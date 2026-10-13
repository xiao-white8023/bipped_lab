"""Preserve a calibrated selector; adjust ONLY decision parameters on validation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.selector.dataset import SelectorShardDataset, ShardBatchSampler
from legged_lab.selector.evaluation import Predictions, collect_predictions, report_predictions
from legged_lab.selector.feature_contract import dataset_feature_contract
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity, sha256
from legged_lab.selector.labels import dataset_safety_label_contract
from legged_lab.selector.metrics import json_safe_metrics
from legged_lab.selector.model import calibrate_probabilities, selector_from_artifact
from legged_lab.selector.selection import SelectionRule, fit_selection_rule


def main():
    parser = argparse.ArgumentParser(description="Validation-only directional decision search; preserve calibration.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive; num-workers must be nonnegative")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("use a new or empty output directory")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if source.get("version") == 5:
        parser.error("version 5 decisions use learned outcomes; use train_selector_outcomes.py")
    baseline = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    model = selector_from_artifact(source)
    reference = selector_from_artifact(baseline)
    if "selection_rule" in baseline:
        parser.error("baseline-checkpoint must be the original two-head model before decision tuning")
    assert_same_policy(source["frozen_policy"], baseline["frozen_policy"])
    for key in ("model_state_dict", "normalizer_state_dict"):
        if (source[key].keys() != baseline[key].keys()
                or any(not torch.equal(tensor, baseline[key][name]) for name, tensor in source[key].items())):
            raise ValueError("baseline and calibrated checkpoint must have identical network/normalizer weights")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    raw_model = torch.nn.Sequential(model.normalizer, model.model).to(device).eval()
    previous_rule = SelectionRule.from_dict(source["selection_rule"]) if "selection_rule" in source else None

    def predict(split):
        dataset = SelectorShardDataset(args.dataset, split)
        assert_same_policy(source["frozen_policy"], dataset_policy_identity(dataset.manifest))
        if (dataset_feature_contract(dataset.manifest) != source["selector_feature_contract"]
                or dataset_safety_label_contract(dataset.manifest) != source["safety_label_contract"]):
            raise ValueError("dataset/checkpoint contracts differ")
        loader = DataLoader(dataset, batch_sampler=ShardBatchSampler(dataset, args.batch_size, shuffle=False),
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        raw = collect_predictions(raw_model, loader, device)
        score = raw.probabilities.to(device)
        calibrated = calibrate_probabilities(score, model.temperatures, model.calibration_biases).cpu()
        original = calibrate_probabilities(score, reference.temperatures.to(device), reference.calibration_biases.to(device)).cpu()
        return raw, calibrated, original

    print("[selector decision] reading validation; network and calibration are frozen", flush=True)
    validation_raw, validation_p, baseline_p = predict("validation")
    scale = model.temperatures.cpu() / reference.temperatures
    bias = reference.calibration_biases - model.calibration_biases.cpu()*scale
    print("[selector decision] searching directional comparisons on validation only", flush=True)
    rule, fit = fit_selection_rule(validation_p, validation_raw.labels, validation_raw.current_mask,
        baseline_probabilities=baseline_p, reference_scale=tuple(scale.tolist()), reference_bias=tuple(bias.tolist()))
    artifact = copy.deepcopy(source)
    artifact["selection_rule"] = rule.to_dict()
    artifact["selection_rule_fit"] = fit
    artifact["source_metrics_before_decision_tuning"] = copy.deepcopy(source.get("metrics", {}))
    artifact["metrics"] = {}
    artifact["decision_tuning_source"] = {"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint), "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_sha256": sha256(args.baseline_checkpoint), "fit_split": "validation",
        "network_changed": False, "normalizer_changed": False, "calibration_changed": False}

    def compare(raw, p, original):
        def prediction(values):
            return Predictions(values, raw.labels, raw.current_mask, raw.audit)
        return {"baseline": report_predictions(prediction(original), None),
                "previous_rule": report_predictions(prediction(p), previous_rule),
                "candidate": report_predictions(prediction(p), rule)}

    validation_report = compare(validation_raw, validation_p, baseline_p)
    artifact["metrics"].update({f"validation/{k}": v for k, v in validation_report["candidate"]["metrics"].items()})
    artifact["metrics"]["validation/decision_guard_passed"] = float(fit["accepted"])
    args.output.mkdir(parents=True, exist_ok=True)
    name = "lp_vp_selector_checkpoint.pt" if fit["accepted"] else "candidate_selector_checkpoint.pt"
    destination = args.output/name
    # Parameters and validation-only acceptance are fixed before test access.
    torch.save(artifact, destination)
    print(f"[selector decision] validation accepted={fit['accepted']}; frozen rule={rule.to_dict()}", flush=True)
    print("[selector decision] reading test for reporting only", flush=True)
    test_raw, test_p, test_baseline = predict("test")
    test_report = compare(test_raw, test_p, test_baseline)
    artifact["metrics"].update({f"test/{k}": v for k, v in test_report["candidate"]["metrics"].items()})
    for key in ("model_state_dict", "normalizer_state_dict", "temperatures", "calibration_biases"):
        left, right = source[key], artifact[key]
        if isinstance(left, dict):
            unchanged = all(torch.equal(tensor, right[n]) for n, tensor in left.items())
        else:
            unchanged = torch.equal(left, right)
        if not unchanged:
            raise RuntimeError(f"decision tuning changed frozen parameters: {key}")
    torch.save(artifact, destination)
    report = {"report_version": 5, "decision_guard": fit["guard"], "selection_rule_fit": fit,
        "validation": validation_report, "test": test_report, "checkpoint": str(destination.resolve()),
        "recommended_checkpoint": str(destination.resolve()) if fit["accepted"] else None,
        "network_unchanged": True, "normalizer_unchanged": True, "calibration_unchanged": True,
        "notes": ["Test labels do not accept, reject, refit or change a rule.",
                  "Reference comparison scores are not reported as failure probabilities.",
                  "A rejected checkpoint is diagnostic only and cannot be exported."]}
    path = args.output/"decision_tuning_report.json"
    path.write_text(json.dumps(json_safe_metrics(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe_metrics({"accepted_on_validation": fit["accepted"], "checkpoint": str(destination),
        "report": str(path), "guard": fit["guard"], "test_candidate": test_report["candidate"]["metrics"],
        "test_checks": test_report["candidate"]["checks"]}), indent=2, sort_keys=True, allow_nan=False), flush=True)
    if not fit["accepted"]:
        print("[selector decision] no candidate passed: source retained; diagnostic checkpoint only", flush=True)


if __name__ == "__main__":
    main()
