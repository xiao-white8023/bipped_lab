"""Recalibrate an existing selector using validation; retain all trained weights.

Only validation rows fit the four calibration parameters and selection
thresholds. Test rows are read afterwards solely for the comparison report.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.selector.dataset import SelectorShardDataset, ShardBatchSampler  # noqa: E402
from legged_lab.selector.evaluation import Predictions, collect_predictions, report_predictions  # noqa: E402
from legged_lab.selector.feature_contract import dataset_feature_contract  # noqa: E402
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity, sha256  # noqa: E402
from legged_lab.selector.labels import dataset_safety_label_contract  # noqa: E402
from legged_lab.selector.metrics import json_safe_metrics, selector_metrics  # noqa: E402
from legged_lab.selector.model import (  # noqa: E402
    calibrate_probabilities, fit_affine_calibration, selector_from_artifact,
)
from legged_lab.selector.selection import SelectionRule, fit_selection_rule  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only affine calibration of an existing selector.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; source is never overwritten.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-error-cases", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_error_cases < 0:
        parser.error("batch-size must be positive; workers/error limits must be nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("use a new or empty output directory")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if source.get("version") == 5:
        parser.error("version 5 includes a learned decision head; use train_selector_outcomes.py so it is preserved")
    model = selector_from_artifact(source).to(device).eval()
    source_rule = SelectionRule.from_dict(source["selection_rule"]) if "selection_rule" in source else None

    def loader_for(split):
        dataset = SelectorShardDataset(args.dataset, split)
        assert_same_policy(source["frozen_policy"], dataset_policy_identity(dataset.manifest))
        if dataset_safety_label_contract(dataset.manifest) != source["safety_label_contract"]:
            raise ValueError("checkpoint/dataset safety labels differ")
        if dataset_feature_contract(dataset.manifest) != source["selector_feature_contract"]:
            raise ValueError("checkpoint/dataset features differ")
        return DataLoader(dataset, batch_sampler=ShardBatchSampler(dataset, args.batch_size, shuffle=False),
            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print("[selector recalibration] reading validation only", flush=True)
    # Identity calibration exposes the ORIGINAL raw network scores. Never
    # invert rounded/saturated previously calibrated probabilities.
    raw_model = torch.nn.Sequential(model.normalizer, model.model).to(device).eval()
    validation_raw = collect_predictions(raw_model, loader_for("validation"), device)
    temperatures, biases, calibration = fit_affine_calibration(
        validation_raw.probabilities.to(device), validation_raw.labels.to(device),
        positive_weights=source.get("positive_weights"),
        initial_temperatures=model.temperatures, initial_biases=model.calibration_biases,
    )
    validation_after = calibrate_probabilities(validation_raw.probabilities.to(device), temperatures, biases)
    rule, rule_fit = fit_selection_rule(
        validation_after, validation_raw.labels.to(device), validation_raw.current_mask.to(device),
        baseline_probabilities=calibrate_probabilities(validation_raw.probabilities.to(device),
            model.temperatures, model.calibration_biases),
        reference_scale=tuple((temperatures / model.temperatures).tolist()),
        reference_bias=tuple((model.calibration_biases-biases*temperatures/model.temperatures).tolist()),
    )
    temperatures, biases = temperatures.cpu(), biases.cpu()
    artifact = copy.deepcopy(source)
    artifact.update({"version": 4, "temperatures": temperatures, "calibration_biases": biases,
        "calibration": calibration, "selection_rule": rule.to_dict(), "selection_rule_fit": rule_fit,
        "recalibration_source": {"checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint), "source_seed": source.get("seed"),
            "fit_split": "validation", "network_weights_changed": False,
            "normalizer_changed": False}})
    # Old metrics describe the old probability mapping/rule, so keep them
    # explicitly as history rather than attaching them to the new predictor.
    artifact["source_metrics_before_recalibration"] = copy.deepcopy(source.get("metrics", {}))
    artifact["metrics"] = {}

    def compare(raw):
        device_scores = raw.probabilities.to(device)
        before = Predictions(calibrate_probabilities(device_scores,
            model.temperatures, model.calibration_biases).cpu(), raw.labels, raw.current_mask, raw.audit)
        after = Predictions(calibrate_probabilities(device_scores,
            temperatures.to(device), biases.to(device)).cpu(), raw.labels, raw.current_mask, raw.audit)
        return {"before": report_predictions(before, source_rule, max_error_cases=args.max_error_cases),
                "after": report_predictions(after, rule, max_error_cases=args.max_error_cases),
                # Same selection rule as the source, isolating probability calibration.
                "calibration_only_same_source_rule": selector_metrics(after.probabilities, after.labels,
                    current_mask=after.current_mask, selection_rule=source_rule)}

    validation_comparison = compare(validation_raw)
    for name, value in validation_comparison["after"]["metrics"].items():
        artifact["metrics"][f"validation/{name}"] = value
    if "validation/loss" in source.get("metrics", {}):
        artifact["metrics"]["validation/loss"] = source["metrics"]["validation/loss"]
    if "training/epochs" in source.get("metrics", {}):
        artifact["metrics"]["training/epochs"] = source["metrics"]["training/epochs"]
    # Freeze and persist the choice BEFORE reading any test rows.
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "lp_vp_selector_checkpoint.pt"
    torch.save(artifact, destination)
    print(f"[selector recalibration] frozen validation-fitted parameters: {rule.to_dict()}", flush=True)
    print("[selector recalibration] reading test for reporting only", flush=True)
    test_raw = collect_predictions(raw_model, loader_for("test"), device)
    test_comparison = compare(test_raw)
    for name, value in test_comparison["after"]["metrics"].items():
        artifact["metrics"][f"test/{name}"] = value
    # Verify inference/calibration did not mutate any trainable weights or normalization.
    for key, state in (("model_state_dict", model.model.state_dict()),
                       ("normalizer_state_dict", model.normalizer.state_dict())):
        if any(not torch.equal(source[key][name], tensor.cpu()) for name, tensor in state.items()):
            raise RuntimeError("recalibration unexpectedly changed network or normalizer weights")
    torch.save(artifact, destination)
    report = {"report_version": 4, "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(destination.resolve()), "calibration": calibration,
        "selection_rule_fit": rule_fit, "validation": validation_comparison, "test": test_comparison,
        "network_weights_unchanged": True, "normalizer_unchanged": True,
        "notes": ["No training samples or test labels fit calibration/thresholds.",
                  "Compare before.rank_comparison and after.rank_comparison to isolate calibration under the original rank rule.",
                  "Primary after metrics use the new danger-to-safe rule; its rule differs from a legacy source."]}
    report_path = args.output / "recalibration_report.json"
    report_path.write_text(json.dumps(json_safe_metrics(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe_metrics({"checkpoint": str(destination), "report": str(report_path),
        "test_before": test_comparison["before"]["metrics"], "test_after": test_comparison["after"]["metrics"],
        "test_checks": test_comparison["after"]["checks"]}), indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
