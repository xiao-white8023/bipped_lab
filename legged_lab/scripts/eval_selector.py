"""Evaluate a trained LP/VP selector against fixed and oracle baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.selector.dataset import SelectorShardDataset  # noqa: E402
from legged_lab.selector.metrics import json_safe_metrics  # noqa: E402
from legged_lab.selector.evaluation import collect_predictions, report_predictions  # noqa: E402
from legged_lab.selector.selection import SelectionRule  # noqa: E402
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity  # noqa: E402
from legged_lab.selector.labels import dataset_safety_label_contract, validate_safety_label_contract  # noqa: E402
from legged_lab.selector.feature_contract import dataset_feature_contract, validate_feature_contract  # noqa: E402
from legged_lab.selector.model import CalibratedSelector, selector_from_artifact  # noqa: E402


def load_selector(path: Path, device: torch.device) -> CalibratedSelector:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return selector_from_artifact(artifact).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the G1 RENet selector.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-class-samples", type=int, default=20,
                        help="Minimum positive AND negative samples for each of two one-second heads.")
    parser.add_argument("--min-contrast-samples", type=int, default=50,
                        help="Minimum LP-only and VP-only one-second failures, each.")
    parser.add_argument("--max-error-cases", type=int, default=100,
                        help="Maximum diagnostic wrong-choice examples PER direction.")
    args = parser.parse_args()
    if (args.min_class_samples < 1 or args.min_contrast_samples < 1
            or args.batch_size < 1 or args.max_error_cases < 0):
        parser.error("coverage thresholds must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = selector_from_artifact(artifact).to(device).eval()
    dataset = SelectorShardDataset(args.dataset, args.split)
    validate_safety_label_contract(artifact.get("safety_label_contract"))
    assert_same_policy(artifact.get("frozen_policy", {}), dataset_policy_identity(dataset.manifest))
    dataset_safety_label_contract(dataset.manifest)
    dataset_feature_contract(dataset.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rule = SelectionRule.from_dict(artifact["selection_rule"]) if "selection_rule" in artifact else None
    predictions = collect_predictions(model, loader, device)
    guard_config = None
    if artifact.get("version") == 5:
        from legged_lab.selector.outcome_decision import OutcomeGuardConfig
        guard_config = OutcomeGuardConfig(**artifact["outcome_guard_config"])
    measured = report_predictions(predictions, rule, min_class_samples=args.min_class_samples,
        min_contrast_samples=args.min_contrast_samples, max_error_cases=args.max_error_cases,
        outcome_guard_config=guard_config)
    report = {
        "split": args.split,
        "frozen_policy": artifact["frozen_policy"],
        "safety_label_contract": artifact["safety_label_contract"],
        "selector_feature_contract": artifact["selector_feature_contract"],
        "samples": len(dataset),
        **measured,
        "report_version": 6 if artifact.get("version") == 5 else 4,
        "checkpoint_version": artifact["version"],
        "calibration": artifact.get("calibration", {"method": "temperature_only"}),
        "coverage_thresholds": {"min_class_samples": args.min_class_samples,
                                "min_contrast_samples": args.min_contrast_samples},
    }
    rendered = json.dumps(json_safe_metrics(report), indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not all(measured["checks"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
