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

from legged_lab.selector.config import SelectorConfig  # noqa: E402
from legged_lab.selector.dataset import SelectorShardDataset  # noqa: E402
from legged_lab.selector.metrics import evaluation_checks, json_safe_metrics, selector_metrics  # noqa: E402
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity  # noqa: E402
from legged_lab.selector.labels import dataset_safety_label_contract, validate_safety_label_contract  # noqa: E402
from legged_lab.selector.feature_contract import dataset_feature_contract, validate_feature_contract  # noqa: E402
from legged_lab.selector.model import CalibratedSelector, FeatureNormalizer, SelectorNetwork  # noqa: E402


def load_selector(path: Path, device: torch.device) -> CalibratedSelector:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_feature_contract(artifact.get("selector_feature_contract"))
    cfg = SelectorConfig(**artifact["selector_config"])
    model = SelectorNetwork(cfg)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    normalizer = FeatureNormalizer(cfg.frame_dim)
    normalizer.load_state_dict(artifact["normalizer_state_dict"], strict=True)
    return CalibratedSelector(model, normalizer, artifact["temperatures"]).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the G1 RENet selector.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-class-samples", type=int, default=20,
                        help="Minimum positive AND negative samples for each of four heads.")
    parser.add_argument("--min-contrast-samples", type=int, default=50,
                        help="Minimum LP-only and VP-only long-horizon failures, each.")
    args = parser.parse_args()
    if args.min_class_samples < 1 or args.min_contrast_samples < 1:
        parser.error("coverage thresholds must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_selector(args.checkpoint, device)
    dataset = SelectorShardDataset(args.dataset, args.split)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_safety_label_contract(artifact.get("safety_label_contract"))
    assert_same_policy(artifact.get("frozen_policy", {}), dataset_policy_identity(dataset.manifest))
    dataset_safety_label_contract(dataset.manifest)
    dataset_feature_contract(dataset.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities, labels = [], []
    fault_metadata, terrain_levels = [], []
    with torch.no_grad():
        for batch in loader:
            probabilities.append(model(batch["features"].to(device)).cpu())
            labels.append(batch["labels"].float())
            if "fault_metadata" in batch:
                fault_metadata.append(batch["fault_metadata"].long())
            terrain_levels.append(batch["terrain_level"].long())
    probabilities_tensor = torch.cat(probabilities)
    labels_tensor = torch.cat(labels)
    metrics = selector_metrics(probabilities_tensor, labels_tensor)
    breakdown: dict[str, dict[str, float]] = {}

    def add_breakdown(name: str, selected: torch.Tensor) -> None:
        if selected.any():
            breakdown[name] = selector_metrics(
                probabilities_tensor[selected], labels_tensor[selected]
            )

    if fault_metadata:
        faults = torch.cat(fault_metadata)
        lp_active, vp_active = faults[:, 6].bool(), faults[:, 7].bool()
        add_breakdown("normal", ~lp_active & ~vp_active)
        add_breakdown("lp_fault", lp_active & ~vp_active)
        add_breakdown("vp_fault", ~lp_active & vp_active)
        add_breakdown("dual_fault", lp_active & vp_active)
        add_breakdown("fault_recovery", (faults[:, 3] > 0) & ((faults[:, 0] > 0) | (faults[:, 1] > 0)))
    levels = torch.cat(terrain_levels)
    for level in torch.unique(levels).tolist():
        add_breakdown(f"terrain_level_{int(level)}", levels == level)
    checks = evaluation_checks(metrics, min_class_samples=args.min_class_samples,
                               min_contrast_samples=args.min_contrast_samples)
    report = {
        "split": args.split,
        "frozen_policy": artifact["frozen_policy"],
        "safety_label_contract": artifact["safety_label_contract"],
        "selector_feature_contract": artifact["selector_feature_contract"],
        "samples": len(dataset),
        "metrics": metrics,
        "breakdown": breakdown,
        "checks": checks,
        "coverage_thresholds": {"min_class_samples": args.min_class_samples,
                                "min_contrast_samples": args.min_contrast_samples},
    }
    rendered = json.dumps(json_safe_metrics(report), indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()