"""Export the calibrated selector to TorchScript with parity validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.selector.config import SelectorConfig  # noqa: E402
from legged_lab.selector.frozen_policy import policy_identity  # noqa: E402
from legged_lab.selector.labels import validate_safety_label_contract  # noqa: E402
from legged_lab.selector.model import CalibratedSelector, FeatureNormalizer, SelectorNetwork  # noqa: E402
from legged_lab.selector.feature_contract import validate_feature_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the G1 RENet selector.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frozen_metadata = policy_identity(artifact.get("frozen_policy", {}))
    label_contract = validate_safety_label_contract(artifact.get("safety_label_contract"))
    feature_contract = validate_feature_contract(artifact.get("selector_feature_contract"))
    cfg = SelectorConfig(**artifact["selector_config"])
    model = SelectorNetwork(cfg)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    normalizer = FeatureNormalizer(cfg.frame_dim)
    normalizer.load_state_dict(artifact["normalizer_state_dict"], strict=True)
    calibrated = CalibratedSelector(model, normalizer, artifact["temperatures"]).eval()
    dummy = torch.randn(8, cfg.history_frames, cfg.frame_dim)
    with torch.inference_mode():
        reference = calibrated(dummy)
    traced = torch.jit.trace(calibrated, dummy, check_trace=True)
    with torch.inference_mode():
        actual = traced(dummy)
    if not torch.allclose(reference, actual, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(
            f"selector TorchScript parity failed: max error {(reference - actual).abs().max().item()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 2,
        "feature_shape": [cfg.history_frames, cfg.frame_dim],
        "output_order": ["lp_0.4s", "lp_1.0s", "vp_0.4s", "vp_1.0s"],
        "mask_contract": {"vp": 0, "lp": 1},
        "source_checkpoint": str(args.checkpoint.resolve()),
        "frozen_policy": frozen_metadata,
        "safety_label_contract": label_contract,
        "selector_feature_contract": feature_contract,
        "metrics": artifact.get("metrics", {}),
    }
    destination = args.output_dir / "lp_vp_selector.pt"
    torch.jit.save(
        traced,
        destination,
        _extra_files={"selector_contract.json": json.dumps(metadata, sort_keys=True)},
    )
    reloaded = torch.jit.load(destination).eval()
    with torch.inference_mode():
        reloaded_output = reloaded(dummy)
    if not torch.allclose(reference, reloaded_output, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError("reloaded selector artifact failed parity validation")
    (args.output_dir / "selector_contract.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[selector] exported {destination}")


if __name__ == "__main__":
    main()