"""Evaluate a saved V6/V7 joint selector without fitting or rewriting weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.selector.joint_data import JointShardDataset, file_sha256, scan_root, validate_manifest
from legged_lab.selector.joint_model import joint_selector_from_artifact
from legged_lab.selector.joint_training import collect_logits, calibrate_q, make_loader, report_joint, save_json, threshold_warning_report
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity
from legged_lab.selector.labels import dataset_safety_label_contract
from legged_lab.selector.joint_timing import collect_failure_steps, report_failure_timing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=30)
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.max_examples < 0:
        p.error("invalid batch/worker/example settings")
    if args.output.exists():
        p.error("choose a new output file")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate env_tienkung or choose --device cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    digest = file_sha256(args.checkpoint)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = joint_selector_from_artifact(artifact).to(device).eval()
    index = scan_root(args.dataset)
    assert_same_policy(artifact["frozen_policy"], dataset_policy_identity(index["manifest"]))
    if (index["feature_contract"] != artifact["source_feature_contract"]
            or dataset_safety_label_contract(index["manifest"]) != artifact["safety_label_contract"]):
        raise ValueError("evaluation source feature/safety contract differs from training")
    dataset = JointShardDataset([index], args.split, model.model.cfg.history_frames)
    loader = make_loader(dataset, args, device)
    prediction = collect_logits(model.model, model.normalizer, loader, device, args.split)
    q = calibrate_q(prediction, model.temperature, model.biases, device)
    report = report_joint(q, prediction, args.max_examples)
    report["failure_timing"] = report_failure_timing(q, prediction, collect_failure_steps(dataset, prediction))
    threshold = artifact.get("warning_threshold_diagnostic", {}).get("threshold")
    if threshold is not None:
        report["threshold_warning_diagnostic"] = threshold_warning_report(q, prediction["labels"], threshold)
    if file_sha256(args.checkpoint) != digest:
        raise RuntimeError("checkpoint changed during evaluation")
    report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
                  dataset=str(args.dataset.resolve()), dataset_manifest_sha256=index["sha256"], split=args.split,
                  calibration_refitted=False, thresholds_refitted=False, weights_updated=False)
    report.update(decision_probability_mode=artifact.get("decision_probability_mode", "temperature_bias"),
                  architecture=model.model.architecture_description())
    save_json(args.output, report)
    print(args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
