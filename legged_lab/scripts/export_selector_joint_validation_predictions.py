"""Export complete V6/V7 validation predictions without fitting or opening other splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-audit-metadata", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    for name in ("repo", "dataset", "checkpoint", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("output exists; choose a new filename")
    if args.output.is_relative_to(args.dataset):
        parser.error("output must be outside the source dataset")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(args.repo))
    import torch
    from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity
    from legged_lab.selector.joint_data import (
        JointShardDataset, check_payload, file_sha256, validate_manifest,
    )
    from legged_lab.selector.joint_model import joint_selector_from_artifact, risks_from_joint
    from legged_lab.selector.joint_training import calibrate_q, collect_logits, make_loader
    from legged_lab.selector.labels import dataset_safety_label_contract
    from legged_lab.selector.outcomes import OUTCOME_ORDER

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the training environment or choose --device cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    manifest_path = args.dataset / "manifest.json"
    checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames, feature_contract = validate_manifest(manifest)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = joint_selector_from_artifact(artifact).to(device).eval()
    assert_same_policy(artifact["frozen_policy"], dataset_policy_identity(manifest))
    if (artifact["source_feature_contract"] != feature_contract
            or artifact["safety_label_contract"] != dataset_safety_label_contract(manifest)):
        raise ValueError("dataset feature/safety contract differs from checkpoint")
    expected_manifest = artifact.get("training_sources", {}).get("base_manifest_sha256")
    if expected_manifest is None or expected_manifest != manifest_sha:
        raise ValueError("dataset must match the checkpoint's recorded base manifest")

    # Do not call scan_root/dataset_index: both also open train and test shards.
    paths = sorted((args.dataset / "validation").glob("shard_*.pt"))
    if not paths or len(paths) != manifest["shards"]["validation"]:
        raise ValueError("validation shard count differs from manifest")
    lengths, records, identity, masks = [], [], [], []
    groups, seen_ids, audit_parts = set(), set(), {}
    optional = ("group_ids", "fault_metadata", "source_metadata", "vp_fault_metadata",
                "terrain_level", "terrain_type", "candidate_time", "execution_metadata")
    audit_fields = None
    for number, path in enumerate(paths):
        before = path.stat()
        data = torch.load(path, map_location="cpu", weights_only=False)
        count = check_payload(data, manifest, path)
        ids = data["scenario_seed"]
        if ids.shape != (count,) or ids.dtype != torch.int64:
            raise ValueError(f"scenario_seed must be int64 [rows]: {path}")
        values = set(ids.tolist())
        if len(values) != count or seen_ids & values:
            raise ValueError(f"duplicate validation sample IDs: {path}")
        seen_ids.update(values)
        group_ids = data["group_ids"]
        if len(group_ids) != count or not all(isinstance(g, str) for g in group_ids):
            raise ValueError(f"invalid validation group IDs: {path}")
        groups.update(group_ids)
        current = data["features"][:, -1, -2].float()
        if not torch.isfinite(current).all() or not ((current == 0) | (current == 1)).all():
            raise ValueError(f"current branch must be binary VP=0/LP=1: {path}")
        identity.append(ids.clone())
        masks.append(current.clone())
        if args.include_audit_metadata:
            fields = {key for key in optional if key in data}
            if audit_fields is not None and fields != audit_fields:
                raise ValueError("audit columns differ between validation shards")
            audit_fields = fields
            for key in fields:
                value = data[key]
                if len(value) != count:
                    raise ValueError(f"audit field length differs: {key}, {path}")
                audit_parts.setdefault(key, []).append(value.clone() if torch.is_tensor(value) else list(value))
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"validation shard changed while reading: {path}")
        lengths.append(count)
        records.append({"path": str(path), "sha256": digest,
                        "size_mtime_ns": [after.st_size, after.st_mtime_ns], "rows": count})
        del data
        print(f"[joint validation export] checked shard {number+1}/{len(paths)}", flush=True)
    if sum(lengths) != manifest["counts"]["validation"]:
        raise ValueError("validation row count differs from manifest")
    group_sha = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
    if manifest.get("group_sha256", {}).get("validation") != group_sha:
        raise ValueError("validation group digest differs from manifest")
    index = {"root": args.dataset, "manifest": manifest, "stored_frames": frames,
             "lengths": {"validation": lengths},
             "file_stats": {r["path"]: r["size_mtime_ns"] for r in records}}
    dataset = JointShardDataset([index], "validation", model.model.cfg.history_frames)
    if dataset.shards != paths:
        raise RuntimeError("validation shard list changed after audit")
    predictions = collect_logits(model.model, model.normalizer,
        make_loader(dataset, args, device), device, "validation only")
    if (not torch.equal(predictions["sample_id"], torch.cat(identity))
            or not torch.equal(predictions["current_mask"], torch.cat(masks))):
        raise RuntimeError("prediction row order differs from validation shard order")
    with torch.no_grad():
        logits = predictions["logits"]
        temperature, biases = model.temperature.cpu(), model.biases.cpu()
        raw_q = logits.softmax(1)
        calibrated_q = calibrate_q(predictions, model.temperature, model.biases, device)
        risks = risks_from_joint(calibrated_q)
    snapshots = [(args.checkpoint, checkpoint_sha), (manifest_path, manifest_sha)]
    snapshots.extend((Path(r["path"]), r["sha256"]) for r in records)
    for path, expected in snapshots:
        if file_sha256(path) != expected:
            raise RuntimeError(f"input changed during prediction export: {path}")
    audit = {key: torch.cat(parts) if torch.is_tensor(parts[0])
             else [item for part in parts for item in part] for key, parts in audit_parts.items()}
    dump = {"format": "joint_selector_validation_predictions", "version": 1,
        "artifact_version": artifact["version"], "architecture": model.model.architecture_description(),
        "decision_probability_mode": artifact.get("decision_probability_mode", "temperature_bias"),
        "split": "validation", "samples": len(logits), "outcome_order": list(OUTCOME_ORDER),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
        "dataset": str(args.dataset), "dataset_manifest_sha256": manifest_sha,
        "model_input_shape": [model.model.cfg.history_frames, 283],
        "raw_logits": logits, "raw_q": raw_q, "calibrated_q": calibrated_q,
        "derived_risks": risks, "labels": predictions["labels"],
        "current_mask": predictions["current_mask"], "sample_id": predictions["sample_id"],
        "temperature": temperature, "biases": biases,
        "shard_index": torch.repeat_interleave(torch.arange(len(paths)), torch.tensor(lengths)),
        "row_in_shard": torch.cat([torch.arange(n) for n in lengths]),
        "validation_shards": records, "audit_metadata": audit,
        "calibration_refitted": False, "thresholds_refitted": False, "weights_updated": False,
        "read_splits": ["validation"], "inputs_unchanged": True,
        "cross_split_isolation_reaudited": False, "device": str(device), "torch_version": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="joint_validation_export_", suffix=".pt", dir=args.output.parent)
    os.close(fd)
    try:
        torch.save(dump, temporary)
        os.link(temporary, args.output)  # atomic publication, refuses any existing target
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"[joint validation export] saved {len(logits)} rows: {args.output}", flush=True)
    print(f"[joint validation export] bytes={args.output.stat().st_size} sha256={file_sha256(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
