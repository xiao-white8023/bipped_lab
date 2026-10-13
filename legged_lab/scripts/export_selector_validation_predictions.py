"""Export fixed V5 candidates on VALIDATION ONLY for decision analysis.

No training, recalibration, decision search, dataset edits or test access.
Stored probabilities remain FP32. Raw feature histories are not exported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export six fixed selector candidates on validation only.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive; num-workers nonnegative")
    args.repo = args.repo.expanduser().resolve()
    if not (args.repo/"legged_lab/selector/outcomes.py").is_file():
        parser.error("--repo must point to the installed V5 selector repository")
    if args.output.exists():
        parser.error("output exists; use a new filename")
    paths = sorted(args.models_dir.glob("outcome_seed_*_snapshot_*.pt"))
    if not paths:
        parser.error("no outcome_seed_*_snapshot_*.pt files in --models-dir")
    if not args.baseline_checkpoint.is_file():
        parser.error("original baseline checkpoint is missing")
    sys.path.insert(0, str(args.repo))
    import torch
    from torch.utils.data import DataLoader
    from legged_lab.selector.dataset import SelectorShardDataset, ShardBatchSampler
    from legged_lab.selector.feature_contract import dataset_feature_contract
    from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity
    from legged_lab.selector.labels import dataset_safety_label_contract
    from legged_lab.selector.model import selector_from_artifact
    from legged_lab.selector.outcomes import OUTCOME_ORDER, outcome_classes, validate_outcome_probabilities
    from legged_lab.selector.selection import current_mask_from_features

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; activate env_tienkung or specify --device cpu")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # This reader opens only the validation shards, never train or test shards.
    dataset = SelectorShardDataset(args.dataset, "validation")
    if not len(dataset):
        parser.error("validation split is empty")
    manifest_sha = sha256(args.dataset/"manifest.json")
    feature_contract = dataset_feature_contract(dataset.manifest)
    label_contract = dataset_safety_label_contract(dataset.manifest)
    frozen_policy = dataset_policy_identity(dataset.manifest)
    loader = DataLoader(dataset,
        batch_sampler=ShardBatchSampler(dataset, args.batch_size, shuffle=False),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0)

    def load_checked(path, outcome):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if outcome:
            if artifact.get("version") != 5 or artifact.get("outcome_order") != list(OUTCOME_ORDER):
                raise ValueError(f"candidate needs V5 four-outcome format: {path.name}")
        elif artifact.get("version") != 3 or "selection_rule" in artifact:
            raise ValueError("baseline must be the original V3 model before decision tuning")
        assert_same_policy(artifact["frozen_policy"], frozen_policy)
        if (artifact.get("selector_feature_contract") != feature_contract
                or artifact.get("safety_label_contract") != label_contract):
            raise ValueError(f"dataset/checkpoint contracts differ: {path.name}")
        expected = (artifact.get("data_sources") or {}).get("base_manifest_sha256")
        if expected is not None and expected != manifest_sha:
            raise ValueError(f"checkpoint belongs to a different dataset: {path.name}")
        return artifact

    @torch.inference_mode()
    def predict(model, outcome, collect_identity=False):
        model.eval()
        probabilities, risks, labels, masks, seeds, groups = [], [], [], [], [], []
        audit = {}
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            if outcome:
                r, q = model.decision_outputs(features)
                validate_outcome_probabilities(q, len(features))
                probabilities.append(q.float().cpu())
            else:
                r = model(features)
            risks.append(r.float().cpu())
            labels.append(batch["labels"].to(dtype=torch.uint8).cpu())
            masks.append(current_mask_from_features(batch["features"]).to(torch.uint8).cpu())
            if collect_identity:
                groups.extend(map(str, batch["group_ids"]))
                seeds.append(batch["scenario_seed"].reshape(-1).long().cpu())
                for key in ("terrain_level", "terrain_type", "candidate_time", "event_identity", "fault_metadata"):
                    if key in batch:
                        audit.setdefault(key, []).append(batch[key].cpu())
                if "future_metrics" in batch:
                    audit.setdefault("first_failure_step_lp_vp", []).append(batch["future_metrics"][:,:,0].short().cpu())
        return (torch.cat(risks), torch.cat(probabilities) if probabilities else None,
                torch.cat(labels), torch.cat(masks), groups,
                torch.cat(seeds) if seeds else None,
                {key: torch.cat(value) for key,value in audit.items()})

    baseline_artifact = load_checked(args.baseline_checkpoint, False)
    model = selector_from_artifact(baseline_artifact).to(device).eval()
    print(f"[validation export] baseline; rows={len(dataset)}; candidates={len(paths)}", flush=True)
    baseline_p, _, labels, masks, groups, seeds, audit = predict(model, False, True)
    del model
    group_names = sorted(set(groups))
    group_sha = hashlib.sha256("\n".join(group_names).encode()).hexdigest()
    if dataset.manifest.get("group_sha256", {}).get("validation") not in (None, group_sha):
        raise ValueError("validation terrain group hash differs from the manifest")
    if len(torch.unique(seeds)) != len(dataset):
        raise ValueError("validation contains duplicate scenario identities")
    group_index = {name: index for index,name in enumerate(group_names)}
    dump = {
        "format": "selector_validation_predictions", "version": 1,
        "split": "validation", "test_rows_exported": 0, "features_exported": False,
        "dataset": str(args.dataset.resolve()), "base_manifest_sha256": manifest_sha,
        "samples": len(dataset), "device": str(device), "outcome_order": list(OUTCOME_ORDER),
        "labels": labels, "current_mask": masks, "scenario_seed": seeds,
        "group_names": group_names,
        "group_index": torch.tensor([group_index[name] for name in groups], dtype=torch.int32),
        "audit": audit, "selector_feature_contract": feature_contract,
        "safety_label_contract": label_contract, "frozen_policy": frozen_policy,
        "baseline": {"checkpoint": str(args.baseline_checkpoint.resolve()),
                     "sha256": sha256(args.baseline_checkpoint), "probabilities": baseline_p},
        "candidates": {},
    }
    for path in paths:
        print(f"[validation export] reading {path.name}", flush=True)
        before = sha256(path)
        artifact = load_checked(path, True)
        for key in ("model_state_dict", "normalizer_state_dict"):
            if (artifact[key].keys() != baseline_artifact[key].keys()
                    or any(not torch.equal(v, baseline_artifact[key][k]) for k,v in artifact[key].items())):
                raise ValueError(f"candidate and baseline risk weights/statistics differ: {path.name}")
        model = selector_from_artifact(artifact).to(device).eval()
        risk, q, y, m, _, _, _ = predict(model, True)
        if not torch.equal(y, labels) or not torch.equal(m, masks):
            raise RuntimeError("validation row ordering differs between candidates")
        if "calibrated_risks" not in dump:
            dump["calibrated_risks"] = risk
        elif not torch.allclose(risk, dump["calibrated_risks"], rtol=1e-5, atol=1e-6):
            raise ValueError("candidate frozen risk outputs/calibration differ")
        if sha256(path) != before:
            raise RuntimeError("checkpoint changed during the export")
        dump["candidates"][path.name] = {
            "checkpoint": str(path.resolve()), "sha256": before,
            "seed": artifact.get("seed"), "best_epoch": artifact.get("outcome_best_epoch"),
            "probabilities": q, "selection_rule": artifact["selection_rule"],
            "outcome_calibration": artifact.get("outcome_calibration"),
            "validation_guard": artifact.get("selection_rule_fit", {}).get("guard"),
        }
        del model, artifact
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    counts = torch.bincount(outcome_classes(labels), minlength=4).tolist()
    dump["outcome_counts"] = counts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="validation_export_tmp_", suffix=".pt", dir=args.output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(dump, temporary_path)
        # Exclusive publication; an existing dump can never be overwritten.
        os.link(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output.resolve()), "size_mib": round(args.output.stat().st_size/(1024**2),2),
        "validation_samples": len(dataset), "outcome_counts": counts, "candidates": len(paths),
        "test_samples": 0, "training_performed": False, "checkpoints_modified": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
