"""Verify filtered-base ancestry before V5 training, without rewriting provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SPLITS = ("train", "validation", "test")
PAIR_ORDER = ("both_safe", "lp_only_failure", "vp_only_failure", "both_failure")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids_and_labels(data):
    n = len(data["features"])
    if tuple(data["features"].shape) != (n, 5, 283) or tuple(data["labels"].shape) != (n, 2):
        raise ValueError("unexpected observable/label shape")
    labels = data["labels"].tolist()
    if any(len(row) != 2 or any(v not in (0, 1) for v in row) for row in labels):
        raise ValueError("labels must be binary LP/VP 1-second outcomes")
    return n, labels


def _filter_ids(data, apply_filter):
    n, labels = _ids_and_labels(data)
    if tuple(data["fault_metadata"].shape) != (n, 8):
        raise ValueError("fault metadata must have eight columns")
    ids, lp_removed, vp_removed = [], 0, 0
    for i, (label, fault) in enumerate(zip(labels, data["fault_metadata"].tolist())):
        if any(fault[j] not in (0, 1) for j in (6, 7)):
            raise ValueError("fault active flags must be binary")
        lp, vp = bool(fault[6]), bool(fault[7])
        a = apply_filter and lp and not vp and bool(label[1])
        b = apply_filter and vp and not lp and bool(label[0])
        lp_removed += int(a)
        vp_removed += int(b)
        if not (a or b):
            ids.append(i)
    return ids, lp_removed, vp_removed


def _compare_rows(original, actual, ids, torch, location):
    if original.keys() != actual.keys():
        raise ValueError(f"filtered fields differ from source: {location}")
    n = len(original["features"])
    index = torch.tensor(ids, dtype=torch.long)
    for key, value in original.items():
        other = actual[key]
        if torch.is_tensor(value):
            if value.ndim < 1 or value.shape[0] != n or not torch.is_tensor(other):
                raise ValueError(f"invalid row field {key}: {location}")
            expected = value if len(ids) == n else value.index_select(0, index)
            equal = (expected.dtype == other.dtype and expected.shape == other.shape
                     and torch.equal(expected, other))
        elif isinstance(value, (list, tuple)) and len(value) == n:
            expected = [value[i] for i in ids]
            equal = isinstance(other, type(value)) and expected == list(other)
        else:
            raise ValueError(f"unsupported row field {key}: {location}")
        if not equal:
            raise ValueError(f"filtered content differs from exact source selection: {location}, {key}")


def verify_filtered_rows(original_index, filtered_index, torch):
    """Compare every retained field against the exact source selection, one shard at a time."""
    original, filtered = original_index["root"], filtered_index["root"]
    snapshot = filtered / "original_manifest.json"
    report_path = filtered / "filter_report.json"
    if not snapshot.is_file() or _sha(snapshot) != original_index["sha256"]:
        raise ValueError("filtered original_manifest.json does not identify the supplied original dataset")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = filtered_index["manifest"].get("metadata", {}).get("offline_filter", {})
    if (report.get("verified") is not True
            or report.get("source_manifest_sha256") != original_index["sha256"]
            or provenance.get("source_manifest_sha256") != original_index["sha256"]):
        raise ValueError("missing or inconsistent filter provenance")
    filtered_splits = report.get("splits_filtered")
    if (not isinstance(filtered_splits, list) or not filtered_splits
            or len(set(filtered_splits)) != len(filtered_splits)
            or not set(filtered_splits) <= set(SPLITS)
            or provenance.get("splits_filtered") != filtered_splits):
        raise ValueError("invalid filtered split declaration")
    for key in ("source_manifest_sha256", "rules", "includes_both_failure_outcomes_in_single_fault_scenarios"):
        if key not in report:
            raise ValueError(f"filter report lacks {key}")
    snapshots = {"original": _sha(original / "manifest.json"), "filtered": _sha(filtered / "manifest.json"),
                 "filter_report": _sha(report_path)}
    verified = {}
    for split in SPLITS:
        output_paths = sorted((filtered / split).glob("shard_*.pt"))
        next_output = 0
        stats = {"before": 0, "after": 0, "removed": 0,
                 "removed_lp_fault_vp_failure": 0, "removed_vp_fault_lp_failure": 0}
        original_paths = sorted((original / split).glob("shard_*.pt"))
        for number, source_path in enumerate(original_paths, 1):
            before = source_path.stat()
            data = torch.load(source_path, map_location="cpu", weights_only=False)
            ids, a, b = _filter_ids(data, split in filtered_splits)
            n = len(data["features"])
            stats["before"] += n
            stats["after"] += len(ids)
            stats["removed"] += a+b
            stats["removed_lp_fault_vp_failure"] += a
            stats["removed_vp_fault_lp_failure"] += b
            if ids:
                if next_output >= len(output_paths):
                    raise ValueError(f"missing filtered shard: {split}")
                target = output_paths[next_output]
                target_before = target.stat()
                child = torch.load(target, map_location="cpu", weights_only=False)
                _compare_rows(data, child, ids, torch, str(target))
                if (target_before.st_size, target_before.st_mtime_ns) != (target.stat().st_size, target.stat().st_mtime_ns):
                    raise RuntimeError(f"filtered shard changed during audit: {target}")
                next_output += 1
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"original shard changed during audit: {source_path}")
            if number % 10 == 0 or number == len(original_paths):
                print(f"[clean lineage] {split} {number}/{len(original_paths)}: "
                      f"kept={stats['after']} removed={stats['removed']}", flush=True)
        recorded = report.get("by_split", {}).get(split, {})
        if (next_output != len(output_paths)
                or stats["before"] != original_index["manifest"]["counts"][split]
                or stats["after"] != filtered_index["manifest"]["counts"][split]
                or any(recorded.get(k) != v for k, v in stats.items())):
            raise ValueError(f"filter report or exact row counts disagree: {split}")
        verified[split] = stats
    for name, path in (("original", original / "manifest.json"), ("filtered", filtered / "manifest.json"),
                       ("filter_report", report_path)):
        if _sha(path) != snapshots[name]:
            raise RuntimeError(f"{name} metadata changed during lineage verification")
    return {"verified": True, "verification": "every retained tensor/list field compared to exact original selection",
            "original_manifest_sha256": original_index["sha256"],
            "filtered_manifest_sha256": filtered_index["sha256"],
            "filter_report_sha256": snapshots["filter_report"], "by_split": verified,
            "splits_filtered": filtered_splits}


def verify_directed_rows(index, torch, directed_contract):
    """Recheck actual saved labels and observable source flags, rather than trust counters."""
    manifest = index["manifest"]
    md = manifest["metadata"]
    info = md.get("supplement_source", {})
    cfg = info.get("config", {})
    mode = cfg.get("fault_source")
    target = {"lp_only": 1, "vp_only": 2, "dual": 3}.get(mode)
    if target is None or md.get("directed_audit_contract") != directed_contract:
        raise ValueError(f"missing/unsupported directed source contract: {index['root']}")
    counts = [0]*4
    for path in sorted((index["root"] / "train").glob("shard_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        n, labels = _ids_and_labels(data)
        if ("source_metadata" not in data or tuple(data["source_metadata"].shape) != (n, 8)
                or tuple(data["fault_metadata"].shape) != (n, 8)):
            raise ValueError(f"missing directed row metadata: {path}")
        for fault, source, label in zip(data["fault_metadata"].tolist(), data["source_metadata"].tolist(), labels):
            if (any(not math.isfinite(float(x)) for x in source)
                    or any(x not in (0, 1) for x in source[:4])):
                raise ValueError(f"invalid source effect flags: {path}")
            lp, vp = bool(fault[0]), bool(fault[1])
            matches = ((lp and not vp) if mode == "lp_only" else
                       (vp and not lp) if mode == "vp_only" else (lp and vp))
            effect = bool(source[0]) if mode == "lp_only" else bool(source[1]) if mode == "vp_only" else bool(source[0] and source[1])
            clean = bool(source[3]) if mode == "lp_only" else bool(source[2]) if mode == "vp_only" else True
            pair = int(label[0])+2*int(label[1])
            if not (matches and effect and clean) or pair != target:
                raise ValueError(f"directed source/outcome/effect check failed: {path}")
            counts[pair] += 1
    if counts != cfg.get("pair_quotas") or sum(counts) != manifest["counts"]["train"]:
        raise ValueError(f"actual directed label counts differ from quotas: {index['root']}")
    return {"verified": True, "fault_source": mode, "pair_counts": counts}


def audit_filtered_training_sources(original_root, filtered_root, supplement_roots=()):
    import torch
    from .supplement_dataset import dataset_index, compatible_manifests
    from .directed_audit import directed_audit_contract
    from .feature_contract import dataset_feature_contract
    from .labels import dataset_safety_label_contract
    from .frozen_policy import dataset_policy_identity

    roots = [Path(p).expanduser().resolve() for p in [original_root, filtered_root, *supplement_roots]]
    if len(set(roots)) != len(roots):
        raise ValueError("original, filtered and supplement dataset roots must be distinct")
    print("[clean lineage] indexing original and filtered datasets", flush=True)
    original, filtered = dataset_index(roots[0]), dataset_index(roots[1])
    for index in (original, filtered):
        dataset_feature_contract(index["manifest"])
        dataset_safety_label_contract(index["manifest"])
        dataset_policy_identity(index["manifest"])
    compatible_manifests(original["manifest"], filtered["manifest"])
    lineage = verify_filtered_rows(original, filtered, torch)
    all_ids = set().union(*original["seeds"].values())
    run_info = original["manifest"].get("metadata", {}).get("run_info", {})
    noise_seeds = {int(run_info.get("fault_seed", int(run_info.get("seed", 42))+2000)) & 0xFFFFFFFF}
    sources = []
    modes = set()
    for root in roots[2:]:
        print(f"[clean lineage] indexing supplement {root}", flush=True)
        current = dataset_index(root)
        md = current["manifest"].get("metadata", {})
        info = md.get("supplement_source", {})
        compatible_manifests(original["manifest"], current["manifest"])
        dataset_feature_contract(current["manifest"])
        if (current["manifest"].get("fixed_split") != "train" or info.get("split") != "train"
                or any(current["manifest"]["counts"][s] for s in ("validation", "test"))):
            raise ValueError(f"supplement must contain training rows only: {root}")
        if info.get("base_manifest_sha256") != original["sha256"]:
            raise ValueError(f"supplement belongs to a different original dataset: {root}")
        if not current["groups"]["train"] <= original["groups"]["train"]:
            raise ValueError(f"supplement contains non-training terrain groups: {root}")
        ids = set().union(*current["seeds"].values())
        if all_ids & ids:
            raise ValueError(f"duplicate candidate IDs across original/supplement sources: {root}")
        all_ids.update(ids)
        fault_seed = md.get("run_info", {}).get("fault_seed")
        if fault_seed is None or (int(fault_seed) & 0xFFFFFFFF) in noise_seeds:
            raise ValueError(f"supplement fault seed is missing or reused: {root}")
        noise_seeds.add(int(fault_seed) & 0xFFFFFFFF)
        checked = verify_directed_rows(current, torch, directed_audit_contract())
        if checked["fault_source"] in modes:
            raise ValueError("supply at most one bank per directed source")
        modes.add(checked["fault_source"])
        status_path = root / "collection_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else None
        sources.append({"root": str(root), "split": "train", "manifest_sha256": current["sha256"],
                        "samples": current["manifest"]["counts"]["train"], "fault_seed": fault_seed,
                        "directed_rows": checked, "original_manifest_sha256": info["base_manifest_sha256"],
                        "collection_status_as_recorded": status,
                        "protected_source_sha256": md.get("protected_source_sha256")})
    if sources and modes != {"lp_only", "vp_only", "dual"}:
        raise ValueError("comparison requires all three directed banks together")
    if sources and any(s["samples"] != 20000 for s in sources):
        raise ValueError("this comparison expects exactly 20,000 rows per directed bank")
    return {"base_manifest_sha256": filtered["sha256"], "base_root": str(filtered["root"]),
            "base_counts": filtered["manifest"]["counts"],
            "verified_original_manifest_sha256": original["sha256"], "original_base_root": str(original["root"]),
            "allowed_parent_manifest_sha256": [original["sha256"], filtered["sha256"]],
            "filtered_lineage": lineage, "additional_sources": sources,
            "training_samples": filtered["manifest"]["counts"]["train"]+sum(s["samples"] for s in sources),
            "validation_and_test": "filtered dataset only; identical across both arms; original test evaluated separately",
            "risk_parent_provenance": "original training provenance retained; only outcome network refit on declared rows"}