"""Audited 5/10/20-frame views; arbitrary independent directed training banks.

Legacy V4 files remain five-frame data. V5 history files require explicit
full-history semantics plus real frame counts and per-frame timestamps.
This module READS the new format; it does not upgrade the simulation collector.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from .directed_audit import directed_audit_contract, source_rejection_masks
from .feature_contract import selector_feature_contract
from .frozen_policy import assert_same_policy, dataset_policy_identity
from .labels import dataset_safety_label_contract
from .outcomes import outcome_classes
from .supplement_dataset import dataset_index

SPLITS = ("train", "validation", "test")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def full_history_contract(frames):
    if frames not in (5, 10, 20):
        raise ValueError("supported histories are 5/10/20 frames")
    c = selector_feature_contract()
    c.update(version=5, feature_shape=[frames, 283], history_initialization="require_full_real_history")
    return c


def validate_manifest(manifest):
    shape = manifest.get("feature_shape")
    if shape not in ([5, 283], [10, 283], [20, 283]):
        raise ValueError("dataset must declare real [5/10/20,283] features")
    frames = shape[0]
    md = manifest.get("metadata", {})
    if md.get("collection_complete") is not True:
        raise ValueError("dataset collection is incomplete")
    if manifest.get("label_order") != ["lp_1.0s", "vp_1.0s"]:
        raise ValueError("dataset must use the existing two one-second labels")
    expected = selector_feature_contract() if manifest.get("version") == 4 else full_history_contract(frames)
    if (manifest.get("version") not in (4, 5) or md.get("selector_feature_contract") != expected
            or (manifest.get("version") == 4 and frames != 5)):
        raise ValueError("history/feature contract differs; changing only a manifest shape is not a conversion")
    dataset_policy_identity(manifest)
    dataset_safety_label_contract(manifest)
    return frames, copy.deepcopy(expected)


def check_payload(data, manifest, location):
    frames, _ = validate_manifest(manifest)
    f, y = data["features"], data["labels"]
    n = len(f)
    if n == 0 or f.shape != (n, frames, 283) or y.shape != (n, 2):
        raise ValueError(f"invalid feature/label shape: {location}")
    outcome_classes(y)
    if manifest["version"] == 5:
        counts = data.get("history_valid_frames")
        times = data.get("history_timestamps")
        end = data.get("feature_end_time")
        start = data.get("candidate_time")
        if (counts is None or counts.shape != (n,) or (counts < frames).any()
                or not torch.isfinite(counts).all() or not torch.equal(counts, counts.round())):
            raise ValueError(f"full-history rows need genuine-frame counts >= {frames}: {location}")
        if (times is None or times.shape != (n, frames) or not torch.isfinite(times).all()
                or not torch.allclose(times[:, 1:].double() - times[:, :-1].double(),
                                      torch.full((n, frames-1), .1, dtype=torch.float64), atol=1e-5, rtol=0)):
            raise ValueError(f"full-history rows need continuous 10 Hz timestamps: {location}")
        if (end is None or start is None or end.shape != (n,) or start.shape != (n,)
                or not torch.isfinite(end).all() or not torch.isfinite(start).all()
                or not torch.allclose(times[:, -1].double(), end.double(), atol=1e-6, rtol=0)
                or not torch.allclose(start.double(), end.double(), atol=1e-6, rtol=0)):
            raise ValueError(f"history end and label boundary differ: {location}")
    return n


def scan_root(root):
    index = dataset_index(root)  # actual shard counts, groups, row IDs and split isolation
    frames, contract = validate_manifest(index["manifest"])
    index["stored_frames"], index["feature_contract"] = frames, contract
    index["lengths"], index["class_counts"] = {}, {}
    index["file_stats"] = {}
    for split in SPLITS:
        lengths, counts = [], torch.zeros(4, dtype=torch.long)
        paths = sorted((index["root"] / split).glob("shard_*.pt"))
        for number, path in enumerate(paths, 1):
            before = path.stat()
            data = torch.load(path, map_location="cpu", weights_only=False)
            lengths.append(check_payload(data, index["manifest"], path))
            counts += torch.bincount(outcome_classes(data["labels"]), minlength=4)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"shard changed during audit: {path}")
            index["file_stats"][str(path)] = [after.st_size, after.st_mtime_ns]
            if number % 10 == 0 or number == len(paths):
                print(f"[joint data] {index['root'].name}/{split} shards={number}/{len(paths)}", flush=True)
        index["lengths"][split], index["class_counts"][split] = lengths, counts.tolist()
    return index


def same_contract(first, other):
    assert_same_policy(dataset_policy_identity(first["manifest"]), dataset_policy_identity(other["manifest"]))
    if (first["feature_contract"] != other["feature_contract"]
            or dataset_safety_label_contract(first["manifest"]) != dataset_safety_label_contract(other["manifest"])):
        raise ValueError("source contracts differ: do not mix stored 5-frame and 20-frame banks")


def verify_lineage(original, filtered):
    """Verify exact retained rows, including every observable/audit field."""
    root = filtered["root"]
    report_path = root / "filter_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = filtered["manifest"]["metadata"].get("offline_filter", {})
    splits = report.get("splits_filtered")
    if (file_sha256(root / "original_manifest.json") != original["sha256"]
            or report.get("verified") is not True or info.get("source_manifest_sha256") != original["sha256"]
            or report.get("source_manifest_sha256") != original["sha256"] or not isinstance(splits, list)
            or not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLITS)
            or info.get("splits_filtered") != splits
            or report.get("includes_both_failure_outcomes_in_single_fault_scenarios") is not True):
        raise ValueError("filtered dataset does not identify the supplied original dataset")
    result = {}
    for split in SPLITS:
        outputs = sorted((root / split).glob("shard_*.pt"))
        position = 0
        stats = dict(before=0, removed=0, after=0, removed_lp_fault_vp_failure=0, removed_vp_fault_lp_failure=0)
        for path in sorted((original["root"] / split).glob("shard_*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=False)
            n, faults, labels = len(data["features"]), data["fault_metadata"], data["labels"].bool()
            if faults.shape != (n, 8) or not ((faults[:, 6:] == 0) | (faults[:, 6:] == 1)).all():
                raise ValueError("filter source needs binary active LP/VP flags")
            lp, vp = faults[:, 6].bool(), faults[:, 7].bool()
            a = lp & ~vp & labels[:, 1] if split in splits else torch.zeros(n, dtype=torch.bool)
            b = vp & ~lp & labels[:, 0] if split in splits else torch.zeros(n, dtype=torch.bool)
            ids = (~(a | b)).nonzero().flatten()
            stats["before"] += n
            stats["after"] += len(ids)
            stats["removed"] += int((a | b).sum())
            stats["removed_lp_fault_vp_failure"] += int(a.sum())
            stats["removed_vp_fault_lp_failure"] += int(b.sum())
            if len(ids):
                if position >= len(outputs):
                    raise ValueError("filtered dataset is missing a shard")
                child = torch.load(outputs[position], map_location="cpu", weights_only=False)
                if data.keys() != child.keys():
                    raise ValueError("filtered/source fields differ")
                for key, value in data.items():
                    actual = child[key]
                    if torch.is_tensor(value):
                        expected = value.index_select(0, ids)
                        ok = torch.is_tensor(actual) and expected.dtype == actual.dtype and torch.equal(expected, actual)
                    else:
                        ok = isinstance(actual, type(value)) and list(actual) == [value[i] for i in ids.tolist()]
                    if not ok:
                        raise ValueError(f"filtered rows differ from exact original selection: {outputs[position]}, {key}")
                position += 1
        recorded = report.get("by_split", {}).get(split, {})
        if (position != len(outputs) or any(recorded.get(k) != v for k, v in stats.items())
                or stats["before"] != original["manifest"]["counts"][split]
                or stats["after"] != filtered["manifest"]["counts"][split]):
            raise ValueError(f"filter ancestry/count mismatch: {split}")
        result[split] = stats
    return {"verified": True, "source_manifest_sha256": original["sha256"],
            "filter_report_sha256": file_sha256(report_path), "by_split": result}


def verify_directed(index):
    manifest = index["manifest"]
    md = manifest["metadata"]
    info = md.get("supplement_source", {})
    cfg = info.get("config", {})
    mode = cfg.get("fault_source")
    pair = {"lp_only": 1, "vp_only": 2, "dual": 3}.get(mode)
    expected_contract = directed_audit_contract()
    if manifest["version"] == 5:
        expected_contract.update(version=2, history_frames=index["stored_frames"])
    if pair is None or md.get("directed_audit_contract") != expected_contract:
        raise ValueError("unsupported directed source/history contract")
    counts = torch.zeros(4, dtype=torch.long)
    for path in sorted((index["root"] / "train").glob("shard_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        reasons = source_rejection_masks(mode, data["fault_metadata"], data.get("source_metadata"))
        if any(v.any() for v in reasons.values()):
            raise ValueError(f"directed source or healthy-side checks fail: {path}")
        targets = outcome_classes(data["labels"])
        if not ((targets == pair) | (targets == 0)).all():
            raise ValueError("directed bank contains an incompatible failure outcome")
        counts += torch.bincount(targets, minlength=4)
    quotas = cfg.get("pair_quotas")
    if quotas is not None and counts.tolist() != quotas:
        raise ValueError("actual directed counts differ from recorded quotas")
    if int(counts.sum()) != manifest["counts"]["train"]:
        raise ValueError("actual directed rows differ from manifest")
    return {"verified": True, "fault_source": mode, "pair_counts": counts.tolist()}


def audit_sources(base_root, supplement_roots=(), original_root=None):
    roots = [Path(p).expanduser().resolve() for p in [base_root, *supplement_roots]]
    if len(set(roots)) != len(roots):
        raise ValueError("dataset roots must be distinct; a bank must not be loaded twice")
    base = scan_root(roots[0])
    original, lineage = base, None
    if base["manifest"]["metadata"].get("offline_filter"):
        if original_root is None:
            raise ValueError("filtered data requires --original-dataset for exact ancestry verification")
        original = scan_root(original_root)
        if original["root"] in roots:
            raise ValueError("original dataset must be distinct from filtered base and banks")
        same_contract(original, base)
        lineage = verify_lineage(original, base)
    elif original_root is not None:
        raise ValueError("--original-dataset is only needed for a filtered base")
    all_ids = set().union(*original["seeds"].values())
    ri = original["manifest"].get("metadata", {}).get("run_info", {})
    noise_seeds = {int(ri.get("fault_seed", int(ri.get("seed", 42))+2000)) & 0xFFFFFFFF}
    indexes, banks = [base], []
    for root in roots[1:]:
        index = scan_root(root)
        same_contract(base, index)
        m, md = index["manifest"], index["manifest"]["metadata"]
        info = md.get("supplement_source", {})
        if (m.get("fixed_split") != "train" or info.get("split") != "train"
                or any(m["counts"][s] for s in ("validation", "test"))
                or info.get("base_manifest_sha256") != original["sha256"]):
            raise ValueError("supplement must be train-only and refer to the original base manifest")
        if not index["groups"]["train"] <= original["groups"]["train"]:
            raise ValueError("supplement terrain group is outside the original training partition")
        ids = set().union(*index["seeds"].values())
        fault_seed = md.get("run_info", {}).get("fault_seed")
        if all_ids & ids or fault_seed is None or (int(fault_seed) & 0xFFFFFFFF) in noise_seeds:
            raise ValueError("supplement has duplicate sample IDs or a reused/missing fault seed")
        all_ids.update(ids)
        noise_seeds.add(int(fault_seed) & 0xFFFFFFFF)
        checked = verify_directed(index)
        banks.append({"root": str(root), "manifest_sha256": index["sha256"], "samples": m["counts"]["train"],
                      "fault_seed": fault_seed, "directed_rows": checked,
                      "collection_status_as_recorded": json.loads((root/"collection_status.json").read_text())
                          if (root/"collection_status.json").is_file() else None})
        indexes.append(index)
    counts = [sum(i["class_counts"]["train"][c] for i in indexes) for c in range(4)]
    audit = {"base_root": str(base["root"]), "base_manifest_sha256": base["sha256"],
        "base_counts": base["manifest"]["counts"], "stored_history_frames": base["stored_frames"],
        "filtered_lineage": lineage, "original_root": str(original["root"]), "original_manifest_sha256": original["sha256"],
        "additional_sources": banks, "training_samples": sum(counts), "train_class_counts": counts,
        "train_class_order": list(("both_safe", "lp_only_failure", "vp_only_failure", "both_failure")),
        "validation_and_test": "base split only; supplements never enter validation/test",
        "inputs_sha256": {str(i["root"]/"manifest.json"): i["sha256"] for i in [original, *indexes]}}
    if lineage:
        audit["inputs_sha256"][str(base["root"]/"filter_report.json")] = lineage["filter_report_sha256"]
        audit["inputs_sha256"][str(base["root"]/"original_manifest.json")] = original["sha256"]
    return indexes, original, audit


def assert_inputs_unchanged(indexes, audit):
    for path, expected in audit["inputs_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"input metadata changed: {path}")
    for index in indexes:
        for path, expected in index["file_stats"].items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise RuntimeError(f"input shard changed: {path}")


class JointShardDataset(Dataset):
    """Sequential shards or a read-only row cache; expose no audit metadata."""
    def __init__(self, indexes, split, history_frames):
        if split not in SPLITS or not indexes:
            raise ValueError("valid split and nonempty source indexes required")
        if split != "train" and len(indexes) != 1:
            raise ValueError("evaluation must use a single explicit split source")
        self.history_frames, self.split = history_frames, split
        self.root, self.manifest = indexes[0]["root"], indexes[0]["manifest"]
        self.shards, self.lengths = [], []
        self.source_counts, self.source_roots, self.shard_stats = [], [], {}
        for index in indexes:
            if history_frames not in (5, 10, 20) or history_frames > index["stored_frames"]:
                raise ValueError(f"requested {history_frames} frames but data stores {index['stored_frames']}; no padding/conversion")
            self.shards.extend(sorted((index["root"] / split).glob("shard_*.pt")))
            self.lengths.extend(index["lengths"][split])
            self.source_counts.append(sum(index["lengths"][split]))
            self.source_roots.append(str(index["root"]))
            self.shard_stats.update(index["file_stats"])
        self.cumulative, total = [], 0
        for size in self.lengths:
            total += size
            self.cumulative.append(total)
        self._cached_index, self._cached_payload = -1, None
        self._cache_paths, self._memmaps, self._cache_stats = None, None, {}

    @property
    def random_access_ready(self):
        return self._cache_paths is not None

    def prepare_random_access(self, directory):
        """One sequential copy to mmap arrays, avoiding one torch.load per random row.

        Only genuine selected history, labels and IDs are cached, in unchanged
        source order. Memory is mapped on demand and shared by the OS across
        workers. No source dataset is modified and no training row is resampled.
        """
        if self.split != "train" or not len(self):
            raise ValueError("random access cache requires nonempty training data")
        directory = Path(directory).resolve()
        if self.random_access_ready or directory.exists():
            raise ValueError("choose a new random-access cache directory")
        directory.parent.mkdir(parents=True, exist_ok=True)
        shapes = {"features": (len(self), self.history_frames, 283),
                  "labels": (len(self), 2), "sample_id": (len(self),)}
        dtypes = {"features": np.dtype("float32"), "labels": np.dtype("float32"),
                  "sample_id": np.dtype("int64")}
        required = sum(math.prod(shape)*dtypes[key].itemsize for key, shape in shapes.items())
        if shutil.disk_usage(directory.parent).free < required + 1024*1024:
            raise RuntimeError(f"random-access cache needs at least {required} bytes of free disk space")
        metadata = {"version": 1, "split": "train", "rows": len(self),
            "history_frames": self.history_frames, "payload_bytes": required,
            "source_roots": self.source_roots, "source_counts": self.source_counts,
            "source_shards": [{"path": str(p), "size_mtime_ns": self.shard_stats[str(p)]} for p in self.shards],
            "arrays": {k: {"shape": list(v), "dtype": str(dtypes[k])} for k, v in shapes.items()}}
        with tempfile.TemporaryDirectory(prefix=".joint_cache_build_", dir=directory.parent) as temporary:
            temporary = Path(temporary)
            arrays = {key: np.lib.format.open_memmap(temporary/f"{key}.npy", mode="w+",
                dtype=dtypes[key], shape=shape) for key, shape in shapes.items()}
            offset = 0
            try:
                for number, (path, count) in enumerate(zip(self.shards, self.lengths), 1):
                    stat = path.stat()
                    if [stat.st_size, stat.st_mtime_ns] != self.shard_stats[str(path)]:
                        raise RuntimeError(f"source shard changed before cache creation: {path}")
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    arrays["features"][offset:offset+count] = data["features"][:, -self.history_frames:].float().numpy()
                    arrays["labels"][offset:offset+count] = data["labels"].float().numpy()
                    arrays["sample_id"][offset:offset+count] = data["scenario_seed"].reshape(-1).long().numpy()
                    stat = path.stat()
                    if [stat.st_size, stat.st_mtime_ns] != self.shard_stats[str(path)]:
                        raise RuntimeError(f"source shard changed during cache creation: {path}")
                    offset += count
                    del data
                    if number == 1 or number % 10 == 0 or number == len(self.shards):
                        print(f"[joint cache] shards={number}/{len(self.shards)} rows={offset}/{len(self)}", flush=True)
                if offset != len(self):
                    raise RuntimeError("random-access cache row count mismatch")
                for array in arrays.values():
                    array.flush()
                (temporary/"cache_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
            finally:
                # Close mmap handles before atomically moving the completed directory.
                for array in arrays.values():
                    array._mmap.close()
                arrays.clear()
            os.replace(temporary, directory)
        self._cache_paths = {key: directory/f"{key}.npy" for key in shapes}
        for path in [*self._cache_paths.values(), directory/"cache_manifest.json"]:
            stat = path.stat()
            self._cache_stats[str(path)] = [stat.st_size, stat.st_mtime_ns]
        self._cached_index, self._cached_payload = -1, None
        return {**metadata, "directory": str(directory), "access": "read_only_numpy_mmap"}

    def assert_random_access_unchanged(self):
        for path, expected in self._cache_stats.items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise RuntimeError(f"training row cache changed: {path}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_memmaps=None, _cached_index=-1, _cached_payload=None)
        return state

    def __getitems__(self, indices):
        """PyTorch batched read: arbitrary rows, restored to sampler order."""
        if not self.random_access_ready:
            return [self[index] for index in indices]
        if self._memmaps is None:
            self._memmaps = {k: np.load(p, mmap_mode="r", allow_pickle=False) for k, p in self._cache_paths.items()}
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= len(self)).any():
            raise IndexError("invalid random-access row indices")
        # Writable batch copies protect immutable cache pages from downstream code.
        values = {k: torch.from_numpy(np.array(v[ids], copy=True)) for k, v in self._memmaps.items()}
        return [{k: v[row] for k, v in values.items()} for row in range(len(ids))]

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.random_access_ready:
            return self.__getitems__([index])[0]
        shard = bisect_right(self.cumulative, index)
        start = self.cumulative[shard-1] if shard else 0
        local = index-start
        if self._cached_index != shard:
            self._cached_payload = torch.load(self.shards[shard], map_location="cpu", weights_only=False)
            self._cached_index = shard
        data = self._cached_payload
        return {"features": data["features"][local, -self.history_frames:].float(),
                "labels": data["labels"][local].float(), "sample_id": data["scenario_seed"][local]}