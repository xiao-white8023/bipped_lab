"""Bounded shard reads, base-split allowlists and multi-root training isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import torch

from .dataset import SPLIT_NAMES, SelectorShardDataset
from .frozen_policy import dataset_policy_identity
from .feature_contract import dataset_feature_contract
from .labels import dataset_safety_label_contract


def manifest_sha256(root):
    return hashlib.sha256((Path(root)/"manifest.json").read_bytes()).hexdigest()


def dataset_index(root):
    """Audit on-disk counts and identities; never load the entire feature set."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    groups, seeds = {}, {}
    for split in SPLIT_NAMES:
        groups[split], seeds[split] = set(), set()
        paths = sorted((root/split).glob("shard_*.pt"))
        if len(paths) != int(manifest["shards"][split]):
            raise RuntimeError(f"shard count mismatch: {root}/{split}")
        count = 0
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            size = len(data["features"])
            if len(data["group_ids"]) != size or len(data["scenario_seed"]) != size:
                raise RuntimeError(f"row identity count mismatch: {path}")
            count += size
            groups[split].update(map(str, data["group_ids"]))
            row_ids = set(map(int, data["scenario_seed"].reshape(-1)))
            if len(row_ids) != size or seeds[split] & row_ids:
                raise RuntimeError(f"duplicate candidate rows: {path}")
            seeds[split].update(row_ids)
        if count != int(manifest["counts"][split]):
            raise RuntimeError(f"sample count mismatch: {root}/{split}")
        expected = manifest.get("group_sha256", {}).get(split)
        actual = hashlib.sha256("\n".join(sorted(groups[split])).encode()).hexdigest()
        if expected is not None and expected != actual:
            raise RuntimeError(f"terrain group hash mismatch: {root}/{split}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right] or seeds[left] & seeds[right]:
            raise RuntimeError(f"dataset split leakage: {root}, {left}/{right}")
    return {"root": root, "manifest": manifest, "groups": groups, "seeds": seeds,
            "sha256": manifest_sha256(root)}


def compatible_manifests(base, other):
    # Paths and tiny parity-test errors may differ between machines. Weights,
    # encoders, precision and the observable/label contracts must match exactly.
    identities = [dataset_policy_identity(m) for m in (base, other)]
    transient = {"source_checkpoint", "reference_policy", "validation"}
    identities = [{k:v for k,v in identity.items() if k not in transient} for identity in identities]
    if identities[0] != identities[1]:
        raise ValueError("supplement frozen RENet identity differs from the base dataset")
    if dataset_feature_contract(base) != dataset_feature_contract(other):
        raise ValueError("supplement selector feature contract differs")
    if dataset_safety_label_contract(base) != dataset_safety_label_contract(other):
        raise ValueError("supplement safety label contract differs")
    for key in ("feature_shape", "label_order"):
        if base.get(key) != other.get(key):
            raise ValueError(f"supplement {key} differs")


def base_collection_partition(root, split):
    index = dataset_index(root)
    dataset_policy_identity(index["manifest"])
    dataset_feature_contract(index["manifest"])
    dataset_safety_label_contract(index["manifest"])
    groups = index["groups"][split]
    if not groups:
        raise ValueError(f"base dataset contains no {split} terrain groups")
    parsed = [re.fullmatch(r"terrain_seed(-?\d+):type(\d+):level(\d+)", group) for group in groups]
    if any(value is None for value in parsed):
        raise ValueError("supplement collection requires terrain_seed/type/level group IDs")
    seeds = {int(value[1]) for value in parsed}
    if len(seeds) != 1:
        raise ValueError("base partition spans several terrain seeds; collect one source dataset at a time")
    return index, groups, seeds.pop(), {(int(value[3]), int(value[2])) for value in parsed}


def restrict_terrain_distribution(distribution, pairs):
    allowed = torch.zeros_like(distribution.counts, dtype=torch.bool)
    for level, kind in pairs:
        if level >= allowed.shape[0] or kind >= allowed.shape[1]:
            raise ValueError("base terrain group is absent from the current histogram")
        allowed[level, kind] = True
    distribution.counts = torch.where(allowed, distribution.counts, 0)
    if distribution.sample_count <= 0:
        raise ValueError("base split allowlist and terrain histogram have no common support")
    return distribution


def audit_training_sources(base_root, supplement_roots=(), stress_roots=()):
    paths = [Path(p).expanduser().resolve() for p in [base_root, *supplement_roots, *stress_roots]]
    if len(set(paths)) != len(paths):
        raise ValueError("dataset roots must be distinct")
    base = dataset_index(paths[0])
    dataset_feature_contract(base["manifest"])
    dataset_safety_label_contract(base["manifest"])
    dataset_policy_identity(base["manifest"])
    all_ids = set().union(*base["seeds"].values())
    base_info = base["manifest"].get("metadata", {}).get("run_info", {})
    noise_seeds = {int(base_info.get("fault_seed", int(base_info.get("seed", 42))+2000)) & 0xFFFFFFFF}
    sources = []
    for number, path in enumerate(paths[1:]):
        current = dataset_index(path)
        manifest = current["manifest"]
        compatible_manifests(base["manifest"], manifest)
        meta = manifest.get("metadata", {})
        info = meta.get("supplement_source", {})
        expected_split = "train" if number < len(supplement_roots) else info.get("split")
        if number >= len(supplement_roots) and expected_split == "train":
            raise ValueError(f"stress source must be validation or test: {path}")
        if expected_split not in SPLIT_NAMES or info.get("split") != expected_split:
            raise ValueError(f"missing/incorrect supplement source split: {path}")
        if manifest.get("fixed_split") != expected_split or any(manifest["counts"][s] for s in SPLIT_NAMES if s != expected_split):
            raise ValueError(f"supplement must contain only its declared split: {path}")
        if info.get("base_manifest_sha256") != base["sha256"]:
            raise ValueError(f"supplement belongs to a different base dataset: {path}")
        if not current["groups"][expected_split] <= base["groups"][expected_split]:
            raise RuntimeError(f"supplement uses terrain groups outside original {expected_split}: {path}")
        ids = set().union(*current["seeds"].values())
        if all_ids & ids:
            raise RuntimeError(f"cross-root duplicate candidate IDs: {path}")
        all_ids |= ids
        fault_seed = meta.get("run_info", {}).get("fault_seed")
        if fault_seed is None or (int(fault_seed) & 0xFFFFFFFF) in noise_seeds:
            raise RuntimeError(f"independent supplements/stress sets require distinct fault seeds: {path}")
        noise_seeds.add(int(fault_seed) & 0xFFFFFFFF)
        sources.append({"root": str(path), "split": expected_split,
                        "manifest_sha256": current["sha256"], "samples": manifest["counts"][expected_split],
                        "fault_seed": fault_seed})
    return {"base_manifest_sha256": base["sha256"], "base_root": str(paths[0]),
            "base_counts": base["manifest"]["counts"], "additional_sources": sources,
            "training_samples": base["manifest"]["counts"]["train"]
                + sum(s["samples"] for s in sources[:len(supplement_roots)]),
            "validation_and_test": "original dataset only; stress sets reported separately"}


class SelectorTrainingDataset(SelectorShardDataset):
    """Flat shards from several roots; collate observable training fields only.

    Older/newer audit columns have different shapes. They are provenance, not
    network inputs, and must never enter a training batch or break its collation.
    """
    def __init__(self, roots, split="train"):
        datasets = [SelectorShardDataset(root, split) for root in roots]
        if not datasets:
            raise ValueError("at least one dataset root is required")
        self.root, self.manifest = datasets[0].root, datasets[0].manifest
        for other in datasets[1:]:
            compatible_manifests(self.manifest, other.manifest)
        self.shards = [path for dataset in datasets for path in dataset.shards]
        self.lengths = [size for dataset in datasets for size in dataset.lengths]
        self.cumulative, total = [], 0
        for size in self.lengths:
            total += size
            self.cumulative.append(total)
        self._cached_index, self._cached_payload = -1, None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return {"features": item["features"], "labels": item["labels"]}
