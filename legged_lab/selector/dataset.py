"""Sharded selector dataset with deterministic group-level splitting."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from .config import CollectionConfig, SelectorConfig
from .labels import FUTURE_METRIC_ORDER
from .collection_audit import FAULT_PARAMETER_ORDER, CANDIDATE_SAFETY_ORDER


SPLIT_NAMES = ("train", "validation", "test")


def verify_dataset_no_group_leak(root: str | Path) -> dict[str, int]:
    """Audit persisted shards for terrain/episode and scenario-seed leakage."""

    root = Path(root)
    groups: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    seeds: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for path in sorted((root / split).glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            groups[split].update(str(value) for value in payload["group_ids"])
            seeds[split].update(
                int(value) for value in payload["scenario_seed"].reshape(-1)
            )
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        group_overlap = groups[left] & groups[right]
        if group_overlap:
            raise RuntimeError(
                f"dataset group leakage between {left} and {right}: {sorted(group_overlap)[:5]}"
            )
        seed_overlap = seeds[left] & seeds[right]
        if seed_overlap:
            raise RuntimeError(
                f"scenario-seed leakage between {left} and {right}: {sorted(seed_overlap)[:5]}"
            )
    return {split: len(groups[split]) for split in SPLIT_NAMES}


def split_from_group(
    group_id: str | int,
    probabilities: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> str:
    """Assign an entire episode/terrain group to one stable split."""

    if abs(sum(probabilities) - 1.0) > 1.0e-6:
        raise ValueError("split probabilities must sum to one")
    digest = hashlib.sha256(str(group_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < probabilities[0]:
        return "train"
    if value < probabilities[0] + probabilities[1]:
        return "validation"
    return "test"


@dataclass
class SelectorBatch:
    features: torch.Tensor
    labels: torch.Tensor
    group_ids: list[str]
    terrain_level: torch.Tensor
    terrain_type: torch.Tensor
    scenario_seed: torch.Tensor
    future_metrics: torch.Tensor | None = None
    fault_metadata: torch.Tensor | None = None
    candidate_time: torch.Tensor | None = None
    feature_end_time: torch.Tensor | None = None
    execution_metadata: torch.Tensor | None = None
    event_identity: torch.Tensor | None = None
    fault_parameters: torch.Tensor | None = None
    first_failure_flags: torch.Tensor | None = None
    first_failure_cause_stable: torch.Tensor | None = None
    candidate_safety: torch.Tensor | None = None

    def validate(self, selector_cfg: SelectorConfig | None = None) -> None:
        selector_cfg = selector_cfg or SelectorConfig()
        batch = self.features.shape[0]
        if self.features.shape != (
            batch,
            selector_cfg.history_frames,
            selector_cfg.frame_dim,
        ):
            raise ValueError("features do not match the [batch, 5, 283] contract")
        if self.labels.shape != (batch, 4):
            raise ValueError("labels must have shape [batch, 4]")
        if len(self.group_ids) != batch:
            raise ValueError("group_ids length must equal batch size")
        for name in ("terrain_level", "terrain_type", "scenario_seed"):
            if getattr(self, name).reshape(-1).shape[0] != batch:
                raise ValueError(f"{name} length must equal batch size")
        labels = self.labels.bool()
        if torch.any(labels[:, 0] & ~labels[:, 1]) or torch.any(
            labels[:, 2] & ~labels[:, 3]
        ):
            raise ValueError(
                "short-horizon failures must also be long-horizon failures"
            )
        if self.future_metrics is not None and self.future_metrics.shape != (
            batch,
            2,
            len(FUTURE_METRIC_ORDER),
        ):
            raise ValueError(
                "future_metrics must contain the current audit columns for both branches"
            )
        if self.fault_metadata is not None and self.fault_metadata.shape != (batch, 8):
            raise ValueError("fault_metadata must have shape [batch, 8]")
        if (self.candidate_time is None) != (self.feature_end_time is None):
            raise ValueError(
                "candidate and feature timestamps must be supplied together"
            )
        if self.candidate_time is not None:
            for value in (self.candidate_time, self.feature_end_time):
                if (
                    value.shape != (batch,)
                    or value.dtype != torch.float64
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        "audit timestamps must be finite float64 [batch] tensors"
                    )
            if not torch.allclose(
                self.candidate_time, self.feature_end_time, rtol=0, atol=1e-6
            ):
                raise ValueError(
                    "candidate snapshot and last feature frame are not aligned"
                )
        if self.execution_metadata is not None and self.execution_metadata.shape != (
            batch,
            3,
        ):
            raise ValueError(
                "execution_metadata must contain mode, dwell and remaining blend steps"
            )
        if self.event_identity is not None and (
            self.event_identity.shape != (batch, 3)
            or self.event_identity.dtype != torch.int64
            or (self.event_identity < 0).any()
        ):
            raise ValueError("event_identity must be nonnegative int64 [batch, 3]")
        for name, columns in (
            ("fault_parameters", len(FAULT_PARAMETER_ORDER)),
            ("candidate_safety", len(CANDIDATE_SAFETY_ORDER)),
        ):
            value = getattr(self, name)
            if value is not None and (
                value.shape != (batch, columns) or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must have finite [batch, {columns}] values")
        if (self.first_failure_flags is None) != (self.first_failure_cause_stable is None):
            raise ValueError("first failure flags and replay agreement must be supplied together")
        if self.first_failure_flags is not None:
            if (self.first_failure_flags.shape != (batch, 2)
                or self.first_failure_flags.dtype != torch.uint8
                or (self.first_failure_flags > 15).any()):
                raise ValueError("first_failure_flags must be uint8 [batch, 2] bitmasks")
            if (self.first_failure_cause_stable.shape != (batch, 2)
                or self.first_failure_cause_stable.dtype != torch.bool):
                raise ValueError("first_failure_cause_stable must be bool [batch, 2]")
            if not torch.equal(self.first_failure_flags != 0, labels[:, [1, 3]]):
                raise ValueError("first failure flags must agree with long-horizon labels")


class SelectorDatasetWriter:
    """Write bounded-size ``.pt`` shards and a self-describing manifest."""

    def __init__(
        self,
        root: str | Path,
        *,
        collection_cfg: CollectionConfig | None = None,
        selector_cfg: SelectorConfig | None = None,
        metadata: dict[str, Any] | None = None,
        fixed_split: str | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.collection_cfg = collection_cfg or CollectionConfig()
        self.selector_cfg = selector_cfg or SelectorConfig()
        self.collection_cfg.validate()
        self.selector_cfg.validate()
        if fixed_split is not None and fixed_split not in SPLIT_NAMES:
            raise ValueError(f"fixed_split must be one of {SPLIT_NAMES}")
        self.fixed_split = fixed_split
        self._pending: dict[str, list[dict[str, Any]]] = {
            name: [] for name in SPLIT_NAMES
        }
        self._pending_count = {name: 0 for name in SPLIT_NAMES}
        self._shard_index = {name: 0 for name in SPLIT_NAMES}
        self._counts = {name: 0 for name in SPLIT_NAMES}
        self._groups = {name: set() for name in SPLIT_NAMES}
        self.metadata = dict(metadata or {})
        self._optional_fields: frozenset[str] | None = None
        for split in SPLIT_NAMES:
            (self.root / split).mkdir(exist_ok=True)

    def append(self, batch: SelectorBatch) -> None:
        batch.validate(self.selector_cfg)
        optional_names = (
            "future_metrics", "fault_metadata", "candidate_time", "feature_end_time",
            "execution_metadata", "event_identity", "fault_parameters",
            "first_failure_flags", "first_failure_cause_stable", "candidate_safety",
        )
        fields = frozenset(name for name in optional_names if getattr(batch, name) is not None)
        if self._optional_fields is not None and fields != self._optional_fields:
            raise ValueError("optional audit fields must be consistent across batches")
        self._optional_fields = fields
        assignments: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
        for index, group_id in enumerate(batch.group_ids):
            split = self.fixed_split or split_from_group(
                group_id, self.collection_cfg.split_probabilities
            )
            assignments[split].append(index)
            self._groups[split].add(str(group_id))
        for split, indices in assignments.items():
            if not indices:
                continue
            ids = torch.tensor(indices, dtype=torch.long, device=batch.features.device)
            item: dict[str, Any] = {
                "features": batch.features[ids].detach().cpu().to(torch.float16),
                "labels": batch.labels[ids].detach().cpu().to(torch.uint8),
                "group_ids": [batch.group_ids[index] for index in indices],
                "terrain_level": batch.terrain_level.reshape(-1)[ids]
                .detach()
                .cpu()
                .to(torch.int8),
                "terrain_type": batch.terrain_type.reshape(-1)[ids]
                .detach()
                .cpu()
                .to(torch.int16),
                "scenario_seed": batch.scenario_seed.reshape(-1)[ids]
                .detach()
                .cpu()
                .to(torch.int64),
            }
            if batch.future_metrics is not None:
                item["future_metrics"] = (
                    batch.future_metrics[ids].detach().cpu().to(torch.float16)
                )
            if batch.fault_metadata is not None:
                item["fault_metadata"] = (
                    batch.fault_metadata[ids].detach().cpu().to(torch.int16)
                )
            for name in ("candidate_time", "feature_end_time", "execution_metadata"):
                value = getattr(batch, name)
                if value is not None:
                    item[name] = value[ids].detach().cpu()
            for name, dtype in (
                ("event_identity", torch.int64), ("fault_parameters", torch.float32),
                ("first_failure_flags", torch.uint8),
                ("first_failure_cause_stable", torch.bool),
                ("candidate_safety", torch.float32),
            ):
                value = getattr(batch, name)
                if value is not None:
                    item[name] = value[ids].detach().cpu().to(dtype)
            self._pending[split].append(item)
            self._pending_count[split] += len(indices)
            while self._pending_count[split] >= self.collection_cfg.shard_size:
                self._flush_one(split, self.collection_cfg.shard_size)

    @staticmethod
    def _take_rows(chunks: list[dict[str, Any]], count: int) -> dict[str, Any]:
        selected: list[dict[str, Any]] = []
        remaining = count
        while remaining > 0:
            chunk = chunks[0]
            chunk_count = chunk["features"].shape[0]
            take = min(remaining, chunk_count)
            head: dict[str, Any] = {}
            tail: dict[str, Any] = {}
            for key, value in chunk.items():
                if key == "group_ids":
                    head[key], tail[key] = value[:take], value[take:]
                else:
                    head[key], tail[key] = value[:take], value[take:]
            selected.append(head)
            if take == chunk_count:
                chunks.pop(0)
            else:
                chunks[0] = tail
            remaining -= take
        output: dict[str, Any] = {}
        for key in selected[0]:
            if key == "group_ids":
                output[key] = [group for item in selected for group in item[key]]
            else:
                output[key] = torch.cat([item[key] for item in selected], dim=0)
        return output

    def _flush_one(self, split: str, count: int) -> None:
        payload = self._take_rows(self._pending[split], count)
        index = self._shard_index[split]
        destination = self.root / split / f"shard_{index:05d}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        self._shard_index[split] += 1
        self._pending_count[split] -= count
        self._counts[split] += count

    def close(self) -> Path:
        for split in SPLIT_NAMES:
            if self._pending_count[split]:
                self._flush_one(split, self._pending_count[split])
        overlap = (
            (self._groups["train"] & self._groups["validation"])
            | (self._groups["train"] & self._groups["test"])
            | (self._groups["validation"] & self._groups["test"])
        )
        if overlap:
            raise RuntimeError(f"dataset group leakage detected: {sorted(overlap)[:5]}")
        manifest = {
            "version": 3,
            "execution_metadata_order": ["mode", "dwell_s", "remaining_blend_steps"],
            "feature_shape": [
                self.selector_cfg.history_frames,
                self.selector_cfg.frame_dim,
            ],
            "label_order": ["lp_0.4s", "lp_1.0s", "vp_0.4s", "vp_1.0s"],
            "future_metric_order_per_branch": list(FUTURE_METRIC_ORDER),
            "counts": self._counts,
            "shards": self._shard_index,
            "group_sha256": {
                split: hashlib.sha256(
                    "\n".join(sorted(groups)).encode("utf-8")
                ).hexdigest()
                for split, groups in self._groups.items()
            },
            "metadata": self.metadata,
            "fixed_split": self.fixed_split,
        }
        destination = self.root / "manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination


class SelectorShardDataset(Dataset):
    """Map-style reader that keeps only the most recently used shard in RAM."""

    def __init__(self, root: str | Path, split: str):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"selector manifest does not exist: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards = sorted((self.root / split).glob("shard_*.pt"))
        expected_shards = int(self.manifest["shards"][split])
        if len(self.shards) != expected_shards:
            raise RuntimeError(
                f"expected {expected_shards} {split} shards, found {len(self.shards)}"
            )
        self.lengths: list[int] = []
        for path in self.shards:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.lengths.append(int(payload["features"].shape[0]))
        self.cumulative: list[int] = []
        total = 0
        for length in self.lengths:
            total += length
            self.cumulative.append(total)
        if total != int(self.manifest["counts"][split]):
            raise RuntimeError("manifest sample count does not match shard contents")
        self._cached_index = -1
        self._cached_payload: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - start
        if shard_index != self._cached_index:
            self._cached_payload = torch.load(
                self.shards[shard_index], map_location="cpu", weights_only=False
            )
            self._cached_index = shard_index
        assert self._cached_payload is not None
        item: dict[str, Any] = {}
        for key, value in self._cached_payload.items():
            if key == "group_ids":
                item[key] = value[local_index]
            else:
                item[key] = value[local_index]
        item["features"] = item["features"].float()
        item["labels"] = item["labels"].float()
        return item


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle shard order and rows without loading a shard for every row.

    A batch visits each shard in contiguous runs. Small shard remainders are
    carried into the next batch so batch size and sample coverage are preserved.
    Each DataLoader worker retains its own one-shard cache; memory stays bounded.
    """

    def __init__(
        self,
        dataset: SelectorShardDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator

    def __len__(self) -> int:
        samples = len(self.dataset)
        return (samples // self.batch_size if self.drop_last else
                (samples + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        shard_order = (torch.randperm(len(self.dataset.lengths), generator=self.generator).tolist()
                       if self.shuffle else range(len(self.dataset.lengths)))
        pending: list[int] = []
        for shard_index in shard_order:
            size = self.dataset.lengths[shard_index]
            start = 0 if shard_index == 0 else self.dataset.cumulative[shard_index - 1]
            rows = (torch.randperm(size, generator=self.generator) + start if self.shuffle
                    else torch.arange(start, start + size)).tolist()
            offset = 0
            if pending:
                take = min(self.batch_size - len(pending), size)
                pending.extend(rows[:take])
                offset = take
                if len(pending) == self.batch_size:
                    yield pending
                    pending = []
            while offset + self.batch_size <= size:
                yield rows[offset:offset + self.batch_size]
                offset += self.batch_size
            pending.extend(rows[offset:])
        if pending and not self.drop_last:
            yield pending

