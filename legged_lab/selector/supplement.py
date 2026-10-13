"""Supplement profiles, actual-outcome quotas and event diversity constraints."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math
import torch

from .config import FaultConfig
from .collection_audit import PAIR_NAMES


PROFILE_NAMES = ("baseline", "mixed", "lp_map", "vp_hard", "dual", "recovery")
LIFECYCLE_NAMES = (
    "normal", "other_fault", "drift_early", "drift_mature",
    "person_approach", "person_near", "person_depart", "person_residual",
    "person_cleared", "recovered",
)


def lifecycle_classes(faults, parameters, noise):
    """Classify the *current* observation; no future event state is used.

    These classes are audit fields, never selector features or safety labels.
    Person residuals require retained cells, rather than just an elapsed timer.
    """
    n = len(faults)
    stage = torch.zeros(n, dtype=torch.long, device=faults.device)
    active = faults[:, 6:].bool().any(-1)
    stage[active] = 1
    planned = faults[:, :2].ne(0).any(-1)
    stage[~active & planned & (faults[:, 3] == 1)] = 9
    lp_active = faults[:, 6].bool()
    drift = lp_active & ((faults[:, 0] == 12) | (faults[:, 0] == 14))
    age = (-parameters[:, 4]).clamp_min(0)
    stage[drift & (age < 2.)] = 2
    stage[drift & (age >= 2.)] = 3
    person = lp_active & ((faults[:, 0] == 15) | (faults[:, 0] == 16))
    if noise is not None:
        phase = noise[:, 7].long()
        for value in (0, 1, 2):
            stage[person & (phase == value)] = 4 + value
        stage[person & (phase == 3) & (noise[:, 8] > 0)] = 7
        stage[person & (phase == 3) & (noise[:, 8] == 0)] = 8
    return stage


def fault_profile(name: str, base: FaultConfig | None = None) -> FaultConfig:
    """Weights use the stable extended LP/VP kind order from faults.py."""
    base = base or FaultConfig()
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown fault profile: {name}")
    if name == "baseline":
        return base
    # LP-only banks cannot generate the rare VP-fails / LP-safe class reliably.
    # Mixed collection therefore gives VP-only events the largest proposal share.
    scenarios = {
        "mixed": (.10, .40, .30, .20), "lp_map": (.05, .05, .85, .05),
        "vp_hard": (.05, .85, .05, .05), "dual": (.05, .10, .10, .75),
        "recovery": (.25, .25, .25, .25),
    }[name]
    lp_weights = (.02, .04, .03, .04, .07, .07, .05, .03, .25, .25, .15)
    vp_weights = (.04, .10, .12, .12, .12, .12, .03, .25, .10)
    cfg = replace(base, extended_faults=True, profile=name,
        lp_kind_weights=lp_weights, vp_kind_weights=vp_weights,
        scenario_probabilities=scenarios,
        min_duration_frames=40, max_duration_frames=120,
        max_start_delay_frames=10, min_recovery_frames=12, max_recovery_frames=25,
        max_dual_stagger_frames=5, vp_occlusion_fraction=(.20, .85),
        vp_scale_range=(.55, 1.55), vp_bias_m_range=(-.60, .60), vp_max_shift_pixels=6,
        lp_height_bias_m_range=(-.30, .30), lp_translation_range_m=(-.40, .40),
        lp_yaw_range_deg=(-12., 12.), lp_drift_m_per_s=(.02, .12),
        lp_drift_deg_per_s=(.3, 3.), severity_probabilities=(.15, .35, .50))
    if name == "recovery":
        cfg = replace(cfg, severity_probabilities=(.45, .40, .15),
                      min_recovery_frames=25, max_recovery_frames=50,
                      lp_map_random_walk_m=.004)
    cfg.validate()
    return cfg


@dataclass(frozen=True)
class SupplementConfig:
    split: str = "train"
    pair_quotas: tuple[int, int, int, int] | None = None
    max_samples_per_event: int = 8
    min_event_spacing_s: float = .2
    require_initial_safe: bool = True
    max_no_progress_batches: int = 1000
    # Dataclass defaults preserve existing callers. The production supplement
    # CLI explicitly enables 15 s real burn-in and 30 s paced collection.
    warmup_s: float = 0.
    min_collection_span_s: float = 0.
    label_interval_frames: int = 1
    randomize_label_phase: bool = False
    max_samples_per_labelled_batch: int = 0

    def validate(self, target_samples):
        if self.split not in ("train", "validation", "test"):
            raise ValueError("invalid supplement split")
        if self.split != "train" and self.pair_quotas is not None:
            raise ValueError("stress evaluation must use predefined faults without outcome quotas")
        if self.pair_quotas is not None and (
            len(self.pair_quotas) != 4 or any(type(q) is not int or q < 0 for q in self.pair_quotas)
            or sum(self.pair_quotas) != target_samples):
            raise ValueError("four pair quotas must sum exactly to target_samples")
        if self.max_samples_per_event < 1 or not math.isfinite(self.min_event_spacing_s) or self.min_event_spacing_s < .2:
            raise ValueError("event cap must be positive and spacing must be at least 0.2 s")
        if type(self.max_no_progress_batches) is not int or self.max_no_progress_batches < 1:
            raise ValueError("max_no_progress_batches must be a positive integer")
        for name in ("warmup_s", "min_collection_span_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.label_interval_frames) is not int or self.label_interval_frames < 1:
            raise ValueError("label_interval_frames must be a positive integer")
        if type(self.randomize_label_phase) is not bool:
            raise ValueError("randomize_label_phase must be boolean")
        if type(self.max_samples_per_labelled_batch) is not int or self.max_samples_per_labelled_batch < 0:
            raise ValueError("max_samples_per_labelled_batch must be a nonnegative integer")


def scaled_pair_quotas(total):
    quotas = [int(total*w) for w in (.25, .25, .375, .125)]
    quotas[0] += total-sum(quotas)
    return tuple(quotas)


def label_classes(labels):
    """Pair index: both safe, LP fails, VP fails, both fail within one second."""
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("outcome labels must be [n,2] [LP_1s, VP_1s]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("outcome labels must be binary")
    return labels[:, 0].long() + 2 * labels[:, 1].long()


class OutcomeQuotaSampler:
    """Runs only on the real candidate path, outside counterfactual forks."""

    def __init__(self, cfg: SupplementConfig, target_samples: int, seed: int):
        cfg.validate(target_samples)
        self.cfg, self.target = cfg, target_samples
        self.generator = torch.Generator().manual_seed(seed)
        self.timing_generator = torch.Generator().manual_seed(seed+991)
        self._label_bucket, self._label_tick = None, None
        self.counts = [0]*4
        self.events = {}
        self.rejections = {name: 0 for name in ("event_cap", "event_spacing", "outcome_quota")}
        self.eligible_counts = [0]*4
        self.collection_start = None
        self.last_time = None
        self.seen_lifecycle = [0]*len(LIFECYCLE_NAMES)
        self.eligible_lifecycle = [0]*len(LIFECYCLE_NAMES)
        self.written_lifecycle = [0]*len(LIFECYCLE_NAMES)
        self.eligible_1s_by_lifecycle = [[0]*4 for _ in LIFECYCLE_NAMES]
        self.written_1s_by_lifecycle = [[0]*4 for _ in LIFECYCLE_NAMES]
        self.written_time_bins = {}
        self.rejections["temporal_budget"] = 0

    def start(self, now):
        if self.collection_start is not None:
            raise RuntimeError("collection time origin is already fixed")
        self.collection_start = self.last_time = float(now)

    def observe_time(self, now):
        if self.collection_start is None:
            self.start(now)
        if float(now) < self.last_time - 1e-6:
            raise ValueError("real collection time moved backwards")
        self.last_time = float(now)

    def sample_budget(self, now):
        self.observe_time(now)
        span = self.cfg.min_collection_span_s
        elapsed = self.last_time - self.collection_start
        fraction = min(1., elapsed / span) if span else 1.
        released = min(self.target, int(math.floor(self.target*fraction + 1e-6)))
        room = max(0, released-sum(self.counts))
        if self.cfg.max_samples_per_labelled_batch:
            room = min(room, self.cfg.max_samples_per_labelled_batch)
        return room

    def should_label_tick(self, tick):
        interval = self.cfg.label_interval_frames
        if not self.cfg.randomize_label_phase:
            return tick % interval == 0
        # One aligned 10 Hz snapshot in each interval, with an independent
        # offset. A fixed 1 s grid can alias 0.2 s/0.5 s on-off fault cycles.
        bucket = tick // interval
        if bucket != self._label_bucket:
            self._label_bucket = bucket
            offset = int(torch.randint(interval, (1,), generator=self.timing_generator))
            self._label_tick = bucket*interval+offset
        return tick == self._label_tick

    def record_unlabelled(self, faults, parameters, noise):
        stages = lifecycle_classes(faults, parameters, noise).cpu()
        for stage, count in enumerate(torch.bincount(stages, minlength=len(LIFECYCLE_NAMES)).tolist()):
            self.seen_lifecycle[stage] += count

    def select(self, good, labels, event_identity, now, *, faults=None, parameters=None, noise=None):
        budget = self.sample_budget(now)
        pairs = label_classes(labels).cpu()
        candidates = good.nonzero().flatten().cpu()
        events = event_identity.cpu().tolist()
        stages = (lifecycle_classes(faults, parameters, noise).cpu()
                  if faults is not None else torch.ones(len(labels), dtype=torch.long))
        if faults is not None:
            self.record_unlabelled(faults, parameters, noise)
        else:
            self.seen_lifecycle[1] += len(labels)
        for i in candidates.tolist():
            self.eligible_counts[int(pairs[i])] += 1
            self.eligible_lifecycle[int(stages[i])] += 1
            self.eligible_1s_by_lifecycle[int(stages[i])][int(pairs[i])] += 1
        order = candidates[torch.randperm(len(candidates), generator=self.generator)].tolist()
        selected = []
        for position, i in enumerate(order):
            if sum(self.counts) >= self.target:
                break
            if len(selected) >= budget:
                self.rejections["temporal_budget"] += len(order)-position
                break
            key = tuple(events[i])
            count, last = self.events.get(key, (0, -math.inf))
            if count >= self.cfg.max_samples_per_event:
                self.rejections["event_cap"] += 1
                continue
            if now-last < self.cfg.min_event_spacing_s-1e-7:
                self.rejections["event_spacing"] += 1
                continue
            category = int(pairs[i])
            if self.cfg.pair_quotas is not None:
                room = self.cfg.pair_quotas[category]-self.counts[category]
                if room <= 0:
                    self.rejections["outcome_quota"] += 1
                    continue
            self.counts[category] += 1
            self.events[key] = (count+1, now)
            self.written_lifecycle[int(stages[i])] += 1
            self.written_1s_by_lifecycle[int(stages[i])][category] += 1
            selected.append(i)
        time_bin = int((self.last_time-self.collection_start)//5)
        if selected:
            self.written_time_bins[str(time_bin)] = self.written_time_bins.get(str(time_bin), 0)+len(selected)
        return torch.tensor(selected, dtype=torch.long, device=good.device)

    def summary(self):
        elapsed = (self.last_time-self.collection_start) if self.collection_start is not None else 0.
        complete = (sum(self.counts) == self.target
            and elapsed+1e-6 >= self.cfg.min_collection_span_s)
        return {
            "config": asdict(self.cfg), "pair_order": list(PAIR_NAMES),
            "accepted_1s": self.counts, "eligible_1s": self.eligible_counts,
            "independent_events": len(self.events), "rejections": self.rejections,
            "missing_1s": ([max(0,q-n) for q,n in zip(self.cfg.pair_quotas, self.counts)]
                             if self.cfg.pair_quotas is not None else None),
            "temporal_coverage": {
                "version": 1, "warmup_s": self.cfg.warmup_s,
                "collection_elapsed_s": elapsed,
                "minimum_collection_span_s": self.cfg.min_collection_span_s,
                "label_interval_frames": self.cfg.label_interval_frames,
                "randomize_label_phase": self.cfg.randomize_label_phase,
                "time_bin_width_s": 5., "written_by_time_bin": self.written_time_bins,
                "lifecycle_order": list(LIFECYCLE_NAMES),
                "stage_counter_scope": "label-attempt snapshots; excludes burn-in and unsampled 10 Hz ticks",
                "mature_drift_age_s": 2.,
                "seen": self.seen_lifecycle, "eligible": self.eligible_lifecycle,
                "written": self.written_lifecycle,
                "eligible_1s_by_lifecycle": self.eligible_1s_by_lifecycle,
                "written_1s_by_lifecycle": self.written_1s_by_lifecycle,
                "missing_observed_stages": [name for name,n in zip(LIFECYCLE_NAMES,self.seen_lifecycle) if n == 0],
                "note": "Pacing is an upper admission bound. Stage counters are diagnostic, not outcome quotas or proof of stage coverage.",
            },
            "complete": complete,
        }
