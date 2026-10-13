"""Stream V4 one-second shards into a JSON audit report without starting Isaac Sim."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from .collection_audit import (
    CANDIDATE_SAFETY_ORDER, FAULT_PARAMETER_ORDER, PAIR_NAMES,
    SCENARIO_NAMES, SEVERITY_NAMES, active_scenario,
)
from .dataset import SPLIT_NAMES
from .labels import CLEARANCE_LIMIT_M, FIRST_FAILURE_BITS, FUTURE_METRIC_ORDER, TILT_LIMIT_RAD
from .lp_map_faults import NOISE_AUDIT_ORDER


LABEL_ORDER = ("lp_1.0s", "vp_1.0s")
AUDIT_FIELDS = ("event_identity", "fault_parameters", "first_failure_flags",
                "first_failure_cause_stable", "candidate_safety")
FAULT_NAMES = (
    "none", "freeze", "delay", "vp_occlusion", "vp_scale", "vp_bias", "vp_shift",
    "vp_collapse", "lp_random_dropout", "lp_block_dropout", "lp_height_bias",
    "lp_registration", "lp_drift", "lp_collapse",
    "lp_map_drift", "lp_person", "lp_compound", "vp_compound", "vp_burst",
)
PARAMETERS_BY_KIND = {
    1: (), 2: ("delay_s",), 3: ("vp_occlusion_fraction",), 4: ("vp_scale",),
    5: ("vp_bias_m",), 6: ("vp_shift_x_px", "vp_shift_y_px"),
    7: ("vp_collapse_normalized",), 8: ("lp_dropout_probability",),
    9: ("lp_block_fraction",), 10: ("lp_height_bias_m",),
    11: ("lp_translation_x_m", "lp_translation_y_m", "lp_yaw_deg"),
    12: ("lp_drift_x_m_per_s", "lp_drift_y_m_per_s", "lp_drift_yaw_deg_per_s"),
    13: (),
    14: ("lp_drift_x_m_per_s", "lp_drift_y_m_per_s", "lp_drift_yaw_deg_per_s"),
    15: (), 16: (),
    17: ("delay_s", "vp_scale", "vp_bias_m", "vp_shift_x_px", "vp_shift_y_px", "vp_occlusion_fraction"),
    18: ("vp_occlusion_fraction", "vp_bias_m"),
}


class Counts:
    def __init__(self):
        self.rows = 0
        self.positive = torch.zeros(2, dtype=torch.long)
        self.pairs = torch.zeros(1, 4, dtype=torch.long)

    def add(self, labels):
        self.rows += len(labels)
        self.positive += labels.long().sum(0)
        for h, (lp, vp) in enumerate(((0, 1),)):
            self.pairs[h] += torch.bincount(
                labels[:, lp].long() + labels[:, vp].long() * 2, minlength=4
            )

    def result(self):
        return {
            "rows": self.rows,
            "positive": self.positive.tolist(),
            "positive_rate": [int(v) / self.rows if self.rows else None for v in self.positive],
            "pair_counts": self.pairs.tolist(),
            "pair_rates": [[int(v) / self.rows if self.rows else None for v in row] for row in self.pairs],
        }


def _group_update(groups, dimension, values, labels, mask=None):
    if mask is None:
        mask = torch.ones(len(labels), dtype=torch.bool)
    for value in torch.unique(values[mask]).tolist():
        groups[dimension][str(value)].add(labels[mask & (values == value)])


def _parameter_update(summary, key, values):
    if not len(values):
        return
    value = summary.setdefault(key, {"rows": 0, "min": float("inf"), "max": -float("inf"), "sum": 0.0})
    value["rows"] += len(values)
    value["min"] = min(value["min"], float(values.min()))
    value["max"] = max(value["max"], float(values.max()))
    value["sum"] += float(values.double().sum())


def analyze_selector_dataset(root: str | Path) -> dict:
    """Analyze complete or interrupted V4 datasets; missing new fields stay missing."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 4 or manifest.get("label_order") != list(LABEL_ORDER):
        raise ValueError("analysis requires V4 one-second / two-head data")
    if manifest.get("future_metric_order_per_branch") != list(FUTURE_METRIC_ORDER):
        raise ValueError("unsupported V4 future metric order")
    overall, splits = Counts(), {s: Counts() for s in SPLIT_NAMES}
    groups = defaultdict(lambda: defaultdict(Counts))
    fields = defaultdict(int)
    times = torch.zeros(2, 51, dtype=torch.long)  # index 0: no observed failure
    time_rows = 0
    causes = torch.zeros(2, 16, dtype=torch.long)
    ambiguous = torch.zeros(2, dtype=torch.long)
    cause_rows = 0
    initial_rows = initial_invalid = initial_failed = 0
    episodes, plans = {}, {}
    event_rows = 0
    parameter_summary = {}
    noise_summary = {}
    audit_contract = manifest.get("metadata", {}).get("collection_audit_contract")
    has_new_contract = audit_contract is not None
    if has_new_contract and (
        audit_contract.get("version") != 2
        or audit_contract.get("fault_parameter_order") != list(FAULT_PARAMETER_ORDER)
        or audit_contract.get("candidate_safety_order") != list(CANDIDATE_SAFETY_ORDER)
        or audit_contract.get("first_failure_bits") != FIRST_FAILURE_BITS
    ):
        raise ValueError("unsupported collection audit field contract")
    for split_index, split in enumerate(SPLIT_NAMES):
        paths = sorted((root / split).glob("shard_*.pt"))
        if len(paths) != manifest["shards"][split]:
            raise ValueError(f"manifest shard count differs for {split}")
        for path in paths:
            # mmap keeps the multi-GB feature tensors off the analysis heap.
            data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            raw_labels = data["labels"]
            if raw_labels.ndim != 2 or raw_labels.shape[1] != 2 or not ((raw_labels == 0) | (raw_labels == 1)).all():
                raise ValueError(f"invalid binary labels in {path}")
            labels = raw_labels.bool()
            n = len(labels)
            if data["features"].shape != (n, 5, 283):
                raise ValueError(f"invalid feature shape in {path}")
            overall.add(labels)
            splits[split].add(labels)
            for field in ("future_metrics", "fault_metadata", "execution_metadata", *AUDIT_FIELDS):
                if field in data:
                    if len(data[field]) != n:
                        raise ValueError(f"misaligned {field} in {path}")
                    fields[field] += n
            if "noise_metadata" in data:
                noise = data["noise_metadata"]
                contract = manifest.get("metadata", {}).get("noise_audit_contract", {})
                if contract.get("version") != 1 or contract.get("order") != list(NOISE_AUDIT_ORDER):
                    raise ValueError(f"unknown noise audit order in {path}")
                if noise.shape != (n, len(NOISE_AUDIT_ORDER)) or not torch.isfinite(noise).all():
                    raise ValueError(f"invalid map noise metadata in {path}")
                fields["noise_metadata"] += n
                for column, name in enumerate(NOISE_AUDIT_ORDER):
                    _parameter_update(noise_summary, name, noise[:, column])
                advanced = data["fault_metadata"][:, 6].bool() & (data["fault_metadata"][:, 0] >= 14)
                people = advanced & ((data["fault_metadata"][:, 0] == 15) | (data["fault_metadata"][:, 0] == 16))
                _group_update(groups, "person_phase", noise[:, 7].long(), labels, people)
                _group_update(groups, "lp_ghost_present", (noise[:, 8] > 0).long(), labels, advanced)
                _group_update(groups, "lp_support_bin", torch.bucketize(noise[:, 11].contiguous(), torch.tensor([.05,.5,.95]), right=True), labels, advanced)
                _group_update(groups, "lp_saturation_bin", torch.bucketize(noise[:, 12].contiguous(), torch.tensor([.01,.1,.5]), right=True), labels, advanced)
            if any(f in data for f in AUDIT_FIELDS) and not has_new_contract:
                raise ValueError("new audit tensors require collection_audit_contract metadata")
            if ("first_failure_flags" in data) != ("first_failure_cause_stable" in data):
                raise ValueError(f"missing paired first failure audit field in {path}")
            for field, columns in (("fault_parameters", len(FAULT_PARAMETER_ORDER)),
                                   ("candidate_safety", len(CANDIDATE_SAFETY_ORDER))):
                if field in data and (data[field].shape != (n, columns) or not torch.isfinite(data[field]).all()):
                    raise ValueError(f"invalid {field} in {path}")
            if "future_metrics" in data:
                first = data["future_metrics"][:, :, 0].float()
                if first.shape != (n, 2) or not torch.isfinite(first).all() or not (first == first.round()).all():
                    raise ValueError(f"invalid first failure times in {path}")
                if not ((first == -1) | ((first >= 1) & (first <= 50))).all():
                    raise ValueError(f"out-of-horizon first failure in {path}")
                if not torch.equal(first >= 1, labels):
                    raise ValueError(f"first failure times disagree with labels in {path}")
                for branch in range(2):
                    times[branch] += torch.bincount(first[:, branch].clamp_min(0).long(), minlength=51)
                time_rows += n
            if "first_failure_flags" in data:
                flags = data["first_failure_flags"].long()
                stable = data["first_failure_cause_stable"].bool()
                if flags.shape != (n, 2) or stable.shape != (n, 2) or not ((flags >= 0) & (flags <= 15)).all():
                    raise ValueError(f"invalid first failure flags in {path}")
                if not torch.equal(flags != 0, labels):
                    raise ValueError(f"first failure flags disagree with labels in {path}")
                for branch in range(2):
                    causes[branch] += torch.bincount(flags[stable[:, branch], branch], minlength=16)
                    ambiguous[branch] += (~stable[:, branch] & labels[:, branch]).sum()
                cause_rows += n
            if "candidate_safety" in data:
                safety = data["candidate_safety"]
                valid = safety[:, 3].bool()
                violations = safety[:, 2].bool() | (valid & ((safety[:, 0] > TILT_LIMIT_RAD) | (safety[:, 1] < CLEARANCE_LIMIT_M)))
                initial_rows += n
                initial_invalid += int((~valid).sum())
                initial_failed += int(violations.sum())
                _group_update(groups, "initial_safety", violations.long(), labels, valid | safety[:, 2].bool())
            for field in ("terrain_level", "terrain_type"):
                _group_update(groups, field, data[field].reshape(-1).long(), labels)
            if "execution_metadata" in data:
                execution = data["execution_metadata"]
                _group_update(groups, "behavior_mode", execution[:, 0].long(), labels)
                _group_update(groups, "transition_active", (execution[:, 2] > 0).long(), labels)
                dwell = torch.bucketize(execution[:, 1].float().contiguous(), torch.tensor([1., 2.]), right=True)
                _group_update(groups, "dwell_bin", dwell, labels)
            if "fault_metadata" in data:
                faults = data["fault_metadata"].long()
                if faults.shape != (n, 8):
                    raise ValueError(f"invalid fault metadata in {path}")
                scenario = active_scenario(faults)
                severity = faults[:, 2]
                if (not ((severity >= 0) & (severity <= 2)).all()
                    or not ((faults[:, :2] >= 0) & (faults[:, :2] < len(FAULT_NAMES))).all()
                    or not ((faults[:, 6:] == 0) | (faults[:, 6:] == 1)).all()):
                    raise ValueError(f"invalid fault metadata in {path}")
                _group_update(groups, "active_scenario", scenario, labels)
                _group_update(groups, "plan_severity", severity, labels)
                _group_update(groups, "active_fault_severity", severity, labels, scenario != 0)
                _group_update(groups, "active_scenario_severity", scenario * 3 + severity, labels)
                _group_update(groups, "phase", faults[:, 3], labels)
                for branch, name in enumerate(("lp", "vp")):
                    active = faults[:, 6 + branch].bool()
                    _group_update(groups, f"active_{name}_fault_kind", faults[:, branch], labels, active)
                    _group_update(groups, f"active_{name}_fault_kind_severity", faults[:, branch] * 3 + severity, labels, active)
                    if "fault_parameters" in data:
                        for kind in torch.unique(faults[active, branch]).tolist():
                            mask = active & (faults[:, branch] == kind)
                            names = [f"{name}_duration_s", f"{name}_start_in_s"]
                            names += [f"{name}_delay_s" if p == "delay_s" else p for p in PARAMETERS_BY_KIND.get(kind, ())]
                            for parameter in names:
                                column = FAULT_PARAMETER_ORDER.index(parameter)
                                values = data["fault_parameters"][mask, column]
                                _parameter_update(parameter_summary, f"{name}:{FAULT_NAMES[kind]}:{parameter}", values)
            if "event_identity" in data:
                identity = data["event_identity"]
                if identity.shape != (n, 3) or identity.dtype != torch.int64 or (identity < 0).any():
                    raise ValueError(f"invalid event identity in {path}")
                event_rows += n
                packed = sum(labels[:, j].long() << j for j in range(2)).tolist()
                fault_rows = data.get("fault_metadata")
                for index, (env_id, episode, schedule) in enumerate(identity.tolist()):
                    ep, plan = (env_id, episode), (env_id, episode, schedule)
                    split_bit = 1 << split_index
                    old = episodes.get(ep, (0, 0, [0, 0, 0]))
                    old[-1][split_index] |= packed[index]
                    episodes[ep] = (old[0] | split_bit, old[1] | packed[index], old[-1])
                    if fault_rows is not None:
                        fm = fault_rows[index]
                        plan_group = int(fm[0] != 0) * 2 + int(fm[1] != 0), int(fm[2])
                    else:
                        plan_group = None
                    old = plans.get(plan, (0, 0, plan_group, [0, 0, 0]))
                    if old[2] != plan_group:
                        raise ValueError(f"fault plan identity reused with different metadata in {path}")
                    old[-1][split_index] |= packed[index]
                    plans[plan] = (old[0] | split_bit, old[1] | packed[index], plan_group, old[-1])
        if splits[split].rows != manifest["counts"][split]:
            raise ValueError(f"manifest row count differs for {split}")

    def event_counts(registry, split_bit=None):
        values = [v for v in registry.values() if split_bit is None or v[0] & split_bit]
        positives = [v[1] if split_bit is None else v[-1][split_bit.bit_length() - 1] for v in values]
        return {"unique": len(values),
                "with_counterfactual_failure_per_head": [sum(bool(v & (1 << j)) for v in positives) for j in range(2)],
                "cross_split": sum(v[0].bit_count() > 1 for v in values)}

    accepted_plans = torch.zeros(4, 3, dtype=torch.long)
    for _, _, category, _ in plans.values():
        if category is not None:
            accepted_plans[category] += 1
    for value in parameter_summary.values():
        value["mean"] = value.pop("sum") / value["rows"]
    for value in noise_summary.values():
        value["mean"] = value.pop("sum") / value["rows"]
    return {
        "report_version": 3,
        "collection_complete": manifest.get("metadata", {}).get("collection_complete", False),
        "label_order": list(LABEL_ORDER), "branch_order": ["lp", "vp"],
        "horizon_order": ["1.0s"], "pair_order": list(PAIR_NAMES),
        "overall": overall.result(), "by_split": {s: c.result() for s, c in splits.items()},
        "group_key_legend": {
            "active_scenario": list(SCENARIO_NAMES), "severity": list(SEVERITY_NAMES),
            "active_scenario_severity": "scenario_index * 3 + severity_index",
            "fault_kind": list(FAULT_NAMES), "fault_kind_severity": "kind_id * 3 + severity_index",
            "phase": {"-1": "before", "0": "scheduled_interval", "1": "recovered"},
            "behavior_mode": {"0": "vp", "1": "lp"},
            "dwell_bin": ["[0,1)", "[1,2)", "[2,inf)"],
            "transition_active": {"0": "no_blend", "1": "blending"},
            "initial_safety": {"0": "within_thresholds", "1": "already_violated"},
            "person_phase": ["approach", "near", "depart", "residual"],
            "lp_support_bin": ["<.05", ".05-.5", ".5-.95", ">=.95"],
            "lp_saturation_bin": ["<.01", ".01-.1", ".1-.5", ">=.5"],
        },
        "by_group": {d: {k: c.result() for k, c in sorted(g.items())} for d, g in sorted(groups.items())},
        "field_rows": {f: fields[f] for f in ("future_metrics", "fault_metadata", "execution_metadata", *AUDIT_FIELDS, "noise_metadata")},
        "first_failure_time": {
            "available_rows": time_rows, "control_dt_s": .02,
            "no_failure_within_1s": times[:, 0].tolist(),
            "step_counts_1_to_50": times[:, 1:].tolist(),
            "note": "absence within 1s does not establish safety after 1s",
        },
        "first_failure_causes": {
            "available_rows": cause_rows, "bits": FIRST_FAILURE_BITS,
            "stable_bitmask_counts_0_to_15": causes.tolist(),
            "ambiguous_failure_rows": ambiguous.tolist(),
            "stable_trigger_counts": {
                name: [int(causes[b, [i for i in range(16) if i & bit]].sum()) for b in range(2)]
                for name, bit in FIRST_FAILURE_BITS.items()
            },
            "note": "first-step triggers can overlap; ambiguous replay causes excluded from stable counts",
        },
        "candidate_safety": {"available_rows": initial_rows, "invalid_rows": initial_invalid,
                             "already_violated_rows": initial_failed},
        "events": {
            "available_rows": event_rows,
            "episodes": event_counts(episodes) if event_rows else None,
            "plans": event_counts(plans) if event_rows else None,
            "by_split": {s: {"episodes": event_counts(episodes, 1 << i), "plans": event_counts(plans, 1 << i)}
                         for i, s in enumerate(SPLIT_NAMES)} if event_rows else None,
            "accepted_unique_plans_by_planned_scenario_severity": accepted_plans.tolist() if event_rows else None,
            "note": "plans include normal schedules; failure counts denote at least one failed counterfactual label, not real falls",
        },
        "fault_parameter_summary": parameter_summary,
        "noise_parameter_summary": noise_summary,
        "supplement_source": manifest.get("metadata", {}).get("supplement_source"),
        "supplement_coverage": manifest.get("metadata", {}).get("supplement_coverage"),
        "collection_audit": manifest.get("metadata", {}).get("collection_audit"),
        "replay_filter_contract": manifest.get("metadata", {}).get("replay_filter_contract"),
        "per_candidate_replay_filter": manifest.get("metadata", {}).get("per_candidate_replay_filter"),
        "per_candidate_replay_diagnostics": manifest.get("metadata", {}).get("per_candidate_replay_diagnostics"),
        "snapshot_replay_validation": manifest.get("metadata", {}).get("snapshot_replay_validation"),
        "run_info": manifest.get("metadata", {}).get("run_info"),
        "fault_config": manifest.get("metadata", {}).get("fault_config"),
        "coverage": manifest.get("metadata", {}).get("coverage"),
        "safety_label_filter": manifest.get("metadata", {}).get("safety_label_filter"),
        "sampling_progress": manifest.get("metadata", {}).get("sampling_progress"),
        "notes": [
            "10/30/60 controls scheduled-plan draws; accepted frame fractions also depend on durations and filtering.",
            "balanced_scenarios balances active fault categories, not safe/failure outcomes.",
            "Only V4 data uses the one-second labels and ordinary-transition contract.",
            "Replay diagnostics compare acceptance rules on the same candidate rollouts; retained counts precede quotas, written counts apply quotas.",
        ],
    }


def write_collection_analysis(root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(root)
    destination = Path(output) if output is not None else root / "collection_analysis.json"
    if destination.resolve() == (root / "manifest.json").resolve():
        raise ValueError("analysis output must not overwrite the dataset manifest")
    report = analyze_selector_dataset(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
