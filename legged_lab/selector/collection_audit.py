"""Collection-only contracts and bounded counters; never selector inputs."""

from __future__ import annotations

import torch

from .labels import FIRST_FAILURE_BITS


SCENARIO_NAMES = ("normal", "vp_only", "lp_only", "dual")
SEVERITY_NAMES = ("light", "medium", "heavy")
PAIR_NAMES = ("both_safe", "lp_only_failure", "vp_only_failure", "both_failure")
FAULT_PARAMETER_ORDER = (
    "lp_duration_s", "vp_duration_s", "lp_delay_s", "vp_delay_s",
    "lp_start_in_s", "vp_start_in_s", "vp_scale", "vp_bias_m",
    "vp_shift_x_px", "vp_shift_y_px", "vp_occlusion_fraction",
    "vp_collapse_normalized", "lp_dropout_probability", "lp_block_fraction",
    "lp_height_bias_m", "lp_translation_x_m", "lp_translation_y_m", "lp_yaw_deg",
    "lp_drift_x_m_per_s", "lp_drift_y_m_per_s", "lp_drift_yaw_deg_per_s",
)
CANDIDATE_SAFETY_ORDER = ("tilt_rad", "clearance_m", "torso_contact", "measurement_valid")
FUNNEL_STAGES = (
    "seen", "history_ready", "observable_future", "episode_horizon",
    "labels_valid", "replay_stable", "written",
)


def collection_audit_contract() -> dict:
    return {
        "version": 2,
        "event_identity_order": ["env_id", "episode_serial", "schedule_count"],
        "event_identity_scope": "collection_run_id",
        "event_identity_meaning": "scheduled_plan_including_normal_intervals",
        "fault_metadata_order": [
            "lp_kind", "vp_kind", "severity", "phase", "remaining_frames",
            "duration_frames", "lp_active", "vp_active",
        ],
        "severity_order": list(SEVERITY_NAMES),
        "fault_parameter_order": list(FAULT_PARAMETER_ORDER),
        "fault_parameter_note": "planned values; interpret only for the matching fault kind",
        "fault_start_in_s": "relative to the last observed sensor frame; negative means already started",
        "first_failure_bits": FIRST_FAILURE_BITS,
        "first_failure_branch_order": ["lp", "vp"],
        "first_failure_cause_stable": "first-step bitmask agrees between repeated branch rollouts",
        "candidate_safety_order": list(CANDIDATE_SAFETY_ORDER),
        "candidate_safety_note": "ground truth at candidate time; audit only",
        "group_ids_meaning": "terrain seed/type/level group, not episode identity",
        "scenario_seed_meaning": "candidate row ID, not fault event identity",
    }


def active_scenario(metadata: torch.Tensor) -> torch.Tensor:
    return metadata[:, 6].long() * 2 + metadata[:, 7].long()


class CollectionAudit:
    """Observe only the real behavior path, outside snapshot/probe forks."""

    def __init__(self, num_envs: int, device):
        self.last_plan = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
        self.observed_plans = torch.zeros(4, 3, dtype=torch.long)
        self.funnel = torch.zeros(len(FUNNEL_STAGES), 4, 3, dtype=torch.long)
        self.pairs = torch.zeros(3, 1, 4, 3, 4, dtype=torch.long)
        self.candidate_batches = 0

    def observe_behavior_plans(self, episode_serial, faults) -> None:
        plan = torch.stack((episode_serial, faults.schedule_count), -1)
        changed = (plan != self.last_plan).any(-1)
        category = (faults.lp_kind != 0).long() * 2 + (faults.vp_kind != 0).long()
        keys = category * 3 + faults.severity
        self.observed_plans += torch.bincount(keys[changed], minlength=12).reshape(4, 3).cpu()
        self.last_plan.copy_(plan)

    def record_candidates(self, faults, labels, stage_masks, written) -> None:
        """Stage masks are cumulative; pair labels use the second replay."""
        keys = active_scenario(faults) * 3 + faults[:, 2].long()
        masks = [torch.ones_like(written), *stage_masks, written]
        if len(masks) != len(FUNNEL_STAGES):
            raise ValueError("unexpected collection audit funnel")
        for index, mask in enumerate(masks):
            self.funnel[index] += torch.bincount(keys[mask], minlength=12).reshape(4, 3).cpu()
        for stage, mask in enumerate((stage_masks[-2], stage_masks[-1], written)):
            for horizon, (lp, vp) in enumerate(((0, 1),)):
                pairs = labels[:, lp].long() + labels[:, vp].long() * 2
                counts = torch.bincount((keys * 4 + pairs)[mask], minlength=48)
                self.pairs[stage, horizon] += counts.reshape(4, 3, 4).cpu()
        self.candidate_batches += 1

    def summary(self) -> dict:
        return {
            "candidate_batches": self.candidate_batches,
            "scenario_order": list(SCENARIO_NAMES),
            "severity_order": list(SEVERITY_NAMES),
            "observed_behavior_plans_by_planned_scenario_severity": self.observed_plans.tolist(),
            "observed_plan_scope": "from collection start through real behavior steps; excludes burn-in and all counterfactual/replay forks",
            "funnel_order": list(FUNNEL_STAGES),
            "funnel_by_active_scenario_severity": self.funnel.tolist(),
            "pair_stage_order": ["before_replay_filter", "after_replay_filter", "written"],
            "pair_horizon_order": ["1.0s"],
            "pair_order": list(PAIR_NAMES),
            "pairs_by_stage_horizon_active_scenario_severity": self.pairs.tolist(),
            "pre_replay_pair_note": "diagnostic second-replay labels, not reliable ground truth for rejected pairs",
        }
