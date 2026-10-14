"""Safety outcomes for one-second counterfactual rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import torch


TILT_LIMIT_RAD = 0.8
CLEARANCE_LIMIT_M = 0.45
GROUND_RADIUS_M = 0.15
# A bitmask preserves simultaneous triggers on the first failure step.
FIRST_FAILURE_BITS = {
    "non_timeout_termination": 1,
    "torso_contact": 2,
    "tilt": 4,
    "low_clearance": 8,
}
FUTURE_METRIC_ORDER = (
    "first_failure_step", "first_reset_step", "max_tilt", "min_clearance",
    "max_tracking_error", "terminated", "torso_contact", "first_timeout_step",
    "first_invalid_step", "observed_steps",
)


def safety_label_contract() -> dict[str, Any]:
    """Version the physical meaning of labels, independently of policy hashes."""

    return {
        "version": 3,
        "horizon_steps": [50],
        "label_order": ["lp_1.0s", "vp_1.0s"],
        "control_dt_s": 0.02,
        "capture": "before_control_step_reset",
        "ground_reference": "nearest_valid_height_scan_hit_to_root_xy",
        "ground_max_radius_m": GROUND_RADIUS_M,
        "tilt_limit_rad": TILT_LIMIT_RAD,
        "clearance_limit_m": CLEARANCE_LIMIT_M,
        "torso_contact": "termination_body_force_history_norm_gt_1N",
        "failure": "non_timeout_termination_or_torso_contact_or_tilt_or_low_clearance",
        "missing_measurement": "reject_pair_if_unknown_before_first_failure",
        "timeout": "reject_incomplete_horizon;final_step_timeout_is_complete",
        "tracking_error": "audit_only",
    }


def validate_safety_label_contract(contract: Any) -> dict[str, Any]:
    if contract != safety_label_contract():
        raise ValueError("selector safety-label contract is missing or outdated; recollect the dataset")
    return copy.deepcopy(contract)


def dataset_safety_label_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    return validate_safety_label_contract(manifest.get("metadata", {}).get("safety_label_contract"))


@dataclass
class SafetyStepMetrics:
    """Detached ground-truth measurements from the completed, unreset step."""

    tilt: torch.Tensor
    clearance: torch.Tensor
    tracking_error: torch.Tensor
    torso_contact: torch.Tensor
    valid: torch.Tensor


def local_clearance(env: Any, max_radius_m: float = GROUND_RADIUS_M) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the closest valid ray hit near the root projection, never a distant obstacle.

    Actual world coordinates avoid assuming a particular flatten order or a
    scan centered on the root. No local ground return means an unknown label.
    """

    root = env.robot.data.root_pos_w
    scanner = getattr(env, "height_scanner", None)
    if scanner is None:
        return torch.zeros_like(root[:, 2]), torch.zeros_like(root[:, 2], dtype=torch.bool)
    hits = scanner.data.ray_hits_w
    valid_hit = torch.isfinite(hits).all(dim=-1)
    ray_starts = getattr(scanner, "ray_starts_w", None)
    max_distance = getattr(getattr(scanner, "cfg", None), "max_distance", None)
    if ray_starts is not None and max_distance is not None:
        distance = (hits - ray_starts).norm(dim=-1)
        valid_hit &= torch.isfinite(distance) & (distance <= max_distance)
    xy_distance_sq = (hits[..., :2] - root[:, None, :2]).square().sum(dim=-1)
    valid_hit &= xy_distance_sq <= max_radius_m ** 2
    distances = torch.where(valid_hit, xy_distance_sq, torch.inf)
    nearest_distance, nearest = distances.min(dim=-1)
    ground_z = hits[..., 2].gather(1, nearest[:, None]).squeeze(1)
    valid = torch.isfinite(root).all(dim=-1) & torch.isfinite(nearest_distance)
    clearance = torch.where(valid, root[:, 2] - ground_z, 0.0)
    return clearance, valid


@torch.no_grad()
def capture_safety_step(env: Any, command: torch.Tensor) -> SafetyStepMetrics:
    """Capture true simulator state before G1RENetEnv resets any robots."""

    projected = env.robot.data.projected_gravity_b
    tilt = torch.acos((-projected[:, 2]).clamp(-1.0, 1.0)).abs()
    clearance, ground_valid = local_clearance(env)
    actual_velocity = env.robot.data.root_lin_vel_b
    angular_velocity = env.robot.data.root_ang_vel_b
    tracking_error = (actual_velocity[:, :2] - command[:, :2]).square().sum(dim=-1)
    tracking_error += (angular_velocity[:, 2] - command[:, 2]).square()
    forces = env.contact_sensor.data.net_forces_w_history[:, :, env.termination_contact_cfg.body_ids]
    torso_contact = (forces.norm(dim=-1) > 1.0).any(dim=(1, 2))
    valid = (
        ground_valid & torch.isfinite(projected).all(dim=-1)
        & torch.isfinite(actual_velocity).all(dim=-1)
        & torch.isfinite(angular_velocity).all(dim=-1)
        & torch.isfinite(command).all(dim=-1)
        & torch.isfinite(forces).all(dim=(1, 2, 3))
    )
    return SafetyStepMetrics(tilt, clearance, tracking_error, torso_contact, valid)


def step_with_safety_capture(env: Any, actions: torch.Tensor, command: torch.Tensor):
    """Attach a collection-only pre-reset hook for exactly one control step.

    G1RENetEnv.step calls reset even for an empty reset batch. The scoped hook
    samples every robot at that point, then delegates to the original reset.
    It restores the exact instance method layout even when step raises, and
    refuses an environment that does not expose this pre-reset boundary.
    """

    original_reset = env.reset
    had_override = "reset" in vars(env)
    previous_override = vars(env).get("reset")
    metrics = None

    def capture_and_reset(*args, **kwargs):
        nonlocal metrics
        if metrics is not None:
            raise RuntimeError("selector safety capture requires one reset boundary per control step")
        metrics = capture_safety_step(env, command)
        return original_reset(*args, **kwargs)

    env.reset = capture_and_reset
    try:
        result = env.step(actions)
    finally:
        if had_override:
            env.reset = previous_override
        else:
            del env.reset
    if metrics is None:
        raise RuntimeError("selector environment did not expose a pre-reset safety capture boundary")
    return result, metrics


@dataclass
class RolloutOutcome:
    fail_1s: torch.Tensor
    first_failure_step: torch.Tensor
    first_reset_step: torch.Tensor
    max_tilt: torch.Tensor
    min_clearance: torch.Tensor
    max_tracking_error: torch.Tensor
    terminated: torch.Tensor
    torso_contact: torch.Tensor
    label_valid: torch.Tensor
    first_timeout_step: torch.Tensor
    first_invalid_step: torch.Tensor
    observed_steps: torch.Tensor
    first_failure_flags: torch.Tensor | None = None
    first_failure_cause_stable: torch.Tensor | None = None

    def metrics(self) -> torch.Tensor:
        return torch.stack(
            (
                self.first_failure_step.float(),
                self.first_reset_step.float(),
                self.max_tilt,
                self.min_clearance,
                self.max_tracking_error,
                self.terminated.float(),
                self.torso_contact.float(),
                self.first_timeout_step.float(),
                self.first_invalid_step.float(),
                self.observed_steps.float(),
            ),
            dim=-1,
        )


class SafetyOutcomeAccumulator:
    """Accumulate only future outcomes; none of these values enter selector features."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device,
        horizon_steps: int = 50,
        tilt_limit_rad: float = TILT_LIMIT_RAD,
        clearance_limit_m: float = CLEARANCE_LIMIT_M,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.horizon_steps = int(horizon_steps)
        self.tilt_limit_rad = float(tilt_limit_rad)
        self.clearance_limit_m = float(clearance_limit_m)
        if type(horizon_steps) is not int or horizon_steps != 50:
            raise ValueError("selector labels require a one-second / 50-step horizon")
        self.reset()

    def reset(self) -> None:
        n = self.num_envs
        self.step_index = 0
        self.fail_1s = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_failure_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.first_failure_flags = torch.zeros(n, dtype=torch.uint8, device=self.device)
        self.first_reset_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.max_tilt = torch.zeros(n, device=self.device)
        self.min_clearance = torch.full((n,), torch.inf, device=self.device)
        self.max_tracking_error = torch.zeros(n, device=self.device)
        self.terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.torso_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.label_valid = torch.ones(n, dtype=torch.bool, device=self.device)
        self.first_timeout_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.first_invalid_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.observed_steps = torch.zeros(n, dtype=torch.long, device=self.device)

    def update(
        self,
        metrics: SafetyStepMetrics,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> None:
        if self.step_index >= self.horizon_steps:
            raise RuntimeError("outcome accumulator received more than the configured horizon")
        for name in SafetyStepMetrics.__dataclass_fields__:
            if getattr(metrics, name).shape != (self.num_envs,):
                raise ValueError(f"safety {name} must have shape {(self.num_envs,)}")
        dones = dones.reshape(self.num_envs).bool()
        time_outs = time_outs.reshape(self.num_envs).bool()
        if torch.any(time_outs & ~dones):
            raise ValueError("a selector timeout must also reset the environment")
        tilt, clearance, tracking_error = metrics.tilt, metrics.clearance, metrics.tracking_error
        finite = torch.isfinite(tilt) & torch.isfinite(clearance) & torch.isfinite(tracking_error)
        measurement_valid = metrics.valid.bool() & finite
        non_timeout_termination = dones.bool() & ~time_outs.bool()
        torso_contact = metrics.torso_contact.bool()
        active = ~self.finished
        self.observed_steps += active.long()
        # Once failure is observed, the one-second failure label is determined.
        # Before that point, missing measurements cannot establish safety.
        unknown = active & ~self.fail_1s & ~measurement_valid
        newly_invalid = unknown & (self.first_invalid_step < 0)
        self.first_invalid_step[newly_invalid] = self.step_index + 1
        self.label_valid &= ~unknown
        timed_out = active & time_outs
        self.first_timeout_step[timed_out] = self.step_index + 1
        if self.step_index + 1 < self.horizon_steps:
            self.label_valid &= ~timed_out
        failure = active & (
            non_timeout_termination
            | torso_contact
            | (measurement_valid & (tilt > self.tilt_limit_rad))
            | (measurement_valid & (clearance < self.clearance_limit_m))
        )
        newly_failed = failure & (self.first_failure_step < 0)
        self.first_failure_step[newly_failed] = self.step_index + 1
        flags = (
            non_timeout_termination.to(torch.uint8)
            | (torso_contact.to(torch.uint8) * 2)
            | ((measurement_valid & (tilt > self.tilt_limit_rad)).to(torch.uint8) * 4)
            | ((measurement_valid & (clearance < self.clearance_limit_m)).to(torch.uint8) * 8)
        )
        self.first_failure_flags[newly_failed] = flags[newly_failed]
        newly_reset = active & dones.bool()
        self.first_reset_step[newly_reset] = self.step_index + 1
        self.fail_1s |= failure
        measured = active & measurement_valid
        self.max_tilt = torch.where(measured, torch.maximum(self.max_tilt, tilt), self.max_tilt)
        self.min_clearance = torch.where(
            measured, torch.minimum(self.min_clearance, clearance), self.min_clearance
        )
        self.max_tracking_error = torch.where(
            measured,
            torch.maximum(self.max_tracking_error, tracking_error),
            self.max_tracking_error,
        )
        self.terminated |= active & non_timeout_termination
        self.torso_contact |= active & torso_contact
        self.finished |= dones.bool()
        self.step_index += 1

    def outcome(self) -> RolloutOutcome:
        if self.step_index != self.horizon_steps:
            raise RuntimeError(
                f"rollout ended after {self.step_index} steps, expected {self.horizon_steps}"
            )
        return RolloutOutcome(
            fail_1s=self.fail_1s.clone(),
            first_failure_step=self.first_failure_step.clone(),
            first_reset_step=self.first_reset_step.clone(),
            max_tilt=self.max_tilt.clone(),
            min_clearance=self.min_clearance.clone(),
            max_tracking_error=self.max_tracking_error.clone(),
            terminated=self.terminated.clone(),
            torso_contact=self.torso_contact.clone(),
            label_valid=self.label_valid.clone(),
            first_timeout_step=self.first_timeout_step.clone(),
            first_invalid_step=self.first_invalid_step.clone(),
            observed_steps=self.observed_steps.clone(),
            first_failure_flags=self.first_failure_flags.clone(),
        )
