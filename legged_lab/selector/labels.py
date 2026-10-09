# 中文导读
# 文件用途：用 reset 前的仿真真实状态生成 LP/VP 在 0.4 秒和 1 秒内的安全失败标签。
# 使用位置：collector 采集标签；dataset 使用审计列定义；训练、评估和导出检查标签版本。
# 说明：版本 2 使用机身投影附近地面，拒绝未知测量和提前超时；跟踪误差只作审计。
# 说明：真实安全指标只生成监督标签，不进入 selector 输入；reset 钩子只在采集单步期间存在。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - safety_label_contract：返回当前标签版本、时间窗、测量方式、失败阈值及无效样本处理约定。
# - validate_safety_label_contract：要求标签约定与当前版本完全一致，拒绝旧版本或缺失的约定。
# - dataset_safety_label_contract：从数据清单元数据中读取并校验标签约定。
# - SafetyStepMetrics：保存完成一个控制步后、尚未 reset 时的倾角、高度、跟踪误差和接触等指标。
# - local_clearance：在机身投影附近寻找最近有效扫描点，计算相对地面高度及测量有效标记。
# - capture_safety_step：读取真实机器人状态、地面和躯干接触历史，计算安全指标并检查有限性。
# - step_with_safety_capture：临时包装环境 reset，在原 step 的 reset 边界采样，并保证退出时恢复原方法。
# - step_with_safety_capture.capture_and_reset：内部回调，先捕获所有环境的安全指标，再调用原 reset。
# - RolloutOutcome：保存两个时间窗的标签、首次失败及 reset 时间、有效性和连续审计指标。
# - RolloutOutcome.metrics：按固定列顺序将 rollout 审计信息堆叠成张量，供数据集保存。
# - SafetyOutcomeAccumulator：逐步累计一个候选分支的未来安全结果，屏蔽 reset 后的新 episode。
# - SafetyOutcomeAccumulator.__init__：设置环境数、设备、短长时间窗及失败阈值，并初始化累计状态。
# - SafetyOutcomeAccumulator.reset：清空标签、计数、首次事件时间和最大/最小审计指标。
# - SafetyOutcomeAccumulator.update：处理当前安全指标、终止和超时，累计失败并标记未知或截断样本。
# - SafetyOutcomeAccumulator.outcome：要求完成配置的 rollout 步数，返回短失败必然包含于长失败的结果副本。
# 中文导读结束

"""Safety outcome aggregation for 0.4 s and 1.0 s counterfactual rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import torch


TILT_LIMIT_RAD = 0.8
CLEARANCE_LIMIT_M = 0.45
GROUND_RADIUS_M = 0.15
FUTURE_METRIC_ORDER = (
    "first_failure_step", "first_reset_step", "max_tilt", "min_clearance",
    "max_tracking_error", "terminated", "torso_contact", "first_timeout_step",
    "first_invalid_step", "observed_steps",
)


def safety_label_contract() -> dict[str, Any]:
    """Version the physical meaning of labels, independently of policy hashes."""

    return {
        "version": 2,
        "horizon_steps": [20, 50],
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
    fail_short: torch.Tensor
    fail_long: torch.Tensor
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
        short_horizon_steps: int = 20,
        long_horizon_steps: int = 50,
        tilt_limit_rad: float = TILT_LIMIT_RAD,
        clearance_limit_m: float = CLEARANCE_LIMIT_M,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.short_horizon_steps = int(short_horizon_steps)
        self.long_horizon_steps = int(long_horizon_steps)
        self.tilt_limit_rad = float(tilt_limit_rad)
        self.clearance_limit_m = float(clearance_limit_m)
        if not 0 < self.short_horizon_steps <= self.long_horizon_steps:
            raise ValueError("invalid selector label horizons")
        self.reset()

    def reset(self) -> None:
        n = self.num_envs
        self.step_index = 0
        self.fail_short = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fail_long = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_failure_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
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
        if self.step_index >= self.long_horizon_steps:
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
        # Once failure is observed, both horizon outcomes are determined.
        # Before that point, missing measurements cannot establish safety.
        unknown = active & ~self.fail_long & ~measurement_valid
        newly_invalid = unknown & (self.first_invalid_step < 0)
        self.first_invalid_step[newly_invalid] = self.step_index + 1
        self.label_valid &= ~unknown
        timed_out = active & time_outs
        self.first_timeout_step[timed_out] = self.step_index + 1
        if self.step_index + 1 < self.long_horizon_steps:
            self.label_valid &= ~timed_out
        failure = active & (
            non_timeout_termination
            | torso_contact
            | (measurement_valid & (tilt > self.tilt_limit_rad))
            | (measurement_valid & (clearance < self.clearance_limit_m))
        )
        newly_failed = failure & (self.first_failure_step < 0)
        self.first_failure_step[newly_failed] = self.step_index + 1
        newly_reset = active & dones.bool()
        self.first_reset_step[newly_reset] = self.step_index + 1
        if self.step_index < self.short_horizon_steps:
            self.fail_short |= failure
        self.fail_long |= failure
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
        if self.step_index != self.long_horizon_steps:
            raise RuntimeError(
                f"rollout ended after {self.step_index} steps, expected {self.long_horizon_steps}"
            )
        # By construction any short failure is also a long failure.
        self.fail_long |= self.fail_short
        return RolloutOutcome(
            fail_short=self.fail_short.clone(),
            fail_long=self.fail_long.clone(),
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
        )
