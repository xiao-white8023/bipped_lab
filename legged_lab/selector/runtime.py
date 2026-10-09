"""Deterministic hard gates, hysteresis, and action transition logic."""

from __future__ import annotations

import torch

from .config import RuntimeConfig
from .contracts import SelectorDecision, SensorHealth


class SelectorStateMachine:
    """Batched online LP/VP choice; mask 1 is LP and mask 0 is VP."""

    REASON_HOLD = 0
    REASON_HARD_TO_LP = 1
    REASON_HARD_TO_VP = 2
    REASON_RISK_TO_LP = 3
    REASON_RISK_TO_VP = 4
    REASON_TIE_TO_VP = 5
    REASON_BOTH_UNRELIABLE = 6

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        initial_mask: float = 0.0,
        cfg: RuntimeConfig | None = None,
    ):
        self.cfg = cfg or RuntimeConfig()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.current_mask = torch.full(
            (self.num_envs,), float(initial_mask), device=self.device
        )
        self.last_switch_time = torch.full(
            (self.num_envs,), -torch.inf, device=self.device, dtype=torch.float64
        )
        self.pending_mask = self.current_mask.clone()
        self.pending_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.lp_invalid_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.vp_invalid_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        initial_mask: float | torch.Tensor = 0.0,
        now: float | torch.Tensor = 0.0,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        mask = torch.as_tensor(initial_mask, device=self.device, dtype=torch.float32)
        time = torch.as_tensor(now, device=self.device, dtype=torch.float64)
        if mask.ndim == 0:
            mask = mask.expand(len(env_ids))
        if time.ndim == 0:
            time = time.expand(len(env_ids))
        self.current_mask[env_ids] = mask.reshape(-1)
        self.last_switch_time[env_ids] = time.reshape(-1)
        self.pending_mask[env_ids] = self.current_mask[env_ids]
        self.pending_count[env_ids] = 0
        self.lp_invalid_count[env_ids] = 0
        self.vp_invalid_count[env_ids] = 0

    def _availability(
        self, health: SensorHealth, invalid_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = health.as_tensor()
        finite = torch.isfinite(values).all(dim=-1)
        invalid_now = (
            (~finite)
            | (health.frame_age > self.cfg.sensor_timeout_s)
            | (health.valid_ratio < self.cfg.min_valid_ratio)
        )
        updated_count = torch.where(
            invalid_now, invalid_count + 1, torch.zeros_like(invalid_count)
        )
        available = (
            finite
            & (health.frame_age <= self.cfg.sensor_timeout_s)
            & (updated_count < self.cfg.invalid_confirmations)
        )
        return available, updated_count

    def step(
        self,
        risks: torch.Tensor,
        lp_health: SensorHealth,
        vp_health: SensorHealth,
        now: float | torch.Tensor,
    ) -> SelectorDecision:
        if risks.shape != (self.num_envs, 4):
            raise ValueError(f"risks must have shape {(self.num_envs, 4)}")
        if not torch.isfinite(risks).all():
            raise ValueError("selector risks must be finite")
        now_tensor = torch.as_tensor(now, device=self.device, dtype=torch.float64)
        if now_tensor.ndim == 0:
            now_tensor = now_tensor.expand(self.num_envs)
        else:
            now_tensor = now_tensor.reshape(self.num_envs)

        lp_available, self.lp_invalid_count = self._availability(
            lp_health, self.lp_invalid_count
        )
        vp_available, self.vp_invalid_count = self._availability(
            vp_health, self.vp_invalid_count
        )
        lp_risk = (
            self.cfg.short_risk_weight * risks[:, 0]
            + self.cfg.long_risk_weight * risks[:, 1]
        )
        vp_risk = (
            self.cfg.short_risk_weight * risks[:, 2]
            + self.cfg.long_risk_weight * risks[:, 3]
        )
        both_high = (lp_risk >= self.cfg.both_unreliable_risk) & (
            vp_risk >= self.cfg.both_unreliable_risk
        )
        both_unavailable = ~lp_available & ~vp_available
        both_unreliable = both_high | both_unavailable

        old_mask = self.current_mask.clone()
        switched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        emergency = torch.zeros_like(switched)
        reason = torch.full(
            (self.num_envs,), self.REASON_HOLD, dtype=torch.long, device=self.device
        )

        using_lp = self.current_mask > 0.5
        hard_to_vp = using_lp & ~lp_available & vp_available
        hard_to_lp = ~using_lp & ~vp_available & lp_available
        if hard_to_vp.any():
            self.current_mask[hard_to_vp] = 0.0
            switched[hard_to_vp] = True
            emergency[hard_to_vp] = True
            reason[hard_to_vp] = self.REASON_HARD_TO_VP
        if hard_to_lp.any():
            self.current_mask[hard_to_lp] = 1.0
            switched[hard_to_lp] = True
            emergency[hard_to_lp] = True
            reason[hard_to_lp] = self.REASON_HARD_TO_LP

        hard_switched = hard_to_vp | hard_to_lp
        if hard_switched.any():
            self.last_switch_time[hard_switched] = now_tensor[hard_switched]
            self.pending_count[hard_switched] = 0
            self.pending_mask[hard_switched] = self.current_mask[hard_switched]

        eligible = ~hard_switched & ~both_unreliable & lp_available & vp_available
        dwell_ok = (now_tensor - self.last_switch_time) >= self.cfg.min_dwell_s
        desired = self.current_mask.clone()
        desired_reason = torch.full_like(reason, self.REASON_HOLD)
        risk_difference = lp_risk - vp_risk
        tie = risk_difference.abs() <= self.cfg.vp_tie_margin
        tie_to_vp = eligible & tie & (self.current_mask > 0.5)
        desired[tie_to_vp] = 0.0
        desired_reason[tie_to_vp] = self.REASON_TIE_TO_VP

        risk_to_vp = (
            eligible
            & (self.current_mask > 0.5)
            & (vp_risk + self.cfg.switch_margin < lp_risk)
        )
        risk_to_lp = (
            eligible
            & (self.current_mask <= 0.5)
            & (lp_risk + self.cfg.switch_margin < vp_risk)
        )
        desired[risk_to_vp] = 0.0
        desired_reason[risk_to_vp] = self.REASON_RISK_TO_VP
        desired[risk_to_lp] = 1.0
        desired_reason[risk_to_lp] = self.REASON_RISK_TO_LP

        wants_change = eligible & dwell_ok & (desired != self.current_mask)
        same_pending = desired == self.pending_mask
        self.pending_count = torch.where(
            wants_change,
            torch.where(
                same_pending,
                self.pending_count + 1,
                torch.ones_like(self.pending_count),
            ),
            torch.zeros_like(self.pending_count),
        )
        self.pending_mask = torch.where(wants_change, desired, self.current_mask)
        confirmed = wants_change & (self.pending_count >= self.cfg.switch_confirmations)
        if confirmed.any():
            self.current_mask[confirmed] = desired[confirmed]
            self.last_switch_time[confirmed] = now_tensor[confirmed]
            switched[confirmed] = True
            reason[confirmed] = desired_reason[confirmed]
            self.pending_count[confirmed] = 0
            self.pending_mask[confirmed] = self.current_mask[confirmed]

        reason[both_unreliable & ~switched] = self.REASON_BOTH_UNRELIABLE
        if not torch.equal(old_mask, self.current_mask):
            changed = old_mask != self.current_mask
            if not torch.equal(changed, switched):
                raise RuntimeError("selector internal switched mask is inconsistent")
        return SelectorDecision(
            selected_mask=self.current_mask[:, None].clone(),
            risks=risks,
            lp_available=lp_available[:, None],
            vp_available=vp_available[:, None],
            both_unreliable=both_unreliable[:, None],
            switched=switched[:, None],
            emergency_switch=emergency[:, None],
            reason_code=reason[:, None],
        )


class ActionBlender:
    """Blend final actions after a binary mask switch; never interpolate masks."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        cfg: RuntimeConfig | None = None,
    ):
        self.cfg = cfg or RuntimeConfig()
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.last_output = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.start_action = self.last_output.clone()
        self.remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.total = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_output[env_ids] = 0
        self.start_action[env_ids] = 0
        self.remaining[env_ids] = 0
        self.total[env_ids] = 1
        self.initialized[env_ids] = False

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).clone()
            for name in (
                "last_output",
                "start_action",
                "remaining",
                "total",
                "initialized",
            )
        }

    def restore(self, state: dict[str, torch.Tensor]) -> None:
        for name, source in state.items():
            getattr(self, name).copy_(source)

    def seed(self, action: torch.Tensor) -> None:
        if action.shape != self.last_output.shape:
            raise ValueError("initial action must match the action contract")
        self.last_output.copy_(action)
        self.start_action.copy_(action)
        self.remaining.zero_()
        self.total.fill_(1)
        self.initialized.fill_(True)

    def step(
        self,
        lp_action: torch.Tensor,
        vp_action: torch.Tensor,
        selected_mask: torch.Tensor,
        switched: torch.Tensor,
        emergency_switch: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.num_envs, self.action_dim)
        if lp_action.shape != expected or vp_action.shape != expected:
            raise ValueError(f"candidate actions must have shape {expected}")
        mask = selected_mask.reshape(self.num_envs, 1) > 0.5
        target = torch.where(mask, lp_action, vp_action)
        if not torch.isfinite(target).all():
            raise ValueError("selected candidate action must be finite")
        first = ~self.initialized
        self.last_output[first] = target[first]
        self.initialized[first] = True

        switch = switched.reshape(self.num_envs).bool()
        emergency = emergency_switch.reshape(self.num_envs).bool()
        if switch.any():
            self.start_action[switch] = self.last_output[switch]
            normal_steps = torch.full_like(
                self.total[switch], self.cfg.normal_blend_steps
            )
            emergency_steps = torch.full_like(
                self.total[switch], self.cfg.emergency_blend_steps
            )
            self.total[switch] = torch.where(
                emergency[switch], emergency_steps, normal_steps
            ).clamp_min(1)
            self.remaining[switch] = self.total[switch]

        blending = self.remaining > 0
        output = target.clone()
        if blending.any():
            completed = self.total[blending] - self.remaining[blending] + 1
            alpha = completed.to(target.dtype) / self.total[blending].to(target.dtype)
            output[blending] = (1.0 - alpha[:, None]) * self.start_action[
                blending
            ] + alpha[:, None] * target[blending]
            self.remaining[blending] -= 1
        self.last_output.copy_(output)
        return output
