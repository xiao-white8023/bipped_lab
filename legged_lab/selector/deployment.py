"""Online and shadow-mode runtime for the exported LP/VP selector artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import RuntimeConfig, SelectorConfig
from .contracts import DualBranchOutput, SelectorDecision, SelectorInput, SelectorOutput
from .features import FeatureHistory, SensorHealthTracker, build_selector_frame
from .runtime import ActionBlender, SelectorStateMachine


@dataclass
class OnlineSelectorStep:
    output: SelectorOutput
    control_action: torch.Tensor
    lp_action: torch.Tensor
    vp_action: torch.Tensor
    uncertainty: torch.Tensor


class OnlineSelectorRuntime:
    """Compose the two immutable artifacts without changing RENet.

    ``step`` may be called at the 50 Hz control rate. Risk inference and the
    state machine update at 10 Hz; both candidate actions refresh every call.
    In shadow mode, the recommendation is returned but the input mask keeps
    control of the robot.
    """

    def __init__(
        self,
        dual_branch: nn.Module,
        selector: nn.Module,
        *,
        proprio_history_dim: int = 780,
        lp_shape: tuple[int, ...] = (11, 17),
        vp_shape: tuple[int, ...] = (2, 18, 32),
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        selector_cfg: SelectorConfig | None = None,
        runtime_cfg: RuntimeConfig | None = None,
        observation_normalizer: nn.Module | None = None,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.proprio_history_dim = int(proprio_history_dim)
        self.lp_shape = tuple(lp_shape)
        self.vp_shape = tuple(vp_shape)
        self.selector_cfg = selector_cfg or SelectorConfig()
        self.runtime_cfg = runtime_cfg or RuntimeConfig()
        self.selector_cfg.validate()
        self.runtime_cfg.validate()
        if self.proprio_history_dim < self.selector_cfg.proprio_dim:
            raise ValueError("proprio history cannot be shorter than the current 78-D frame")
        self.dual_branch = dual_branch.to(self.device).eval()
        self.selector = selector.to(self.device).eval()
        self.observation_normalizer = (
            observation_normalizer.to(self.device).eval()
            if observation_normalizer is not None
            else nn.Identity()
        )
        self.lp_health = SensorHealthTracker(self.num_envs, self.lp_shape, self.device)
        self.vp_health = SensorHealthTracker(self.num_envs, self.vp_shape, self.device)
        self.history = FeatureHistory(self.num_envs, self.selector_cfg, self.device)
        self.machine = SelectorStateMachine(
            self.num_envs, device=self.device, cfg=self.runtime_cfg
        )
        self.blender = ActionBlender(
            self.num_envs,
            self.selector_cfg.action_dim,
            device=self.device,
            cfg=self.runtime_cfg,
        )
        self.initialized = False
        self.last_decision_time = torch.full(
            (self.num_envs,), -torch.inf, device=self.device
        )
        self.last_decision: SelectorDecision | None = None
        self.last_uncertainty = torch.zeros(self.num_envs, 4, device=self.device)

    @staticmethod
    def reason_name(code: int) -> str:
        names = {
            SelectorStateMachine.REASON_HOLD: "hold",
            SelectorStateMachine.REASON_HARD_TO_LP: "hard_to_lp",
            SelectorStateMachine.REASON_HARD_TO_VP: "hard_to_vp",
            SelectorStateMachine.REASON_RISK_TO_LP: "risk_to_lp",
            SelectorStateMachine.REASON_RISK_TO_VP: "risk_to_vp",
            SelectorStateMachine.REASON_TIE_TO_VP: "tie_to_vp",
            SelectorStateMachine.REASON_BOTH_UNRELIABLE: "both_unreliable",
        }
        return names.get(int(code), "unknown")

    def reset(
        self,
        *,
        initial_mask: float | torch.Tensor = 0.0,
        timestamp: float | torch.Tensor = 0.0,
    ) -> None:
        self.lp_health.reset()
        self.vp_health.reset()
        self.history.reset()
        self.machine.reset(initial_mask=initial_mask, now=timestamp)
        self.blender.reset()
        self.last_decision_time.fill_(-torch.inf)
        self.last_decision = None
        self.last_uncertainty.zero_()
        self.initialized = True

    def _validate_input(self, value: SelectorInput) -> None:
        expected = {
            "proprio": (self.num_envs, self.proprio_history_dim),
            "lp_height_map": (self.num_envs, *self.lp_shape),
            "vp_depth_history": (self.num_envs, *self.vp_shape),
        }
        for name, shape in expected.items():
            actual = getattr(value, name)
            if tuple(actual.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(actual.shape)}")

    @torch.no_grad()
    def step(self, value: SelectorInput, *, shadow: bool = False) -> OnlineSelectorStep:
        self._validate_input(value)
        timestamp = value.timestamp.to(self.device).reshape(self.num_envs).float()
        current_mask = value.current_mask.to(self.device).reshape(self.num_envs).float()
        if not self.initialized:
            self.reset(initial_mask=current_mask, timestamp=timestamp)

        proprio = value.proprio.to(self.device)
        lp_map = value.lp_height_map.to(self.device)
        vp_depth = value.vp_depth_history.to(self.device)
        policy_observation = torch.cat(
            (
                proprio,
                lp_map.reshape(self.num_envs, -1),
                vp_depth.reshape(self.num_envs, -1),
                current_mask[:, None],
            ),
            dim=-1,
        )
        raw = self.dual_branch(policy_observation)
        if not isinstance(raw, (tuple, list)) or len(raw) != 6:
            raise RuntimeError("renet_dual_branch.pt must return six tensors")
        branches = DualBranchOutput(*raw)

        lp_valid = torch.isfinite(lp_map) & (lp_map != -1.1)
        vp_valid = torch.isfinite(vp_depth) & (vp_depth >= 0.0) & (vp_depth <= 1.0)
        lp_health = self.lp_health.update(
            lp_map,
            value.lp_timestamp.to(self.device),
            timestamp,
            valid_mask=lp_valid,
        )
        vp_health = self.vp_health.update(
            vp_depth,
            value.vp_timestamp.to(self.device),
            timestamp,
            valid_mask=vp_valid,
            abnormal_mask=~vp_valid,
        )

        due = timestamp - self.last_decision_time >= 0.1 - 1.0e-6
        if due.any() and not due.all():
            raise RuntimeError("batched runtime requires synchronized 10 Hz selector timestamps")
        decision_updated = bool(due.all())
        if decision_updated:
            normalized = self.observation_normalizer(policy_observation)
            current_proprio = normalized[
                :, self.proprio_history_dim - self.selector_cfg.proprio_dim : self.proprio_history_dim
            ]
            frame = build_selector_frame(
                current_proprio,
                branches,
                lp_health,
                vp_health,
                current_mask,
                value.time_since_switch.to(self.device),
                self.selector_cfg,
            )
            history = self.history.append(frame)
            prediction = self.selector(history)
            if isinstance(prediction, (tuple, list)):
                risks, uncertainty = prediction
            else:
                risks = prediction
                uncertainty = torch.zeros_like(risks)
            self.last_decision = self.machine.step(risks, lp_health, vp_health, timestamp)
            self.last_uncertainty = uncertainty
            self.last_decision_time.copy_(timestamp)
        if self.last_decision is None:
            raise RuntimeError("selector did not produce an initial decision")

        decision = self.last_decision
        if shadow:
            control_action = current_mask[:, None] * branches.lp_action + (
                1.0 - current_mask[:, None]
            ) * branches.vp_action
        else:
            switched = (
                decision.switched
                if decision_updated
                else torch.zeros_like(decision.switched)
            )
            emergency = (
                decision.emergency_switch
                if decision_updated
                else torch.zeros_like(decision.emergency_switch)
            )
            control_action = self.blender.step(
                branches.lp_action,
                branches.vp_action,
                decision.selected_mask,
                switched,
                emergency,
            )
        public = SelectorOutput(
            selected_mask=decision.selected_mask,
            lp_risk_short=decision.lp_risk_short,
            lp_risk_long=decision.lp_risk_long,
            vp_risk_short=decision.vp_risk_short,
            vp_risk_long=decision.vp_risk_long,
            lp_available=decision.lp_available,
            vp_available=decision.vp_available,
            both_unreliable=decision.both_unreliable,
            switch_reason=decision.reason_code,
        )
        return OnlineSelectorStep(
            output=public,
            control_action=control_action,
            lp_action=branches.lp_action,
            vp_action=branches.vp_action,
            uncertainty=self.last_uncertainty,
        )


class ShadowLogWriter:
    """Append deployment-observable selector records as JSON Lines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")

    @staticmethod
    def _values(tensor: torch.Tensor) -> list[Any]:
        return tensor.detach().cpu().reshape(-1).tolist()

    def append(
        self,
        value: SelectorInput,
        step: OnlineSelectorStep,
        *,
        robot_state: dict[str, Any] | None = None,
    ) -> None:
        output = step.output
        record = {
            "timestamp": self._values(value.timestamp),
            "actual_mask": self._values(value.current_mask),
            "suggested_mask": self._values(output.selected_mask),
            "lp_risk_short": self._values(output.lp_risk_short),
            "lp_risk_long": self._values(output.lp_risk_long),
            "vp_risk_short": self._values(output.vp_risk_short),
            "vp_risk_long": self._values(output.vp_risk_long),
            "lp_available": self._values(output.lp_available),
            "vp_available": self._values(output.vp_available),
            "both_unreliable": self._values(output.both_unreliable),
            "switch_reason": self._values(output.switch_reason),
            "robot_state": robot_state or {},
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "ShadowLogWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
