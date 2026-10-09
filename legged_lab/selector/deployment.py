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
from .feature_contract import validate_feature_contract
from .frozen_policy import assert_same_policy
from .labels import validate_safety_label_contract


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

    Auto mode owns execution mode/switch time after initialization. A manual
    override must call reset with the actual mode and switch timestamp. Shadow
    mode reads the external actual mode/dwell every call. LP packet time is the
    local-map publication time; LP source time is the underlying scan time.
    """

    @classmethod
    def from_artifacts(cls, dual_branch_path, selector_path, **kwargs):
        dual_extra = {"renet_dual_branch_contract.json": ""}
        selector_extra = {"selector_contract.json": ""}
        dual = torch.jit.load(
            str(dual_branch_path), map_location="cpu", _extra_files=dual_extra
        )
        selector = torch.jit.load(
            str(selector_path), map_location="cpu", _extra_files=selector_extra
        )
        try:
            dual_metadata = json.loads(dual_extra["renet_dual_branch_contract.json"])
            selector_metadata = json.loads(selector_extra["selector_contract.json"])
        except (ValueError, TypeError) as error:
            raise ValueError(
                "both TorchScript artifacts must embed their identity contracts"
            ) from error
        feature_contract = validate_feature_contract(
            selector_metadata.get("selector_feature_contract")
        )
        validate_safety_label_contract(selector_metadata.get("safety_label_contract"))
        assert_same_policy(
            dual_metadata.get("frozen_policy", {}),
            selector_metadata.get("frozen_policy", {}),
        )
        cfg = kwargs.get("runtime_cfg") or RuntimeConfig()
        if (cfg.normal_blend_steps, cfg.emergency_blend_steps) != (
            feature_contract["transition"]["normal_steps"],
            feature_contract["transition"]["emergency_steps"],
        ):
            raise ValueError(
                "runtime action transition differs from the collected label contract"
            )
        return cls(dual, selector, **kwargs)

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
            raise ValueError(
                "proprio history cannot be shorter than the current 78-D frame"
            )
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
        self.shadow_mode = None
        self.last_decision_time = torch.full(
            (self.num_envs,), -torch.inf, device=self.device, dtype=torch.float64
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
        self.shadow_mode = None

    def _validate_input(self, value: SelectorInput) -> None:
        expected = {
            "proprio": (self.num_envs, self.proprio_history_dim),
            "lp_height_map": (self.num_envs, *self.lp_shape),
            "vp_depth_history": (self.num_envs, *self.vp_shape),
        }
        for name, shape in expected.items():
            actual = getattr(value, name)
            if tuple(actual.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(actual.shape)}"
                )
        for name, data in (
            ("lp_valid_mask", value.lp_height_map),
            ("vp_valid_support", value.vp_depth_history),
        ):
            support = getattr(value, name)
            if support is None or support.shape != data.shape:
                raise ValueError(
                    f"{name} must contain raw validity before filling/clipping, with the sensor shape"
                )
        for name in (
            "timestamp",
            "lp_timestamp",
            "vp_timestamp",
            "lp_packet_timestamp",
        ):
            if (
                getattr(value, name) is None
                or getattr(value, name).dtype != torch.float64
            ):
                raise ValueError(
                    f"{name} must be float64; converting a rounded epoch float32 clock cannot recover precision"
                )

    @torch.no_grad()
    def step(self, value: SelectorInput, *, shadow: bool = False) -> OnlineSelectorStep:
        self._validate_input(value)
        timestamp = value.timestamp.to(self.device).reshape(self.num_envs).double()
        current_mask = value.current_mask.to(self.device).reshape(self.num_envs).float()
        if not self.initialized:
            self.reset(initial_mask=current_mask, timestamp=timestamp)
            self.machine.last_switch_time.copy_(
                timestamp
                - value.time_since_switch.to(self.device)
                .reshape(self.num_envs)
                .double()
            )
        if self.shadow_mode is not None and self.shadow_mode != shadow:
            self.machine.reset(initial_mask=current_mask, now=timestamp)
            self.machine.last_switch_time.copy_(
                timestamp
                - value.time_since_switch.to(self.device)
                .reshape(self.num_envs)
                .double()
            )
            self.last_decision_time.fill_(-torch.inf)
            self.last_decision = None
            self.blender.seed(
                value.proprio.to(self.device)[:, -self.selector_cfg.action_dim :]
            )
        self.shadow_mode = shadow
        if not self.blender.initialized.all():
            # The frozen policy uses action observation scale 1.0. The newest
            # proprio frame ends with the last executed, blended action.
            self.blender.seed(
                value.proprio.to(self.device)[:, -self.selector_cfg.action_dim :]
            )
        if shadow:
            self.machine.current_mask.copy_(current_mask)
            self.machine.last_switch_time.copy_(
                timestamp
                - value.time_since_switch.to(self.device)
                .reshape(self.num_envs)
                .double()
            )
        else:
            current_mask = self.machine.current_mask.clone()

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

        lp_health = self.lp_health.update(
            lp_map,
            value.lp_timestamp.to(self.device),
            timestamp,
            valid_mask=value.lp_valid_mask.to(self.device),
            packet_timestamp=None
            if value.lp_packet_timestamp is None
            else value.lp_packet_timestamp.to(self.device),
        )
        vp_health = self.vp_health.update(
            vp_depth,
            value.vp_timestamp.to(self.device),
            timestamp,
            valid_mask=value.vp_valid_support.to(self.device),
            packet_timestamp=None
            if value.vp_packet_timestamp is None
            else value.vp_packet_timestamp.to(self.device),
        )

        due = timestamp - self.last_decision_time >= 0.1 - 1.0e-6
        if due.any() and not due.all():
            raise RuntimeError(
                "batched runtime requires synchronized 10 Hz selector timestamps"
            )
        decision_updated = bool(due.all())
        if decision_updated:
            normalized = self.observation_normalizer(policy_observation)
            current_proprio = normalized[
                :,
                self.proprio_history_dim
                - self.selector_cfg.proprio_dim : self.proprio_history_dim,
            ]
            frame = build_selector_frame(
                current_proprio,
                branches,
                lp_health,
                vp_health,
                current_mask,
                (timestamp - self.machine.last_switch_time).clamp_min(0).float(),
                self.selector_cfg,
            )
            history = self.history.append(frame)
            prediction = self.selector(history)
            if isinstance(prediction, (tuple, list)):
                risks, uncertainty = prediction
            else:
                risks = prediction
                uncertainty = torch.zeros_like(risks)
            self.last_decision = self.machine.step(
                risks, lp_health, vp_health, timestamp
            )
            self.last_uncertainty = uncertainty
            self.last_decision_time.copy_(timestamp)
        if self.last_decision is None:
            raise RuntimeError("selector did not produce an initial decision")

        decision = self.last_decision
        if shadow:
            control_action = torch.where(
                current_mask[:, None] > 0.5, branches.lp_action, branches.vp_action
            )
            self.blender.seed(control_action)
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
