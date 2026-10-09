"""Observable health metrics and the fixed 283-D selector feature layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import SelectorConfig
from .contracts import DualBranchOutput, SensorHealth


@dataclass
class _HealthSnapshot:
    last_data: torch.Tensor
    last_timestamp: torch.Tensor
    update_interval: torch.Tensor
    temporal_l1: torch.Tensor
    frozen_ratio: torch.Tensor
    initialized: torch.Tensor


class SensorHealthTracker:
    """Track health using only values and timestamps available on the robot."""

    def __init__(self, num_envs: int, sensor_shape: tuple[int, ...], device: str | torch.device):
        self.num_envs = int(num_envs)
        self.sensor_shape = tuple(sensor_shape)
        self.device = torch.device(device)
        self.last_data = torch.zeros((self.num_envs, *self.sensor_shape), device=self.device)
        self.last_timestamp = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self.update_interval = torch.zeros(self.num_envs, device=self.device)
        self.temporal_l1 = torch.zeros(self.num_envs, device=self.device)
        self.frozen_ratio = torch.zeros(self.num_envs, device=self.device)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_data[env_ids] = 0
        self.last_timestamp[env_ids] = -torch.inf
        self.update_interval[env_ids] = 0
        self.temporal_l1[env_ids] = 0
        self.frozen_ratio[env_ids] = 0
        self.initialized[env_ids] = False

    def snapshot(self) -> _HealthSnapshot:
        return _HealthSnapshot(
            last_data=self.last_data.clone(),
            last_timestamp=self.last_timestamp.clone(),
            update_interval=self.update_interval.clone(),
            temporal_l1=self.temporal_l1.clone(),
            frozen_ratio=self.frozen_ratio.clone(),
            initialized=self.initialized.clone(),
        )

    def restore(self, snapshot: _HealthSnapshot) -> None:
        for name in _HealthSnapshot.__dataclass_fields__:
            target = getattr(self, name)
            source = getattr(snapshot, name)
            if target.shape != source.shape:
                raise ValueError(f"health snapshot shape mismatch for {name}")
            target.copy_(source)

    def update(
        self,
        data: torch.Tensor,
        timestamp: torch.Tensor,
        now: torch.Tensor | float,
        *,
        valid_mask: torch.Tensor | None = None,
        abnormal_mask: torch.Tensor | None = None,
    ) -> SensorHealth:
        if data.shape != self.last_data.shape:
            raise ValueError(f"sensor data must have shape {tuple(self.last_data.shape)}, got {tuple(data.shape)}")
        timestamp = timestamp.reshape(self.num_envs).to(device=self.device, dtype=torch.float32)
        now_tensor = torch.as_tensor(now, device=self.device, dtype=torch.float32)
        if now_tensor.ndim == 0:
            now_tensor = now_tensor.expand(self.num_envs)
        else:
            now_tensor = now_tensor.reshape(self.num_envs)

        finite = torch.isfinite(data)
        if valid_mask is None:
            valid = finite
        else:
            if valid_mask.shape != data.shape:
                raise ValueError("valid_mask must match sensor data")
            valid = finite & valid_mask.bool()
        if abnormal_mask is None:
            abnormal = ~valid
        else:
            if abnormal_mask.shape != data.shape:
                raise ValueError("abnormal_mask must match sensor data")
            abnormal = (~valid) | abnormal_mask.bool()

        flat_data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).reshape(self.num_envs, -1)
        flat_valid = valid.reshape(self.num_envs, -1)
        flat_abnormal = abnormal.reshape(self.num_envs, -1)
        new_packet = (~self.initialized) | (timestamp > self.last_timestamp + 1.0e-7)

        comparable = flat_valid & torch.isfinite(self.last_data).reshape(self.num_envs, -1)
        differences = (flat_data - self.last_data.reshape(self.num_envs, -1)).abs()
        count = comparable.sum(dim=-1).clamp_min(1)
        temporal_l1 = (differences * comparable).sum(dim=-1) / count
        frozen_ratio = ((differences <= 1.0e-6) & comparable).sum(dim=-1) / count
        temporal_l1 = torch.where(self.initialized, temporal_l1, torch.zeros_like(temporal_l1))
        frozen_ratio = torch.where(self.initialized, frozen_ratio, torch.zeros_like(frozen_ratio))

        finite_previous_time = torch.isfinite(self.last_timestamp)
        interval = torch.where(
            finite_previous_time,
            (timestamp - self.last_timestamp).clamp_min(0.0),
            torch.zeros_like(timestamp),
        )
        self.update_interval.copy_(torch.where(new_packet, interval, self.update_interval))
        # These two observables are evaluated at the selector tick even when
        # the packet timestamp is stale.  That makes a true frozen stream read
        # temporal_l1=0 and frozen_ratio=1 while frame_age keeps increasing.
        self.temporal_l1.copy_(temporal_l1)
        self.frozen_ratio.copy_(frozen_ratio)
        if new_packet.any():
            self.last_data[new_packet] = data[new_packet]
            self.last_timestamp[new_packet] = timestamp[new_packet]
            self.initialized[new_packet] = True

        valid_count = flat_valid.sum(dim=-1)
        valid_ratio = valid_count.float() / flat_valid.shape[1]
        abnormal_ratio = flat_abnormal.float().mean(dim=-1)
        safe_count = valid_count.clamp_min(1)
        spatial_mean = (flat_data * flat_valid).sum(dim=-1) / safe_count
        centered = (flat_data - spatial_mean[:, None]) * flat_valid
        spatial_std = torch.sqrt(centered.square().sum(dim=-1) / safe_count)
        spatial_mean = torch.where(valid_count > 0, spatial_mean, torch.zeros_like(spatial_mean))
        spatial_std = torch.where(valid_count > 0, spatial_std, torch.zeros_like(spatial_std))
        frame_age = torch.where(
            self.initialized,
            (now_tensor - self.last_timestamp).clamp_min(0.0),
            torch.full_like(now_tensor, torch.inf),
        )
        return SensorHealth(
            frame_age=frame_age,
            update_interval=self.update_interval.clone(),
            valid_ratio=valid_ratio,
            abnormal_ratio=abnormal_ratio,
            temporal_l1=self.temporal_l1.clone(),
            frozen_ratio=self.frozen_ratio.clone(),
            spatial_mean=spatial_mean,
            spatial_std=spatial_std,
        )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.ndim != len(shape) or any(expected >= 0 and actual != expected for actual, expected in zip(tensor.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def build_selector_frame(
    proprio: torch.Tensor,
    branches: DualBranchOutput,
    lp_health: SensorHealth,
    vp_health: SensorHealth,
    current_mask: torch.Tensor,
    time_since_switch: torch.Tensor,
    cfg: SelectorConfig | None = None,
) -> torch.Tensor:
    """Build the exact 283-D per-frame feature vector from observable values."""

    cfg = cfg or SelectorConfig()
    cfg.validate()
    batch = proprio.shape[0]
    _require_shape("proprio", proprio, (batch, cfg.proprio_dim))
    _require_shape("lp_latent", branches.lp_latent, (batch, cfg.latent_dim))
    _require_shape("vp_latent", branches.vp_latent, (batch, cfg.latent_dim))
    _require_shape("lp_velocity", branches.lp_velocity, (batch, cfg.velocity_dim))
    _require_shape("vp_velocity", branches.vp_velocity, (batch, cfg.velocity_dim))
    _require_shape("lp_action", branches.lp_action, (batch, cfg.action_dim))
    _require_shape("vp_action", branches.vp_action, (batch, cfg.action_dim))

    lp_health_tensor = lp_health.as_tensor()
    vp_health_tensor = vp_health.as_tensor()
    _require_shape("lp_health", lp_health_tensor, (batch, cfg.health_dim_per_sensor))
    _require_shape("vp_health", vp_health_tensor, (batch, cfg.health_dim_per_sensor))

    latent_difference = branches.lp_latent - branches.vp_latent
    velocity_difference = branches.lp_velocity - branches.vp_velocity
    action_difference = branches.lp_action - branches.vp_action
    cross = torch.cat(
        (
            latent_difference.norm(dim=-1, keepdim=True),
            F.cosine_similarity(branches.lp_latent, branches.vp_latent, dim=-1, eps=1.0e-8)[:, None],
            velocity_difference,
            action_difference.norm(dim=-1, keepdim=True),
            action_difference.abs().amax(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    _require_shape("cross_features", cross, (batch, cfg.cross_dim))
    state = torch.cat(
        (
            current_mask.reshape(batch, 1).to(proprio.dtype),
            time_since_switch.reshape(batch, 1).to(proprio.dtype),
        ),
        dim=-1,
    )
    frame = torch.cat(
        (
            proprio,
            branches.lp_latent,
            branches.vp_latent,
            branches.lp_velocity,
            branches.vp_velocity,
            branches.lp_action,
            branches.vp_action,
            lp_health_tensor.to(proprio.dtype),
            vp_health_tensor.to(proprio.dtype),
            cross,
            state,
        ),
        dim=-1,
    )
    _require_shape("selector_frame", frame, (batch, cfg.frame_dim))
    return frame


class FeatureHistory:
    """Five-frame, 10 Hz history with first-frame replication on reset."""

    def __init__(self, num_envs: int, cfg: SelectorConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or SelectorConfig()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.buffer = torch.zeros(
            self.num_envs,
            self.cfg.history_frames,
            self.cfg.frame_dim,
            device=self.device,
        )
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.buffer[env_ids] = 0
        self.initialized[env_ids] = False

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        _require_shape("frame", frame, (self.num_envs, self.cfg.frame_dim))
        first = ~self.initialized
        if first.any():
            self.buffer[first] = frame[first, None].expand(-1, self.cfg.history_frames, -1)
        subsequent = self.initialized
        if subsequent.any():
            self.buffer[subsequent, :-1] = self.buffer[subsequent, 1:].clone()
            self.buffer[subsequent, -1] = frame[subsequent]
        self.initialized[:] = True
        return self.buffer.clone()

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.buffer.clone(), self.initialized.clone()

    def restore(self, snapshot: tuple[torch.Tensor, torch.Tensor]) -> None:
        buffer, initialized = snapshot
        if buffer.shape != self.buffer.shape or initialized.shape != self.initialized.shape:
            raise ValueError("feature-history snapshot shape mismatch")
        self.buffer.copy_(buffer)
        self.initialized.copy_(initialized)
