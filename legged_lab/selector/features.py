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
    last_valid: torch.Tensor
    last_timestamp: torch.Tensor
    last_packet_timestamp: torch.Tensor
    update_interval: torch.Tensor
    temporal_l1: torch.Tensor
    frozen_ratio: torch.Tensor
    valid_ratio: torch.Tensor
    abnormal_ratio: torch.Tensor
    spatial_mean: torch.Tensor
    spatial_std: torch.Tensor
    initialized: torch.Tensor


class SensorHealthTracker:
    """Cache statistics per packet; query packet age at any control rate.

    valid_mask may be boolean or fractional raw valid support in [0,1]. The
    source timestamp measures freshness; packet_timestamp identifies arrivals.
    A repeated packet never overwrites temporal/spatial statistics. All time
    arithmetic uses float64; only durations enter the FP32 feature vector.
    """

    def __init__(self, num_envs: int, sensor_shape: tuple[int, ...], device):
        self.num_envs, self.sensor_shape = int(num_envs), tuple(sensor_shape)
        self.device = torch.device(device)
        self.last_data = torch.zeros((num_envs, *sensor_shape), device=self.device)
        self.last_valid = torch.zeros_like(self.last_data)
        self.last_timestamp = torch.full(
            (num_envs,), -torch.inf, device=self.device, dtype=torch.float64
        )
        self.last_packet_timestamp = self.last_timestamp.clone()
        for name in (
            "update_interval",
            "temporal_l1",
            "frozen_ratio",
            "valid_ratio",
            "abnormal_ratio",
            "spatial_mean",
            "spatial_std",
        ):
            setattr(self, name, torch.zeros(num_envs, device=self.device))
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for name in _HealthSnapshot.__dataclass_fields__:
            value = getattr(self, name)
            value[env_ids] = (
                -torch.inf if name in ("last_timestamp", "last_packet_timestamp") else 0
            )

    def snapshot(self):
        return _HealthSnapshot(
            **{
                name: getattr(self, name).clone()
                for name in _HealthSnapshot.__dataclass_fields__
            }
        )

    def restore(self, snapshot):
        for name in _HealthSnapshot.__dataclass_fields__:
            target, source = getattr(self, name), getattr(snapshot, name)
            if target.shape != source.shape:
                raise ValueError(f"health snapshot shape mismatch for {name}")
            target.copy_(source)

    def update(
        self,
        data,
        timestamp,
        now,
        *,
        valid_mask=None,
        abnormal_mask=None,
        packet_timestamp=None,
    ):
        if data.shape != self.last_data.shape:
            raise ValueError(
                f"sensor data must have shape {tuple(self.last_data.shape)}"
            )
        source = torch.as_tensor(
            timestamp, device=self.device, dtype=torch.float64
        ).reshape(self.num_envs)
        clock = (
            torch.as_tensor(now, device=self.device, dtype=torch.float64)
            .reshape(-1)
            .expand(self.num_envs)
        )
        packet = (
            source
            if packet_timestamp is None
            else torch.as_tensor(
                packet_timestamp, device=self.device, dtype=torch.float64
            ).reshape(self.num_envs)
        )
        if not torch.isfinite(clock).all():
            raise ValueError("health clock must be finite")
        # A delayed/rewound source packet is still a new arrival. It must not
        # inherit the younger source timestamp from the previous packet.
        new = torch.isfinite(packet) & (
            (~self.initialized) | (packet != self.last_packet_timestamp)
        )
        if new.any():
            weights = torch.isfinite(data).float()
            if valid_mask is not None:
                if valid_mask.shape != data.shape:
                    raise ValueError("valid support must match sensor data")
                if (
                    not torch.isfinite(valid_mask).all()
                    or ((valid_mask < 0) | (valid_mask > 1)).any()
                ):
                    raise ValueError("valid support must be finite and in [0,1]")
                weights *= valid_mask.float()
            safe = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            flat, valid = safe.flatten(1), weights.flatten(1)
            previous = self.last_data.flatten(1)
            common = torch.minimum(valid, self.last_valid.flatten(1))
            count = common.sum(-1).clamp_min(1.0)
            diff = (flat - previous).abs()
            temporal = (diff * common).sum(-1) / count
            frozen = ((diff <= 1e-6).float() * common).sum(-1) / count
            temporal = torch.where(self.initialized, temporal, 0.0)
            frozen = torch.where(self.initialized, frozen, 0.0)
            denominator = valid.sum(-1).clamp_min(1.0)
            mean = (flat * valid).sum(-1) / denominator
            std = torch.sqrt(
                ((flat - mean[:, None]).square() * valid).sum(-1) / denominator
            )
            ratio = valid.mean(-1)
            abnormal = 1.0 - valid
            if abnormal_mask is not None:
                if abnormal_mask.shape != data.shape:
                    raise ValueError("abnormal mask must match sensor data")
                abnormal = torch.maximum(abnormal, abnormal_mask.flatten(1).float())
            interval = torch.where(
                self.initialized,
                (packet - self.last_packet_timestamp).clamp_min(0.0),
                0.0,
            ).float()
            values = dict(
                temporal_l1=temporal,
                frozen_ratio=frozen,
                update_interval=interval,
                valid_ratio=ratio,
                abnormal_ratio=abnormal.mean(-1),
                spatial_mean=mean,
                spatial_std=std,
            )
            for name, value in values.items():
                getattr(self, name)[new] = value[new]
            self.last_data[new] = safe[new]
            self.last_valid[new] = weights[new]
            self.last_timestamp[new] = source[new]
            self.last_packet_timestamp[new] = packet[new]
            self.initialized[new] = True
        age = (clock - self.last_timestamp).clamp_min(0.0)
        age = torch.where(
            self.initialized & torch.isfinite(self.last_timestamp), age, torch.inf
        ).float()
        return SensorHealth(
            frame_age=age,
            **{
                name: getattr(self, name).clone()
                for name in (
                    "update_interval",
                    "valid_ratio",
                    "abnormal_ratio",
                    "temporal_l1",
                    "frozen_ratio",
                    "spatial_mean",
                    "spatial_std",
                )
            },
        )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.ndim != len(shape) or any(
        expected >= 0 and actual != expected
        for actual, expected in zip(tensor.shape, shape)
    ):
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
            F.cosine_similarity(
                branches.lp_latent, branches.vp_latent, dim=-1, eps=1.0e-8
            )[:, None],
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

    def __init__(
        self,
        num_envs: int,
        cfg: SelectorConfig | None = None,
        device: str | torch.device = "cpu",
    ):
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
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.buffer[env_ids] = 0
        self.initialized[env_ids] = False

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        _require_shape("frame", frame, (self.num_envs, self.cfg.frame_dim))
        first = ~self.initialized
        if first.any():
            self.buffer[first] = frame[first, None].expand(
                -1, self.cfg.history_frames, -1
            )
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
        if (
            buffer.shape != self.buffer.shape
            or initialized.shape != self.initialized.shape
        ):
            raise ValueError("feature-history snapshot shape mismatch")
        self.buffer.copy_(buffer)
        self.initialized.copy_(initialized)
