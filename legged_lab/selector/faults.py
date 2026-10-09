"""Vectorized, snapshot-safe LP/VP structural fault injection.

Faults are applied to copies of the exact tensors consumed by the frozen
policy.  Consequently the module is isolated from the normal G1 environment:
when a fault is inactive the returned sensor tensors are bit-identical to the
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
import torch.nn.functional as F

from .config import FaultConfig


class FaultKind(IntEnum):
    NONE = 0
    FREEZE = 1
    DELAY = 2
    VP_OCCLUSION = 3
    VP_SCALE = 4
    VP_BIAS = 5
    VP_SHIFT = 6
    VP_COLLAPSE = 7
    LP_RANDOM_DROPOUT = 8
    LP_BLOCK_DROPOUT = 9
    LP_HEIGHT_BIAS = 10
    LP_REGISTRATION = 11
    LP_DRIFT = 12
    LP_COLLAPSE = 13


VP_FAULTS = (
    FaultKind.FREEZE,
    FaultKind.DELAY,
    FaultKind.VP_OCCLUSION,
    FaultKind.VP_SCALE,
    FaultKind.VP_BIAS,
    FaultKind.VP_SHIFT,
    FaultKind.VP_COLLAPSE,
)
LP_FAULTS = (
    FaultKind.FREEZE,
    FaultKind.DELAY,
    FaultKind.LP_RANDOM_DROPOUT,
    FaultKind.LP_BLOCK_DROPOUT,
    FaultKind.LP_HEIGHT_BIAS,
    FaultKind.LP_REGISTRATION,
    FaultKind.LP_DRIFT,
    FaultKind.LP_COLLAPSE,
)


@dataclass
class FaultSnapshot:
    tensors: dict[str, torch.Tensor]
    cursor: int
    generator_state: torch.Tensor


@dataclass
class FaultBatch:
    lp: torch.Tensor
    vp: torch.Tensor
    lp_timestamp: torch.Tensor
    vp_timestamp: torch.Tensor
    lp_active: torch.Tensor
    vp_active: torch.Tensor


class SelectorFaultManager:
    """One independently sampled fault schedule per simulated environment."""

    _SNAPSHOT_TENSORS = (
        "step_count",
        "lp_kind",
        "vp_kind",
        "severity",
        "start_frame",
        "duration_frames",
        "lp_delay",
        "vp_delay",
        "vp_scale",
        "vp_bias",
        "vp_shift_x",
        "vp_shift_y",
        "vp_collapse_value",
        "lp_dropout",
        "lp_bias",
        "lp_shift_x",
        "lp_shift_y",
        "lp_yaw",
        "lp_drift_xy",
        "lp_drift_yaw",
        "vp_occlusion_mask",
        "lp_block_mask",
        "lp_buffer",
        "vp_buffer",
        "lp_time_buffer",
        "vp_time_buffer",
        "buffer_initialized",
        "lp_frozen",
        "vp_frozen",
        "lp_frozen_time",
        "vp_frozen_time",
        "lp_freeze_initialized",
        "vp_freeze_initialized",
    )

    def __init__(
        self,
        num_envs: int,
        *,
        lp_shape: tuple[int, int] = (11, 17),
        vp_shape: tuple[int, int, int] = (2, 18, 32),
        device: str | torch.device = "cpu",
        seed: int = 42,
        cfg: FaultConfig | None = None,
    ):
        self.cfg = cfg or FaultConfig()
        self.cfg.validate()
        self.num_envs = int(num_envs)
        self.lp_shape = tuple(lp_shape)
        self.vp_shape = tuple(vp_shape)
        self.device = torch.device(device)
        generator_device = self.device.type if self.device.type in ("cpu", "cuda") else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed))
        n = self.num_envs
        capacity = self.cfg.max_delay_frames + 1
        self.step_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.lp_kind = torch.zeros(n, dtype=torch.long, device=self.device)
        self.vp_kind = torch.zeros(n, dtype=torch.long, device=self.device)
        self.severity = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.duration_frames = torch.ones(n, dtype=torch.long, device=self.device)
        self.lp_delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self.vp_delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self.vp_scale = torch.ones(n, device=self.device)
        self.vp_bias = torch.zeros(n, device=self.device)
        self.vp_shift_x = torch.zeros(n, dtype=torch.long, device=self.device)
        self.vp_shift_y = torch.zeros(n, dtype=torch.long, device=self.device)
        self.vp_collapse_value = torch.zeros(n, device=self.device)
        self.lp_dropout = torch.zeros(n, device=self.device)
        self.lp_bias = torch.zeros(n, device=self.device)
        self.lp_shift_x = torch.zeros(n, dtype=torch.long, device=self.device)
        self.lp_shift_y = torch.zeros(n, dtype=torch.long, device=self.device)
        self.lp_yaw = torch.zeros(n, device=self.device)
        self.lp_drift_xy = torch.zeros(n, 2, device=self.device)
        self.lp_drift_yaw = torch.zeros(n, device=self.device)
        self.vp_occlusion_mask = torch.zeros((n, *self.vp_shape[-2:]), dtype=torch.bool, device=self.device)
        self.lp_block_mask = torch.zeros((n, *self.lp_shape), dtype=torch.bool, device=self.device)
        self.lp_buffer = torch.full(
            (n, capacity, *self.lp_shape), self.cfg.lp_invalid_height, device=self.device
        )
        self.vp_buffer = torch.ones((n, capacity, *self.vp_shape), device=self.device)
        self.lp_time_buffer = torch.zeros(n, capacity, device=self.device)
        self.vp_time_buffer = torch.zeros(n, capacity, device=self.device)
        self.buffer_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.lp_frozen = torch.full((n, *self.lp_shape), self.cfg.lp_invalid_height, device=self.device)
        self.vp_frozen = torch.ones((n, *self.vp_shape), device=self.device)
        self.lp_frozen_time = torch.zeros(n, device=self.device)
        self.vp_frozen_time = torch.zeros(n, device=self.device)
        self.lp_freeze_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.vp_freeze_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.cursor = -1
        self.reset()

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def _randint(self, low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high, shape, device=self.device, generator=self.generator)

    @staticmethod
    def _severity_bounds(level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.where(level == 0, 0.0, torch.where(level == 1, 0.35, 0.70))
        upper = torch.where(level == 0, 0.35, torch.where(level == 1, 0.70, 1.0))
        return lower, upper

    def _strength(self, env_ids: torch.Tensor) -> torch.Tensor:
        level = self.severity[env_ids]
        lower, upper = self._severity_bounds(level)
        return lower + (upper - lower) * self._rand(len(env_ids))

    def _sample_signed(self, env_ids: torch.Tensor, max_abs: float, minimum_abs: float = 0.0) -> torch.Tensor:
        strength = self._strength(env_ids)
        magnitude = minimum_abs + (max_abs - minimum_abs) * strength
        sign = torch.where(self._rand(len(env_ids)) < 0.5, -1.0, 1.0)
        return magnitude * sign

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = len(env_ids)
        if count == 0:
            return
        scenario = torch.multinomial(
            torch.tensor(self.cfg.scenario_probabilities, device=self.device),
            count,
            replacement=True,
            generator=self.generator,
        )
        severity = torch.multinomial(
            torch.tensor(self.cfg.severity_probabilities, device=self.device),
            count,
            replacement=True,
            generator=self.generator,
        )
        self.severity[env_ids] = severity
        self.lp_kind[env_ids] = int(FaultKind.NONE)
        self.vp_kind[env_ids] = int(FaultKind.NONE)
        lp_fault = (scenario == 2) | (scenario == 3)
        vp_fault = (scenario == 1) | (scenario == 3)
        if lp_fault.any():
            choices = torch.tensor([int(kind) for kind in LP_FAULTS], device=self.device)
            self.lp_kind[env_ids[lp_fault]] = choices[
                self._randint(0, len(choices), (int(lp_fault.sum().item()),))
            ]
        if vp_fault.any():
            choices = torch.tensor([int(kind) for kind in VP_FAULTS], device=self.device)
            self.vp_kind[env_ids[vp_fault]] = choices[
                self._randint(0, len(choices), (int(vp_fault.sum().item()),))
            ]

        self.step_count[env_ids] = 0
        self.start_frame[env_ids] = self._randint(
            0, self.cfg.max_start_delay_frames + 1, (count,)
        )
        strength = self._strength(env_ids)
        duration_span = self.cfg.max_duration_frames - self.cfg.min_duration_frames
        self.duration_frames[env_ids] = (
            self.cfg.min_duration_frames + torch.round(strength * duration_span).long()
        )
        self.lp_delay[env_ids] = 1 + torch.round(strength * (self.cfg.max_delay_frames - 1)).long()
        self.vp_delay[env_ids] = 1 + torch.round(strength * (self.cfg.max_delay_frames - 1)).long()
        self.vp_scale[env_ids] = 1.0 + self._sample_signed(env_ids, 0.15)
        self.vp_bias[env_ids] = self._sample_signed(env_ids, 0.20) / self.cfg.vp_depth_max_m
        shifts = torch.round(1 + strength * (self.cfg.vp_max_shift_pixels - 1)).long()
        self.vp_shift_x[env_ids] = shifts * torch.where(self._rand(count) < 0.5, -1, 1)
        self.vp_shift_y[env_ids] = shifts * torch.where(self._rand(count) < 0.5, -1, 1)
        # 0: all-zero depth, 1: all maximum-range depth, -1: explicit
        # invalid-depth sentinel.  NaN is deliberately avoided because the
        # frozen actor must never receive non-finite inputs.
        collapse_values = torch.tensor((0.0, 1.0, -1.0), device=self.device)
        self.vp_collapse_value[env_ids] = collapse_values[
            self._randint(0, len(collapse_values), (count,))
        ]
        lo, hi = self.cfg.lp_dropout_fraction
        self.lp_dropout[env_ids] = lo + strength * (hi - lo)
        self.lp_bias[env_ids] = self._sample_signed(env_ids, 0.15)
        max_cell_shift = max(1, round(max(abs(v) for v in self.cfg.lp_translation_range_m) / 0.1))
        lp_shifts = torch.round(1 + strength * (max_cell_shift - 1)).long()
        self.lp_shift_x[env_ids] = lp_shifts * torch.where(self._rand(count) < 0.5, -1, 1)
        self.lp_shift_y[env_ids] = lp_shifts * torch.where(self._rand(count) < 0.5, -1, 1)
        self.lp_yaw[env_ids] = self._sample_signed(env_ids, max(abs(v) for v in self.cfg.lp_yaw_range_deg))
        drift_speed = self.cfg.lp_drift_m_per_s[1] * strength
        drift_angle = 2 * math.pi * self._rand(count)
        self.lp_drift_xy[env_ids, 0] = drift_speed * drift_angle.cos()
        self.lp_drift_xy[env_ids, 1] = drift_speed * drift_angle.sin()
        self.lp_drift_yaw[env_ids] = self._sample_signed(env_ids, self.cfg.lp_drift_deg_per_s[1])
        self._sample_rectangles(env_ids, strength)

        self.buffer_initialized[env_ids] = False
        self.lp_freeze_initialized[env_ids] = False
        self.vp_freeze_initialized[env_ids] = False
        self.lp_buffer[env_ids] = self.cfg.lp_invalid_height
        self.vp_buffer[env_ids] = 1.0
        self.lp_time_buffer[env_ids] = 0.0
        self.vp_time_buffer[env_ids] = 0.0

    def _sample_rectangles(self, env_ids: torch.Tensor, strength: torch.Tensor) -> None:
        self.vp_occlusion_mask[env_ids] = False
        self.lp_block_mask[env_ids] = False
        vp_h, vp_w = self.vp_shape[-2:]
        lp_h, lp_w = self.lp_shape
        vp_lo, vp_hi = self.cfg.vp_occlusion_fraction
        lp_lo, lp_hi = self.cfg.lp_block_fraction
        vp_fraction = vp_lo + strength * (vp_hi - vp_lo)
        lp_fraction = lp_lo + strength * (lp_hi - lp_lo)
        for local_index, env_id in enumerate(env_ids.tolist()):
            vp_area = max(1, round(float(vp_fraction[local_index]) * vp_h * vp_w))
            vp_rect_h = max(1, min(vp_h, round(math.sqrt(vp_area * vp_h / vp_w))))
            vp_rect_w = max(1, min(vp_w, math.ceil(vp_area / vp_rect_h)))
            top = int(self._randint(0, vp_h - vp_rect_h + 1, (1,)).item())
            left = int(self._randint(0, vp_w - vp_rect_w + 1, (1,)).item())
            self.vp_occlusion_mask[env_id, top : top + vp_rect_h, left : left + vp_rect_w] = True

            lp_area = max(1, round(float(lp_fraction[local_index]) * lp_h * lp_w))
            lp_rect_h = max(1, min(lp_h, round(math.sqrt(lp_area * lp_h / lp_w))))
            lp_rect_w = max(1, min(lp_w, math.ceil(lp_area / lp_rect_h)))
            top = int(self._randint(0, lp_h - lp_rect_h + 1, (1,)).item())
            left = int(self._randint(0, lp_w - lp_rect_w + 1, (1,)).item())
            self.lp_block_mask[env_id, top : top + lp_rect_h, left : left + lp_rect_w] = True

    def snapshot(self) -> FaultSnapshot:
        return FaultSnapshot(
            tensors={name: getattr(self, name).clone() for name in self._SNAPSHOT_TENSORS},
            cursor=self.cursor,
            generator_state=self.generator.get_state().clone(),
        )

    def restore(self, snapshot: FaultSnapshot) -> None:
        missing = set(self._SNAPSHOT_TENSORS) - set(snapshot.tensors)
        if missing:
            raise ValueError(f"fault snapshot is missing fields: {sorted(missing)}")
        for name in self._SNAPSHOT_TENSORS:
            target = getattr(self, name)
            source = snapshot.tensors[name]
            if target.shape != source.shape:
                raise ValueError(f"fault snapshot shape mismatch for {name}")
            target.copy_(source)
        self.cursor = int(snapshot.cursor)
        self.generator.set_state(snapshot.generator_state)

    def unobservable_onset_within(self, horizon_frames: int) -> torch.Tensor:
        """Flag candidates with a not-yet-observed total collapse in the label window."""

        if horizon_frames <= 0:
            raise ValueError("horizon_frames must be positive")
        future_start = (self.start_frame >= self.step_count) & (
            self.start_frame < self.step_count + int(horizon_frames)
        )
        collapse = (self.lp_kind == int(FaultKind.LP_COLLAPSE)) | (
            self.vp_kind == int(FaultKind.VP_COLLAPSE)
        )
        return future_start & collapse

    def audit_metadata(self) -> torch.Tensor:
        """Return generation-only metadata; these columns never enter the selector.

        Columns are LP kind, VP kind, severity, phase (-1 before/0 active/1
        recovered), remaining frames, duration frames, and LP/VP active flags.
        """

        observed_step = (self.step_count - 1).clamp_min(0)
        active = (observed_step >= self.start_frame) & (
            observed_step < self.start_frame + self.duration_frames
        )
        phase = torch.where(
            observed_step < self.start_frame,
            torch.full_like(observed_step, -1),
            torch.where(active, torch.zeros_like(observed_step), torch.ones_like(observed_step)),
        )
        remaining = (self.start_frame + self.duration_frames - self.step_count).clamp_min(0)
        lp_active = active & (self.lp_kind != int(FaultKind.NONE))
        vp_active = active & (self.vp_kind != int(FaultKind.NONE))
        return torch.stack(
            (
                self.lp_kind,
                self.vp_kind,
                self.severity,
                phase,
                remaining,
                self.duration_frames,
                lp_active.long(),
                vp_active.long(),
            ),
            dim=-1,
        )

    def _push_buffers(
        self,
        lp: torch.Tensor,
        vp: torch.Tensor,
        lp_timestamp: torch.Tensor,
        vp_timestamp: torch.Tensor,
    ) -> None:
        capacity = self.cfg.max_delay_frames + 1
        self.cursor = (self.cursor + 1) % capacity
        first = ~self.buffer_initialized
        if first.any():
            self.lp_buffer[first] = lp[first, None]
            self.vp_buffer[first] = vp[first, None]
            self.lp_time_buffer[first] = lp_timestamp[first, None]
            self.vp_time_buffer[first] = vp_timestamp[first, None]
            self.buffer_initialized[first] = True
        self.lp_buffer[:, self.cursor] = lp
        self.vp_buffer[:, self.cursor] = vp
        self.lp_time_buffer[:, self.cursor] = lp_timestamp
        self.vp_time_buffer[:, self.cursor] = vp_timestamp

    def _delayed(self, buffer: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
        capacity = self.cfg.max_delay_frames + 1
        slots = torch.remainder(self.cursor - delays, capacity)
        ids = torch.arange(self.num_envs, device=self.device)
        return buffer[ids, slots]

    @staticmethod
    def _integer_shift(data: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, fill: float) -> torch.Tensor:
        output = data.clone()
        for env_id in torch.nonzero((dx != 0) | (dy != 0), as_tuple=False).flatten().tolist():
            shift_x, shift_y = int(dx[env_id]), int(dy[env_id])
            shifted = torch.roll(data[env_id], shifts=(shift_y, shift_x), dims=(-2, -1))
            if shift_y > 0:
                shifted[..., :shift_y, :] = fill
            elif shift_y < 0:
                shifted[..., shift_y:, :] = fill
            if shift_x > 0:
                shifted[..., :, :shift_x] = fill
            elif shift_x < 0:
                shifted[..., :, shift_x:] = fill
            output[env_id] = shifted
        return output

    def _rotate_lp(self, data: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
        angle = torch.deg2rad(yaw_deg)
        cosine, sine = angle.cos(), angle.sin()
        theta = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=data.dtype)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        grid = F.affine_grid(theta, (self.num_envs, 1, *self.lp_shape), align_corners=True)
        valid = (data != self.cfg.lp_invalid_height).to(data.dtype)
        sampled = F.grid_sample(data[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
        sampled_valid = F.grid_sample(valid[:, None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0]
        return torch.where(sampled_valid > 0.5, sampled, torch.full_like(sampled, self.cfg.lp_invalid_height))

    def _transform_lp(
        self,
        data: torch.Tensor,
        translation_m: torch.Tensor,
        yaw_deg: torch.Tensor,
        *,
        resolution_m: float = 0.1,
    ) -> torch.Tensor:
        """Apply a continuous planar transform while preserving invalid cells."""

        angle = torch.deg2rad(yaw_deg)
        cosine, sine = angle.cos(), angle.sin()
        theta = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=data.dtype)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        width_m = max((self.lp_shape[1] - 1) * resolution_m, resolution_m)
        height_m = max((self.lp_shape[0] - 1) * resolution_m, resolution_m)
        theta[:, 0, 2] = -2.0 * translation_m[:, 0] / width_m
        theta[:, 1, 2] = -2.0 * translation_m[:, 1] / height_m
        grid = F.affine_grid(theta, (self.num_envs, 1, *self.lp_shape), align_corners=True)
        valid = (data != self.cfg.lp_invalid_height).to(data.dtype)
        sampled = F.grid_sample(
            data[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[:, 0]
        sampled_valid = F.grid_sample(
            valid[:, None], grid, mode="nearest", padding_mode="zeros", align_corners=True
        )[:, 0]
        return torch.where(
            sampled_valid > 0.5,
            sampled,
            torch.full_like(sampled, self.cfg.lp_invalid_height),
        )

    def apply(
        self,
        lp: torch.Tensor,
        vp: torch.Tensor,
        lp_timestamp: torch.Tensor,
        vp_timestamp: torch.Tensor,
    ) -> FaultBatch:
        """Apply one 10 Hz fault step without mutating the input tensors."""

        if lp.shape != (self.num_envs, *self.lp_shape):
            raise ValueError(f"LP tensor must have shape {(self.num_envs, *self.lp_shape)}")
        if vp.shape != (self.num_envs, *self.vp_shape):
            raise ValueError(f"VP tensor must have shape {(self.num_envs, *self.vp_shape)}")
        lp_timestamp = lp_timestamp.reshape(self.num_envs).to(self.device)
        vp_timestamp = vp_timestamp.reshape(self.num_envs).to(self.device)
        self._push_buffers(lp, vp, lp_timestamp, vp_timestamp)

        active = (self.step_count >= self.start_frame) & (
            self.step_count < self.start_frame + self.duration_frames
        )
        lp_active = active & (self.lp_kind != int(FaultKind.NONE))
        vp_active = active & (self.vp_kind != int(FaultKind.NONE))
        lp_out, vp_out = lp.clone(), vp.clone()
        lp_time, vp_time = lp_timestamp.clone(), vp_timestamp.clone()

        lp_freeze = lp_active & (self.lp_kind == int(FaultKind.FREEZE))
        new_lp_freeze = lp_freeze & ~self.lp_freeze_initialized
        if new_lp_freeze.any():
            self.lp_frozen[new_lp_freeze] = lp[new_lp_freeze]
            self.lp_frozen_time[new_lp_freeze] = lp_timestamp[new_lp_freeze]
            self.lp_freeze_initialized[new_lp_freeze] = True
        lp_out[lp_freeze] = self.lp_frozen[lp_freeze]
        lp_time[lp_freeze] = self.lp_frozen_time[lp_freeze]

        vp_freeze = vp_active & (self.vp_kind == int(FaultKind.FREEZE))
        new_vp_freeze = vp_freeze & ~self.vp_freeze_initialized
        if new_vp_freeze.any():
            self.vp_frozen[new_vp_freeze] = vp[new_vp_freeze]
            self.vp_frozen_time[new_vp_freeze] = vp_timestamp[new_vp_freeze]
            self.vp_freeze_initialized[new_vp_freeze] = True
        vp_out[vp_freeze] = self.vp_frozen[vp_freeze]
        vp_time[vp_freeze] = self.vp_frozen_time[vp_freeze]

        lp_delay_mask = lp_active & (self.lp_kind == int(FaultKind.DELAY))
        if lp_delay_mask.any():
            delayed = self._delayed(self.lp_buffer, self.lp_delay)
            delayed_time = self._delayed(self.lp_time_buffer, self.lp_delay)
            lp_out[lp_delay_mask] = delayed[lp_delay_mask]
            lp_time[lp_delay_mask] = delayed_time[lp_delay_mask]
        vp_delay_mask = vp_active & (self.vp_kind == int(FaultKind.DELAY))
        if vp_delay_mask.any():
            delayed = self._delayed(self.vp_buffer, self.vp_delay)
            delayed_time = self._delayed(self.vp_time_buffer, self.vp_delay)
            vp_out[vp_delay_mask] = delayed[vp_delay_mask]
            vp_time[vp_delay_mask] = delayed_time[vp_delay_mask]

        occluded = vp_active & (self.vp_kind == int(FaultKind.VP_OCCLUSION))
        if occluded.any():
            mask = self.vp_occlusion_mask[occluded, None].expand(-1, self.vp_shape[0], -1, -1)
            vp_out[occluded] = torch.where(mask, torch.ones_like(vp_out[occluded]), vp_out[occluded])
        scaled = vp_active & (self.vp_kind == int(FaultKind.VP_SCALE))
        if scaled.any():
            vp_out[scaled] = (
                vp_out[scaled] * self.vp_scale[scaled, None, None, None]
            ).clamp(0.0, 1.0)
        biased_vp = vp_active & (self.vp_kind == int(FaultKind.VP_BIAS))
        if biased_vp.any():
            vp_out[biased_vp] = (
                vp_out[biased_vp] + self.vp_bias[biased_vp, None, None, None]
            ).clamp(0.0, 1.0)
        shifted = vp_active & (self.vp_kind == int(FaultKind.VP_SHIFT))
        if shifted.any():
            candidate = self._integer_shift(vp_out, self.vp_shift_x, self.vp_shift_y, 1.0)
            vp_out[shifted] = candidate[shifted]
        collapsed = vp_active & (self.vp_kind == int(FaultKind.VP_COLLAPSE))
        if collapsed.any():
            vp_out[collapsed] = self.vp_collapse_value[collapsed, None, None, None]

        random_dropout = lp_active & (self.lp_kind == int(FaultKind.LP_RANDOM_DROPOUT))
        if random_dropout.any():
            dropout_mask = self._rand(self.num_envs, *self.lp_shape) < self.lp_dropout[:, None, None]
            mask = dropout_mask & random_dropout[:, None, None]
            lp_out[mask] = self.cfg.lp_invalid_height
        block = lp_active & (self.lp_kind == int(FaultKind.LP_BLOCK_DROPOUT))
        if block.any():
            mask = self.lp_block_mask & block[:, None, None]
            lp_out[mask] = self.cfg.lp_invalid_height
        biased = lp_active & (self.lp_kind == int(FaultKind.LP_HEIGHT_BIAS))
        if biased.any():
            valid = lp_out[biased] != self.cfg.lp_invalid_height
            candidate = (lp_out[biased] + self.lp_bias[biased, None, None]).clamp(*self.cfg.lp_height_clip)
            lp_out[biased] = torch.where(valid, candidate, lp_out[biased])
        registered = lp_active & (self.lp_kind == int(FaultKind.LP_REGISTRATION))
        if registered.any():
            candidate = self._integer_shift(
                self._rotate_lp(lp_out, self.lp_yaw),
                self.lp_shift_x,
                self.lp_shift_y,
                self.cfg.lp_invalid_height,
            )
            lp_out[registered] = candidate[registered]
        drifting = lp_active & (self.lp_kind == int(FaultKind.LP_DRIFT))
        if drifting.any():
            elapsed_s = (self.step_count.float() - self.start_frame.float()).clamp_min(0) * self.cfg.update_period_s
            yaw = self.lp_drift_yaw * elapsed_s
            candidate = self._transform_lp(
                lp_out,
                self.lp_drift_xy * elapsed_s[:, None],
                yaw,
            )
            lp_out[drifting] = candidate[drifting]
        lp_collapsed = lp_active & (self.lp_kind == int(FaultKind.LP_COLLAPSE))
        lp_out[lp_collapsed] = self.cfg.lp_invalid_height

        self.step_count += 1
        return FaultBatch(
            lp=lp_out,
            vp=vp_out,
            lp_timestamp=lp_time,
            vp_timestamp=vp_time,
            lp_active=lp_active,
            vp_active=vp_active,
        )
