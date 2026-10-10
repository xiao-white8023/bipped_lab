"""Row-local recurring faults, with data/quality/timestamps kept atomic."""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
import torch
import torch.nn.functional as F
from .config import FaultConfig
from .randomness import PerEnvironmentRNG


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
    cursor: torch.Tensor
    generator_state: torch.Tensor


@dataclass
class FaultBatch:
    lp: torch.Tensor
    vp: torch.Tensor
    lp_timestamp: torch.Tensor
    vp_timestamp: torch.Tensor
    lp_active: torch.Tensor
    vp_active: torch.Tensor
    lp_valid: torch.Tensor
    vp_valid: torch.Tensor
    lp_packet_timestamp: torch.Tensor
    vp_packet_timestamp: torch.Tensor


class SelectorFaultManager:
    def __init__(
        self,
        num_envs,
        *,
        lp_shape=(11, 17),
        vp_shape=(2, 18, 32),
        device="cpu",
        seed=42,
        cfg=None,
    ):
        self.cfg = cfg or FaultConfig()
        self.cfg.validate()
        self.num_envs, self.lp_shape, self.vp_shape = (
            int(num_envs),
            tuple(lp_shape),
            tuple(vp_shape),
        )
        self.device = torch.device(device)
        self.generator = PerEnvironmentRNG(num_envs, seed, self.device)
        n, capacity = num_envs, self.cfg.max_delay_frames + 1
        integer = (
            "step_count",
            "lp_kind",
            "vp_kind",
            "severity",
            "start_frame",
            "duration_frames",
            "lp_start_frame",
            "vp_start_frame",
            "lp_duration_frames",
            "vp_duration_frames",
            "next_schedule_frame",
            "lp_delay",
            "vp_delay",
            "vp_shift_x",
            "vp_shift_y",
            "schedule_count",
        )
        scalar = (
            "vp_scale",
            "vp_bias",
            "vp_collapse_value",
            "lp_dropout",
            "lp_bias",
            "lp_yaw",
            "lp_drift_yaw",
        )
        for name in integer:
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=self.device))
        for name in scalar:
            setattr(self, name, torch.zeros(n, device=self.device))
        self.lp_drift_xy = torch.zeros(n, 2, device=self.device)
        self.lp_translation_xy = torch.zeros(n, 2, device=self.device)
        self.vp_occlusion_mask = torch.zeros(
            n, *vp_shape[-2:], device=self.device, dtype=torch.bool
        )
        self.lp_block_mask = torch.zeros(
            n, *lp_shape, device=self.device, dtype=torch.bool
        )
        self.lp_buffer = torch.full(
            (n, capacity, *lp_shape), self.cfg.lp_invalid_height, device=self.device
        )
        self.vp_buffer = torch.ones(n, capacity, *vp_shape, device=self.device)
        self.lp_valid_buffer = torch.zeros_like(self.lp_buffer)
        self.vp_valid_buffer = torch.zeros_like(self.vp_buffer)
        self.lp_time_buffer = torch.zeros(
            n, capacity, device=self.device, dtype=torch.float64
        )
        self.vp_time_buffer = torch.zeros_like(self.lp_time_buffer)
        self.lp_packet_buffer = torch.zeros_like(self.lp_time_buffer)
        self.vp_packet_buffer = torch.zeros_like(self.lp_time_buffer)
        self.buffer_initialized = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.lp_frozen = self.lp_buffer[:, 0].clone()
        self.vp_frozen = self.vp_buffer[:, 0].clone()
        self.lp_frozen_valid = torch.zeros_like(self.lp_frozen)
        self.vp_frozen_valid = torch.zeros_like(self.vp_frozen)
        self.lp_frozen_time = torch.zeros(n, device=self.device, dtype=torch.float64)
        self.vp_frozen_time = torch.zeros_like(self.lp_frozen_time)
        self.lp_frozen_packet = torch.zeros_like(self.lp_frozen_time)
        self.vp_frozen_packet = torch.zeros_like(self.lp_frozen_time)
        self.lp_freeze_initialized = torch.zeros_like(self.buffer_initialized)
        self.vp_freeze_initialized = torch.zeros_like(self.buffer_initialized)
        self.cursor = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self._SNAPSHOT_TENSORS = tuple(
            name
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor) and name != "cursor"
        )
        self.reset()

    def _severity_strength(self, ids, shape=()):
        level = self.severity[ids]
        lo = level.float().reshape(-1, *([1] * len(shape))) * 0.35
        level = level.reshape(-1, *([1] * len(shape)))
        hi = torch.where(level == 2, 1.0, lo + 0.35)
        return lo + (hi - lo) * self.generator.uniform(ids, shape)

    def _signed_range(self, ids, bounds, *, neutral=0.0, shape=()):
        """Sample within exact bounds; severity measures distance from neutral.

        The nearest allowed value is the anchor for one-sided bounds. Equal
        bounds, including zero ranges, remain exact instead of adding a
        hard-coded minimum fault.
        """
        low, high = bounds
        anchor = min(max(neutral, low), high)
        choose_low = self.generator.uniform(ids, shape) < 0.5
        if anchor == low:
            choose_low = torch.zeros_like(choose_low)
        elif anchor == high:
            choose_low = torch.ones_like(choose_low)
        endpoint = torch.where(choose_low, low, high)
        return anchor + self._severity_strength(ids, shape) * (endpoint - anchor)

    def _schedule(self, ids, initial=False):
        if len(ids) == 0:
            return
        scenario = self.generator.categorical(self.cfg.scenario_probabilities, ids)
        self.severity[ids] = self.generator.categorical(
            self.cfg.severity_probabilities, ids
        )
        self.lp_kind[ids] = 0
        self.vp_kind[ids] = 0
        for flags, kinds, target in (
            ((scenario == 2) | (scenario == 3), LP_FAULTS, self.lp_kind),
            ((scenario == 1) | (scenario == 3), VP_FAULTS, self.vp_kind),
        ):
            chosen = ids[flags]
            options = torch.tensor([int(k) for k in kinds], device=self.device)
            target[chosen] = options[self.generator.randint(0, len(kinds), chosen)]
        level = self.severity[ids]
        lo = level.float() * 0.35
        hi = torch.where(level == 2, 1.0, lo + 0.35)
        strength = lo + (hi - lo) * self.generator.uniform(ids)
        delay = (
            self.generator.randint(0, self.cfg.max_start_delay_frames + 1, ids)
            if initial
            else torch.zeros_like(ids)
        )
        self.start_frame[ids] = self.step_count[ids] + delay
        span = self.cfg.max_duration_frames - self.cfg.min_duration_frames
        self.duration_frames[ids] = (
            self.cfg.min_duration_frames + torch.round(strength * span).long()
        )
        self.lp_start_frame[ids] = self.start_frame[ids]
        self.vp_start_frame[ids] = self.start_frame[ids]
        self.lp_duration_frames[ids] = self.duration_frames[ids]
        self.vp_duration_frames[ids] = self.duration_frames[ids]
        dual = ids[scenario == 3]
        stagger = self.generator.randint(0, self.cfg.max_dual_stagger_frames + 1, dual)
        lp_first = self.generator.uniform(dual) < 0.5
        self.lp_start_frame[dual] += torch.where(lp_first, 0, stagger)
        self.vp_start_frame[dual] += torch.where(lp_first, stagger, 0)
        self.vp_duration_frames[dual] = self.generator.randint(
            self.cfg.min_duration_frames, self.cfg.max_duration_frames + 1, dual
        )
        self.duration_frames[ids] = (
            torch.maximum(
                self.lp_start_frame[ids] + self.lp_duration_frames[ids],
                self.vp_start_frame[ids] + self.vp_duration_frames[ids],
            )
            - self.start_frame[ids]
        )
        recovery = self.generator.randint(
            self.cfg.min_recovery_frames, self.cfg.max_recovery_frames + 1, ids
        )
        self.next_schedule_frame[ids] = (
            self.start_frame[ids] + self.duration_frames[ids] + recovery
        )
        self.lp_delay[ids] = (
            1 + torch.round(strength * (self.cfg.max_delay_frames - 1)).long()
        )
        self.vp_delay[ids] = self.lp_delay[ids]
        self.vp_scale[ids] = self._signed_range(ids, self.cfg.vp_scale_range, neutral=1.0)
        self.vp_bias[ids] = self._signed_range(ids, self.cfg.vp_bias_m_range) / self.cfg.vp_depth_max_m
        shifts = (strength * self.cfg.vp_max_shift_pixels).round().long()
        for name in ("vp_shift_x", "vp_shift_y"):
            getattr(self, name)[ids] = shifts * torch.where(
                self.generator.uniform(ids) < 0.5, -1, 1
            )
        self.vp_collapse_value[ids] = torch.tensor(
            [0.0, 1.0, -1.0], device=self.device
        )[self.generator.randint(0, 3, ids)]
        low, high = self.cfg.lp_dropout_fraction
        self.lp_dropout[ids] = low + strength * (high - low)
        self.lp_bias[ids] = self._signed_range(ids, self.cfg.lp_height_bias_m_range)
        self.lp_translation_xy[ids] = self._signed_range(ids, self.cfg.lp_translation_range_m, shape=(2,))
        self.lp_yaw[ids] = self._signed_range(ids, self.cfg.lp_yaw_range_deg)
        angle = 2 * math.pi * self.generator.uniform(ids)
        low, high = self.cfg.lp_drift_m_per_s
        speed = low + strength * (high - low)
        self.lp_drift_xy[ids] = torch.stack(
            (speed * angle.cos(), speed * angle.sin()), -1
        )
        low, high = self.cfg.lp_drift_deg_per_s
        yaw_speed = low + self._severity_strength(ids) * (high - low)
        self.lp_drift_yaw[ids] = yaw_speed * torch.where(self.generator.uniform(ids) < 0.5, -1.0, 1.0)
        self.lp_freeze_initialized[ids] = False
        self.vp_freeze_initialized[ids] = False
        self.schedule_count[ids] += 1
        for name, shape, bounds in (
            ("vp_occlusion_mask", self.vp_shape[-2:], self.cfg.vp_occlusion_fraction),
            ("lp_block_mask", self.lp_shape, self.cfg.lp_block_fraction),
        ):
            h, w = shape
            fraction = bounds[0] + strength * (bounds[1] - bounds[0])
            rh = (fraction * h * w * h / w).sqrt().round().long().clamp(1, h)
            rw = (fraction * h * w / rh).ceil().long().clamp(1, w)
            top = (self.generator.uniform(ids) * (h - rh + 1)).long()
            left = (self.generator.uniform(ids) * (w - rw + 1)).long()
            yy = torch.arange(h, device=self.device)[None, :, None]
            xx = torch.arange(w, device=self.device)[None, None, :]
            getattr(self, name)[ids] = (
                (yy >= top[:, None, None])
                & (yy < (top + rh)[:, None, None])
                & (xx >= left[:, None, None])
                & (xx < (left + rw)[:, None, None])
                & (fraction > 0)[:, None, None]
            )

    def reset(self, env_ids=None):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids.to(self.device).long()
        )
        self.step_count[ids] = 0
        self.schedule_count[ids] = 0
        self.cursor[ids] = -1
        self.buffer_initialized[ids] = False
        self.lp_valid_buffer[ids] = 0
        self.vp_valid_buffer[ids] = 0
        self._schedule(ids, initial=True)

    def snapshot(self):
        return FaultSnapshot(
            {n: getattr(self, n).clone() for n in self._SNAPSHOT_TENSORS},
            self.cursor.clone(),
            self.generator.get_state(),
        )

    def restore(self, snapshot):
        for name, value in snapshot.tensors.items():
            getattr(self, name).copy_(value)
        self.cursor.copy_(snapshot.cursor)
        self.generator.set_state(snapshot.generator_state)

    def unobservable_onset_within(self, horizon_frames):
        lp_pending = (self.lp_start_frame >= self.step_count) & (
            self.lp_start_frame < self.step_count + horizon_frames
        )
        vp_pending = (self.vp_start_frame >= self.step_count) & (
            self.vp_start_frame < self.step_count + horizon_frames
        )
        # The next schedule is sampled in the future and is absent from the
        # observable feature history. Do not label a candidate across that draw.
        reschedule = (self.next_schedule_frame >= self.step_count) & (
            self.next_schedule_frame < self.step_count + horizon_frames
        )
        return (
            reschedule
            | (lp_pending & (self.lp_kind != 0))
            | (vp_pending & (self.vp_kind != 0))
        )

    def _active(self, observed):
        return (
            (observed >= self.lp_start_frame)
            & (observed < self.lp_start_frame + self.lp_duration_frames)
            & (self.lp_kind != 0),
            (observed >= self.vp_start_frame)
            & (observed < self.vp_start_frame + self.vp_duration_frames)
            & (self.vp_kind != 0),
        )

    def audit_metadata(self):
        observed = (self.step_count - 1).clamp_min(0)
        la, va = self._active(observed)
        phase = torch.where(
            observed < self.start_frame,
            -1,
            torch.where(observed < self.start_frame + self.duration_frames, 0, 1),
        )
        remaining = (
            self.start_frame + self.duration_frames - self.step_count
        ).clamp_min(0)
        return torch.stack(
            (
                self.lp_kind,
                self.vp_kind,
                self.severity,
                phase,
                remaining,
                self.duration_frames,
                la.long(),
                va.long(),
            ),
            -1,
        )

    def audit_parameters(self):
        """Actual sampled plan parameters in FAULT_PARAMETER_ORDER, without RNG draws."""
        dt = self.cfg.update_period_s
        observed = (self.step_count - 1).clamp_min(0)
        return torch.stack(
            (
                self.lp_duration_frames * dt, self.vp_duration_frames * dt,
                self.lp_delay * dt, self.vp_delay * dt,
                (self.lp_start_frame - observed) * dt,
                (self.vp_start_frame - observed) * dt,
                self.vp_scale, self.vp_bias * self.cfg.vp_depth_max_m,
                self.vp_shift_x.float(), self.vp_shift_y.float(),
                self.vp_occlusion_mask.float().mean((-2, -1)),
                self.vp_collapse_value, self.lp_dropout,
                self.lp_block_mask.float().mean((-2, -1)), self.lp_bias,
                self.lp_translation_xy[:, 0], self.lp_translation_xy[:, 1],
                self.lp_yaw, self.lp_drift_xy[:, 0], self.lp_drift_xy[:, 1],
                self.lp_drift_yaw,
            ), -1,
        )

    def _delayed(self, buffer, delays):
        ids = torch.arange(self.num_envs, device=self.device)
        return buffer[ids, (self.cursor - delays) % (self.cfg.max_delay_frames + 1)]

    @staticmethod
    def _integer_shift(
        data: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, fill: float
    ) -> torch.Tensor:
        n, h, w = data.shape[0], data.shape[-2], data.shape[-1]
        y = torch.arange(h, device=data.device)[None, :, None] - dy[:, None, None]
        x = torch.arange(w, device=data.device)[None, None, :] - dx[:, None, None]
        inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        indices = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).reshape(n, 1, h * w)
        shifted = data.reshape(n, -1, h * w).gather(
            2, indices.expand(n, data.reshape(n, -1, h * w).shape[1], h * w)
        )
        return torch.where(inside.reshape(n, 1, h * w), shifted, fill).reshape_as(data)

    def _rotate_lp(self, data: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
        return self._transform_lp(data, torch.zeros(data.shape[0], 2, device=data.device), yaw_deg)

    def _transform_lp(
        self,
        data: torch.Tensor,
        translation_m: torch.Tensor,
        yaw_deg: torch.Tensor,
        *,
        resolution_m: float | None = None,
        valid_support: torch.Tensor | None = None,
        return_support: bool = False,
    ):
        """Sample at R(yaw) * xy - translation, in physical metres.

        Height and raw support use the same bilinear grid. Unknown heights
        contribute zero weight; -1.1 is never part of a height average. A
        rectangular map needs physical aspect factors in normalized coordinates.
        Supports a subset of rows so unrelated environments need no warp.
        """
        n = data.shape[0]
        resolution_m = self.cfg.lp_resolution_m if resolution_m is None else resolution_m
        if not math.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("LP transform resolution must be finite and positive")
        if data.shape != (n, *self.lp_shape) or translation_m.shape != (n, 2) or yaw_deg.shape != (n,):
            raise ValueError("LP transform data, translation and yaw shapes must match")
        if valid_support is not None and valid_support.shape != data.shape:
            raise ValueError("LP transform support must match height")

        angle = torch.deg2rad(yaw_deg)
        cosine, sine = angle.cos(), angle.sin()
        theta = torch.zeros(n, 2, 3, device=data.device, dtype=data.dtype)
        width_m = max((self.lp_shape[1] - 1) * resolution_m, resolution_m)
        height_m = max((self.lp_shape[0] - 1) * resolution_m, resolution_m)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine * height_m / width_m
        theta[:, 1, 0] = sine * width_m / height_m
        theta[:, 1, 1] = cosine
        theta[:, 0, 2] = -2.0 * translation_m[:, 0] / width_m
        theta[:, 1, 2] = -2.0 * translation_m[:, 1] / height_m
        grid = F.affine_grid(
            theta, (n, 2, *self.lp_shape), align_corners=True
        )
        finite = torch.isfinite(data) & (data != self.cfg.lp_invalid_height)
        weights = finite.to(data.dtype)
        if valid_support is not None:
            weights = weights * valid_support.to(data.dtype)
        weighted_height = torch.where(finite, data, 0.0) * weights
        sampled = F.grid_sample(
            torch.stack((weighted_height, weights), 1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        support = sampled[:, 1].clamp(0, 1)
        has_support = support > 1.0e-6
        height = torch.where(
            has_support,
            sampled[:, 0] / support.clamp_min(1.0e-6),
            torch.full_like(support, self.cfg.lp_invalid_height),
        )
        support = torch.where(has_support, support, 0.0)
        return (height, support) if return_support else height

    def apply(
        self,
        lp,
        vp,
        lp_timestamp,
        vp_timestamp,
        *,
        lp_valid=None,
        vp_valid=None,
        lp_packet_timestamp=None,
        vp_packet_timestamp=None,
        env_ids=None,
    ):
        for data, shape in ((lp, self.lp_shape), (vp, self.vp_shape)):
            if data.shape != (self.num_envs, *shape):
                raise ValueError(
                    "fault input shape differs from configured sensor layout"
                )
        for data, support in ((lp, lp_valid), (vp, vp_valid)):
            if support is not None and (
                support.shape != data.shape
                or not torch.isfinite(support).all()
                or ((support < 0) | (support > 1)).any()
            ):
                raise ValueError(
                    "raw valid support must match data and be finite in [0,1]"
                )
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids.to(self.device).long()
        )
        lp_valid = (
            (torch.isfinite(lp) & (lp != self.cfg.lp_invalid_height)).float()
            if lp_valid is None
            else lp_valid.float()
        )
        vp_valid = (
            (torch.isfinite(vp) & (vp >= 0) & (vp <= 1)).float()
            if vp_valid is None
            else vp_valid.float()
        )
        lp_time = (
            lp_timestamp.to(self.device, torch.float64).reshape(self.num_envs).clone()
        )
        vp_time = (
            vp_timestamp.to(self.device, torch.float64).reshape(self.num_envs).clone()
        )
        lp_packet = (
            lp_time.clone()
            if lp_packet_timestamp is None
            else lp_packet_timestamp.to(self.device, torch.float64)
            .reshape(self.num_envs)
            .clone()
        )
        vp_packet = (
            vp_time.clone()
            if vp_packet_timestamp is None
            else vp_packet_timestamp.to(self.device, torch.float64)
            .reshape(self.num_envs)
            .clone()
        )
        due = ids[self.step_count[ids] >= self.next_schedule_frame[ids]]
        self._schedule(due)
        capacity = self.cfg.max_delay_frames + 1
        self.cursor[ids] = (self.cursor[ids] + 1) % capacity
        first = ids[~self.buffer_initialized[ids]]
        for buffer, value in (
            (self.lp_buffer, lp),
            (self.vp_buffer, vp),
            (self.lp_valid_buffer, lp_valid),
            (self.vp_valid_buffer, vp_valid),
            (self.lp_time_buffer, lp_time),
            (self.vp_time_buffer, vp_time),
            (self.lp_packet_buffer, lp_packet),
            (self.vp_packet_buffer, vp_packet),
        ):
            buffer[first] = value[first, None]
            buffer[ids, self.cursor[ids]] = value[ids]
        self.buffer_initialized[ids] = True
        selected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        selected[ids] = True
        la, va = self._active(self.step_count)
        la &= selected
        va &= selected
        out_lp, out_vp = lp.clone(), vp.clone()
        lv, vv = lp_valid.clone(), vp_valid.clone()
        for name, kinds, on, data, support, time, packet in (
            ("lp", self.lp_kind, la, out_lp, lv, lp_time, lp_packet),
            ("vp", self.vp_kind, va, out_vp, vv, vp_time, vp_packet),
        ):
            freeze = on & (kinds == int(FaultKind.FREEZE))
            initialized = getattr(self, name + "_freeze_initialized")
            new = freeze & ~initialized
            getattr(self, name + "_frozen")[new] = data[new]
            getattr(self, name + "_frozen_valid")[new] = support[new]
            getattr(self, name + "_frozen_time")[new] = time[new]
            initialized[new] = True
            getattr(self, name + "_frozen_packet")[new] = packet[new]
            data[freeze] = getattr(self, name + "_frozen")[freeze]
            support[freeze] = getattr(self, name + "_frozen_valid")[freeze]
            time[freeze] = getattr(self, name + "_frozen_time")[freeze]
            packet[freeze] = getattr(self, name + "_frozen_packet")[freeze]
            delayed = on & (kinds == int(FaultKind.DELAY))
            delays = getattr(self, name + "_delay")
            data[delayed] = self._delayed(getattr(self, name + "_buffer"), delays)[
                delayed
            ]
            support[delayed] = self._delayed(
                getattr(self, name + "_valid_buffer"), delays
            )[delayed]
            time[delayed] = self._delayed(getattr(self, name + "_time_buffer"), delays)[
                delayed
            ]
            packet[delayed] = self._delayed(
                getattr(self, name + "_packet_buffer"), delays
            )[delayed]
        occluded = va & (self.vp_kind == int(FaultKind.VP_OCCLUSION))
        mask = (
            self.vp_occlusion_mask[:, None].expand_as(vv)
            & occluded[:, None, None, None]
        )
        out_vp[mask] = 1.0
        vv[mask] = 0.0
        for kind, scale, bias in (
            (FaultKind.VP_SCALE, self.vp_scale, torch.zeros_like(self.vp_bias)),
            (FaultKind.VP_BIAS, torch.ones_like(self.vp_scale), self.vp_bias),
        ):
            chosen = va & (self.vp_kind == int(kind))
            out_vp[chosen] = (
                out_vp[chosen] * scale[chosen, None, None, None]
                + bias[chosen, None, None, None]
            ).clamp(0, 1)
        shifted = va & (self.vp_kind == int(FaultKind.VP_SHIFT))
        out_vp[shifted] = self._integer_shift(
            out_vp, self.vp_shift_x, self.vp_shift_y, 1.0
        )[shifted]
        vv[shifted] = self._integer_shift(vv, self.vp_shift_x, self.vp_shift_y, 0.0)[
            shifted
        ]
        collapsed = va & (self.vp_kind == int(FaultKind.VP_COLLAPSE))
        out_vp[collapsed] = self.vp_collapse_value[collapsed, None, None, None]
        vv[collapsed] = 0.0
        dropout_ids = (
            (la & (self.lp_kind == int(FaultKind.LP_RANDOM_DROPOUT)))
            .nonzero()
            .flatten()
        )
        if len(dropout_ids):
            drop = (
                self.generator.uniform(dropout_ids, self.lp_shape)
                < self.lp_dropout[dropout_ids, None, None]
            )
            patch = out_lp[dropout_ids]
            patch[drop] = self.cfg.lp_invalid_height
            out_lp[dropout_ids] = patch
            patch = lv[dropout_ids]
            patch[drop] = 0.0
            lv[dropout_ids] = patch
        blocked = la & (self.lp_kind == int(FaultKind.LP_BLOCK_DROPOUT))
        mask = self.lp_block_mask & blocked[:, None, None]
        out_lp[mask] = self.cfg.lp_invalid_height
        lv[mask] = 0.0
        biased = la & (self.lp_kind == int(FaultKind.LP_HEIGHT_BIAS))
        candidate = (out_lp + self.lp_bias[:, None, None]).clamp(
            *self.cfg.lp_height_clip
        )
        out_lp[biased] = torch.where(
            lv[biased] > 0, candidate[biased], self.cfg.lp_invalid_height
        )
        registered = (la & (self.lp_kind == int(FaultKind.LP_REGISTRATION))).nonzero().flatten()
        if len(registered):
            out_lp[registered], lv[registered] = self._transform_lp(
                out_lp[registered], self.lp_translation_xy[registered], self.lp_yaw[registered],
                valid_support=lv[registered], return_support=True,
            )
        drifting = (la & (self.lp_kind == int(FaultKind.LP_DRIFT))).nonzero().flatten()
        if len(drifting):
            elapsed = ((self.step_count[drifting] - self.lp_start_frame[drifting]).clamp_min(0).float()
                       * self.cfg.update_period_s)
            out_lp[drifting], lv[drifting] = self._transform_lp(
                out_lp[drifting], self.lp_drift_xy[drifting] * elapsed[:, None], self.lp_drift_yaw[drifting] * elapsed,
                valid_support=lv[drifting], return_support=True,
            )
        collapsed = la & (self.lp_kind == int(FaultKind.LP_COLLAPSE))
        out_lp[collapsed] = self.cfg.lp_invalid_height
        lv[collapsed] = 0.0
        lv = lv.clamp(0, 1)
        vv = vv.clamp(0, 1)
        self.step_count[ids] += 1
        return FaultBatch(
            out_lp, out_vp, lp_time, vp_time, la, va, lv, vv, lp_packet, vp_packet
        )

