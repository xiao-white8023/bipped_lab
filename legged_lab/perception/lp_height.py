"""Simulator-independent PGTT height sampling, packet noise and snapshot history.

The PGTT grids are policy query layouts rather than physical LiDAR patterns.
Rows follow increasing ``y`` and columns follow increasing ``x``.  A PGTT
policy consumes exactly one (possibly delayed) height snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math

import torch


PGTT_HEIGHT_SOURCE = "pgtt_full_atten"


@dataclass
class PgttNoiseCfg:
    """Residual mapping errors for the AME height grid."""

    enable: bool = True
    start_iteration: int = 0
    force_active: bool = False
    height_std_range: tuple = (0.01, 0.03)
    height_bias_range: tuple = (-0.02, 0.02)
    xy_bias_range: tuple = (-0.02, 0.02)
    yaw_bias_deg_range: tuple = (-1.0, 1.0)
    dropout_range: tuple = (0.0, 0.05)
    quantize: bool = True
    map_resolution: float = 0.1
    max_delay_packets: int = 2
    block_probability: float = 0.2
    block_x_size: tuple = (0.1, 0.3)
    block_y_size: tuple = (0.1, 0.4)
    block_duration_packets: tuple = (1, 3)


@dataclass
class LPHeightCfg:
    """Single-snapshot 11x17 height grid for the AME LP encoder."""

    resolution: float = 0.1
    full_size: tuple = (1.6, 1.0)  # (x, y), metres
    history_frames: int = 1
    update_period: float = 0.1
    invalid_height: float = -1.1
    height_clip: tuple = (-1.0, 1.0)
    noise: PgttNoiseCfg = field(default_factory=PgttNoiseCfg)

    # Runtime geometry.  ``resolve_pgtt_height_cfg`` is the sole writer.
    shape: tuple = (11, 17)
    x_range: tuple = (-0.8, 0.8)
    y_range: tuple = (-0.5, 0.5)


def _grid_count(span: float, resolution: float) -> int:
    intervals = span / resolution
    if not math.isclose(intervals, round(intervals), abs_tol=1e-6):
        raise ValueError("PGTT map extents must be exact multiples of its resolution")
    return round(intervals) + 1


def resolve_pgtt_height_cfg(cfg: LPHeightCfg, source: str) -> LPHeightCfg:
    """Validate and resolve the sole supported AME height-grid contract."""

    if source != PGTT_HEIGHT_SOURCE:
        raise ValueError(f"lp_height_source must be {PGTT_HEIGHT_SOURCE!r}, got {source!r}")

    runtime = copy.deepcopy(cfg)
    if runtime.resolution <= 0:
        raise ValueError("PGTT resolution must be positive")
    if runtime.history_frames != 1:
        raise ValueError("PGTT policies consume exactly one height snapshot")
    if runtime.full_size[0] <= 0 or runtime.full_size[1] <= 0:
        raise ValueError("PGTT full map size must be positive")
    if not runtime.height_clip[0] < runtime.height_clip[1]:
        raise ValueError("PGTT height_clip must be increasing")
    if runtime.height_clip[0] <= runtime.invalid_height <= runtime.height_clip[1]:
        raise ValueError("PGTT invalid_height must lie outside the valid clipping interval")

    x_half, y_half = runtime.full_size[0] / 2, runtime.full_size[1] / 2
    full_shape = (
        _grid_count(runtime.full_size[1], runtime.resolution),
        _grid_count(runtime.full_size[0], runtime.resolution),
    )
    if full_shape != (11, 17):
        raise ValueError(
            f"The current policy contract requires a full 11x17 grid, got {full_shape}"
        )
    runtime.y_range = (-y_half, y_half)
    runtime.x_range = (-x_half, x_half)
    runtime.shape = full_shape
    return runtime


def grid_points(cfg: LPHeightCfg, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return ``[rows_y, cols_x, 2]`` query points in policy flatten order."""

    rows, cols = cfg.shape
    x = torch.linspace(cfg.x_range[0], cfg.x_range[1], cols, device=device)
    y = torch.linspace(cfg.y_range[0], cfg.y_range[1], rows, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def observation_contract(cfg: LPHeightCfg, source: str, *, height_offset: float, height_scale: float) -> dict:
    """Build the strict serializable contract stored in new checkpoints."""

    if source != PGTT_HEIGHT_SOURCE:
        raise ValueError("LP observation contracts require source='pgtt_full_atten'")
    return {
        "version": 3,
        "source": source,
        "layout": "rectangular_full",
        "shape": list(cfg.shape),
        "history_frames": 1,
        "update_period": cfg.update_period,
        "frame": "capture_torso_yaw",
        "order": "single/y_ascending/x_ascending",
        "sample_xy": grid_points(cfg).tolist(),
        "height_offset": height_offset,
        "height_scale": height_scale,
        "height_clip": list(cfg.height_clip),
        "invalid_height": cfg.invalid_height,
        "lp_target": "critic_11x17_full_187_fixed_clip",
    }


class HeightSnapshotHistory:
    """A one-frame policy snapshot plus private packet slots for delay noise."""

    def __init__(self, num_envs: int, cfg: LPHeightCfg, device, max_delay: int = 0):
        if cfg.history_frames != 1 or max_delay < 0:
            raise ValueError("PGTT supports one policy snapshot and nonnegative packet delay")
        self.frames, self.capacity = 1, 1 + max_delay
        self.device = device
        self.height = torch.full((num_envs, self.capacity, *cfg.shape), cfg.invalid_height, device=device)
        self.confidence = torch.zeros_like(self.height)
        self.time = torch.zeros((num_envs, self.capacity), device=device)
        self.pose = torch.zeros((num_envs, self.capacity, 7), device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_scan = torch.full_like(self.count, -1)
        self.invalid_height = cfg.invalid_height

    def reset(self, ids):
        self.height[ids] = self.invalid_height
        self.confidence[ids] = 0
        self.time[ids] = 0
        self.pose[ids] = 0
        self.count[ids] = 0
        self.last_scan[ids] = -1

    def push(self, height, confidence, timestamp, pose, scan_counts):
        ids = ((scan_counts > 0) & (scan_counts != self.last_scan)).nonzero().flatten()
        slots = self.count[ids] % self.capacity
        self.height[ids, slots] = height[ids]
        self.confidence[ids, slots] = confidence[ids]
        self.time[ids, slots] = timestamp[ids]
        self.pose[ids, slots] = pose[ids]
        self.count[ids] += 1
        self.last_scan[ids] = scan_counts[ids]
        return ids

    def read(self, now, delay):
        offsets = torch.minimum(delay, (self.count - 1).clamp_min(0))
        slots = (self.count - 1 - offsets).clamp_min(0) % self.capacity
        ids = torch.arange(len(self.count), device=self.device)
        return (
            self.height[ids, slots, None],
            self.confidence[ids, slots, None],
            (now - self.time[ids, slots]).clamp_min(0)[:, None],
        )


class PgttMapNoise:
    """Episode-fixed spatial errors and packet-correlated map residuals.

    ``apply`` must only be called for new RayCaster packets.  Quantized samples
    that map to the same cell share noise and dropout.  The output remains one
    snapshot even though up to three packets are retained by the delay buffer.
    """

    def __init__(self, num_envs: int, cfg: LPHeightCfg, device):
        self.cfg, self.noise, self.device = cfg, cfg.noise, device
        self.nominal = grid_points(cfg, device).reshape(-1, 2)
        n, k = num_envs, len(self.nominal)
        self.xy = self.nominal[None].repeat(n, 1, 1)
        self.query_xy = self.xy.clone()
        self.representative = torch.arange(k, device=device)[None].repeat(n, 1)
        self.std = torch.zeros(n, device=device)
        self.bias = torch.zeros(n, device=device)
        self.dropout = torch.zeros(n, device=device)
        self.delay = torch.zeros(n, dtype=torch.long, device=device)
        self.block_remaining = torch.zeros_like(self.delay)
        self.block_center = torch.zeros(n, 2, device=device)
        self.block_size = torch.zeros(n, 2, device=device)
        self.last_valid = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.last_height = torch.full((n, k), cfg.invalid_height, device=device)
        self.last_scan = torch.full_like(self.delay, -1)
        self._validate()
        self.reset(torch.arange(n, device=device))

    def _validate(self):
        cfg = self.noise
        for name in (
            "height_std_range",
            "height_bias_range",
            "xy_bias_range",
            "yaw_bias_deg_range",
            "dropout_range",
            "block_x_size",
            "block_y_size",
        ):
            bounds = getattr(cfg, name)
            if len(bounds) != 2 or bounds[0] > bounds[1] or not all(math.isfinite(v) for v in bounds):
                raise ValueError(f"Invalid PGTT noise range: {name}")
        if not 0 <= cfg.dropout_range[0] <= cfg.dropout_range[1] <= 1 or cfg.height_std_range[0] < 0:
            raise ValueError("Invalid PGTT dropout or height standard deviation")
        if not 0 <= cfg.block_probability <= 1:
            raise ValueError("PGTT block_probability must be in [0, 1]")
        if cfg.map_resolution <= 0 or cfg.max_delay_packets < 0 or int(cfg.max_delay_packets) != cfg.max_delay_packets:
            raise ValueError("Invalid PGTT map resolution or packet delay")
        if cfg.start_iteration < 0 or int(cfg.start_iteration) != cfg.start_iteration:
            raise ValueError("PGTT noise start_iteration must be a nonnegative integer")
        duration = cfg.block_duration_packets
        if not 1 <= duration[0] <= duration[1] or any(int(value) != value for value in duration):
            raise ValueError("PGTT block durations must be positive integer packet counts")

    def _uniform(self, shape, bounds, ids=None, tag="pgtt_parameters"):
        rng = getattr(self, "selector_randomness", None)
        if rng is None:
            return torch.empty(shape, device=self.device).uniform_(*bounds)
        return bounds[0] + (bounds[1] - bounds[0]) * rng.uniform(ids, tuple(shape[1:]), tag)

    def _randint(self, low, high, ids, tag):
        rng = getattr(self, "selector_randomness", None)
        return torch.randint(low, high, (len(ids),), device=self.device) if rng is None else rng.randint(low, high, ids, tag)

    def reset(self, ids):
        cfg, count = self.noise, len(ids)
        self.last_scan[ids] = -1
        self.last_valid[ids] = False
        self.last_height[ids] = self.cfg.invalid_height
        self.block_remaining[ids] = 0
        self.xy[ids] = self.nominal
        self.query_xy[ids] = self.nominal
        self.representative[ids] = torch.arange(len(self.nominal), device=self.device)
        self.delay[ids] = 0
        self.std[ids] = 0
        self.bias[ids] = 0
        self.dropout[ids] = 0
        if not cfg.enable:
            return

        self.std[ids] = self._uniform((count,), cfg.height_std_range, ids)
        self.bias[ids] = self._uniform((count,), cfg.height_bias_range, ids)
        self.dropout[ids] = self._uniform((count,), cfg.dropout_range, ids)
        self.delay[ids] = self._randint(0, cfg.max_delay_packets + 1, ids, "pgtt_delay")
        xy = self.nominal[None].expand(count, -1, -1).clone()
        if cfg.quantize:
            phase = self._uniform((count, 1, 2), (-cfg.map_resolution / 2, cfg.map_resolution / 2), ids)
            cells = torch.round((xy - phase) / cfg.map_resolution).long()
            xy = cells * cfg.map_resolution + phase
            env_column = torch.arange(count, device=self.device)[:, None, None].expand(count, len(self.nominal), 1)
            keys = torch.cat((env_column, cells), dim=-1).reshape(-1, 3)
            _, inverse = torch.unique(keys, dim=0, return_inverse=True)
            index = torch.arange(len(keys), device=self.device)
            first = torch.full_like(index, len(keys))
            first.scatter_reduce_(0, inverse, index, reduce="amin", include_self=True)
            self.representative[ids] = first[inverse].reshape(count, -1) % len(self.nominal)
        self.query_xy[ids] = xy

        yaw = torch.deg2rad(self._uniform((count, 1), cfg.yaw_bias_deg_range, ids))
        x, y = xy.unbind(-1)
        self.xy[ids] = torch.stack((x * yaw.cos() - y * yaw.sin(), x * yaw.sin() + y * yaw.cos()), dim=-1)
        self.xy[ids] += self._uniform((count, 1, 2), cfg.xy_bias_range, ids)

    def apply(
        self,
        world_height,
        valid,
        torso_z,
        scan_counts,
        *,
        height_offset,
        height_scale,
        enable_noise: bool | None = None,
    ):
        ids = ((scan_counts > 0) & (scan_counts != self.last_scan)).nonzero().flatten()
        count, rays = len(ids), len(self.nominal)
        cfg = self.noise
        z = world_height[ids].reshape(count, rays).clone()
        valid = valid[ids].reshape(count, rays).clone()
        noise_active = cfg.enable if enable_noise is None else cfg.enable and enable_noise
        if noise_active and count:
            representative = self.representative[ids]
            rng = getattr(self, "selector_randomness", None)
            gaussian = torch.randn_like(z) if rng is None else rng.normal(ids, (rays,), "pgtt_gaussian")
            uniform = torch.rand_like(z) if rng is None else rng.uniform(ids, (rays,), "pgtt_dropout")
            z += gaussian.gather(1, representative) * self.std[ids, None] + self.bias[ids, None]
            valid &= uniform.gather(1, representative) >= self.dropout[ids, None]

            new_block = (self.block_remaining[ids] == 0) & (
                (torch.rand(count, device=self.device) if rng is None else rng.uniform(ids, (), "pgtt_block")) < cfg.block_probability
            )
            block_ids = ids[new_block]
            if len(block_ids):
                centers = self._randint(0, rays, block_ids, "pgtt_block_center")
                self.block_center[block_ids] = self.nominal[centers]
                self.block_size[block_ids, 0] = self._uniform((len(block_ids),), cfg.block_x_size, block_ids)
                self.block_size[block_ids, 1] = self._uniform((len(block_ids),), cfg.block_y_size, block_ids)
                self.block_remaining[block_ids] = self._randint(cfg.block_duration_packets[0], cfg.block_duration_packets[1] + 1,
                                                   block_ids, "pgtt_block_duration")
            blocked = (
                (self.query_xy[ids] - self.block_center[ids, None]).abs()
                <= self.block_size[ids, None] / 2
            ).all(dim=-1)
            valid &= ~(blocked & (self.block_remaining[ids, None] > 0))
            self.block_remaining[ids] = (self.block_remaining[ids] - 1).clamp_min(0)

        height = ((torso_z[ids, None] - z - height_offset) * height_scale).clamp(*self.cfg.height_clip)
        self.last_height[ids] = torch.where(valid, height, self.cfg.invalid_height)
        self.last_valid[ids] = valid
        self.last_scan[ids] = scan_counts[ids]
        return (
            self.last_height.view(-1, *self.cfg.shape),
            self.last_valid.view(-1, *self.cfg.shape).float(),
        )
