"""Time-ordered Livox MID-360 ray pattern support.

The bundled angle sequence is derived from Livox's MIT-licensed Gazebo
simulation data.  It contains the azimuth and zenith angle of every source
sample.  A frame advances through the source sequence at the physical point
rate, while ``downsample`` controls how many rays are actually cast.

The current ray caster treats each 0.1-second packet as a snapshot at one body
pose.  Per-ray time offsets are exposed by ``GroupedRayCaster.ray_time_offsets``
so a later mapping stage can add motion distortion or deskewing explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


_DEFAULT_SEQUENCE_FILE = Path(__file__).resolve().parent / "data" / "mid360_angles.npy"
_DIRECTION_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def source_points_per_scan(cfg: "Mid360PatternCfg") -> int:
    """Return the number of physical source samples advanced per scan."""

    if cfg.point_rate_hz <= 0:
        raise ValueError(f"point_rate_hz must be positive, got {cfg.point_rate_hz}.")
    if cfg.scan_period <= 0.0:
        raise ValueError(f"scan_period must be positive, got {cfg.scan_period}.")
    if cfg.downsample < 1:
        raise ValueError(f"downsample must be at least one, got {cfg.downsample}.")

    exact_count = cfg.point_rate_hz * cfg.scan_period
    rounded_count = round(exact_count)
    if not math.isclose(exact_count, rounded_count, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            "point_rate_hz * scan_period must be an integer number of source samples, "
            f"got {exact_count}."
        )
    return int(rounded_count)


def sampled_point_offsets(cfg: "Mid360PatternCfg", device: str | torch.device) -> torch.Tensor:
    """Return source-sequence offsets retained after temporal downsampling."""

    return torch.arange(0, source_points_per_scan(cfg), cfg.downsample, dtype=torch.long, device=device)


def load_mid360_sequence_directions(
    cfg: "Mid360PatternCfg", device: str | torch.device
) -> torch.Tensor:
    """Load and cache the complete MID-360 source sequence as unit vectors."""

    sequence_path = Path(cfg.sequence_file).expanduser().resolve()
    if not sequence_path.is_file():
        raise FileNotFoundError(
            f"MID-360 angle sequence not found: {sequence_path}. "
            "The bundled file should be located under grouped_ray_caster/data/."
        )

    device_string = str(torch.device(device))
    cache_key = (str(sequence_path), device_string)
    if cache_key in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[cache_key]

    quantized_angles = np.load(sequence_path, allow_pickle=False)
    if quantized_angles.ndim != 2 or quantized_angles.shape[1] != 2:
        raise ValueError(
            "MID-360 angle sequence must have shape (num_samples, 2) for "
            f"(azimuth, zenith), got {quantized_angles.shape}."
        )
    if quantized_angles.dtype != np.uint16:
        raise ValueError(
            "MID-360 angle sequence must use uint16 hundredths of a degree, "
            f"got {quantized_angles.dtype}."
        )
    if quantized_angles.shape[0] < source_points_per_scan(cfg):
        raise ValueError(
            "MID-360 angle sequence is shorter than one scan: "
            f"{quantized_angles.shape[0]} < {source_points_per_scan(cfg)}."
        )

    # The resource stores unsigned hundredths of a degree.  Copying removes
    # the read-only mmap flag before transferring it to Torch.
    angles = torch.from_numpy(np.array(quantized_angles, dtype=np.uint16, copy=True)).to(
        device=device, dtype=torch.float32
    )
    angles = torch.deg2rad(angles / 100.0)
    azimuth = angles[:, 0]
    zenith = angles[:, 1]

    sin_zenith = torch.sin(zenith)
    directions = torch.stack(
        (
            sin_zenith * torch.cos(azimuth),
            sin_zenith * torch.sin(azimuth),
            torch.cos(zenith),
        ),
        dim=-1,
    ).contiguous()
    _DIRECTION_CACHE[cache_key] = directions
    return directions


def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the first temporally ordered, downsampled MID-360 scan.

    IsaacLab calls a pattern function once during sensor initialization.  The
    :class:`GroupedRayCaster` recognizes :class:`Mid360PatternCfg` and advances
    this initial frame before every subsequent ray cast.
    """

    sequence = load_mid360_sequence_directions(cfg, device)
    offsets = sampled_point_offsets(cfg, device)
    indices = (offsets + cfg.phase_offset) % sequence.shape[0]
    ray_directions = sequence[indices]
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Configuration for a time-ordered Livox MID-360 scan packet.

    ``downsample`` reduces ray-casting cost without changing scan time.  For
    example, the defaults advance 20,000 physical points every 0.1 seconds but
    cast 4,000 rays selected at their original temporal positions.
    """

    func: Callable = mid360_pattern

    sequence_file: str = str(_DEFAULT_SEQUENCE_FILE)
    """Path to a ``uint16[:, 2]`` NumPy array of azimuth/zenith in 0.01 degree."""

    point_rate_hz: int = 200_000
    """Physical source point rate. MID-360 first-return output is 200 kpoints/s."""

    scan_period: float = 0.1
    """Duration represented by one returned scan packet in seconds."""

    downsample: int = 5
    """Temporal stride through the source points. Five produces 4,000 rays per scan."""

    phase_offset: int = 0
    """Deterministic initial source-sequence offset, before optional randomization."""

    randomize_phase: bool = True
    """Randomize each environment's source phase at initialization and reset."""
