"""Foot support rays clipped to the sole outline of the configured binary STL."""

from __future__ import annotations

import math
from pathlib import Path
import struct
from collections.abc import Callable

import numpy as np
from scipy.spatial import ConvexHull
import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def load_sole_outline(mesh_path: str, slice_tolerance: float = .001) -> np.ndarray:
    """Return the XY convex outline within 1 mm of the visual sole's lowest Z."""
    if not math.isfinite(slice_tolerance) or slice_tolerance <= 0:
        raise ValueError("sole_slice_tolerance must be positive and finite.")
    blob = Path(mesh_path).read_bytes()
    if len(blob) < 84:
        raise ValueError("Foot sole pattern requires a valid binary STL.")
    count = struct.unpack_from('<I', blob, 80)[0]
    if count == 0 or len(blob) != 84 + 50 * count:
        raise ValueError("Foot sole pattern requires a valid binary STL.")
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    vertices = np.frombuffer(blob, dtype, count=count, offset=84)['vertices'].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("Foot sole STL vertices must be finite.")
    bottom = np.unique(vertices[vertices[:, 2] <= vertices[:, 2].min() + slice_tolerance, :2], axis=0)
    hull = ConvexHull(bottom)
    return bottom[hull.vertices].astype(np.float64)


def foot_sole_pattern(cfg: "FootSolePatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a symmetric 1 cm grid in the foot link frame, inset from its sole."""
    if not math.isfinite(cfg.resolution) or cfg.resolution <= 0:
        raise ValueError("Foot sole resolution must be positive and finite.")
    if not math.isfinite(cfg.edge_margin) or cfg.edge_margin < 0:
        raise ValueError("Foot sole edge_margin must be nonnegative and finite.")
    outline = load_sole_outline(cfg.mesh_path, cfg.sole_slice_tolerance)
    hull = ConvexHull(outline)
    low = np.ceil(outline.min(axis=0) / cfg.resolution).astype(int)
    high = np.floor(outline.max(axis=0) / cfg.resolution).astype(int)
    x = np.arange(low[0], high[0] + 1) * cfg.resolution
    y = np.arange(low[1], high[1] + 1) * cfg.resolution
    xx, yy = np.meshgrid(x, y, indexing='xy')
    candidates = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    # Hull equations have unit outward normals: signed distances <= -margin
    # ensure rounded corners/outside points never contribute to support ratios.
    distances = candidates @ hull.equations[:, :2].T + hull.equations[:, 2]
    keep = (distances <= -cfg.edge_margin + 1e-9).all(axis=-1)
    points = candidates[keep]
    if len(points) == 0:
        raise ValueError("Foot sole pattern contains no rays; reduce resolution or edge_margin.")
    starts = torch.zeros(len(points), 3, dtype=torch.float32, device=device)
    starts[:, :2] = torch.as_tensor(points, dtype=torch.float32, device=device)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.
    return starts, directions


@configclass
class FootSolePatternCfg(PatternBaseCfg):
    func: Callable = foot_sole_pattern
    mesh_path: str = ""
    resolution: float = .01
    edge_margin: float = .002
    sole_slice_tolerance: float = .001
