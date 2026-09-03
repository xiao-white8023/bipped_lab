# Copyright (c) 2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tri-mesh gap terrain with a finite, ray-castable bottom surface."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshGapTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def gap_with_floor_terrain(
    difficulty: float,
    cfg: MeshGapWithFloorTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the standard square-ring gap with a bottom at finite depth.

    The exposed top surfaces remain at ``z=0``.  Unlike IsaacLab's standard
    ``MeshGapTerrainCfg``, the annular gap contains a horizontal collision
    surface at ``z=-gap_depth`` so downward LiDAR rays obtain a physical hit.
    """

    gap_width_low, gap_width_high = cfg.gap_width_range
    gap_depth_low, gap_depth_high = cfg.gap_depth_range
    if gap_width_low < 0.0 or gap_width_high < gap_width_low:
        raise ValueError(f"gap_width_range must be non-negative and increasing, got {cfg.gap_width_range}.")
    if gap_depth_low <= 0.0 or gap_depth_high < gap_depth_low:
        raise ValueError(f"gap_depth_range must be positive and increasing, got {cfg.gap_depth_range}.")

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    gap_width = gap_width_low + difficulty * (gap_width_high - gap_width_low)
    gap_depth = gap_depth_low + difficulty * (gap_depth_high - gap_depth_low)
    inner_size = (cfg.platform_width + 2.0 * gap_width,) * 2
    if inner_size[0] >= min(cfg.size):
        raise ValueError(
            "platform_width + 2 * gap_width must fit inside the terrain: "
            f"inner_size={inner_size}, terrain_size={cfg.size}."
        )

    terrain_center_xy = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    wall_height = max(1.0, gap_depth)
    top_center = (*terrain_center_xy, -0.5 * wall_height)

    meshes: list[trimesh.Trimesh] = []
    # Original outer ground ring and central spawn platform, both with z=0 tops.
    meshes += make_border(cfg.size, inner_size, wall_height, top_center)
    platform_dim = (cfg.platform_width, cfg.platform_width, wall_height)
    meshes.append(
        trimesh.creation.box(
            platform_dim,
            trimesh.transformations.translation_matrix(top_center),
        )
    )

    # Annular floor exposed only inside the gap.  A one-metre-thick box keeps
    # the mesh closed while its upper surface is exactly at -gap_depth.
    floor_thickness = 1.0
    floor_center = (*terrain_center_xy, -gap_depth - 0.5 * floor_thickness)
    meshes += make_border(
        inner_size,
        (cfg.platform_width, cfg.platform_width),
        floor_thickness,
        floor_center,
    )

    origin = np.array([terrain_center_xy[0], terrain_center_xy[1], 0.0])
    return meshes, origin


@configclass
class MeshGapWithFloorTerrainCfg(MeshGapTerrainCfg):
    """Configuration for a square-ring gap with a finite bottom."""

    function = gap_with_floor_terrain

    gap_depth_range: tuple[float, float] = MISSING
    """Minimum and maximum vertical depth of the gap bottom below z=0, in metres."""
