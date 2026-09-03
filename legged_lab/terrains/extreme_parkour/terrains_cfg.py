# Copyright (c) 2023 Xuxin Cheng, Kexin Shi, Deepak Pathak
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Adapted for Isaac Lab from Extreme Parkour, commit
# d2ffe27ba59a3229fad22a9fc94c38010bb1f519. See NOTICE and LICENSE in this
# directory. This file is for non-commercial use under CC BY-NC 4.0.

"""Configuration objects for the Isaac Lab Extreme Parkour terrains."""

from __future__ import annotations

import math

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import terrains as terrain_functions


class ExtremeParkourTerrainCfgMixin:
    """Parameters shared by all migrated Extreme Parkour height fields."""

    platform_length: float = 2.5
    platform_height: float = 0.0
    num_obstacles: int = 6
    start_position_x: float = 1.0
    edge_padding_width: float = 0.1
    edge_padding_height: float = 0.0
    roughness_range: tuple[float, float] = (0.02, 0.06)
    roughness_step: float = 0.005
    roughness_downsampled_scale: float = 0.075


@configclass
class HfParkourTerrainCfg(HfTerrainBaseCfg, ExtremeParkourTerrainCfgMixin):
    """Alternating tilted stepping stones over a pit."""

    function = terrain_functions.parkour_terrain

    pit_depth_range: tuple[float, float] = (0.2, 1.0)
    stone_length_lower_range: tuple[float, float] = (0.9, 0.6)
    stone_length_upper_range: tuple[float, float] = (1.0, 0.8)
    stone_width: float = 1.0
    last_stone_length: float = 1.6
    x_spacing_lower_range: tuple[float, float] = (-0.1, -0.1)
    x_spacing_upper_range: tuple[float, float] = (0.1, 0.4)
    y_offset_lower_range: tuple[float, float] = (0.2, 0.2)
    y_offset_upper_range: tuple[float, float] = (0.3, 0.4)
    incline_height_range: tuple[float, float] = (0.0, 0.25)
    last_incline_height_range: tuple[float, float] = (0.1, 0.25)
    final_platform_clearance: float = 0.05


@configclass
class HfParkourHurdleTerrainCfg(HfTerrainBaseCfg, ExtremeParkourTerrainCfgMixin):
    """Laterally offset hurdles; ``flat=True`` produces the rough-flat baseline."""

    function = terrain_functions.parkour_hurdle_terrain

    hurdle_length_range: tuple[float, float] = (0.1, 0.4)
    hurdle_height_lower_range: tuple[float, float] = (0.1, 0.2)
    hurdle_height_upper_range: tuple[float, float] = (0.15, 0.4)
    x_spacing_range: tuple[float, float] = (1.2, 2.2)
    y_offset_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width_range: tuple[float, float] = (0.4, 0.8)
    flat: bool = False


@configclass
class HfParkourStepTerrainCfg(HfTerrainBaseCfg, ExtremeParkourTerrainCfgMixin):
    """Large ascending and descending step platforms."""

    function = terrain_functions.parkour_step_terrain

    step_height_range: tuple[float, float] = (0.1, 0.45)
    step_run_range: tuple[float, float] = (0.3, 1.5)
    y_offset_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width_range: tuple[float, float] = (0.5, 1.0)


@configclass
class HfParkourGapTerrainCfg(HfTerrainBaseCfg, ExtremeParkourTerrainCfgMixin):
    """Winding valid corridors interrupted by transverse gaps."""

    function = terrain_functions.parkour_gap_terrain

    gap_width_range: tuple[float, float] = (0.1, 0.8)
    gap_depth_range: tuple[float, float] = (0.2, 1.0)
    x_spacing_range: tuple[float, float] = (0.8, 1.5)
    y_offset_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width_range: tuple[float, float] = (0.6, 1.2)


def make_extreme_parkour_terrains_cfg(
    geometry_scale: float = 1.0, seed: int | None = 0
) -> TerrainGeneratorCfg:
    """Build the five-terrain Extreme Parkour curriculum.

    ``geometry_scale`` uniformly scales physical dimensions and height-field
    resolution while keeping the 10-by-20 curriculum topology unchanged.
    """
    geometry_scale = float(geometry_scale)
    if not math.isfinite(geometry_scale) or geometry_scale <= 0.0:
        raise ValueError(f"geometry_scale must be a finite positive number, got {geometry_scale!r}.")

    def scaled(value: float) -> float:
        return value * geometry_scale

    def scaled_pair(values: tuple[float, float]) -> tuple[float, float]:
        return scaled(values[0]), scaled(values[1])

    common = {
        "platform_length": scaled(2.5),
        "platform_height": 0.0,
        "num_obstacles": 6,
        "start_position_x": scaled(1.0),
        "edge_padding_width": scaled(0.1),
        "edge_padding_height": 0.0,
        "roughness_range": scaled_pair((0.02, 0.06)),
        "roughness_step": scaled(0.005),
        "roughness_downsampled_scale": scaled(0.075),
    }

    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=(scaled(18.0), scaled(4.0)),
        border_width=scaled(5.0),
        border_height=scaled(1.0),
        num_rows=10,
        num_cols=20,
        horizontal_scale=scaled(0.05),
        vertical_scale=scaled(0.005),
        slope_threshold=1.5,
        use_cache=False,
        sub_terrains={
            "parkour": HfParkourTerrainCfg(
                proportion=0.2,
                pit_depth_range=scaled_pair((0.2, 1.0)),
                stone_length_lower_range=scaled_pair((0.9, 0.6)),
                stone_length_upper_range=scaled_pair((1.0, 0.8)),
                stone_width=scaled(1.0),
                last_stone_length=scaled(1.6),
                x_spacing_lower_range=scaled_pair((-0.1, -0.1)),
                x_spacing_upper_range=scaled_pair((0.1, 0.4)),
                y_offset_lower_range=scaled_pair((0.2, 0.2)),
                y_offset_upper_range=scaled_pair((0.3, 0.4)),
                incline_height_range=scaled_pair((0.0, 0.25)),
                last_incline_height_range=scaled_pair((0.1, 0.25)),
                final_platform_clearance=scaled(0.05),
                **common,
            ),
            "parkour_hurdle": HfParkourHurdleTerrainCfg(
                proportion=0.2,
                hurdle_length_range=scaled_pair((0.1, 0.4)),
                hurdle_height_lower_range=scaled_pair((0.1, 0.2)),
                hurdle_height_upper_range=scaled_pair((0.15, 0.4)),
                x_spacing_range=scaled_pair((1.2, 2.2)),
                y_offset_range=scaled_pair((-0.4, 0.4)),
                half_valid_width_range=scaled_pair((0.4, 0.8)),
                flat=False,
                **common,
            ),
            "parkour_flat": HfParkourHurdleTerrainCfg(
                proportion=0.2,
                hurdle_length_range=scaled_pair((0.1, 0.4)),
                hurdle_height_lower_range=scaled_pair((0.1, 0.2)),
                hurdle_height_upper_range=scaled_pair((0.15, 0.3)),
                x_spacing_range=scaled_pair((1.2, 2.2)),
                y_offset_range=scaled_pair((-0.4, 0.4)),
                half_valid_width_range=scaled_pair((0.45, 1.0)),
                flat=True,
                **common,
            ),
            "parkour_step": HfParkourStepTerrainCfg(
                proportion=0.2,
                step_height_range=scaled_pair((0.1, 0.45)),
                step_run_range=scaled_pair((0.3, 1.5)),
                y_offset_range=scaled_pair((-0.4, 0.4)),
                half_valid_width_range=scaled_pair((0.5, 1.0)),
                **common,
            ),
            "parkour_gap": HfParkourGapTerrainCfg(
                proportion=0.2,
                gap_width_range=scaled_pair((0.1, 0.8)),
                gap_depth_range=scaled_pair((0.2, 1.0)),
                x_spacing_range=scaled_pair((0.8, 1.5)),
                y_offset_range=scaled_pair((-0.4, 0.4)),
                half_valid_width_range=scaled_pair((0.6, 1.2)),
                **common,
            ),
        },
    )


EXTREME_PARKOUR_TERRAINS_CFG = make_extreme_parkour_terrains_cfg()

