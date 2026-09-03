# Copyright (c) 2023 Xuxin Cheng, Kexin Shi, Deepak Pathak
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Non-commercial Isaac Lab adaptation of the Extreme Parkour terrains."""

from .terrains import parkour_gap_terrain, parkour_hurdle_terrain, parkour_step_terrain, parkour_terrain
from .terrains_cfg import (
    EXTREME_PARKOUR_TERRAINS_CFG,
    HfParkourGapTerrainCfg,
    HfParkourHurdleTerrainCfg,
    HfParkourStepTerrainCfg,
    HfParkourTerrainCfg,
    make_extreme_parkour_terrains_cfg,
)

__all__ = [
    "EXTREME_PARKOUR_TERRAINS_CFG",
    "HfParkourGapTerrainCfg",
    "HfParkourHurdleTerrainCfg",
    "HfParkourStepTerrainCfg",
    "HfParkourTerrainCfg",
    "make_extreme_parkour_terrains_cfg",
    "parkour_gap_terrain",
    "parkour_hurdle_terrain",
    "parkour_step_terrain",
    "parkour_terrain",
]

