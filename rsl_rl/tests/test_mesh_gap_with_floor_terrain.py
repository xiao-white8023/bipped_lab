from __future__ import annotations

import numpy as np

from legged_lab.terrains.mesh_gap_terrain import MeshGapWithFloorTerrainCfg
from legged_lab.terrains.terrain_generator_cfg import ROUGH_TERRAINS_CFG


def _make_cfg() -> MeshGapWithFloorTerrainCfg:
    return MeshGapWithFloorTerrainCfg(
        proportion=1.0,
        size=(8.0, 8.0),
        gap_width_range=(0.1, 0.5),
        gap_depth_range=(0.2, 1.0),
        platform_width=3.0,
    )


def _floor_meshes(meshes):
    # Four outer-border meshes + one platform mesh precede the four floor-ring meshes.
    return meshes[5:]


def test_gap_floor_depth_tracks_curriculum_difficulty():
    cfg = _make_cfg()

    easy_meshes, easy_origin = cfg.function(0.0, cfg)
    hard_meshes, hard_origin = cfg.function(1.0, cfg)

    assert len(easy_meshes) == 9
    assert len(hard_meshes) == 9
    np.testing.assert_allclose(easy_origin, (4.0, 4.0, 0.0))
    np.testing.assert_allclose(hard_origin, easy_origin)
    for floor in _floor_meshes(easy_meshes):
        np.testing.assert_allclose(floor.bounds[1, 2], -0.2)
    for floor in _floor_meshes(hard_meshes):
        np.testing.assert_allclose(floor.bounds[1, 2], -1.0)


def test_gap_floor_width_tracks_curriculum_difficulty():
    cfg = _make_cfg()

    easy_meshes, _ = cfg.function(0.0, cfg)
    hard_meshes, _ = cfg.function(1.0, cfg)

    easy_floor_bounds = np.stack([mesh.bounds for mesh in _floor_meshes(easy_meshes)])
    hard_floor_bounds = np.stack([mesh.bounds for mesh in _floor_meshes(hard_meshes)])
    np.testing.assert_allclose(easy_floor_bounds[:, 0, :2].min(axis=0), (2.4, 2.4))
    np.testing.assert_allclose(easy_floor_bounds[:, 1, :2].max(axis=0), (5.6, 5.6))
    np.testing.assert_allclose(hard_floor_bounds[:, 0, :2].min(axis=0), (2.0, 2.0))
    np.testing.assert_allclose(hard_floor_bounds[:, 1, :2].max(axis=0), (6.0, 6.0))


def test_rough_terrain_uses_gap_with_floor():
    gap_cfg = ROUGH_TERRAINS_CFG.sub_terrains["gap"]

    assert isinstance(gap_cfg, MeshGapWithFloorTerrainCfg)
    assert gap_cfg.gap_width_range == (0.1, 0.5)
    assert gap_cfg.gap_depth_range == (0.2, 1.0)
    assert gap_cfg.platform_width == 3.0
