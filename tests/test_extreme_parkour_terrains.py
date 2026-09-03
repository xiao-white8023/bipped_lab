# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Tests for the Isaac Lab Extreme Parkour terrain adaptation."""

from isaaclab.app import AppLauncher

# Isaac Sim must be running before importing the rest of Isaac Lab.
simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest

from isaaclab.terrains import TerrainGenerator

from legged_lab.terrains.extreme_parkour import (
    EXTREME_PARKOUR_TERRAINS_CFG,
    make_extreme_parkour_terrains_cfg,
)


TERRAIN_NAMES = ("parkour", "parkour_hurdle", "parkour_flat", "parkour_step", "parkour_gap")


def _prepared_sub_terrain(name: str, *, scale: float = 1.0, seed: int = 0):
    parent_cfg = make_extreme_parkour_terrains_cfg(geometry_scale=scale, seed=seed)
    cfg = parent_cfg.sub_terrains[name].copy()
    cfg.size = parent_cfg.size
    cfg.horizontal_scale = parent_cfg.horizontal_scale
    cfg.vertical_scale = parent_cfg.vertical_scale
    cfg.slope_threshold = parent_cfg.slope_threshold
    cfg.seed = parent_cfg.seed
    return cfg


def _generate(name: str, difficulty: float, *, scale: float = 1.0, seed: int = 0):
    cfg = _prepared_sub_terrain(name, scale=scale, seed=seed)
    meshes, origin = cfg.function(difficulty, cfg)
    assert len(meshes) == 1
    return cfg, meshes[0], origin


@pytest.mark.parametrize("name", TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty", (0.0, 0.5, 1.0))
def test_sub_terrain_mesh_contract(name: str, difficulty: float):
    cfg, mesh, origin = _generate(name, difficulty)

    assert mesh.vertices.ndim == 2 and mesh.vertices.shape[1] == 3
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < mesh.vertices.shape[0]
    assert np.isfinite(mesh.vertices).all()
    assert np.isfinite(origin).all()

    np.testing.assert_allclose(mesh.bounds[0, :2], (0.0, 0.0), atol=1.0e-6)
    np.testing.assert_allclose(mesh.bounds[1, :2], cfg.size, atol=1.0e-6)
    np.testing.assert_allclose(origin, (cfg.start_position_x, cfg.size[1] / 2.0, 0.0), atol=1.0e-6)
    assert 0.0 <= origin[0] <= cfg.platform_length


def test_terrain_geometry_characteristics_and_difficulty():
    _, parkour, _ = _generate("parkour", 1.0)
    assert parkour.vertices[:, 2].min() < -0.15
    assert parkour.vertices[:, 2].max() > 0.1

    _, hurdle_easy, _ = _generate("parkour_hurdle", 0.0)
    _, hurdle_hard, _ = _generate("parkour_hurdle", 1.0)
    assert hurdle_hard.vertices[:, 2].max() > hurdle_easy.vertices[:, 2].max()

    _, step_easy, _ = _generate("parkour_step", 0.0)
    _, step_hard, _ = _generate("parkour_step", 1.0)
    assert step_hard.vertices[:, 2].max() > step_easy.vertices[:, 2].max()

    flat_cfg, flat, _ = _generate("parkour_flat", 1.0)
    assert np.abs(flat.vertices[:, 2]).max() <= flat_cfg.roughness_range[1] + flat_cfg.vertical_scale

    gap_easy_cfg, gap_easy, _ = _generate("parkour_gap", 0.0)
    gap_hard_cfg, gap_hard, _ = _generate("parkour_gap", 1.0)
    easy_width = _largest_transverse_gap_width(gap_easy_cfg, gap_easy)
    hard_width = _largest_transverse_gap_width(gap_hard_cfg, gap_hard)
    assert hard_width > easy_width


def _largest_transverse_gap_width(cfg, mesh) -> float:
    shape = (
        int(round(cfg.size[0] / cfg.horizontal_scale)) + 1,
        int(round(cfg.size[1] / cfg.horizontal_scale)) + 1,
    )
    heights = mesh.vertices[:, 2].reshape(shape)
    # A transverse gap is deep over almost all of the Y dimension. Edge padding
    # leaves a narrow non-gap strip, hence the 90% threshold.
    transverse_rows = np.mean(heights < -0.1, axis=1) > 0.9
    longest_run = 0
    current_run = 0
    for is_gap in transverse_rows:
        current_run = current_run + 1 if is_gap else 0
        longest_run = max(longest_run, current_run)
    return longest_run * cfg.horizontal_scale


@pytest.mark.parametrize("name", TERRAIN_NAMES)
def test_generation_is_reproducible_without_global_rng_pollution(name: str):
    np.random.seed(1234)
    state_before = np.random.get_state()
    _, mesh_a, origin_a = _generate(name, 0.63, seed=91)
    state_after = np.random.get_state()
    _, mesh_b, origin_b = _generate(name, 0.63, seed=91)

    np.testing.assert_array_equal(mesh_a.vertices, mesh_b.vertices)
    np.testing.assert_array_equal(mesh_a.faces, mesh_b.faces)
    np.testing.assert_array_equal(origin_a, origin_b)
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_factory_defaults_and_uniform_scaling():
    cfg = EXTREME_PARKOUR_TERRAINS_CFG
    assert cfg.curriculum is True
    assert cfg.size == (18.0, 4.0)
    assert cfg.num_rows == 10
    assert cfg.num_cols == 20
    assert cfg.horizontal_scale == pytest.approx(0.05)
    assert cfg.vertical_scale == pytest.approx(0.005)
    assert tuple(cfg.sub_terrains) == TERRAIN_NAMES
    assert sum(sub_cfg.proportion for sub_cfg in cfg.sub_terrains.values()) == pytest.approx(1.0)

    half = make_extreme_parkour_terrains_cfg(geometry_scale=0.5, seed=7)
    assert half.size == pytest.approx((9.0, 2.0))
    assert half.border_width == pytest.approx(2.5)
    assert half.horizontal_scale == pytest.approx(0.025)
    assert half.vertical_scale == pytest.approx(0.0025)
    assert half.sub_terrains["parkour"].stone_width == pytest.approx(0.5)
    assert half.sub_terrains["parkour_gap"].gap_width_range == pytest.approx((0.05, 0.4))

    half_cfg, half_mesh, half_origin = _generate("parkour_flat", 0.5, scale=0.5, seed=7)
    np.testing.assert_allclose(half_mesh.bounds[:, :2], ((0.0, 0.0), half_cfg.size), atol=1.0e-6)
    np.testing.assert_allclose(
        half_origin,
        (half_cfg.start_position_x, half_cfg.size[1] / 2.0, 0.0),
        atol=1.0e-6,
    )

    with pytest.raises(ValueError, match="geometry_scale"):
        make_extreme_parkour_terrains_cfg(geometry_scale=0.0)


def test_small_curriculum_terrain_generator_integration():
    cfg = make_extreme_parkour_terrains_cfg(seed=17)
    cfg.num_rows = 2
    cfg.num_cols = 5
    generator = TerrainGenerator(cfg=cfg, device="cpu")

    assert generator.terrain_origins.shape == (2, 5, 3)
    expected_width = cfg.size[0] * cfg.num_rows + 2.0 * cfg.border_width
    expected_length = cfg.size[1] * cfg.num_cols + 2.0 * cfg.border_width
    actual_size = generator.terrain_mesh.bounds[1] - generator.terrain_mesh.bounds[0]
    assert actual_size[0] == pytest.approx(expected_width)
    assert actual_size[1] == pytest.approx(expected_length)

    expected_x = np.array([-17.0, 1.0])
    expected_y = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    np.testing.assert_allclose(generator.terrain_origins[:, 0, 0], expected_x, atol=1.0e-6)
    np.testing.assert_allclose(generator.terrain_origins[0, :, 1], expected_y, atol=1.0e-6)
    np.testing.assert_allclose(generator.terrain_origins[:, :, 2], 0.0, atol=1.0e-6)
