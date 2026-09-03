# Copyright (c) 2023 Xuxin Cheng, Kexin Shi, Deepak Pathak
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Adapted for Isaac Lab from Extreme Parkour, commit
# d2ffe27ba59a3229fad22a9fc94c38010bb1f519. See NOTICE and LICENSE in this
# directory. This file is for non-commercial use under CC BY-NC 4.0.

"""Isaac Lab terrain generators adapted from Extreme Parkour.

The original implementation writes into Isaac Gym ``SubTerrain`` height fields.
These functions build the same terrain families and return Isaac Lab's standard
``(list[trimesh.Trimesh], origin)`` result instead. Route goals intentionally do
not form part of this migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

if TYPE_CHECKING:
    from . import terrains_cfg


_INT16_INFO = np.iinfo(np.int16)


def _lerp(bounds: tuple[float, float], difficulty: float) -> float:
    """Linearly interpolate a value over the Isaac Lab difficulty interval."""
    return bounds[0] + difficulty * (bounds[1] - bounds[0])


def _validate_common_cfg(cfg) -> None:
    if len(cfg.size) != 2 or cfg.size[0] <= 0.0 or cfg.size[1] <= 0.0:
        raise ValueError(f"Terrain size must contain two positive values, got {cfg.size!r}.")
    if cfg.horizontal_scale <= 0.0 or cfg.vertical_scale <= 0.0:
        raise ValueError("Terrain horizontal_scale and vertical_scale must be positive.")
    if cfg.platform_length <= 0.0 or cfg.platform_length >= cfg.size[0]:
        raise ValueError(
            f"platform_length must be within (0, size[0]); got {cfg.platform_length} for size {cfg.size}."
        )
    if cfg.num_obstacles < 1:
        raise ValueError(f"num_obstacles must be positive, got {cfg.num_obstacles}.")
    if cfg.start_position_x < 0.0 or cfg.start_position_x > cfg.platform_length:
        raise ValueError(
            "start_position_x must lie on the initial platform; "
            f"got {cfg.start_position_x} with platform length {cfg.platform_length}."
        )
    if cfg.roughness_range[0] < 0.0 or cfg.roughness_range[1] < cfg.roughness_range[0]:
        raise ValueError(f"Invalid roughness_range: {cfg.roughness_range!r}.")
    if cfg.roughness_step <= 0.0:
        raise ValueError(f"roughness_step must be positive, got {cfg.roughness_step}.")
    if cfg.roughness_downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "roughness_downsampled_scale must be at least horizontal_scale; "
            f"got {cfg.roughness_downsampled_scale} < {cfg.horizontal_scale}."
        )


def _normalize_difficulty(difficulty: float) -> float:
    difficulty = float(difficulty)
    if not np.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"Terrain difficulty must be finite and within [0, 1], got {difficulty!r}.")
    return difficulty


def _make_rng(difficulty: float, cfg, terrain_salt: int) -> np.random.Generator:
    """Create a deterministic per-terrain RNG without changing NumPy global state."""
    seed = getattr(cfg, "seed", None)
    if seed is None:
        # Match Isaac Lab's global-seed behavior when TerrainGeneratorCfg.seed is None.
        seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    difficulty_key = int(round(difficulty * 1_000_000_000))
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, terrain_salt, difficulty_key])
    return np.random.default_rng(sequence)


def _height_field_shape(cfg) -> tuple[int, int]:
    # One extra sample makes the mesh span the complete configured physical size.
    width_cells = max(1, int(round(cfg.size[0] / cfg.horizontal_scale)))
    length_cells = max(1, int(round(cfg.size[1] / cfg.horizontal_scale)))
    return width_cells + 1, length_cells + 1


def _meters_to_horizontal_pixels(value: float, cfg, *, minimum: int = 0) -> int:
    return max(minimum, int(round(value / cfg.horizontal_scale)))


def _meters_to_vertical_units(value: float, cfg) -> int:
    return int(round(value / cfg.vertical_scale))


def _sample_integer(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample like ``np.random.randint(low, high)`` while tolerating quantized ranges."""
    if high <= low:
        return int(low)
    return int(rng.integers(low, high))


def _write_patch(
    height_field: np.ndarray,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    value: int | np.ndarray,
) -> None:
    """Write a scalar or array patch after clipping it to the terrain bounds."""
    clipped_x_start = max(0, x_start)
    clipped_x_end = min(height_field.shape[0], x_end)
    clipped_y_start = max(0, y_start)
    clipped_y_end = min(height_field.shape[1], y_end)
    if clipped_x_start >= clipped_x_end or clipped_y_start >= clipped_y_end:
        return

    if np.isscalar(value):
        height_field[clipped_x_start:clipped_x_end, clipped_y_start:clipped_y_end] = value
        return

    value_array = np.asarray(value)
    value_x_start = clipped_x_start - x_start
    value_x_end = value_x_start + clipped_x_end - clipped_x_start
    value_y_start = clipped_y_start - y_start
    value_y_end = value_y_start + clipped_y_end - clipped_y_start
    height_field[clipped_x_start:clipped_x_end, clipped_y_start:clipped_y_end] = value_array[
        value_x_start:value_x_end, value_y_start:value_y_end
    ]


def _apply_edge_padding(height_field: np.ndarray, cfg) -> None:
    pad_width = _meters_to_horizontal_pixels(cfg.edge_padding_width, cfg)
    if pad_width <= 0:
        return
    pad_width = min(pad_width, min(height_field.shape) // 2)
    pad_height = _meters_to_vertical_units(cfg.edge_padding_height, cfg)
    height_field[:pad_width, :] = pad_height
    height_field[-pad_width:, :] = pad_height
    height_field[:, :pad_width] = pad_height
    height_field[:, -pad_width:] = pad_height


def _interpolate_random_grid(coarse: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a small random grid using NumPy only."""
    coarse_x = np.linspace(0.0, 1.0, coarse.shape[0])
    coarse_y = np.linspace(0.0, 1.0, coarse.shape[1])
    output_x = np.linspace(0.0, 1.0, output_shape[0])
    output_y = np.linspace(0.0, 1.0, output_shape[1])

    along_x = np.empty((output_shape[0], coarse.shape[1]), dtype=np.float64)
    for column in range(coarse.shape[1]):
        along_x[:, column] = np.interp(output_x, coarse_x, coarse[:, column])

    output = np.empty(output_shape, dtype=np.float64)
    for row in range(output_shape[0]):
        output[row, :] = np.interp(output_y, coarse_y, along_x[row, :])
    return output


def _add_roughness(height_field: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Overlay the low-frequency random-uniform roughness used by the source terrains."""
    max_height = float(rng.uniform(cfg.roughness_range[0], cfg.roughness_range[1]))
    max_height_units = abs(_meters_to_vertical_units(max_height, cfg))
    step_units = max(1, abs(_meters_to_vertical_units(cfg.roughness_step, cfg)))
    possible_heights = np.arange(
        -max_height_units,
        max_height_units + step_units,
        step_units,
        dtype=np.int32,
    )
    if possible_heights.size == 0:
        return height_field

    coarse_shape = (
        max(2, int(np.ceil(cfg.size[0] / cfg.roughness_downsampled_scale)) + 1),
        max(2, int(np.ceil(cfg.size[1] / cfg.roughness_downsampled_scale)) + 1),
    )
    coarse = rng.choice(possible_heights, size=coarse_shape)
    roughness = np.rint(_interpolate_random_grid(coarse, height_field.shape)).astype(np.int32)
    combined = height_field.astype(np.int32) + roughness
    return np.clip(combined, _INT16_INFO.min, _INT16_INFO.max).astype(np.int16)


def _height_field_to_terrain(height_field: np.ndarray, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    vertices, faces = convert_height_field_to_mesh(
        height_field,
        horizontal_scale=cfg.horizontal_scale,
        vertical_scale=cfg.vertical_scale,
        slope_threshold=cfg.slope_threshold,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    origin = np.array([cfg.start_position_x, 0.5 * cfg.size[1], 0.0], dtype=np.float64)
    return [mesh], origin


def parkour_terrain(
    difficulty: float, cfg: terrains_cfg.HfParkourTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate alternating, laterally tilted stepping stones over a deep pit."""
    _validate_common_cfg(cfg)
    difficulty = _normalize_difficulty(difficulty)
    rng = _make_rng(difficulty, cfg, terrain_salt=11)

    pit_depth = rng.uniform(*cfg.pit_depth_range)
    height_field = np.full(
        _height_field_shape(cfg),
        -_meters_to_vertical_units(pit_depth, cfg),
        dtype=np.int16,
    )
    center_y = height_field.shape[1] // 2
    platform_length = _meters_to_horizontal_pixels(cfg.platform_length, cfg, minimum=1)
    height_field[:platform_length, :] = _meters_to_vertical_units(cfg.platform_height, cfg)

    stone_length_low = _lerp(cfg.stone_length_lower_range, difficulty)
    stone_length_high = _lerp(cfg.stone_length_upper_range, difficulty)
    stone_length = float(rng.uniform(stone_length_low, stone_length_high))
    # Preserve the source code's 0.2 m length quantization.
    stone_length = 2.0 * round(stone_length / 2.0, 1)
    stone_length_px = _meters_to_horizontal_pixels(stone_length, cfg, minimum=1)
    stone_width_px = _meters_to_horizontal_pixels(cfg.stone_width, cfg, minimum=1)
    last_stone_length_px = _meters_to_horizontal_pixels(cfg.last_stone_length, cfg, minimum=1)

    spacing_low = stone_length_px + _meters_to_horizontal_pixels(
        _lerp(cfg.x_spacing_lower_range, difficulty), cfg
    )
    spacing_high = stone_length_px + _meters_to_horizontal_pixels(
        _lerp(cfg.x_spacing_upper_range, difficulty), cfg
    )
    spacing_low = max(1, spacing_low)
    spacing_high = max(spacing_low + 1, spacing_high)

    lateral_low = _meters_to_horizontal_pixels(_lerp(cfg.y_offset_lower_range, difficulty), cfg)
    lateral_high = _meters_to_horizontal_pixels(_lerp(cfg.y_offset_upper_range, difficulty), cfg)
    lateral_low = max(0, lateral_low)
    lateral_high = max(lateral_low + 1, lateral_high)

    incline_height = _meters_to_vertical_units(_lerp(cfg.incline_height_range, difficulty), cfg)
    last_incline_height = _meters_to_vertical_units(_lerp(cfg.last_incline_height_range, difficulty), cfg)

    x_position = platform_length - _sample_integer(rng, spacing_low, spacing_high) + stone_length_px // 2
    left_right_flag = int(rng.integers(0, 2))
    last_x_end = platform_length
    for stone_index in range(cfg.num_obstacles):
        x_position += _sample_integer(rng, spacing_low, spacing_high)
        direction = 1 if left_right_flag else -1
        y_position = center_y + direction * _sample_integer(rng, lateral_low, lateral_high)

        if stone_index == cfg.num_obstacles - 1:
            x_position += last_stone_length_px // 4
            current_length = last_stone_length_px
            current_incline = last_incline_height
        else:
            current_length = stone_length_px
            current_incline = incline_height

        tilted_profile = np.linspace(-current_incline, current_incline, stone_width_px) * direction
        stone_patch = np.tile(tilted_profile, (current_length, 1)).astype(np.int16)
        x_start = x_position - current_length // 2
        x_end = x_start + current_length
        y_start = y_position - stone_width_px // 2
        y_end = y_start + stone_width_px
        _write_patch(height_field, x_start, x_end, y_start, y_end, stone_patch)
        last_x_end = max(last_x_end, x_end)
        left_right_flag = 1 - left_right_flag

    platform_clearance = _meters_to_horizontal_pixels(cfg.final_platform_clearance, cfg)
    final_platform_start = min(height_field.shape[0], last_x_end + platform_clearance)
    height_field[final_platform_start:, :] = _meters_to_vertical_units(cfg.platform_height, cfg)
    _apply_edge_padding(height_field, cfg)
    height_field = _add_roughness(height_field, cfg, rng)
    return _height_field_to_terrain(height_field, cfg)


def parkour_hurdle_terrain(
    difficulty: float, cfg: terrains_cfg.HfParkourHurdleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate laterally offset hurdles, or the source rough-flat baseline."""
    _validate_common_cfg(cfg)
    difficulty = _normalize_difficulty(difficulty)
    rng = _make_rng(difficulty, cfg, terrain_salt=23 if not cfg.flat else 29)

    height_field = np.zeros(_height_field_shape(cfg), dtype=np.int16)
    center_y = height_field.shape[1] // 2
    platform_length = _meters_to_horizontal_pixels(cfg.platform_length, cfg, minimum=1)
    height_field[:platform_length, :] = _meters_to_vertical_units(cfg.platform_height, cfg)

    hurdle_length = _meters_to_horizontal_pixels(_lerp(cfg.hurdle_length_range, difficulty), cfg, minimum=1)
    x_spacing_low = _meters_to_horizontal_pixels(cfg.x_spacing_range[0], cfg, minimum=1)
    x_spacing_high = _meters_to_horizontal_pixels(cfg.x_spacing_range[1], cfg, minimum=1)
    x_spacing_high = max(x_spacing_low + 1, x_spacing_high)
    y_offset_low = _meters_to_horizontal_pixels(cfg.y_offset_range[0], cfg)
    y_offset_high = _meters_to_horizontal_pixels(cfg.y_offset_range[1], cfg)
    y_offset_high = max(y_offset_low + 1, y_offset_high)

    half_valid_width = rng.uniform(*cfg.half_valid_width_range)
    half_valid_width_px = _meters_to_horizontal_pixels(half_valid_width, cfg, minimum=1)
    hurdle_height_low = _meters_to_vertical_units(_lerp(cfg.hurdle_height_lower_range, difficulty), cfg)
    hurdle_height_high = _meters_to_vertical_units(_lerp(cfg.hurdle_height_upper_range, difficulty), cfg)
    hurdle_height_high = max(hurdle_height_low + 1, hurdle_height_high)

    x_position = platform_length
    for _ in range(cfg.num_obstacles):
        x_position += _sample_integer(rng, x_spacing_low, x_spacing_high)
        y_offset = _sample_integer(rng, y_offset_low, y_offset_high)
        if cfg.flat:
            continue

        hurdle_height = _sample_integer(rng, hurdle_height_low, hurdle_height_high)
        x_start = x_position - hurdle_length // 2
        x_end = x_start + hurdle_length
        y_start = center_y + y_offset - half_valid_width_px
        y_end = center_y + y_offset + half_valid_width_px
        _write_patch(height_field, x_start, x_end, y_start, y_end, hurdle_height)

    _apply_edge_padding(height_field, cfg)
    height_field = _add_roughness(height_field, cfg, rng)
    return _height_field_to_terrain(height_field, cfg)


def parkour_step_terrain(
    difficulty: float, cfg: terrains_cfg.HfParkourStepTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate large platforms that ascend and then descend along the course."""
    _validate_common_cfg(cfg)
    difficulty = _normalize_difficulty(difficulty)
    rng = _make_rng(difficulty, cfg, terrain_salt=37)

    height_field = np.zeros(_height_field_shape(cfg), dtype=np.int16)
    center_y = height_field.shape[1] // 2
    platform_length = _meters_to_horizontal_pixels(cfg.platform_length, cfg, minimum=1)
    height_field[:platform_length, :] = _meters_to_vertical_units(cfg.platform_height, cfg)

    step_height_m = _lerp(cfg.step_height_range, difficulty)
    step_height = _meters_to_vertical_units(step_height_m, cfg)
    run_low = _meters_to_horizontal_pixels(cfg.step_run_range[0] + step_height_m, cfg, minimum=1)
    run_high = _meters_to_horizontal_pixels(cfg.step_run_range[1] + step_height_m, cfg, minimum=1)
    run_high = max(run_low + 1, run_high)
    y_offset_low = _meters_to_horizontal_pixels(cfg.y_offset_range[0], cfg)
    y_offset_high = _meters_to_horizontal_pixels(cfg.y_offset_range[1], cfg)
    y_offset_high = max(y_offset_low + 1, y_offset_high)
    half_valid_width = _meters_to_horizontal_pixels(
        rng.uniform(*cfg.half_valid_width_range), cfg, minimum=1
    )

    x_position = platform_length
    stair_height = 0
    for step_index in range(cfg.num_obstacles):
        run_length = _sample_integer(rng, run_low, run_high)
        y_offset = _sample_integer(rng, y_offset_low, y_offset_high)
        if step_index < cfg.num_obstacles // 2:
            stair_height += step_height
        elif step_index > cfg.num_obstacles // 2:
            stair_height -= step_height

        x_end = x_position + run_length
        y_start = center_y + y_offset - half_valid_width
        y_end = center_y + y_offset + half_valid_width
        _write_patch(height_field, x_position, x_end, y_start, y_end, stair_height)
        x_position = x_end

    _apply_edge_padding(height_field, cfg)
    height_field = _add_roughness(height_field, cfg, rng)
    return _height_field_to_terrain(height_field, cfg)


def parkour_gap_terrain(
    difficulty: float, cfg: terrains_cfg.HfParkourGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a winding corridor interrupted by full-width gaps."""
    _validate_common_cfg(cfg)
    difficulty = _normalize_difficulty(difficulty)
    rng = _make_rng(difficulty, cfg, terrain_salt=41)

    height_field = np.zeros(_height_field_shape(cfg), dtype=np.int16)
    center_y = height_field.shape[1] // 2
    platform_length = _meters_to_horizontal_pixels(cfg.platform_length, cfg, minimum=1)
    height_field[:platform_length, :] = _meters_to_vertical_units(cfg.platform_height, cfg)

    gap_width_m = _lerp(cfg.gap_width_range, difficulty)
    gap_width = _meters_to_horizontal_pixels(gap_width_m, cfg, minimum=1)
    gap_depth = -_meters_to_vertical_units(rng.uniform(*cfg.gap_depth_range), cfg)
    spacing_low = _meters_to_horizontal_pixels(cfg.x_spacing_range[0] + gap_width_m, cfg, minimum=1)
    spacing_high = _meters_to_horizontal_pixels(cfg.x_spacing_range[1] + gap_width_m, cfg, minimum=1)
    spacing_high = max(spacing_low + 1, spacing_high)
    y_offset_low = _meters_to_horizontal_pixels(cfg.y_offset_range[0], cfg)
    y_offset_high = _meters_to_horizontal_pixels(cfg.y_offset_range[1], cfg)
    y_offset_high = max(y_offset_low + 1, y_offset_high)
    half_valid_width = _meters_to_horizontal_pixels(
        rng.uniform(*cfg.half_valid_width_range), cfg, minimum=1
    )

    x_position = platform_length
    previous_x = x_position
    for _ in range(cfg.num_obstacles):
        x_position += _sample_integer(rng, spacing_low, spacing_high)
        y_offset = _sample_integer(rng, y_offset_low, y_offset_high)

        gap_start = x_position - gap_width // 2
        _write_patch(height_field, gap_start, gap_start + gap_width, 0, height_field.shape[1], gap_depth)

        corridor_start = center_y + y_offset - half_valid_width
        corridor_end = center_y + y_offset + half_valid_width
        _write_patch(height_field, previous_x, x_position, 0, corridor_start, gap_depth)
        _write_patch(
            height_field,
            previous_x,
            x_position,
            corridor_end,
            height_field.shape[1],
            gap_depth,
        )
        previous_x = x_position

    _apply_edge_padding(height_field, cfg)
    height_field = _add_roughness(height_field, cfg, rng)
    return _height_field_to_terrain(height_field, cfg)

