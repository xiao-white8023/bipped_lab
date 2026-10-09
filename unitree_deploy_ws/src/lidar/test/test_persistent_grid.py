"""Tests for the persistent, covariance-aware world elevation grid."""

from lidar.persistent_grid import PersistentElevationGrid

import numpy as np


def make_grid(**overrides):
    parameters = {
        'resolution': 0.1,
        'variance_min': 1.0e-6,
        'variance_max': 1.0,
        'mahalanobis_threshold': 3.0,
        'candidate_confirmations': 3,
        'candidate_tolerance': 0.05,
        'candidate_timeout': 1.0,
        'conflict_variance_increase': 0.0,
    }
    parameters.update(overrides)
    return PersistentElevationGrid(**parameters)


def test_cells_remain_valid_after_ten_seconds_without_an_age_limit():
    """Stopping for longer than Phase 2's history must not erase terrain."""
    grid = make_grid()
    grid.update(np.array([[0.0, 0.0, -0.5]]), np.array([0.01]), 100.0)

    sampled = grid.sample(
        np.array([[0.0, 0.0]]),
        current_stamp=120.0,
        max_valid_variance=0.1,
        max_age_seconds=0.0,
    )

    assert sampled['valid'][0]
    assert np.isclose(sampled['height'][0], -0.5)
    assert np.isclose(sampled['height_age'][0], 20.0)
    assert np.isclose(sampled['freshest_age'][0], 20.0)


def test_unobserved_cell_uncertainty_grows_without_a_hard_expiry():
    grid = make_grid(variance_growth_per_second=0.001)
    grid.update(np.array([[0.0, 0.0, -0.5]]), np.array([0.01]), 0.0)

    early = grid.sample(np.array([[0.0, 0.0]]), 5.0, 0.1)
    late = grid.sample(np.array([[0.0, 0.0]]), 20.0, 0.1)

    assert early['valid'][0]
    assert late['valid'][0]
    assert np.isclose(early['variance'][0], 0.015)
    assert np.isclose(late['variance'][0], 0.03)


def test_optional_age_limit_invalidates_and_prunes_old_cells():
    grid = make_grid()
    grid.update(np.array([[0.0, 0.0, -0.5]]), np.array([0.01]), 0.0)

    sampled = grid.sample(
        np.array([[0.0, 0.0]]), 11.0, 0.1, max_age_seconds=10.0
    )
    removed = grid.prune(
        np.array([0.0, 0.0]), 2.0, 11.0, max_age_seconds=10.0
    )

    assert not sampled['valid'][0]
    assert removed == 1
    assert len(grid) == 0


def test_spatial_window_moves_with_the_robot():
    """Persistence is local in space, not an unbounded global map."""
    grid = make_grid()
    grid.update(
        np.array([[0.0, 0.0, -0.5], [3.0, 0.0, -0.4]]),
        np.array([0.01, 0.01]),
        0.0,
    )

    removed = grid.prune(np.array([3.0, 0.0]), 1.0, 1.0)

    assert removed == 1
    assert len(grid) == 1
    assert grid.sample(np.array([[3.0, 0.0]]), 1.0, 0.1)['valid'][0]


def test_covariance_weighted_fusion_prefers_precise_measurement():
    grid = make_grid()
    grid.update(np.array([[0.0, 0.0, 0.0]]), np.array([0.04]), 0.0)
    grid.update(np.array([[0.0, 0.0, 0.2]]), np.array([0.01]), 0.1)

    sampled = grid.sample(np.array([[0.0, 0.0]]), 0.1, 0.1)

    assert sampled['valid'][0]
    assert np.isclose(sampled['height'][0], 0.16)
    assert np.isclose(sampled['variance'][0], 0.008)
    assert sampled['observations'][0] == 2


def test_conflicting_surface_requires_repeated_confirmation():
    """One outlier cannot immediately overwrite a persistent cell."""
    grid = make_grid(mahalanobis_threshold=2.0)
    grid.update(np.array([[0.0, 0.0, 0.0]]), np.array([0.001]), 0.0)

    first = grid.update(
        np.array([[0.0, 0.0, 0.5]]), np.array([0.001]), 0.1
    )
    second = grid.update(
        np.array([[0.0, 0.0, 0.51]]), np.array([0.001]), 0.2
    )
    before_confirmation = grid.sample(
        np.array([[0.0, 0.0]]), 0.2, 0.1
    )
    third = grid.update(
        np.array([[0.0, 0.0, 0.49]]), np.array([0.001]), 0.3
    )
    after_confirmation = grid.sample(
        np.array([[0.0, 0.0]]), 0.3, 0.1
    )

    assert first['conflicts'] == 1 and first['replaced'] == 0
    assert second['conflicts'] == 1 and second['replaced'] == 0
    assert np.isclose(before_confirmation['height'][0], 0.0)
    assert third['replaced'] == 1
    assert np.isclose(after_confirmation['height'][0], 0.5, atol=0.01)


def test_one_scan_keeps_highest_measurement_per_world_cell():
    grid = make_grid()
    counters = grid.update(
        np.array(
            [
                [0.01, 0.01, -0.6],
                [0.02, -0.01, -0.4],
            ]
        ),
        np.array([0.01, 0.02]),
        0.0,
    )

    sampled = grid.sample(np.array([[0.0, 0.0]]), 0.0, 0.1)

    assert counters['measurements'] == 1
    assert np.isclose(sampled['height'][0], -0.4)


def test_source_timestamp_rewind_starts_a_new_coordinate_epoch():
    grid = make_grid()
    grid.update(np.array([[0.0, 0.0, -0.5]]), np.array([0.01]), 100.0)

    counters = grid.update(
        np.array([[1.0, 0.0, -0.4]]), np.array([0.01]), 50.0
    )

    assert counters['rewound']
    assert len(grid) == 1
    assert not grid.sample(np.array([[0.0, 0.0]]), 50.0, 0.1)['valid'][0]
    assert grid.sample(np.array([[1.0, 0.0]]), 50.0, 0.1)['valid'][0]


def test_sampling_snapshot_is_isolated_from_later_grid_updates():
    grid = make_grid()
    grid.update(np.array([[0.0, 0.0, -0.5]]), np.array([0.01]), 0.0)
    snapshot = grid.sampling_snapshot()

    grid.update(np.array([[0.0, 0.0, -0.4]]), np.array([0.01]), 0.1)

    original = snapshot.sample(np.array([[0.0, 0.0]]), 0.1, 0.1)
    updated = grid.sample(np.array([[0.0, 0.0]]), 0.1, 0.1)
    assert np.isclose(original['height'][0], -0.5)
    assert np.isclose(updated['height'][0], -0.45)


def test_hysteretic_indices_reject_boundary_jitter_then_switch_once():
    grid = make_grid()
    previous = np.array([[0, 0]], dtype=np.int64)

    for x in (0.049, 0.051, 0.048, 0.052, 0.069):
        selected = grid.hysteretic_cell_indices(
            np.array([[x, 0.0]]), previous, hysteresis_m=0.02
        )
        assert selected[0, 0] == 0
        previous = selected

    selected = grid.hysteretic_cell_indices(
        np.array([[0.071, 0.0]]), previous, hysteresis_m=0.02
    )
    assert selected[0, 0] == 1


def test_sample_can_use_an_explicit_hysteretic_cell_key():
    grid = make_grid()
    grid.update(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.8]]),
        np.array([0.01, 0.01]),
        0.0,
    )
    query = np.array([[0.051, 0.0]])

    nearest = grid.sample(query, 0.0, 0.1)
    held = grid.sample(
        query,
        0.0,
        0.1,
        cell_indices=np.array([[0, 0]], dtype=np.int64),
    )

    assert np.isclose(nearest['height'][0], 0.8)
    assert np.isclose(held['height'][0], 0.0)
