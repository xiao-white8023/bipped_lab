"""Unit tests for the first-stage elevation-map baseline statistics."""

from lidar.baseline_elevation_analyzer import GridStatistics, finite_summary


import numpy as np


def test_finite_summary_ignores_nonfinite_values_and_scales():
    """Nonfinite samples must not contaminate scalar summaries."""
    result = finite_summary([0.001, np.nan, 0.003], scale=1000.0)

    assert result['count'] == 2
    assert np.isclose(result['mean'], 2.0)
    assert np.isclose(result['min'], 1.0)
    assert np.isclose(result['max'], 3.0)


def test_grid_statistics_tracks_coverage_heights_and_missing_runs():
    """Coverage and missing-run statistics must be cell-local."""
    statistics = GridStatistics(rows=2, cols=2)

    first = statistics.add(
        [0.1, -1.1, 0.2, -1.1],
        [1, 0, 1, 0],
    )
    second = statistics.add(
        [0.3, -1.1, -1.1, -1.1],
        [1, 0, 0, 0],
    )
    arrays = statistics.arrays()

    assert first['valid_count'] == 2
    assert second['valid_count'] == 1
    np.testing.assert_allclose(
        arrays['valid_ratio'], [[1.0, 0.0], [0.5, 0.0]]
    )
    np.testing.assert_allclose(
        arrays['height_mean'][[0, 1], [0, 0]], [0.2, 0.2]
    )
    np.testing.assert_array_equal(
        statistics.longest_invalid_run, [[0, 2], [1, 2]]
    )


def test_grid_statistics_detects_contract_violations():
    """Invalid masks and heights must produce explicit error counts."""
    statistics = GridStatistics(rows=2, cols=2)

    result = statistics.add(
        [-1.0, 1.1, np.nan, -1.1],
        [0, 1, 2, 0],
    )

    assert result['valid_value_errors'] == 1
    assert result['invalid_encoding_errors'] == 1
    assert result['nonfinite_valid_errors'] == 1
    assert result['valid_range_errors'] == 1
