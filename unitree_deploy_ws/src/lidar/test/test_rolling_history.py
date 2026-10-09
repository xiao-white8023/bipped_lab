"""Tests for the source-time rolling point history."""

from lidar.rolling_history import highest_point_per_cell, TimedPointHistory

import numpy as np


def test_highest_point_per_cell_keeps_actual_highest_samples():
    """Per-packet reduction must retain one real, highest point per XY cell."""
    points = np.array(
        [
            [-0.04, 0.01, -0.60],
            [0.03, -0.02, -0.45],
            [0.21, 0.01, -0.70],
            [0.22, 0.02, -0.50],
        ],
        dtype=np.float64,
    )

    reduced = highest_point_per_cell(points, 1.6, 1.0, 0.1)

    assert reduced.shape == (2, 3)
    np.testing.assert_allclose(
        reduced[np.argsort(reduced[:, 0])],
        [[0.03, -0.02, -0.45], [0.22, 0.02, -0.50]],
    )


def test_history_uses_source_time_and_reports_point_age():
    """Packets inside the configured ten-second window remain available."""
    history = TimedPointHistory(duration_seconds=10.0, max_frames=200)
    history.push(100.0, np.array([[1.0, 2.0, 3.0]]))
    history.push(109.9, np.array([[4.0, 5.0, 6.0]]))

    points, ages = history.points_and_ages(109.9)

    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(ages, [9.9, 0.0])
    assert np.isclose(history.span_seconds, 9.9)


def test_history_expires_old_packets_and_bounds_frame_count():
    """Both source-time duration and frame count must bound memory use."""
    history = TimedPointHistory(duration_seconds=10.0, max_frames=2)
    point = np.zeros((1, 3), dtype=np.float64)
    history.push(0.0, point)
    history.push(5.0, point)
    history.push(9.0, point)
    assert len(history) == 2

    history.push(20.0, point)
    assert len(history) == 1
    assert history.point_count == 1


def test_history_clears_when_rosbag_time_moves_backwards():
    """Looped bag playback or sensor resets must not mix coordinate epochs."""
    history = TimedPointHistory(duration_seconds=10.0, max_frames=200)
    history.push(100.0, np.ones((2, 3), dtype=np.float64))

    rewound = history.push(50.0, np.zeros((1, 3), dtype=np.float64))

    assert rewound
    assert len(history) == 1
    assert history.point_count == 1
