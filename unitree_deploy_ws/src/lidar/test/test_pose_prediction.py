"""Tests for bounded source-pose latency compensation."""

from lidar.pose_prediction import PosePredictor

import numpy as np


def make_predictor(**overrides):
    parameters = {
        'window_seconds': 0.4,
        'minimum_span_seconds': 0.15,
        'maximum_horizon_seconds': 0.25,
        'maximum_clock_age_seconds': 1.0,
        'linear_deadband': 0.02,
        'yaw_rate_deadband': 0.02,
        'yaw_prediction_gain': 0.5,
        'maximum_linear_speed': 1.5,
        'maximum_vertical_speed': 0.5,
        'maximum_yaw_rate': 2.0,
    }
    parameters.update(overrides)
    return PosePredictor(**parameters)


def test_constant_velocity_prediction_compensates_source_age():
    predictor = make_predictor()
    for stamp in (0.0, 0.1, 0.2, 0.3):
        predictor.update(stamp, np.array([0.8 * stamp, 0.0, 0.5]), 0.0)

    position, yaw, horizon, compatible = predictor.predict(0.48)

    assert compatible
    assert np.isclose(horizon, 0.18)
    assert np.isclose(position[0], 0.8 * 0.48)
    assert np.isclose(position[2], 0.5)
    assert np.isclose(yaw, 0.0)


def test_prediction_horizon_and_velocity_are_bounded():
    predictor = make_predictor(maximum_linear_speed=1.0)
    predictor.update(0.0, np.array([0.0, 0.0, 0.0]), 0.0)
    predictor.update(0.2, np.array([2.0, 0.0, 0.0]), 0.0)

    position, _, horizon, compatible = predictor.predict(0.8)

    assert compatible
    assert np.isclose(horizon, 0.25)
    assert np.isclose(position[0], 2.25)


def test_stationary_pose_noise_is_rejected_by_deadband():
    predictor = make_predictor(linear_deadband=0.05)
    samples = (0.000, 0.003, -0.002, 0.002, 0.000)
    for index, x in enumerate(samples):
        predictor.update(
            index * 0.1, np.array([x, 0.0, 0.5]), 0.002 * (-1) ** index
        )

    position, yaw, horizon, compatible = predictor.predict(0.58)

    assert compatible
    assert np.isclose(horizon, 0.18)
    assert np.allclose(position, np.array([0.0, 0.0, 0.5]))
    assert np.isclose(yaw, 0.002)


def test_incompatible_or_stale_clock_disables_prediction():
    predictor = make_predictor(maximum_clock_age_seconds=0.5)
    predictor.update(10.0, np.array([1.0, 2.0, 0.5]), 0.2)

    position, yaw, horizon, compatible = predictor.predict(20.0)

    assert not compatible
    assert horizon == 0.0
    assert np.allclose(position, np.array([1.0, 2.0, 0.5]))
    assert np.isclose(yaw, 0.2)


def test_yaw_rate_fit_handles_pi_wrap():
    predictor = make_predictor()
    predictor.update(0.0, np.zeros(3), 3.10)
    predictor.update(0.2, np.zeros(3), -3.083185307179586)

    _, yaw, horizon, compatible = predictor.predict(0.4)

    assert compatible
    assert np.isclose(horizon, 0.2)
    assert np.isclose(yaw, -3.033185307179586)
