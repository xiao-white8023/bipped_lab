"""Short-horizon pose prediction for latency-compensated map sampling."""

from collections import deque
import math

import numpy as np


def angle_difference(first, second):
    """Return the wrapped angle ``first - second`` in radians."""
    return math.atan2(
        math.sin(first - second), math.cos(first - second)
    )


class PosePredictor:
    """
    Estimate torso velocity from timestamped poses and extrapolate briefly.

    FAST-LIO leaves ``Odometry.twist`` at zero, so velocity is fitted from a
    short history of pose samples.  Prediction is deliberately bounded: it is
    only intended to compensate sensor and processing latency, never to act as
    a stand-alone odometry source.
    """

    def __init__(
        self,
        window_seconds=0.25,
        minimum_span_seconds=0.10,
        maximum_horizon_seconds=0.25,
        maximum_clock_age_seconds=1.0,
        linear_deadband=0.08,
        yaw_rate_deadband=0.04,
        yaw_prediction_gain=0.5,
        maximum_linear_speed=1.5,
        maximum_vertical_speed=0.5,
        maximum_yaw_rate=2.0,
    ):
        self.window_seconds = float(window_seconds)
        self.minimum_span_seconds = float(minimum_span_seconds)
        self.maximum_horizon_seconds = float(maximum_horizon_seconds)
        self.maximum_clock_age_seconds = float(
            maximum_clock_age_seconds
        )
        self.linear_deadband = float(linear_deadband)
        self.yaw_rate_deadband = float(yaw_rate_deadband)
        self.yaw_prediction_gain = float(yaw_prediction_gain)
        self.maximum_linear_speed = float(maximum_linear_speed)
        self.maximum_vertical_speed = float(maximum_vertical_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self._validate_configuration()
        self.reset()

    def _validate_configuration(self):
        if self.window_seconds <= 0.0:
            raise ValueError('window_seconds must be positive')
        if not 0.0 < self.minimum_span_seconds <= self.window_seconds:
            raise ValueError(
                'minimum_span_seconds must lie in (0, window_seconds]'
            )
        if self.maximum_horizon_seconds < 0.0:
            raise ValueError('maximum_horizon_seconds must be non-negative')
        if self.maximum_clock_age_seconds < self.maximum_horizon_seconds:
            raise ValueError(
                'maximum_clock_age_seconds must cover the prediction horizon'
            )
        if min(self.linear_deadband, self.yaw_rate_deadband) < 0.0:
            raise ValueError('prediction deadbands must be non-negative')
        if not 0.0 <= self.yaw_prediction_gain <= 1.0:
            raise ValueError('yaw_prediction_gain must lie in [0, 1]')
        if min(
            self.maximum_linear_speed,
            self.maximum_vertical_speed,
            self.maximum_yaw_rate,
        ) <= 0.0:
            raise ValueError('prediction rate limits must be positive')

    def reset(self):
        """Discard pose history and all estimated rates."""
        self._history = deque()
        self._latest_stamp = None
        self._latest_position = None
        self._latest_yaw = None
        self._latest_unwrapped_yaw = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._yaw_rate = 0.0

    @property
    def ready(self):
        """Return whether at least one pose is available."""
        return self._latest_stamp is not None

    @property
    def latest_stamp(self):
        """Return the newest pose source timestamp or ``None``."""
        return self._latest_stamp

    @staticmethod
    def _least_squares_slope(times, values):
        centered = times - np.mean(times)
        denominator = float(centered @ centered)
        if denominator <= 1.0e-12:
            return np.zeros(values.shape[1], dtype=np.float64)
        return centered @ values / denominator

    def _estimate_rates(self):
        if len(self._history) < 2:
            self._velocity.fill(0.0)
            self._yaw_rate = 0.0
            return
        stamps = np.asarray(
            [sample[0] for sample in self._history], dtype=np.float64
        )
        if stamps[-1] - stamps[0] < self.minimum_span_seconds:
            self._velocity.fill(0.0)
            self._yaw_rate = 0.0
            return
        positions = np.stack([sample[1] for sample in self._history])
        unwrapped_yaw = np.asarray(
            [sample[2] for sample in self._history], dtype=np.float64
        )[:, None]
        velocity = self._least_squares_slope(stamps, positions)
        yaw_rate = float(
            self._least_squares_slope(stamps, unwrapped_yaw)[0]
        )

        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed < self.linear_deadband:
            velocity[:2] = 0.0
        elif horizontal_speed > self.maximum_linear_speed:
            velocity[:2] *= self.maximum_linear_speed / horizontal_speed
        velocity[2] = float(
            np.clip(
                velocity[2],
                -self.maximum_vertical_speed,
                self.maximum_vertical_speed,
            )
        )
        if abs(velocity[2]) < self.linear_deadband:
            velocity[2] = 0.0
        if abs(yaw_rate) < self.yaw_rate_deadband:
            yaw_rate = 0.0
        self._velocity = velocity
        self._yaw_rate = float(
            np.clip(
                yaw_rate, -self.maximum_yaw_rate, self.maximum_yaw_rate
            )
        )

    def update(self, stamp, position, yaw):
        """Insert one source-time pose sample."""
        stamp = float(stamp)
        position = np.asarray(position, dtype=np.float64)
        yaw = float(yaw)
        if position.shape != (3,):
            raise ValueError('position must have shape (3,)')
        if not np.isfinite(position).all() or not math.isfinite(yaw):
            raise ValueError('pose samples must be finite')
        if self._latest_stamp is not None and stamp <= self._latest_stamp:
            if stamp < self._latest_stamp - 1.0e-6:
                self.reset()
            else:
                return False

        if self._latest_yaw is None:
            unwrapped_yaw = yaw
        else:
            unwrapped_yaw = self._latest_unwrapped_yaw + angle_difference(
                yaw, self._latest_yaw
            )
        self._history.append((stamp, position.copy(), unwrapped_yaw))
        while (
            len(self._history) > 1
            and stamp - self._history[0][0] > self.window_seconds
        ):
            self._history.popleft()
        self._latest_stamp = stamp
        self._latest_position = position.copy()
        self._latest_yaw = yaw
        self._latest_unwrapped_yaw = unwrapped_yaw
        self._estimate_rates()
        return True

    def predict(self, target_stamp):
        """Return position, yaw, horizon and clock-compatibility flag."""
        if not self.ready:
            raise RuntimeError('cannot predict before the first pose sample')
        age = float(target_stamp) - self._latest_stamp
        clock_compatible = (
            0.0 <= age <= self.maximum_clock_age_seconds
        )
        horizon = (
            min(age, self.maximum_horizon_seconds)
            if clock_compatible
            else 0.0
        )
        position = self._latest_position + self._velocity * horizon
        yaw = (
            self._latest_yaw
            + self.yaw_prediction_gain * self._yaw_rate * horizon
        )
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return position, yaw, float(horizon), clock_compatible
