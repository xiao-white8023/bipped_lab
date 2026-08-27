"""Low-pass filtering for G1 squat policy actions."""

from __future__ import annotations

import math

import numpy as np


def compute_low_pass_alpha(control_dt: float, cutoff_hz: float) -> float:
    """Return the exact-discretization coefficient of a first-order low-pass filter."""
    control_dt = float(control_dt)
    cutoff_hz = float(cutoff_hz)
    if control_dt <= 0.0:
        raise ValueError(f"control_dt must be positive, got {control_dt}.")

    nyquist_hz = 0.5 / control_dt
    if not 0.0 < cutoff_hz < nyquist_hz:
        raise ValueError(
            "action_filter.cutoff_hz must satisfy "
            f"0 < cutoff_hz < {nyquist_hz:g} Hz for control_dt={control_dt:g}, "
            f"got {cutoff_hz:g} Hz."
        )

    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * control_dt)


class FirstOrderLowPassFilter:
    """Stateful first-order low-pass filter for a fixed-size action vector."""

    def __init__(
        self,
        size: int,
        control_dt: float,
        cutoff_hz: float,
        enabled: bool = True,
    ) -> None:
        if size <= 0:
            raise ValueError(f"filter size must be positive, got {size}.")

        self.size = int(size)
        self.enabled = bool(enabled)
        self.cutoff_hz = float(cutoff_hz)
        self.alpha = (
            compute_low_pass_alpha(control_dt, cutoff_hz)
            if self.enabled
            else 1.0
        )
        self._state = np.zeros(self.size, dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current filtered action."""
        return self._state.copy()

    def reset(self, value: np.ndarray | None = None) -> None:
        """Reset the filter to zeros or to a supplied action vector."""
        if value is None:
            self._state.fill(0.0)
            return
        self._state[:] = self._validate_action(value)

    def update(self, action: np.ndarray) -> np.ndarray:
        """Filter one action sample and return a copy of the new state."""
        action = self._validate_action(action)
        if self.enabled:
            self._state += np.float32(self.alpha) * (action - self._state)
        else:
            self._state[:] = action
        return self._state.copy()

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.size,):
            raise ValueError(
                f"action must have shape ({self.size},), got {action.shape}."
            )
        return action
