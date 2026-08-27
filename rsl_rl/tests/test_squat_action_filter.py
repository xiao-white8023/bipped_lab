from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "leggeddeploy"))

from squat_action_filter import (  # noqa: E402
    FirstOrderLowPassFilter,
    compute_low_pass_alpha,
)


def test_five_hz_alpha_at_fifty_hz_control_rate():
    alpha = compute_low_pass_alpha(control_dt=0.02, cutoff_hz=5.0)
    expected = 1.0 - math.exp(-2.0 * math.pi * 5.0 * 0.02)

    assert alpha == pytest.approx(expected)
    assert alpha == pytest.approx(0.4665119089)


def test_filter_starts_at_zero_and_converges_monotonically_to_step():
    action_filter = FirstOrderLowPassFilter(
        size=3,
        control_dt=0.02,
        cutoff_hz=5.0,
    )
    target = np.ones(3, dtype=np.float32)

    first = action_filter.update(target)
    outputs = [first]
    outputs.extend(action_filter.update(target) for _ in range(9))
    values = np.array([output[0] for output in outputs])

    np.testing.assert_allclose(first, action_filter.alpha * target, rtol=1.0e-6)
    assert np.all(np.diff(values) > 0.0)
    assert values[-1] < 1.0
    assert values[-1] == pytest.approx(
        1.0 - (1.0 - action_filter.alpha) ** 10,
        rel=1.0e-6,
    )


def test_filter_attenuates_alternating_high_frequency_action():
    action_filter = FirstOrderLowPassFilter(
        size=1,
        control_dt=0.02,
        cutoff_hz=5.0,
    )

    outputs = []
    for step in range(300):
        raw = np.array([1.0 if step % 2 == 0 else -1.0], dtype=np.float32)
        outputs.append(float(action_filter.update(raw)[0]))

    assert max(abs(value) for value in outputs[-50:]) < 0.31


def test_disabled_filter_is_an_exact_passthrough():
    action_filter = FirstOrderLowPassFilter(
        size=3,
        control_dt=0.02,
        cutoff_hz=5.0,
        enabled=False,
    )
    raw = np.array([0.3, -0.2, 1.1], dtype=np.float32)

    filtered = action_filter.update(raw)

    np.testing.assert_array_equal(filtered, raw)
    np.testing.assert_array_equal(action_filter.state, raw)


@pytest.mark.parametrize("cutoff_hz", [0.0, -1.0, 25.0, 30.0])
def test_invalid_cutoff_is_rejected(cutoff_hz):
    with pytest.raises(ValueError, match="cutoff_hz"):
        compute_low_pass_alpha(control_dt=0.02, cutoff_hz=cutoff_hz)


def test_wrong_action_dimension_is_rejected():
    action_filter = FirstOrderLowPassFilter(
        size=15,
        control_dt=0.02,
        cutoff_hz=5.0,
    )

    with pytest.raises(ValueError, match=r"shape \(15,\)"):
        action_filter.update(np.zeros(14, dtype=np.float32))
