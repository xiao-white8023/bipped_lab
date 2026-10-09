"""Tests for the deployed G1 RENet D435i observation preprocessing."""

from g1_depth.depth_policy import (
    build_policy_projection,
    decode_depth_image,
    gaussian_blur_3x3_sigma1,
    PinholeIntrinsics,
    PolicyDepthHistory,
    preprocess_policy_depth,
)

import numpy as np
import pytest


def training_intrinsics():
    """Return intrinsics whose pixel centers are the 64x36 training rays."""
    horizontal_half = np.tan(np.deg2rad(89.51) / 2.0)
    vertical_half = np.tan(np.deg2rad(58.29) / 2.0)
    return PinholeIntrinsics(
        width=64,
        height=36,
        fx=64.0 / (2.0 * horizontal_half),
        fy=36.0 / (2.0 * vertical_half),
        cx=31.5,
        cy=17.5,
    )


def test_projection_reproduces_exact_training_crop_pixel_centres():
    """The target grid must select the same pixel centers as simulator crop."""
    projection = build_policy_projection(training_intrinsics())

    expected_u, expected_v = np.meshgrid(
        np.arange(16, 48), np.arange(18, 36)
    )
    np.testing.assert_allclose(projection.source_u, expected_u, atol=1.0e-12)
    np.testing.assert_allclose(projection.source_v, expected_v, atol=1.0e-12)
    assert (projection.output_height, projection.output_width) == (18, 32)
    assert projection.coverage == 1.0


def test_decode_16uc1_honours_row_padding_and_scale():
    """Padded uint16 images must decode from millimetres without row drift."""
    first_row = np.array([100, 250, 999], dtype='<u2').tobytes()
    second_row = np.array([1000, 2500, 888], dtype='<u2').tobytes()

    decoded = decode_depth_image(
        first_row + second_row,
        height=2,
        width=2,
        step=6,
        encoding='16UC1',
        depth_unit_scale=0.001,
    )

    np.testing.assert_allclose(decoded, [[0.1, 0.25], [1.0, 2.5]])


def test_decode_big_endian_32fc1():
    """Float depth decoding must honour the ROS image byte-order flag."""
    source = np.array([[0.5, 1.25]], dtype='>f4')

    decoded = decode_depth_image(
        source.tobytes(),
        height=1,
        width=2,
        step=8,
        encoding='32FC1',
        is_bigendian=True,
    )

    np.testing.assert_allclose(decoded, [[0.5, 1.25]])


def test_measured_zero_follows_the_training_blur_then_clip_path():
    """A D435i hole is a finite 0 m, not inf/nan.

    The training pipeline only rewrites inf/nan before its blur, so a measured
    zero must reach the blur unchanged and be turned into the near plane by the
    [0.1, 2.5] clip.
    """
    projection = build_policy_projection(training_intrinsics())
    source = np.zeros((36, 64), dtype=np.float32)

    normalized, stats = preprocess_policy_depth(source, projection)

    np.testing.assert_allclose(normalized, 0.1 / 2.5, atol=2.0e-7)
    assert stats.valid_ratio == 0.0
    assert stats.projection_coverage == 1.0
    assert stats.minimum_metres == np.float32(0.1)
    assert stats.maximum_metres == np.float32(0.1)


def test_non_finite_depth_is_filled_with_the_far_plane():
    """inf/nan is the only invalid value the training pipeline rewrites."""
    projection = build_policy_projection(training_intrinsics())
    source = np.full((36, 64), np.inf, dtype=np.float32)
    source[0, 0] = np.nan

    normalized, stats = preprocess_policy_depth(source, projection)

    np.testing.assert_allclose(normalized, 1.0, atol=2.0e-7)
    assert stats.valid_ratio == 0.0


def test_legacy_max_mode_still_pushes_every_invalid_sample_to_the_far_plane():
    """invalid_depth_mode='max' preserves the pre-training-parity behaviour."""
    projection = build_policy_projection(training_intrinsics())
    source = np.zeros((36, 64), dtype=np.float32)

    normalized, stats = preprocess_policy_depth(
        source, projection, invalid_depth_mode='max'
    )

    np.testing.assert_allclose(normalized, 1.0, atol=2.0e-7)
    assert stats.valid_ratio == 0.0


def test_training_mode_keeps_a_single_hole_local_after_the_blur():
    """One measured zero may only darken its own 3x3 neighbourhood."""
    projection = build_policy_projection(training_intrinsics())
    source = np.full((36, 64), 1.0, dtype=np.float32)
    source[27, 32] = 0.0

    normalized, stats = preprocess_policy_depth(source, projection)
    expected = np.clip(
        gaussian_blur_3x3_sigma1(source[18:36, 16:48]), 0.1, 2.5
    ) / 2.5

    np.testing.assert_allclose(normalized, expected, rtol=1.0e-6, atol=1.0e-7)
    # row 27 - crop_top 18, column 32 - crop_left 16
    assert normalized[9, 16] < 1.0
    assert abs(float(normalized[9, 16]) - 1.0) > 0.1
    assert stats.valid_ratio < 1.0
    assert stats.valid_ratio > 0.99


def test_blur_disabled_feeds_the_projected_frame_directly():
    """blur_enable=False must skip the training blur but keep clip + normalize."""
    projection = build_policy_projection(training_intrinsics())
    source = np.linspace(0.1, 2.5, 36 * 64, dtype=np.float32).reshape(36, 64)

    blurred, _ = preprocess_policy_depth(source, projection)
    raw, stats = preprocess_policy_depth(source, projection, blur_enable=False)
    expected = np.clip(source[18:36, 16:48], 0.1, 2.5) / 2.5

    np.testing.assert_allclose(raw, expected, rtol=1.0e-6, atol=1.0e-7)
    assert stats.valid_ratio == 1.0
    # The unblurred frame keeps a wider value range than the blurred one.
    assert float(raw.std()) > float(blurred.std())


def test_blur_disabled_still_clips_and_normalizes_holes():
    """Measured zeros must still land on the near plane without the blur."""
    projection = build_policy_projection(training_intrinsics())
    source = np.zeros((36, 64), dtype=np.float32)

    normalized, stats = preprocess_policy_depth(
        source, projection, blur_enable=False
    )

    np.testing.assert_allclose(normalized, 0.1 / 2.5, atol=2.0e-7)
    assert stats.valid_ratio == 0.0


def test_bilinear_training_mode_keeps_measured_zeros_in_the_mix():
    """The bilinear control path must apply the same invalid-depth contract."""
    projection = build_policy_projection(training_intrinsics())
    source = np.full((36, 64), 1.0, dtype=np.float32)
    source[27, 32] = 0.0

    training, _ = preprocess_policy_depth(
        source, projection, sampling_mode='bilinear'
    )
    legacy, _ = preprocess_policy_depth(
        source, projection, sampling_mode='bilinear', invalid_depth_mode='max'
    )

    # training: the zero only drags interpolated samples down, never above 0.4
    assert float(training.max()) <= 0.4 + 1.0e-6
    assert float(training.min()) < 0.4
    # max: the same zero becomes the far plane and pushes samples up instead
    assert float(legacy.min()) == np.float32(0.4)
    assert float(legacy.max()) > 0.4


def test_invalid_depth_mode_is_validated():
    """An unknown mode must fail loudly instead of silently picking a fill."""
    projection = build_policy_projection(training_intrinsics())
    source = np.zeros((36, 64), dtype=np.float32)

    with pytest.raises(ValueError):
        preprocess_policy_depth(source, projection, invalid_depth_mode='nearest')


def test_preprocessing_matches_crop_blur_clip_then_normalize():
    """Deployment must reproduce the deterministic training blur exactly."""
    projection = build_policy_projection(training_intrinsics())
    source = np.linspace(0.1, 2.5, 36 * 64, dtype=np.float32).reshape(36, 64)

    normalized, stats = preprocess_policy_depth(source, projection)
    expected = gaussian_blur_3x3_sigma1(source[18:36, 16:48]) / 2.5

    np.testing.assert_allclose(normalized, expected, rtol=1.0e-6, atol=1.0e-7)
    assert normalized.shape == (18, 32)
    assert stats.valid_ratio == 1.0


def test_fixed_gaussian_blur_has_expected_sigma_one_kernel():
    """The deployment blur is deterministic and contains no random noise."""
    source = np.zeros((5, 5), dtype=np.float32)
    source[2, 2] = 1.0
    blurred = gaussian_blur_3x3_sigma1(source)
    one_d = np.exp(-0.5 * np.asarray([-1.0, 0.0, 1.0], dtype=np.float32) ** 2)
    one_d /= one_d.sum()
    expected = np.zeros((5, 5), dtype=np.float32)
    expected[1:4, 1:4] = np.outer(one_d, one_d)
    np.testing.assert_allclose(blurred, expected, rtol=1.0e-6, atol=1.0e-7)


def test_history_duplicates_first_frame_and_orders_oldest_to_newest():
    """Initialization and temporal flatten order must match the simulator."""
    history = PolicyDepthHistory(frame_count=2, capture_period=0.1)
    first = np.full((18, 32), 0.25, dtype=np.float32)
    second = np.full((18, 32), 0.75, dtype=np.float32)

    initialized, rewound = history.push(10.0, first)
    too_soon, _ = history.push(10.05, second)
    updated, _ = history.push(10.1, second)

    assert not rewound
    assert too_soon is None
    np.testing.assert_allclose(initialized[0], first)
    np.testing.assert_allclose(initialized[1], first)
    np.testing.assert_allclose(updated[0], first)
    np.testing.assert_allclose(updated[1], second)


def test_history_rewind_starts_a_new_two_frame_epoch():
    """Looped bags must not mix depth frames from different clock epochs."""
    history = PolicyDepthHistory(frame_count=2, capture_period=0.1)
    old = np.full((18, 32), 0.25, dtype=np.float32)
    new = np.full((18, 32), 0.9, dtype=np.float32)
    history.push(100.0, old)

    reset, rewound = history.push(50.0, new)

    assert rewound
    np.testing.assert_allclose(reset[0], new)
    np.testing.assert_allclose(reset[1], new)


def test_history_keeps_ten_hz_when_camera_runs_slightly_above_thirty_hz():
    """A 30.08 Hz camera must not turn the depth history into a 7.52 Hz stream."""
    history = PolicyDepthHistory(capture_period=0.1)
    frame = np.zeros((18, 32), dtype=np.float32)
    accepted = []
    for stamp in np.arange(0.0, 30.0, 1.0 / 30.08):
        due = history.is_due(stamp)
        output, _ = history.push(stamp, frame)
        assert due == (output is not None)
        if output is not None:
            accepted.append(stamp)

    rate = (len(accepted) - 1) / (accepted[-1] - accepted[0])
    assert 9.98 < rate < 10.02
    assert np.max(np.diff(accepted)) < 0.135


def test_history_skips_missed_deadlines_without_a_catch_up_burst():
    """After camera silence, use fresh frames without replaying missed ticks."""
    history = PolicyDepthHistory(capture_period=0.1)
    old = np.zeros((18, 32), dtype=np.float32)
    new = np.ones((18, 32), dtype=np.float32)
    history.push(10.0, old)
    resumed, rewound = history.push(12.035, new)
    assert not rewound
    np.testing.assert_array_equal(resumed[0], old)
    np.testing.assert_array_equal(resumed[1], new)
    assert not history.is_due(12.036)
    assert not history.is_due(12.099)
    assert history.is_due(12.1)


def test_history_rewind_resets_capture_deadline():
    """A new clock epoch must not wait on the previous epoch's deadline."""
    history = PolicyDepthHistory(capture_period=0.1)
    frame = np.zeros((18, 32), dtype=np.float32)
    history.push(100.0, frame)
    _, rewound = history.push(50.0, frame)
    assert rewound
    assert not history.is_due(50.05)
    assert history.is_due(50.1)
