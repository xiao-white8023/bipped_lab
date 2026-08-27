from __future__ import annotations

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import pytest
import torch

from rsl_rl.utils import RecoveryAMPFeatureBuilder, RecoveryNPZMotionLoader
from rsl_rl.utils.recovery_amp_features import (
    BODY_ANG_VEL_DIM,
    BODY_ANG_VEL_SLICE,
    BODY_LIN_VEL_DIM,
    BODY_LIN_VEL_SLICE,
    BODY_ORI_6D_SLICE,
    BODY_ORI_DIM,
    BODY_POS_DIM,
    BODY_POS_SLICE,
    FEATURE_DIM,
    G1_RECOVERY_AMP_BODY_NAMES,
    matrix_from_quat_wxyz,
    quat_mul_wxyz,
)


MOTION_DIR = (
    REPO_ROOT
    / "legged_lab"
    / "envs"
    / "g1"
    / "datasets"
    / "npz"
    / "recovery_candidates_v3"
)


@pytest.fixture(scope="module")
def loader():
    return RecoveryNPZMotionLoader(motion_dir=MOTION_DIR, device="cpu")


@pytest.fixture(scope="module")
def builder(loader):
    return RecoveryAMPFeatureBuilder(loader.body_names)


def _sample_state(loader, batch_size=64):
    torch.manual_seed(3201)
    return loader.sample_state_batch(batch_size)


def test_real_npz_feature_shape_layout_and_finiteness(loader, builder):
    feature = builder.build_from_state(_sample_state(loader, batch_size=512))

    assert len(G1_RECOVERY_AMP_BODY_NAMES) == 13
    assert builder.num_target_bodies == 13
    assert builder.feature_dim == FEATURE_DIM == 195
    assert BODY_POS_DIM == 39
    assert BODY_ORI_DIM == 78
    assert BODY_LIN_VEL_DIM == 39
    assert BODY_ANG_VEL_DIM == 39
    assert feature.shape == (512, 195)
    assert feature[:, BODY_POS_SLICE].shape == (512, 39)
    assert feature[:, BODY_ORI_6D_SLICE].shape == (512, 78)
    assert feature[:, BODY_LIN_VEL_SLICE].shape == (512, 39)
    assert feature[:, BODY_ANG_VEL_SLICE].shape == (512, 39)
    assert torch.isfinite(feature).all()


def test_quaternion_sign_invariance(loader, builder):
    state = _sample_state(loader)
    sign_flipped_state = dict(state)
    sign_flipped_state["body_quat_w"] = -state["body_quat_w"]

    torch.testing.assert_close(
        builder.build_from_state(sign_flipped_state),
        builder.build_from_state(state),
        atol=2e-6,
        rtol=2e-6,
    )


def test_global_translation_invariance(loader, builder):
    state = _sample_state(loader)
    translated_state = dict(state)
    translation = torch.tensor([5.0, -3.0, 2.0], dtype=state["body_pos_w"].dtype)
    translated_state["body_pos_w"] = state["body_pos_w"] + translation

    torch.testing.assert_close(
        builder.build_from_state(translated_state),
        builder.build_from_state(state),
        atol=3e-6,
        rtol=3e-6,
    )


def test_global_rotation_invariance(loader, builder):
    state = _sample_state(loader)
    dtype = state["body_quat_w"].dtype
    half_angle = torch.tensor(math.pi / 4.0, dtype=dtype)
    world_rotation = torch.stack(
        (
            torch.cos(half_angle),
            torch.zeros((), dtype=dtype),
            torch.zeros((), dtype=dtype),
            torch.sin(half_angle),
        )
    )
    rotation_matrix = matrix_from_quat_wxyz(world_rotation)

    rotated_state = dict(state)
    for field in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        rotated_state[field] = torch.matmul(state[field], rotation_matrix.transpose(-1, -2))
    rotated_state["body_quat_w"] = quat_mul_wxyz(
        world_rotation, state["body_quat_w"]
    )

    torch.testing.assert_close(
        builder.build_from_state(rotated_state),
        builder.build_from_state(state),
        atol=5e-6,
        rtol=5e-6,
    )


def test_velocity_uses_each_target_body_local_frame(loader, builder):
    num_bodies = len(loader.body_names)
    body_pos_w = torch.zeros((1, num_bodies, 3))
    body_quat_w = torch.zeros((1, num_bodies, 4))
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = torch.zeros((1, num_bodies, 3))
    body_ang_vel_w = torch.zeros((1, num_bodies, 3))

    target_index = loader.body_index(G1_RECOVERY_AMP_BODY_NAMES[0])
    half_angle = math.pi / 4.0
    body_quat_w[0, target_index] = torch.tensor(
        [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    )
    body_lin_vel_w[0, target_index] = torch.tensor([1.0, 0.0, 0.0])
    body_ang_vel_w[0, target_index] = torch.tensor([1.0, 0.0, 0.0])

    feature = builder.build(
        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
    )
    expected_local_velocity = torch.tensor([[0.0, -1.0, 0.0]])
    expected_rotation_6d = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]])
    actual_orientation = feature[
        :, BODY_ORI_6D_SLICE.start : BODY_ORI_6D_SLICE.start + 6
    ]
    actual_linear = feature[:, BODY_LIN_VEL_SLICE.start : BODY_LIN_VEL_SLICE.start + 3]
    actual_angular = feature[:, BODY_ANG_VEL_SLICE.start : BODY_ANG_VEL_SLICE.start + 3]
    torch.testing.assert_close(actual_orientation, expected_rotation_6d, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_linear, expected_local_velocity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_angular, expected_local_velocity, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual_linear, torch.tensor([[1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("transition_dt", [0.01, 0.02])
def test_real_npz_transition_features(loader, builder, transition_dt):
    state, next_state = loader.sample_transition_batch(512, transition_dt=transition_dt)
    amp_state, amp_next_state = builder.build_transition(state, next_state)

    assert amp_state.shape == (512, 195)
    assert amp_next_state.shape == (512, 195)
    assert torch.isfinite(amp_state).all()
    assert torch.isfinite(amp_next_state).all()
