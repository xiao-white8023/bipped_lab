"""Frozen dimensions, joint order, and gains for the 23-DOF G1 RENet policy."""

from __future__ import annotations

import numpy as np


JOINT_COUNT = 23
PROPRIO_FRAME_DIM = 78
PROPRIO_HISTORY_FRAMES = 10
PROPRIO_HISTORY_DIM = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_FRAMES
ELEVATION_DIM = 11 * 17
DEPTH_HISTORY_DIM = 2 * 18 * 32
POLICY_INPUT_DIM = PROPRIO_HISTORY_DIM + ELEVATION_DIM + DEPTH_HISTORY_DIM + 1
ACTION_SCALE = 0.25
CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0

# Exact Isaac Lab order used by g1_renet, not Unitree motor order.
POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
)

# policy[i] reads/writes Unitree LowState/LowCmd motor index POLICY_TO_MOTOR[i].
# The unused 23-DOF slots are 13, 14, 20, 21, 27, 28 (and 29..34).
POLICY_TO_MOTOR = np.asarray(
    [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26],
    dtype=np.int64,
)
ACTIVE_MOTOR_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26],
    dtype=np.int64,
)

# sim2sim_renet.py stores these in compact Unitree/MuJoCo order.
_DEFAULT_Q_MOTOR_COMPACT = np.asarray(
    [
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        0.0,
        0.35, 0.18, 0.0, 0.87, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0,
    ],
    dtype=np.float32,
)
_KP_MOTOR_COMPACT = np.asarray(
    [
        200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
        200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
        200.0,
        100.0, 100.0, 50.0, 50.0, 15.0,
        100.0, 100.0, 50.0, 50.0, 15.0,
    ],
    dtype=np.float32,
)
_KD_MOTOR_COMPACT = np.asarray(
    [
        5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
        5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
        5.0,
        2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0,
    ],
    dtype=np.float32,
)


def _compact_motor_to_full(values: np.ndarray) -> np.ndarray:
    full = np.zeros(35, dtype=np.float32)
    full[ACTIVE_MOTOR_INDICES] = values
    return full


DEFAULT_Q_MOTOR = _compact_motor_to_full(_DEFAULT_Q_MOTOR_COMPACT)
KP_MOTOR = _compact_motor_to_full(_KP_MOTOR_COMPACT)
KD_MOTOR = _compact_motor_to_full(_KD_MOTOR_COMPACT)
DEFAULT_Q_POLICY = DEFAULT_Q_MOTOR[POLICY_TO_MOTOR].copy()


def policy_to_motor(values: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    """Map one 23-value policy-order vector into the 35 Unitree motor slots."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape != (JOINT_COUNT,):
        raise ValueError(f"expected {(JOINT_COUNT,)}, got {array.shape}")
    result = np.full(35, fill, dtype=np.float32)
    result[POLICY_TO_MOTOR] = array
    return result


def motor_to_policy(values: np.ndarray) -> np.ndarray:
    """Read the controlled joints from a 35-value Unitree motor vector."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape != (35,):
        raise ValueError(f"expected {(35,)}, got {array.shape}")
    return array[POLICY_TO_MOTOR].copy()


if len(POLICY_JOINT_NAMES) != JOINT_COUNT:
    raise RuntimeError("policy joint-name contract must contain 23 entries")
if len(set(POLICY_TO_MOTOR.tolist())) != JOINT_COUNT:
    raise RuntimeError("policy-to-motor map must be one-to-one")
if not np.array_equal(np.sort(POLICY_TO_MOTOR), ACTIVE_MOTOR_INDICES):
    raise RuntimeError("policy-to-motor map does not cover the 23 active motors")
