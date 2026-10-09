"""Build the exact 10 x 78 proprioception block consumed by g1_renet."""

from __future__ import annotations

import numpy as np

from .contracts import (
    CLIP_ACTIONS,
    CLIP_OBSERVATIONS,
    DEFAULT_Q_POLICY,
    JOINT_COUNT,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_DIM,
    PROPRIO_HISTORY_FRAMES,
)


def projected_gravity_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Match Isaac Lab projected_gravity_b for Unitree [w, x, y, z]."""

    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    qw, qx, qy, qz = quat
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def build_proprio_frame(
    angular_velocity: np.ndarray,
    projected_gravity: np.ndarray,
    command: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Assemble [ang_vel, gravity, command, q-q0, dq, previous_action]."""

    angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
    projected_gravity = np.asarray(projected_gravity, dtype=np.float32)
    command = np.asarray(command, dtype=np.float32)
    joint_position = np.asarray(joint_position, dtype=np.float32)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    for name, value, shape in (
        ("angular_velocity", angular_velocity, (3,)),
        ("projected_gravity", projected_gravity, (3,)),
        ("command", command, (3,)),
        ("joint_position", joint_position, (JOINT_COUNT,)),
        ("joint_velocity", joint_velocity, (JOINT_COUNT,)),
        ("previous_action", previous_action, (JOINT_COUNT,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    frame = np.concatenate(
        (
            angular_velocity,
            projected_gravity,
            command,
            joint_position - DEFAULT_Q_POLICY,
            joint_velocity,
            np.clip(previous_action, -CLIP_ACTIONS, CLIP_ACTIONS),
        )
    ).astype(np.float32, copy=False)
    if frame.shape != (PROPRIO_FRAME_DIM,):
        raise RuntimeError(f"proprio frame has unexpected shape {frame.shape}")
    return frame


class ProprioHistory:
    """Oldest-to-newest history with reset behavior matching sim2sim_renet."""

    def __init__(self) -> None:
        self._frames = np.zeros(
            (PROPRIO_HISTORY_FRAMES, PROPRIO_FRAME_DIM), dtype=np.float32
        )
        self._initialized = False

    def reset(self) -> None:
        self._frames.fill(0.0)
        self._initialized = False

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (PROPRIO_FRAME_DIM,):
            raise ValueError(f"frame must have shape {(PROPRIO_FRAME_DIM,)}, got {frame.shape}")
        if not self._initialized:
            self._frames[:] = frame
            self._initialized = True
        else:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        result = np.clip(
            self._frames.reshape(PROPRIO_HISTORY_DIM),
            -CLIP_OBSERVATIONS,
            CLIP_OBSERVATIONS,
        )
        return result.astype(np.float32, copy=False)
