#!/usr/bin/env python3
"""Run the standalone G1 Recovery policy in MuJoCo.

This script mirrors the IsaacLab ``g1_recovery`` deployment contract:

* G1 23-DoF model and 23-D whole-body action;
* 50 Hz policy over a 200 Hz MuJoCo simulation;
* actor observation ``[ang_vel_b, projected_gravity_b, joint_pos_error,
  joint_vel, last_action]`` (75 values) with 10-frame history;
* dataset-based fallen reset poses from ``recovery_reset_states_v1``;
* IsaacLab joint order at the policy boundary and MuJoCo joint order at the
  simulator boundary.

Run ``play_recovery.py`` once without ``--no_export`` to create the default
TorchScript policy used here.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml"
)
DEFAULT_RESET_DIR = (
    REPO_ROOT
    / "legged_lab/envs/g1/datasets/npz/recovery_reset_states_v1"
)


# MuJoCo XML order. This is also the actuator order in the official 23-DoF
# model, but addresses are still resolved by name below instead of assumed.
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


# IsaacLab order recorded by the Recovery dataset and seen by the actor.
ISAAC_JOINT_NAMES = (
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


NUM_JOINTS = 23
NUM_ACTIONS = 23
OBS_PER_STEP = 75
OBS_HISTORY_LENGTH = 10
POLICY_OBS_DIM = OBS_PER_STEP * OBS_HISTORY_LENGTH
PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0
CLIP_OBSERVATIONS = 100.0
SOFT_JOINT_LIMIT_FACTOR = 0.90
MAX_EPISODE_STEPS = 300
SUCCESS_HEIGHT = 0.85 * 0.793
SUCCESS_UPRIGHT = 0.93
SUCCESS_HOLD_STEPS = 25


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _decode_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got {array.shape}.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _string_tuple(value: np.ndarray, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    result = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    return tuple(result)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the quaternion's local frame."""
    quat = np.asarray(quat, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if quat.shape != (4,) or vector.shape != (3,):
        raise ValueError("quat/vector must have shapes (4,) and (3,).")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Quaternion must be finite and non-zero.")
    quat = quat / norm
    qw = quat[0]
    qvec = quat[1:]
    return (
        vector * (2.0 * qw * qw - 1.0)
        - 2.0 * qw * np.cross(qvec, vector)
        + 2.0 * qvec * np.dot(qvec, vector)
    )


def projected_gravity_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return IsaacLab's unit gravity vector expressed in the pelvis frame."""
    return quat_rotate_inverse_wxyz(
        quat,
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ).astype(np.float32)


def _default_joint_position(name: str) -> float:
    if "hip_pitch" in name:
        return -0.20
    if "knee" in name:
        return 0.42
    if "ankle_pitch" in name:
        return -0.23
    if "shoulder_pitch" in name:
        return 0.35
    if name == "left_shoulder_roll_joint":
        return 0.18
    if name == "right_shoulder_roll_joint":
        return -0.18
    if "elbow" in name:
        return 0.87
    return 0.0


def _joint_control_parameters(name: str) -> tuple[float, float, float]:
    """Return the IsaacLab (Kp, Kd, effort limit) for one joint."""
    if "hip_pitch" in name:
        return 200.0, 5.0, 88.0
    if "hip_roll" in name:
        return 150.0, 5.0, 139.0
    if "hip_yaw" in name:
        return 150.0, 5.0, 88.0
    if "knee" in name:
        return 200.0, 5.0, 139.0
    if "ankle" in name:
        return 20.0, 2.0, 35.0
    if name == "waist_yaw_joint":
        return 200.0, 5.0, 88.0
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return 100.0, 2.0, 25.0
    if "shoulder_yaw" in name or "elbow" in name:
        return 50.0, 2.0, 25.0
    if "wrist_roll" in name:
        return 15.0, 2.0, 25.0
    raise KeyError(f"No Recovery control parameters for joint {name!r}.")


@dataclass(frozen=True)
class RecoveryResetState:
    """One whole-body fallen state expressed in the IsaacLab convention."""

    path: Path
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


def discover_reset_states(reset_dir: Path) -> tuple[Path, ...]:
    """Return naturally sorted Recovery reset-state files."""
    reset_dir = reset_dir.expanduser().resolve()
    if not reset_dir.is_dir():
        raise NotADirectoryError(f"Recovery reset directory not found: {reset_dir}")
    state_files = tuple(sorted(reset_dir.glob("*.npz"), key=_natural_key))
    if not state_files:
        raise FileNotFoundError(f"No Recovery reset NPZ files found in {reset_dir}")
    return state_files


def load_reset_state(path: Path) -> RecoveryResetState:
    """Load one validated Stage-4 Recovery reset state without torch/IsaacLab."""
    required_keys = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
        "root_body_index",
        "root_body_name",
        "quaternion_convention",
        "velocity_frame",
        "body_state_frame",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise KeyError(f"{path.name}: missing required keys {missing}.")

        joint_names = _string_tuple(data["joint_names"], "joint_names")
        body_names = _string_tuple(data["body_names"], "body_names")
        if joint_names != ISAAC_JOINT_NAMES:
            raise ValueError(
                f"{path.name}: Recovery joint order does not match g1_recovery."
            )
        root_index = int(np.asarray(data["root_body_index"]).reshape(-1)[0])
        root_name = _decode_scalar(data["root_body_name"], "root_body_name")
        if root_name != "pelvis" or body_names[root_index] != "pelvis":
            raise ValueError(f"{path.name}: root body must be pelvis.")
        if _decode_scalar(
            data["quaternion_convention"], "quaternion_convention"
        ).upper() != "WXYZ":
            raise ValueError(f"{path.name}: quaternion convention must be WXYZ.")
        if _decode_scalar(data["velocity_frame"], "velocity_frame").lower() != "world":
            raise ValueError(f"{path.name}: reset velocities must be world-frame.")
        if _decode_scalar(
            data["body_state_frame"], "body_state_frame"
        ).lower() != "link_world":
            raise ValueError(f"{path.name}: body state frame must be link_world.")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isclose(fps, 50.0, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{path.name}: reset-state fps must be 50, got {fps}.")

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

    expected_shapes = {
        "joint_pos": (1, NUM_JOINTS),
        "joint_vel": (1, NUM_JOINTS),
        "body_pos_w": (1, len(body_names), 3),
        "body_quat_w": (1, len(body_names), 4),
        "body_lin_vel_w": (1, len(body_names), 3),
        "body_ang_vel_w": (1, len(body_names), 3),
    }
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{path.name}: {name} must have shape {expected_shapes[name]}, "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{path.name}: {name} contains NaN or Inf.")

    root_quat = body_quat[0, root_index].copy()
    root_quat_norm = np.linalg.norm(root_quat)
    if not np.isfinite(root_quat_norm) or root_quat_norm < 1.0e-12:
        raise ValueError(f"{path.name}: root quaternion is invalid.")
    root_quat /= root_quat_norm
    return RecoveryResetState(
        path=path,
        root_pos_w=body_pos[0, root_index].copy(),
        root_quat_w=root_quat,
        root_lin_vel_w=body_lin_vel[0, root_index].copy(),
        root_ang_vel_w=body_ang_vel[0, root_index].copy(),
        joint_pos=joint_pos[0].copy(),
        joint_vel=joint_vel[0].copy(),
    )


def find_latest_exported_policy() -> Path | None:
    """Find the newest exported policy below logs/g1_recovery."""
    candidates = sorted(
        (REPO_ROOT / "logs/g1_recovery").glob("*/exported/policy.pt")
    )
    return candidates[-1].resolve() if candidates else None


class G1RecoveryMujocoRunner:
    """Execute an exported G1 Recovery actor against the 23-DoF MuJoCo model."""

    def __init__(
        self,
        policy_path: Path,
        model_path: Path,
        reset_files: tuple[Path, ...],
        state_index: int | None,
        random_states: bool,
        zero_initial_velocity: bool,
        seed: int,
    ) -> None:
        """Load and validate the policy, model, mappings, and reset dataset."""
        self.policy_path = policy_path.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"TorchScript policy not found: {self.policy_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        print(f"[INFO] Loading MuJoCo model: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        print(f"[INFO] Loading TorchScript policy: {self.policy_path}")
        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.policy.eval()

        self.reset_files = reset_files
        self.fixed_state_index = state_index
        self.random_states = random_states
        self.zero_initial_velocity = zero_initial_velocity
        self.rng = np.random.default_rng(seed)

        self._build_model_mappings()
        self._build_control_parameters()

        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.obs_history = np.zeros(POLICY_OBS_DIM, dtype=np.float32)
        self.episode_step = 0
        self.success_hold_counter = 0
        self.current_state_index = -1

        self._validate_policy_contract()
        self.reset_episode()

        print(f"[INFO] Control rate: {1.0 / POLICY_DT:.1f} Hz")
        print(
            f"[INFO] Actor observation: {OBS_PER_STEP} x "
            f"{OBS_HISTORY_LENGTH} = {POLICY_OBS_DIM}"
        )
        print(f"[INFO] Whole-body action dimension: {NUM_ACTIONS}")

    def _build_model_mappings(self) -> None:
        free_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        if free_joint_id < 0 or self.model.jnt_type[free_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("MuJoCo model is missing floating_base_joint.")
        self.root_qpos_addr = int(self.model.jnt_qposadr[free_joint_id])
        self.root_qvel_addr = int(self.model.jnt_dofadr[free_joint_id])

        actual_joint_names = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            actual_joint_names.append(
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )
            )
        if tuple(actual_joint_names) != MUJOCO_JOINT_NAMES:
            raise RuntimeError(
                "MuJoCo 23-DoF joint order mismatch.\n"
                f"Expected: {MUJOCO_JOINT_NAMES}\n"
                f"Actual:   {tuple(actual_joint_names)}"
            )

        qpos_addr = []
        qvel_addr = []
        actuator_ids = []
        joint_ranges = []
        for name in MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if joint_id < 0 or actuator_id < 0:
                raise RuntimeError(
                    f"MuJoCo model is missing joint/actuator {name!r}."
                )
            qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_addr.append(int(self.model.jnt_dofadr[joint_id]))
            actuator_ids.append(actuator_id)
            joint_ranges.append(self.model.jnt_range[joint_id].copy())

        self.qpos_addr_mj = np.asarray(qpos_addr, dtype=np.int32)
        self.qvel_addr_mj = np.asarray(qvel_addr, dtype=np.int32)
        self.actuator_ids_mj = np.asarray(actuator_ids, dtype=np.int32)

        # G1_23CFG assigns 0.01 armature to every implicit actuator. The raw
        # MuJoCo XML does not carry that IsaacLab actuator setting, so apply it
        # explicitly; this is especially important for contact-rich fallen
        # reset poses where zero-armature joints can produce huge QACC spikes.
        self.model.dof_armature[self.qvel_addr_mj] = 0.01

        mj_index = {name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)}
        isaac_index = {name: index for index, name in enumerate(ISAAC_JOINT_NAMES)}
        self.mj_to_isaac_idx = np.asarray(
            [mj_index[name] for name in ISAAC_JOINT_NAMES], dtype=np.int32
        )
        self.isaac_to_mj_idx = np.asarray(
            [isaac_index[name] for name in MUJOCO_JOINT_NAMES], dtype=np.int32
        )

        joint_ranges_mj = np.asarray(joint_ranges, dtype=np.float64)
        joint_mid = joint_ranges_mj.mean(axis=1)
        joint_half_range = 0.5 * np.diff(joint_ranges_mj, axis=1).reshape(-1)
        self.soft_joint_lower_mj = (
            joint_mid - SOFT_JOINT_LIMIT_FACTOR * joint_half_range
        )
        self.soft_joint_upper_mj = (
            joint_mid + SOFT_JOINT_LIMIT_FACTOR * joint_half_range
        )

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if self.torso_body_id < 0:
            raise RuntimeError("MuJoCo model is missing torso_link.")

    def _build_control_parameters(self) -> None:
        self.default_joint_pos_mj = np.asarray(
            [_default_joint_position(name) for name in MUJOCO_JOINT_NAMES],
            dtype=np.float64,
        )
        self.default_joint_pos_isaac = self.default_joint_pos_mj[
            self.mj_to_isaac_idx
        ]
        parameters = np.asarray(
            [_joint_control_parameters(name) for name in MUJOCO_JOINT_NAMES],
            dtype=np.float64,
        )
        self.kp_mj = parameters[:, 0]
        self.kd_mj = parameters[:, 1]
        self.effort_limit_mj = parameters[:, 2]

    def _validate_policy_contract(self) -> None:
        dummy_obs = torch.zeros(1, POLICY_OBS_DIM, dtype=torch.float32)
        with torch.inference_mode():
            dummy_action = self.policy(dummy_obs)
        if tuple(dummy_action.shape) != (1, NUM_ACTIONS):
            raise RuntimeError(
                "Recovery TorchScript policy contract mismatch: expected "
                f"input [1, {POLICY_OBS_DIM}] -> output [1, {NUM_ACTIONS}], "
                f"got output {tuple(dummy_action.shape)}."
            )
        if not torch.isfinite(dummy_action).all():
            raise RuntimeError("Recovery policy produced NaN/Inf for zero input.")

    def mj_to_isaac(self, values_mj: np.ndarray) -> np.ndarray:
        """Reorder a 23-D vector from MuJoCo order to IsaacLab order."""
        return values_mj[self.mj_to_isaac_idx]

    def isaac_to_mj(self, values_isaac: np.ndarray) -> np.ndarray:
        """Reorder a 23-D vector from IsaacLab order to MuJoCo order."""
        return values_isaac[self.isaac_to_mj_idx]

    def _select_state_index(self) -> int:
        if self.random_states:
            return int(self.rng.integers(len(self.reset_files)))
        if self.fixed_state_index is not None:
            return self.fixed_state_index
        return 0

    def reset_episode(self) -> np.ndarray:
        """Reset MuJoCo from an NPZ state and return a filled actor history."""
        self.current_state_index = self._select_state_index()
        state = load_reset_state(self.reset_files[self.current_state_index])

        mujoco.mj_resetData(self.model, self.data)

        root_qpos = self.root_qpos_addr
        root_qvel = self.root_qvel_addr
        root_pos = state.root_pos_w.copy()
        # Match G1RecoveryEnv: preserve source height, place X/Y at env origin.
        root_pos[:2] = 0.0
        self.data.qpos[root_qpos : root_qpos + 3] = root_pos
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = state.root_quat_w

        if not self.zero_initial_velocity:
            self.data.qvel[root_qvel : root_qvel + 3] = state.root_lin_vel_w
            self.data.qvel[root_qvel + 3 : root_qvel + 6] = (
                quat_rotate_inverse_wxyz(
                    state.root_quat_w,
                    state.root_ang_vel_w,
                )
            )
            self.data.qvel[self.qvel_addr_mj] = self.isaac_to_mj(
                state.joint_vel
            )

        self.data.qpos[self.qpos_addr_mj] = self.isaac_to_mj(state.joint_pos)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.action.fill(0.0)
        self.episode_step = 0
        self.success_hold_counter = 0
        current_obs = self._current_observation()
        self.obs_history = np.tile(current_obs, OBS_HISTORY_LENGTH).astype(
            np.float32
        )

        print(
            f"[INFO] Reset state {self.current_state_index + 1}/"
            f"{len(self.reset_files)}: {state.path.name}, "
            f"pelvis_h={self.data.qpos[root_qpos + 2]:.3f}"
        )
        return self.obs_history.copy()

    def _current_observation(self) -> np.ndarray:
        joint_pos_mj = self.data.qpos[self.qpos_addr_mj].copy()
        joint_vel_mj = self.data.qvel[self.qvel_addr_mj].copy()
        root_quat = self.data.qpos[
            self.root_qpos_addr + 3 : self.root_qpos_addr + 7
        ].copy()
        ang_vel_body = self.data.qvel[
            self.root_qvel_addr + 3 : self.root_qvel_addr + 6
        ].copy()

        observation = np.concatenate(
            (
                ang_vel_body,
                projected_gravity_from_quat_wxyz(root_quat),
                self.mj_to_isaac(joint_pos_mj)
                - self.default_joint_pos_isaac,
                self.mj_to_isaac(joint_vel_mj),
                self.action,
            )
        ).astype(np.float32)
        if observation.shape != (OBS_PER_STEP,):
            raise RuntimeError(
                f"Recovery observation must have {OBS_PER_STEP} values, "
                f"got {observation.shape}."
            )
        if not np.isfinite(observation).all():
            raise RuntimeError("Recovery observation contains NaN or Inf.")
        return observation

    def append_observation(self) -> np.ndarray:
        """Append the current 75-D state and return the clipped history."""
        self.obs_history = np.roll(self.obs_history, -OBS_PER_STEP)
        self.obs_history[-OBS_PER_STEP:] = self._current_observation()
        return np.clip(
            self.obs_history,
            -CLIP_OBSERVATIONS,
            CLIP_OBSERVATIONS,
        ).astype(np.float32)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run one actor inference and store the clipped Isaac-order action."""
        obs_tensor = torch.from_numpy(observation).unsqueeze(0)
        with torch.inference_mode():
            action_tensor = self.policy(obs_tensor)
        action = action_tensor.detach().cpu().numpy().reshape(-1)
        if action.shape != (NUM_ACTIONS,) or not np.isfinite(action).all():
            raise RuntimeError(
                "Recovery policy must return one finite 23-D action, got "
                f"shape={action.shape}."
            )
        self.action = np.clip(
            action,
            -CLIP_ACTIONS,
            CLIP_ACTIONS,
        ).astype(np.float32)
        return self.action

    def position_target_mj(self) -> np.ndarray:
        """Convert the current actor action into a soft-limit-safe target."""
        target_isaac = (
            self.default_joint_pos_isaac
            + ACTION_SCALE * self.action.astype(np.float64)
        )
        target_mj = self.isaac_to_mj(target_isaac)
        return np.clip(
            target_mj,
            self.soft_joint_lower_mj,
            self.soft_joint_upper_mj,
        )

    def apply_pd_control(self) -> None:
        """Apply the IsaacLab-equivalent saturated PD torque in MuJoCo."""
        target_mj = self.position_target_mj()
        joint_pos_mj = self.data.qpos[self.qpos_addr_mj]
        joint_vel_mj = self.data.qvel[self.qvel_addr_mj]
        torque_mj = (
            self.kp_mj * (target_mj - joint_pos_mj)
            - self.kd_mj * joint_vel_mj
        )
        torque_mj = np.clip(
            torque_mj,
            -self.effort_limit_mj,
            self.effort_limit_mj,
        )
        self.data.ctrl[self.actuator_ids_mj] = torque_mj

    def recovery_status(self) -> tuple[bool, bool, float, float]:
        """Update and return success/timeout state and pose diagnostics."""
        pelvis_height = float(self.data.qpos[self.root_qpos_addr + 2])
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_up_z = float(torso_rotation[2, 2])
        ready = (
            pelvis_height >= SUCCESS_HEIGHT
            and torso_up_z >= SUCCESS_UPRIGHT
        )
        self.success_hold_counter = (
            self.success_hold_counter + 1 if ready else 0
        )
        success = self.success_hold_counter >= SUCCESS_HOLD_STEPS
        timeout = self.episode_step >= MAX_EPISODE_STEPS and not success
        return success, timeout, pelvis_height, torso_up_z

    def run(
        self,
        duration_s: float,
        loop_episodes: bool,
        headless: bool,
        realtime: bool,
        print_every: int,
    ) -> None:
        """Run policy and physics steps until an outcome or duration limit."""
        observation = self.obs_history.copy()
        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.lookat[:] = self.data.qpos[
                self.root_qpos_addr : self.root_qpos_addr + 3
            ]
            viewer.cam.distance = 2.5
            viewer.sync()

        total_policy_steps = 0
        total_physics_steps = 0
        try:
            while total_physics_steps * PHYSICS_DT < duration_s:
                if viewer is not None and not viewer.is_running():
                    break

                self.infer(observation)
                for _ in range(DECIMATION):
                    step_start = time.perf_counter()
                    self.apply_pd_control()
                    mujoco.mj_step(self.model, self.data)
                    total_physics_steps += 1
                    if viewer is not None:
                        viewer.sync()
                    if realtime:
                        sleep_s = PHYSICS_DT - (time.perf_counter() - step_start)
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)

                self.episode_step += 1
                total_policy_steps += 1
                success, timeout, pelvis_h, torso_up_z = self.recovery_status()
                observation = self.append_observation()

                if print_every and total_policy_steps % print_every == 0:
                    print(
                        f"[INFO] t={total_physics_steps * PHYSICS_DT:.2f}s, "
                        f"episode_step={self.episode_step}, "
                        f"pelvis_h={pelvis_h:.3f}, "
                        f"torso_up_z={torso_up_z:.3f}, "
                        f"hold={self.success_hold_counter}/{SUCCESS_HOLD_STEPS}"
                    )

                if success or timeout:
                    outcome = "SUCCESS" if success else "TIMEOUT"
                    print(
                        f"[INFO] Recovery {outcome}: "
                        f"state={self.current_state_index + 1}, "
                        f"duration={self.episode_step * POLICY_DT:.2f}s, "
                        f"pelvis_h={pelvis_h:.3f}, torso_up_z={torso_up_z:.3f}"
                    )
                    if not loop_episodes:
                        break
                    observation = self.reset_episode()
                    if viewer is not None:
                        viewer.sync()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            if viewer is not None:
                viewer.close()
        print("[INFO] MuJoCo Recovery simulation finished.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G1 Recovery IsaacLab-to-MuJoCo policy playback."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=(
            "Exported TorchScript policy. By default, use the newest "
            "logs/g1_recovery/*/exported/policy.pt."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="23-DoF G1 MuJoCo XML scene.",
    )
    parser.add_argument(
        "--reset_dir",
        type=Path,
        default=DEFAULT_RESET_DIR,
        help="Directory containing Recovery reset_state_*.npz files.",
    )
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--state",
        type=int,
        default=None,
        help="1-based reset-state index. The default is state 1.",
    )
    state_group.add_argument(
        "--random_state",
        action="store_true",
        help="Sample a random reset state at the start of every episode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random-state seed.")
    parser.add_argument(
        "--zero_initial_velocity",
        action="store_true",
        help="Keep the NPZ pose but zero root and joint reset velocities.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum total simulation duration in seconds.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reset and continue after Recovery success or the 6 s timeout.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--no_realtime",
        action="store_true",
        help="Do not sleep to match wall-clock simulation time.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Print status every N policy steps; 0 disables periodic output.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate model, reset state, and policy I/O without stepping physics.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run or validate the MuJoCo deployment."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.state is not None and args.state <= 0:
        parser.error("--state is 1-based and must be positive.")
    if args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.print_every < 0:
        parser.error("--print_every must be non-negative.")

    policy_path = (
        args.policy.expanduser().resolve()
        if args.policy is not None
        else find_latest_exported_policy()
    )
    if policy_path is None:
        parser.error(
            "No exported g1_recovery policy was found. Run play_recovery.py "
            "once without --no_export, or pass --policy explicitly."
        )

    reset_files = discover_reset_states(args.reset_dir)
    state_index = args.state - 1 if args.state is not None else None
    if state_index is not None and state_index >= len(reset_files):
        parser.error(
            f"--state {args.state} requested, but {len(reset_files)} reset "
            "states are available."
        )

    runner = G1RecoveryMujocoRunner(
        policy_path=policy_path,
        model_path=args.model,
        reset_files=reset_files,
        state_index=state_index,
        random_states=args.random_state,
        zero_initial_velocity=args.zero_initial_velocity,
        seed=args.seed,
    )
    if args.validate_only:
        action = runner.infer(runner.obs_history.copy())
        print(
            "[SUCCESS] Recovery sim2sim contract validated: "
            f"obs={runner.obs_history.shape}, action={action.shape}, "
            f"state={runner.current_state_index + 1}."
        )
        return

    runner.run(
        duration_s=args.duration,
        loop_episodes=args.loop,
        headless=args.headless,
        realtime=not args.no_realtime,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
