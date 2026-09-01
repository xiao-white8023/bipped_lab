#!/usr/bin/env python3
"""Run the standalone G1 Recovery policy in MuJoCo.

The implementation follows the structure of ``sim2sim_rough.py`` while
preserving the standalone Recovery deployment contract:

* 23 whole-body actions in IsaacLab joint order;
* 75 proprioceptive values per policy step with a 10-step history;
* a 50 Hz policy controlling a 200 Hz MuJoCo simulation;
* IsaacLab-equivalent action scaling, soft joint limits, PD gains, armature,
  and effort limits;
* an upright initialization and stabilization phase followed by a standing
  standby mode; viewer perturbation releases the root and activates Recovery.

After stabilization, use the MuJoCo viewer perturbation controls to push or
drag the robot. The policy then runs continuously without automatic success,
timeout, or reset handling. Run ``play_recovery.py`` once without
``--no_export`` to create the default TorchScript policy.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF_DIR = REPO_ROOT / "legged_lab/assets/g1/g1_23dof"
DEFAULT_MODEL = MJCF_DIR / "g1_23dof_rev_1_0.xml"


# MuJoCo XML and actuator order for the official 23-DoF model.
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


# IsaacLab order seen by the Recovery actor.
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


class G1RecoverySim2SimCfg:
    """Standalone MuJoCo settings matching the G1 Recovery training task."""

    class sim:
        duration = 30.0
        num_actions = 23
        num_obs_per_step = 75
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        soft_joint_limit_factor = 0.90

    class robot:
        init_height = 0.793
        joint_armature = 0.01
        stabilize_duration = 2.0
        recovery_trigger_height_ratio = 0.85
        recovery_trigger_upright = 0.93


def _read_mjcf_text(xml_path: Path) -> tuple[str, bool]:
    """Read MJCF text and strip any prefix before the opening mujoco tag."""
    text = xml_path.read_text(encoding="utf-8")
    mujoco_start = text.find("<mujoco")
    if mujoco_start < 0:
        return text, False
    return text[mujoco_start:], mujoco_start > 0


def _parse_xml(xml_path: Path) -> ET.ElementTree:
    try:
        text, _ = _read_mjcf_text(xml_path)
        return ET.ElementTree(ET.fromstring(text))
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"Failed to parse MJCF XML: {xml_path}") from exc


def _mirror_meshdir(
    source_xml: Path,
    source_root: ET.Element,
    temp_dir: Path,
) -> None:
    compiler = source_root.find("compiler")
    if compiler is None:
        return
    meshdir = compiler.attrib.get("meshdir")
    if not meshdir:
        return

    source_meshdir = (source_xml.parent / meshdir).resolve()
    target_meshdir = temp_dir / meshdir
    if not source_meshdir.is_dir() or target_meshdir.exists():
        return

    target_meshdir.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_meshdir.symlink_to(source_meshdir, target_is_directory=True)
    except OSError:
        shutil.copytree(source_meshdir, target_meshdir)


def _scene_has_custom_terrain(scene_root: ET.Element) -> bool:
    worldbody = scene_root.find("worldbody")
    return worldbody is not None and next(worldbody.iter("geom"), None) is not None


def _remove_default_floor(tree: ET.ElementTree) -> int:
    """Remove the robot XML's default floor before including custom terrain."""

    def remove_from(parent: ET.Element) -> int:
        removed = 0
        for child in list(parent):
            if child.tag == "geom" and child.attrib.get("name") == "floor":
                parent.remove(child)
                removed += 1
            else:
                removed += remove_from(child)
        return removed

    removed = 0
    for worldbody in tree.getroot().findall("worldbody"):
        removed += remove_from(worldbody)
    return removed


def prepare_mjcf_for_custom_scene(
    model_path: Path,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Prepare a self-contained temporary MJCF when a custom scene is used."""
    model_path = model_path.expanduser().resolve()
    _, model_needs_sanitizing = _read_mjcf_text(model_path)
    scene_tree = _parse_xml(model_path)
    scene_root = scene_tree.getroot()
    include = next(scene_root.iter("include"), None)
    strip_included_floor = bool(
        include is not None
        and include.attrib.get("file")
        and _scene_has_custom_terrain(scene_root)
    )
    if not model_needs_sanitizing and not strip_included_floor:
        return model_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="g1_recovery_mjcf_")
    temp_path = Path(temp_dir.name)
    if include is not None and include.attrib.get("file"):
        include_file = include.attrib["file"]
        robot_xml = (model_path.parent / include_file).resolve()
        robot_tree = _parse_xml(robot_xml)
        if strip_included_floor:
            removed = _remove_default_floor(robot_tree)
            if removed:
                print(
                    f"[INFO] Removed {removed} default floor geom(s) from "
                    "the included robot XML."
                )
        _mirror_meshdir(robot_xml, robot_tree.getroot(), temp_path)

        patched_robot_path = temp_path / robot_xml.name
        robot_tree.write(patched_robot_path, encoding="unicode")
        include.attrib["file"] = robot_xml.name

    _mirror_meshdir(model_path, scene_root, temp_path)
    patched_scene_path = temp_path / model_path.name
    scene_tree.write(patched_scene_path, encoding="unicode")
    print(f"[INFO] Using temporary MuJoCo scene: {patched_scene_path}")
    return patched_scene_path, temp_dir


def get_available_scenes(mjcf_dir: Path) -> dict[str, Path]:
    scenes: dict[str, Path] = {}
    if not mjcf_dir.is_dir():
        return scenes
    for path in sorted(mjcf_dir.glob("*_scene.xml")):
        scenes[path.name.removesuffix("_scene.xml")] = path.resolve()
    for filename in ("scene.xml", "flat_scene.xml", "rough_scene.xml"):
        path = mjcf_dir / filename
        if path.is_file():
            name = path.stem.removesuffix("_scene")
            scenes.setdefault(name, path.resolve())
    return scenes


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
    """Return IsaacLab's unit gravity vector in the pelvis frame."""
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
    """Return the Recovery training (Kp, Kd, effort limit) for one joint."""
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


def find_latest_exported_policy() -> Path | None:
    """Find the most recently modified exported Recovery TorchScript policy."""
    candidates = list(
        (REPO_ROOT / "logs/g1_recovery").glob("*/exported/policy.pt")
    )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve() if candidates else None


class G1RecoveryMujocoRunner:
    """Execute an exported G1 Recovery actor against a MuJoCo model."""

    def __init__(
        self,
        cfg: G1RecoverySim2SimCfg,
        policy_path: Path,
        model_path: Path,
    ) -> None:
        self.cfg = cfg
        self.policy_path = policy_path.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"TorchScript policy not found: {self.policy_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        print(f"[INFO] Loading MuJoCo model: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        print(f"[INFO] Loading TorchScript policy: {self.policy_path}")
        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.policy.eval()

        self.viewer = None
        self.init_variables()
        self.build_joint_mappings()
        self.build_control_parameters()
        self.set_initial_pose()
        self._fill_observation_history()
        self.validate_policy_contract()

        control_hz = 1.0 / (self.cfg.sim.dt * self.cfg.sim.decimation)
        obs_dim = (
            self.cfg.sim.num_obs_per_step
            * self.cfg.sim.actor_obs_history_length
        )
        print(f"[INFO] Control frequency: {control_hz:.1f} Hz")
        print(
            f"[INFO] Actor observation: {self.cfg.sim.num_obs_per_step} x "
            f"{self.cfg.sim.actor_obs_history_length} = {obs_dim}"
        )
        print(f"[INFO] Whole-body action dimension: {self.cfg.sim.num_actions}")

    def init_variables(self) -> None:
        self.policy_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions
        if self.model.nu < self.num_actions:
            raise RuntimeError(
                f"MuJoCo model has {self.model.nu} actuators, expected at "
                f"least {self.num_actions}."
            )

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_history = np.zeros(
            self.cfg.sim.num_obs_per_step
            * self.cfg.sim.actor_obs_history_length,
            dtype=np.float32,
        )
        self.policy_step = 0

    def build_joint_mappings(self) -> None:
        free_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        if (
            free_joint_id < 0
            or self.model.jnt_type[free_joint_id]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
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
            print("[WARNING] MuJoCo joint order differs from the reference XML.")
            print(f"[WARNING] Expected: {MUJOCO_JOINT_NAMES}")
            print(f"[WARNING] Actual:   {tuple(actual_joint_names)}")

        missing_joints = sorted(
            set(MUJOCO_JOINT_NAMES) - set(actual_joint_names)
        )
        if missing_joints:
            raise RuntimeError(
                f"Model is missing Recovery joints: {missing_joints}"
            )

        joint_ids = []
        qpos_addr = []
        qvel_addr = []
        joint_ranges = []
        for name in MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            joint_ids.append(joint_id)
            qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_addr.append(int(self.model.jnt_dofadr[joint_id]))
            joint_ranges.append(self.model.jnt_range[joint_id].copy())

        actuator_by_joint: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if joint_name is not None:
                actuator_by_joint.setdefault(joint_name, actuator_id)
        missing_actuators = sorted(
            set(MUJOCO_JOINT_NAMES) - set(actuator_by_joint)
        )
        if missing_actuators:
            raise RuntimeError(
                "Model is missing actuators for Recovery joints: "
                f"{missing_actuators}"
            )

        self.joint_ids_mj = np.asarray(joint_ids, dtype=np.int32)
        self.qpos_addr_mj = np.asarray(qpos_addr, dtype=np.int32)
        self.qvel_addr_mj = np.asarray(qvel_addr, dtype=np.int32)
        self.ctrl_addr_mj = np.asarray(
            [actuator_by_joint[name] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )
        self.model.dof_armature[self.qvel_addr_mj] = (
            self.cfg.robot.joint_armature
        )

        mujoco_indices = {
            name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)
        }
        isaac_indices = {
            name: index for index, name in enumerate(ISAAC_JOINT_NAMES)
        }
        self.mujoco_to_isaac_idx = np.asarray(
            [mujoco_indices[name] for name in ISAAC_JOINT_NAMES],
            dtype=np.int32,
        )
        self.isaac_to_mujoco_idx = np.asarray(
            [isaac_indices[name] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )

        joint_ranges_mj = np.asarray(joint_ranges, dtype=np.float64)
        joint_mid = joint_ranges_mj.mean(axis=1)
        joint_half_range = 0.5 * np.diff(
            joint_ranges_mj,
            axis=1,
        ).reshape(-1)
        factor = self.cfg.sim.soft_joint_limit_factor
        self.soft_joint_lower_mj = joint_mid - factor * joint_half_range
        self.soft_joint_upper_mj = joint_mid + factor * joint_half_range

        self.torso_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso_link",
        )
        if self.torso_body_id < 0:
            raise RuntimeError("MuJoCo model is missing torso_link.")

        print("[INFO] Joint and actuator mappings are ready.")
        print(f"[INFO] MuJoCo first 6 joints: {MUJOCO_JOINT_NAMES[:6]}")
        print(f"[INFO] IsaacLab first 6 joints: {ISAAC_JOINT_NAMES[:6]}")
        print(f"[INFO] qpos addresses: {self.qpos_addr_mj.tolist()}")
        print(f"[INFO] qvel addresses: {self.qvel_addr_mj.tolist()}")
        print(f"[INFO] ctrl addresses: {self.ctrl_addr_mj.tolist()}")
        print(
            "[INFO] MuJoCo joint armature: "
            f"{self.cfg.robot.joint_armature}"
        )

    def build_control_parameters(self) -> None:
        self.default_dof_pos_mj = np.asarray(
            [_default_joint_position(name) for name in MUJOCO_JOINT_NAMES],
            dtype=np.float64,
        )
        self.default_dof_pos_isaac = self.mj_to_isaac(
            self.default_dof_pos_mj
        )
        parameters = np.asarray(
            [_joint_control_parameters(name) for name in MUJOCO_JOINT_NAMES],
            dtype=np.float64,
        )
        self.kp_mj = parameters[:, 0]
        self.kd_mj = parameters[:, 1]
        self.effort_limit_mj = parameters[:, 2]

    def validate_policy_contract(self) -> None:
        expected_obs_dim = (
            self.cfg.sim.num_obs_per_step
            * self.cfg.sim.actor_obs_history_length
        )
        dummy_obs = torch.zeros(1, expected_obs_dim, dtype=torch.float32)
        with torch.inference_mode():
            dummy_action = self.policy(dummy_obs)
        if tuple(dummy_action.shape) != (1, self.num_actions):
            raise RuntimeError(
                "Recovery TorchScript policy contract mismatch: expected "
                f"input [1, {expected_obs_dim}] -> output "
                f"[1, {self.num_actions}], got {tuple(dummy_action.shape)}."
            )
        if not torch.isfinite(dummy_action).all():
            raise RuntimeError("Recovery policy produced NaN/Inf for zero input.")

    def mj_to_isaac(self, values_mj: np.ndarray) -> np.ndarray:
        return values_mj[self.mujoco_to_isaac_idx]

    def isaac_to_mj(self, values_isaac: np.ndarray) -> np.ndarray:
        return values_isaac[self.isaac_to_mujoco_idx]

    def set_initial_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        root_qpos = self.root_qpos_addr
        self.data.qpos[root_qpos : root_qpos + 3] = [
            0.0,
            0.0,
            self.cfg.robot.init_height,
        ]
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_addr_mj] = self.default_dof_pos_mj
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.action.fill(0.0)
        self.policy_step = 0
        mujoco.mj_forward(self.model, self.data)
        print(
            f"[INFO] Upright initial pelvis height: "
            f"{self.data.qpos[root_qpos + 2]:.3f} m"
        )

    def _current_observation(self) -> np.ndarray:
        joint_pos_mj = self.data.qpos[self.qpos_addr_mj].copy()
        joint_vel_mj = self.data.qvel[self.qvel_addr_mj].copy()
        root_quat = self.data.qpos[
            self.root_qpos_addr + 3 : self.root_qpos_addr + 7
        ].copy()
        root_ang_vel_b = self.data.qvel[
            self.root_qvel_addr + 3 : self.root_qvel_addr + 6
        ].copy()

        current_obs = np.concatenate(
            (
                root_ang_vel_b,
                projected_gravity_from_quat_wxyz(root_quat),
                self.mj_to_isaac(joint_pos_mj)
                - self.default_dof_pos_isaac,
                self.mj_to_isaac(joint_vel_mj),
                self.action,
            )
        ).astype(np.float32)
        expected_shape = (self.cfg.sim.num_obs_per_step,)
        if current_obs.shape != expected_shape:
            raise RuntimeError(
                f"Expected single observation shape {expected_shape}, "
                f"got {current_obs.shape}."
            )
        if not np.isfinite(current_obs).all():
            raise RuntimeError("Recovery observation contains NaN or Inf.")
        return current_obs

    def get_obs(self) -> np.ndarray:
        current_obs = self._current_observation()
        obs_dim = self.cfg.sim.num_obs_per_step
        self.obs_history = np.roll(self.obs_history, shift=-obs_dim)
        self.obs_history[-obs_dim:] = current_obs
        return np.clip(
            self.obs_history,
            -self.cfg.sim.clip_observations,
            self.cfg.sim.clip_observations,
        ).astype(np.float32)

    def _fill_observation_history(self) -> None:
        self.obs_history.fill(0.0)
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()

    def infer(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        with torch.inference_mode():
            action_tensor = self.policy(obs_tensor)
        action = action_tensor.detach().cpu().numpy().reshape(-1)
        if action.shape != (self.num_actions,) or not np.isfinite(action).all():
            raise RuntimeError(
                "Recovery policy must return one finite 23-D action, got "
                f"shape={action.shape}."
            )
        self.action[:] = np.clip(
            action,
            -self.cfg.sim.clip_actions,
            self.cfg.sim.clip_actions,
        )
        return self.action.copy()

    def position_control(self) -> np.ndarray:
        target_isaac = (
            self.default_dof_pos_isaac
            + self.cfg.sim.action_scale * self.action.astype(np.float64)
        )
        target_mj = self.isaac_to_mj(target_isaac)
        target_mj = np.clip(
            target_mj,
            self.soft_joint_lower_mj,
            self.soft_joint_upper_mj,
        )
        if not np.isfinite(target_mj).all():
            raise RuntimeError("Recovery position target contains NaN or Inf.")
        return target_mj

    def pd_control(self, target_q_mj: np.ndarray) -> np.ndarray:
        joint_pos_mj = self.data.qpos[self.qpos_addr_mj]
        joint_vel_mj = self.data.qvel[self.qvel_addr_mj]
        torque_mj = (
            self.kp_mj * (target_q_mj - joint_pos_mj)
            - self.kd_mj * joint_vel_mj
        )
        torque_mj = np.clip(
            torque_mj,
            -self.effort_limit_mj,
            self.effort_limit_mj,
        )
        if not np.isfinite(torque_mj).all():
            raise RuntimeError("Recovery PD torque contains NaN or Inf.")
        return torque_mj

    def _apply_control(self) -> None:
        target_q_mj = self.position_control()
        torque_mj = self.pd_control(target_q_mj)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_addr_mj] = torque_mj

    def _require_finite_state(self) -> None:
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("MuJoCo qpos contains NaN or Inf.")
        if not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo qvel contains NaN or Inf.")

    def _hold_upright_root(self) -> None:
        """Apply the same root hold used during the stabilization phase."""
        root_qpos = self.root_qpos_addr
        root_qvel = self.root_qvel_addr
        self.data.qpos[root_qpos : root_qpos + 2] = 0.0
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[root_qvel : root_qvel + 6] = 0.0

    def recovery_triggered(self) -> tuple[bool, float, float]:
        """Return whether manual perturbation has moved the robot out of stand."""
        pelvis_height = float(self.data.qpos[self.root_qpos_addr + 2])
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_up_z = float(torso_rotation[2, 2])
        height_threshold = (
            self.cfg.robot.recovery_trigger_height_ratio
            * self.cfg.robot.init_height
        )
        external_force_active = bool(
            np.linalg.norm(self.data.xfrc_applied, axis=1).max(initial=0.0)
            > 1.0e-6
        )
        triggered = (
            external_force_active
            or pelvis_height < height_threshold
            or torso_up_z < self.cfg.robot.recovery_trigger_upright
        )
        return triggered, pelvis_height, torso_up_z

    def stabilize_robot(self, duration: float, realtime: bool) -> None:
        """Hold the upright root attitude while joints settle under PD control."""
        if duration <= 0.0:
            self._fill_observation_history()
            return

        target_q_mj = self.default_dof_pos_mj.copy()
        num_steps = int(round(duration / self.cfg.sim.dt))
        for step in range(num_steps):
            step_start = time.perf_counter()
            self._hold_upright_root()

            torque_mj = self.pd_control(target_q_mj)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.ctrl_addr_mj] = torque_mj
            mujoco.mj_step(self.model, self.data)
            self._require_finite_state()

            if self.viewer is not None and step % 10 == 0:
                self.viewer.sync()
            if realtime:
                sleep_s = self.cfg.sim.dt - (
                    time.perf_counter() - step_start
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        root_qpos = self.root_qpos_addr
        root_qvel = self.root_qvel_addr
        self.data.qpos[root_qpos : root_qpos + 2] = 0.0
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[root_qvel : root_qvel + 6] = 0.0
        self.action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        self._fill_observation_history()
        print(
            f"[INFO] Stabilized pelvis height: "
            f"{self.data.qpos[root_qpos + 2]:.4f} m"
        )

    def run(
        self,
        headless: bool,
        realtime: bool,
        print_every: int,
        stabilize_duration: float,
    ) -> None:
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = self.data.qpos[
                self.root_qpos_addr : self.root_qpos_addr + 3
            ]
            self.viewer.cam.distance = 2.5
            self.viewer.sync()

        self.stabilize_robot(stabilize_duration, realtime=realtime)
        if self.viewer is not None:
            print(
                "\n[INFO] Use the MuJoCo viewer perturbation controls to "
                "push or drag the robot."
            )
            print(
                "[INFO] The Recovery policy activates after pelvis height or "
                "torso uprightness drops below the training threshold."
            )
            print("[INFO] Close the viewer or press Ctrl+C to exit.\n")

        run_start_sim_time = float(self.data.time)
        total_physics_steps = 0
        recovery_active = False
        try:
            while self.data.time - run_start_sim_time < self.cfg.sim.duration:
                if self.viewer is not None and not self.viewer.is_running():
                    break

                observation = self.get_obs()
                triggered, pelvis_h, torso_up_z = self.recovery_triggered()
                if triggered and not recovery_active:
                    recovery_active = True
                    print(
                        "[INFO] Manual disturbance detected; activating "
                        f"Recovery policy at pelvis_h={pelvis_h:.3f}, "
                        f"torso_up_z={torso_up_z:.3f}."
                    )
                if recovery_active:
                    self.infer(observation)
                else:
                    self.action.fill(0.0)
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.perf_counter()
                    if not recovery_active:
                        self._hold_upright_root()
                    self._apply_control()
                    mujoco.mj_step(self.model, self.data)
                    self._require_finite_state()
                    total_physics_steps += 1
                    if self.viewer is not None:
                        self.viewer.sync()
                    if realtime:
                        sleep_s = self.cfg.sim.dt - (
                            time.perf_counter() - step_start
                        )
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)

                self.policy_step += 1
                if print_every and self.policy_step % print_every == 0:
                    _, pelvis_h, torso_up_z = self.recovery_triggered()
                    elapsed = total_physics_steps * self.cfg.sim.dt
                    mode = "recovery" if recovery_active else "standby"
                    print(
                        f"[INFO] t={elapsed:.2f}s, "
                        f"mode={mode}, "
                        f"pelvis_h={pelvis_h:.3f}, "
                        f"torso_up_z={torso_up_z:.3f}, "
                        f"action_norm={np.linalg.norm(self.action):.3f}"
                    )
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        print("[INFO] MuJoCo Recovery simulation finished.")


def _build_parser(
    available_scenes: dict[str, Path],
) -> argparse.ArgumentParser:
    scene_names = sorted(available_scenes)
    parser = argparse.ArgumentParser(
        description="G1 Recovery 23-DoF IsaacLab-to-MuJoCo Sim2Sim.",
        formatter_class=argparse.RawTextHelpFormatter,
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
        help="MuJoCo XML used when --scene and --scene-file are omitted.",
    )
    parser.add_argument(
        "--scene",
        choices=scene_names if scene_names else None,
        default=None,
        help=(
            "Built-in scene name. Available: "
            f"{', '.join(scene_names) if scene_names else 'none'}"
        ),
    )
    parser.add_argument(
        "--scene-file",
        "--scene_file",
        dest="scene_file",
        type=Path,
        default=None,
        help="Custom scene XML path; takes priority over --scene.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=G1RecoverySim2SimCfg.sim.duration,
        help="Policy playback duration in seconds, excluding stabilization.",
    )
    parser.add_argument(
        "--stabilize-duration",
        "--stabilize_duration",
        dest="stabilize_duration",
        type=float,
        default=G1RecoverySim2SimCfg.robot.stabilize_duration,
        help="Upright PD stabilization duration in seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--no-realtime",
        "--no_realtime",
        dest="no_realtime",
        action="store_true",
        help="Do not sleep to match wall-clock simulation time.",
    )
    parser.add_argument(
        "--print-every",
        "--print_every",
        dest="print_every",
        type=int,
        default=25,
        help="Print status every N policy steps; 0 disables output.",
    )
    parser.add_argument(
        "--list-scenes",
        "--list_scenes",
        dest="list_scenes",
        action="store_true",
        help="List built-in scene XML files and exit.",
    )
    parser.add_argument(
        "--check-mapping",
        "--check_mapping",
        dest="check_mapping",
        action="store_true",
        help="Validate and print joint/actuator mappings, then exit.",
    )
    parser.add_argument(
        "--validate-only",
        "--validate_only",
        dest="validate_only",
        action="store_true",
        help="Validate model, observation, policy, target, and torque contracts.",
    )
    return parser


def main() -> None:
    available_scenes = get_available_scenes(MJCF_DIR)
    parser = _build_parser(available_scenes)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.stabilize_duration < 0.0:
        parser.error("--stabilize-duration must be non-negative.")
    if args.print_every < 0:
        parser.error("--print-every must be non-negative.")

    if args.list_scenes:
        print("Available scenes:")
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {path.name}")
        print(f"Scene directory: {MJCF_DIR}")
        return

    policy_path = (
        args.policy.expanduser().resolve()
        if args.policy is not None
        else find_latest_exported_policy()
    )
    if policy_path is None:
        parser.error(
            "No exported g1_recovery policy was found. Run "
            "play_recovery.py once without --no_export, or pass --policy."
        )

    if args.scene_file is not None:
        model_path = args.scene_file.expanduser().resolve()
    elif args.scene is not None:
        model_path = available_scenes[args.scene]
    else:
        model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"MuJoCo XML does not exist: {model_path}")

    cfg = G1RecoverySim2SimCfg()
    cfg.sim.duration = args.duration

    print("=" * 60)
    print("G1 Recovery 23-DoF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {policy_path}")
    print(f"MuJoCo XML: {model_path}")
    if args.scene is not None:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    load_model_path, temp_mjcf_dir = prepare_mjcf_for_custom_scene(model_path)
    try:
        runner = G1RecoveryMujocoRunner(
            cfg=cfg,
            policy_path=policy_path,
            model_path=load_model_path,
        )
        if args.check_mapping:
            print("[SUCCESS] Recovery joint/actuator mapping validated.")
            return

        if args.validate_only:
            observation = runner.get_obs()
            action = runner.infer(observation)
            target = runner.position_control()
            torque = runner.pd_control(target)
            print(
                "[SUCCESS] Recovery sim2sim contract validated: "
                f"obs={observation.shape}, action={action.shape}, "
                f"target={target.shape}, torque={torque.shape}."
            )
            return

        runner.run(
            headless=args.headless,
            realtime=not args.no_realtime,
            print_every=args.print_every,
            stabilize_duration=args.stabilize_duration,
        )
    finally:
        if temp_mjcf_dir is not None:
            temp_mjcf_dir.cleanup()


if __name__ == "__main__":
    main()
