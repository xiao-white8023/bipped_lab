# ======================================================
# L1 OP模式
# R1 VP模式
# ========================================================

import argparse
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torchvision.transforms as T


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from joystick import JoyStick, JoystickButton  # noqa: E402

    JOYSTICK_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    JoyStick = None
    JoystickButton = None
    JOYSTICK_IMPORT_ERROR = exc


# MuJoCo XML joint order for the 23DOF G1 model.
MUJOCO_DOF_NAMES = [
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
]

# Isaac Lab joint order used by g1_renet/g1_rough training.
LAB_DOF_NAMES = [
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
]


class G1RENetSim2SimCfg:
    class sim:
        sim_duration = 100.0
        num_actions = 23
        num_obs_per_step = 78
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        print_mapping = False

    class command:
        lin_vel_x = (-0.6, 1.0)
        lin_vel_y = (-0.5, 0.5)
        ang_vel_z = (-1.57, 1.57)
        deadzone = 0.1

    class gait:
        enable = True
        period = 0.8
        offset = 0.5

    class robot:
        init_height = 0.793
        joint_armature = 0.1
        joint_damping = 0.001
        joint_frictionloss = 0.1

        camera_name = "depth_camera"
        camera_body_name = "torso_link"
        camera_height = 36
        camera_width = 64
        camera_pos = "0.0576235 0.01753 0.42987"
        camera_xyaxes = "0 -1 0 0.7384553406258838 0 0.6743023875837234"
        camera_fovy = "58.29"
        depth_crop = (18, 0, 16, 16)  # up, down, left, right: 36x64 -> 18x32
        depth_history_frames = 2
        depth_update_interval = 5
        depth_min = 0.1
        depth_max = 2.5
        show_depth = True
        debug_depth = False
        auto_patch_camera = True

    class renet:
        # Paper/env convention: 1.0 = OP, 0.0 = VP.
        estimator_mask_dim = 1
        default_estimator = "vp"


def _read_mjcf_text(xml_path: str) -> tuple[str, bool]:
    with open(xml_path, "r", encoding="utf-8") as f:
        text = f.read()

    mujoco_start = text.find("<mujoco")
    if mujoco_start < 0:
        return text, False
    return text[mujoco_start:], mujoco_start > 0


def _parse_xml(xml_path: str) -> ET.ElementTree:
    try:
        text, _ = _read_mjcf_text(xml_path)
        return ET.ElementTree(ET.fromstring(text))
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse MJCF XML: {xml_path}") from exc


def xml_contains_camera(xml_path: str, camera_name: str, seen: set[str] | None = None) -> bool:
    xml_path = os.path.abspath(xml_path)
    if seen is None:
        seen = set()
    if xml_path in seen or not os.path.isfile(xml_path):
        return False
    seen.add(xml_path)

    tree = _parse_xml(xml_path)
    root = tree.getroot()
    for camera in root.iter("camera"):
        if camera.attrib.get("name") == camera_name:
            return True

    xml_dir = os.path.dirname(xml_path)
    for include in root.iter("include"):
        include_file = include.attrib.get("file")
        if include_file and xml_contains_camera(os.path.join(xml_dir, include_file), camera_name, seen):
            return True
    return False


def _mirror_meshdir(source_xml: str, source_root: ET.Element, temp_dir: str) -> None:
    compiler = source_root.find("compiler")
    if compiler is None:
        return

    meshdir = compiler.attrib.get("meshdir")
    if not meshdir:
        return

    source_meshdir = os.path.join(os.path.dirname(source_xml), meshdir)
    target_meshdir = os.path.join(temp_dir, meshdir)
    if not os.path.isdir(source_meshdir) or os.path.exists(target_meshdir):
        return

    os.makedirs(os.path.dirname(target_meshdir), exist_ok=True)
    try:
        os.symlink(source_meshdir, target_meshdir, target_is_directory=True)
    except OSError:
        shutil.copytree(source_meshdir, target_meshdir)


def _scene_has_custom_terrain(scene_root: ET.Element) -> bool:
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        return False
    return any(child.tag == "geom" for child in worldbody.iter())


def _remove_default_floor_from_tree(tree: ET.ElementTree) -> int:
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


def _add_depth_camera_to_tree(tree: ET.ElementTree, cfg: G1RENetSim2SimCfg) -> None:
    root = tree.getroot()
    if any(camera.attrib.get("name") == cfg.robot.camera_name for camera in root.iter("camera")):
        return

    target_body = None
    for body in root.iter("body"):
        if body.attrib.get("name") == cfg.robot.camera_body_name:
            target_body = body
            break
    if target_body is None:
        raise RuntimeError(f"Cannot find body '{cfg.robot.camera_body_name}' for RENet depth camera.")

    camera = ET.Element(
        "camera",
        {
            "name": cfg.robot.camera_name,
            "pos": cfg.robot.camera_pos,
            "xyaxes": cfg.robot.camera_xyaxes,
            "fovy": cfg.robot.camera_fovy,
        },
    )

    insert_at = 0
    for idx, child in enumerate(list(target_body)):
        if child.tag in {"inertial", "joint", "site"}:
            insert_at = idx + 1
    target_body.insert(insert_at, camera)


def prepare_mjcf_with_depth_camera(model_path: str, cfg: G1RENetSim2SimCfg) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """Return a MuJoCo XML path that has the RENet depth camera.

    The g1_23dof robot XML defines depth_camera. This helper still keeps scene
    files robust by sanitizing non-XML prefixes, removing the robot XML's
    built-in flat floor from custom terrain scenes, and by adding the training
    camera pose to custom models that do not have it yet.
    """

    _, model_needs_sanitizing = _read_mjcf_text(model_path)
    source_dir = os.path.dirname(os.path.abspath(model_path))
    scene_tree = _parse_xml(model_path)
    scene_root = scene_tree.getroot()
    include = next(scene_root.iter("include"), None)
    strip_included_floor = bool(
        include is not None and include.attrib.get("file") and _scene_has_custom_terrain(scene_root)
    )
    has_camera = xml_contains_camera(model_path, cfg.robot.camera_name)
    if has_camera and not model_needs_sanitizing and not strip_included_floor:
        return model_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="g1_renet_mjcf_")

    if include is not None and include.attrib.get("file"):
        include_file = include.attrib["file"]
        robot_xml = os.path.join(source_dir, include_file)
        robot_tree = _parse_xml(robot_xml)
        if strip_included_floor:
            removed_floors = _remove_default_floor_from_tree(robot_tree)
            if removed_floors:
                print(f"[INFO] Removed {removed_floors} default floor geom(s) from included robot XML.")
        if not xml_contains_camera(robot_xml, cfg.robot.camera_name):
            _add_depth_camera_to_tree(robot_tree, cfg)
        _mirror_meshdir(robot_xml, robot_tree.getroot(), temp_dir.name)

        patched_robot_name = os.path.basename(robot_xml)
        patched_robot_path = os.path.join(temp_dir.name, patched_robot_name)
        robot_tree.write(patched_robot_path, encoding="unicode")

        include.attrib["file"] = patched_robot_name
        patched_scene_path = os.path.join(temp_dir.name, os.path.basename(model_path))
        scene_tree.write(patched_scene_path, encoding="unicode")
        print(f"[INFO] Using temporary RENet MJCF scene: {patched_scene_path}")
        return patched_scene_path, temp_dir

    if not has_camera:
        _add_depth_camera_to_tree(scene_tree, cfg)
    _mirror_meshdir(model_path, scene_root, temp_dir.name)
    patched_model_path = os.path.join(temp_dir.name, os.path.basename(model_path))
    scene_tree.write(patched_model_path, encoding="unicode")
    print(f"[INFO] Using temporary RENet MJCF model: {patched_model_path}")
    return patched_model_path, temp_dir


class G1RENetMujocoRunner:
    def __init__(self, cfg: G1RENetSim2SimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        self.estimator = self.cfg.renet.default_estimator.lower()
        if self.estimator not in {"op", "vp"}:
            raise ValueError("Default RENet estimator must be 'op' or 'vp'.")
        self.estimator_mask_value = self.estimator_to_mask(self.estimator)
        self._temp_mjcf_dir = None

        load_model_path = model_path
        if self.cfg.robot.auto_patch_camera:
            load_model_path, self._temp_mjcf_dir = prepare_mjcf_with_depth_camera(model_path, self.cfg)

        print(f"[INFO] Loading MuJoCo model: {load_model_path}")
        self.model = mujoco.MjModel.from_xml_path(load_model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        self.init_depth_camera()
        self.init_joystick()

        control_hz = 1.0 / (self.cfg.sim.dt * self.cfg.sim.decimation)
        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        depth_dim = self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
        total_actor_dim = proprio_dim + depth_dim + self.cfg.renet.estimator_mask_dim
        print(f"[INFO] Control frequency: {control_hz:.1f} Hz")
        print(f"[INFO] RENet estimator: {self.estimator.upper()} (mask={self.estimator_mask_value:.1f})")
        print(f"[INFO] Actor obs: proprio={proprio_dim}, depth={depth_dim}, mask=1, total={total_actor_dim}")

    @staticmethod
    def estimator_to_mask(estimator: str) -> float:
        return 1.0 if estimator == "op" else 0.0

    def set_estimator(self, estimator: str, announce: bool = True) -> None:
        estimator = estimator.lower()
        if estimator not in {"op", "vp"}:
            raise ValueError("RENet estimator must be 'op' or 'vp'.")
        if estimator == "vp" and self.renderer is None:
            if announce:
                print("[WARNING] VP requested, but depth renderer is unavailable. Staying on OP.")
            estimator = "op"
        if estimator == self.estimator:
            return
        self.estimator = estimator
        self.estimator_mask_value = self.estimator_to_mask(estimator)
        if announce:
            print(f"[INFO] RENet estimator switched to {self.estimator.upper()} (mask={self.estimator_mask_value:.1f})")

    def current_mode_text(self) -> str:
        return f"{self.estimator.upper()} mask={self.estimator_mask_value:.1f}"

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions

        if self.model.nu < self.num_actions:
            raise RuntimeError(f"MuJoCo model has {self.model.nu} actuators, expected at least {self.num_actions}.")

        crop_up, crop_down, crop_left, crop_right = self.cfg.robot.depth_crop
        self.depth_height = self.cfg.robot.camera_height - crop_up - crop_down
        self.depth_width = self.cfg.robot.camera_width - crop_left - crop_right
        if self.depth_height <= 0 or self.depth_width <= 0:
            raise RuntimeError(f"Invalid RENet depth crop: {self.cfg.robot.depth_crop}.")

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.command_vel = np.zeros(3, dtype=np.float32)
        self.episode_length_buf = 0
        self.phase = 0.0
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.leg_phase = np.zeros(2, dtype=np.float32)

        self.default_dof_pos = np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.42,
                -0.23,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.42,
                -0.23,
                0.0,
                0.0,
                0.35,
                0.18,
                0.0,
                0.87,
                0.0,
                0.35,
                -0.18,
                0.0,
                0.87,
                0.0,
            ],
            dtype=np.float32,
        )

        self.kps = np.array(
            [
                200.0,
                150.0,
                150.0,
                200.0,
                20.0,
                20.0,
                200.0,
                150.0,
                150.0,
                200.0,
                20.0,
                20.0,
                200.0,
                100.0,
                100.0,
                50.0,
                50.0,
                15.0,
                100.0,
                100.0,
                50.0,
                50.0,
                15.0,
            ],
            dtype=np.float32,
        )
        self.kds = np.array(
            [
                5.0,
                5.0,
                5.0,
                5.0,
                2.0,
                2.0,
                5.0,
                5.0,
                5.0,
                5.0,
                2.0,
                2.0,
                5.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
            dtype=np.float32,
        )
        self.torque_limits = np.array(
            [
                88.0,
                139.0,
                88.0,
                139.0,
                35.0,
                35.0,
                88.0,
                139.0,
                88.0,
                139.0,
                35.0,
                35.0,
                88.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
            ],
            dtype=np.float32,
        )

        self.obs_history = np.zeros(
            self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,
            dtype=np.float32,
        )
        self.depth_buffer = deque(
            [
                torch.zeros((self.depth_height, self.depth_width), dtype=torch.float32)
                for _ in range(self.cfg.robot.depth_history_frames)
            ],
            maxlen=self.cfg.robot.depth_history_frames,
        )
        self.blur_transform = T.GaussianBlur(kernel_size=3, sigma=1.0)

        self.viewer = None
        self.renderer = None
        self.fig = None
        self.ax = None
        self.im = None
        self.joystick = None
        self.use_joystick = False

    def build_joint_mappings(self) -> None:
        real_mujoco_names = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                real_mujoco_names.append(name)

        if real_mujoco_names != MUJOCO_DOF_NAMES:
            print("\n========== MuJoCo joint order mismatch ==========")
            for idx, (expected, actual) in enumerate(zip(MUJOCO_DOF_NAMES, real_mujoco_names)):
                if expected != actual:
                    print(f"[WARN] index {idx}: expected {expected}, got {actual}")
            if len(real_mujoco_names) != len(MUJOCO_DOF_NAMES):
                print(f"[WARN] expected {len(MUJOCO_DOF_NAMES)} joints, got {len(real_mujoco_names)}")
            print("================================================\n")

        missing = sorted(set(LAB_DOF_NAMES) - set(real_mujoco_names))
        if missing:
            raise RuntimeError(f"Model is missing joints required by g1_renet: {missing}")

        self.qpos_addr_mj = []
        self.qvel_addr_mj = []
        for name in MUJOCO_DOF_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Model is missing MuJoCo joint: {name}")
            self.qpos_addr_mj.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_addr_mj.append(int(self.model.jnt_dofadr[joint_id]))

        self.qpos_addr_mj = np.asarray(self.qpos_addr_mj, dtype=np.int32)
        self.qvel_addr_mj = np.asarray(self.qvel_addr_mj, dtype=np.int32)
        self.model.dof_armature[self.qvel_addr_mj] = self.cfg.robot.joint_armature
        self.model.dof_damping[self.qvel_addr_mj] = self.cfg.robot.joint_damping
        self.model.dof_frictionloss[self.qvel_addr_mj] = self.cfg.robot.joint_frictionloss

        actuator_by_joint = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None and joint_name not in actuator_by_joint:
                actuator_by_joint[joint_name] = actuator_id

        missing_actuators = sorted(set(MUJOCO_DOF_NAMES) - set(actuator_by_joint))
        if missing_actuators:
            raise RuntimeError(f"Model is missing actuators for joints: {missing_actuators}")
        self.ctrl_addr_mj = np.asarray([actuator_by_joint[name] for name in MUJOCO_DOF_NAMES], dtype=np.int32)

        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.mujoco_to_lab_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
        self.lab_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        self.default_dof_pos_lab = self.mj_to_lab(self.default_dof_pos)

        print("[INFO] Joint mapping ready.")
        print(f"[INFO] MuJoCo first 6 joints: {MUJOCO_DOF_NAMES[:6]}")
        print(f"[INFO] Isaac first 6 joints: {LAB_DOF_NAMES[:6]}")
        if self.cfg.sim.print_mapping:
            print("[INFO] Isaac action -> MuJoCo joint mapping:")
            for lab_idx, joint_name in enumerate(LAB_DOF_NAMES):
                mj_idx = mujoco_indices[joint_name]
                print(
                    f"  action[{lab_idx:02d}] {joint_name} -> "
                    f"mj[{mj_idx:02d}], qpos[{self.qpos_addr_mj[mj_idx]}], ctrl[{self.ctrl_addr_mj[mj_idx]}]"
                )

    def set_initial_pose(self) -> None:
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_addr_mj] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] Initial height: {self.data.qpos[2]:.3f} m")

    def init_depth_camera(self) -> None:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg.robot.camera_name)
        if camera_id < 0:
            print(f"[WARNING] MuJoCo camera '{self.cfg.robot.camera_name}' is missing. VP is unavailable.")
            self.set_estimator("op", announce=False)
            return

        print(
            f"[INFO] Initializing depth renderer: "
            f"{self.cfg.robot.camera_width}x{self.cfg.robot.camera_height}, camera={self.cfg.robot.camera_name}"
        )
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.cfg.robot.camera_height,
            width=self.cfg.robot.camera_width,
        )
        self.renderer.enable_depth_rendering()

        if self.cfg.robot.show_depth:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            self.ax.set_title("G1 RENet Depth Camera")

    def init_joystick(self) -> None:
        if JoyStick is None:
            self.use_joystick = False
            print(f"[WARNING] Controller support is unavailable: {JOYSTICK_IMPORT_ERROR}. Command will stay zero.")
            return

        try:
            self.joystick = JoyStick()
            self.use_joystick = True
            print("[INFO] Xbox controller connected.")
        except RuntimeError as exc:
            self.use_joystick = False
            print(f"[WARNING] Controller init failed: {exc}. Command will stay zero.")

    def mj_to_lab(self, array_mj: np.ndarray) -> np.ndarray:
        return array_mj[self.mujoco_to_lab_idx]

    def lab_to_mj(self, array_lab: np.ndarray) -> np.ndarray:
        return array_lab[self.lab_to_mujoco_idx]

    def get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        return np.array(
            [
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ],
            dtype=np.float32,
        )

    def get_proprio_obs(self) -> np.ndarray:
        dof_pos_mj = self.data.qpos[self.qpos_addr_mj].copy()
        dof_vel_mj = self.data.qvel[self.qvel_addr_mj].copy()
        ang_vel_body = self.data.qvel[3:6].copy()
        projected_gravity = self.get_gravity_orientation(self.data.qpos[3:7].copy())

        joint_pos_lab = self.mj_to_lab(dof_pos_mj - self.default_dof_pos)
        joint_vel_lab = self.mj_to_lab(dof_vel_mj)

        obs = np.concatenate(
            [
                ang_vel_body,
                projected_gravity,
                self.command_vel,
                joint_pos_lab,
                joint_vel_lab,
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.cfg.sim.num_obs_per_step:
            raise RuntimeError(f"Expected single proprio obs dim {self.cfg.sim.num_obs_per_step}, got {obs.shape[0]}.")

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def build_policy_obs(self) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        proprio_history = self.get_proprio_obs()
        proprio_tensor = torch.from_numpy(proprio_history).float()
        depth_flat = torch.stack(list(self.depth_buffer)).reshape(-1).float()
        estimator_mask = torch.tensor([self.estimator_mask_value], dtype=torch.float32)
        policy_obs = torch.cat([proprio_tensor, depth_flat, estimator_mask], dim=-1).unsqueeze(0)

        expected_dim = (
            self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
            + self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
            + self.cfg.renet.estimator_mask_dim
        )
        if policy_obs.shape[-1] != expected_dim:
            raise RuntimeError(f"Expected RENet actor obs dim {expected_dim}, got {policy_obs.shape[-1]}.")
        return policy_obs, proprio_history, depth_flat

    def update_depth_vision(self, noise_active: bool = False) -> None:
        if self.renderer is None:
            return

        self.renderer.update_scene(self.data, camera=self.cfg.robot.camera_name)
        raw_depth = self.renderer.render()
        if self.cfg.robot.debug_depth:
            print(f"[DEBUG] Raw depth min={raw_depth.min():.3f}, max={raw_depth.max():.3f}")

        if self.cfg.robot.show_depth and self.ax is not None:
            vis_depth = np.clip(raw_depth, self.cfg.robot.depth_min, self.cfg.robot.depth_max)
            vis_depth = (vis_depth - self.cfg.robot.depth_min) / (self.cfg.robot.depth_max - self.cfg.robot.depth_min)
            vis_img = 1.0 - vis_depth
            if self.im is None:
                self.im = self.ax.imshow(vis_img, cmap="gray", vmin=0, vmax=1)
            else:
                self.im.set_data(vis_img)
            title = f"G1 RENet Depth Camera [{self.current_mode_text()}]"
            if noise_active:
                title += " [COLLAPSE]"
            self.ax.set_title(title, color="red" if noise_active else "black", fontweight="bold" if noise_active else "normal")
            self.fig.canvas.flush_events()

        depth_tensor = torch.tensor(raw_depth.copy(), dtype=torch.float32)
        crop_up, crop_down, crop_left, crop_right = self.cfg.robot.depth_crop
        height_end = self.cfg.robot.camera_height - crop_down
        width_end = self.cfg.robot.camera_width - crop_right
        depth_cropped = depth_tensor[crop_up:height_end, crop_left:width_end]

        depth_cropped[torch.isinf(depth_cropped)] = self.cfg.robot.depth_max
        depth_cropped[torch.isnan(depth_cropped)] = self.cfg.robot.depth_max

        if noise_active:
            depth_cropped = self.apply_depth_collapse(depth_cropped)

        depth_blurred = self.blur_transform(depth_cropped.unsqueeze(0)).squeeze(0)
        clip_min = 0.0 if noise_active else self.cfg.robot.depth_min
        depth_clipped = torch.clip(depth_blurred, min=clip_min, max=self.cfg.robot.depth_max)
        depth_normalized = depth_clipped / self.cfg.robot.depth_max

        if self.cfg.robot.debug_depth:
            print(
                "[DEBUG] RENet depth "
                f"shape={tuple(depth_normalized.shape)}, "
                f"min={depth_normalized.min().item():.3f}, "
                f"max={depth_normalized.max().item():.3f}, "
                f"mean={depth_normalized.mean().item():.3f}"
            )

        self.depth_buffer.append(depth_normalized)

    def apply_depth_collapse(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        # Y is used as an explicit visual-collapse test. OP/VP selection itself
        # still comes only from the estimator mask.
        return torch.zeros_like(depth_tensor)

    def calculate_gait_para(self) -> None:
        if not self.cfg.gait.enable:
            return
        t = self.episode_length_buf * self.dt
        self.phase = (t % self.cfg.gait.period) / self.cfg.gait.period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.cfg.gait.offset) % 1.0
        self.leg_phase[0] = self.phase_left
        self.leg_phase[1] = self.phase_right

    def apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.cfg.command.deadzone else value

    def update_command_from_joystick(self) -> tuple[bool, bool]:
        if not self.use_joystick:
            return False, False

        self.joystick.update()
        if self.joystick.is_button_pressed(JoystickButton.SELECT):
            print("[INFO] Controller SELECT pressed, exiting.")
            return True, False

        if self.joystick.is_button_pressed(JoystickButton.A):
            self.command_vel[:] = 0.0

        if self.joystick.is_button_released(JoystickButton.START):
            print("[INFO] Controller START released, resetting robot.")
            self.reset_robot()

        if self.joystick.is_button_pressed(JoystickButton.L1):
            self.set_estimator("op")
        if self.joystick.is_button_pressed(JoystickButton.R1):
            self.set_estimator("vp")

        ax1 = self.apply_deadzone(self.joystick.get_axis_value(1))
        ax0 = self.apply_deadzone(self.joystick.get_axis_value(0))
        ax3 = self.apply_deadzone(self.joystick.get_axis_value(3))

        self.command_vel[0] = np.clip(-ax1 * 1.0, *self.cfg.command.lin_vel_x)
        self.command_vel[1] = np.clip(-ax0 * 0.5, *self.cfg.command.lin_vel_y)
        self.command_vel[2] = np.clip(-ax3 * 1.57, *self.cfg.command.ang_vel_z)
        depth_noise_active = self.estimator == "vp" and self.joystick.is_button_pressed(JoystickButton.Y)
        return False, depth_noise_active

    def position_control(self) -> np.ndarray:
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab_to_mj(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.qpos_addr_mj]
        dq = self.data.qvel[self.qvel_addr_mj]
        tau = (target_q - q) * self.kps - dq * self.kds
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def reset_robot(self) -> None:
        self.action[:] = 0.0
        self.command_vel[:] = 0.0
        self.episode_length_buf = 0
        self.leg_phase[:] = 0.0
        self.obs_history[:] = 0.0
        for frame in self.depth_buffer:
            frame.zero_()
        self.set_initial_pose()
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_proprio_obs()
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision()

    def stabilize_robot(self, duration: float = 2.0) -> None:
        target_pos = self.default_dof_pos.copy()
        num_steps = int(duration / self.cfg.sim.dt)
        for i in range(num_steps):
            self.data.qpos[0:2] = 0.0
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[0:6] = 0.0

            tau = self.pd_control(target_pos)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.ctrl_addr_mj] = tau
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                if i % 10 == 0:
                    self.viewer.sync()
                time.sleep(self.cfg.sim.dt)

        self.obs_history[:] = 0.0
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_proprio_obs()
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision()
        print(f"[INFO] Stabilized height: {self.data.qpos[2]:.4f} m")

    def run(self) -> None:
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.stabilize_robot()

        print("\n[INFO] Xbox: left stick = vx/vy, right stick X = yaw rate, A = stop, START = reset, SELECT = exit")
        print("[INFO] RENet: L1 = OP/mask 1, R1 = VP/mask 0. Hold Y in VP mode to feed collapsed depth.")
        print(f"[INFO] Current RENet mode: {self.current_mode_text()}")
        print("[INFO] Press Ctrl+C in terminal to exit.\n")

        debug_counter = 0
        try:
            while self.viewer.is_running():
                should_exit, depth_noise_active = self.update_command_from_joystick()
                if should_exit:
                    break

                if self.episode_length_buf % self.cfg.robot.depth_update_interval == 0:
                    self.update_depth_vision(noise_active=depth_noise_active)

                policy_obs, proprio_history, depth_flat = self.build_policy_obs()
                with torch.inference_mode():
                    action_tensor = self.policy(policy_obs)
                self.action[:] = action_tensor.squeeze(0).detach().cpu().numpy()[: self.num_actions]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

                debug_counter += 1
                if debug_counter <= 3:
                    latest_obs = proprio_history[-self.cfg.sim.num_obs_per_step :]
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  policy obs shape: {tuple(policy_obs.shape)}")
                    print(f"  estimator: {self.estimator.upper()}, mask={self.estimator_mask_value:.1f}")
                    print(f"  latest proprio first 9: {latest_obs[:9]}")
                    print(f"  depth flat shape: {tuple(depth_flat.shape)}, mean={depth_flat.mean().item():.3f}")
                    print(f"  command: {self.command_vel}")
                    print(f"  action first 6 (Isaac order): {self.action[:6]}")

                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:] = 0.0
                    self.data.ctrl[self.ctrl_addr_mj] = tau
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()

                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)

                self.episode_length_buf += 1
                self.calculate_gait_para()

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"mode={self.current_mode_text()}, "
                        f"cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], "
                        f"phase=[{self.leg_phase[0]:.2f}, {self.leg_phase[1]:.2f}], "
                        f"h={self.data.qpos[2]:.3f}m"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            if self.viewer is not None:
                self.viewer.close()
            if self.renderer is not None:
                self.renderer.close()
            if self.fig is not None:
                plt.close(self.fig)
            if self._temp_mjcf_dir is not None:
                self._temp_mjcf_dir.cleanup()
            print("[INFO] Simulation finished.")


def find_latest_exported_policy(log_root: str) -> str:
    candidates = []
    if os.path.isdir(log_root):
        for run_name in os.listdir(log_root):
            policy_path = os.path.join(log_root, run_name, "exported", "policy.pt")
            if os.path.isfile(policy_path):
                candidates.append(policy_path)
    if not candidates:
        return os.path.join(log_root, "exported", "policy.pt")
    return max(candidates, key=os.path.getmtime)


def get_available_scenes(mjcf_dir: str) -> dict[str, str]:
    scenes = {}
    if os.path.isdir(mjcf_dir):
        for filename in os.listdir(mjcf_dir):
            if filename in ["scene.xml", "flat_scene.xml", "rough_scene.xml", "slope_scene.xml"]:
                scenes[filename.replace(".xml", "").replace("_scene", "")] = os.path.join(mjcf_dir, filename)
            elif filename.endswith("_scene.xml"):
                scenes[filename.replace("_scene.xml", "")] = os.path.join(mjcf_dir, filename)
    return scenes


def main() -> None:
    legged_lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    mjcf_dir = os.path.join(legged_lab_root, "legged_lab/assets/g1/g1_23dof")

    default_policy = find_latest_exported_policy(os.path.join(legged_lab_root, "logs/g1_renet"))
    default_model = os.path.join(mjcf_dir, "g1_23dof_rev_1_0.xml")
    available_scenes = get_available_scenes(mjcf_dir)
    scene_names = list(available_scenes.keys())

    parser = argparse.ArgumentParser(
        description="G1 RENet 23DOF Sim2Sim. Observation layout: proprio history | depth history | estimator mask.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--policy", type=str, default=default_policy, help="Path to exported g1_renet policy.pt")
    parser.add_argument("--model", type=str, default=default_model, help="MuJoCo XML path used when --scene is omitted")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=scene_names if scene_names else None,
        help=f"Scene name. Available: {', '.join(scene_names) if scene_names else 'none'}",
    )
    parser.add_argument("--scene-file", type=str, default=None, help="Custom scene XML path, higher priority than --scene")
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument("--list-scenes", action="store_true", help="List available scene XML files")
    parser.add_argument("--check-mapping", action="store_true", help="Print joint/actuator mapping and exit")
    parser.add_argument("--no-depth-view", action="store_true", help="Disable matplotlib depth image window")
    parser.add_argument("--debug-depth", action="store_true", help="Print depth min/max statistics")
    parser.add_argument(
        "--no-auto-camera",
        action="store_true",
        help="Do not create a temporary depth_camera when the 23DOF XML lacks one",
    )
    args = parser.parse_args()

    if args.list_scenes:
        print("\nAvailable scenes:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"Scene directory: {mjcf_dir}")
        return

    if args.scene_file:
        model_path = args.scene_file
    elif args.scene:
        model_path = available_scenes[args.scene]
    else:
        model_path = args.model

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file does not exist: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"[ERROR] MuJoCo XML does not exist: {model_path}")
        sys.exit(1)

    cfg = G1RENetSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    cfg.sim.print_mapping = args.check_mapping
    cfg.robot.show_depth = not args.no_depth_view
    cfg.robot.debug_depth = args.debug_depth
    cfg.robot.auto_patch_camera = not args.no_auto_camera

    print("=" * 60)
    print("G1 RENet 23DOF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"MuJoCo XML: {model_path}")
    print(f"Initial estimator: {cfg.renet.default_estimator.upper()} (controller L1=OP, R1=VP)")
    if args.scene:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    runner = G1RENetMujocoRunner(
        cfg=cfg,
        policy_path=args.policy,
        model_path=model_path,
    )
    if args.check_mapping:
        print("[INFO] Mapping check finished; viewer was not launched.")
        return

    runner.run()


if __name__ == "__main__":
    main()
