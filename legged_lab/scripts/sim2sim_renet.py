"""G1 23DOF LP/VP RENet deployment in MuJoCo.

Observation layout: proprio[780] | LiDAR elevation[187] | depth[1152] | mask[1].
The estimator mask is 1 for LP and 0 for VP.
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torchvision.transforms as T


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from legged_lab.perception.rolling_elevation_map import RollingElevationMap  # noqa: E402

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

# Keep the same self-return set as the IsaacLab GroupedRayCaster training scene.
G1_23DOF_LINKS = [
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
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
        joint_armature = 0.01
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

    class lidar:
        body_name = "torso_link"
        offset_pos = (0.0002835, 0.00003, 0.428434)
        # WXYZ quaternion for URDF rpy=(pi, 0.0511207, 0).
        offset_quat_wxyz = (0.0, 0.9996733521, 0.0, -0.0255575638)
        point_rate = 200_000
        scan_period = 0.1
        downsample = 5
        min_distance = 0.1
        max_distance = 5.0
        random_phase = True
        angle_file = os.path.join(
            REPO_ROOT,
            "legged_lab/sensors/grouped_ray_caster/data/mid360_angles.npy",
        )

    class elevation_map:
        history_frames = 8
        history_seconds = 0.8
        resolution = 0.1
        internal_size = (3.2, 3.2)
        output_size = (1.6, 1.0)
        invalid_height = -1.0
        height_clip = (-1.0, 1.0)
        height_scan_offset = 0.5
        height_scale = 1.0

    class odometry:
        delay_seconds = 0.04
        position_white_std = 0.01
        yaw_white_std = math.radians(0.5)
        xy_random_walk = 0.005
        z_random_walk = 0.002
        yaw_random_walk = math.radians(0.2)

    class renet:
        # Current convention: 1.0 = LP, 0.0 = VP.
        estimator_mask_dim = 1
        default_estimator = "lp"


def quat_wxyz_to_matrix(quat: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Convert a WXYZ quaternion to a 3x3 rotation matrix."""

    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_yaw(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


@dataclass
class TorsoPose:
    position: np.ndarray
    rotation: np.ndarray

    @property
    def yaw(self) -> float:
        return rotation_yaw(self.rotation)

    def clone(self) -> "TorsoPose":
        return TorsoPose(self.position.copy(), self.rotation.copy())


class Mid360ScanPattern:
    """Time-ordered, downsampled MID-360 direction packets."""

    def __init__(self, cfg: G1RENetSim2SimCfg, seed: int):
        self.cfg = cfg
        quantized = np.load(cfg.lidar.angle_file, allow_pickle=False)
        if quantized.ndim != 2 or quantized.shape[1] != 2 or quantized.dtype != np.uint16:
            raise RuntimeError(
                "MID-360 angle resource must be uint16 with shape (num_samples, 2)."
            )
        angles = np.deg2rad(quantized.astype(np.float64) / 100.0)
        azimuth = angles[:, 0]
        zenith = angles[:, 1]
        sin_zenith = np.sin(zenith)
        self.sequence = np.column_stack(
            (
                sin_zenith * np.cos(azimuth),
                sin_zenith * np.sin(azimuth),
                np.cos(zenith),
            )
        )
        exact_source_count = cfg.lidar.point_rate * cfg.lidar.scan_period
        self.source_points_per_scan = int(round(exact_source_count))
        if not math.isclose(exact_source_count, self.source_points_per_scan):
            raise ValueError("LiDAR point_rate * scan_period must be an integer.")
        self.sample_offsets = np.arange(
            0,
            self.source_points_per_scan,
            cfg.lidar.downsample,
            dtype=np.int64,
        )
        self.rays_per_scan = int(self.sample_offsets.size)
        self._rng = np.random.default_rng(seed)
        self.cursor = 0
        self.reset()

    def reset(self) -> None:
        self.cursor = (
            int(self._rng.integers(0, self.sequence.shape[0]))
            if self.cfg.lidar.random_phase
            else 0
        )

    def next_directions(self) -> np.ndarray:
        indices = (self.cursor + self.sample_offsets) % self.sequence.shape[0]
        directions = self.sequence[indices]
        self.cursor = (self.cursor + self.source_points_per_scan) % self.sequence.shape[0]
        return directions


class PoseSourceManager:
    """Provide ground-truth or a reproducible delayed/noisy odometry pose."""

    def __init__(self, cfg: G1RENetSim2SimCfg, dt: float, seed: int, source: str):
        self.cfg = cfg
        self.dt = float(dt)
        self.delay_steps = max(0, int(round(cfg.odometry.delay_seconds / self.dt)))
        self._rng = np.random.default_rng(seed)
        self.source = "truth"
        self._history = deque(maxlen=self.delay_steps + 1)
        self._position_walk = np.zeros(3, dtype=np.float64)
        self._yaw_walk = 0.0
        self._initialized = False
        self.set_source(source)

    def set_source(self, source: str, truth_pose: TorsoPose | None = None) -> None:
        source = source.lower()
        if source not in {"truth", "noisy_odom"}:
            raise ValueError("Pose source must be 'truth' or 'noisy_odom'.")
        self.source = source
        if truth_pose is not None:
            self.reset(truth_pose)

    def reset(self, truth_pose: TorsoPose) -> None:
        self._history.clear()
        for _ in range(self.delay_steps + 1):
            self._history.append(truth_pose.clone())
        self._position_walk.fill(0.0)
        self._yaw_walk = 0.0
        self._initialized = True

    def step(self, truth_pose: TorsoPose) -> TorsoPose:
        if not self._initialized:
            self.reset(truth_pose)
        self._history.append(truth_pose.clone())
        delayed = self._history[0]

        walk_scale = math.sqrt(self.dt)
        self._position_walk[:2] += self._rng.normal(
            0.0, self.cfg.odometry.xy_random_walk * walk_scale, size=2
        )
        self._position_walk[2] += self._rng.normal(
            0.0, self.cfg.odometry.z_random_walk * walk_scale
        )
        self._yaw_walk += float(
            self._rng.normal(0.0, self.cfg.odometry.yaw_random_walk * walk_scale)
        )

        if self.source == "truth":
            return truth_pose.clone()

        white_position = self._rng.normal(
            0.0, self.cfg.odometry.position_white_std, size=3
        )
        white_yaw = float(self._rng.normal(0.0, self.cfg.odometry.yaw_white_std))
        position = delayed.position + self._position_walk + white_position
        rotation = yaw_rotation(self._yaw_walk + white_yaw) @ delayed.rotation
        return TorsoPose(position, rotation)


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
    def __init__(
        self,
        cfg: G1RENetSim2SimCfg,
        policy_path: str,
        model_path: str,
        *,
        estimator: str | None = None,
        pose_source: str = "truth",
        seed: int = 0,
        lidar_dropout: bool = False,
        depth_collapse: bool = False,
        show_lidar_map: bool = False,
        debug_lidar: bool = False,
        headless: bool = False,
    ):
        self.cfg = cfg
        self.seed = int(seed)
        self.headless = bool(headless)
        self.pose_source_name = pose_source.lower()
        self.lidar_dropout = bool(lidar_dropout)
        self.depth_collapse = bool(depth_collapse)
        self.show_lidar_map = bool(show_lidar_map)
        self.debug_lidar = bool(debug_lidar)
        self.estimator = (estimator or self.cfg.renet.default_estimator).lower()
        if self.estimator not in {"lp", "vp"}:
            raise ValueError("Default RENet estimator must be 'lp' or 'vp'.")
        self.estimator_mask_value = self.estimator_to_mask(self.estimator)
        self._temp_mjcf_dir = None

        load_model_path = model_path
        if self.cfg.robot.auto_patch_camera:
            load_model_path, self._temp_mjcf_dir = prepare_mjcf_with_depth_camera(model_path, self.cfg)

        print(f"[INFO] Loading MuJoCo model: {load_model_path}")
        self.model = mujoco.MjModel.from_xml_path(load_model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        self.configure_sensor_geom_groups()

        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        self.init_lidar()
        self.init_depth_camera()
        self.init_joystick()
        self.validate_policy_contract()

        control_hz = 1.0 / (self.cfg.sim.dt * self.cfg.sim.decimation)
        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        lidar_dim = self.elevation_map.output_rows * self.elevation_map.output_cols
        depth_dim = self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
        total_actor_dim = proprio_dim + lidar_dim + depth_dim + self.cfg.renet.estimator_mask_dim
        print(f"[INFO] Control frequency: {control_hz:.1f} Hz")
        print(f"[INFO] RENet estimator: {self.estimator.upper()} (mask={self.estimator_mask_value:.1f})")
        print(
            f"[INFO] Actor obs: proprio={proprio_dim}, LiDAR={lidar_dim}, "
            f"depth={depth_dim}, mask=1, total={total_actor_dim}"
        )

    @staticmethod
    def estimator_to_mask(estimator: str) -> float:
        return 1.0 if estimator == "lp" else 0.0

    def set_estimator(self, estimator: str, announce: bool = True) -> None:
        estimator = estimator.lower()
        if estimator not in {"lp", "vp"}:
            raise ValueError("RENet estimator must be 'lp' or 'vp'.")
        if estimator == "vp" and self.renderer is None:
            if announce:
                print("[WARNING] VP requested, but depth renderer is unavailable. Staying on LP.")
            estimator = "lp"
        if estimator == self.estimator:
            return
        self.estimator = estimator
        self.estimator_mask_value = self.estimator_to_mask(estimator)
        if announce:
            print(f"[INFO] RENet estimator switched to {self.estimator.upper()} (mask={self.estimator_mask_value:.1f})")

    def set_pose_source(self, source: str, announce: bool = True) -> None:
        truth_pose = self.get_truth_torso_pose()
        self.pose_provider.set_source(source, truth_pose)
        self.pose_source_name = self.pose_provider.source
        self.current_map_pose = truth_pose.clone()
        self.elevation_map.reset()
        self.scan_count = 0
        if announce:
            print(f"[INFO] Elevation-map pose source switched to {self.pose_source_name}; map history cleared.")

    def current_mode_text(self) -> str:
        return f"{self.estimator.upper()} mask={self.estimator_mask_value:.1f}"

    def configure_sensor_geom_groups(self) -> None:
        """Expose terrain plus the training-time G1 self-return link visuals."""

        free_joints = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_joints.size == 0:
            raise RuntimeError("G1 MuJoCo model has no free joint/root body.")
        robot_root_body = int(self.model.jnt_bodyid[int(free_joints[0])])

        def is_robot_body(body_id: int) -> bool:
            current = int(body_id)
            while current > 0:
                if current == robot_root_body:
                    return True
                current = int(self.model.body_parentid[current])
            return False

        allowed = set(G1_23DOF_LINKS)
        visible_robot_geoms = 0
        hidden_robot_geoms = 0
        static_geoms = 0
        visible_bodies = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if not is_robot_body(body_id):
                self.model.geom_group[geom_id] = 0
                static_geoms += 1
                continue
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name in allowed and int(self.model.geom_group[geom_id]) == 1:
                self.model.geom_group[geom_id] = 1
                visible_robot_geoms += 1
                visible_bodies.add(body_name)
            else:
                # Group 5 stays collision-active, but is hidden from LiDAR and depth.
                self.model.geom_group[geom_id] = 5
                hidden_robot_geoms += 1

        self.sensor_geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.sensor_scene_option = mujoco.MjvOption()
        self.sensor_scene_option.geomgroup[:] = 0
        self.sensor_scene_option.geomgroup[0:2] = 1
        self.ray_visible_body_names = visible_bodies
        print(
            "[INFO] Sensor geom groups: "
            f"terrain/static={static_geoms}, allowed self visuals={visible_robot_geoms}, "
            f"hidden robot geoms={hidden_robot_geoms}."
        )

    def init_lidar(self) -> None:
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.lidar.body_name
        )
        if self.torso_body_id < 0:
            raise RuntimeError(f"MuJoCo body '{self.cfg.lidar.body_name}' is missing.")
        self.lidar_offset_pos = np.asarray(self.cfg.lidar.offset_pos, dtype=np.float64)
        self.lidar_offset_rotation = quat_wxyz_to_matrix(self.cfg.lidar.offset_quat_wxyz)
        self.lidar_pattern = Mid360ScanPattern(self.cfg, self.seed)
        expected_rays = int(
            round(self.cfg.lidar.point_rate * self.cfg.lidar.scan_period)
            // self.cfg.lidar.downsample
        )
        if self.lidar_pattern.rays_per_scan != expected_rays:
            raise RuntimeError(
                f"Expected {expected_rays} MID-360 rays, got {self.lidar_pattern.rays_per_scan}."
            )
        self.lidar_scan_interval = int(round(self.cfg.lidar.scan_period / self.dt))
        if not math.isclose(
            self.lidar_scan_interval * self.dt,
            self.cfg.lidar.scan_period,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("LiDAR scan period must be divisible by the control period.")
        self.elevation_map = RollingElevationMap(
            num_envs=1,
            device="cpu",
            history_frames=self.cfg.elevation_map.history_frames,
            history_seconds=self.cfg.elevation_map.history_seconds,
            resolution=self.cfg.elevation_map.resolution,
            internal_size=self.cfg.elevation_map.internal_size,
            output_size=self.cfg.elevation_map.output_size,
            invalid_height=self.cfg.elevation_map.invalid_height,
            height_clip=self.cfg.elevation_map.height_clip,
        )
        self.pose_provider = PoseSourceManager(
            self.cfg, self.dt, self.seed + 1, self.pose_source_name
        )
        truth_pose = self.get_truth_torso_pose()
        self.pose_provider.reset(truth_pose)
        self.current_truth_pose = truth_pose
        self.current_map_pose = truth_pose.clone()
        self.scan_count = 0
        self.latest_lidar_height = torch.full(
            (self.elevation_map.output_rows, self.elevation_map.output_cols),
            self.cfg.elevation_map.invalid_height,
            dtype=torch.float32,
        )
        self.latest_lidar_confidence = torch.zeros_like(self.latest_lidar_height)
        self.latest_lidar_valid_fraction = 0.0
        self.last_lidar_hit_count = 0
        self.lidar_geomid = np.empty(self.lidar_pattern.rays_per_scan, dtype=np.int32)
        self.lidar_distance = np.empty(self.lidar_pattern.rays_per_scan, dtype=np.float64)
        print(
            f"[INFO] MID-360: {self.lidar_pattern.rays_per_scan} rays/scan, "
            f"period={self.cfg.lidar.scan_period:.3f}s, range="
            f"[{self.cfg.lidar.min_distance:.1f}, {self.cfg.lidar.max_distance:.1f}]m."
        )
        if self.show_lidar_map:
            plt.ion()
            self.lidar_fig, self.lidar_axes = plt.subplots(1, 2, figsize=(8, 3))
            self.lidar_fig.suptitle("G1 MID-360 rolling elevation map")

    def validate_policy_contract(self) -> None:
        """Reject stale exports and verify the current LP/VP deployment contract."""

        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        lidar_dim = self.elevation_map.output_rows * self.elevation_map.output_cols
        depth_dim = self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
        obs_dim = proprio_dim + lidar_dim + depth_dim + self.cfg.renet.estimator_mask_dim
        mask_index = obs_dim - 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        base = 0.05 * torch.randn(1, obs_dim, generator=generator)
        lp_obs = base.clone()
        lp_obs[:, mask_index] = 1.0
        vp_obs = base.clone()
        vp_obs[:, mask_index] = 0.0
        lp_depth_changed = lp_obs.clone()
        lp_depth_changed[:, proprio_dim + lidar_dim : mask_index] += 0.5
        vp_lidar_changed = vp_obs.clone()
        vp_lidar_changed[:, proprio_dim : proprio_dim + lidar_dim] += 0.5
        lp_lidar_changed = lp_obs.clone()
        lp_lidar_changed[:, proprio_dim : proprio_dim + lidar_dim] += 0.5
        vp_depth_changed = vp_obs.clone()
        vp_depth_changed[:, proprio_dim + lidar_dim : mask_index] += 0.5
        try:
            with torch.inference_mode():
                lp_action = self.policy(lp_obs)
                vp_action = self.policy(vp_obs)
                lp_depth_action = self.policy(lp_depth_changed)
                vp_lidar_action = self.policy(vp_lidar_changed)
                lp_lidar_action = self.policy(lp_lidar_changed)
                vp_depth_action = self.policy(vp_depth_changed)
        except Exception as exc:
            raise RuntimeError(
                "Policy does not accept the current 2120-dimensional LP/VP observation. "
                "Re-export it with play_new_renet.py or export.py."
            ) from exc
        if tuple(lp_action.shape) != (1, self.num_actions) or not torch.isfinite(lp_action).all():
            raise RuntimeError(
                f"Policy output must be finite with shape (1, {self.num_actions}), "
                f"got {tuple(lp_action.shape)}."
            )
        if torch.allclose(lp_action, vp_action, rtol=1.0e-7, atol=1.0e-8):
            raise RuntimeError(
                "Policy ignores the current estimator mask; this is likely a stale export."
            )
        if not torch.allclose(lp_action, lp_depth_action, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError("LP action depends on inactive depth input; export contract is invalid.")
        if not torch.allclose(vp_action, vp_lidar_action, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError("VP action depends on inactive LiDAR input; export contract is invalid.")
        if torch.allclose(lp_action, lp_lidar_action, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError("LP action is insensitive to LiDAR; export contract is invalid.")
        if torch.allclose(vp_action, vp_depth_action, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError("VP action is insensitive to depth; export contract is invalid.")
        print(f"[INFO] Policy contract validated: input={obs_dim}, output={self.num_actions}, LP/VP dynamic mask.")

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
        self.lidar_fig = None
        self.lidar_axes = None
        self.lidar_images = None
        self.joystick = None
        self.use_joystick = False
        self._joystick_lidar_dropout = False
        self._joystick_depth_collapse = False

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
            self.set_estimator("lp", announce=False)
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
        if self.headless:
            self.use_joystick = False
            return
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

    def get_truth_torso_pose(self) -> TorsoPose:
        position = np.asarray(self.data.xpos[self.torso_body_id], dtype=np.float64).copy()
        rotation = np.asarray(self.data.xmat[self.torso_body_id], dtype=np.float64).reshape(3, 3).copy()
        return TorsoPose(position, rotation)

    def refresh_pose_estimate(self) -> None:
        self.current_truth_pose = self.get_truth_torso_pose()
        self.current_map_pose = self.pose_provider.step(self.current_truth_pose)

    def _raycast_mid360(self, directions_sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast one physical packet and return world hits, sensor origin, and geom IDs."""

        truth = self.current_truth_pose
        sensor_position = truth.position + truth.rotation @ self.lidar_offset_pos
        directions_world = directions_sensor @ self.lidar_offset_rotation.T @ truth.rotation.T
        self.lidar_geomid.fill(-1)
        self.lidar_distance.fill(np.inf)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            sensor_position,
            np.ascontiguousarray(directions_world.reshape(-1), dtype=np.float64),
            self.sensor_geomgroup,
            1,
            -1,
            self.lidar_geomid,
            self.lidar_distance,
            directions_world.shape[0],
            self.cfg.lidar.max_distance,
        )
        valid = (
            (self.lidar_geomid >= 0)
            & np.isfinite(self.lidar_distance)
            & (self.lidar_distance >= self.cfg.lidar.min_distance)
            & (self.lidar_distance <= self.cfg.lidar.max_distance)
        )
        hits_world = np.full((directions_world.shape[0], 3), np.inf, dtype=np.float64)
        hits_world[valid] = (
            sensor_position[None, :]
            + self.lidar_distance[valid, None] * directions_world[valid]
        )
        return hits_world, sensor_position, self.lidar_geomid.copy()

    @staticmethod
    def _transform_points_between_pose_frames(
        hits_world: np.ndarray,
        truth_pose: TorsoPose,
        map_pose: TorsoPose,
    ) -> np.ndarray:
        """Express physical hits in the selected odometry/world frame."""

        transformed = np.full_like(hits_world, np.inf)
        valid = np.isfinite(hits_world).all(axis=1)
        local = (hits_world[valid] - truth_pose.position) @ truth_pose.rotation
        transformed[valid] = local @ map_pose.rotation.T + map_pose.position
        return transformed

    def update_lidar_scan(self, dropout_active: bool = False) -> bool:
        """Advance the scan phase and ingest one packet unless it is dropped."""

        directions_sensor = self.lidar_pattern.next_directions()
        if dropout_active:
            return False
        hits_world, _, geom_ids = self._raycast_mid360(directions_sensor)
        hits_map = self._transform_points_between_pose_frames(
            hits_world,
            self.current_truth_pose,
            self.current_map_pose,
        )
        sensor_position_map = (
            self.current_map_pose.position
            + self.current_map_pose.rotation @ self.lidar_offset_pos
        )
        self.scan_count += 1
        self.elevation_map.update_from_point_cloud(
            torch.from_numpy(hits_map.astype(np.float32, copy=False)).unsqueeze(0),
            torch.from_numpy(sensor_position_map.astype(np.float32, copy=False)).unsqueeze(0),
            torch.from_numpy(self.current_map_pose.position.astype(np.float32, copy=False)).unsqueeze(0),
            torch.tensor([self.current_map_pose.yaw], dtype=torch.float32),
            torch.tensor([self.scan_count], dtype=torch.long),
            torch.tensor([self.data.time], dtype=torch.float32),
            min_distance=self.cfg.lidar.min_distance,
            max_distance=self.cfg.lidar.max_distance,
        )
        self.last_lidar_hit_count = int(np.isfinite(hits_map).all(axis=1).sum())
        if self.debug_lidar:
            names = []
            for geom_id in np.unique(geom_ids[geom_ids >= 0]):
                body_id = int(self.model.geom_bodyid[int(geom_id)])
                names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world")
            sensor_z = float(sensor_position_map[2])
            print(
                f"[LIDAR] scan={self.scan_count}, hits={self.last_lidar_hit_count}/"
                f"{self.lidar_pattern.rays_per_scan}, sensor_z={sensor_z:.3f}, "
                f"bodies={sorted(set(names))}"
            )
        return True

    def get_lidar_observation(self) -> torch.Tensor:
        pose = self.current_map_pose
        height, confidence = self.elevation_map.get_observation(
            torch.from_numpy(pose.position.astype(np.float32, copy=False)).unsqueeze(0),
            torch.tensor([pose.yaw], dtype=torch.float32),
            torch.tensor([self.data.time], dtype=torch.float32),
            height_offset=self.cfg.elevation_map.height_scan_offset,
            height_scale=self.cfg.elevation_map.height_scale,
        )
        self.latest_lidar_height = height[0]
        self.latest_lidar_confidence = confidence[0]
        self.latest_lidar_valid_fraction = float((confidence[0] > 0.0).float().mean().item())
        if not torch.isfinite(self.latest_lidar_height).all():
            raise RuntimeError("LiDAR elevation observation contains NaN/Inf.")
        self.update_lidar_visualization()
        return self.latest_lidar_height

    def update_lidar_visualization(self) -> None:
        if not self.show_lidar_map or self.lidar_axes is None:
            return
        arrays = (
            self.latest_lidar_height.detach().cpu().numpy(),
            (self.latest_lidar_confidence > 0.0).detach().cpu().numpy(),
        )
        titles = ("relative height", "validity")
        cmaps = ("terrain", "gray")
        if self.lidar_images is None:
            self.lidar_images = []
            for axis, array, title, cmap in zip(self.lidar_axes, arrays, titles, cmaps):
                image = axis.imshow(array, origin="lower", cmap=cmap, vmin=-1.0 if title == "relative height" else 0.0, vmax=1.0)
                axis.set_title(title)
                self.lidar_images.append(image)
        else:
            for image, array in zip(self.lidar_images, arrays):
                image.set_data(array)
        self.lidar_fig.canvas.flush_events()

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

    def build_policy_obs(self) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        proprio_history = self.get_proprio_obs()
        proprio_tensor = torch.from_numpy(proprio_history).float()
        lidar_flat = self.get_lidar_observation().reshape(-1).float()
        depth_flat = torch.stack(list(self.depth_buffer)).reshape(-1).float()
        estimator_mask = torch.tensor([self.estimator_mask_value], dtype=torch.float32)
        policy_obs = torch.cat(
            [proprio_tensor, lidar_flat, depth_flat, estimator_mask], dim=-1
        ).unsqueeze(0)

        expected_dim = (
            self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
            + self.elevation_map.output_rows * self.elevation_map.output_cols
            + self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
            + self.cfg.renet.estimator_mask_dim
        )
        if policy_obs.shape[-1] != expected_dim:
            raise RuntimeError(f"Expected RENet actor obs dim {expected_dim}, got {policy_obs.shape[-1]}.")
        return policy_obs, proprio_history, lidar_flat, depth_flat

    def update_depth_vision(self, noise_active: bool = False) -> None:
        if self.renderer is None:
            return

        self.renderer.update_scene(
            self.data,
            camera=self.cfg.robot.camera_name,
            scene_option=self.sensor_scene_option,
        )
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
        # Y is used as an explicit visual-collapse test. LP/VP selection itself
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

    def update_command_from_joystick(self) -> tuple[bool, bool, bool]:
        if not self.use_joystick:
            return False, self.depth_collapse, self.lidar_dropout

        self.joystick.update()
        if self.joystick.is_button_pressed(JoystickButton.SELECT):
            print("[INFO] Controller SELECT pressed, exiting.")
            return True, self.depth_collapse, self.lidar_dropout

        if self.joystick.is_button_pressed(JoystickButton.A):
            self.command_vel[:] = 0.0

        if self.joystick.is_button_released(JoystickButton.START):
            print("[INFO] Controller START released, resetting robot.")
            self.reset_robot()

        if self.joystick.is_button_pressed(JoystickButton.L1):
            self.set_estimator("lp")
        if self.joystick.is_button_pressed(JoystickButton.R1):
            self.set_estimator("vp")
        if self.joystick.is_button_released(JoystickButton.B):
            next_source = "noisy_odom" if self.pose_source_name == "truth" else "truth"
            self.set_pose_source(next_source)

        ax1 = self.apply_deadzone(self.joystick.get_axis_value(1))
        ax0 = self.apply_deadzone(self.joystick.get_axis_value(0))
        ax3 = self.apply_deadzone(self.joystick.get_axis_value(3))

        self.command_vel[0] = np.clip(-ax1 * 1.0, *self.cfg.command.lin_vel_x)
        self.command_vel[1] = np.clip(-ax0 * 0.5, *self.cfg.command.lin_vel_y)
        self.command_vel[2] = np.clip(-ax3 * 1.57, *self.cfg.command.ang_vel_z)
        depth_noise_active = self.depth_collapse or self.joystick.is_button_pressed(JoystickButton.Y)
        lidar_dropout_active = self.lidar_dropout or self.joystick.is_button_pressed(JoystickButton.X)
        return False, depth_noise_active, lidar_dropout_active

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
        self.lidar_pattern.reset()
        self.elevation_map.reset()
        self.scan_count = 0
        truth_pose = self.get_truth_torso_pose()
        self.pose_provider.reset(truth_pose)
        self.current_truth_pose = truth_pose
        self.current_map_pose = self.pose_provider.step(truth_pose)
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_proprio_obs()
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision(noise_active=self.depth_collapse)
        self.update_lidar_scan(dropout_active=self.lidar_dropout)
        self.get_lidar_observation()

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
        self.elevation_map.reset()
        self.scan_count = 0
        self.lidar_pattern.reset()
        truth_pose = self.get_truth_torso_pose()
        self.pose_provider.reset(truth_pose)
        self.current_truth_pose = truth_pose
        self.current_map_pose = self.pose_provider.step(truth_pose)
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_proprio_obs()
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision(noise_active=self.depth_collapse)
        self.update_lidar_scan(dropout_active=self.lidar_dropout)
        self.get_lidar_observation()
        print(f"[INFO] Stabilized height: {self.data.qpos[2]:.4f} m")

    def run(self) -> None:
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.stabilize_robot()

        print("\n[INFO] Xbox: left stick=vx/vy, right stick X=yaw rate, A=stop, START=reset, SELECT=exit")
        print("[INFO] RENet: L1=LP/mask 1, R1=VP/mask 0, B=truth/noisy_odom.")
        print("[INFO] Fault injection: hold X=drop LiDAR packets, hold Y=zero depth frames.")
        print(f"[INFO] Current RENet mode: {self.current_mode_text()}")
        print(f"[INFO] Map pose source: {self.pose_source_name}")
        print(f"[INFO] Duration: {self.cfg.sim.sim_duration:.1f}s after stabilization. Press Ctrl+C to exit.\n")

        debug_counter = 0
        control_start_time = float(self.data.time)
        try:
            while (
                float(self.data.time) - control_start_time < self.cfg.sim.sim_duration
                and (self.viewer is None or self.viewer.is_running())
            ):
                should_exit, depth_noise_active, lidar_dropout_active = self.update_command_from_joystick()
                if should_exit:
                    break

                self.refresh_pose_estimate()
                if self.episode_length_buf % self.cfg.robot.depth_update_interval == 0:
                    self.update_depth_vision(noise_active=depth_noise_active)
                if (
                    self.episode_length_buf > 0
                    and self.episode_length_buf % self.lidar_scan_interval == 0
                ):
                    self.update_lidar_scan(dropout_active=lidar_dropout_active)

                policy_obs, proprio_history, lidar_flat, depth_flat = self.build_policy_obs()
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
                    print(
                        f"  LiDAR flat shape: {tuple(lidar_flat.shape)}, "
                        f"valid={self.latest_lidar_valid_fraction:.1%}, scans={self.scan_count}"
                    )
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
                    if self.viewer is not None:
                        self.viewer.sync()

                    elapsed = time.time() - step_start
                    if self.viewer is not None and self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)

                self.episode_length_buf += 1
                self.calculate_gait_para()

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"mode={self.current_mode_text()}, "
                        f"pose={self.pose_source_name}, lidar_valid={self.latest_lidar_valid_fraction:.1%}, "
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
            if self.lidar_fig is not None:
                plt.close(self.lidar_fig)
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
        description=(
            "G1 RENet 23DOF Sim2Sim. Observation layout: "
            "proprio history | LiDAR elevation | depth history | estimator mask."
        ),
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
    parser.add_argument(
        "--estimator",
        choices=("lp", "vp"),
        default="lp",
        help="Initial control branch: LP uses LiDAR and VP uses depth",
    )
    parser.add_argument(
        "--pose-source",
        choices=("truth", "noisy_odom"),
        default="truth",
        help="Pose source used by the rolling elevation map",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for LiDAR phase and noisy odometry")
    parser.add_argument("--lidar-dropout", action="store_true", help="Persistently drop new LiDAR packets")
    parser.add_argument("--depth-collapse", action="store_true", help="Persistently feed zero depth frames")
    parser.add_argument("--show-lidar-map", action="store_true", help="Show LiDAR height and validity maps")
    parser.add_argument("--debug-lidar", action="store_true", help="Print per-packet LiDAR diagnostics")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer")
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

    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.headless and args.show_lidar_map:
        parser.error("--show-lidar-map cannot be used together with --headless")

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
    cfg.robot.show_depth = not args.no_depth_view and not args.headless
    cfg.robot.debug_depth = args.debug_depth
    cfg.robot.auto_patch_camera = not args.no_auto_camera

    print("=" * 60)
    print("G1 RENet 23DOF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"MuJoCo XML: {model_path}")
    print(f"Initial estimator: {args.estimator.upper()} (controller L1=LP, R1=VP)")
    print(f"Elevation-map pose source: {args.pose_source}")
    if args.scene:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    runner = G1RENetMujocoRunner(
        cfg=cfg,
        policy_path=args.policy,
        model_path=model_path,
        estimator=args.estimator,
        pose_source=args.pose_source,
        seed=args.seed,
        lidar_dropout=args.lidar_dropout,
        depth_collapse=args.depth_collapse,
        show_lidar_map=args.show_lidar_map,
        debug_lidar=args.debug_lidar,
        headless=args.headless,
    )
    if args.check_mapping:
        print("[INFO] Mapping check finished; viewer was not launched.")
        return

    runner.run()


if __name__ == "__main__":
    main()
