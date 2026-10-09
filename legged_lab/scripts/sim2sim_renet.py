"""MuJoCo evaluation of the pgtt_full_atten RENet LP/VP policy.

The LP branch receives a single 11x17 ground-only height snapshot sampled by
native MuJoCo vertical rays. The VP branch receives two depth frames.
"""

import argparse
import json
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


# Keep the MuJoCo deployment grid self-contained. These values deliberately
# duplicate the policy observation contract instead of importing the
# IsaacLab/training-side PGTT pattern or its coordinate builder.
PGTT_HEIGHT_SOURCE = "pgtt_full_atten"


@dataclass(frozen=True)
class MujocoPgttGridCfg:
    """Native MuJoCo vertical-ray grid in torso-yaw coordinates."""

    shape: tuple[int, int]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float = 0.1
    invalid_height: float = -1.1

    @property
    def ray_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def sample_xy(self) -> np.ndarray:
        """Return ``[num_rays, 2]`` in y-ascending/x-ascending order."""

        rows, cols = self.shape
        x = np.linspace(self.x_range[0], self.x_range[1], cols, dtype=np.float64)
        y = np.linspace(self.y_range[0], self.y_range[1], rows, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def make_mujoco_pgtt_grid(source: str) -> MujocoPgttGridCfg:
    """Build one deployment grid without consulting IsaacLab-side helpers."""

    if source != PGTT_HEIGHT_SOURCE:
        raise ValueError("MuJoCo RENet supports only pgtt_full_atten")
    cfg = MujocoPgttGridCfg(shape=(11, 17), x_range=(-0.8, 0.8), y_range=(-0.5, 0.5))

    rows = round((cfg.y_range[1] - cfg.y_range[0]) / cfg.resolution) + 1
    cols = round((cfg.x_range[1] - cfg.x_range[0]) / cfg.resolution) + 1
    if (rows, cols) != cfg.shape:
        raise RuntimeError(
            f"MuJoCo PGTT grid range/resolution implies {(rows, cols)}, got {cfg.shape}."
        )
    return cfg


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
        # Gaussian blur applied to every VP depth frame. It used to be a fixed
        # sigma=1.0; it is now adjustable at runtime from the controller:
        # L3 toggles it, D-pad up/down (and the UP/DOWN buttons) change sigma,
        # R3 restores the initial sigma.
        depth_blur_enable = True
        depth_blur_sigma = 1.0
        depth_blur_kernel_size = 3
        depth_blur_sigma_min = 0.0
        depth_blur_sigma_max = 4.0
        depth_blur_sigma_step = 0.5
        auto_patch_camera = True

    class lp_sensor:
        body_name = "torso_link"
        scan_period = 0.1

    class lp_height:
        max_packet_age = 0.8
        height_clip = (-1.0, 1.0)
        height_scan_offset = 0.5
        height_scale = 1.0


    class renet:
        # Current convention: 1.0 = LP, 0.0 = VP.
        estimator_mask_dim = 1
        default_estimator = "lp"
        lp_height_source = "pgtt_full_atten"
        pgtt_map_backend = "mujoco_ray"


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
        self.lidar_dropout = bool(lidar_dropout)
        self.depth_collapse = bool(depth_collapse)
        self.show_lidar_map = bool(show_lidar_map)
        self.debug_lidar = bool(debug_lidar)
        self.lp_height_source = self.cfg.renet.lp_height_source
        if self.lp_height_source != "pgtt_full_atten":
            raise ValueError("MuJoCo RENet supports only pgtt_full_atten")
        if self.cfg.renet.pgtt_map_backend != "mujoco_ray":
            raise ValueError("pgtt_map_backend must be 'mujoco_ray'.")
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

        print(f"[INFO] Loading policy: {policy_path}")
        extra_files = {"renet_contract.json": ""}
        self.policy = torch.jit.load(
            policy_path, map_location="cpu", _extra_files=extra_files
        )
        self.policy.eval()
        metadata_text = extra_files["renet_contract.json"]
        if isinstance(metadata_text, bytes):
            metadata_text = metadata_text.decode("utf-8")
        try:
            self.policy_metadata = json.loads(metadata_text) if metadata_text else None
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Exported policy contains invalid renet_contract.json metadata."
            ) from exc
        self.configure_sensor_geom_groups()

        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        self.init_lidar()
        self.init_depth_camera()
        self.init_joystick()
        self.validate_policy_contract()

        control_hz = 1.0 / (self.cfg.sim.dt * self.cfg.sim.decimation)
        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        lidar_dim = math.prod(self.lp_height_shape)
        depth_dim = self.cfg.robot.depth_history_frames * self.depth_height * self.depth_width
        total_actor_dim = proprio_dim + lidar_dim + depth_dim + self.cfg.renet.estimator_mask_dim
        print(f"[INFO] Control frequency: {control_hz:.1f} Hz")
        print(f"[INFO] RENet estimator: {self.estimator.upper()} (mask={self.estimator_mask_value:.1f})")
        print(
            f"[INFO] LP height source: {self.lp_height_source}, "
            f"shape={self.lp_height_shape}, invalid={self.lp_invalid_height}, "
            "backend=mujoco_ray"
        )
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
        ray_visible_robot_geoms = 0
        viewer_only_robot_geoms = 0
        hidden_collision_geoms = 0
        static_geoms = 0
        visible_bodies = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            original_group = int(self.model.geom_group[geom_id])
            if not is_robot_body(body_id):
                self.model.geom_group[geom_id] = 0
                static_geoms += 1
                continue
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if original_group == 1 and body_name in allowed:
                self.model.geom_group[geom_id] = 1
                ray_visible_robot_geoms += 1
                visible_bodies.add(body_name)
            elif original_group == 1:
                # Group 2 remains visible in the interactive viewer, but is
                # excluded by the LiDAR/depth sensor mask below.
                self.model.geom_group[geom_id] = 2
                viewer_only_robot_geoms += 1
            else:
                # Collision geoms stay physics-active but hidden from sensors
                # and from the normal visual rendering of the robot.
                self.model.geom_group[geom_id] = 5
                hidden_collision_geoms += 1

        self.sensor_geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        # Match the training scanner's ground-only query mesh.
        self.lidar_geomgroup = self.sensor_geomgroup.copy()
        self.lidar_geomgroup[1] = 0
        self.sensor_scene_option = mujoco.MjvOption()
        self.sensor_scene_option.geomgroup[:] = 0
        self.sensor_scene_option.geomgroup[0:2] = 1
        self.ray_visible_body_names = visible_bodies
        print(
            "[INFO] Sensor geom groups: "
            f"terrain/static={static_geoms}, allowed self visuals={ray_visible_robot_geoms}, "
            f"viewer-only robot visuals={viewer_only_robot_geoms}, "
            f"hidden collision geoms={hidden_collision_geoms}; "
            f"LiDAR self returns={'kept' if self.lidar_geomgroup[1] else 'filtered'}."
        )

    def init_lidar(self) -> None:
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.lp_sensor.body_name
        )
        if self.torso_body_id < 0:
            raise RuntimeError(f"MuJoCo body '{self.cfg.lp_sensor.body_name}' is missing.")
        self.lidar_scan_interval = int(round(self.cfg.lp_sensor.scan_period / self.dt))
        if not math.isclose(self.lidar_scan_interval * self.dt, self.cfg.lp_sensor.scan_period, abs_tol=1e-9):
            raise ValueError("LP scan period must be divisible by the control period.")
        self.pgtt_runtime_cfg = make_mujoco_pgtt_grid(self.lp_height_source)
        self.lp_height_shape = tuple(self.pgtt_runtime_cfg.shape)
        self.lp_invalid_height = self.pgtt_runtime_cfg.invalid_height
        truth_pose = self.get_truth_torso_pose()
        self.current_truth_pose = truth_pose
        self.scan_count = 0
        self.latest_lidar_height = torch.full(self.lp_height_shape, self.lp_invalid_height, dtype=torch.float32)
        self.latest_lidar_confidence = torch.zeros_like(self.latest_lidar_height)
        self.latest_lidar_valid_fraction = 0.0
        self.latest_lidar_scan_time = float("-inf")
        self.last_lidar_hit_count = 0
        self.pgtt_sample_xy = self.pgtt_runtime_cfg.sample_xy()
        self.pgtt_ray_geomid = np.full(1, -1, dtype=np.int32)
        print(f"[INFO] MuJoCo mj_ray -> pgtt_full_atten: shape={self.lp_height_shape}")
        if self.show_lidar_map:
            plt.ion()
            self.lidar_fig, self.lidar_axes = plt.subplots(1, 2, figsize=(8, 3))
            self.lidar_fig.suptitle("G1 pgtt_full_atten native MuJoCo mj_ray")

    def validate_policy_contract(self) -> None:
        """Reject stale exports and verify the current LP/VP deployment contract."""

        policy_metadata = getattr(self, "policy_metadata", None)
        if policy_metadata is None:
            raise RuntimeError("pgtt_full_atten requires embedded AME metadata; re-export the checkpoint.")
        if policy_metadata.get("lp_height_source") != "pgtt_full_atten":
            raise RuntimeError("Policy must be exported for pgtt_full_atten.")
        exported_contract = policy_metadata.get("lp_observation_contract", {})
        if tuple(exported_contract.get("shape", ())) != self.lp_height_shape:
            raise RuntimeError("Exported LP shape does not match the 11x17 height grid.")
        exported_encoder = policy_metadata.get("lp_encoder_contract")
        expected_encoder = {
            "version": 1,
            "encoder_type": "ame_point_cross_attention",
            "proprio_history_frames": 5,
            "height_history_frames": 1,
            "map_shape": [11, 17],
            "map_token_dim": 64,
            "attention_heads": 16,
            "sample_order": "y_ascending/x_ascending",
            "grid_x_range": [-0.8, 0.8],
            "grid_y_range": [-0.5, 0.5],
            "coordinate_features": "x_y_encoded_height",
            "uses_lp_gru": False,
            "reconstruction_region": "full_187",
        }
        if exported_encoder != expected_encoder:
            raise RuntimeError(
                "pgtt_full_atten policy is missing or has an incompatible AME "
                "point-cross-attention encoder contract. Re-export it with the "
                "current play_new_renet.py."
            )
        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        lidar_dim = math.prod(self.lp_height_shape)
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
                f"Policy does not accept the {obs_dim}-dimensional {self.lp_height_source} "
                "LP/VP observation. Re-export a checkpoint trained with the same "
                "--lp_height_source using play_new_renet.py."
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
        print(
            f"[INFO] Policy contract validated: source={self.lp_height_source}, "
            f"input={obs_dim}, output={self.num_actions}, LP/VP dynamic mask."
        )

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
        self.depth_blur_enabled = bool(self.cfg.robot.depth_blur_enable)
        self.depth_blur_sigma_min = float(self.cfg.robot.depth_blur_sigma_min)
        self.depth_blur_sigma_max = float(self.cfg.robot.depth_blur_sigma_max)
        if self.depth_blur_sigma_min > self.depth_blur_sigma_max:
            raise ValueError("depth_blur_sigma_min must not exceed depth_blur_sigma_max")
        self.depth_blur_sigma = min(
            max(float(self.cfg.robot.depth_blur_sigma), self.depth_blur_sigma_min),
            self.depth_blur_sigma_max,
        )
        self.blur_transform = self.make_depth_blur(self.depth_blur_sigma)

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
        self._depth_blur_buttons: dict[int, bool] = {}
        self._depth_blur_hat_vertical = 0

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


    def _update_pgtt_mujoco_ray_scan(self, dropout_active: bool = False) -> bool:
        """Query MuJoCo geoms with one native ``mj_ray`` per PGTT grid cell."""

        if dropout_active:
            return False
        if self.pgtt_sample_xy is None or self.pgtt_ray_geomid is None:
            raise RuntimeError("PGTT native MuJoCo ray backend was not initialized.")

        pose = self.current_truth_pose
        yaw = pose.yaw
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        local_x = self.pgtt_sample_xy[:, 0]
        local_y = self.pgtt_sample_xy[:, 1]
        starts = np.empty((self.pgtt_sample_xy.shape[0], 3), dtype=np.float64)
        starts[:, 0] = pose.position[0] + cos_yaw * local_x - sin_yaw * local_y
        starts[:, 1] = pose.position[1] + sin_yaw * local_x + cos_yaw * local_y
        starts[:, 2] = pose.position[2] + 20.0
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        distances = np.full(starts.shape[0], -1.0, dtype=np.float64)
        geom_ids = np.full(starts.shape[0], -1, dtype=np.int32)
        for ray_index, ray_start in enumerate(starts):
            self.pgtt_ray_geomid[0] = -1
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                ray_start,
                direction,
                self.lidar_geomgroup,
                1,
                -1,
                self.pgtt_ray_geomid,
            )
            if 0.0 <= distance <= 40.0:
                distances[ray_index] = distance
                geom_ids[ray_index] = self.pgtt_ray_geomid[0]

        valid = distances >= 0.0
        height = torch.full(
            self.lp_height_shape,
            self.lp_invalid_height,
            dtype=torch.float32,
        )
        confidence = torch.zeros(self.lp_height_shape, dtype=torch.float32)
        if np.any(valid):
            hit_z = starts[valid, 2] - distances[valid]
            encoded = np.clip(
                pose.position[2] - hit_z - self.cfg.lp_height.height_scan_offset,
                self.cfg.lp_height.height_clip[0],
                self.cfg.lp_height.height_clip[1],
            ).astype(np.float32, copy=False)
            flat_height = height.reshape(-1)
            flat_confidence = confidence.reshape(-1)
            valid_tensor = torch.from_numpy(valid)
            flat_height[valid_tensor] = torch.from_numpy(encoded)
            flat_confidence[valid_tensor] = 1.0

        self.scan_count += 1
        self.last_lidar_hit_count = int(valid.sum())
        self.latest_lidar_scan_time = float(self.data.time)
        self.latest_lidar_height = height
        self.latest_lidar_confidence = confidence
        self.latest_lidar_valid_fraction = float(valid.mean())
        if self.debug_lidar:
            names = []
            for geom_id in np.unique(geom_ids[geom_ids >= 0]):
                body_id = int(self.model.geom_bodyid[int(geom_id)])
                names.append(
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                    or "world"
                )
            print(
                f"[MUJOCO_RAY] scan={self.scan_count}, hits={self.last_lidar_hit_count}/"
                f"{starts.shape[0]}, torso_z={pose.position[2]:.3f}, "
                f"bodies={sorted(set(names))}"
            )
        return True


    def update_lidar_scan(self, dropout_active: bool = False) -> bool:
        return self._update_pgtt_mujoco_ray_scan(dropout_active)

    def get_lidar_observation(self) -> torch.Tensor:
        if float(self.data.time) - self.latest_lidar_scan_time > self.cfg.lp_height.max_packet_age:
            self.latest_lidar_height.fill_(self.lp_invalid_height)
            self.latest_lidar_confidence.zero_()
            self.latest_lidar_valid_fraction = 0.0
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
            + math.prod(self.lp_height_shape)
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
            if self.depth_blur_enabled and self.depth_blur_sigma > 0.0:
                title += f" [BLUR sigma={self.depth_blur_sigma:.2f}]"
            else:
                title += " [BLUR OFF]"
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

        if self.depth_blur_enabled and self.depth_blur_sigma > 0.0:
            depth_blurred = self.blur_transform(depth_cropped.unsqueeze(0)).squeeze(0)
        else:
            depth_blurred = depth_cropped
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

    def apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.cfg.command.deadzone else value

    def make_depth_blur(self, sigma: float):
        """Build the Gaussian blur applied to the VP depth frames."""
        kernel_size = int(self.cfg.robot.depth_blur_kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("depth_blur_kernel_size must be a positive odd integer")
        # torchvision divides by sigma**2, so sigma must stay strictly positive.
        return T.GaussianBlur(kernel_size=kernel_size, sigma=max(float(sigma), 1.0e-3))

    def _button_rose(self, button) -> bool:
        """Rising edge of a controller button; JoyStick only reports levels."""
        pressed = bool(self.joystick.is_button_pressed(button))
        previous = self._depth_blur_buttons.get(int(button), False)
        self._depth_blur_buttons[int(button)] = pressed
        return pressed and not previous

    def _dpad_vertical_rise(self) -> int:
        """+1/-1 when the D-pad hat leaves the centre, 0 while held or released."""
        hat = self.joystick.get_hat_direction(0)
        vertical = int(hat[1]) if hat else 0
        previous = self._depth_blur_hat_vertical
        self._depth_blur_hat_vertical = vertical
        if vertical == 0 or vertical == previous:
            return 0
        return 1 if vertical > 0 else -1

    def update_depth_blur_from_joystick(self) -> None:
        """Controller control of the VP depth Gaussian blur."""
        if not self.use_joystick:
            return

        if self._button_rose(JoystickButton.L3):
            self.depth_blur_enabled = not self.depth_blur_enabled
            print(
                f"[INFO] Depth Gaussian blur {'ON' if self.depth_blur_enabled else 'OFF'} "
                f"(sigma={self.depth_blur_sigma:.2f})."
            )
        if self._button_rose(JoystickButton.R3):
            self.depth_blur_sigma = min(
                max(float(self.cfg.robot.depth_blur_sigma), self.depth_blur_sigma_min),
                self.depth_blur_sigma_max,
            )
            self.blur_transform = self.make_depth_blur(self.depth_blur_sigma)
            print(f"[INFO] Depth Gaussian blur sigma reset to {self.depth_blur_sigma:.2f}.")

        step = float(self.cfg.robot.depth_blur_sigma_step)
        steps = self._dpad_vertical_rise()
        if self._button_rose(JoystickButton.UP):
            steps += 1
        if self._button_rose(JoystickButton.DOWN):
            steps -= 1
        if steps == 0:
            return

        sigma = min(
            max(self.depth_blur_sigma + steps * step, self.depth_blur_sigma_min),
            self.depth_blur_sigma_max,
        )
        if abs(sigma - self.depth_blur_sigma) < 1.0e-9:
            return
        self.depth_blur_sigma = sigma
        self.blur_transform = self.make_depth_blur(sigma)
        print(
            f"[INFO] Depth Gaussian blur sigma = {sigma:.2f} "
            f"({'ON' if self.depth_blur_enabled else 'OFF'})."
        )

    def update_command_from_joystick(self) -> tuple[bool, bool, bool]:
        if not self.use_joystick:
            return False, self.depth_collapse, self.lidar_dropout

        self.joystick.update()
        self.update_depth_blur_from_joystick()
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
        self.obs_history[:] = 0.0
        for frame in self.depth_buffer:
            frame.zero_()
        self.set_initial_pose()
        self.scan_count = 0
        truth_pose = self.get_truth_torso_pose()
        self.current_truth_pose = truth_pose
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
        self.scan_count = 0
        truth_pose = self.get_truth_torso_pose()
        self.current_truth_pose = truth_pose
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
            # Groups 0/1/2 are terrain, ray-visible robot visuals, and the
            # remaining robot visuals. Group 5 contains collision geometry.
            self.viewer.opt.geomgroup[:] = 0
            self.viewer.opt.geomgroup[0:3] = 1
        self.stabilize_robot()

        print("\n[INFO] Xbox: left stick=vx/vy, right stick X=yaw rate, A=stop, START=reset, SELECT=exit")
        print("[INFO] RENet: L1=LP/mask 1, R1=VP/mask 0.")
        print("[INFO] Fault injection: hold X=drop LiDAR packets, hold Y=zero depth frames.")
        print(
            "[INFO] VP depth Gaussian blur: "
            f"{'ON' if self.depth_blur_enabled else 'OFF'} "
            f"sigma={self.depth_blur_sigma:.2f} "
            f"(L3=toggle, D-pad up/down=sigma +/-{float(self.cfg.robot.depth_blur_sigma_step):.1f}, "
            f"R3=reset)."
        )
        print(f"[INFO] Current RENet mode: {self.current_mode_text()}")
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

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"mode={self.current_mode_text()}, "
                        f"lp_valid={self.latest_lidar_valid_fraction:.1%}, "
                        f"cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], "
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


def parse_lp_height_source(value: str) -> str:
    if value != "pgtt_full_atten":
        raise argparse.ArgumentTypeError("MuJoCo RENet supports only pgtt_full_atten")
    return value


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
    parser.add_argument(
        "--pgtt-map-backend",
        choices=("mujoco_ray",),
        default="mujoco_ray",
        help=(
            "AME height input uses native MuJoCo vertical rays."
        ),
    )
    parser.add_argument("--scene-file", type=str, default=None, help="Custom scene XML path, higher priority than --scene")
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--lp-height-source",
        "--lp_height_source",
        dest="lp_height_source",
        type=parse_lp_height_source,
        metavar="{pgtt_full_atten}",
        default="pgtt_full_atten",
        help="AME LP height contract (pgtt_full_atten)",
    )
    parser.add_argument(
        "--estimator",
        choices=("lp", "vp"),
        default="lp",
        help="Initial control branch: LP uses LiDAR and VP uses depth",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for simulation")
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
        "--no-depth-blur",
        action="store_true",
        help="Start with the VP depth Gaussian blur disabled (L3 toggles it at runtime)",
    )
    parser.add_argument(
        "--depth-blur-sigma",
        type=float,
        default=None,
        help="Initial Gaussian blur sigma for VP depth frames (default 1.0; D-pad up/down adjusts it)",
    )
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
    cfg.robot.depth_blur_enable = not args.no_depth_blur
    if args.depth_blur_sigma is not None:
        if args.depth_blur_sigma < 0.0:
            parser.error("--depth-blur-sigma must not be negative")
        cfg.robot.depth_blur_sigma = args.depth_blur_sigma
    cfg.robot.auto_patch_camera = not args.no_auto_camera
    cfg.renet.lp_height_source = args.lp_height_source
    cfg.renet.pgtt_map_backend = args.pgtt_map_backend

    print("=" * 60)
    print("G1 RENet 23DOF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"MuJoCo XML: {model_path}")
    print(f"Initial estimator: {args.estimator.upper()} (controller L1=LP, R1=VP)")
    print(f"Requested LP height contract: {args.lp_height_source or 'auto'}")
    print(f"Requested PGTT map backend: {args.pgtt_map_backend}")
    print(
        "VP depth Gaussian blur: "
        f"{'on' if cfg.robot.depth_blur_enable else 'off'} "
        f"(sigma={cfg.robot.depth_blur_sigma:g}, "
        f"range=[{cfg.robot.depth_blur_sigma_min:g}, {cfg.robot.depth_blur_sigma_max:g}], "
        f"step={cfg.robot.depth_blur_sigma_step:g})"
    )
    if args.scene:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    runner = G1RENetMujocoRunner(
        cfg=cfg,
        policy_path=args.policy,
        model_path=model_path,
        estimator=args.estimator,
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
