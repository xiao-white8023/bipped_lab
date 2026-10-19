"""Run the independent g1_depth TorchScript policy in MuJoCo.

The policy consumes ten proprioception frames and two metric-depth frames.
Native MuJoCo rays reproduce the training camera without an OpenGL renderer.
Isaac Lab, AMP components, and privileged terrain observations are not needed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF_DIR = REPO_ROOT / "legged_lab/assets/g1/g1_23dof"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default order in the current Isaac Lab G1 23DOF articulation. The loaded
# policy's named contract determines the actual observation/action order.
ISAAC_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

# Same visual self-return links as G1_23DOF_LINKS in the training asset.
SELF_RETURN_BODIES = frozenset((
    "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link",
))
PROPRIO_ORDER = ["angular_velocity", "projected_gravity", "command",
                 "joint_position_offset", "joint_velocity", "previous_action"]
SCALE_NAMES = ("ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions")
COMMAND_LIMITS = np.array([[-0.6, 1.0], [-0.5, 0.5], [-1.57, 1.57]])


@dataclass
class G1DepthSim2SimCfg:
    dt: float = 0.005
    decimation: int = 4
    duration: float = 100.0
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    init_height: float = 0.793
    headless: bool = False
    use_joystick: bool = True
    show_depth: bool = True
    debug_depth: bool = False
    depth_noise: bool = False
    proprio_noise: bool = False
    seed: int = 0

    def validate(self):
        if not math.isclose(self.dt, 0.005) or type(self.decimation) is not int or self.decimation != 4:
            raise ValueError("g1_depth sim2sim requires dt=0.005 and decimation=4.")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be finite and positive.")
        if not math.isfinite(self.init_height) or self.init_height <= 0:
            raise ValueError("init_height must be finite and positive.")
        if np.asarray(self.command).shape != (3,) or not np.isfinite(self.command).all():
            raise ValueError("command must contain three finite velocities.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be a nonnegative integer below 2**63-1.")


def joint_control_parameters(name: str) -> tuple[float, float, float, float, float]:
    """Return (default position, Kp, Kd, torque limit, armature) from G1_23CFG."""
    if name not in ISAAC_JOINT_NAMES:
        raise ValueError(f"Unsupported G1 joint: {name}")
    if "hip_pitch" in name:
        return -0.20, 200.0, 5.0, 88.0, 0.01
    if "hip_roll" in name:
        return 0.0, 150.0, 5.0, 139.0, 0.01
    if "hip_yaw" in name:
        return 0.0, 150.0, 5.0, 88.0, 0.01
    if "knee" in name:
        return 0.42, 200.0, 5.0, 139.0, 0.01
    if "ankle" in name:
        return (-0.23 if "pitch" in name else 0.0), 20.0, 2.0, 35.0, 0.01
    if name == "waist_yaw_joint":
        return 0.0, 200.0, 5.0, 88.0, 0.01
    if "shoulder_pitch" in name:
        return 0.35, 100.0, 2.0, 25.0, 0.01
    if "shoulder_roll" in name:
        return (0.18 if name.startswith("left") else -0.18), 100.0, 2.0, 25.0, 0.01
    if "shoulder_yaw" in name or "elbow" in name:
        return (0.87 if "elbow" in name else 0.0), 50.0, 2.0, 25.0, 0.01
    return 0.0, 15.0, 2.0, 25.0, 0.01


def validate_depth_contract(metadata: dict) -> dict:
    """Validate the embedded v1 inference contract before loading a scene."""
    if not isinstance(metadata, dict) or metadata.get("task") != "g1_depth" or metadata.get("version") != 1:
        raise ValueError("Expected an exported g1_depth v1 policy; export it with play_depth.py.")
    if metadata.get("input_shape") != ["batch", 1932] or metadata.get("output_shape") != ["batch", 23]:
        raise ValueError("g1_depth policy must accept [N,1932] and return [N,23].")
    c = metadata.get("observation_contract")
    if not isinstance(c, dict):
        raise ValueError("Missing depth observation contract; re-export with play_depth.py.")
    expected = {
        "version": 1, "task": "g1_depth", "actor_observation_dim": 1932,
        "critic_observation_dim": 1037, "action_dim": 23,
        "proprio_history_frames": 10, "proprio_frame_dim": 78,
        "proprio_order": PROPRIO_ORDER, "depth_shape": [2, 18, 32],
        "sequence_order": "oldest_to_current", "camera_size": [36, 64],
        "depth_normalization": "clip_metres_divide_depth_max",
        "depth_invalid_fill": "nonfinite_to_depth_max", "camera_convention": "world",
        "camera_ray_alignment": "yaw",
    }
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f"Incompatible g1_depth observation contract: {key}={c.get(key)!r}, expected {value!r}.")
    names = c.get("joint_names")
    if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
            or len(names) != 23 or set(names) != set(ISAAC_JOINT_NAMES)):
        raise ValueError("The depth policy must name all 23 distinct G1 joints.")
    positive = ("depth_capture_period_s", "depth_min_m", "depth_max_m", "camera_focal_length",
                "camera_horizontal_aperture", "camera_vertical_aperture", "control_dt_s",
                "action_scale", "clip_observations", "clip_actions")
    for key in positive:
        value = c.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Depth contract {key} must be finite and positive.")
    if not math.isclose(c["control_dt_s"], .02) or not math.isclose(c["depth_capture_period_s"], .1):
        raise ValueError("g1_depth requires 50 Hz control and 10 Hz depth capture.")
    if c["depth_min_m"] >= c["depth_max_m"]:
        raise ValueError("depth_min_m must be below depth_max_m.")
    scales = c.get("obs_scales")
    if not isinstance(scales, dict) or set(scales) != set(SCALE_NAMES):
        raise ValueError("Missing proprioception observation scales in depth contract.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scales.values()):
        raise ValueError("Observation scales must be finite numbers.")
    for key, size in (("camera_position", 3), ("camera_rotation", 4)):
        value = c.get(key)
        if (not isinstance(value, list) or len(value) != size or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
            raise ValueError(f"Invalid {key} in depth contract.")
    if not np.linalg.norm(c["camera_rotation"]) > 1e-8:
        raise ValueError("Camera rotation must be a nonzero WXYZ quaternion.")
    crop = c.get("depth_crop")
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int or v < 0 for v in crop):
        raise ValueError("depth_crop must contain four nonnegative integers.")
    if (36 - crop[0] - crop[1], 64 - crop[2] - crop[3]) != (18, 32):
        raise ValueError("Depth crop must produce 18x32 pixels.")
    return c


def load_depth_policy(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TorchScript policy not found: {path}")
    extra = {"depth_contract.json": ""}
    try:
        policy = torch.jit.load(str(path), map_location="cpu", _extra_files=extra).eval()
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Expected exported policy.pt, not a training checkpoint; use play_depth.py to export.") from exc
    if not extra["depth_contract.json"]:
        raise ValueError("Policy has no embedded depth_contract.json; re-export with play_depth.py.")
    metadata = json.loads(extra["depth_contract.json"])
    contract = validate_depth_contract(metadata)
    with torch.inference_mode():
        output = policy(torch.zeros(1, 1932))
    if not isinstance(output, torch.Tensor) or output.shape != (1, 23) or not torch.isfinite(output).all():
        raise ValueError("Depth policy must return 23 finite actions for a [1,1932] observation.")
    return policy, contract


def quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Expected a finite nonzero WXYZ quaternion.")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def pinhole_rays(contract: dict) -> np.ndarray:
    """Pixel-centre unit rays in the training camera's +X-forward frame."""
    height, width = contract["camera_size"]
    fx = width * contract["camera_focal_length"] / contract["camera_horizontal_aperture"]
    fy = height * contract["camera_focal_length"] / contract["camera_vertical_aperture"]
    u, v = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5, indexing="xy")
    rays = np.stack((np.ones_like(u), -(u-width/2)/fx, -(v-height/2)/fy), axis=-1)
    return np.ascontiguousarray((rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3))


def gaussian_blur_3x3(depth: torch.Tensor) -> torch.Tensor:
    """Same sigma=1, reflection padding as torchvision.transforms.GaussianBlur."""
    axis = torch.arange(-1, 2, dtype=depth.dtype, device=depth.device)
    kernel = torch.exp(-.5 * axis.square())
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    return F.conv2d(F.pad(depth[None, None], (1, 1, 1, 1), mode="reflect"), kernel)[0, 0]


def process_depth(raw_depth, contract: dict, *, noise: bool = False, generator=None) -> torch.Tensor:
    """Crop, optionally blur/add metric noise, clip, then divide by depth_max."""
    depth = torch.as_tensor(raw_depth, dtype=torch.float32).clone()
    height, width = contract["camera_size"]
    if depth.shape != (height, width):
        raise ValueError(f"Expected raw depth {(height, width)}, received {tuple(depth.shape)}.")
    up, down, left, right = contract["depth_crop"]
    depth = depth[up:height-down, left:width-right]
    minimum, maximum = contract["depth_min_m"], contract["depth_max_m"]
    depth = torch.where(torch.isfinite(depth) & (depth >= 0), depth, maximum)
    if noise:
        if torch.rand(1, generator=generator).item() < .5:
            depth = gaussian_blur_3x3(depth)
        normalized_distance = ((depth-minimum) / (maximum-minimum)).clamp(0, 1)
        sigma = .10 * normalized_distance.square()
        depth = depth + torch.randn(depth.shape, dtype=depth.dtype, generator=generator) * sigma
    return depth.clamp(minimum, maximum) / maximum


def proprio_noise_vector(scales: dict) -> np.ndarray:
    result = np.zeros(78, dtype=np.float32)
    result[:3] = .2 * scales["ang_vel"]
    result[3:6] = .05 * scales["projected_gravity"]
    result[9:32] = .01 * scales["joint_pos"]
    result[32:55] = 1.5 * scales["joint_vel"]
    return result


def _read_mjcf(path: Path) -> ET.Element:
    text = path.read_text()
    start = text.find("<mujoco")
    if start < 0:
        raise ValueError(f"No <mujoco> root in scene: {path}")
    return ET.fromstring(text[start:])


def prepare_mjcf(model_path: str | Path):
    """Copy include trees to a temporary directory and preserve source assets.

    Custom scenes replace only the included robot's built-in world floor. This
    prevents both duplicate geom names and an invisible floor filling gaps.
    """
    source = Path(model_path).expanduser().resolve()
    documents = {}

    def read(path, ancestors=()):
        if path in ancestors:
            raise ValueError(f"Cyclic MJCF include: {path}")
        if path in documents:
            return
        root = _read_mjcf(path)
        documents[path] = root
        for item in root.iter("include"):
            read((path.parent / item.attrib["file"]).resolve(), (*ancestors, path))

    read(source)
    robot_documents = {p for p, root in documents.items()
                       if any(j.tag == "freejoint" or j.get("type") == "free" for j in root.iter())}
    custom_terrain = any(p not in robot_documents and any(
        next(world.iter("geom"), None) is not None for world in root.findall("worldbody"))
        for p, root in documents.items())
    # MJCF asset directories are global, including a compiler inside an include.
    directories = {}
    for p, root in documents.items():
        for compiler in root.findall("compiler"):
            for key in ("assetdir", "meshdir", "texturedir"):
                if compiler.get(key):
                    directories[key] = (p.parent / compiler.get(key)).resolve()
    temporary = tempfile.TemporaryDirectory(prefix="g1_depth_mjcf_")
    try:
        copies = {p: Path(temporary.name) / f"scene_{i}.xml" for i, p in enumerate(documents)}
        for p, root in documents.items():
            if custom_terrain and p in robot_documents:
                for world in root.findall("worldbody"):
                    for geom in list(world.findall("geom")):
                        if geom.get("name") == "floor" and geom.get("type") == "plane":
                            world.remove(geom)
            for item in root.iter("include"):
                item.set("file", str(copies[(p.parent / item.attrib["file"]).resolve()]))
            for asset in root.findall("asset"):
                for item in asset.iter():
                    if item.get("file"):
                        key = {"mesh": "meshdir", "texture": "texturedir"}.get(item.tag, "assetdir")
                        directory = directories.get(key, directories.get("assetdir", p.parent))
                        item.set("file", str((directory / item.get("file")).resolve()))
            for compiler in root.findall("compiler"):
                for key in ("assetdir", "meshdir", "texturedir"):
                    compiler.attrib.pop(key, None)
                compiler.set("strippath", "false")
            ET.ElementTree(root).write(copies[p], encoding="unicode")
        return copies[source], temporary
    except BaseException:
        temporary.cleanup()
        raise


@dataclass
class JointMapping:
    names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    root_body: int
    root_qpos: int
    root_qvel: int
    default: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    armature: np.ndarray

    @classmethod
    def from_model(cls, model, names=ISAAC_JOINT_NAMES):
        names = tuple(names)
        if len(names) != 23 or set(names) != set(ISAAC_JOINT_NAMES):
            raise ValueError("Expected 23 distinct policy joint names.")
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        actual = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                  for i in range(model.njnt) if i not in free}
        if len(free) != 1 or actual != set(names) or model.nu != 23:
            raise ValueError("MuJoCo model must contain one free root, the 23 G1 joints and 23 motors.")
        joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids):
            raise ValueError("All controlled G1 joints must be hinges.")
        actuators = {}
        for i in range(model.nu):
            joint = int(model.actuator_trnid[i, 0])
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT or joint in actuators:
                raise ValueError("Each G1 joint requires exactly one direct motor actuator.")
            if (not np.allclose(model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
                    or model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                    or model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE
                    or model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE
                    or not math.isclose(model.actuator_gainprm[i, 0], 1.0)):
                raise ValueError("g1_depth requires unit-gear direct torque motors.")
            actuators[joint] = i
        if set(actuators) != set(joint_ids):
            raise ValueError("G1 joint/actuator mapping is incomplete.")
        root = int(free[0])
        root_body = int(model.jnt_bodyid[root])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body) != "pelvis":
            raise ValueError("The G1 floating root body must be pelvis.")
        params = np.asarray([joint_control_parameters(n) for n in names], dtype=np.float64)
        return cls(names, joint_ids, model.jnt_qposadr[joint_ids].copy(), model.jnt_dofadr[joint_ids].copy(),
                   np.array([actuators[j] for j in joint_ids]), root_body,
                   int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root]),
                   *(params[:, i].copy() for i in range(5)))

    def print_mapping(self):
        for i, name in enumerate(self.names):
            print(f"  action[{i:02d}] {name}: qpos[{self.qpos[i]}], qvel[{self.qvel[i]}], ctrl[{self.ctrl[i]}]")


def configure_sensor_geom_groups(model, root_body: int) -> np.ndarray:
    """Select training visual self returns while retaining physical collisions."""
    visible = set()
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i])
        parent = body
        while parent > 0 and parent != root_body:
            parent = int(model.body_parentid[parent])
        if parent != root_body:
            model.geom_group[i] = 0  # terrain and scene objects
        elif model.geom_group[i] == 1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            model.geom_group[i] = 1 if name in SELF_RETURN_BODIES else 2
            if name in SELF_RETURN_BODIES:
                visible.add(name)
        else:
            model.geom_group[i] = 5  # collisions remain active, hidden from camera
    if visible != SELF_RETURN_BODIES:
        raise ValueError(f"Missing training visual self-return meshes: {sorted(SELF_RETURN_BODIES-visible)}")
    return np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)


class G1DepthMujocoRunner:
    def __init__(self, cfg: G1DepthSim2SimCfg, policy_path: str | Path, model_path: str | Path):
        self.cfg = cfg
        cfg.validate()
        self.viewer = self.joystick = self.figure = self._plt = None
        self._temporary = None
        self._exit_event, self._reset_event, self._stop_event = (threading.Event() for _ in range(3))
        self._stopped = False
        self.policy_path = Path(policy_path).expanduser().resolve()
        self.source_model_path = Path(model_path).expanduser().resolve()
        try:
            torch.set_num_threads(1)
            self.policy, self.contract = load_depth_policy(self.policy_path)
            self.model_path, self._temporary = prepare_mjcf(self.source_model_path)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.model.opt.timestep = cfg.dt
            self.data = mujoco.MjData(self.model)
            self.mapping = JointMapping.from_model(self.model, self.contract["joint_names"])
            self.model.dof_armature[self.mapping.qvel] = self.mapping.armature
            self.sensor_geomgroup = configure_sensor_geom_groups(self.model, self.mapping.root_body)
            self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            if self.torso_id < 0:
                raise ValueError("MuJoCo G1 model has no torso_link for the depth camera.")
            self.camera_rays = pinhole_rays(self.contract)
            self.camera_rotation = quat_wxyz_to_matrix(self.contract["camera_rotation"])
            self.ray_distances = np.empty(len(self.camera_rays), dtype=np.float64)
            self.ray_geom_ids = np.empty(len(self.camera_rays), dtype=np.int32)
            self.depth_rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
            self.proprio_rng = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
            self.noise_vector = proprio_noise_vector(self.contract["obs_scales"])
            self.policy_dt = self.contract["control_dt_s"]
            self.command_vel = np.clip(np.asarray(cfg.command, dtype=np.float32), *COMMAND_LIMITS.T).astype(np.float32)
            self.action = np.zeros(23, dtype=np.float64)
            self.proprio_history = np.zeros((10, 78), dtype=np.float32)
            self.depth_history = torch.zeros(2, 18, 32)
            self.control_steps = self.capture_count = self.reset_count = 0
            self._mse_sum = np.zeros(3)
            self._height_min = math.inf
            self._max_action = self._max_torque = self._action_saturated = self._torque_saturated = 0.0
            self._depth_min, self._depth_max = math.inf, -math.inf
            self.reset_robot(count=False)
            print(f"[INFO] g1_depth policy: {self.policy_path}")
            print(f"[INFO] MuJoCo source: {self.source_model_path}")
            print(f"[INFO] obs=1932, actions=23; physics=200 Hz, policy=50 Hz, depth=10 Hz (native rays)")
            print(f"[INFO] command={self.command_vel.tolist()}, proprio_noise={cfg.proprio_noise}, depth_noise={cfg.depth_noise}")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def check_state(self):
        if not all(np.isfinite(v).all() for v in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
            raise RuntimeError("Nonfinite MuJoCo state; stopping g1_depth control.")
        for warning in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC):
            if self.data.warning[warning].number:
                raise RuntimeError(f"MuJoCo reported {warning.name}; stopping g1_depth control.")

    def reset_robot(self, *, count=True):
        if self._stopped and self.joystick is None:
            self.command_vel[:] = np.clip(self.cfg.command, *COMMAND_LIMITS.T)
        mujoco.mj_resetData(self.model, self.data)
        root = self.mapping.root_qpos
        self.data.qpos[root:root+3] = [0, 0, self.cfg.init_height]
        self.data.qpos[root+3:root+7] = [1, 0, 0, 0]
        self.data.qpos[self.mapping.qpos] = self.mapping.default
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.action.fill(0)
        self.proprio_history.fill(0)
        self.depth_history.zero_()
        self._last_proprio_time = self._last_capture_time = None
        self.capture_times = []
        self._stopped = False
        mujoco.mj_forward(self.model, self.data)
        self.build_policy_obs()
        if count:
            self.reset_count += 1

    def sample_proprio(self) -> np.ndarray:
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.mapping.root_body, velocity, 0)
        rotation = self.data.xmat[self.mapping.root_body].reshape(3, 3)
        # MuJoCo's local body velocity is in the inertial frame (ximat).
        # Training root_ang_vel_b uses the pelvis link frame (xmat).
        angular_body = rotation.T @ velocity[:3]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        s = self.contract["obs_scales"]
        result = np.concatenate((
            angular_body * s["ang_vel"], gravity * s["projected_gravity"],
            self.command_vel * s["commands"],
            (self.data.qpos[self.mapping.qpos] - self.mapping.default) * s["joint_pos"],
            self.data.qvel[self.mapping.qvel] * s["joint_vel"], self.action * s["actions"],
        )).astype(np.float32)
        if self.cfg.proprio_noise:
            result += (2 * torch.rand(78, generator=self.proprio_rng).numpy()-1) * self.noise_vector
        return result

    def sample_raw_depth(self) -> np.ndarray:
        torso_rotation = self.data.xmat[self.torso_id].reshape(3, 3)
        origin = self.data.xpos[self.torso_id] + torso_rotation @ np.asarray(self.contract["camera_position"])
        rotation = torso_rotation @ self.camera_rotation
        directions = np.ascontiguousarray(self.camera_rays @ rotation.T)
        cutoff = self.contract["depth_max_m"] / self.camera_rays[:, 0].min()
        mujoco.mj_multiRay(self.model, self.data, origin, directions.reshape(-1), self.sensor_geomgroup,
                          1, -1, self.ray_geom_ids, self.ray_distances, len(directions), cutoff)
        # distance_to_image_plane is the camera-forward component, not ray range.
        depth = self.ray_distances * self.camera_rays[:, 0]
        invalid = (self.ray_distances < 0) | (self.ray_geom_ids < 0) | ~np.isfinite(depth)
        depth[invalid] = np.inf
        return depth.reshape(self.contract["camera_size"])

    def update_depth(self):
        now = float(self.data.time)
        if self._last_capture_time is not None and now-self._last_capture_time < self.contract["depth_capture_period_s"]-1e-8:
            return
        self.raw_depth = self.sample_raw_depth()
        current = process_depth(self.raw_depth, self.contract, noise=self.cfg.depth_noise, generator=self.depth_rng)
        if self._last_capture_time is None:
            self.depth_history[:] = current
        else:
            self.depth_history[0] = self.depth_history[1].clone()
            self.depth_history[1] = current
        self._last_capture_time = now
        self.capture_times.append(now)
        self.capture_count += 1
        self._depth_min = min(self._depth_min, float(current.min()))
        self._depth_max = max(self._depth_max, float(current.max()))
        if self.cfg.debug_depth:
            print(f"[DEPTH] t={now:.3f}s valid={np.isfinite(self.raw_depth).mean():.3f} "
                  f"normalized=[{float(current.min()):.4f},{float(current.max()):.4f}]")
        self.update_depth_view()

    def build_policy_obs(self) -> torch.Tensor:
        self.check_state()
        mujoco.mj_forward(self.model, self.data)
        now = float(self.data.time)
        if self._last_proprio_time != now:
            current = self.sample_proprio()
            if self._last_proprio_time is None:
                self.proprio_history[:] = current
            else:
                self.proprio_history[:-1] = self.proprio_history[1:].copy()
                self.proprio_history[-1] = current
            self._last_proprio_time = now
        # Commands may change between calls at the same simulation instant.
        self.proprio_history[-1, 6:9] = self.command_vel * self.contract["obs_scales"]["commands"]
        self.update_depth()
        clip = self.contract["clip_observations"]
        proprio = torch.from_numpy(np.clip(self.proprio_history, -clip, clip).reshape(-1))
        obs = torch.cat((proprio, self.depth_history.reshape(-1))).unsqueeze(0)
        if obs.shape != (1, 1932) or not torch.isfinite(obs).all():
            raise RuntimeError("Nonfinite or malformed g1_depth policy observation.")
        return obs

    def pd_control(self, target) -> np.ndarray:
        torque = ((target-self.data.qpos[self.mapping.qpos]) * self.mapping.kp
                  - self.data.qvel[self.mapping.qvel] * self.mapping.kd)
        if not np.isfinite(torque).all():
            raise RuntimeError("Nonfinite G1 PD torque.")
        self._torque_saturated += float((np.abs(torque) > self.mapping.torque_limit).mean())
        return np.clip(torque, -self.mapping.torque_limit, self.mapping.torque_limit)

    def step(self):
        observation = self.build_policy_obs()
        with torch.inference_mode():
            output = self.policy(observation)
        if not isinstance(output, torch.Tensor) or output.shape != (1, 23) or not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite or malformed g1_depth action; stopping control.")
        raw = output[0].detach().cpu().numpy().astype(np.float64)
        clip = self.contract["clip_actions"]
        self._action_saturated += float((np.abs(raw) > clip).mean())
        self.action[:] = np.clip(raw, -clip, clip)
        self._max_action = max(self._max_action, float(np.abs(self.action).max()))
        target = self.mapping.default + self.contract["action_scale"] * self.action
        for _ in range(self.cfg.decimation):
            torque = self.pd_control(target)
            self.data.ctrl[:] = 0
            self.data.ctrl[self.mapping.ctrl] = torque
            self._max_torque = max(self._max_torque, float(np.abs(torque).max()))
            mujoco.mj_step(self.model, self.data)
            self.check_state()
        mujoco.mj_forward(self.model, self.data)
        self.control_steps += 1
        rotation = self.data.xmat[self.mapping.root_body].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        velocity = self.data.qvel[self.mapping.root_qvel:self.mapping.root_qvel+3]
        vx, vy = velocity[:2]
        self._mse_sum[:2] += (np.array([vx*math.cos(yaw)+vy*math.sin(yaw),
                                      -vx*math.sin(yaw)+vy*math.cos(yaw)]) - self.command_vel[:2])**2
        angular = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.mapping.root_body, angular, 0)
        self._mse_sum[2] += (angular[2]-self.command_vel[2])**2
        self._height_min = min(self._height_min, float(self.data.qpos[self.mapping.root_qpos+2]))
        return observation

    def key_callback(self, key):
        if key in (ord("R"), ord("r")):
            self._reset_event.set()
        elif key == 32:
            self._stop_event.set()
        elif key == 256:  # GLFW_KEY_ESCAPE
            self._exit_event.set()

    def init_interfaces(self):
        if self.cfg.headless:
            return
        import glfw

        if not glfw.init():
            raise RuntimeError("Cannot initialize the MuJoCo display. Check DISPLAY/X11 access or use --headless.")
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        with self.viewer.lock():
            self.viewer.opt.geomgroup[:] = 0
            self.viewer.opt.geomgroup[:3] = 1
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mapping.root_body
            self.viewer.cam.distance = 3.0
        if self.cfg.use_joystick:
            try:
                from legged_lab.scripts.joystick import JoyStick, JoystickButton

                self.joystick = JoyStick()
                self.joystick_buttons = JoystickButton
            except (ImportError, RuntimeError) as exc:
                pygame = sys.modules.get("pygame")
                if pygame is not None:
                    pygame.quit()
                print(f"[INFO] No usable joystick; keeping fixed velocity commands: {exc}")
        if self.cfg.show_depth:
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            self.figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            self.depth_artists = [axes[0].imshow(np.zeros((36, 64)), vmin=self.contract["depth_min_m"],
                                                vmax=self.contract["depth_max_m"], cmap="viridis"),
                                  axes[1].imshow(np.zeros((18, 32)), vmin=0, vmax=1, cmap="viridis")]
            axes[0].set_title("Raw depth (m)")
            axes[1].set_title("Policy depth (normalized)")
            self.figure.tight_layout()
            self.figure.canvas.mpl_connect("key_press_event", lambda e: self.key_callback(
                {"r": ord("R"), " ": 32, "escape": 256}.get(e.key, -1)))
            self.update_depth_view()
        print("[INFO] R / Start: reset and resume input; Space: stop; Esc / Select: exit; A: hold to stop.")

    def update_depth_view(self):
        if self.figure is None or not self._plt.fignum_exists(self.figure.number):
            return
        maximum = self.contract["depth_max_m"]
        self.depth_artists[0].set_data(np.clip(np.nan_to_num(self.raw_depth, nan=maximum,
                                                          posinf=maximum, neginf=maximum),
                                              self.contract["depth_min_m"], maximum))
        self.depth_artists[1].set_data(self.depth_history[-1].numpy())
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def update_inputs(self):
        if self._stop_event.is_set():
            self._stop_event.clear()
            self._stopped = True
            self.command_vel[:] = 0
        if self._reset_event.is_set():
            self._reset_event.clear()
            self.reset_robot()
        if self.joystick is None:
            return
        self.joystick.update()
        buttons = self.joystick_buttons
        if self.joystick.is_button_pressed(buttons.SELECT):
            self._exit_event.set()
            return
        if self.joystick.is_button_released(buttons.START):
            self.reset_robot()
        if self.joystick.is_button_pressed(buttons.A) or self._stopped:
            self.command_vel[:] = 0
            return
        axes = np.array([-self.joystick.get_axis_value(i) for i in (1, 0, 3)])
        axes[np.abs(axes) < .1] = 0
        positive = COMMAND_LIMITS[:, 1]
        negative = -COMMAND_LIMITS[:, 0]
        self.command_vel[:] = np.clip(axes * np.where(axes >= 0, positive, negative), *COMMAND_LIMITS.T)

    def summary(self, wall_time: float = 0.0) -> dict:
        count = max(1, self.control_steps)
        return {
            "task": "g1_depth", "policy": str(self.policy_path), "model": str(self.source_model_path),
            "observation_dim": 1932, "action_dim": 23, "depth_backend": "mujoco_multi_ray",
            "command": self.command_vel.tolist(), "control_steps": self.control_steps,
            "simulation_time_s": self.control_steps*self.policy_dt,
            "wall_time_s": wall_time,
            "real_time_factor": self.control_steps*self.policy_dt/wall_time if wall_time > 0 else 0.0,
            "depth_captures": self.capture_count, "resets": self.reset_count,
            "proprio_noise": self.cfg.proprio_noise, "depth_noise": self.cfg.depth_noise,
            "tracking_xy_mse": float(self._mse_sum[:2].sum()/count),
            "tracking_yaw_mse": float(self._mse_sum[2]/count),
            "root_height_min_m": self._height_min if self.control_steps else float(self.data.qpos[self.mapping.root_qpos+2]),
            "root_height_final_m": float(self.data.qpos[self.mapping.root_qpos+2]),
            "depth_normalized_min": self._depth_min, "depth_normalized_max": self._depth_max,
            "max_abs_action": self._max_action, "max_abs_torque_nm": self._max_torque,
            "action_saturated_fraction": self._action_saturated/count,
            "torque_saturated_fraction": self._torque_saturated/(count*self.cfg.decimation),
        }

    def run(self) -> dict:
        self.init_interfaces()
        started = time.perf_counter()
        steps = math.ceil(self.cfg.duration/self.policy_dt - 1e-9)
        for _ in range(steps):
            if self._exit_event.is_set() or (self.viewer is not None and not self.viewer.is_running()):
                break
            tick = time.perf_counter()
            with self.viewer.lock() if self.viewer is not None else nullcontext():
                self.update_inputs()
                if self._exit_event.is_set():
                    break
                self.step()
            if self.viewer is not None:
                self.viewer.sync()
                remaining = self.policy_dt-(time.perf_counter()-tick)
                if remaining > 0:
                    time.sleep(remaining)
            if self.control_steps % 100 == 0:
                print(f"[INFO] steps={self.control_steps}, t={self.data.time:.2f}s, "
                      f"height={self.data.qpos[self.mapping.root_qpos+2]:.3f}m, captures={self.capture_count}")
        report = self.summary(time.perf_counter()-started)
        print("[SUMMARY] " + json.dumps(report, sort_keys=True))
        return report

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.figure is not None:
            self._plt.close(self.figure)
            self.figure = None
        if self.joystick is not None:
            import pygame

            self.joystick.joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
            self.joystick = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def get_available_scenes(directory=MJCF_DIR):
    directory = Path(directory)
    scenes = {"flat": directory / "g1_23dof_rev_1_0.xml"}
    scenes.update({p.stem.removesuffix("_scene"): p for p in sorted(directory.glob("*_scene.xml"))})
    return scenes


def find_latest_exported_policy(log_root=REPO_ROOT / "logs/g1_depth"):
    paths = list(Path(log_root).glob("*/exported/policy.pt"))
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, str(p))) if paths else None


def main():
    scenes = get_available_scenes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="Exported g1_depth policy.pt; default: newest logs/g1_depth/*/exported/policy.pt")
    parser.add_argument("--model", type=Path, default=scenes["flat"], help="Robot MJCF used when no scene is specified")
    parser.add_argument("--scene", choices=sorted(scenes), help="Built-in scene (default: flat)")
    parser.add_argument("--scene-file", type=Path, help="Custom MJCF; overrides --scene and --model")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--vx", type=float, default=.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true", help="No viewer, joystick or depth window; no OpenGL required")
    parser.add_argument("--no-joystick", action="store_true")
    parser.add_argument("--no-depth-view", action="store_true")
    parser.add_argument("--debug-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-noise", action="store_true", help="Training Gaussian depth noise and probabilistic blur")
    parser.add_argument("--proprio-noise", action="store_true", help="Training uniform proprioception noise")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--check-mapping", action="store_true", help="Print default joint mapping without loading a policy")
    parser.add_argument("--report", type=Path, help="Optional JSON run summary")
    args = parser.parse_args()
    if args.list_scenes:
        for name, path in scenes.items():
            print(f"{name}: {path}")
        return
    cfg = G1DepthSim2SimCfg(duration=args.duration, command=(args.vx, args.vy, args.wz),
        headless=args.headless, use_joystick=not args.no_joystick, show_depth=not args.no_depth_view,
        debug_depth=args.debug_depth, depth_noise=args.depth_noise, proprio_noise=args.proprio_noise, seed=args.seed)
    try:
        cfg.validate()
        model_path = args.scene_file or scenes.get(args.scene) or args.model
        if args.check_mapping:
            prepared, temporary = prepare_mjcf(model_path)
            try:
                model = mujoco.MjModel.from_xml_path(str(prepared))
                JointMapping.from_model(model).print_mapping()
            finally:
                temporary.cleanup()
            return
        policy_path = args.policy or find_latest_exported_policy()
        if policy_path is None:
            parser.error("No exported g1_depth policy found. Supply --policy /path/to/exported/policy.pt.")
        with G1DepthMujocoRunner(cfg, policy_path, model_path) as runner:
            report = runner.run()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (ValueError, RuntimeError, FileNotFoundError, ET.ParseError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
