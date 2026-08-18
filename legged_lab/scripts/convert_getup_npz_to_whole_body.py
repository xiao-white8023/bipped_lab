#!/usr/bin/env python3
"""Convert retargeted G1 MuJoCo NPZ motions to whole-body IsaacLab NPZ.

The converter follows the data-flow used by AMP_mjlab's ``csv_to_npz.py``:

1. pose is the source of truth;
2. root/joint positions are resampled before velocities are reconstructed;
3. root position and joint position use linear interpolation;
4. root orientation uses quaternion SLERP;
5. root linear, root angular, and joint velocities are reconstructed on the
   final output timeline;
6. the current repository's actual G1 23-DoF IsaacLab articulation supplies
   all rigid-body link states through forward kinematics; and
7. final joint states are stored in the IsaacLab articulation joint order; and
8. no training-time AMP features or body selection are performed here.

Source and IsaacLab public quaternions are both WXYZ.  MuJoCo free-joint
angular velocity is body-local, but it is used only for diagnostics and is
rotated to world before comparison.  Recomputed angular velocity and all
saved velocities are in the simulation world frame.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SOURCE_QPOS_WIDTH = 30
SOURCE_QVEL_WIDTH = 29
NUM_G1_JOINTS = 23
QUAT_EPS = 1.0e-8
QUAT_NORM_TOL = 1.0e-3
ROOT_POSE_TOL = 2.0e-4
ROOT_VELOCITY_TOL = 2.0e-4
JOINT_STATE_TOL = 2.0e-5


@dataclass(frozen=True)
class SourceMotion:
    """Validated source arrays in their original source ordering."""

    path: Path
    qpos: np.ndarray
    qvel: np.ndarray
    fps: float
    split_points: np.ndarray
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class ResampledMotion:
    """Pose and reconstructed velocity arrays in source joint order."""

    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    output_split_points: np.ndarray
    source_reference_root_lin_vel_w: np.ndarray
    source_reference_root_ang_vel_w: np.ndarray
    source_reference_joint_vel: np.ndarray


def _add_converter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input_npz",
        type=Path,
        default=None,
        help="Retargeted MuJoCo-style G1 NPZ input.",
    )
    parser.add_argument(
        "--output_npz",
        type=Path,
        default=None,
        help="AMP_mjlab-style whole-body NPZ output.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Output sampling frequency in Hz (default: 50).",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Inspect the source NPZ without launching IsaacLab.",
    )
    parser.add_argument(
        "--inspect_output",
        type=Path,
        default=None,
        help="Inspect an already converted output NPZ without launching IsaacLab.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Debug-only cap on total output frames; derivatives remain segment-local.",
    )


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert G1 23-DoF retargeted MuJoCo NPZ pose data into an "
            "AMP_mjlab-style whole-body IsaacLab NPZ."
        ),
        add_help=add_help,
    )
    _add_converter_arguments(parser)
    return parser


def _string_list(value: np.ndarray, name: str) -> list[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={array.shape}.")

    result: list[str] = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    return result


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape={array.shape}.")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}.")
    return scalar


def _validate_split_points(value: np.ndarray | None, num_frames: int) -> np.ndarray:
    if value is None or np.asarray(value).size == 0:
        return np.asarray([0, num_frames], dtype=np.int64)

    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"split_points must be one-dimensional, got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("split_points must contain integer frame indices.")
    splits = raw.astype(np.int64, copy=False)
    if len(splits) < 2:
        raise ValueError("split_points must contain at least [0, num_frames].")
    if int(splits[0]) != 0 or int(splits[-1]) != num_frames:
        raise ValueError(
            "split_points must cover the complete source array: "
            f"expected first=0 and last={num_frames}, got {splits.tolist()}."
        )
    if np.any(np.diff(splits) <= 0):
        raise ValueError(f"split_points must be strictly increasing: {splits.tolist()}.")
    return splits.copy()


def _load_source(path: Path) -> SourceMotion:
    input_path = path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")

    # Object-valued metadata is intentionally not unpickled.  The required
    # arrays in the retargeted source format are plain numeric/string arrays.
    with np.load(input_path, allow_pickle=False) as data:
        required = ("qpos", "qvel", "frequency", "joint_names")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing required source keys {missing}; keys={data.files}.")

        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        fps = _scalar_float(data["frequency"], "frequency")
        names = _string_list(data["joint_names"], "joint_names")
        raw_splits = data["split_points"] if "split_points" in data.files else None

    if qpos.ndim != 2 or qpos.shape[1] != SOURCE_QPOS_WIDTH:
        raise ValueError(
            f"qpos must have shape [T,{SOURCE_QPOS_WIDTH}], got {qpos.shape}."
        )
    if qvel.ndim != 2 or qvel.shape[1] != SOURCE_QVEL_WIDTH:
        raise ValueError(
            f"qvel must have shape [T,{SOURCE_QVEL_WIDTH}], got {qvel.shape}."
        )
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError(
            f"qpos/qvel frame mismatch: {qpos.shape[0]} vs {qvel.shape[0]}."
        )
    if qpos.shape[0] == 0:
        raise ValueError("Source motion contains no frames.")
    if fps <= 0.0:
        raise ValueError(f"frequency must be positive, got {fps}.")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or Inf.")
    if not np.isfinite(qvel).all():
        raise ValueError("qvel contains NaN or Inf.")

    if len(names) != NUM_G1_JOINTS + 1:
        raise ValueError(
            f"joint_names must contain root + {NUM_G1_JOINTS} joints, got {len(names)}."
        )
    if names[0] != "root":
        raise ValueError(f"joint_names[0] must be 'root', got {names[0]!r}.")
    joint_names = names[1:]
    duplicates = sorted({name for name in joint_names if joint_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source joint names: {duplicates}.")

    splits = _validate_split_points(raw_splits, qpos.shape[0])
    root_quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(root_quat_norm < QUAT_EPS):
        bad = np.flatnonzero(root_quat_norm < QUAT_EPS)[:10].tolist()
        raise ValueError(f"Near-zero source root quaternion at global frames {bad}.")

    return SourceMotion(
        path=input_path,
        qpos=qpos,
        qvel=qvel,
        fps=fps,
        split_points=splits,
        joint_names=tuple(joint_names),
    )


def _normalize_and_make_continuous(quat_wxyz: np.ndarray) -> tuple[np.ndarray, int]:
    quat = np.asarray(quat_wxyz, dtype=np.float64).copy()
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"Quaternion array must have shape [T,4], got {quat.shape}.")
    norms = np.linalg.norm(quat, axis=1)
    if not np.isfinite(norms).all() or np.any(norms < QUAT_EPS):
        bad = np.flatnonzero(~np.isfinite(norms) | (norms < QUAT_EPS))[:10].tolist()
        raise ValueError(f"Invalid quaternion norm at segment-local frames {bad}.")
    quat /= norms[:, None]

    sign_flips = 0
    for frame_id in range(1, len(quat)):
        if float(np.dot(quat[frame_id - 1], quat[frame_id])) < 0.0:
            quat[frame_id] *= -1.0
            sign_flips += 1
    return quat, sign_flips


def _frame_blend(
    source_frames: int, source_fps: float, output_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_frames == 1:
        zeros = np.zeros(len(output_times), dtype=np.int64)
        return zeros, zeros, np.zeros(len(output_times), dtype=np.float64)

    source_coordinate = output_times * source_fps
    index_0 = np.floor(source_coordinate + 1.0e-12).astype(np.int64)
    index_0 = np.clip(index_0, 0, source_frames - 1)
    index_1 = np.minimum(index_0 + 1, source_frames - 1)
    blend = source_coordinate - index_0
    blend[index_0 == source_frames - 1] = 0.0
    return index_0, index_1, np.clip(blend, 0.0, 1.0)


def _lerp_samples(
    values: np.ndarray,
    index_0: np.ndarray,
    index_1: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    alpha = blend[:, None]
    return values[index_0] * (1.0 - alpha) + values[index_1] * alpha


def _quat_slerp_wxyz(
    quat: np.ndarray,
    index_0: np.ndarray,
    index_1: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    """Proper shortest-path quaternion SLERP, retaining WXYZ layout."""

    q0 = quat[index_0]
    q1 = quat[index_1].copy()
    dot = np.sum(q0 * q1, axis=1)

    # Temporal continuity normally makes dot positive.  Keep shortest-path
    # handling here as a local safety property of SLERP as well.
    negative = dot < 0.0
    q1[negative] *= -1.0
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    result = np.empty_like(q0)
    near = dot > 0.9995
    if np.any(near):
        alpha = blend[near, None]
        result[near] = q0[near] * (1.0 - alpha) + q1[near] * alpha

    far = ~near
    if np.any(far):
        theta = np.arccos(dot[far])
        sin_theta = np.sin(theta)
        alpha = blend[far]
        weight_0 = np.sin((1.0 - alpha) * theta) / sin_theta
        weight_1 = np.sin(alpha * theta) / sin_theta
        result[far] = q0[far] * weight_0[:, None] + q1[far] * weight_1[:, None]

    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = np.asarray(quat, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _axis_angle_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quat, dtype=np.float64).copy()
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized[normalized[:, 0] < 0.0] *= -1.0
    vector = normalized[:, 1:]
    vector_norm = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(normalized[:, 0], -1.0, 1.0))
    result = np.empty_like(vector)
    regular = vector_norm > 1.0e-10
    result[regular] = vector[regular] * (angle[regular] / vector_norm[regular])[:, None]
    result[~regular] = 2.0 * vector[~regular]
    return result


def _quat_apply_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate body-local vectors into world without changing quaternion layout."""

    quat_vector = quat[:, 1:]
    twice_cross = 2.0 * np.cross(quat_vector, vector)
    return vector + quat[:, :1] * twice_cross + np.cross(quat_vector, twice_cross)


def _numeric_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) == 1:
        return np.zeros_like(values)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, dt, axis=0, edge_order=edge_order)


def _so3_derivative_wxyz(rotations: np.ndarray, dt: float) -> np.ndarray:
    """AMP_mjlab-equivalent SO(3) derivative in the world frame.

    Interior samples use ``q[t+1] * conjugate(q[t-1]) / (2*dt)``.  The
    relative rotation is a left/world-frame delta.  As in AMP_mjlab, the first
    and last interior velocities are copied to the endpoints.
    """

    num_frames = len(rotations)
    if num_frames == 1:
        return np.zeros((1, 3), dtype=np.float64)
    if num_frames == 2:
        relative = _quat_multiply_wxyz(
            rotations[1:2], _quat_conjugate_wxyz(rotations[0:1])
        )
        omega = _axis_angle_from_quat_wxyz(relative) / dt
        return np.repeat(omega, 2, axis=0)

    previous = rotations[:-2]
    following = rotations[2:]
    relative = _quat_multiply_wxyz(following, _quat_conjugate_wxyz(previous))
    interior = _axis_angle_from_quat_wxyz(relative) / (2.0 * dt)
    return np.concatenate((interior[:1], interior, interior[-1:]), axis=0)


def _output_times(num_source_frames: int, source_fps: float, output_fps: float) -> np.ndarray:
    """Build the AMP_mjlab-style output timeline ``[0, duration)``.

    AMP_mjlab's converter samples with ``arange(0, duration, output_dt)``;
    therefore a sample exactly at ``duration`` is intentionally excluded.
    A one-frame source has zero duration, so retain one sample at t=0.
    """

    duration = (num_source_frames - 1) / source_fps
    if duration <= 0.0:
        return np.zeros(1, dtype=np.float64)

    times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
    if len(times) == 0:
        return np.zeros(1, dtype=np.float64)
    return times


def _resample_source(
    source: SourceMotion, output_fps: float, max_frames: int | None
) -> ResampledMotion:
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError(f"--output_fps must be positive and finite, got {output_fps}.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"--max_frames must be positive, got {max_frames}.")

    result: dict[str, list[np.ndarray]] = {
        "root_pos_w": [],
        "root_quat_w": [],
        "root_lin_vel_w": [],
        "root_ang_vel_w": [],
        "joint_pos": [],
        "joint_vel": [],
        "source_reference_root_lin_vel_w": [],
        "source_reference_root_ang_vel_w": [],
        "source_reference_joint_vel": [],
    }
    output_splits = [0]
    remaining = max_frames

    for trajectory_index, (start, end) in enumerate(
        zip(source.split_points[:-1], source.split_points[1:])
    ):
        start_i, end_i = int(start), int(end)
        source_frames = end_i - start_i
        duration = (source_frames - 1) / source.fps
        times = _output_times(source_frames, source.fps, output_fps)

        pose = source.qpos[start_i:end_i]
        source_velocity = source.qvel[start_i:end_i]
        root_quat, sign_flips = _normalize_and_make_continuous(pose[:, 3:7])
        index_0, index_1, blend = _frame_blend(source_frames, source.fps, times)

        root_pos = _lerp_samples(pose[:, 0:3], index_0, index_1, blend)
        joint_pos = _lerp_samples(pose[:, 7:30], index_0, index_1, blend)
        root_quat_resampled = _quat_slerp_wxyz(root_quat, index_0, index_1, blend)

        dt = 1.0 / output_fps
        root_lin_vel = _numeric_gradient(root_pos, dt)
        joint_vel = _numeric_gradient(joint_pos, dt)
        root_ang_vel = _so3_derivative_wxyz(root_quat_resampled, dt)

        # Source qvel is reference-only.  MuJoCo free-joint angular velocity is
        # body-local, so rotate it to world before comparing to the recomputed
        # world-frame angular velocity.
        source_root_ang_vel_w = _quat_apply_wxyz(root_quat, source_velocity[:, 3:6])
        reference_root_lin = _lerp_samples(source_velocity[:, 0:3], index_0, index_1, blend)
        reference_root_ang = _lerp_samples(source_root_ang_vel_w, index_0, index_1, blend)
        reference_joint = _lerp_samples(source_velocity[:, 6:29], index_0, index_1, blend)

        take = len(times) if remaining is None else min(len(times), remaining)
        if take == 0:
            break
        for key, value in (
            ("root_pos_w", root_pos),
            ("root_quat_w", root_quat_resampled),
            ("root_lin_vel_w", root_lin_vel),
            ("root_ang_vel_w", root_ang_vel),
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("source_reference_root_lin_vel_w", reference_root_lin),
            ("source_reference_root_ang_vel_w", reference_root_ang),
            ("source_reference_joint_vel", reference_joint),
        ):
            result[key].append(value[:take])

        output_splits.append(output_splits[-1] + take)
        print(
            f"Trajectory {trajectory_index}: source_frames={source_frames}, "
            f"source_fps={source.fps:g}, duration_s={duration:.6f}, "
            f"output_frames={take}{' (truncated)' if take < len(times) else ''}, "
            f"output_fps={output_fps:g}, source_quaternion_sign_flips={sign_flips}"
        )

        if remaining is not None:
            remaining -= take
            if remaining == 0:
                break

    if not result["root_pos_w"]:
        raise RuntimeError("Resampling produced no output frames.")

    def concatenate(key: str) -> np.ndarray:
        return np.concatenate(result[key], axis=0)

    return ResampledMotion(
        root_pos_w=concatenate("root_pos_w"),
        root_quat_w=concatenate("root_quat_w"),
        root_lin_vel_w=concatenate("root_lin_vel_w"),
        root_ang_vel_w=concatenate("root_ang_vel_w"),
        joint_pos=concatenate("joint_pos"),
        joint_vel=concatenate("joint_vel"),
        output_split_points=np.asarray(output_splits, dtype=np.int64),
        source_reference_root_lin_vel_w=concatenate("source_reference_root_lin_vel_w"),
        source_reference_root_ang_vel_w=concatenate("source_reference_root_ang_vel_w"),
        source_reference_joint_vel=concatenate("source_reference_joint_vel"),
    )


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def _quat_angle_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Promote simulator float32 quaternions before normalization.  Performing
    # the norm in float32 can turn an identical rotation into a spurious
    # ~1e-3 rad angle through acos' sensitivity near dot=1.
    left_f64 = np.asarray(left, dtype=np.float64)
    right_f64 = np.asarray(right, dtype=np.float64)
    left_n = left_f64 / np.linalg.norm(left_f64, axis=-1, keepdims=True)
    right_n = right_f64 / np.linalg.norm(right_f64, axis=-1, keepdims=True)
    dot = np.abs(np.sum(left_n * right_n, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _trajectory_and_local_frame(global_frame: int, split_points: np.ndarray) -> tuple[int, int]:
    trajectory = int(np.searchsorted(split_points, global_frame, side="right") - 1)
    trajectory = min(trajectory, len(split_points) - 2)
    return trajectory, global_frame - int(split_points[trajectory])


def _assert_finite(
    name: str,
    array: np.ndarray,
    split_points: np.ndarray,
    element_names: Sequence[str] | None = None,
) -> None:
    invalid = np.argwhere(~np.isfinite(array))
    if len(invalid) == 0:
        return
    first = invalid[0]
    global_frame = int(first[0])
    trajectory, local_frame = _trajectory_and_local_frame(global_frame, split_points)
    detail = ""
    if element_names is not None and array.ndim >= 3:
        element_id = int(first[1])
        detail = f", element_index={element_id}, element_name={element_names[element_id]!r}"
    elif element_names is not None and array.ndim == 2 and len(element_names) == array.shape[1]:
        element_id = int(first[1])
        detail = f", element_index={element_id}, element_name={element_names[element_id]!r}"
    raise ValueError(
        f"{name} contains NaN/Inf at trajectory={trajectory}, local_frame={local_frame}, "
        f"global_frame={global_frame}{detail}, full_index={first.tolist()}."
    )


def _build_joint_mapping(
    source_names: Sequence[str], target_names: Sequence[str]
) -> np.ndarray:
    if len(source_names) != NUM_G1_JOINTS:
        raise RuntimeError(f"Source has {len(source_names)} joints, expected {NUM_G1_JOINTS}.")
    if len(target_names) != NUM_G1_JOINTS:
        raise RuntimeError(
            f"IsaacLab G1 articulation has {len(target_names)} joints, expected {NUM_G1_JOINTS}."
        )

    source_duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
    target_duplicates = sorted({name for name in target_names if target_names.count(name) > 1})
    if source_duplicates or target_duplicates:
        raise RuntimeError(
            f"Duplicate joint names: source={source_duplicates}, IsaacLab={target_duplicates}."
        )

    source_index = {name: index for index, name in enumerate(source_names)}
    target_set = set(target_names)
    missing = [name for name in target_names if name not in source_index]
    unexpected = [name for name in source_names if name not in target_set]
    if missing or unexpected:
        raise RuntimeError(
            "Source/IsaacLab joint sets do not match. "
            f"Missing source joints required by IsaacLab: {missing}; "
            f"unexpected source joints: {unexpected}."
        )
    return np.asarray([source_index[name] for name in target_names], dtype=np.int64)


def _run_isaaclab_fk(
    args: argparse.Namespace,
    motion: ResampledMotion,
    source_joint_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str], list[str], int]:
    """Write every deterministic frame and read all articulation link states."""

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from legged_lab.assets.g1.g1_29 import G1_23CFG

    @configclass
    class ConverterSceneCfg(InteractiveSceneCfg):
        robot = G1_23CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / float(args.output_fps),
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ConverterSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    isaac_joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    target_from_source = _build_joint_mapping(list(source_joint_names), isaac_joint_names)
    print(f"IsaacLab joint order ({len(isaac_joint_names)}): {isaac_joint_names}")
    print(f"IsaacLab body order ({len(body_names)}): {body_names}")
    print(f"source->IsaacLab joint source indices: {target_from_source.tolist()}")

    device = robot.device
    root_pose = torch.as_tensor(
        np.concatenate((motion.root_pos_w, motion.root_quat_w), axis=1),
        dtype=torch.float32,
        device=device,
    )
    root_velocity = torch.as_tensor(
        np.concatenate((motion.root_lin_vel_w, motion.root_ang_vel_w), axis=1),
        dtype=torch.float32,
        device=device,
    )
    joint_pos_source = torch.as_tensor(motion.joint_pos, dtype=torch.float32, device=device)
    joint_vel_source = torch.as_tensor(motion.joint_vel, dtype=torch.float32, device=device)
    mapping = torch.as_tensor(target_from_source, dtype=torch.long, device=device)
    joint_pos_target = joint_pos_source.index_select(1, mapping)
    joint_vel_target = joint_vel_source.index_select(1, mapping)

    num_frames = root_pose.shape[0]
    num_bodies = len(body_names)
    body_pos = torch.empty((num_frames, num_bodies, 3), dtype=torch.float32, device=device)
    body_quat = torch.empty((num_frames, num_bodies, 4), dtype=torch.float32, device=device)
    body_lin_vel = torch.empty((num_frames, num_bodies, 3), dtype=torch.float32, device=device)
    body_ang_vel = torch.empty((num_frames, num_bodies, 3), dtype=torch.float32, device=device)
    sim_joint_pos = torch.empty_like(joint_pos_target)
    sim_joint_vel = torch.empty_like(joint_vel_target)

    root_body_index: int | None = None
    for frame_id in range(num_frames):
        # The link-specific APIs preserve the semantics of MuJoCo qpos root
        # position (root actor/link frame), and all passed velocities are world.
        robot.write_root_link_pose_to_sim(root_pose[frame_id : frame_id + 1])
        robot.write_root_link_velocity_to_sim(root_velocity[frame_id : frame_id + 1])
        robot.write_joint_state_to_sim(
            joint_pos_target[frame_id : frame_id + 1],
            joint_vel_target[frame_id : frame_id + 1],
        )

        # No physics integration step is taken.  forward() updates simulator
        # kinematics, then scene.update() refreshes the asset data buffers before
        # whole-body link states are read.
        sim.forward()
        scene.update(1.0 / float(args.output_fps))
        frame_body_pos = robot.data.body_link_pos_w[0]
        frame_body_quat = robot.data.body_link_quat_w[0]
        frame_body_lin_vel = robot.data.body_link_lin_vel_w[0]
        frame_body_ang_vel = robot.data.body_link_ang_vel_w[0]

        if root_body_index is None:
            position_error = torch.linalg.vector_norm(
                frame_body_pos - root_pose[frame_id, :3], dim=-1
            )
            normalized_body_quat = torch.nn.functional.normalize(frame_body_quat, dim=-1)
            normalized_root_quat = torch.nn.functional.normalize(
                root_pose[frame_id, 3:7], dim=-1
            )
            quat_dot = torch.abs(
                torch.sum(normalized_body_quat * normalized_root_quat[None, :], dim=-1)
            )
            angle_error = 2.0 * torch.acos(torch.clamp(quat_dot, -1.0, 1.0))
            score = position_error + angle_error
            root_body_index = int(torch.argmin(score).item())
            root_position_error = float(position_error[root_body_index].item())
            root_angle_error = float(angle_error[root_body_index].item())
            if root_position_error > ROOT_POSE_TOL or root_angle_error > ROOT_POSE_TOL:
                raise RuntimeError(
                    "Could not identify an articulation body matching the written root link pose: "
                    f"best_body={body_names[root_body_index]!r}, "
                    f"position_error={root_position_error:.3e}, "
                    f"angle_error={root_angle_error:.3e}."
                )
            print(
                f"Auto-detected root rigid body: index={root_body_index}, "
                f"name={body_names[root_body_index]!r}"
            )

        body_pos[frame_id].copy_(frame_body_pos)
        body_quat[frame_id].copy_(frame_body_quat)
        body_lin_vel[frame_id].copy_(frame_body_lin_vel)
        body_ang_vel[frame_id].copy_(frame_body_ang_vel)
        sim_joint_pos[frame_id].copy_(robot.data.joint_pos[0])
        sim_joint_vel[frame_id].copy_(robot.data.joint_vel[0])

        completed = frame_id + 1
        if completed == num_frames or completed % 500 == 0:
            print(f"FK frames: {completed}/{num_frames}")

    assert root_body_index is not None

    arrays = {
        "body_pos_w": body_pos.cpu().numpy(),
        "body_quat_w": body_quat.cpu().numpy(),
        "body_lin_vel_w": body_lin_vel.cpu().numpy(),
        "body_ang_vel_w": body_ang_vel.cpu().numpy(),
        "sim_joint_pos": sim_joint_pos.cpu().numpy(),
        "sim_joint_vel": sim_joint_vel.cpu().numpy(),
        "joint_pos_target": joint_pos_target.cpu().numpy(),
        "joint_vel_target": joint_vel_target.cpu().numpy(),
    }
    return arrays, isaac_joint_names, body_names, root_body_index


def _validate_and_report(
    source: SourceMotion,
    motion: ResampledMotion,
    simulator_arrays: dict[str, np.ndarray],
    body_names: Sequence[str],
    root_body_index: int,
) -> dict[str, float]:
    body_pos = simulator_arrays["body_pos_w"]
    body_quat = simulator_arrays["body_quat_w"]
    body_lin_vel = simulator_arrays["body_lin_vel_w"]
    body_ang_vel = simulator_arrays["body_ang_vel_w"]

    for name, array, names in (
        ("joint_pos", motion.joint_pos, source.joint_names),
        ("joint_vel", motion.joint_vel, source.joint_names),
        ("body_pos_w", body_pos, body_names),
        ("body_quat_w", body_quat, body_names),
        ("body_lin_vel_w", body_lin_vel, body_names),
        ("body_ang_vel_w", body_ang_vel, body_names),
    ):
        _assert_finite(name, array, motion.output_split_points, names)

    root_quat_norm = np.linalg.norm(motion.root_quat_w, axis=1)
    body_quat_norm = np.linalg.norm(body_quat, axis=2)
    root_pos_error = float(
        np.max(np.abs(body_pos[:, root_body_index] - motion.root_pos_w))
    )
    root_angle_error = float(
        np.max(_quat_angle_error(body_quat[:, root_body_index], motion.root_quat_w))
    )
    root_lin_vel_rmse = _rmse(body_lin_vel[:, root_body_index], motion.root_lin_vel_w)
    root_ang_vel_rmse = _rmse(body_ang_vel[:, root_body_index], motion.root_ang_vel_w)
    joint_pos_error = float(
        np.max(np.abs(simulator_arrays["sim_joint_pos"] - simulator_arrays["joint_pos_target"]))
    )
    joint_vel_error = float(
        np.max(np.abs(simulator_arrays["sim_joint_vel"] - simulator_arrays["joint_vel_target"]))
    )

    source_root_lin_rmse = _rmse(
        motion.source_reference_root_lin_vel_w, motion.root_lin_vel_w
    )
    source_root_ang_rmse = _rmse(
        motion.source_reference_root_ang_vel_w, motion.root_ang_vel_w
    )
    source_joint_rmse = _rmse(motion.source_reference_joint_vel, motion.joint_vel)
    source_root_lin_max = float(
        np.max(
            np.abs(motion.source_reference_root_lin_vel_w - motion.root_lin_vel_w)
        )
    )
    source_root_ang_max = float(
        np.max(
            np.abs(motion.source_reference_root_ang_vel_w - motion.root_ang_vel_w)
        )
    )
    source_joint_max = float(
        np.max(np.abs(motion.source_reference_joint_vel - motion.joint_vel))
    )

    metrics = {
        "root_pos_max_abs_error": root_pos_error,
        "root_quat_angle_error_max": root_angle_error,
        "root_lin_vel_rmse": root_lin_vel_rmse,
        "root_ang_vel_rmse": root_ang_vel_rmse,
        "joint_pos_max_abs_error": joint_pos_error,
        "joint_vel_max_abs_error": joint_vel_error,
        "source_vs_recomputed_root_lin_vel_rmse": source_root_lin_rmse,
        "source_vs_recomputed_root_ang_vel_rmse": source_root_ang_rmse,
        "source_vs_recomputed_joint_vel_rmse": source_joint_rmse,
        "source_vs_recomputed_root_lin_vel_max_abs": source_root_lin_max,
        "source_vs_recomputed_root_ang_vel_max_abs": source_root_ang_max,
        "source_vs_recomputed_joint_vel_max_abs": source_joint_max,
        "root_quat_norm_min": float(np.min(root_quat_norm)),
        "root_quat_norm_max": float(np.max(root_quat_norm)),
        "body_quat_norm_min": float(np.min(body_quat_norm)),
        "body_quat_norm_max": float(np.max(body_quat_norm)),
    }
    for name, value in metrics.items():
        print(f"Validation/{name}={value:.9g}")
    print("Validation/all_core_arrays_finite=True")

    if root_pos_error > ROOT_POSE_TOL or root_angle_error > ROOT_POSE_TOL:
        raise RuntimeError(
            "Root pose consistency validation failed: "
            f"position={root_pos_error:.3e}, quaternion_angle={root_angle_error:.3e}."
        )
    if root_lin_vel_rmse > ROOT_VELOCITY_TOL or root_ang_vel_rmse > ROOT_VELOCITY_TOL:
        raise RuntimeError(
            "Root velocity consistency validation failed: "
            f"linear_rmse={root_lin_vel_rmse:.3e}, angular_rmse={root_ang_vel_rmse:.3e}."
        )
    if joint_pos_error > JOINT_STATE_TOL or joint_vel_error > JOINT_STATE_TOL:
        raise RuntimeError(
            "Joint consistency validation failed: "
            f"position={joint_pos_error:.3e}, velocity={joint_vel_error:.3e}."
        )
    if (
        np.max(np.abs(root_quat_norm - 1.0)) > QUAT_NORM_TOL
        or np.max(np.abs(body_quat_norm - 1.0)) > QUAT_NORM_TOL
    ):
        raise RuntimeError("Quaternion norm validation failed.")
    return metrics


def _save_output(
    output_path: Path,
    source: SourceMotion,
    motion: ResampledMotion,
    simulator_arrays: dict[str, np.ndarray],
    isaac_joint_names: Sequence[str],
    body_names: Sequence[str],
    root_body_index: int,
    output_fps: float,
) -> Path:
    resolved = output_path.expanduser().resolve()
    if resolved == source.path:
        raise ValueError("--output_npz must not overwrite the source NPZ.")
    resolved.parent.mkdir(parents=True, exist_ok=True)

    output_splits = motion.output_split_points.astype(np.int64, copy=False)
    payload: dict[str, np.ndarray] = {
        # Match AMP_mjlab's core output convention: fps is a length-1 array,
        # and joint states are saved in the articulation's actual IsaacLab order.
        "fps": np.asarray([output_fps], dtype=np.float32),
        "joint_pos": np.asarray(simulator_arrays["sim_joint_pos"], dtype=np.float32),
        "joint_vel": np.asarray(simulator_arrays["sim_joint_vel"], dtype=np.float32),
        "body_pos_w": np.asarray(simulator_arrays["body_pos_w"], dtype=np.float32),
        "body_quat_w": np.asarray(simulator_arrays["body_quat_w"], dtype=np.float32),
        "body_lin_vel_w": np.asarray(simulator_arrays["body_lin_vel_w"], dtype=np.float32),
        "body_ang_vel_w": np.asarray(simulator_arrays["body_ang_vel_w"], dtype=np.float32),
        "joint_names": np.asarray(isaac_joint_names, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
        "source_fps": np.asarray(source.fps, dtype=np.float32),
        "source_frame_count": np.asarray(source.qpos.shape[0], dtype=np.int64),
        "source_file": np.asarray(str(source.path), dtype=np.str_),
        "source_split_points": source.split_points.astype(np.int64, copy=False),
        "output_split_points": output_splits,
        "split_points": output_splits,
        "root_pos_w": np.asarray(motion.root_pos_w, dtype=np.float32),
        "root_quat_w": np.asarray(motion.root_quat_w, dtype=np.float32),
        "root_lin_vel_w": np.asarray(motion.root_lin_vel_w, dtype=np.float32),
        "root_ang_vel_w": np.asarray(motion.root_ang_vel_w, dtype=np.float32),
        "root_body_index": np.asarray(root_body_index, dtype=np.int64),
        "root_body_name": np.asarray(body_names[root_body_index], dtype=np.str_),
        "quaternion_convention": np.asarray("wxyz", dtype=np.str_),
        "source_quaternion_convention": np.asarray("wxyz", dtype=np.str_),
        "velocity_frame": np.asarray("world", dtype=np.str_),
        "source_root_angular_velocity_frame": np.asarray("body_local", dtype=np.str_),
        "body_state_frame": np.asarray("link_world", dtype=np.str_),
    }
    np.savez(resolved, **payload)
    print(f"Saved whole-body motion: {resolved}")
    return resolved


def _safe_array_description(data: Any, key: str) -> str:
    try:
        array = np.asarray(data[key])
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            return "object array (not loaded; pickle is disabled)"
        return f"unreadable: {error}"
    suffix = f", empty={array.size == 0}"
    if array.size <= 30:
        suffix += f", value={array.tolist()!r}"
    return f"shape={array.shape}, dtype={array.dtype}{suffix}"


def inspect_source(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {resolved}")
    print(f"Source file: {resolved}")
    with np.load(resolved, allow_pickle=False) as data:
        print(f"Keys ({len(data.files)}): {data.files}")
        for key in data.files:
            print(f"  {key}: {_safe_array_description(data, key)}")

    source = _load_source(resolved)
    total_duration = sum(
        (int(end) - int(start) - 1) / source.fps
        for start, end in zip(source.split_points[:-1], source.split_points[1:])
    )
    root_norm = np.linalg.norm(source.qpos[:, 3:7], axis=1)
    sign_flips = 0
    for start, end in zip(source.split_points[:-1], source.split_points[1:]):
        quat = source.qpos[int(start) : int(end), 3:7]
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        sign_flips += int(np.count_nonzero(np.sum(quat[1:] * quat[:-1], axis=1) < 0.0))

    print(f"qpos shape: {source.qpos.shape}")
    print(f"qvel shape: {source.qvel.shape}")
    print(f"frequency: {source.fps:g} Hz")
    print(f"frame count: {source.qpos.shape[0]}")
    print(f"trajectory count: {len(source.split_points) - 1}")
    print(f"duration_s (sum of segment durations): {total_duration:.6f}")
    print(f"split_points: {source.split_points.tolist()}")
    print(f"joint_names: {['root', *source.joint_names]}")
    print(f"qpos min/max: {np.min(source.qpos):.9g} / {np.max(source.qpos):.9g}")
    print(f"qvel min/max: {np.min(source.qvel):.9g} / {np.max(source.qvel):.9g}")
    print(
        f"root height min/max: {np.min(source.qpos[:, 2]):.9g} / "
        f"{np.max(source.qpos[:, 2]):.9g}"
    )
    print(f"root quaternion norm min/max: {np.min(root_norm):.9g} / {np.max(root_norm):.9g}")
    print(f"quaternion temporal sign flips (segment-local): {sign_flips}")
    print(f"qpos finite: {bool(np.isfinite(source.qpos).all())}")
    print(f"qvel finite: {bool(np.isfinite(source.qvel).all())}")

    body_keys = ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")
    with np.load(resolved, allow_pickle=False) as data:
        body_empty: list[bool] = []
        for key in body_keys:
            if key not in data.files:
                print(f"{key}: missing")
                body_empty.append(True)
                continue
            array = np.asarray(data[key])
            print(f"{key}: shape={array.shape}, empty={array.size == 0}")
            body_empty.append(array.size == 0)
    print(f"body-related source arrays all empty/missing: {all(body_empty)}")
    print("Source quaternion convention: wxyz")
    print("Source qvel root angular velocity frame: body_local (reference only)")


def inspect_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Output NPZ not found: {resolved}")
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )
    with np.load(resolved, allow_pickle=False) as data:
        print(f"Output file: {resolved}")
        print(f"Keys ({len(data.files)}): {data.files}")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing output keys: {missing}.")
        for key in data.files:
            print(f"  {key}: {_safe_array_description(data, key)}")

        fps = _scalar_float(data["fps"], "fps")
        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        body_pos = np.asarray(data["body_pos_w"])
        body_quat = np.asarray(data["body_quat_w"])
        body_lin_vel = np.asarray(data["body_lin_vel_w"])
        body_ang_vel = np.asarray(data["body_ang_vel_w"])
        joint_names = _string_list(data["joint_names"], "joint_names")
        body_names = _string_list(data["body_names"], "body_names")
        splits = np.asarray(
            data["output_split_points"] if "output_split_points" in data.files else data["split_points"],
            dtype=np.int64,
        )

    arrays = (joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel)
    if any(array.shape[0] != joint_pos.shape[0] for array in arrays):
        raise ValueError("Output core arrays do not have the same frame count.")
    duration = sum(
        max(int(end) - int(start) - 1, 0) / fps
        for start, end in zip(splits[:-1], splits[1:])
    )
    quat_norm = np.linalg.norm(body_quat, axis=2)
    print(f"fps: {fps:g}")
    print(f"frames: {joint_pos.shape[0]}")
    print(f"duration_s (sum of segment durations): {duration:.6f}")
    print(f"joint count: {len(joint_names)}")
    print(f"body count: {len(body_names)}")
    print(f"joint names: {joint_names}")
    print(f"body names: {body_names}")
    print(f"joint_pos shape: {joint_pos.shape}")
    print(f"joint_vel shape: {joint_vel.shape}")
    print(f"body_pos_w shape: {body_pos.shape}")
    print(f"body_quat_w shape: {body_quat.shape}")
    print(f"body_lin_vel_w shape: {body_lin_vel.shape}")
    print(f"body_ang_vel_w shape: {body_ang_vel.shape}")
    print(f"body quaternion norm min/max: {np.min(quat_norm):.9g} / {np.max(quat_norm):.9g}")
    print(f"all core arrays finite: {all(bool(np.isfinite(array).all()) for array in arrays)}")
    print(f"output split_points: {splits.tolist()}")


def _validate_mode_arguments(args: argparse.Namespace) -> None:
    if args.inspect_only and args.inspect_output is not None:
        raise ValueError("Use only one of --inspect_only and --inspect_output.")
    if args.inspect_output is not None:
        if args.output_npz is not None:
            raise ValueError("--output_npz is not used with --inspect_output.")
        return
    if args.input_npz is None:
        raise ValueError("--input_npz is required for source inspection or conversion.")
    if args.inspect_only:
        if args.output_npz is not None:
            raise ValueError("--output_npz is not used with --inspect_only.")
        return
    if args.output_npz is None:
        raise ValueError("--output_npz is required for conversion.")


def main() -> None:
    # Inspect modes deliberately stop before AppLauncher is imported/started.
    preliminary_parser = _build_parser(add_help=False)
    preliminary_args, _ = preliminary_parser.parse_known_args()
    if preliminary_args.inspect_only or preliminary_args.inspect_output is not None:
        parser = _build_parser(add_help=True)
        args = parser.parse_args()
        _validate_mode_arguments(args)
        if args.inspect_output is not None:
            inspect_output(args.inspect_output)
        else:
            assert args.input_npz is not None
            inspect_source(args.input_npz)
        return

    # AppLauncher must be imported before any other Isaac/Omniverse module.
    from isaaclab.app import AppLauncher

    parser = _build_parser(add_help=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _validate_mode_arguments(args)
    assert args.input_npz is not None and args.output_npz is not None

    source = _load_source(args.input_npz)
    motion = _resample_source(source, args.output_fps, args.max_frames)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        simulator_arrays, isaac_joint_names, body_names, root_body_index = _run_isaaclab_fk(
            args, motion, source.joint_names
        )
        _validate_and_report(
            source, motion, simulator_arrays, body_names, root_body_index
        )
        saved_path = _save_output(
            args.output_npz,
            source,
            motion,
            simulator_arrays,
            isaac_joint_names,
            body_names,
            root_body_index,
            args.output_fps,
        )
        print(f"Inspect with: {Path(sys.argv[0])} --inspect_output {saved_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()