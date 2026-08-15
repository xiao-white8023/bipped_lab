#!/usr/bin/env python3
"""
Crop a LocoMuJoCo Unitree G1 trajectory (.npz) into a reset-ready Recovery state file.

This script is deliberately separate from AMP expert conversion.

Output frame layout (59-D), designed to map directly to IsaacLab reset APIs:

    [0:3]    root_pos_local_w      (3)
    [3:7]    root_quat_wxyz_w      (4)
    [7:10]   root_lin_vel_w        (3)
    [10:13]  root_ang_vel_w        (3)
    [13:36]  joint_pos_23          (23)
    [36:59]  joint_vel_23          (23)

Important conventions:
- MuJoCo free-joint qpos quaternion is WXYZ and represents world orientation.
- MuJoCo free-joint linear velocity is in the world frame.
- MuJoCo free-joint angular velocity is in the LOCAL body frame.
  This script rotates it into the WORLD frame before saving, because IsaacLab
  write_root_link_velocity_to_sim() expects world-frame linear/angular velocity.
- By default source root X/Y are removed (set to zero) because environment reset
  placement should be supplied by the target IsaacLab env origin. Root Z is kept.
- Joint order is explicitly stored in metadata and is the current bipped_lab
  G1 23-DoF order.

The resulting file is intended to be consumed as:

    root_pose = frame[..., 0:7]
    root_vel  = frame[..., 7:13]
    joint_pos = frame[..., 13:36]
    joint_vel = frame[..., 36:59]

For a target IsaacLab environment:
    root_pose[..., 0:3] += env_origin
    robot.write_root_link_pose_to_sim(root_pose, env_ids)
    robot.write_root_link_velocity_to_sim(root_vel, env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

Do NOT pass this file through remove_locked_ankles.py.
Do NOT use it as D_REC AMP expert data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# MuJoCo mjtJoint enum values.
MJ_JNT_FREE = 0
MJ_JNT_BALL = 1
MJ_JNT_SLIDE = 2
MJ_JNT_HINGE = 3

RESET_FRAME_SIZE = 59
FORMAT_NAME = "G1RecoveryResetV1"

TARGET_JOINTS: List[str] = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

FRAME_LAYOUT = {
    "root_pos_local_w": [0, 3],
    "root_quat_wxyz_w": [3, 7],
    "root_lin_vel_w": [7, 10],
    "root_ang_vel_w": [10, 13],
    "joint_pos_23": [13, 36],
    "joint_vel_23": [36, 59],
}


def _to_string_list(value: np.ndarray) -> List[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    out: List[str] = []
    for item in arr.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        out.append(str(item))
    return out


def _scalar_float(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def _build_joint_index_maps(
    joint_names: List[str],
    jnt_type: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Reproduce LocoMuJoCo qpos/qvel indexing from joint metadata."""
    jnt_type = np.asarray(jnt_type).reshape(-1)
    if len(joint_names) != len(jnt_type):
        raise ValueError(
            f"joint_names has {len(joint_names)} entries but jnt_type has {len(jnt_type)}."
        )

    qpos_map: Dict[str, np.ndarray] = {}
    qvel_map: Dict[str, np.ndarray] = {}
    qpos_cursor = 0
    qvel_cursor = 0

    for name, joint_type_raw in zip(joint_names, jnt_type):
        joint_type = int(joint_type_raw)

        if joint_type == MJ_JNT_FREE:
            qpos_dim, qvel_dim = 7, 6
        elif joint_type == MJ_JNT_BALL:
            qpos_dim, qvel_dim = 4, 3
        elif joint_type in (MJ_JNT_SLIDE, MJ_JNT_HINGE):
            qpos_dim, qvel_dim = 1, 1
        else:
            raise ValueError(
                f"Unsupported MuJoCo joint type {joint_type} for '{name}'."
            )

        qpos_map[name] = np.arange(
            qpos_cursor, qpos_cursor + qpos_dim, dtype=np.int64
        )
        qvel_map[name] = np.arange(
            qvel_cursor, qvel_cursor + qvel_dim, dtype=np.int64
        )
        qpos_cursor += qpos_dim
        qvel_cursor += qvel_dim

    return qpos_map, qvel_map


def _select_trajectory_slice(
    num_frames: int,
    split_points: np.ndarray | None,
    trajectory_index: int,
) -> Tuple[int, int, np.ndarray]:
    """Return global [start, end) for one trajectory in a stacked NPZ."""
    if split_points is None or np.asarray(split_points).size == 0:
        splits = np.asarray([0, num_frames], dtype=np.int64)
    else:
        splits = np.asarray(split_points, dtype=np.int64).reshape(-1)

        if splits[0] != 0:
            splits = np.concatenate(([0], splits))

        if splits[-1] != num_frames:
            if splits[-1] < num_frames:
                splits = np.concatenate((splits, [num_frames]))
            else:
                raise ValueError(
                    f"split_points ends at {splits[-1]}, beyond qpos length {num_frames}."
                )

    if np.any(np.diff(splits) <= 0):
        raise ValueError(
            f"split_points must be strictly increasing, got {splits.tolist()}."
        )

    num_trajectories = len(splits) - 1
    if trajectory_index < 0 or trajectory_index >= num_trajectories:
        raise IndexError(
            f"trajectory_index={trajectory_index} invalid; "
            f"file contains {num_trajectories} trajectories."
        )

    start = int(splits[trajectory_index])
    end = int(splits[trajectory_index + 1])
    return start, end, splits


def _resolve_local_crop(
    trajectory_num_frames: int,
    fps: float,
    start_time: float | None,
    end_time: float | None,
    start_frame: int | None,
    end_frame: int | None,
) -> Tuple[int, int]:
    """
    Resolve a crop as local [start, end) frame indices.

    Time mode uses an inclusive time interval [start_time, end_time]:
      start = first frame whose timestamp >= start_time
      end   = one past last frame whose timestamp <= end_time

    Frame mode uses Python slicing semantics [start_frame, end_frame).
    """
    use_time = start_time is not None or end_time is not None
    use_frame = start_frame is not None or end_frame is not None

    if use_time and use_frame:
        raise ValueError(
            "Use either --start-time/--end-time OR --start-frame/--end-frame, not both."
        )

    if use_frame:
        s = 0 if start_frame is None else int(start_frame)
        e = trajectory_num_frames if end_frame is None else int(end_frame)

        if s < 0 or e < 0:
            raise ValueError("Frame indices must be >= 0.")
        if s >= e:
            raise ValueError(f"Require start_frame < end_frame, got {s} >= {e}.")
        if e > trajectory_num_frames:
            raise ValueError(
                f"end_frame={e} exceeds trajectory frame count {trajectory_num_frames}."
            )
        return s, e

    duration_last_frame = (
        (trajectory_num_frames - 1) / fps if trajectory_num_frames > 0 else 0.0
    )

    s_time = 0.0 if start_time is None else float(start_time)
    e_time = duration_last_frame if end_time is None else float(end_time)

    if not math.isfinite(s_time) or not math.isfinite(e_time):
        raise ValueError("start/end time must be finite.")
    if s_time < 0.0:
        raise ValueError(f"start_time must be >= 0, got {s_time}.")
    if e_time < s_time:
        raise ValueError(
            f"Require end_time >= start_time, got {e_time} < {s_time}."
        )
    if e_time > duration_last_frame + 1e-9:
        raise ValueError(
            f"end_time={e_time:.6f}s exceeds trajectory last-frame time "
            f"{duration_last_frame:.6f}s."
        )

    # First source sample at or after start_time.
    s = int(math.ceil(s_time * fps - 1e-9))
    # Include the last source sample whose timestamp is <= end_time.
    last_inclusive = int(math.floor(e_time * fps + 1e-9))
    e = last_inclusive + 1

    s = max(0, min(s, trajectory_num_frames - 1))
    e = max(s + 1, min(e, trajectory_num_frames))
    return s, e


def _normalize_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quaternion [T,4], got {quat_wxyz.shape}.")

    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norm[:, 0] < 1e-8):
        bad = np.where(norm[:, 0] < 1e-8)[0][:10].tolist()
        raise ValueError(f"Near-zero root quaternion at local crop frames {bad}.")

    return quat_wxyz / norm


def _mujoco_local_ang_vel_to_world(
    quat_wxyz_world: np.ndarray,
    ang_vel_local: np.ndarray,
) -> np.ndarray:
    """
    Convert MuJoCo free-joint angular velocity from body-local frame to world frame.

    scipy Rotation expects XYZW quaternion ordering.
    """
    quat_xyzw = quat_wxyz_world[:, [1, 2, 3, 0]]
    rot_w_from_b = Rotation.from_quat(quat_xyzw)
    return rot_w_from_b.apply(np.asarray(ang_vel_local, dtype=np.float64))


def convert(
    input_npz: Path,
    output_txt: Path,
    trajectory_index: int,
    start_time: float | None,
    end_time: float | None,
    start_frame: int | None,
    end_frame: int | None,
    source_fps_override: float | None,
    keep_source_xy: bool,
) -> None:
    if not input_npz.is_file():
        raise FileNotFoundError(f"Input file not found: {input_npz}")

    with np.load(input_npz, allow_pickle=True) as data:
        keys = list(data.files)
        print(f"[INFO] Input: {input_npz}")
        print(f"[INFO] NPZ keys ({len(keys)}): {keys}")

        required = ["qpos", "qvel", "joint_names", "jnt_type"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"Missing required key(s): {missing}. Available keys: {keys}"
            )

        qpos_all = np.asarray(data["qpos"], dtype=np.float64)
        qvel_all = np.asarray(data["qvel"], dtype=np.float64)
        joint_names = _to_string_list(data["joint_names"])
        jnt_type = np.asarray(data["jnt_type"]).reshape(-1)
        split_points = (
            np.asarray(data["split_points"]) if "split_points" in data else None
        )

        if qpos_all.ndim != 2 or qvel_all.ndim != 2:
            raise ValueError(
                f"Expected qpos/qvel 2-D. Got qpos={qpos_all.shape}, qvel={qvel_all.shape}."
            )
        if qpos_all.shape[0] != qvel_all.shape[0]:
            raise ValueError(
                f"qpos/qvel frame count mismatch: "
                f"{qpos_all.shape[0]} vs {qvel_all.shape[0]}."
            )
        if not np.isfinite(qpos_all).all() or not np.isfinite(qvel_all).all():
            raise ValueError("Input qpos/qvel contains NaN or Inf.")

        if source_fps_override is not None:
            fps = float(source_fps_override)
            print(f"[INFO] Using --source-fps override: {fps:.6f} Hz")
        elif "frequency" in data:
            fps = _scalar_float(data["frequency"])
            print(f"[INFO] Source frequency from NPZ: {fps:.6f} Hz")
        else:
            raise KeyError(
                "NPZ has no 'frequency'. Pass --source-fps explicitly."
            )

        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"FPS must be positive and finite, got {fps}.")

        qpos_map, qvel_map = _build_joint_index_maps(joint_names, jnt_type)

        expected_qpos_dim = max(int(idx[-1]) for idx in qpos_map.values()) + 1
        expected_qvel_dim = max(int(idx[-1]) for idx in qvel_map.values()) + 1
        if qpos_all.shape[1] != expected_qpos_dim:
            raise ValueError(
                f"qpos width {qpos_all.shape[1]} does not match metadata "
                f"width {expected_qpos_dim}."
            )
        if qvel_all.shape[1] != expected_qvel_dim:
            raise ValueError(
                f"qvel width {qvel_all.shape[1]} does not match metadata "
                f"width {expected_qvel_dim}."
            )

        free_joint_names = [
            name
            for name, joint_type in zip(joint_names, jnt_type)
            if int(joint_type) == MJ_JNT_FREE
        ]
        if "root" in free_joint_names:
            root_name = "root"
        elif len(free_joint_names) == 1:
            root_name = free_joint_names[0]
        elif not free_joint_names:
            raise RuntimeError(
                "No MuJoCo free joint found; cannot build floating-base reset state."
            )
        else:
            raise RuntimeError(
                f"Multiple free joints found and none is named 'root': {free_joint_names}"
            )

        missing_target = [
            name for name in TARGET_JOINTS if name not in qpos_map
        ]
        if missing_target:
            raise KeyError(
                "NPZ is not compatible with current bipped_lab Unitree G1 23-DoF. "
                f"Missing joints: {missing_target}"
            )

        for name in TARGET_JOINTS:
            if len(qpos_map[name]) != 1 or len(qvel_map[name]) != 1:
                raise ValueError(
                    f"Target joint '{name}' is not 1-DoF: "
                    f"qpos_idx={qpos_map[name]}, qvel_idx={qvel_map[name]}"
                )

        traj_global_start, traj_global_end, splits = _select_trajectory_slice(
            num_frames=qpos_all.shape[0],
            split_points=split_points,
            trajectory_index=trajectory_index,
        )

        traj_num_frames = traj_global_end - traj_global_start
        local_start, local_end = _resolve_local_crop(
            trajectory_num_frames=traj_num_frames,
            fps=fps,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        global_start = traj_global_start + local_start
        global_end = traj_global_start + local_end

        qpos = qpos_all[global_start:global_end]
        qvel = qvel_all[global_start:global_end]

        root_qpos_idx = qpos_map[root_name]
        root_qvel_idx = qvel_map[root_name]
        if len(root_qpos_idx) != 7 or len(root_qvel_idx) != 6:
            raise ValueError(
                f"Root '{root_name}' is not a standard MuJoCo free joint: "
                f"qpos_dim={len(root_qpos_idx)}, qvel_dim={len(root_qvel_idx)}"
            )

        root_qpos = qpos[:, root_qpos_idx]
        root_qvel = qvel[:, root_qvel_idx]

        # MuJoCo free joint qpos = xyz + qw qx qy qz.
        root_pos = root_qpos[:, 0:3].copy()
        root_quat_wxyz = _normalize_quaternion_wxyz(root_qpos[:, 3:7])

        # For portable reset placement, default to zero source X/Y while preserving
        # the source root height above its original ground.
        if not keep_source_xy:
            root_pos[:, 0:2] = 0.0

        # MuJoCo free joint qvel:
        #   first 3: linear velocity in WORLD frame
        #   last  3: angular velocity in LOCAL body frame
        root_lin_vel_w = root_qvel[:, 0:3].copy()
        root_ang_vel_local = root_qvel[:, 3:6].copy()
        root_ang_vel_w = _mujoco_local_ang_vel_to_world(
            root_quat_wxyz,
            root_ang_vel_local,
        )

        joint_pos = np.stack(
            [qpos[:, int(qpos_map[name][0])] for name in TARGET_JOINTS],
            axis=1,
        )
        joint_vel = np.stack(
            [qvel[:, int(qvel_map[name][0])] for name in TARGET_JOINTS],
            axis=1,
        )

        frames = np.concatenate(
            [
                root_pos,          # 3
                root_quat_wxyz,    # 4
                root_lin_vel_w,    # 3
                root_ang_vel_w,    # 3
                joint_pos,         # 23
                joint_vel,         # 23
            ],
            axis=1,
        ).astype(np.float32)

        if frames.ndim != 2 or frames.shape[1] != RESET_FRAME_SIZE:
            raise RuntimeError(
                f"Internal error: expected [T,{RESET_FRAME_SIZE}], got {frames.shape}."
            )
        if frames.shape[0] <= 0:
            raise RuntimeError("Crop produced zero frames.")
        if not np.isfinite(frames).all():
            raise ValueError("Converted reset frames contain NaN or Inf.")

        quat_norm = np.linalg.norm(frames[:, 3:7], axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError(
                "Output root quaternion is not unit length. "
                f"norm range=[{quat_norm.min()}, {quat_norm.max()}]"
            )

        # Useful reset-specific diagnostics.
        local_first_time = local_start / fps
        local_last_time = (local_end - 1) / fps

        payload = {
            "Format": FORMAT_NAME,
            "Version": 1,
            "FrameDuration": 1.0 / fps,
            "FPS": fps,
            "FrameSize": RESET_FRAME_SIZE,
            "QuaternionConvention": "WXYZ",
            "RootPoseFrame": "world orientation; position local to target env placement",
            "RootLinearVelocityFrame": "world",
            "RootAngularVelocityFrame": "world",
            "SourceMuJoCoAngularVelocityFrame": "body-local",
            "JointNames": TARGET_JOINTS,
            "FrameLayout": FRAME_LAYOUT,
            "RootXYMode": "source" if keep_source_xy else "zeroed_for_env_origin_placement",
            "Source": {
                "InputNPZ": str(input_npz),
                "TrajectoryIndex": int(trajectory_index),
                "TrajectoryGlobalFrameRange": [
                    int(traj_global_start),
                    int(traj_global_end),
                ],
                "CropLocalFrameRange": [
                    int(local_start),
                    int(local_end),
                ],
                "CropGlobalFrameRange": [
                    int(global_start),
                    int(global_end),
                ],
                "CropLocalTimeRangeSeconds": [
                    float(local_first_time),
                    float(local_last_time),
                ],
                "SplitPoints": splits.tolist(),
            },
            "Frames": frames.tolist(),
        }

        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with output_txt.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")

        print()
        print("[SUCCESS] Recovery reset crop written.")
        print(f"[SUCCESS] Output: {output_txt}")
        print(f"[SUCCESS] Format: {FORMAT_NAME}")
        print(f"[SUCCESS] Frames: {frames.shape[0]}")
        print(f"[SUCCESS] Dimension: {frames.shape[1]}")
        print(f"[SUCCESS] FPS: {fps:.6f}")
        print(
            "[SUCCESS] Local crop time actually stored: "
            f"{local_first_time:.6f}s -> {local_last_time:.6f}s"
        )
        print(
            "[SUCCESS] Local crop frames [start,end): "
            f"[{local_start}, {local_end})"
        )
        print(
            "[SUCCESS] Global source frames [start,end): "
            f"[{global_start}, {global_end})"
        )
        print(
            "[SUCCESS] Quaternion norm range: "
            f"[{quat_norm.min():.7f}, {quat_norm.max():.7f}]"
        )
        print(
            "[SUCCESS] Root angular velocity was converted "
            "MuJoCo body-local -> IsaacLab-compatible world frame."
        )
        if keep_source_xy:
            print("[INFO] Source root X/Y were preserved.")
        else:
            print(
                "[INFO] Source root X/Y were zeroed. "
                "At reset, add the target env origin / desired XY placement."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop a LocoMuJoCo Unitree G1 NPZ into a 59-D reset-ready "
            "Recovery physical-state file."
        )
    )

    parser.add_argument(
        "--input_npz",
        type=Path,
        required=True,
        help="Source LocoMuJoCo Unitree G1 NPZ.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        required=True,
        help="Output reset-state JSON/TXT file.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Trajectory index if source NPZ contains split_points. Default: 0.",
    )

    time_group = parser.add_argument_group("time crop")
    time_group.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Crop start time in seconds, relative to selected trajectory.",
    )
    time_group.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Crop end time in seconds, relative to selected trajectory. Inclusive.",
    )

    frame_group = parser.add_argument_group("frame crop")
    frame_group.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Local start frame relative to selected trajectory.",
    )
    frame_group.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Local end frame EXCLUSIVE, relative to selected trajectory.",
    )

    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override NPZ frequency metadata only if necessary.",
    )
    parser.add_argument(
        "--keep-source-xy",
        action="store_true",
        help=(
            "Preserve source root X/Y. Default behavior zeroes source X/Y so "
            "the reset state can be placed relative to an IsaacLab env origin."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert(
        input_npz=args.input_npz.expanduser().resolve(),
        output_txt=args.output_txt.expanduser().resolve(),
        trajectory_index=args.trajectory_index,
        start_time=args.start_time,
        end_time=args.end_time,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        source_fps_override=args.source_fps,
        keep_source_xy=args.keep_source_xy,
    )


if __name__ == "__main__":
    main()
