#!/usr/bin/env python3
"""Convert a LocoMuJoCo UnitreeG1 trajectory (.npz) to bipped_lab visualization motion JSON/TXT.

Output frame layout expected by bipped_lab G1 visualize_motion():
    root_pos(3)
    root_euler_XYZ(3)
    joint_pos(23)
    root_vel(6)
    joint_vel(23)
Total: 58 values per frame.

This script intentionally does NOT run MuJoCo or IsaacLab. It only converts the
stored LocoMuJoCo qpos/qvel trajectory into the visualization format already
consumed by bipped_lab. The converted file can then be replayed in IsaacLab via
play_amp_animation.py, which recomputes the hand/foot FK used by AMP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# MuJoCo mjtJoint enum values.
MJ_JNT_FREE = 0
MJ_JNT_BALL = 1
MJ_JNT_SLIDE = 2
MJ_JNT_HINGE = 3


# Exact 23-DoF order expected by the current bipped_lab G1 visualization code:
# left leg(6) -> right leg(6) -> waist(1) -> left arm(5) -> right arm(5)
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
    joint_names: List[str], jnt_type: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Reproduce LocoMuJoCo's qpos/qvel indexing from joint names and joint types."""
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
            raise ValueError(f"Unsupported MuJoCo joint type {joint_type} for '{name}'.")

        qpos_map[name] = np.arange(qpos_cursor, qpos_cursor + qpos_dim, dtype=np.int64)
        qvel_map[name] = np.arange(qvel_cursor, qvel_cursor + qvel_dim, dtype=np.int64)
        qpos_cursor += qpos_dim
        qvel_cursor += qvel_dim

    return qpos_map, qvel_map


def _select_trajectory_slice(
    num_frames: int,
    split_points: np.ndarray | None,
    trajectory_index: int,
) -> Tuple[int, int, np.ndarray]:
    """Return [start, end) for one trajectory stored inside a stacked LocoMuJoCo file."""
    if split_points is None or np.asarray(split_points).size == 0:
        splits = np.asarray([0, num_frames], dtype=np.int64)
    else:
        splits = np.asarray(split_points, dtype=np.int64).reshape(-1)

        # Be tolerant of old/simple files that contain only starts and omit the final end.
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
        raise ValueError(f"split_points must be strictly increasing, got {splits.tolist()}.")

    num_trajectories = len(splits) - 1
    if trajectory_index < 0 or trajectory_index >= num_trajectories:
        raise IndexError(
            f"trajectory_index={trajectory_index} is invalid; file contains "
            f"{num_trajectories} trajectory/trajectories."
        )

    start = int(splits[trajectory_index])
    end = int(splits[trajectory_index + 1])
    return start, end, splits


def _root_quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo root quaternion [w,x,y,z] to the XYZ Euler convention used by bipped_lab."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quaternion array [T,4], got {quat_wxyz.shape}.")

    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        bad = np.where(norms[:, 0] < 1e-8)[0][:10].tolist()
        raise ValueError(f"Invalid near-zero root quaternion at local frame(s): {bad}")

    quat_wxyz = quat_wxyz / norms

    # scipy Rotation expects [x, y, z, w].
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_euler("XYZ", degrees=False)


def convert(
    input_npz: Path,
    output_txt: Path,
    trajectory_index: int,
    motion_weight: float,
    root_z_offset: float,
    source_fps_override: float | None,
    inspect_only: bool,
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
                "This does not look like a current LocoMuJoCo Trajectory NPZ. "
                f"Missing required key(s): {missing}. Available keys: {keys}"
            )

        qpos_all = np.asarray(data["qpos"], dtype=np.float64)
        qvel_all = np.asarray(data["qvel"], dtype=np.float64)
        joint_names = _to_string_list(data["joint_names"])
        jnt_type = np.asarray(data["jnt_type"]).reshape(-1)
        split_points = np.asarray(data["split_points"]) if "split_points" in data else None

        if qpos_all.ndim != 2 or qvel_all.ndim != 2:
            raise ValueError(
                f"Expected qpos/qvel to be 2-D. Got qpos={qpos_all.shape}, qvel={qvel_all.shape}."
            )
        if qpos_all.shape[0] != qvel_all.shape[0]:
            raise ValueError(
                f"qpos and qvel frame counts differ: {qpos_all.shape[0]} vs {qvel_all.shape[0]}."
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
                "NPZ has no 'frequency' key. Pass --source-fps explicitly, e.g. --source-fps 30."
            )

        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"FPS must be positive and finite, got {fps}.")

        qpos_map, qvel_map = _build_joint_index_maps(joint_names, jnt_type)

        expected_qpos_dim = max(int(idx[-1]) for idx in qpos_map.values()) + 1
        expected_qvel_dim = max(int(idx[-1]) for idx in qvel_map.values()) + 1
        if qpos_all.shape[1] != expected_qpos_dim:
            raise ValueError(
                f"qpos width {qpos_all.shape[1]} does not match joint metadata width {expected_qpos_dim}."
            )
        if qvel_all.shape[1] != expected_qvel_dim:
            raise ValueError(
                f"qvel width {qvel_all.shape[1]} does not match joint metadata width {expected_qvel_dim}."
            )

        free_joint_names = [
            name for name, jt in zip(joint_names, jnt_type) if int(jt) == MJ_JNT_FREE
        ]
        if not free_joint_names:
            raise RuntimeError("No MuJoCo free joint found; cannot extract floating root pose.")
        if "root" in free_joint_names:
            root_name = "root"
        elif len(free_joint_names) == 1:
            root_name = free_joint_names[0]
        else:
            raise RuntimeError(
                f"Multiple free joints found and none is named 'root': {free_joint_names}"
            )

        missing_target_joints = [name for name in TARGET_JOINTS if name not in qpos_map]
        if missing_target_joints:
            raise KeyError(
                "The NPZ is not compatible with the expected Unitree G1 23-DoF joint set. "
                f"Missing: {missing_target_joints}\n"
                f"Available joints: {joint_names}"
            )

        for name in TARGET_JOINTS:
            if len(qpos_map[name]) != 1 or len(qvel_map[name]) != 1:
                raise ValueError(
                    f"Target joint '{name}' is not a 1-DoF hinge/slide joint: "
                    f"qpos_idx={qpos_map[name]}, qvel_idx={qvel_map[name]}"
                )

        start, end, splits = _select_trajectory_slice(
            qpos_all.shape[0], split_points, trajectory_index
        )
        qpos = qpos_all[start:end]
        qvel = qvel_all[start:end]

        print(f"[INFO] qpos shape: {qpos_all.shape}")
        print(f"[INFO] qvel shape: {qvel_all.shape}")
        print(f"[INFO] joint count: {len(joint_names)}")
        print(f"[INFO] floating root joint: {root_name}")
        print(f"[INFO] split_points: {splits.tolist()}")
        print(
            f"[INFO] Selected trajectory {trajectory_index}: global frames [{start}, {end}), "
            f"count={end - start}"
        )
        print("[INFO] Target 23-DoF joint order:")
        for i, name in enumerate(TARGET_JOINTS):
            print(
                f"  {i:02d}: {name:<30s} "
                f"qpos={int(qpos_map[name][0]):>2d} qvel={int(qvel_map[name][0]):>2d}"
            )

        if inspect_only:
            print("[INFO] --inspect-only requested; no output file written.")
            return

        root_qpos_idx = qpos_map[root_name]
        root_qvel_idx = qvel_map[root_name]
        if len(root_qpos_idx) != 7 or len(root_qvel_idx) != 6:
            raise ValueError(
                f"Root '{root_name}' is not a standard free joint: "
                f"qpos dim={len(root_qpos_idx)}, qvel dim={len(root_qvel_idx)}"
            )

        root_qpos = qpos[:, root_qpos_idx]  # xyz + qw qx qy qz
        root_vel = qvel[:, root_qvel_idx]   # 6-D MuJoCo free-joint velocity

        root_pos = root_qpos[:, 0:3].copy()
        root_pos[:, 2] += float(root_z_offset)
        root_euler_xyz = _root_quat_wxyz_to_euler_xyz(root_qpos[:, 3:7])

        joint_pos = np.stack(
            [qpos[:, int(qpos_map[name][0])] for name in TARGET_JOINTS], axis=1
        )
        joint_vel = np.stack(
            [qvel[:, int(qvel_map[name][0])] for name in TARGET_JOINTS], axis=1
        )

        frames = np.concatenate(
            [
                root_pos,         # 3
                root_euler_xyz,   # 3
                joint_pos,        # 23
                root_vel,         # 6
                joint_vel,        # 23
            ],
            axis=1,
        ).astype(np.float32)

        if frames.shape[1] != 58:
            raise RuntimeError(f"Internal error: expected output width 58, got {frames.shape}.")
        if not np.isfinite(frames).all():
            raise ValueError("Converted frames contain NaN or Inf.")

        output_txt.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "LoopMode": "Wrap",
            "FrameDuration": 1.0 / fps,
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": float(motion_weight),
            "Frames": frames.tolist(),
        }

        with output_txt.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        duration_s = (frames.shape[0] - 1) / fps if frames.shape[0] > 1 else 0.0
        print("\n[SUCCESS] Conversion finished.")
        print(f"[SUCCESS] Output: {output_txt}")
        print(f"[SUCCESS] Frames: {frames.shape[0]}")
        print(f"[SUCCESS] Dimension: {frames.shape[1]}")
        print(f"[SUCCESS] FPS: {fps:.6f}")
        print(f"[SUCCESS] FrameDuration: {1.0 / fps:.9f} s")
        print(f"[SUCCESS] Approx. duration: {duration_s:.3f} s")
        print(f"[SUCCESS] root_z_offset applied here: {root_z_offset:+.4f} m")
        print(
            "[NOTE] Your current bipped_lab visualize_motion() additionally adds +0.1 m "
            "to root z during replay. This converter defaults to 0.0 m to avoid double offsets."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LocoMuJoCo UnitreeG1 .npz trajectory to the 58-D "
            "bipped_lab visualization motion format."
        )
    )
    parser.add_argument(
        "--input_npz",
        type=Path,
        required=True,
        help="Path to the LocoMuJoCo UnitreeG1 .npz file.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        required=True,
        help="Output visualization JSON/TXT path.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Which trajectory to extract when split_points contains multiple clips. Default: 0.",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.5,
        help="MotionWeight field written to output. Default: 0.5.",
    )
    parser.add_argument(
        "--root-z-offset",
        type=float,
        default=0.0,
        help=(
            "Optional root-height offset in meters applied during conversion. "
            "Default: 0.0. Note: current bipped_lab visualize_motion() already adds +0.1 m."
        ),
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override source FPS only if the NPZ has no/incorrect frequency metadata.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print NPZ structure and joint mapping without writing an output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        input_npz=args.input_npz.expanduser().resolve(),
        output_txt=args.output_txt.expanduser().resolve(),
        trajectory_index=args.trajectory_index,
        motion_weight=args.motion_weight,
        root_z_offset=args.root_z_offset,
        source_fps_override=args.source_fps,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()