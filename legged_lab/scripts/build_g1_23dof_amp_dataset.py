#!/usr/bin/env python3
"""Build the G1 23-DoF locomotion AMP dataset from GMR pickle files.

The training layout is intentionally generated independently from the 58-D
visualization layout. The legacy default is:

    19 non-ankle joint positions
    19 non-ankle joint velocities
    left hand, right hand, left foot, right foot positions in the root frame

``include_ankles=True`` exports all 23 joint positions and velocities instead,
matching the current ``G1RENetEnv.get_amp_obs_for_expert_trans`` (58-D AMP).
58-D preview files contain root-pose fields and are a different format.

The end-effector positions are evaluated with the same G1 23-DoF MJCF used by
GMR.  This avoids the historical bug where deleting four columns from a 58-D
visualization frame left root-pose values in a nominally 50-D AMP frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES_23 = [
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

ANKLE_INDICES = {4, 5, 10, 11}
AMP_JOINT_INDICES = [index for index in range(23) if index not in ANKLE_INDICES]
AMP_JOINT_NAMES = [JOINT_NAMES_23[index] for index in AMP_JOINT_INDICES]
END_EFFECTOR_NAMES = [
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

CLIP_SOURCES = {
    "walk_forward": "ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz",
    "walk_backward": "ACCAD/Female1Walking_c3d/B5_-_walk_backwards_stageii.npz",
    "walk_left": "ACCAD/Female1Walking_c3d/B22_-_side_step_left_stageii.npz",
    "walk_right": "ACCAD/Female1Walking_c3d/B23_-_side_step_right_stageii.npz",
    "turn_in_place_left": "ACCAD/Female1Walking_c3d/B9_-_walk_turn_left_(90)_stageii.npz",
    "turn_in_place_right": "ACCAD/Female1Walking_c3d/B12_-_walk_turn_right_(90)_stageii.npz",
}

# The walking clips contain a straight approach, one natural 90-degree turn,
# and a straight departure.  Keep the actual turning strides and remove only
# the approach/departure translation.  Linear XY detrending preserves the
# measured pelvis sway while making the resulting root motion turn in place.
TURN_PREPROCESSING = {
    "turn_in_place_left": ("turn_walk_left_full.pkl", 90, 151),
    "turn_in_place_right": ("turn_walk_right_full.pkl", 90, 151),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _load_gmr_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        motion = pickle.load(stream)

    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required.difference(motion)
    if missing:
        raise KeyError(f"{path} is missing fields: {sorted(missing)}")

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    root_pos = _as_float_array(motion["root_pos"], "root_pos")
    root_rot_xyzw = _as_float_array(motion["root_rot"], "root_rot")
    dof_pos = _as_float_array(motion["dof_pos"], "dof_pos")

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: fps must be positive, got {fps}")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos must have shape (T, 3), got {root_pos.shape}")
    if root_rot_xyzw.shape != (root_pos.shape[0], 4):
        raise ValueError(f"{path}: root_rot must have shape (T, 4), got {root_rot_xyzw.shape}")
    if dof_pos.shape != (root_pos.shape[0], 23):
        raise ValueError(f"{path}: dof_pos must have shape (T, 23), got {dof_pos.shape}")
    if root_pos.shape[0] < 3:
        raise ValueError(f"{path}: at least three frames are required")

    quat_norm = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError(f"{path}: root_rot contains a zero quaternion")
    root_rot_xyzw = root_rot_xyzw / quat_norm
    for frame_id in range(1, len(root_rot_xyzw)):
        if np.dot(root_rot_xyzw[frame_id - 1], root_rot_xyzw[frame_id]) < 0.0:
            root_rot_xyzw[frame_id] *= -1.0

    return fps, root_pos, root_rot_xyzw, dof_pos


def _prepare_turn_pkls(input_dir: Path) -> None:
    """Create deterministic in-place turns from natural walking-turn GMR output."""
    for clip_name, (full_name, start, stop) in TURN_PREPROCESSING.items():
        full_path = input_dir / full_name
        output_path = input_dir / f"{clip_name}.pkl"
        with full_path.open("rb") as stream:
            motion = pickle.load(stream)

        frame_count = len(motion["root_pos"])
        if not (0 <= start < stop <= frame_count):
            raise ValueError(
                f"Invalid crop [{start}:{stop}] for {full_path} with {frame_count} frames"
            )

        cropped: dict[str, Any] = {}
        for key, value in motion.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == frame_count:
                cropped[key] = value[start:stop].copy()
            else:
                cropped[key] = value

        root_pos = np.asarray(cropped["root_pos"], dtype=np.float64).copy()
        phase = np.linspace(0.0, 1.0, len(root_pos))[:, None]
        xy_trend = root_pos[0, :2] + phase * (root_pos[-1, :2] - root_pos[0, :2])
        root_pos[:, :2] -= xy_trend
        cropped["root_pos"] = root_pos
        cropped["preprocessing"] = {
            "source_gmr_pkl": full_name,
            "frame_slice": [start, stop],
            "root_xy": "linear_detrend_preserving_pelvis_sway",
        }

        with output_path.open("wb") as stream:
            pickle.dump(cropped, stream)
        print(f"Prepared {output_path.name} from {full_name}[{start}:{stop}]")


def _angular_velocity(root_rot_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rotations = Rotation.from_quat(root_rot_xyzw)
    velocity = np.empty((len(rotations), 3), dtype=np.float64)
    velocity[0] = (rotations[0].inv() * rotations[1]).as_rotvec() / dt
    velocity[-1] = (rotations[-2].inv() * rotations[-1]).as_rotvec() / dt
    velocity[1:-1] = (rotations[:-2].inv() * rotations[2:]).as_rotvec() / (2.0 * dt)
    return velocity


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, dt, axis=0, edge_order=2)


def _ground_root_height(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the lowest collision foot geom exactly on the ground each frame."""
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in FOOT_BODY_NAMES
    }
    foot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in foot_body_ids
        and int(model.geom_contype[geom_id]) != 0
    ]
    if floor_id < 0 or not foot_geom_ids:
        raise ValueError("The MJCF must contain a floor and collision geoms for both feet")

    grounded = root_pos.copy()
    height_shifts = np.empty(len(root_pos), dtype=np.float64)
    nearest_points = np.empty(6, dtype=np.float64)
    for frame_id in range(len(root_pos)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_pos[frame_id]
        data.qpos[3:7] = root_rot_xyzw[frame_id, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame_id]
        mujoco.mj_forward(model, data)
        lowest_distance = min(
            mujoco.mj_geomDistance(
                model,
                data,
                floor_id,
                geom_id,
                2.0,
                nearest_points,
            )
            for geom_id in foot_geom_ids
        )
        height_shifts[frame_id] = -lowest_distance
        grounded[frame_id, 2] += height_shifts[frame_id]

    return grounded, height_shifts


def _forward_kinematics_features(
    model: mujoco.MjModel,
    dof_pos: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in END_EFFECTOR_NAMES
    ]
    if root_id < 0 or any(body_id < 0 for body_id in body_ids):
        raise ValueError("The MJCF does not contain every required AMP body")

    output = np.empty((len(dof_pos), 12), dtype=np.float64)
    for frame_id, joints in enumerate(dof_pos):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = 0.0
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[7:] = joints
        mujoco.mj_forward(model, data)
        root_rotation_world = data.xmat[root_id].reshape(3, 3)
        local_positions = [
            root_rotation_world.T @ (data.xpos[body_id] - data.xpos[root_id])
            for body_id in body_ids
        ]
        output[frame_id] = np.concatenate(local_positions)
    return output


def _write_motion(path: Path, frames: np.ndarray, fps: float, extra_metadata: dict | None = None,
                  *, motion_weight: float = 1.0) -> None:
    if not np.isfinite(motion_weight) or motion_weight <= 0:
        raise ValueError("MotionWeight must be positive and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        stream.write('  "LoopMode": "Clamp",\n')
        stream.write(f'  "FrameDuration": {1.0 / fps:.12g},\n')
        stream.write('  "EnableCycleOffsetPosition": false,\n')
        stream.write('  "EnableCycleOffsetRotation": false,\n')
        stream.write(f'  "MotionWeight": {float(motion_weight)},\n')
        for key, value in (extra_metadata or {}).items():
            stream.write(f'  {json.dumps(key)}: {json.dumps(value)},\n')
        stream.write('  "Frames": [\n')
        for frame_id, frame in enumerate(frames):
            values = ", ".join(f"{value:.8f}" for value in frame)
            suffix = "," if frame_id + 1 < len(frames) else ""
            stream.write(f"    [{values}]{suffix}\n")
        stream.write("  ]\n}\n")


def _motion_metrics(
    model: mujoco.MjModel,
    fps: float,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    end_positions: np.ndarray,
    grounding_height_shifts: np.ndarray,
) -> dict[str, Any]:
    joint_limits = model.jnt_range[1:]
    lower_violation = np.maximum(joint_limits[:, 0] - dof_pos, 0.0)
    upper_violation = np.maximum(dof_pos - joint_limits[:, 1], 0.0)
    max_limit_violation = float(max(lower_violation.max(), upper_violation.max()))

    rotations = Rotation.from_quat(root_rot_xyzw)
    yaw = np.unwrap(rotations.as_euler("XYZ", degrees=False)[:, 2])
    displacement_world = root_pos[-1] - root_pos[0]
    displacement_initial_root = rotations[0].inv().apply(displacement_world)

    return {
        "frames": int(len(root_pos)),
        "fps": float(fps),
        "duration_s": float((len(root_pos) - 1) / fps),
        "root_height_m": {
            "min": float(root_pos[:, 2].min()),
            "mean": float(root_pos[:, 2].mean()),
            "max": float(root_pos[:, 2].max()),
        },
        "root_displacement_world_m": displacement_world.tolist(),
        "root_displacement_initial_root_m": displacement_initial_root.tolist(),
        "root_yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "max_joint_speed_rad_s": float(np.abs(dof_vel).max()),
        "max_joint_limit_violation_rad": max_limit_violation,
        "collision_grounding_height_shift_m": {
            "min": float(grounding_height_shifts.min()),
            "mean": float(grounding_height_shifts.mean()),
            "max": float(grounding_height_shifts.max()),
        },
        "end_effector_position_range_m": {
            name: {
                "min": end_positions[:, 3 * index : 3 * index + 3].min(axis=0).tolist(),
                "max": end_positions[:, 3 * index : 3 * index + 3].max(axis=0).tolist(),
            }
            for index, name in enumerate(END_EFFECTOR_NAMES)
        },
    }


def build_clip(
    model: mujoco.MjModel,
    input_path: Path,
    visualization_path: Path,
    amp_path: Path,
    *,
    preserve_root_pose: bool = False,
    include_ankles: bool = False,
    motion_weight: float = 1.0,
) -> dict[str, Any]:
    fps, root_pos, root_rot_xyzw, dof_pos = _load_gmr_motion(input_path)
    # Reviewed motions may already include shoe-mesh-aware grounding. Preserve
    # those poses rather than applying a second collision-only height edit.
    if preserve_root_pose:
        grounding_height_shifts = np.zeros(len(root_pos), dtype=np.float64)
    else:
        root_pos, grounding_height_shifts = _ground_root_height(
            model, root_pos, root_rot_xyzw, dof_pos,
        )
    dt = 1.0 / fps
    root_lin_vel = _finite_difference(root_pos, dt)
    root_ang_vel = _angular_velocity(root_rot_xyzw, dt)
    dof_vel = _finite_difference(dof_pos, dt)
    end_positions = _forward_kinematics_features(model, dof_pos)

    euler_xyz = np.unwrap(
        Rotation.from_quat(root_rot_xyzw).as_euler("XYZ", degrees=False),
        axis=0,
    )
    visualization_frames = np.concatenate(
        [root_pos, euler_xyz, dof_pos, root_lin_vel, root_ang_vel, dof_vel],
        axis=1,
    )
    amp_frames = np.concatenate(
        [
            dof_pos if include_ankles else dof_pos[:, AMP_JOINT_INDICES],
            dof_vel if include_ankles else dof_vel[:, AMP_JOINT_INDICES],
            end_positions,
        ],
        axis=1,
    )

    if visualization_frames.shape[1] != 58:
        raise RuntimeError(f"Visualization frame width is {visualization_frames.shape[1]}, expected 58")
    expected_amp_dim = 58 if include_ankles else 50
    if amp_frames.shape[1] != expected_amp_dim:
        raise RuntimeError(f"AMP frame width is {amp_frames.shape[1]}, expected {expected_amp_dim}")
    if not np.isfinite(visualization_frames).all() or not np.isfinite(amp_frames).all():
        raise ValueError(f"{input_path}: generated frames contain NaN or Inf")

    metrics = _motion_metrics(
        model,
        fps,
        root_pos,
        root_rot_xyzw,
        dof_pos,
        dof_vel,
        end_positions,
        grounding_height_shifts,
    )
    if metrics["max_joint_limit_violation_rad"] > 1.0e-5:
        raise ValueError(
            f"{input_path}: retargeted joints exceed the MJCF limits by "
            f"{metrics['max_joint_limit_violation_rad']:.6g} rad"
        )

    _write_motion(visualization_path, visualization_frames, fps, motion_weight=motion_weight)
    _write_motion(amp_path, amp_frames, fps, extra_metadata={
        "FrameLayout": "g1_amp_joint_pos_vel_end_effectors_v1",
        "JointCount": 23 if include_ankles else 19,
        "AMPJointNames": JOINT_NAMES_23 if include_ankles else AMP_JOINT_NAMES,
    }, motion_weight=motion_weight)
    metrics.update(
        {
            "gmr_pkl": str(input_path),
            "gmr_pkl_sha256": _sha256(input_path),
            "visualization_file": str(visualization_path),
            "visualization_sha256": _sha256(visualization_path),
            "amp_file": str(amp_path),
            "amp_sha256": _sha256(amp_path),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the six GMR PKLs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset root to populate")
    parser.add_argument("--mjcf", type=Path, required=True, help="G1 23-DoF MJCF used for retargeting")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.expanduser().resolve()))
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    if model_joint_names != JOINT_NAMES_23:
        raise ValueError(
            "MJCF joint order does not match the g1_renet contract:\n"
            f"expected={JOINT_NAMES_23}\nactual={model_joint_names}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    _prepare_turn_pkls(input_dir)
    results: dict[str, Any] = {}
    for clip_name in CLIP_SOURCES:
        input_path = input_dir / f"{clip_name}.pkl"
        results[clip_name] = build_clip(
            model,
            input_path,
            output_dir / "visualization" / f"{clip_name}.txt",
            output_dir / "amp" / f"{clip_name}.txt",
        )
        print(
            f"{clip_name}: {results[clip_name]['frames']} frames, "
            f"{results[clip_name]['duration_s']:.2f}s, "
            f"yaw={results[clip_name]['root_yaw_change_deg']:.1f} deg"
        )

    manifest = {
        "dataset": "g1_23dof_locomotion_v1",
        "format_version": 1,
        "robot": "Unitree G1 23-DoF",
        "mjcf": str(args.mjcf.expanduser().resolve()),
        "mjcf_sha256": _sha256(args.mjcf.expanduser().resolve()),
        "source_dataset": "ACCAD from AMASS, SMPL-X stageii",
        "retargeter": "YanjieZe/GMR unitree_g1_23dof",
        "retarget_settings": {
            "target_fps": 30,
            "max_iter": 50,
            "damping": 0.5,
            "smooth_alpha": 0.08,
            "align_ground": True,
            "height_adjust_save": True,
            "derived_visualization_grounding": (
                "per-frame root-Z correction using the lowest G1 foot collision geom"
            ),
        },
        "source_files": CLIP_SOURCES,
        "turn_preprocessing": {
            clip_name: {
                "source_gmr_pkl": spec[0],
                "frame_slice_python": [spec[1], spec[2]],
                "root_xy": "linear detrend preserving measured pelvis sway",
            }
            for clip_name, spec in TURN_PREPROCESSING.items()
        },
        "joint_order_23": JOINT_NAMES_23,
        "amp_joint_order_19": AMP_JOINT_NAMES,
        "amp_end_effector_order": END_EFFECTOR_NAMES,
        "amp_layout": "joint_pos_19, joint_vel_19, root-frame end_effector_pos_12",
        "visualization_layout": "root_pos_3, root_euler_xyz_3, joint_pos_23, root_lin_vel_3, root_ang_vel_3, joint_vel_23",
        "clips": results,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
