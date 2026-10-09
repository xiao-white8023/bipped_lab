#!/usr/bin/env python3
"""Apply reproducible morphology-aware arm calibration to a G1 motion PKL."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


LEFT_KNEE = 3
LEFT_HIP_PITCH = 0
LEFT_ANKLE_PITCH = 4
LEFT_ANKLE_ROLL = 5
RIGHT_KNEE = 9
RIGHT_HIP_PITCH = 6
RIGHT_ANKLE_PITCH = 10
RIGHT_ANKLE_ROLL = 11
LEFT_HIP_YAW = 2
RIGHT_HIP_YAW = 8
WAIST_YAW = 12
LEFT_SHOULDER_PITCH = 13
LEFT_SHOULDER_ROLL = 14
LEFT_SHOULDER_YAW = 15
LEFT_ELBOW = 16
RIGHT_SHOULDER_PITCH = 18
RIGHT_SHOULDER_ROLL = 19
RIGHT_SHOULDER_YAW = 20
RIGHT_ELBOW = 21


def _ground_root_height(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    foot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in foot_body_ids
        and int(model.geom_contype[geom_id]) != 0
    ]
    nearest = np.empty(6, dtype=np.float64)
    grounded = root_pos.copy()
    for frame_id in range(len(grounded)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = grounded[frame_id]
        data.qpos[3:7] = root_rot_xyzw[frame_id, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame_id]
        mujoco.mj_forward(model, data)
        lowest = min(
            mujoco.mj_geomDistance(model, data, floor_id, geom_id, 2.0, nearest)
            for geom_id in foot_geom_ids
        )
        grounded[frame_id, 2] -= lowest
    return grounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shoulder-pitch-offset", type=float, default=0.10)
    parser.add_argument("--elbow-offset", type=float, default=0.10)
    parser.add_argument("--knee-offset", type=float, default=0.0)
    parser.add_argument("--knee-flex-scale", type=float, default=1.0)
    parser.add_argument("--left-knee-scale", type=float, default=1.0)
    parser.add_argument("--right-knee-scale", type=float, default=1.0)
    parser.add_argument("--left-knee-offset", type=float, default=0.0)
    parser.add_argument("--right-knee-offset", type=float, default=0.0)
    parser.add_argument(
        "--knee-min",
        type=float,
        default=None,
        help="Optional minimum knee flexion in radians; use 0 to remove hyperextension without bending every frame.",
    )
    parser.add_argument("--symmetrize-shoulder-pitch", action="store_true")
    parser.add_argument("--shoulder-swing-scale", type=float, default=1.0)
    parser.add_argument("--symmetrize-shoulder-frontal", action="store_true")
    parser.add_argument("--shoulder-roll-scale", type=float, default=1.0)
    parser.add_argument("--shoulder-yaw-scale", type=float, default=1.0)
    parser.add_argument("--symmetrize-elbows", action="store_true")
    parser.add_argument("--waist-yaw-scale", type=float, default=1.0)
    parser.add_argument("--hip-yaw-scale", type=float, default=1.0)
    parser.add_argument("--left-hip-pitch-offset", type=float, default=0.0)
    parser.add_argument("--right-hip-pitch-offset", type=float, default=0.0)
    parser.add_argument("--hip-roll-scale", type=float, default=1.0)
    parser.add_argument("--hip-roll-mean-scale", type=float, default=1.0)
    parser.add_argument(
        "--preserve-foot-roll",
        action="store_true",
        help="Compensate removed hip roll at the ankle to retain the retargeted foot orientation.",
    )
    parser.add_argument(
        "--preserve-foot-pitch",
        action="store_true",
        help="Compensate hip/knee sagittal calibration at the ankle.",
    )
    parser.add_argument("--root-yaw-sway-scale", type=float, default=1.0)
    parser.add_argument(
        "--compensate-root-yaw",
        action="store_true",
        help="Move removed root-yaw sway into both hip-yaw joints and the waist to preserve world-space limb headings.",
    )
    args = parser.parse_args()

    with args.input.expanduser().resolve().open("rb") as stream:
        motion = pickle.load(stream)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64).copy()
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 23:
        raise ValueError(f"Expected a (T, 23) dof_pos array, got {dof_pos.shape}")
    original_leg_sagittal = dof_pos[
        :, [LEFT_HIP_PITCH, LEFT_KNEE, RIGHT_HIP_PITCH, RIGHT_KNEE]
    ].copy()

    if args.symmetrize_shoulder_pitch:
        left = dof_pos[:, LEFT_SHOULDER_PITCH]
        right = dof_pos[:, RIGHT_SHOULDER_PITCH]
        left_mean = float(left.mean())
        right_mean = float(right.mean())
        common_mean = 0.5 * (left_mean + right_mean)
        # The antisymmetric component is the human gait arm swing.  Removing
        # the common-mode component prevents both arms drifting forward or
        # backward together while retaining their contralateral leg phase.
        swing = 0.5 * ((left - left_mean) - (right - right_mean))
        swing *= args.shoulder_swing_scale
        dof_pos[:, LEFT_SHOULDER_PITCH] = common_mean + swing
        dof_pos[:, RIGHT_SHOULDER_PITCH] = common_mean - swing

    if args.symmetrize_shoulder_frontal:
        # G1's roll/yaw signs are mirrored.  Keeping only the mirrored
        # component removes mocap calibration bias that otherwise leaves one
        # arm farther away from the torso than the other.
        roll = 0.5 * (
            dof_pos[:, LEFT_SHOULDER_ROLL]
            - dof_pos[:, RIGHT_SHOULDER_ROLL]
        )
        yaw = 0.5 * (
            dof_pos[:, RIGHT_SHOULDER_YAW]
            - dof_pos[:, LEFT_SHOULDER_YAW]
        )
        dof_pos[:, LEFT_SHOULDER_ROLL] = args.shoulder_roll_scale * roll
        dof_pos[:, RIGHT_SHOULDER_ROLL] = -args.shoulder_roll_scale * roll
        dof_pos[:, LEFT_SHOULDER_YAW] = -args.shoulder_yaw_scale * yaw
        dof_pos[:, RIGHT_SHOULDER_YAW] = args.shoulder_yaw_scale * yaw
    else:
        dof_pos[:, [LEFT_SHOULDER_ROLL, RIGHT_SHOULDER_ROLL]] *= (
            args.shoulder_roll_scale
        )
        dof_pos[:, [LEFT_SHOULDER_YAW, RIGHT_SHOULDER_YAW]] *= (
            args.shoulder_yaw_scale
        )

    if args.symmetrize_elbows:
        left_mean = float(dof_pos[:, LEFT_ELBOW].mean())
        right_mean = float(dof_pos[:, RIGHT_ELBOW].mean())
        common_mean = 0.5 * (left_mean + right_mean)
        dof_pos[:, LEFT_ELBOW] += common_mean - left_mean
        dof_pos[:, RIGHT_ELBOW] += common_mean - right_mean

    dof_pos[:, WAIST_YAW] *= args.waist_yaw_scale
    dof_pos[:, [LEFT_HIP_YAW, RIGHT_HIP_YAW]] *= args.hip_yaw_scale

    original_hip_roll = dof_pos[:, [1, 7]].copy()
    dof_pos[:, [1, 7]] *= args.hip_roll_scale

    if args.hip_roll_mean_scale != 1.0:
        # Keep the alternating coronal balance motion, but reduce the static
        # inward lean of both thighs caused by the human/robot hip-width
        # mismatch.  Left/right hip-roll signs are mirrored on G1.
        left_roll = dof_pos[:, 1]
        right_roll = dof_pos[:, 7]
        mirrored_mean = 0.5 * (float(left_roll.mean()) - float(right_roll.mean()))
        target_left_mean = args.hip_roll_mean_scale * mirrored_mean
        target_right_mean = -target_left_mean
        dof_pos[:, 1] += target_left_mean - float(left_roll.mean())
        dof_pos[:, 7] += target_right_mean - float(right_roll.mean())

    if args.preserve_foot_roll:
        removed_hip_roll = original_hip_roll - dof_pos[:, [1, 7]]
        dof_pos[:, [LEFT_ANKLE_ROLL, RIGHT_ANKLE_ROLL]] += removed_hip_roll

    dof_pos[:, [LEFT_SHOULDER_PITCH, RIGHT_SHOULDER_PITCH]] += (
        args.shoulder_pitch_offset
    )
    dof_pos[:, [LEFT_ELBOW, RIGHT_ELBOW]] += args.elbow_offset
    dof_pos[:, [LEFT_KNEE, RIGHT_KNEE]] += args.knee_offset
    dof_pos[:, [LEFT_KNEE, RIGHT_KNEE]] *= args.knee_flex_scale
    dof_pos[:, LEFT_KNEE] = (
        args.left_knee_scale * dof_pos[:, LEFT_KNEE] + args.left_knee_offset
    )
    dof_pos[:, RIGHT_KNEE] = (
        args.right_knee_scale * dof_pos[:, RIGHT_KNEE] + args.right_knee_offset
    )
    dof_pos[:, LEFT_HIP_PITCH] += args.left_hip_pitch_offset
    dof_pos[:, RIGHT_HIP_PITCH] += args.right_hip_pitch_offset
    if args.knee_min is not None:
        dof_pos[:, [LEFT_KNEE, RIGHT_KNEE]] = np.maximum(
            dof_pos[:, [LEFT_KNEE, RIGHT_KNEE]], args.knee_min
        )
    if args.preserve_foot_pitch:
        left_delta = (
            dof_pos[:, LEFT_HIP_PITCH] - original_leg_sagittal[:, 0]
            + dof_pos[:, LEFT_KNEE] - original_leg_sagittal[:, 1]
        )
        right_delta = (
            dof_pos[:, RIGHT_HIP_PITCH] - original_leg_sagittal[:, 2]
            + dof_pos[:, RIGHT_KNEE] - original_leg_sagittal[:, 3]
        )
        dof_pos[:, LEFT_ANKLE_PITCH] -= left_delta
        dof_pos[:, RIGHT_ANKLE_PITCH] -= right_delta

    root_rot = np.asarray(motion["root_rot"], dtype=np.float64).copy()
    if args.root_yaw_sway_scale != 1.0:
        root_euler = Rotation.from_quat(root_rot).as_euler("XYZ")
        yaw = np.unwrap(root_euler[:, 2])
        heading = np.linspace(yaw[0], yaw[-1], len(yaw))
        adjusted_yaw = heading + args.root_yaw_sway_scale * (yaw - heading)
        if args.compensate_root_yaw:
            removed_yaw = yaw - adjusted_yaw
            dof_pos[:, [LEFT_HIP_YAW, RIGHT_HIP_YAW]] += removed_yaw[:, None]
            dof_pos[:, WAIST_YAW] += removed_yaw
        root_euler[:, 2] = adjusted_yaw
        root_rot = Rotation.from_euler("XYZ", root_euler).as_quat()

    # Exact G1 23-DoF limits for the four adjusted joints.
    dof_pos[:, LEFT_SHOULDER_PITCH] = np.clip(
        dof_pos[:, LEFT_SHOULDER_PITCH], -3.0892, 2.6704
    )
    dof_pos[:, RIGHT_SHOULDER_PITCH] = np.clip(
        dof_pos[:, RIGHT_SHOULDER_PITCH], -3.0892, 2.6704
    )
    dof_pos[:, [LEFT_ELBOW, RIGHT_ELBOW]] = np.clip(
        dof_pos[:, [LEFT_ELBOW, RIGHT_ELBOW]], -1.0472, 2.0944
    )

    model_path = Path(
        "/home/cjh/GMR/assets/unitree_g1_23dof/g1_23dof_rev_1_0.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_limits = model.jnt_range[1:]
    dof_pos = np.clip(dof_pos, joint_limits[:, 0], joint_limits[:, 1])

    motion["dof_pos"] = dof_pos
    motion["root_pos"] = _ground_root_height(
        model,
        np.asarray(motion["root_pos"], dtype=np.float64),
        root_rot,
        dof_pos,
    )
    motion["root_rot"] = root_rot
    motion["arm_posture_calibration"] = {
        "shoulder_pitch_offset_rad": args.shoulder_pitch_offset,
        "elbow_offset_rad": args.elbow_offset,
        "knee_offset_rad": args.knee_offset,
        "knee_flex_scale": args.knee_flex_scale,
        "left_knee_scale": args.left_knee_scale,
        "right_knee_scale": args.right_knee_scale,
        "left_knee_offset_rad": args.left_knee_offset,
        "right_knee_offset_rad": args.right_knee_offset,
        "knee_min_rad": args.knee_min,
        "symmetrize_shoulder_pitch": args.symmetrize_shoulder_pitch,
        "shoulder_swing_scale": args.shoulder_swing_scale,
        "symmetrize_shoulder_frontal": args.symmetrize_shoulder_frontal,
        "shoulder_roll_scale": args.shoulder_roll_scale,
        "shoulder_yaw_scale": args.shoulder_yaw_scale,
        "symmetrize_elbows": args.symmetrize_elbows,
        "waist_yaw_scale": args.waist_yaw_scale,
        "hip_yaw_scale": args.hip_yaw_scale,
        "left_hip_pitch_offset_rad": args.left_hip_pitch_offset,
        "right_hip_pitch_offset_rad": args.right_hip_pitch_offset,
        "hip_roll_scale": args.hip_roll_scale,
        "hip_roll_mean_scale": args.hip_roll_mean_scale,
        "preserve_foot_roll": args.preserve_foot_roll,
        "preserve_foot_pitch": args.preserve_foot_pitch,
        "root_yaw_sway_scale": args.root_yaw_sway_scale,
        "compensate_root_yaw": args.compensate_root_yaw,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(motion, stream)
    print(f"Saved calibrated motion to {output}")


if __name__ == "__main__":
    main()
