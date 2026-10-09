#!/usr/bin/env python3
"""Convert a retargeted G1 23-DoF CSV into the 58-D preview format.

Expected CSV row layout::

    root_xyz(3), root_quat_xyzw(4), joint_pos(23)

The output is consumed by ``play_amp_animation.py --motion_file``.  Joint
angles and root orientation are preserved.  By default only a constant XY
offset is removed so the motion starts near the viewer origin.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

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


def _load_urdf_limits(urdf_path: Path) -> tuple[list[str], np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    limits: list[tuple[float, float]] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") not in {"revolute", "continuous"}:
            continue
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if joint.attrib.get("type") == "continuous":
            lower, upper = -np.inf, np.inf
        elif limit is None:
            raise ValueError(f"URDF revolute joint '{name}' has no limit element")
        else:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        names.append(name)
        limits.append((lower, upper))
    return names, np.asarray(limits, dtype=np.float64)


def _normalize_quaternions(quat_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("CSV contains a zero root quaternion")
    normalized = quat_xyzw / norms
    for index in range(1, len(normalized)):
        if np.dot(normalized[index - 1], normalized[index]) < 0.0:
            normalized[index] *= -1.0
    return normalized


def _angular_velocity(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rotations = Rotation.from_quat(quat_xyzw)
    velocity = np.empty((len(rotations), 3), dtype=np.float64)
    velocity[0] = (rotations[0].inv() * rotations[1]).as_rotvec() / dt
    velocity[-1] = (rotations[-2].inv() * rotations[-1]).as_rotvec() / dt
    velocity[1:-1] = (
        rotations[:-2].inv() * rotations[2:]
    ).as_rotvec() / (2.0 * dt)
    return velocity


def _ground_root_to_urdf(
    urdf_path: Path,
    root_pos: np.ndarray,
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Put the lowest left/right foot collision sphere on Z=0 per frame."""
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    if model.nq != len(JOINT_NAMES_23):
        raise ValueError(
            f"Expected a fixed-base 23-DoF URDF for grounding, got nq={model.nq}"
        )
    data = mujoco.MjData(model)
    foot_bodies = {"left_ankle_roll_link", "right_ankle_roll_link"}
    foot_geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if (
            body_name in foot_bodies
            and int(model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        ):
            foot_geom_ids.append(geom_id)
    if len(foot_geom_ids) != 8:
        raise ValueError(
            "Expected four collision spheres per foot in the current URDF, "
            f"found {len(foot_geom_ids)} total."
        )

    grounded = root_pos.copy()
    shifts = np.empty(len(joint_pos), dtype=np.float64)
    for frame_index, joints in enumerate(joint_pos):
        data.qpos[:] = joints
        mujoco.mj_forward(model, data)
        lowest_local_z = min(
            float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
            for geom_id in foot_geom_ids
        )
        target_root_z = -lowest_local_z
        shifts[frame_index] = target_root_z - root_pos[frame_index, 2]
        grounded[frame_index, 2] = target_root_z
    return grounded, shifts


def _write_preview(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "LoopMode": "Clamp",
        "FrameDuration": 1.0 / fps,
        "EnableCycleOffsetPosition": False,
        "EnableCycleOffsetRotation": False,
        "MotionWeight": 1.0,
        "Frames": frames.tolist(),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--preserve-root-xy",
        action="store_true",
        help="Keep the source's constant global XY offset instead of centering frame 0.",
    )
    parser.add_argument(
        "--ground-feet-to-urdf",
        action="store_true",
        help="Adjust only root Z so the lowest current-URDF foot collider is on the floor.",
    )
    args = parser.parse_args()

    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError(f"--fps must be positive and finite, got {args.fps}")

    csv_path = args.input_csv.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    motion = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 30:
        raise ValueError(
            f"Expected CSV shape (T, 30), got {motion.shape} from {csv_path}"
        )
    if motion.shape[0] < 3 or not np.isfinite(motion).all():
        raise ValueError("Motion must have at least three finite frames")

    urdf_joint_names, joint_limits = _load_urdf_limits(urdf_path)
    if urdf_joint_names != JOINT_NAMES_23:
        raise ValueError(
            "URDF joint order does not match the source CSV contract:\n"
            f"expected={JOINT_NAMES_23}\nactual={urdf_joint_names}"
        )

    root_pos = motion[:, :3].copy()
    root_quat_xyzw = _normalize_quaternions(motion[:, 3:7].copy())
    joint_pos = motion[:, 7:].copy()
    if not args.preserve_root_xy:
        root_pos[:, :2] -= root_pos[0, :2]

    lower_violation = np.maximum(joint_limits[:, 0] - joint_pos, 0.0)
    upper_violation = np.maximum(joint_pos - joint_limits[:, 1], 0.0)
    max_violation = float(max(lower_violation.max(), upper_violation.max()))
    if max_violation > 1.0e-6:
        flat_index = int(np.argmax(np.maximum(lower_violation, upper_violation)))
        frame_index, joint_index = np.unravel_index(flat_index, joint_pos.shape)
        raise ValueError(
            "Source motion exceeds the current URDF joint limits: "
            f"max={max_violation:.6g} rad at frame={frame_index}, "
            f"joint={JOINT_NAMES_23[joint_index]}"
        )

    grounding_shifts = np.zeros(len(root_pos), dtype=np.float64)
    if args.ground_feet_to_urdf:
        root_pos, grounding_shifts = _ground_root_to_urdf(
            urdf_path,
            root_pos,
            joint_pos,
        )

    dt = 1.0 / args.fps
    root_euler_xyz = np.unwrap(
        Rotation.from_quat(root_quat_xyzw).as_euler("XYZ", degrees=False),
        axis=0,
    )
    root_lin_vel = np.gradient(root_pos, dt, axis=0, edge_order=2)
    root_ang_vel = _angular_velocity(root_quat_xyzw, dt)
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2)
    frames = np.concatenate(
        [
            root_pos,
            root_euler_xyz,
            joint_pos,
            root_lin_vel,
            root_ang_vel,
            joint_vel,
        ],
        axis=1,
    )
    if frames.shape[1] != 58 or not np.isfinite(frames).all():
        raise RuntimeError(f"Generated invalid preview frames: {frames.shape}")

    _write_preview(output_path, frames, args.fps)
    displacement = root_pos[-1] - root_pos[0]
    print(f"Wrote {output_path}")
    print(f"  frames: {len(frames)}")
    print(f"  fps: {args.fps:.6f}")
    print(f"  duration: {(len(frames) - 1) * dt:.6f} s")
    print(f"  root displacement: {displacement.tolist()} m")
    print(f"  root height range: [{root_pos[:, 2].min():.6f}, {root_pos[:, 2].max():.6f}] m")
    print(
        "  URDF grounding shift: "
        f"[{grounding_shifts.min():.6f}, {grounding_shifts.max():.6f}] m"
    )
    print(f"  max URDF joint-limit violation: {max_violation:.3e} rad")


if __name__ == "__main__":
    main()
