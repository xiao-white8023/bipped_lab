#!/usr/bin/env python3
"""Retarget a selected LaFAN1 BVH interval to the local G1 23-DoF model."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np


GMR_ROOT = Path("/home/cjh/GMR")
G1_XML = GMR_ROOT / "assets/unitree_g1_23dof/g1_23dof_rev_1_0.xml"
G1_IK_CONFIG = (
    GMR_ROOT
    / "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_23dof.json"
)


def _smooth_qpos(previous: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None or alpha <= 0.0:
        return current
    output = (1.0 - alpha) * current + alpha * previous
    current_quat = current[3:7].copy()
    if np.dot(previous[3:7], current_quat) < 0.0:
        current_quat *= -1.0
    quat = (1.0 - alpha) * current_quat + alpha * previous[3:7]
    output[3:7] = quat / np.linalg.norm(quat)
    return output


def _ground_root_height(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    """Ground both collision feet without changing any joint angle."""
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
    parser.add_argument("--bvh-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=150)
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--smooth-alpha", type=float, default=0.08)
    args = parser.parse_args()

    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting import params
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    params.IK_CONFIG_DICT["bvh_lafan1"]["unitree_g1_23dof"] = G1_IK_CONFIG
    frames, human_height = load_bvh_file(str(args.bvh_file), format="lafan1")
    if not 0 <= args.start_frame < args.end_frame <= len(frames):
        raise ValueError(
            f"Invalid frame interval [{args.start_frame}:{args.end_frame}] "
            f"for {len(frames)} source frames"
        )
    frames = frames[args.start_frame : args.end_frame]

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="unitree_g1_23dof",
        actual_human_height=human_height,
        damping=args.damping,
        verbose=False,
    )
    retargeter.max_iter = args.max_iter

    qpos_frames: list[np.ndarray] = []
    previous = None
    for frame in frames:
        qpos = retargeter.retarget(frame)
        qpos = _smooth_qpos(previous, qpos, args.smooth_alpha)
        previous = qpos.copy()
        qpos_frames.append(qpos)

    qpos = np.asarray(qpos_frames, dtype=np.float64)
    root_pos = qpos[:, :3].copy()
    root_pos[:, :2] -= root_pos[0, :2]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]].copy()
    dof_pos = qpos[:, 7:].copy()
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    root_pos = _ground_root_height(model, root_pos, root_rot_xyzw, dof_pos)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(
            {
                "fps": float(args.fps),
                "root_pos": root_pos,
                "root_rot": root_rot_xyzw,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
                "source": {
                    "dataset": "Ubisoft LaFAN1",
                    "bvh_file": str(args.bvh_file.expanduser().resolve()),
                    "frame_slice": [args.start_frame, args.end_frame],
                },
            },
            stream,
        )
    print(f"Saved {len(qpos)} frames to {output}")


if __name__ == "__main__":
    main()
