#!/usr/bin/env python3
"""Compare Recovery AMP features before and after exact IsaacLab state playback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RSL_RL_PATH = REPO_ROOT / "rsl_rl"
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))

DEFAULT_MOTION_DIR = (
    REPO_ROOT
    / "legged_lab"
    / "envs"
    / "g1"
    / "datasets"
    / "npz"
    / "recovery_candidates_v3"
)


# AppLauncher must precede all other IsaacLab/Omniverse imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Check expert/policy parity for the shared 195D Recovery AMP feature builder."
)
parser.add_argument(
    "--motion_dir",
    type=Path,
    default=DEFAULT_MOTION_DIR,
    help="Directory containing final Stage-2 Recovery NPZ clips.",
)
parser.add_argument(
    "--motion_id",
    type=int,
    default=0,
    help="Zero-based motion index after filename sorting.",
)
parser.add_argument(
    "--frame_id",
    type=int,
    default=None,
    help="Non-boundary frame index; defaults to the selected clip midpoint.",
)
parser.add_argument(
    "--atol",
    type=float,
    default=1e-5,
    help="Maximum allowed absolute error for every feature block.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.motion_id < 0:
    parser.error("--motion_id must be non-negative.")
if args.atol <= 0.0:
    parser.error("--atol must be positive.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from legged_lab.assets.g1.g1_29 import G1_23CFG
from rsl_rl.utils import RecoveryAMPFeatureBuilder, RecoveryNPZMotionLoader
from rsl_rl.utils.recovery_amp_features import (
    BODY_ANG_VEL_SLICE,
    BODY_LIN_VEL_SLICE,
    BODY_ORI_6D_SLICE,
    BODY_POS_SLICE,
)


@configclass
class ParitySceneCfg(InteractiveSceneCfg):
    robot = G1_23CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


FEATURE_BLOCKS = {
    "body_pos_b": BODY_POS_SLICE,
    "body_ori_6d": BODY_ORI_6D_SLICE,
    "body_lin_vel_b": BODY_LIN_VEL_SLICE,
    "body_ang_vel_b": BODY_ANG_VEL_SLICE,
}


def _progress(message: str) -> None:
    print(f"[Recovery AMP parity] {message}", flush=True)


def _assert_exact_name_order(label, robot_names, npz_names) -> None:
    robot_names = tuple(robot_names)
    npz_names = tuple(npz_names)
    if robot_names != npz_names:
        raise RuntimeError(
            f"{label} order mismatch.\nNPZ:   {npz_names}\nRobot: {robot_names}"
        )


def _batched_frame_state(loader, motion_id: int, frame_id: int):
    clip = loader.clips[motion_id]
    state = loader.get_state_at_time(motion_id, frame_id / clip.fps)
    return {field: value.unsqueeze(0) for field, value in state.items()}


def run_parity_test() -> dict[str, dict[str, float]]:
    _progress(f"Loading Recovery NPZ motions from {args.motion_dir}.")
    loader = RecoveryNPZMotionLoader(
        motion_dir=args.motion_dir,
        device=args.device,
    )
    if args.motion_id >= loader.num_motions:
        raise ValueError(
            f"--motion_id={args.motion_id} but only {loader.num_motions} motions were loaded."
        )
    clip = loader.clips[args.motion_id]
    frame_id = clip.num_frames // 2 if args.frame_id is None else args.frame_id
    if frame_id <= 0 or frame_id >= clip.num_frames - 1:
        raise ValueError(
            f"frame_id must be non-boundary for {clip.name}: "
            f"1 <= frame_id <= {clip.num_frames - 2}, got {frame_id}."
        )

    _progress(
        f"Creating num_envs=1 scene on {args.device} for motion_id={args.motion_id}, "
        f"frame_id={frame_id}."
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=clip.frame_duration,
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ParitySceneCfg(num_envs=1, env_spacing=2.0))
    _progress("Resetting simulation and scene.")
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    _progress("Scene reset complete; validating exact joint/body name order.")

    _assert_exact_name_order("Joint", robot.joint_names, loader.joint_names)
    _assert_exact_name_order("Body", robot.body_names, loader.body_names)

    expert_state = _batched_frame_state(loader, args.motion_id, frame_id)
    root_index = loader.root_body_index
    root_pose = torch.cat(
        (
            expert_state["body_pos_w"][:, root_index],
            expert_state["body_quat_w"][:, root_index],
        ),
        dim=-1,
    )
    root_velocity = torch.cat(
        (
            expert_state["body_lin_vel_w"][:, root_index],
            expert_state["body_ang_vel_w"][:, root_index],
        ),
        dim=-1,
    )
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_root_link_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(
        expert_state["joint_pos"], expert_state["joint_vel"]
    )

    # Exact state playback only: refresh FK/data buffers without a physics step.
    _progress("State written; refreshing FK and articulation data buffers.")
    sim.forward()
    scene.update(clip.frame_duration)
    _progress("FK refresh complete; building expert and policy features.")

    builder = RecoveryAMPFeatureBuilder(loader.body_names)
    expert_feature = builder.build_from_state(expert_state)
    policy_feature = builder.build(
        robot.data.body_link_pos_w,
        robot.data.body_link_quat_w,
        robot.data.body_link_lin_vel_w,
        robot.data.body_link_ang_vel_w,
    )
    if expert_feature.shape != (1, 195) or policy_feature.shape != (1, 195):
        raise RuntimeError(
            f"Unexpected feature shapes: expert={expert_feature.shape}, "
            f"policy={policy_feature.shape}."
        )

    print(
        f"Motion: {clip.name}\nFrame: {frame_id}/{clip.num_frames - 1}\n"
        f"Device: {robot.device}\nFeature dim: {builder.feature_dim}",
        flush=True,
    )
    results = {}
    failed_blocks = []
    for block_name, block_slice in FEATURE_BLOCKS.items():
        absolute_error = torch.abs(
            expert_feature[:, block_slice] - policy_feature[:, block_slice]
        )
        max_error = float(absolute_error.max().item())
        mean_error = float(absolute_error.mean().item())
        results[block_name] = {
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
        }
        print(
            f"{block_name:18s} max_abs_error={max_error:.9e} "
            f"mean_abs_error={mean_error:.9e}",
            flush=True,
        )
        if max_error > args.atol:
            failed_blocks.append((block_name, max_error))

    if failed_blocks:
        raise RuntimeError(
            f"Recovery AMP parity failed with atol={args.atol}: {failed_blocks}."
        )
    print(f"Recovery AMP expert/policy parity passed with atol={args.atol}.", flush=True)
    return results


if __name__ == "__main__":
    try:
        run_parity_test()
    finally:
        # This is a one-shot headless diagnostic with no writers or persistent
        # simulator state, so immediate cleanup avoids slow Replicator teardown.
        simulation_app.close(skip_cleanup=True)
