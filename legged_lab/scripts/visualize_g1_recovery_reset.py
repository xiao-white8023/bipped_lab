#!/usr/bin/env python3
"""Visualize and quality-check physical G1 Recovery trajectory-frame resets.

This tool always uses :class:`G1RecoveryEnv` and its shared reset path. Freeze
mode performs no physics integration. Hold mode advances PhysX directly while
holding the exact sampled reset joint positions; it never executes a policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RSL_RL_PATH = REPO_ROOT / "rsl_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num_envs",
        type=int,
        default=9,
        help="Number of G1 environments. Ignored when fixed IDs are supplied.",
    )
    parser.add_argument(
        "--mode",
        choices=("freeze", "hold"),
        default="freeze",
        help="Freeze exact reset poses or run low-level PhysX hold QC.",
    )
    parser.add_argument(
        "--motion_ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional deterministic zero-based trajectory IDs.",
    )
    parser.add_argument(
        "--frame_ids",
        nargs="*",
        type=int,
        default=None,
        help="Integer frame IDs paired one-to-one with --motion_ids.",
    )
    parser.add_argument(
        "--resample_interval_s",
        type=float,
        default=3.0,
        help="Wall-clock interval between random reset batches.",
    )
    parser.add_argument(
        "--hold_s",
        type=float,
        default=0.5,
        help="Physics duration per batch in hold mode.",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        help="Continuously show batches (the default).",
    )
    run_mode.add_argument(
        "--once",
        dest="loop",
        action="store_false",
        help="Process exactly one reset batch and exit.",
    )
    parser.set_defaults(loop=True)
    parser.add_argument(
        "--camera_height",
        type=float,
        default=None,
        help="Optional camera height above the environment-grid center.",
    )
    parser.add_argument(
        "--print_metadata",
        dest="print_metadata",
        action="store_true",
        default=True,
        help="Print reset source metadata (enabled by default).",
    )
    parser.add_argument(
        "--no_print_metadata",
        dest="print_metadata",
        action="store_false",
        help="Suppress the reset source metadata table.",
    )


# AppLauncher must precede all remaining IsaacLab/Omniverse imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Visualize exact G1 Recovery trajectory-frame resets."
)
_add_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.headless:
    parser.error("G1 Recovery reset visualization requires GUI; do not use --headless.")
if args.num_envs <= 0:
    parser.error("--num_envs must be positive.")
if (args.motion_ids is None) != (args.frame_ids is None):
    parser.error("--motion_ids and --frame_ids must be supplied together.")
if args.motion_ids is not None and not args.motion_ids:
    parser.error("Fixed trajectory-frame IDs require at least one pair.")
if args.motion_ids is not None and len(args.motion_ids) != len(args.frame_ids):
    parser.error("--motion_ids and --frame_ids must have equal lengths.")
if args.motion_ids is not None and any(value < 0 for value in args.motion_ids):
    parser.error("--motion_ids must contain non-negative zero-based IDs.")
if args.frame_ids is not None and any(value < 0 for value in args.frame_ids):
    parser.error("--frame_ids must contain non-negative zero-based IDs.")
if args.resample_interval_s <= 0.0:
    parser.error("--resample_interval_s must be positive.")
if args.hold_s <= 0.0:
    parser.error("--hold_s must be positive.")
if args.camera_height is not None and args.camera_height <= 0.0:
    parser.error("--camera_height must be positive.")
if args.motion_ids is not None:
    args.num_envs = len(args.motion_ids)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch

from legged_lab.envs.g1.g1_recovery_cfg import G1RecoveryEnvCfg
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv


RENDER_PERIOD_S = 1.0 / 30.0
HOLD_RENDER_DECIMATION = 3


def _set_grid_camera(env: G1RecoveryEnv) -> None:
    origins = env.scene.env_origins.detach().cpu()
    minimum = origins.min(dim=0).values
    maximum = origins.max(dim=0).values
    center = 0.5 * (minimum + maximum)
    span_x = float(maximum[0] - minimum[0])
    span_y = float(maximum[1] - minimum[1])
    grid_extent = max(span_x, span_y, float(env.cfg.scene.env_spacing))
    diagonal_offset = max(3.5, 0.85 * grid_extent + 2.5)
    auto_height = max(2.8, 0.65 * grid_extent + 2.0)
    camera_height = (
        auto_height if args.camera_height is None else args.camera_height
    )
    eye = (
        float(center[0]) + diagonal_offset,
        float(center[1]) + diagonal_offset,
        float(center[2]) + camera_height,
    )
    target = (float(center[0]), float(center[1]), float(center[2]) + 0.5)
    env.sim.set_camera_view(eye=eye, target=target)
    print(
        "Camera covers environment grid: "
        f"center=({target[0]:.2f}, {target[1]:.2f}), "
        f"span=({span_x:.2f}, {span_y:.2f}) m, eye={eye}."
    )


def _print_reset_batch(
    env: G1RecoveryEnv,
    motion_ids: torch.Tensor,
    frame_ids: torch.Tensor,
) -> None:
    motion_id_list = motion_ids.detach().cpu().tolist()
    frame_id_list = frame_ids.detach().cpu().tolist()
    print("\n" + "=" * 174)
    print("Recovery Reset Batch")
    print("motion IDs:", motion_id_list)
    print("frame IDs :", frame_id_list)
    if args.print_metadata:
        sample = env.reset_motion_loader.get_frame_batch(motion_ids, frame_ids)
        root_index = env.reset_motion_loader.root_body_index
        _, torso_up_z = env._compute_torso_upright()
        root_lin_speed = torch.linalg.vector_norm(
            sample["body_lin_vel_w"][:, root_index], dim=-1
        )
        root_ang_speed = torch.linalg.vector_norm(
            sample["body_ang_vel_w"][:, root_index], dim=-1
        )
        mean_joint_speed = torch.mean(torch.abs(sample["joint_vel"]), dim=-1)
        header = (
            f"{'env':>3} {'motion_id':>9} {'frame_id':>8} {'source_file':<38} "
            f"{'pelvis_height':>14} {'torso_up_z':>11} "
            f"{'root_lin_speed':>14} {'root_ang_speed':>14} "
            f"{'mean_joint_speed':>16}"
        )
        print(header)
        print("-" * len(header))
        for env_index, (motion_id, frame_id) in enumerate(
            zip(motion_id_list, frame_id_list)
        ):
            source = env.reset_motion_loader.clips[motion_id].path.name
            if len(source) > 38:
                source = source[:35] + "..."
            print(
                f"{env_index:>3} {motion_id:>9} {frame_id:>8} {source:<38} "
                f"{float(sample['body_pos_w'][env_index, root_index, 2]):>14.3f} "
                f"{float(torso_up_z[env_index]):>11.3f} "
                f"{float(root_lin_speed[env_index]):>14.3f} "
                f"{float(root_ang_speed[env_index]):>14.3f} "
                f"{float(mean_joint_speed[env_index]):>16.3f}"
            )
    print("=" * 174, flush=True)


def _reset_batch(
    env: G1RecoveryEnv,
    env_ids: torch.Tensor,
    fixed_motion_ids: torch.Tensor | None,
    fixed_frame_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fixed_motion_ids is None:
        env.reset(env_ids)
    else:
        env.reset_to_motion_frames(env_ids, fixed_motion_ids, fixed_frame_ids)
    motion_ids = env.last_reset_motion_ids[env_ids].clone()
    frame_ids = env.last_reset_frame_ids[env_ids].clone()
    _print_reset_batch(env, motion_ids, frame_ids)
    return motion_ids, frame_ids


def _quaternion_angle(initial_quat, final_quat):
    initial_quat = torch.nn.functional.normalize(initial_quat, dim=-1)
    final_quat = torch.nn.functional.normalize(final_quat, dim=-1)
    dot = torch.abs(torch.sum(initial_quat * final_quat, dim=-1))
    return 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))


def _run_hold_qc(
    env: G1RecoveryEnv,
    env_ids: torch.Tensor,
    motion_ids: torch.Tensor,
    frame_ids: torch.Tensor,
) -> None:
    initial_root_pos = env.robot.data.root_link_pos_w[env_ids].clone()
    initial_root_quat = env.robot.data.root_link_quat_w[env_ids].clone()
    max_root_lin_speed = torch.linalg.vector_norm(
        env.robot.data.root_link_lin_vel_w[env_ids], dim=-1
    )
    max_root_ang_speed = torch.linalg.vector_norm(
        env.robot.data.root_link_ang_vel_w[env_ids], dim=-1
    )

    hold_steps = max(1, round(args.hold_s / env.physics_dt))
    zero_velocity_target = torch.zeros_like(env.last_reset_joint_pos[env_ids])
    for step_index in range(hold_steps):
        if not simulation_app.is_running():
            return
        env.robot.set_joint_position_target(
            env.last_reset_joint_pos[env_ids], env_ids=env_ids
        )
        env.robot.set_joint_velocity_target(
            zero_velocity_target, env_ids=env_ids
        )
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        max_root_lin_speed = torch.maximum(
            max_root_lin_speed,
            torch.linalg.vector_norm(
                env.robot.data.root_link_lin_vel_w[env_ids], dim=-1
            ),
        )
        max_root_ang_speed = torch.maximum(
            max_root_ang_speed,
            torch.linalg.vector_norm(
                env.robot.data.root_link_ang_vel_w[env_ids], dim=-1
            ),
        )
        if step_index % HOLD_RENDER_DECIMATION == 0:
            env.sim.render()

    final_root_pos = env.robot.data.root_link_pos_w[env_ids]
    final_root_quat = env.robot.data.root_link_quat_w[env_ids]
    root_displacement = torch.linalg.vector_norm(
        final_root_pos - initial_root_pos, dim=-1
    )
    root_angle_change = _quaternion_angle(initial_root_quat, final_root_quat)

    physical_tensors = (
        final_root_pos,
        final_root_quat,
        env.robot.data.root_link_lin_vel_w[env_ids],
        env.robot.data.root_link_ang_vel_w[env_ids],
        env.robot.data.joint_pos[env_ids],
        env.robot.data.joint_vel[env_ids],
    )
    if not all(torch.isfinite(tensor).all() for tensor in physical_tensors):
        raise RuntimeError("Hold QC produced NaN or Inf physical state values.")

    print("\n" + "=" * 128)
    print("Recovery Reset Hold QC (no automatic good/bad threshold)")
    print(
        f"{'env':>3} {'motion_id':>9} {'frame_id':>8} {'root_disp_m':>14} "
        f"{'root_angle_change_rad':>22} "
        f"{'max_root_lin_speed':>20} {'max_root_ang_speed':>20}"
    )
    motion_id_list = motion_ids.detach().cpu().tolist()
    frame_id_list = frame_ids.detach().cpu().tolist()
    for row_index, (motion_id, frame_id) in enumerate(
        zip(motion_id_list, frame_id_list)
    ):
        print(
            f"{int(env_ids[row_index]):>3} {motion_id:>9} {frame_id:>8} "
            f"{float(root_displacement[row_index]):>14.6f} "
            f"{float(root_angle_change[row_index]):>22.6f} "
            f"{float(max_root_lin_speed[row_index]):>20.6f} "
            f"{float(max_root_ang_speed[row_index]):>20.6f}"
        )
    print("=" * 128, flush=True)


def _run_freeze(
    env: G1RecoveryEnv,
    env_ids: torch.Tensor,
    fixed_motion_ids: torch.Tensor | None,
    fixed_frame_ids: torch.Tensor | None,
) -> None:
    _reset_batch(env, env_ids, fixed_motion_ids, fixed_frame_ids)
    env.sim.render()
    if not args.loop:
        return

    last_random_reset_time = time.perf_counter()
    while simulation_app.is_running():
        env.sim.render()
        if fixed_motion_ids is None:
            now = time.perf_counter()
            if now - last_random_reset_time >= args.resample_interval_s:
                _reset_batch(
                    env,
                    env_ids,
                    fixed_motion_ids=None,
                    fixed_frame_ids=None,
                )
                last_random_reset_time = now
        time.sleep(RENDER_PERIOD_S)


def _run_hold(
    env: G1RecoveryEnv,
    env_ids: torch.Tensor,
    fixed_motion_ids: torch.Tensor | None,
    fixed_frame_ids: torch.Tensor | None,
) -> None:
    while simulation_app.is_running():
        batch_start_time = time.perf_counter()
        motion_ids, frame_ids = _reset_batch(
            env, env_ids, fixed_motion_ids, fixed_frame_ids
        )
        _run_hold_qc(env, env_ids, motion_ids, frame_ids)
        env.sim.render()
        if not args.loop:
            return

        if fixed_motion_ids is None:
            while simulation_app.is_running():
                elapsed = time.perf_counter() - batch_start_time
                if elapsed >= args.resample_interval_s:
                    break
                env.sim.render()
                time.sleep(min(RENDER_PERIOD_S, args.resample_interval_s - elapsed))


def main() -> None:
    cfg = G1RecoveryEnvCfg()
    cfg.device = args.device
    cfg.scene.num_envs = args.num_envs
    # The training config's flat terrain generator deliberately distributes
    # origins across large terrain tiles. A plane preserves the same flat-ground
    # reset semantics while arranging GUI QC environments on env_spacing grid.
    cfg.scene.terrain_type = "plane"
    cfg.scene.terrain_generator = None
    env = G1RecoveryEnv(cfg=cfg, headless=False)
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    fixed_motion_ids = None
    fixed_frame_ids = None
    if args.motion_ids is not None:
        fixed_motion_ids = torch.tensor(
            args.motion_ids, dtype=torch.long, device=env.device
        )
        fixed_frame_ids = torch.tensor(
            args.frame_ids, dtype=torch.long, device=env.device
        )

    _set_grid_camera(env)
    print(
        f"G1 Recovery reset visualizer: mode={args.mode}, "
        f"num_envs={env.num_envs}, loop={args.loop}."
    )
    if args.mode == "freeze":
        _run_freeze(env, env_ids, fixed_motion_ids, fixed_frame_ids)
    else:
        _run_hold(env, env_ids, fixed_motion_ids, fixed_frame_ids)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=True)
