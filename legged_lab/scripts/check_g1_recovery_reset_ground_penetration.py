#!/usr/bin/env python3
"""Sequentially visualize exact G1 Recovery trajectory frames for manual QC.

- Exactly one G1 robot.
- One integer trajectory frame is shown at a time.
- Each frame is frozen for 2 seconds by default.
- No env.step() and no sim.step().
- Terminal prints current index / total count / motion/frame ID / file name.
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--motion_dir",
        type=Path,
        default=None,
        help="Optional Recovery trajectory directory override.",
    )
    parser.add_argument(
        "--display_s",
        type=float,
        default=2.0,
        help="Seconds to display each trajectory frame. Default: 2.0.",
    )
    parser.add_argument(
        "--motion_id",
        type=int,
        default=0,
        help="Zero-based motion ID after filename sorting. Default: 0.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Zero-based first integer frame to display. Default: 0.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Optional zero-based inclusive last frame.",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=2.6,
        help="Camera horizontal distance from robot. Default: 2.6 m.",
    )
    parser.add_argument(
        "--camera_height",
        type=float,
        default=1.5,
        help="Camera height above environment origin. Default: 1.5 m.",
    )
    parser.add_argument(
        "--camera_side",
        choices=("front", "side", "diag"),
        default="diag",
        help="Initial camera direction.",
    )
    parser.add_argument(
        "--render_fps",
        type=float,
        default=30.0,
        help="Viewport refresh rate while a reset pose is frozen.",
    )


# AppLauncher must be created before importing the rest of IsaacLab/Omniverse.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Sequential manual QC for exact G1 Recovery trajectory frames."
)
add_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.headless:
    parser.error("This visualizer requires GUI. Do not use --headless.")
if args.display_s <= 0.0:
    parser.error("--display_s must be > 0.")
if args.motion_id < 0:
    parser.error("--motion_id must be >= 0.")
if args.start_frame < 0:
    parser.error("--start_frame must be >= 0.")
if args.end_frame is not None and args.end_frame < args.start_frame:
    parser.error("--end_frame must be >= --start_frame.")
if args.camera_distance <= 0.0:
    parser.error("--camera_distance must be > 0.")
if args.camera_height <= 0.0:
    parser.error("--camera_height must be > 0.")
if args.render_fps <= 0.0:
    parser.error("--render_fps must be > 0.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch

from legged_lab.envs.g1.g1_recovery_cfg import G1RecoveryEnvCfg
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv


def configure_motion_dir(cfg: G1RecoveryEnvCfg) -> Path | None:
    """Apply an optional trajectory directory override."""
    if args.motion_dir is None:
        if hasattr(cfg, "reset") and hasattr(cfg.reset, "motion_dir"):
            value = getattr(cfg.reset, "motion_dir")
            if value:
                return Path(str(value)).expanduser().resolve()
        return None

    motion_dir = args.motion_dir.expanduser().resolve()
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")

    if not hasattr(cfg, "reset") or not hasattr(cfg.reset, "motion_dir"):
        raise RuntimeError(
            "G1RecoveryEnvCfg has no reset.motion_dir field; "
            "--motion_dir cannot be applied."
        )

    cfg.reset.motion_dir = str(motion_dir)
    return motion_dir


def get_motion_file_name(loader, motion_id: int) -> str:
    """Return the trajectory NPZ file name using loader ordering."""
    return loader.clips[motion_id].path.name


def set_camera(env: G1RecoveryEnv) -> None:
    """Place the camera around the single environment."""
    origin = env.scene.env_origins[0].detach().cpu()

    target = (
        float(origin[0]),
        float(origin[1]),
        float(origin[2]) + 0.45,
    )

    distance = float(args.camera_distance)
    height = float(origin[2]) + float(args.camera_height)

    if args.camera_side == "front":
        eye = (target[0] + distance, target[1], height)
    elif args.camera_side == "side":
        eye = (target[0], target[1] + distance, height)
    else:
        offset = distance / (2.0 ** 0.5)
        eye = (target[0] + offset, target[1] + offset, height)

    env.sim.set_camera_view(eye=eye, target=target)
    print(f"Camera eye={eye}, target={target}", flush=True)


@torch.no_grad()
def show_frame(
    env: G1RecoveryEnv,
    motion_id: int,
    frame_id: int,
    current_index: int,
    total_to_show: int,
    total_motion_frames: int,
) -> bool:
    """Load one exact trajectory frame and freeze-render it."""
    if not simulation_app.is_running():
        return False

    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    motion_ids = torch.tensor([motion_id], dtype=torch.long, device=env.device)
    frame_ids = torch.tensor([frame_id], dtype=torch.long, device=env.device)

    # Use the exact deterministic reset path from the real Recovery environment.
    env.reset_to_motion_frames(env_ids, motion_ids, frame_ids)

    # Update kinematics/Fabric only. Do NOT integrate physics.
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)

    file_name = get_motion_file_name(env.reset_motion_loader, motion_id)

    print(
        f"[{current_index}/{total_to_show}] "
        f"motion_id={motion_id}/{env.reset_motion_loader.num_motions - 1} | "
        f"frame_id={frame_id}/{total_motion_frames - 1} | "
        f"file={file_name}",
        flush=True,
    )

    env.sim.render()

    frame_dt = 1.0 / float(args.render_fps)
    end_time = time.perf_counter() + float(args.display_s)

    while simulation_app.is_running():
        now = time.perf_counter()
        if now >= end_time:
            break

        frame_start = now
        env.sim.render()

        sleep_time = frame_dt - (time.perf_counter() - frame_start)
        if sleep_time > 0.0:
            remaining = max(0.0, end_time - time.perf_counter())
            time.sleep(min(sleep_time, remaining))

    return simulation_app.is_running()


def main() -> None:
    cfg = G1RecoveryEnvCfg()
    cfg.device = args.device

    # Exactly one robot/environment.
    cfg.scene.num_envs = 1

    # Use simple flat ground for manual reset QC.
    cfg.scene.terrain_type = "plane"
    cfg.scene.terrain_generator = None

    motion_dir = configure_motion_dir(cfg)

    env = G1RecoveryEnv(cfg=cfg, headless=False)
    loader = env.reset_motion_loader

    if args.motion_id >= loader.num_motions:
        raise IndexError(
            f"--motion_id={args.motion_id} is out of range; "
            f"dataset has {loader.num_motions} motions."
        )
    clip = loader.clips[args.motion_id]
    total_frames = int(clip.num_frames)

    start_frame = int(args.start_frame)
    if start_frame >= total_frames:
        raise IndexError(
            f"--start_frame={start_frame} is out of range. "
            f"Motion {clip.name!r} has {total_frames} frames."
        )

    if args.end_frame is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(int(args.end_frame), total_frames - 1)

    total_to_show = end_frame - start_frame + 1

    set_camera(env)

    print()
    print("=" * 100)
    print("G1 Recovery Trajectory-Frame Sequential Manual QC")
    print(f"Motion            : {args.motion_id} | {clip.path.name}")
    print(f"Total frames      : {total_frames}")
    print(f"Viewing frame_ids : {start_frame} .. {end_frame}")
    print(f"Frames this run   : {total_to_show}")
    print(f"Time per frame    : {args.display_s:.2f} s")
    print("Physics           : OFF (freeze only)")
    if motion_dir is not None:
        print(f"Motion directory  : {motion_dir}")
    print("=" * 100, flush=True)

    for current_index, frame_id in enumerate(
        range(start_frame, end_frame + 1),
        start=1,
    ):
        keep_running = show_frame(
            env=env,
            motion_id=args.motion_id,
            frame_id=frame_id,
            current_index=current_index,
            total_to_show=total_to_show,
            total_motion_frames=total_frames,
        )

        if not keep_running:
            print("Viewport closed. Stopping manual QC.", flush=True)
            return

    print()
    print("=" * 100)
    print(f"Finished: displayed {total_to_show}/{total_to_show} trajectory frames.")
    print("The final trajectory frame remains visible until the window is closed.")
    print("=" * 100, flush=True)

    while simulation_app.is_running():
        env.sim.render()
        time.sleep(1.0 / float(args.render_fps))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
