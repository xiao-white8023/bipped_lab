#!/usr/bin/env python3
"""Sequentially visualize G1 Recovery reset states for manual inspection.

- Exactly one G1 robot.
- One reset state/file is shown at a time.
- Each state is frozen for 2 seconds by default.
- No env.step() and no sim.step().
- Terminal prints current index / total count / state_id / file name.
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
        "--reset_dir",
        type=Path,
        default=None,
        help="Optional reset-state directory override.",
    )
    parser.add_argument(
        "--display_s",
        type=float,
        default=2.0,
        help="Seconds to display each reset state. Default: 2.0.",
    )
    parser.add_argument(
        "--start_state",
        type=int,
        default=0,
        help="Zero-based first state_id to display. Default: 0.",
    )
    parser.add_argument(
        "--end_state",
        type=int,
        default=None,
        help="Optional zero-based inclusive last state_id.",
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
    description="Sequential manual QC for G1 Recovery reset states."
)
add_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.headless:
    parser.error("This visualizer requires GUI. Do not use --headless.")
if args.display_s <= 0.0:
    parser.error("--display_s must be > 0.")
if args.start_state < 0:
    parser.error("--start_state must be >= 0.")
if args.end_state is not None and args.end_state < args.start_state:
    parser.error("--end_state must be >= --start_state.")
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


def configure_reset_dir(cfg: G1RecoveryEnvCfg) -> Path | None:
    """Apply an optional reset directory override."""
    if args.reset_dir is None:
        if hasattr(cfg, "reset") and hasattr(cfg.reset, "state_dir"):
            value = getattr(cfg.reset, "state_dir")
            if value:
                return Path(str(value)).expanduser().resolve()
        return None

    reset_dir = args.reset_dir.expanduser().resolve()
    if not reset_dir.is_dir():
        raise FileNotFoundError(f"Reset directory not found: {reset_dir}")

    if not hasattr(cfg, "reset") or not hasattr(cfg.reset, "state_dir"):
        raise RuntimeError(
            "G1RecoveryEnvCfg has no reset.state_dir field; "
            "--reset_dir cannot be applied."
        )

    cfg.reset.state_dir = str(reset_dir)
    return reset_dir


def get_state_file_name(loader, state_id: int) -> str:
    """Return the reset NPZ file name using loader ordering."""
    state_files = getattr(loader, "state_files", None)
    if state_files is not None and len(state_files) > state_id:
        return Path(str(state_files[state_id])).name

    files = getattr(loader, "files", None)
    if files is not None and len(files) > state_id:
        return Path(str(files[state_id])).name

    return f"state_{state_id:04d}.npz"


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
def show_state(
    env: G1RecoveryEnv,
    state_id: int,
    current_index: int,
    total_to_show: int,
    total_dataset_states: int,
) -> bool:
    """Load one exact reset state and freeze-render it."""
    if not simulation_app.is_running():
        return False

    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    state_ids = torch.tensor([state_id], dtype=torch.long, device=env.device)

    # Use the exact deterministic reset path from the real Recovery environment.
    env.reset_to_state_ids(env_ids, state_ids)

    # Update kinematics/Fabric only. Do NOT integrate physics.
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)

    file_name = get_state_file_name(env.reset_state_loader, state_id)

    print(
        f"[{current_index}/{total_to_show}] "
        f"state_id={state_id}/{total_dataset_states - 1} | "
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

    reset_dir = configure_reset_dir(cfg)

    env = G1RecoveryEnv(cfg=cfg, headless=False)
    loader = env.reset_state_loader

    total_states = int(loader.num_states)
    if total_states <= 0:
        raise RuntimeError("Recovery reset loader contains no states.")

    start_state = int(args.start_state)
    if start_state >= total_states:
        raise IndexError(
            f"--start_state={start_state} is out of range. "
            f"Dataset has {total_states} states."
        )

    if args.end_state is None:
        end_state = total_states - 1
    else:
        end_state = min(int(args.end_state), total_states - 1)

    total_to_show = end_state - start_state + 1

    set_camera(env)

    print()
    print("=" * 100)
    print("G1 Recovery Reset Sequential Manual QC")
    print(f"Total reset files : {total_states}")
    print(f"Viewing state_ids : {start_state} .. {end_state}")
    print(f"Files this run    : {total_to_show}")
    print(f"Time per file     : {args.display_s:.2f} s")
    print("Physics           : OFF (freeze only)")
    if reset_dir is not None:
        print(f"Reset directory   : {reset_dir}")
    print("=" * 100, flush=True)

    for current_index, state_id in enumerate(
        range(start_state, end_state + 1),
        start=1,
    ):
        keep_running = show_state(
            env=env,
            state_id=state_id,
            current_index=current_index,
            total_to_show=total_to_show,
            total_dataset_states=total_states,
        )

        if not keep_running:
            print("Viewport closed. Stopping manual QC.", flush=True)
            return

    print()
    print("=" * 100)
    print(f"Finished: displayed {total_to_show}/{total_to_show} reset files.")
    print("The final reset state remains visible until the window is closed.")
    print("=" * 100, flush=True)

    while simulation_app.is_running():
        env.sim.render()
        time.sleep(1.0 / float(args.render_fps))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
