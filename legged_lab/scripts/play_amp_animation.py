# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.

"""Replay a G1 visualization trajectory and optionally save AMP expert frames.

By default, playback uses the exact ``FrameDuration`` stored in the input
visualization file. Passing ``--fps`` explicitly performs temporal resampling at
that target rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure the repository-local rsl_rl package has priority.
CURRENT_DIR = Path(__file__).resolve().parent
LOCAL_RSL_RL_PATH = (CURRENT_DIR / "../../rsl_rl").resolve()
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))

# Isaac/Omniverse modules must not be imported before AppLauncher starts.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay a G1 motion and generate AMP expert data.")
parser.add_argument("--task", type=str, required=True, help="Registered G1 task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Saving requires 1.")
parser.add_argument("--seed", type=int, default=None, help="Optional scene seed override.")
parser.add_argument("--save_path", type=str, default=None, help="Output AMP expert JSON/TXT path.")
parser.add_argument(
    "--fps",
    type=float,
    default=None,
    help="Optional output/playback FPS. Omit to preserve the source FrameDuration.",
)
parser.add_argument(
    "--motion_weight",
    type=float,
    default=0.5,
    help="MotionWeight written to the generated AMP expert file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is not None and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Safe to import simulation-dependent project modules after launching the app.
from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402


def _set_if_present(obj, attribute: str, value) -> None:
    if obj is not None and hasattr(obj, attribute):
        setattr(obj, attribute, value)


def _disable_generation_randomization(env_cfg) -> None:
    """Disable common randomization/noise sources without assuming every field exists."""
    if hasattr(env_cfg, "noise"):
        _set_if_present(env_cfg.noise, "add_noise", False)

    scene = getattr(env_cfg, "scene", None)
    if scene is not None:
        height_scanner = getattr(scene, "height_scanner", None)
        _set_if_present(height_scanner, "enable_height_scan", False)

        lidar = getattr(scene, "lidar", None)
        _set_if_present(lidar, "enable_lidar", False)

        depth_camera = getattr(scene, "depth_camera", None)
        _set_if_present(depth_camera, "enable_depth_camera", False)

        camera = getattr(scene, "camera", None)
        _set_if_present(camera, "add_camera", False)

        left_feet_ray_caster = getattr(scene, "left_feet_ray_caster", None)
        _set_if_present(left_feet_ray_caster, "add_left_feet_ray_caster", False)

        right_feet_ray_caster = getattr(scene, "right_feet_ray_caster", None)
        _set_if_present(right_feet_ray_caster, "add_right_feet_ray_caster", False)

    domain_rand = getattr(env_cfg, "domain_rand", None)
    events = getattr(domain_rand, "events", None)
    if events is not None:
        for event_name in (
            "push_robot",
            "physics_material",
            "add_base_mass",
            "reset_base",
            "reset_robot_joints",
        ):
            _set_if_present(events, event_name, None)

    action_delay = getattr(domain_rand, "action_delay", None)
    _set_if_present(action_delay, "enable", False)


def _write_motion_file(
    save_path: str,
    frames: np.ndarray,
    frame_duration: float,
    motion_weight: float,
) -> None:
    output_path = Path(save_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim != 2 or frames.shape[1] != 58:
        raise ValueError(f"Expected AMP expert data with shape (T, 58), got {frames.shape}.")
    if not np.isfinite(frames).all():
        raise ValueError("Generated AMP expert frames contain NaN or Inf values.")

    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": float(frame_duration),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": float(motion_weight),
        "Frames": frames.tolist(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(
        f"Saved AMP expert motion to '{output_path}': "
        f"frames={frames.shape[0]}, dim={frames.shape[1]}, "
        f"FrameDuration={frame_duration:.9f}, fps={1.0 / frame_duration:.6f}."
    )


def play_amp_animation() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    _disable_generation_randomization(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    if hasattr(env_cfg, "commands"):
        _set_if_present(env_cfg.commands, "debug_vis", False)

    if args_cli.seed is not None:
        scene_seed = int(args_cli.seed)
    else:
        scene_seed = int(agent_cfg.seed)
    _set_if_present(env_cfg.scene, "seed", scene_seed)

    if args_cli.save_path and int(args_cli.num_envs) != 1:
        raise ValueError("Generating one AMP expert trajectory requires --num_envs=1.")

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    source_num_frames = int(env.amp_loader_display.trajectory_num_frames[0])
    source_frame_duration = float(env.amp_loader_display.trajectory_frame_durations[0])
    source_duration = float(env.amp_loader_display.trajectory_lens[0])

    if source_num_frames < 2:
        raise ValueError("The source motion must contain at least two frames.")

    if args_cli.fps is None:
        output_frame_duration = source_frame_duration
        sample_times = np.arange(source_num_frames, dtype=np.float64) * source_frame_duration
    else:
        if not np.isfinite(args_cli.fps) or args_cli.fps <= 0.0:
            raise ValueError(f"--fps must be positive and finite, got {args_cli.fps}.")
        output_frame_duration = 1.0 / float(args_cli.fps)
        # Include t=0 and the final source time without sampling beyond it.
        output_num_frames = int(np.floor(source_duration / output_frame_duration + 1e-9)) + 1
        sample_times = np.arange(output_num_frames, dtype=np.float64) * output_frame_duration
        sample_times = np.minimum(sample_times, source_duration)

    print(
        "Starting G1 motion playback: "
        f"source_frames={source_num_frames}, source_fps={1.0 / source_frame_duration:.6f}, "
        f"output_frames={len(sample_times)}, output_fps={1.0 / output_frame_duration:.6f}."
    )

    all_frames: list[np.ndarray] = []
    for frame_index, sample_time in enumerate(sample_times):
        if not simulation_app.is_running():
            print(f"Simulation stopped after {frame_index} frames.")
            break

        frame = env.visualize_motion(float(sample_time))

        if args_cli.save_path:
            frame_np = frame.detach().cpu().numpy()
            if frame_np.shape != (1, 58):
                raise ValueError(
                    "env.visualize_motion() must return shape (1, 58) for full G1 23-DoF AMP, "
                    f"but returned {frame_np.shape}."
                )
            all_frames.append(frame_np[0].copy())

    if args_cli.save_path:
        if not all_frames:
            raise RuntimeError("No AMP expert frames were generated.")
        _write_motion_file(
            save_path=args_cli.save_path,
            frames=np.stack(all_frames, axis=0),
            frame_duration=output_frame_duration,
            motion_weight=args_cli.motion_weight,
        )


if __name__ == "__main__":
    try:
        play_amp_animation()
    finally:
        simulation_app.close()
