# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.

"""Replay a 58D G1 visualization trajectory and save AMP expert frames.

By default, playback uses the exact `FrameDuration` stored in the input
visualization file. Passing `--fps` explicitly performs temporal resampling at
that target rate.

During playback, the terminal displays:
- output frame index
- corresponding source frame index
- current playback time
- total motion duration

Locomotion export preserves the existing 58D full-AMP layout. Recovery export
appends ``robot.data.projected_gravity_b`` after IsaacLab replay, producing a
61D full-AMP frame. The gravity is deliberately not reconstructed from the
visualization Euler angles so expert and policy Recovery AMP use the exact same
IsaacLab definition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# Ensure the repository-local rsl_rl package has priority.
CURRENT_DIR = Path(__file__).resolve().parent
LOCAL_RSL_RL_PATH = (CURRENT_DIR / "../../rsl_rl").resolve()
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))


# Isaac/Omniverse modules must not be imported before AppLauncher starts.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Replay a G1 motion and generate AMP expert data."
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Registered G1 task name.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments. Saving requires 1.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional scene seed override.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Output AMP expert JSON/TXT path.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help=(
        "Optional 58D visualization motion to replay. When provided, this "
        "temporarily overrides env_cfg.amp_motion_files_display."
    ),
)
parser.add_argument(
    "--amp_profile",
    choices=("locomotion", "recovery"),
    default="locomotion",
    help="AMP export layout: locomotion=58D, recovery=61D with projected gravity.",
)
parser.add_argument(
    "--start_time",
    type=float,
    default=None,
    help="Optional crop start time in seconds (default: 0).",
)
parser.add_argument(
    "--end_time",
    type=float,
    default=None,
    help="Optional crop end time in seconds (default: source duration).",
)
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
parser.add_argument(
    "--clip_ranges",
    type=str,
    default=None,
    help=(
        "Optional comma-separated time ranges to crop, e.g. "
        "'12.9-18.65,28.5-34.4,40.1-44.8'. "
        "Each range is saved as an independent AMP motion file."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is not None and "sensor" in args_cli.task:
    args_cli.enable_cameras = True
if args_cli.save_path is not None and not args_cli.headless:
    print(
        "[play_amp_animation] WARNING: exporting without --headless enables "
        "the GUI and per-frame rendering, which is substantially slower.",
        flush=True,
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Safe to import simulation-dependent project modules after launching the app.
import torch  # noqa: E402

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
        _set_if_present(
            left_feet_ray_caster,
            "add_left_feet_ray_caster",
            False,
        )

        right_feet_ray_caster = getattr(scene, "right_feet_ray_caster", None)
        _set_if_present(
            right_feet_ray_caster,
            "add_right_feet_ray_caster",
            False,
        )

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
    amp_profile: str,
    start_time: float,
    end_time: float,
) -> None:
    output_path = Path(save_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_dim = {"locomotion": 58, "recovery": 61}[amp_profile]
    if frames.ndim != 2 or frames.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {amp_profile} AMP expert data with shape "
            f"(T, {expected_dim}), got {frames.shape}."
        )
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
        f"Saved AMP expert motion to '{output_path}':\n"
        f"  AMP profile   = {amp_profile}\n"
        f"  frame count   = {frames.shape[0]}\n"
        f"  dimension     = {frames.shape[1]}\n"
        f"  FrameDuration = {frame_duration:.9f}\n"
        f"  fps           = {1.0 / frame_duration:.6f}\n"
        f"  start_time    = {start_time:.6f} s\n"
        f"  end_time      = {end_time:.6f} s"
    )



def _parse_clip_ranges(
    clip_ranges: str | None,
    source_duration: float,
) -> list[tuple[float, float]]:
    """Parse --clip_ranges into validated (start_time, end_time) pairs."""
    if clip_ranges is None or not clip_ranges.strip():
        return []

    ranges: list[tuple[float, float]] = []

    for item in clip_ranges.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid clip range '{item}'. "
                "Expected format like '12.9-18.65,28.5-34.4'."
            )

        try:
            start_time = float(parts[0].strip())
            end_time = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric clip range '{item}'."
            ) from exc

        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError(f"Clip range must be finite, got '{item}'.")
        if start_time < 0.0:
            raise ValueError(
                f"Clip start time must be >= 0, got {start_time}."
            )
        if end_time <= start_time:
            raise ValueError(
                f"Clip end time must be greater than start time, got '{item}'."
            )
        if end_time > source_duration + 1e-8:
            raise ValueError(
                f"Clip '{item}' exceeds source duration "
                f"{source_duration:.6f} s."
            )

        ranges.append((start_time, end_time))

    if not ranges:
        raise ValueError("--clip_ranges was provided but no valid ranges were found.")

    return ranges


def _resolve_time_range(
    start_time: float | None,
    end_time: float | None,
    source_duration: float,
) -> tuple[float, float] | None:
    """Resolve and validate the optional single crop range."""
    if start_time is None and end_time is None:
        return None

    resolved_start = 0.0 if start_time is None else float(start_time)
    resolved_end = source_duration if end_time is None else float(end_time)
    if not np.isfinite(resolved_start) or not np.isfinite(resolved_end):
        raise ValueError("--start_time and --end_time must be finite.")
    if not 0.0 <= resolved_start < resolved_end <= source_duration + 1e-8:
        raise ValueError(
            "Expected 0 <= start_time < end_time <= source_duration, got "
            f"{resolved_start:.6f} <= {resolved_end:.6f} with "
            f"source_duration={source_duration:.6f}."
        )
    return resolved_start, min(resolved_end, source_duration)


def _make_clip_sample_times(
    start_time: float,
    end_time: float,
    frame_duration: float,
) -> np.ndarray:
    """Generate timestamps inside one crop range without sampling beyond it."""
    clip_duration = end_time - start_time

    num_regular_frames = (
        int(np.floor(clip_duration / frame_duration + 1e-9)) + 1
    )
    sample_times = (
        start_time
        + np.arange(num_regular_frames, dtype=np.float64) * frame_duration
    )

    # Preserve the requested end point if it does not lie exactly on the
    # sampling grid. This makes manual time-based cropping intuitive.
    if end_time - sample_times[-1] > 1e-8:
        sample_times = np.concatenate(
            [sample_times, np.array([end_time], dtype=np.float64)]
        )

    return np.minimum(sample_times, end_time)


def _clip_save_path(
    save_path: str,
    clip_index: int,
    num_clips: int,
) -> str:
    """Create an independent output path for each cropped clip."""
    path = Path(save_path).expanduser()

    if num_clips == 1:
        return str(path)

    suffix = path.suffix
    stem = path.stem
    clip_name = f"{stem}_{clip_index:02d}{suffix}"
    return str(path.with_name(clip_name))


def play_amp_animation() -> None:
    print("[play_amp_animation] Preparing task configuration...", flush=True)
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    _disable_generation_randomization(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if args_cli.motion_file is not None:
        motion_path = Path(args_cli.motion_file).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(f"Visualization motion file not found: {motion_path}")
        env_cfg.amp_motion_files_display = [str(motion_path)]

    if hasattr(env_cfg, "commands"):
        _set_if_present(env_cfg.commands, "debug_vis", False)

    if args_cli.seed is not None:
        scene_seed = int(args_cli.seed)
    else:
        scene_seed = int(agent_cfg.seed)
    _set_if_present(env_cfg.scene, "seed", scene_seed)

    if args_cli.save_path and int(args_cli.num_envs) != 1:
        raise ValueError(
            "Generating one AMP expert trajectory requires --num_envs=1."
        )

    print(
        "[play_amp_animation] Creating the IsaacLab environment. The first "
        "URDF import/cache build can take 1-2 minutes; URDF merge and Fabric "
        "missing-visual warnings during this stage are non-fatal.",
        flush=True,
    )
    environment_start_time = time.perf_counter()
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    print(
        "[play_amp_animation] Environment ready in "
        f"{time.perf_counter() - environment_start_time:.1f} s.",
        flush=True,
    )

    source_num_frames = int(
        env.amp_loader_display.trajectory_num_frames[0]
    )
    source_frame_duration = float(
        env.amp_loader_display.trajectory_frame_durations[0]
    )
    source_duration = float(
        env.amp_loader_display.trajectory_lens[0]
    )

    if source_num_frames < 2:
        raise ValueError(
            "The source motion must contain at least two frames."
        )

    if args_cli.fps is None:
        output_frame_duration = source_frame_duration
    else:
        if not np.isfinite(args_cli.fps) or args_cli.fps <= 0.0:
            raise ValueError(
                f"--fps must be positive and finite, got {args_cli.fps}."
            )
        output_frame_duration = 1.0 / float(args_cli.fps)

    clip_ranges = _parse_clip_ranges(
        args_cli.clip_ranges,
        source_duration=source_duration,
    )
    single_time_range = _resolve_time_range(
        args_cli.start_time,
        args_cli.end_time,
        source_duration,
    )
    if clip_ranges and single_time_range is not None:
        raise ValueError("Use either --clip_ranges or --start_time/--end_time, not both.")
    if single_time_range is not None:
        clip_ranges = [single_time_range]

    # No crop ranges: preserve the original full-motion behavior.
    if not clip_ranges:
        if args_cli.fps is None:
            sample_times = (
                np.arange(source_num_frames, dtype=np.float64)
                * source_frame_duration
            )
        else:
            output_num_frames = (
                int(
                    np.floor(
                        source_duration / output_frame_duration + 1e-9
                    )
                )
                + 1
            )
            sample_times = (
                np.arange(output_num_frames, dtype=np.float64)
                * output_frame_duration
            )
            sample_times = np.minimum(sample_times, source_duration)

        playback_ranges = [(0.0, source_duration, sample_times)]
    else:
        playback_ranges = [
            (
                start_time,
                end_time,
                _make_clip_sample_times(
                    start_time,
                    end_time,
                    output_frame_duration,
                ),
            )
            for start_time, end_time in clip_ranges
        ]

    print(
        "Starting G1 motion playback:\n"
        f"  AMP profile       = {args_cli.amp_profile}\n"
        f"  source_frames     = {source_num_frames}\n"
        f"  source_fps        = {1.0 / source_frame_duration:.6f}\n"
        f"  source_duration   = {source_duration:.6f} s\n"
        f"  output_fps        = {1.0 / output_frame_duration:.6f}\n"
        f"  crop_ranges       = {len(clip_ranges)}\n"
    )

    if clip_ranges:
        print("Selected crop ranges:")
        for clip_index, (start_time, end_time) in enumerate(
            clip_ranges, start=1
        ):
            print(
                f"  clip {clip_index:02d}: "
                f"{start_time:.3f} -> {end_time:.3f} s "
                f"({end_time - start_time:.3f} s)"
            )
        print()

    num_playback_ranges = len(playback_ranges)

    for clip_index, (start_time, end_time, sample_times) in enumerate(
        playback_ranges,
        start=1,
    ):
        total_output_frames = len(sample_times)
        all_frames: list[np.ndarray] = []

        if clip_ranges:
            print(
                f"\nPlaying clip {clip_index:02d}/{num_playback_ranges}: "
                f"{start_time:.3f} -> {end_time:.3f} s"
            )

        for frame_index, sample_time in enumerate(sample_times):
            if not simulation_app.is_running():
                print(
                    f"\nSimulation stopped after {frame_index} frames "
                    f"in clip {clip_index}."
                )
                break

            source_frame_index = int(
                round(float(sample_time) / source_frame_duration)
            )
            source_frame_index = min(
                max(source_frame_index, 0),
                source_num_frames - 1,
            )

            if clip_ranges:
                prefix = f"Clip {clip_index:02d} | "
            else:
                prefix = ""

            print(
                "\r"
                f"{prefix}"
                f"Output Frame: "
                f"{frame_index:5d}/{total_output_frames - 1:5d} | "
                f"Source Frame: "
                f"{source_frame_index:5d}/{source_num_frames - 1:5d} | "
                f"Time: "
                f"{float(sample_time):8.3f}/{source_duration:8.3f} s",
                end="",
                flush=True,
            )

            base_amp_frame = env.visualize_motion(float(sample_time))

            if args_cli.save_path:
                if tuple(base_amp_frame.shape) != (1, 58):
                    raise ValueError(
                        "env.visualize_motion() must return shape (1, 58) "
                        "for full G1 23-DoF AMP, "
                        f"but returned {tuple(base_amp_frame.shape)}."
                    )

                amp_frame = base_amp_frame[0]
                if args_cli.amp_profile == "recovery":
                    # This is the same simulator-state field used by
                    # G1RENetEnv.get_recovery_amp_obs() during policy rollout.
                    projected_gravity = env.robot.data.projected_gravity_b
                    if tuple(projected_gravity.shape) != (1, 3):
                        raise ValueError(
                            "robot.data.projected_gravity_b must have shape (1, 3) "
                            "while exporting with --num_envs=1, got "
                            f"{tuple(projected_gravity.shape)}."
                        )
                    gravity = projected_gravity[0]
                    if not gravity.isfinite().all():
                        raise ValueError("projected_gravity_b contains NaN or Inf.")
                    gravity_norm = float(gravity.norm().item())
                    if abs(gravity_norm - 1.0) >= 1e-3:
                        raise ValueError(
                            "projected_gravity_b must have unit norm, got "
                            f"{gravity_norm:.8f}."
                        )
                    amp_frame = torch.cat((amp_frame, gravity), dim=-1)

                all_frames.append(amp_frame.detach().cpu().numpy().copy())

        print()

        if args_cli.save_path:
            if not all_frames:
                raise RuntimeError(
                    f"No AMP expert frames were generated for clip {clip_index}."
                )

            current_save_path = _clip_save_path(
                save_path=args_cli.save_path,
                clip_index=clip_index,
                num_clips=num_playback_ranges,
            )

            _write_motion_file(
                save_path=current_save_path,
                frames=np.stack(all_frames, axis=0),
                frame_duration=output_frame_duration,
                motion_weight=args_cli.motion_weight,
                amp_profile=args_cli.amp_profile,
                start_time=start_time,
                end_time=end_time,
            )


if __name__ == "__main__":
    try:
        play_amp_animation()
    finally:
        simulation_app.close()
