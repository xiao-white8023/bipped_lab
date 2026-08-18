#!/usr/bin/env python3
"""Play Stage-2 Recovery NPZ clips in IsaacLab for visual inspection.

This is a DATA PLAYBACK tool, not a physics rollout:
- every frame directly writes the recorded pelvis/root-link pose,
  recorded root-link velocity, and recorded 23-DoF joint state;
- SimulationContext.forward() updates articulation kinematics/fabric;
- SimulationContext.render() refreshes the viewport;
- no policy, PD tracking, gravity integration, or contact dynamics are used
  to change the recorded pose.

That behavior is intentional: the purpose is to inspect the exact Stage-2
motion data before it is accepted as Recovery expert data.

Expected repository import:
    from legged_lab.assets.g1.g1_29 import G1_23CFG

Examples
--------
Play candidate 1 repeatedly:
    python play_recovery_clips.py \
        --motion_dir path/to/RecoveryCandidates \
        --clip 1 \
        --loop

Play all candidates once:
    python play_recovery_clips.py \
        --motion_dir path/to/RecoveryCandidates \
        --all

Play one NPZ directly at half speed:
    python play_recovery_clips.py \
        --input_npz recovery_03.npz \
        --speed 0.5
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--motion_dir",
        type=Path,
        help="Directory containing Stage-2 recovery candidate .npz files.",
    )
    source.add_argument(
        "--input_npz",
        type=Path,
        help="Play one Stage-2 recovery .npz directly.",
    )

    select = parser.add_mutually_exclusive_group()
    select.add_argument(
        "--clip",
        type=int,
        default=None,
        help="1-based clip number after sorting files in --motion_dir.",
    )
    select.add_argument(
        "--all",
        action="store_true",
        help="Play every .npz in --motion_dir sequentially.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the selected clip. With --all, loop the complete playlist.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 0.5=half speed, 1=real time, 2=double speed.",
    )
    parser.add_argument(
        "--pause_between",
        type=float,
        default=1.0,
        help="Seconds to hold the final frame between clips when using --all.",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=2.6,
        help="Initial viewport camera distance from the motion center.",
    )
    parser.add_argument(
        "--camera_height",
        type=float,
        default=1.5,
        help="Initial viewport camera height.",
    )
    parser.add_argument(
        "--camera_side",
        choices=("front", "side", "diag"),
        default="diag",
        help="Initial camera direction.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Print playback progress every N frames. 0 disables per-frame progress.",
    )


# AppLauncher must be imported before the remaining Isaac/Omniverse modules.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay Stage-2 G1 Recovery NPZ clips in the IsaacLab viewport."
)
_add_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.speed <= 0.0:
    parser.error("--speed must be > 0.")
if args.pause_between < 0.0:
    parser.error("--pause_between must be >= 0.")
if args.clip is not None and args.clip <= 0:
    parser.error("--clip is 1-based and must be >= 1.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from legged_lab.assets.g1.g1_29 import G1_23CFG


CORE_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class MotionClip:
    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_names: tuple[str, ...]
    root_body_index: int
    root_body_name: str

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps


@configclass
class PlaybackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    robot = G1_23CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got shape={array.shape}.")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {result}.")
    return result


def _string_list(value: np.ndarray, name: str) -> list[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={array.shape}.")
    result: list[str] = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    return result


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def discover_motion_files() -> list[Path]:
    if args.input_npz is not None:
        path = args.input_npz.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input NPZ not found: {path}")
        return [path]

    assert args.motion_dir is not None
    directory = args.motion_dir.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Motion directory not found: {directory}")

    files = sorted(directory.glob("*.npz"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    print(f"\nFound {len(files)} NPZ clip(s):")
    for index, path in enumerate(files, start=1):
        print(f"  [{index:02d}] {path.name}")

    if args.all:
        return files

    clip_index = args.clip if args.clip is not None else 1
    if clip_index > len(files):
        raise IndexError(
            f"--clip {clip_index} requested, but directory has only {len(files)} clips."
        )
    return [files[clip_index - 1]]


def load_clip(path: Path) -> MotionClip:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in CORE_KEYS if key not in data.files]
        if missing:
            raise KeyError(
                f"{path.name}: missing required Stage-2 keys {missing}; "
                f"available={data.files}."
            )

        if "joint_names" not in data.files:
            raise KeyError(
                f"{path.name}: joint_names metadata is required for safe playback."
            )
        if "body_names" not in data.files:
            raise KeyError(
                f"{path.name}: body_names metadata is required for safe playback."
            )

        fps = _scalar_float(data["fps"], "fps")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        joint_names = tuple(_string_list(data["joint_names"], "joint_names"))
        body_names = _string_list(data["body_names"], "body_names")

        if "root_body_index" in data.files:
            root_body_index = int(np.asarray(data["root_body_index"]).reshape(-1)[0])
        elif "pelvis" in body_names:
            root_body_index = body_names.index("pelvis")
        else:
            raise ValueError(
                f"{path.name}: cannot resolve root body; expected root_body_index or pelvis."
            )

        if not (0 <= root_body_index < len(body_names)):
            raise ValueError(
                f"{path.name}: root_body_index={root_body_index} out of range."
            )
        root_body_name = body_names[root_body_index]

    frame_count = joint_pos.shape[0]
    if joint_pos.ndim != 2:
        raise ValueError(f"{path.name}: joint_pos must be [T,J], got {joint_pos.shape}.")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"{path.name}: joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}."
        )
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(
            f"{path.name}: body_pos_w must be [T,B,3], got {body_pos_w.shape}."
        )
    if body_quat_w.shape[:2] != body_pos_w.shape[:2] or body_quat_w.shape[-1] != 4:
        raise ValueError(
            f"{path.name}: body_quat_w must be [T,B,4], got {body_quat_w.shape}."
        )
    if body_lin_vel_w.shape != body_pos_w.shape:
        raise ValueError(f"{path.name}: body_lin_vel_w shape mismatch.")
    if body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError(f"{path.name}: body_ang_vel_w shape mismatch.")

    arrays = (
        joint_pos,
        joint_vel,
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
        body_ang_vel_w,
    )
    if any(array.shape[0] != frame_count for array in arrays):
        raise ValueError(f"{path.name}: core arrays do not share one frame count.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path.name}: one or more core arrays contain NaN/Inf.")

    quat_norm = np.linalg.norm(body_quat_w[:, root_body_index], axis=1)
    if np.max(np.abs(quat_norm - 1.0)) > 1.0e-3:
        raise ValueError(
            f"{path.name}: root quaternion norm deviates too far from one."
        )

    return MotionClip(
        path=path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos_w=body_pos_w[:, root_body_index],
        root_quat_w=body_quat_w[:, root_body_index],
        root_lin_vel_w=body_lin_vel_w[:, root_body_index],
        root_ang_vel_w=body_ang_vel_w[:, root_body_index],
        joint_names=joint_names,
        root_body_index=root_body_index,
        root_body_name=root_body_name,
    )


def validate_robot_joint_order(robot, clip: MotionClip) -> None:
    robot_names = tuple(robot.joint_names)
    if robot_names != clip.joint_names:
        print("\nERROR: NPZ joint order does not match the current IsaacLab articulation.")
        print(f"NPZ ({len(clip.joint_names)}): {list(clip.joint_names)}")
        print(f"Robot ({len(robot_names)}): {list(robot_names)}")

        if set(robot_names) == set(clip.joint_names):
            mismatch = [
                (i, clip.joint_names[i], robot_names[i])
                for i in range(min(len(robot_names), len(clip.joint_names)))
                if clip.joint_names[i] != robot_names[i]
            ]
            print(f"Order mismatches: {mismatch[:10]}")
        raise RuntimeError(
            "Refusing playback because joint order is unsafe. "
            "Rebuild Stage 1 against this exact G1_23CFG."
        )


def set_camera_for_clip(sim: SimulationContext, clip: MotionClip) -> None:
    center_xy = np.median(clip.root_pos_w[:, :2], axis=0)
    target = (
        float(center_xy[0]),
        float(center_xy[1]),
        0.70,
    )

    d = float(args.camera_distance)
    h = float(args.camera_height)
    if args.camera_side == "front":
        eye = (target[0] + d, target[1], h)
    elif args.camera_side == "side":
        eye = (target[0], target[1] + d, h)
    else:
        scale = d / (2.0 ** 0.5)
        eye = (target[0] + scale, target[1] + scale, h)

    sim.set_camera_view(eye=eye, target=target)


def prepare_tensors(robot, clip: MotionClip) -> dict[str, torch.Tensor]:
    device = robot.device
    root_pose = torch.as_tensor(
        np.concatenate((clip.root_pos_w, clip.root_quat_w), axis=1),
        dtype=torch.float32,
        device=device,
    )
    root_velocity = torch.as_tensor(
        np.concatenate((clip.root_lin_vel_w, clip.root_ang_vel_w), axis=1),
        dtype=torch.float32,
        device=device,
    )
    joint_pos = torch.as_tensor(
        clip.joint_pos,
        dtype=torch.float32,
        device=device,
    )
    joint_vel = torch.as_tensor(
        clip.joint_vel,
        dtype=torch.float32,
        device=device,
    )
    return {
        "root_pose": root_pose,
        "root_velocity": root_velocity,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def write_frame(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot,
    tensors: dict[str, torch.Tensor],
    frame_id: int,
    dt: float,
) -> None:
    robot.write_root_link_pose_to_sim(
        tensors["root_pose"][frame_id : frame_id + 1]
    )
    robot.write_root_link_velocity_to_sim(
        tensors["root_velocity"][frame_id : frame_id + 1]
    )
    robot.write_joint_state_to_sim(
        tensors["joint_pos"][frame_id : frame_id + 1],
        tensors["joint_vel"][frame_id : frame_id + 1],
    )

    # Exact data playback: update FK/rendering but do not integrate physics.
    sim.forward()
    scene.update(dt)
    sim.render()


def hold_last_frame(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot,
    tensors: dict[str, torch.Tensor],
    clip: MotionClip,
    seconds: float,
) -> None:
    if seconds <= 0.0 or not simulation_app.is_running():
        return

    dt = 1.0 / clip.fps
    deadline = time.perf_counter() + seconds
    while simulation_app.is_running() and time.perf_counter() < deadline:
        frame_start = time.perf_counter()
        write_frame(
            sim,
            scene,
            robot,
            tensors,
            clip.num_frames - 1,
            dt,
        )
        remaining = dt - (time.perf_counter() - frame_start)
        if remaining > 0.0:
            time.sleep(remaining)


def play_clip(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot,
    clip: MotionClip,
    playlist_index: int,
    playlist_size: int,
) -> None:
    validate_robot_joint_order(robot, clip)
    tensors = prepare_tensors(robot, clip)
    set_camera_for_clip(sim, clip)

    source_start = None
    source_end = None
    with np.load(clip.path, allow_pickle=False) as data:
        if "source_start_s" in data.files:
            source_start = float(np.asarray(data["source_start_s"]).reshape(-1)[0])
        if "source_end_s" in data.files:
            source_end = float(np.asarray(data["source_end_s"]).reshape(-1)[0])

    print("\n" + "=" * 86)
    print(
        f"PLAY [{playlist_index}/{playlist_size}] {clip.path.name}\n"
        f"  frames={clip.num_frames}, fps={clip.fps:g}, "
        f"duration={clip.num_frames / clip.fps:.3f}s, "
        f"speed={args.speed:g}x\n"
        f"  root={clip.root_body_name!r}, joints={len(clip.joint_names)}"
    )
    if source_start is not None and source_end is not None:
        print(
            f"  source timeline: {source_start:.3f}s -> {source_end:.3f}s"
        )
    print("=" * 86)

    playback_dt = 1.0 / (clip.fps * args.speed)
    data_dt = 1.0 / clip.fps

    for frame_id in range(clip.num_frames):
        if not simulation_app.is_running():
            return

        wall_start = time.perf_counter()
        write_frame(
            sim=sim,
            scene=scene,
            robot=robot,
            tensors=tensors,
            frame_id=frame_id,
            dt=data_dt,
        )

        if args.print_every > 0 and (
            frame_id == 0
            or (frame_id + 1) % args.print_every == 0
            or frame_id + 1 == clip.num_frames
        ):
            print(
                f"\r  frame {frame_id + 1:4d}/{clip.num_frames:4d} | "
                f"motion t={frame_id / clip.fps:6.2f}s",
                end="",
                flush=True,
            )

        elapsed = time.perf_counter() - wall_start
        remaining = playback_dt - elapsed
        if remaining > 0.0:
            time.sleep(remaining)

    if args.print_every > 0:
        print()

    hold_last_frame(
        sim,
        scene,
        robot,
        tensors,
        clip,
        float(args.pause_between),
    )


def main() -> None:
    files = discover_motion_files()
    clips = [load_clip(path) for path in files]

    first_fps = clips[0].fps
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / first_fps,
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)

    scene_cfg = PlaybackSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    print(f"\nIsaacLab robot joint order ({len(robot.joint_names)}):")
    print(list(robot.joint_names))
    print(f"IsaacLab body count: {len(robot.body_names)}")

    try:
        while simulation_app.is_running():
            for index, clip in enumerate(clips, start=1):
                if not simulation_app.is_running():
                    return
                play_clip(
                    sim=sim,
                    scene=scene,
                    robot=robot,
                    clip=clip,
                    playlist_index=index,
                    playlist_size=len(clips),
                )

            if not args.loop:
                print("\nPlayback finished. Close the IsaacLab window or press Ctrl+C.")
                final_clip = clips[-1]
                final_tensors = prepare_tensors(robot, final_clip)
                while simulation_app.is_running():
                    write_frame(
                        sim,
                        scene,
                        robot,
                        final_tensors,
                        final_clip.num_frames - 1,
                        1.0 / final_clip.fps,
                    )
                    time.sleep(1.0 / 30.0)
                return
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
