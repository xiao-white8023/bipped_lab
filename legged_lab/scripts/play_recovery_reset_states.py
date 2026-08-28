#!/usr/bin/env python3
"""Playback one-state Recovery reset NPZ files for visual quality control.

Each state is written directly to G1_23CFG and refreshed with ``sim.forward``.
No physics step or rollout is performed.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_recovery_reset_dataset import DEFAULT_OUTPUT_DIR, load_source_npz


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reset_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing one-state reset_state_*.npz files.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--state",
        type=int,
        default=None,
        help="Display one 1-based state index after natural filename sorting.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Display all states sequentially.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="With --all, start from this 1-based state index.",
    )
    parser.add_argument(
        "--pause_s",
        type=float,
        default=1.0,
        help="Wall-clock seconds to hold each state without stepping physics.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the selected state or playlist.",
    )


# AppLauncher must precede the remaining IsaacLab/Omniverse imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Playback Stage-4.2a G1 Recovery reset states without physics rollout."
)
_add_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.state is not None and args.state <= 0:
    parser.error("--state is 1-based and must be positive.")
if args.start_index <= 0:
    parser.error("--start_index is 1-based and must be positive.")
if args.pause_s < 0.0:
    parser.error("--pause_s must be non-negative.")
if args.state is not None and args.start_index != 1:
    parser.error("--start_index is only meaningful with --all.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from legged_lab.assets.g1.g1_29 import G1_23CFG


DISPLAY_METADATA = (
    "source_file",
    "source_frame",
    "source_time_s",
    "pelvis_height",
    "torso_up_z",
    "root_lin_speed",
    "root_ang_speed",
    "mean_joint_speed",
)


@dataclass(frozen=True)
class ResetState:
    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_body_index: int
    metadata: dict[str, str | int | float]


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


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _scalar(data, key: str, cast):
    if key not in data.files:
        raise KeyError(f"{key} metadata is required for reset-state playback.")
    array = np.asarray(data[key])
    if array.size != 1:
        raise ValueError(f"{key} must contain one scalar, got {array.shape}.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return cast(item)


def discover_state_files() -> list[tuple[int, Path]]:
    reset_dir = args.reset_dir.expanduser().resolve()
    if not reset_dir.is_dir():
        raise NotADirectoryError(f"Reset-state directory not found: {reset_dir}")
    files = sorted(reset_dir.glob("reset_state_*.npz"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No reset_state_*.npz files found in {reset_dir}")

    if args.state is not None:
        if args.state > len(files):
            raise IndexError(
                f"--state {args.state} requested, but only {len(files)} states exist."
            )
        return [(args.state, files[args.state - 1])]

    if args.all:
        if args.start_index > len(files):
            raise IndexError(
                f"--start_index {args.start_index}, but only {len(files)} states exist."
            )
        return list(enumerate(files[args.start_index - 1 :], start=args.start_index))

    return [(1, files[0])]


def load_reset_state(path: Path) -> ResetState:
    source = load_source_npz(path)
    if source.num_frames != 1:
        raise ValueError(
            f"{path.name}: reset-state core arrays must preserve T=1, "
            f"got T={source.num_frames}."
        )

    with np.load(path, allow_pickle=False) as data:
        metadata = {
            "source_file": _scalar(data, "source_file", str),
            "source_frame": _scalar(data, "source_frame", int),
            "source_time_s": _scalar(data, "source_time_s", float),
            "pelvis_height": _scalar(data, "pelvis_height", float),
            "torso_up_z": _scalar(data, "torso_up_z", float),
            "root_lin_speed": _scalar(data, "root_lin_speed", float),
            "root_ang_speed": _scalar(data, "root_ang_speed", float),
            "mean_joint_speed": _scalar(data, "mean_joint_speed", float),
        }

    payload = source.payload
    return ResetState(
        path=path,
        fps=source.fps,
        joint_pos=payload["joint_pos"],
        joint_vel=payload["joint_vel"],
        body_pos_w=payload["body_pos_w"],
        body_quat_w=payload["body_quat_w"],
        body_lin_vel_w=payload["body_lin_vel_w"],
        body_ang_vel_w=payload["body_ang_vel_w"],
        joint_names=source.joint_names,
        body_names=source.body_names,
        root_body_index=source.root_body_index,
        metadata=metadata,
    )


def validate_robot_order(robot, state: ResetState) -> None:
    robot_joint_names = tuple(robot.joint_names)
    robot_body_names = tuple(robot.body_names)
    if robot_joint_names != state.joint_names:
        raise RuntimeError(
            f"{state.path.name}: joint order mismatch.\n"
            f"NPZ:   {state.joint_names}\nRobot: {robot_joint_names}"
        )
    if robot_body_names != state.body_names:
        raise RuntimeError(
            f"{state.path.name}: body order mismatch.\n"
            f"NPZ:   {state.body_names}\nRobot: {robot_body_names}"
        )


def prepare_tensors(robot, state: ResetState) -> dict[str, torch.Tensor]:
    root_index = state.root_body_index
    root_pose = np.concatenate(
        (state.body_pos_w[:, root_index], state.body_quat_w[:, root_index]), axis=-1
    )
    root_velocity = np.concatenate(
        (
            state.body_lin_vel_w[:, root_index],
            state.body_ang_vel_w[:, root_index],
        ),
        axis=-1,
    )
    return {
        "root_pose": torch.as_tensor(root_pose, dtype=torch.float32, device=robot.device),
        "root_velocity": torch.as_tensor(
            root_velocity, dtype=torch.float32, device=robot.device
        ),
        "joint_pos": torch.as_tensor(
            state.joint_pos, dtype=torch.float32, device=robot.device
        ),
        "joint_vel": torch.as_tensor(
            state.joint_vel, dtype=torch.float32, device=robot.device
        ),
    }


def write_state(sim, scene, robot, tensors, dt: float) -> None:
    robot.write_root_link_pose_to_sim(tensors["root_pose"])
    robot.write_root_link_velocity_to_sim(tensors["root_velocity"])
    robot.write_joint_state_to_sim(tensors["joint_pos"], tensors["joint_vel"])

    # FK/data refresh only: intentionally no sim.step().
    sim.forward()
    scene.update(dt)
    sim.render()


def display_state(sim, scene, robot, state: ResetState, index: int, total: int) -> None:
    validate_robot_order(robot, state)
    tensors = prepare_tensors(robot, state)
    root_position = tensors["root_pose"][0, :3].detach().cpu().tolist()
    sim.set_camera_view(
        eye=(root_position[0] + 2.0, root_position[1] + 2.0, 1.5),
        target=(root_position[0], root_position[1], 0.5),
    )
    write_state(sim, scene, robot, tensors, 1.0 / state.fps)

    print("\n" + "=" * 88)
    print(f"RESET STATE [{index}/{total}] {state.path.name}")
    for name in DISPLAY_METADATA:
        print(f"  {name}: {state.metadata[name]}")
    print("=" * 88, flush=True)

    deadline = time.perf_counter() + args.pause_s
    while simulation_app.is_running() and time.perf_counter() < deadline:
        sim.render()
        time.sleep(min(1.0 / 30.0, max(0.0, deadline - time.perf_counter())))


def main() -> None:
    indexed_files = discover_state_files()
    states = [(index, load_reset_state(path)) for index, path in indexed_files]

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / states[0][1].fps,
        device=args.device,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(PlaybackSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]

    print(f"Loaded G1_23CFG with {len(robot.joint_names)} joints and {len(robot.body_names)} bodies.")
    try:
        while simulation_app.is_running():
            for index, state in states:
                if not simulation_app.is_running():
                    return
                display_state(
                    sim=sim,
                    scene=scene,
                    robot=robot,
                    state=state,
                    index=index,
                    total=len(indexed_files),
                )
            if not args.loop:
                return
    finally:
        simulation_app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
