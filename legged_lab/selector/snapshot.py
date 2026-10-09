"""Full-state snapshots for paired LP/VP counterfactual rollouts.

The implementation is intentionally outside ``G1RENetEnv``.  It captures all
known policy histories plus tensor state owned by command and sensor helpers.
The collector must run the replay-equivalence check before accepting data,
because PhysX contact warm-start caches are not a public snapshot API.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch


@dataclass
class ObjectTensorState:
    tensors: dict[str, torch.Tensor]
    tensor_lists: dict[str, list[torch.Tensor]]
    tensor_dicts: dict[str, dict[Any, torch.Tensor]]
    scalars: dict[str, int | float | bool]


@dataclass
class RENetSnapshot:
    root_state: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    objects: dict[str, ObjectTensorState]
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None
    numpy_rng: tuple[Any, ...]
    python_rng: tuple[Any, ...]
    simulator_time: float | None
    simulator_step: int | None


def _capture_object(obj: Any) -> ObjectTensorState:
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, int | float | bool] = {}
    tensor_lists: dict[str, list[torch.Tensor]] = {}
    tensor_dicts: dict[str, dict[Any, torch.Tensor]] = {}
    if obj is None or not hasattr(obj, "__dict__"):
        return ObjectTensorState(
            tensors=tensors,
            tensor_lists=tensor_lists,
            tensor_dicts=tensor_dicts,
            scalars=scalars,
        )
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value.clone()
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(item, torch.Tensor) for item in value
        ):
            tensor_lists[name] = [item.clone() for item in value]
        elif isinstance(value, dict) and value and all(
            isinstance(item, torch.Tensor) for item in value.values()
        ):
            tensor_dicts[name] = {key: item.clone() for key, item in value.items()}
        elif name in ("_pointer", "cursor", "sim_step_counter") and isinstance(value, (int, float, bool)):
            scalars[name] = value
    return ObjectTensorState(
        tensors=tensors,
        tensor_lists=tensor_lists,
        tensor_dicts=tensor_dicts,
        scalars=scalars,
    )


def _restore_object(obj: Any, state: ObjectTensorState) -> None:
    for name, source in state.tensors.items():
        if not hasattr(obj, name):
            raise RuntimeError(f"snapshot target lost tensor field {type(obj).__name__}.{name}")
        target = getattr(obj, name)
        if not isinstance(target, torch.Tensor) or target.shape != source.shape:
            raise RuntimeError(f"snapshot tensor contract changed for {type(obj).__name__}.{name}")
        target.copy_(source)
    for name, sources in state.tensor_lists.items():
        target = getattr(obj, name, None)
        if not isinstance(target, (list, tuple)) or len(target) != len(sources):
            raise RuntimeError(f"snapshot tensor-list contract changed for {type(obj).__name__}.{name}")
        for target_tensor, source in zip(target, sources):
            if not isinstance(target_tensor, torch.Tensor) or target_tensor.shape != source.shape:
                raise RuntimeError(
                    f"snapshot tensor-list item changed for {type(obj).__name__}.{name}"
                )
            target_tensor.copy_(source)
    for name, sources in state.tensor_dicts.items():
        target = getattr(obj, name, None)
        if not isinstance(target, dict) or set(target) != set(sources):
            raise RuntimeError(f"snapshot tensor-dict contract changed for {type(obj).__name__}.{name}")
        for key, source in sources.items():
            target_tensor = target[key]
            if not isinstance(target_tensor, torch.Tensor) or target_tensor.shape != source.shape:
                raise RuntimeError(
                    f"snapshot tensor-dict item changed for {type(obj).__name__}.{name}[{key!r}]"
                )
            target_tensor.copy_(source)
    for name, value in state.scalars.items():
        setattr(obj, name, value)


class RENetSnapshotManager:
    """Capture and restore one complete vectorized G1 RENet simulation state."""

    _TOP_LEVEL_NAMES = (
        "episode_length_buf",
        "time_out_buf",
        "reset_buf",
        "phase",
        "phase_left",
        "phase_right",
        "leg_phase",
        "action",
        "renet_estimator_mask",
        "depth_buffer",
        "depth_last_capture_time",
        "depth_initialized",
        "lp_height_history",
        "lidar_height_map",
        "lidar_confidence",
        "lidar_map_observation",
        "lp_missing_duration",
        "lp_last_diag_time",
        "avg_feet_force_per_step",
        "avg_feet_speed_per_step",
    )

    def __init__(self, env: Any):
        self.env = env
        required = ("robot", "scene", "sim", "actor_obs_buffer", "critic_obs_buffer", "action_buffer")
        missing = [name for name in required if not hasattr(env, name)]
        if missing:
            raise TypeError(f"environment is not a compatible G1RENetEnv; missing {missing}")

    def _objects(self) -> dict[str, Any]:
        env = self.env
        action_circular = getattr(getattr(env, "action_buffer", None), "_circular_buffer", None)
        sensors = {
            name: sensor for name, sensor in getattr(env.scene, "sensors", {}).items()
        }
        return {
            "env": env,
            "actor_obs_buffer": getattr(env, "actor_obs_buffer", None),
            "critic_obs_buffer": getattr(env, "critic_obs_buffer", None),
            "action_buffer": getattr(env, "action_buffer", None),
            "action_circular_buffer": action_circular,
            "command_generator": getattr(env, "command_generator", None),
            "reward_manager": getattr(env, "reward_manager", None),
            "event_manager": getattr(env, "event_manager", None),
            "terrain": getattr(env.scene, "terrain", None),
            "lp_history": getattr(env, "lp_history", None),
            "pgtt_noise": getattr(env, "pgtt_noise", None),
            "camera": getattr(env, "camera", None),
            "pgtt_height_scanner": getattr(env, "pgtt_height_scanner", None),
            **{f"sensor_{name}": sensor for name, sensor in sensors.items()},
            **{
                f"sensor_data_{name}": getattr(sensor, "data", None)
                for name, sensor in sensors.items()
            },
        }

    def capture(self) -> RENetSnapshot:
        env = self.env
        root_state = env.robot.data.root_state_w.clone()
        joint_position = env.robot.data.joint_pos.clone()
        joint_velocity = env.robot.data.joint_vel.clone()
        objects = self._objects()
        object_states = {name: _capture_object(obj) for name, obj in objects.items() if obj is not None}
        # Environment tensor fields are explicitly constrained so that newly
        # added non-state tensors are not accidentally copied in future.
        env_state = object_states["env"]
        env_state.tensors = {
            name: getattr(env, name).clone()
            for name in self._TOP_LEVEL_NAMES
            if hasattr(env, name) and isinstance(getattr(env, name), torch.Tensor)
        }
        env_state.tensor_lists = {}
        env_state.tensor_dicts = {}
        env_state.scalars = {
            "sim_step_counter": int(getattr(env, "sim_step_counter", 0)),
            "renet_training_iteration": int(getattr(env, "renet_training_iteration", 0)),
            "_pgtt_noise_stage_active": bool(
                getattr(env, "_pgtt_noise_stage_active", False)
            ),
        }
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        simulator_time = getattr(env.sim, "_current_time", None)
        simulator_step = getattr(env.sim, "_current_time_step_index", None)
        return RENetSnapshot(
            root_state=root_state,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            objects=object_states,
            torch_rng=torch.random.get_rng_state().clone(),
            cuda_rng=[state.clone() for state in cuda_rng] if cuda_rng is not None else None,
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            simulator_time=float(simulator_time) if simulator_time is not None else None,
            simulator_step=int(simulator_step) if simulator_step is not None else None,
        )

    def restore(self, snapshot: RENetSnapshot) -> None:
        env = self.env
        env.robot.write_root_state_to_sim(snapshot.root_state)
        env.robot.write_joint_state_to_sim(snapshot.joint_position, snapshot.joint_velocity)
        objects = self._objects()
        for name, state in snapshot.objects.items():
            if name not in objects or objects[name] is None:
                raise RuntimeError(f"snapshot target object {name!r} is unavailable")
            _restore_object(objects[name], state)
        torch.random.set_rng_state(snapshot.torch_rng)
        if snapshot.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshot.cuda_rng)
        np.random.set_state(snapshot.numpy_rng)
        random.setstate(snapshot.python_rng)
        if snapshot.simulator_time is not None and hasattr(env.sim, "_current_time"):
            env.sim._current_time = snapshot.simulator_time
        if snapshot.simulator_step is not None and hasattr(env.sim, "_current_time_step_index"):
            env.sim._current_time_step_index = snapshot.simulator_step
        env.scene.write_data_to_sim()
        env.sim.forward()


@torch.no_grad()
def replay_equivalence(
    manager: RENetSnapshotManager,
    rollout_fn,
    *,
    trajectory_atol: float = 1.0e-4,
) -> dict[str, float | bool]:
    """Run the same branch twice; collectors must call this before writing data."""

    snapshot = manager.capture()
    first = rollout_fn()
    manager.restore(snapshot)
    second = rollout_fn()
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("rollout_fn must return a trajectory tensor")
    if first.shape != second.shape:
        raise RuntimeError("replay trajectories have different shapes")
    error = (first - second).abs()
    max_error = float(error.max().item()) if error.numel() else 0.0
    rmse = float(torch.sqrt(error.square().mean()).item()) if error.numel() else 0.0
    return {
        "passed": bool(max_error <= trajectory_atol),
        "max_abs_error": max_error,
        "rmse": rmse,
        "atol": trajectory_atol,
    }
