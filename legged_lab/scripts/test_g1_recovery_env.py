#!/usr/bin/env python3
"""Stage-4.2c smoke test for G1 Recovery reset visualization/QC APIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RSL_RL_PATH = REPO_ROOT / "rsl_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))

# AppLauncher must precede all other IsaacLab/Omniverse imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Smoke-test Stage-4.2c G1 Recovery reset visualization/QC APIs."
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch

from legged_lab.envs.g1.g1_recovery_cfg import G1RecoveryEnvCfg
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv


NUM_ENVS = 4
ACTOR_OBS_DIM = 300
CRITIC_OBS_DIM = 312
AMP_OBS_DIM = 195
EXPERT_TRANSITION_DT = 0.02


def _assert_finite_shape(tensor, shape, label):
    if tensor.shape != shape:
        raise AssertionError(
            f"{label}: expected shape {shape}, got {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{label} contains NaN or Inf values.")


def _check_step_output(step_output):
    actor_obs, reward, done, infos = step_output
    critic_obs = infos["observations"]["critic"]
    _assert_finite_shape(actor_obs, (NUM_ENVS, ACTOR_OBS_DIM), "actor_obs")
    _assert_finite_shape(critic_obs, (NUM_ENVS, CRITIC_OBS_DIM), "critic_obs")
    _assert_finite_shape(reward, (NUM_ENVS,), "reward")
    if torch.count_nonzero(reward).item() != 0:
        raise AssertionError("Stage-4.1 task reward must remain exactly zero.")
    if done.shape != (NUM_ENVS,) or done.dtype != torch.bool:
        raise AssertionError(f"done has invalid shape/dtype: {done.shape}, {done.dtype}.")
    if not torch.isfinite(done.float()).all():
        raise AssertionError("done contains invalid values.")
    return actor_obs, reward, done, infos


def _assert_dataset_reset(env, env_ids=None):
    if env_ids is None:
        env_ids = torch.arange(NUM_ENVS, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    state_ids = env.last_reset_state_ids[env_ids]
    if state_ids.shape != (env_ids.numel(),):
        raise AssertionError(
            f"last_reset_state_ids has invalid shape {tuple(state_ids.shape)}."
        )
    if torch.any(state_ids < 0) or torch.any(
        state_ids >= env.reset_state_loader.num_states
    ):
        raise AssertionError("last_reset_state_ids contains an invalid dataset ID.")

    reference = env.reset_state_loader.get_state_batch(state_ids)
    env_origins = env.scene.env_origins[env_ids]
    expected_root_pos = reference["root_pos_w"].clone()
    expected_root_pos[:, :2] = env_origins[:, :2]
    expected_root_pos[:, 2] = (
        env_origins[:, 2] + reference["root_pos_w"][:, 2]
    )

    atol = 2.0e-5
    torch.testing.assert_close(
        env.robot.data.root_link_pos_w[env_ids],
        expected_root_pos,
        rtol=1.0e-5,
        atol=atol,
    )

    robot_quat = torch.nn.functional.normalize(
        env.robot.data.root_link_quat_w[env_ids], dim=-1
    )
    source_quat = torch.nn.functional.normalize(reference["root_quat_w"], dim=-1)
    orientation_dot = torch.abs(torch.sum(robot_quat * source_quat, dim=-1))
    torch.testing.assert_close(
        orientation_dot,
        torch.ones_like(orientation_dot),
        rtol=0.0,
        atol=atol,
    )

    comparisons = (
        (
            env.robot.data.root_link_lin_vel_w[env_ids],
            reference["root_lin_vel_w"],
            "root linear velocity",
        ),
        (
            env.robot.data.root_link_ang_vel_w[env_ids],
            reference["root_ang_vel_w"],
            "root angular velocity",
        ),
        (
            env.robot.data.joint_pos[env_ids],
            reference["joint_pos"],
            "joint position",
        ),
        (
            env.robot.data.joint_vel[env_ids],
            reference["joint_vel"],
            "joint velocity",
        ),
        (
            env.joint_pos_target[env_ids],
            reference["joint_pos"],
            "environment joint position target",
        ),
        (
            env.robot.data.joint_pos_target[env_ids],
            reference["joint_pos"],
            "actuator joint position target",
        ),
        (
            env.last_reset_joint_pos[env_ids],
            reference["joint_pos"],
            "cached reset joint position",
        ),
    )
    for actual, expected, label in comparisons:
        try:
            torch.testing.assert_close(
                actual, expected, rtol=1.0e-5, atol=atol
            )
        except AssertionError as error:
            raise AssertionError(f"Dataset reset did not preserve {label}.") from error

    torch.testing.assert_close(
        env.robot.data.joint_vel_target[env_ids],
        torch.zeros_like(reference["joint_vel"]),
        rtol=0.0,
        atol=0.0,
    )
    return reference


def run_smoke_test():
    cfg = G1RecoveryEnvCfg()
    cfg.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.max_episode_length_s = 0.08
    state_dir = Path(cfg.reset.state_dir)
    if not state_dir.is_dir() or not any(state_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"Recovery reset-state dataset is missing or empty: {state_dir}. "
            "Run legged_lab/scripts/build_recovery_reset_dataset.py first."
        )
    env = G1RecoveryEnv(cfg=cfg, headless=args.headless)

    if env.physics_dt != 0.005:
        raise AssertionError(f"physics_dt must be 0.005, got {env.physics_dt}.")
    if env.cfg.sim.decimation != 4:
        raise AssertionError(
            f"decimation must be 4, got {env.cfg.sim.decimation}."
        )
    if env.step_dt != 0.02:
        raise AssertionError(f"step_dt must be 0.02, got {env.step_dt}.")
    if EXPERT_TRANSITION_DT != env.step_dt:
        raise AssertionError("Expert and policy Recovery transition_dt must match.")
    if env.num_actions != 23:
        raise AssertionError(f"num_actions must be 23, got {env.num_actions}.")
    if env.max_episode_length != 4:
        raise AssertionError(
            f"0.08 s must produce four policy steps, got {env.max_episode_length}."
        )
    if env.last_reset_state_ids.shape != (NUM_ENVS,):
        raise AssertionError(
            f"last_reset_state_ids must be {(NUM_ENVS,)}, "
            f"got {tuple(env.last_reset_state_ids.shape)}."
        )
    _assert_dataset_reset(env)

    deterministic_state_ids = torch.arange(
        NUM_ENVS, dtype=torch.long, device=env.device
    )
    all_env_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=env.device)
    env.reset_to_state_ids(all_env_ids, deterministic_state_ids)
    if not torch.equal(env.last_reset_state_ids, deterministic_state_ids):
        raise AssertionError(
            "reset_to_state_ids did not preserve deterministic IDs [0, 1, 2, 3]."
        )
    _assert_dataset_reset(env)

    try:
        env.reset_to_state_ids([0, 1], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("reset_to_state_ids must reject unequal ID lengths.")
    try:
        env.reset_to_state_ids([0], [env.reset_state_loader.num_states])
    except IndexError:
        pass
    else:
        raise AssertionError("reset_to_state_ids must reject out-of-range state IDs.")

    freeze_reference = {
        "root_pos": env.robot.data.root_link_pos_w.clone(),
        "root_quat": env.robot.data.root_link_quat_w.clone(),
        "joint_pos": env.robot.data.joint_pos.clone(),
        "joint_vel": env.robot.data.joint_vel.clone(),
    }
    for _ in range(3):
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
    freeze_actual = {
        "root_pos": env.robot.data.root_link_pos_w,
        "root_quat": env.robot.data.root_link_quat_w,
        "joint_pos": env.robot.data.joint_pos,
        "joint_vel": env.robot.data.joint_vel,
    }
    for key in freeze_reference:
        torch.testing.assert_close(
            freeze_actual[key], freeze_reference[key], rtol=0.0, atol=0.0
        )

    # Hold QC intentionally selects exactly two environments for ten physics steps.
    hold_env_ids = torch.tensor([0, 1], dtype=torch.long, device=env.device)
    episode_length_before_hold = env.episode_length_buf.clone()
    zero_velocity_target = torch.zeros_like(
        env.last_reset_joint_pos[hold_env_ids]
    )
    for _ in range(10):
        env.robot.set_joint_position_target(
            env.last_reset_joint_pos[hold_env_ids], env_ids=hold_env_ids
        )
        env.robot.set_joint_velocity_target(
            zero_velocity_target, env_ids=hold_env_ids
        )
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    hold_tensors = (
        env.robot.data.root_link_pos_w[hold_env_ids],
        env.robot.data.root_link_quat_w[hold_env_ids],
        env.robot.data.root_link_lin_vel_w[hold_env_ids],
        env.robot.data.root_link_ang_vel_w[hold_env_ids],
        env.robot.data.joint_pos[hold_env_ids],
        env.robot.data.joint_vel[hold_env_ids],
    )
    if not all(torch.isfinite(tensor).all() for tensor in hold_tensors):
        raise AssertionError("Ten-step two-environment Hold QC produced NaN or Inf.")
    if not torch.equal(env.episode_length_buf, episode_length_before_hold):
        raise AssertionError("Low-level Hold QC modified episode_length_buf.")

    # Restore an exact reset before exercising the normal policy/timeout path.
    env.reset_to_state_ids(all_env_ids, deterministic_state_ids)
    _assert_dataset_reset(env)

    actor_current, critic_current = env.compute_current_observations()
    _assert_finite_shape(actor_current, (NUM_ENVS, 75), "actor_current")
    _assert_finite_shape(critic_current, (NUM_ENVS, 78), "critic_current")
    actor_obs, infos = env.get_observations()
    _assert_finite_shape(actor_obs, (NUM_ENVS, ACTOR_OBS_DIM), "initial actor_obs")
    _assert_finite_shape(
        infos["observations"]["critic"],
        (NUM_ENVS, CRITIC_OBS_DIM),
        "initial critic_obs",
    )
    amp_obs = env.get_amp_obs_for_expert_trans()
    _assert_finite_shape(amp_obs, (NUM_ENVS, AMP_OBS_DIM), "initial amp_obs")

    zero_actions = torch.zeros(
        (NUM_ENVS, env.num_actions), device=env.device
    )
    _, _, done, _ = _check_step_output(env.step(zero_actions))
    if torch.any(done):
        raise AssertionError("Recovery terminated before its timeout.")
    torch.testing.assert_close(
        env.joint_pos_target,
        env.robot.data.default_joint_pos,
        rtol=0.0,
        atol=1.0e-7,
    )

    one_actions = torch.ones_like(zero_actions)
    _, _, done, _ = _check_step_output(env.step(one_actions))
    if torch.any(done):
        raise AssertionError("Recovery terminated before its timeout.")
    soft_limits = env.robot.data.soft_joint_pos_limits
    expected_one_target = torch.clamp(
        env.robot.data.default_joint_pos + 0.25,
        min=soft_limits[..., 0],
        max=soft_limits[..., 1],
    )
    torch.testing.assert_close(
        env.joint_pos_target, expected_one_target, rtol=0.0, atol=1.0e-7
    )
    if torch.any(env.joint_pos_target < soft_limits[..., 0]) or torch.any(
        env.joint_pos_target > soft_limits[..., 1]
    ):
        raise AssertionError("Joint position target escaped soft joint limits.")

    random_actions = torch.rand_like(zero_actions) * 2.0 - 1.0
    _, _, done, _ = _check_step_output(env.step(random_actions))
    if torch.any(done):
        raise AssertionError("Recovery terminated before its timeout.")
    amp_obs = env.get_amp_obs_for_expert_trans()
    _assert_finite_shape(amp_obs, (NUM_ENVS, AMP_OBS_DIM), "random-step amp_obs")

    actor_obs, _, done, infos = _check_step_output(env.step(one_actions))
    if not torch.all(done):
        raise AssertionError("All four smoke-test environments must time out.")
    if env.reset_env_ids.numel() != NUM_ENVS:
        raise AssertionError(
            f"Expected {NUM_ENVS} reset envs, got {env.reset_env_ids.numel()}."
        )
    if "terminal_amp_states" not in infos:
        raise AssertionError("Timeout infos are missing terminal_amp_states.")
    if not torch.equal(infos["time_outs"], done):
        raise AssertionError("Only timeout termination is allowed in Stage 4.1.")
    terminal_amp_states = infos["terminal_amp_states"]
    _assert_finite_shape(
        terminal_amp_states,
        (env.reset_env_ids.numel(), AMP_OBS_DIM),
        "terminal_amp_states",
    )
    pre_reset_reference = env.pre_reset_amp_obs[env.reset_env_ids]
    torch.testing.assert_close(
        terminal_amp_states, pre_reset_reference, rtol=0.0, atol=0.0
    )
    _assert_dataset_reset(env, env.reset_env_ids)

    actor_history = actor_obs.reshape(NUM_ENVS, 4, 75)
    critic_history = infos["observations"]["critic"].reshape(NUM_ENVS, 4, 78)
    torch.testing.assert_close(
        actor_history,
        actor_history[:, -1:].expand_as(actor_history),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        critic_history,
        critic_history[:, -1:].expand_as(critic_history),
        rtol=0.0,
        atol=0.0,
    )

    print(
        "G1 Recovery Stage 4.2c smoke test passed: "
        "physics_dt=0.005, decimation=4, step_dt=0.02, "
        "deterministic/freeze/hold reset QC passed, "
        "actor=300D, critic=312D, AMP=195D.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        run_smoke_test()
    finally:
        # This one-shot smoke test creates no writers or persistent simulator state.
        simulation_app.close(skip_cleanup=True)
