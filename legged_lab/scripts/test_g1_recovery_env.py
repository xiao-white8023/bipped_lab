#!/usr/bin/env python3
"""Smoke test for standalone G1 Recovery V1 reward and termination semantics."""

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

from legged_lab.envs.g1.g1_recovery_cfg import (
    G1RecoveryAgentCfg,
    G1RecoveryEnvCfg,
)
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv
from rsl_rl.runners import AmpOnPolicyRunner
from rsl_rl.utils import RecoveryAMPData


NUM_ENVS = 4
ACTOR_OBS_DIM = 300
CRITIC_OBS_DIM = 312
AMP_OBS_DIM = 195
AMP_TRANSITION_DIM = 390
EXPERT_TRANSITION_DT = 0.02
AMP_REWARD_COEF = 0.1
AMP_TASK_REWARD_LERP = 0.75
AMP_HIDDEN_DIMS = [1024, 512, 256]
AMP_PRELOAD_TRANSITIONS = 200000
REQUIRED_LOG_KEYS = {
    "RecoveryReward/height_raw",
    "RecoveryReward/height_scaled",
    "RecoveryReward/joint_acc",
    "RecoveryReward/joint_pos_limits",
    "RecoveryReward/action_rate",
    "RecoveryReward/self_collision",
    "RecoveryReward/task_total",
    "Recovery/height_ok_ratio",
    "Recovery/upright_ok_ratio",
    "Recovery/ready_now_ratio",
    "Recovery/success_count",
    "Recovery/success_ratio",
    "Recovery/timeout_count",
    "Recovery/success_hold_counter_mean",
    "Recovery/pelvis_height_mean",
    "Recovery/torso_up_z_mean",
    "RecoverySelfCollision/cost_mean",
    "RecoverySelfCollision/hit_env_ratio",
}


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
    if done.shape != (NUM_ENVS,) or done.dtype != torch.bool:
        raise AssertionError(f"done has invalid shape/dtype: {done.shape}, {done.dtype}.")
    if not torch.isfinite(done.float()).all():
        raise AssertionError("done contains invalid values.")
    if not REQUIRED_LOG_KEYS.issubset(infos.get("log", {})):
        missing = REQUIRED_LOG_KEYS - set(infos.get("log", {}))
        raise AssertionError(f"Recovery logs are missing keys: {sorted(missing)}")
    if not all(
        torch.isfinite(torch.as_tensor(value)).all()
        for value in infos["log"].values()
    ):
        raise AssertionError("Recovery logs contain NaN or Inf values.")
    return actor_obs, reward, done, infos


def _assert_discriminator_architecture(discriminator):
    if discriminator.input_dim != AMP_TRANSITION_DIM:
        raise AssertionError(
            "Recovery discriminator input must be 390D, got "
            f"{discriminator.input_dim}."
        )

    trunk_modules = list(discriminator.trunk)
    if len(trunk_modules) != 2 * len(AMP_HIDDEN_DIMS):
        raise AssertionError(
            "Recovery discriminator trunk must contain one Linear/ReLU pair "
            "per hidden dimension."
        )

    expected_input_dim = AMP_TRANSITION_DIM
    for layer_index, expected_output_dim in enumerate(AMP_HIDDEN_DIMS):
        linear = trunk_modules[2 * layer_index]
        activation = trunk_modules[2 * layer_index + 1]
        if not isinstance(linear, torch.nn.Linear):
            raise AssertionError(
                "Recovery discriminator trunk contains a non-Linear layer."
            )
        if not isinstance(activation, torch.nn.ReLU):
            raise AssertionError("Recovery discriminator must use ReLU activations.")
        if (
            linear.in_features != expected_input_dim
            or linear.out_features != expected_output_dim
        ):
            raise AssertionError(
                "Recovery discriminator Linear dimensions do not match "
                f"{[AMP_TRANSITION_DIM, *AMP_HIDDEN_DIMS]}."
            )
        expected_input_dim = expected_output_dim

    if (
        discriminator.amp_linear.in_features != AMP_HIDDEN_DIMS[-1]
        or discriminator.amp_linear.out_features != 1
    ):
        raise AssertionError(
            "Recovery discriminator output head must be Linear(256, 1)."
        )


def _assert_amp_reward_math(runner):
    """Verify pure style reward, Task/AMP mixing, range, and log semantics."""

    discriminator = runner.alg.discriminator
    saved_state = {
        name: tensor.detach().clone()
        for name, tensor in discriminator.state_dict().items()
    }
    saved_lerp = discriminator.task_reward_lerp
    was_training = discriminator.training

    state_dim = discriminator.input_dim // 2
    state = torch.zeros((1, state_dim), device=runner.device)
    next_state = torch.zeros_like(state)
    discriminator_values = (-4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 6.0)
    style_rewards = []

    try:
        with torch.no_grad():
            for parameter in discriminator.parameters():
                parameter.zero_()

        for index, discriminator_value in enumerate(discriminator_values):
            with torch.no_grad():
                discriminator.amp_linear.bias.fill_(discriminator_value)

            task_reward = torch.tensor(
                [-0.03 + 0.02 * index], device=runner.device
            )
            expected_discriminator_output = torch.tensor(
                [[discriminator_value]], device=runner.device
            )
            expected_style_reward = AMP_REWARD_COEF * torch.clamp(
                1.0
                - 0.25
                * torch.square(expected_discriminator_output.squeeze(-1) - 1.0),
                min=0.0,
            )

            discriminator.task_reward_lerp = 0.0
            actual_style_reward, actual_discriminator_output = (
                discriminator.predict_amp_reward(
                    state,
                    next_state,
                    task_reward,
                    normalizer=None,
                )
            )
            torch.testing.assert_close(
                actual_discriminator_output,
                expected_discriminator_output,
                rtol=0.0,
                atol=1.0e-7,
            )
            torch.testing.assert_close(
                actual_style_reward,
                expected_style_reward,
                rtol=1.0e-6,
                atol=1.0e-7,
            )

            discriminator.task_reward_lerp = AMP_TASK_REWARD_LERP
            actual_mixed_reward, actual_discriminator_output = (
                discriminator.predict_amp_reward(
                    state,
                    next_state,
                    task_reward,
                    normalizer=None,
                )
            )
            expected_mixed_reward = (
                (1.0 - AMP_TASK_REWARD_LERP) * expected_style_reward
                + AMP_TASK_REWARD_LERP * task_reward
            )
            torch.testing.assert_close(
                actual_mixed_reward,
                expected_mixed_reward,
                rtol=1.0e-6,
                atol=1.0e-7,
            )

            log_infos = {}
            runner._record_amp_reward_logs(
                log_infos,
                task_reward,
                actual_mixed_reward,
                actual_discriminator_output,
            )
            expected_logs = {
                "AMP/style_reward_mean": expected_style_reward.mean(),
                "AMP/task_reward_mean": task_reward.mean(),
                "AMP/mixed_reward_mean": expected_mixed_reward.mean(),
                "AMP/reward_mean": expected_mixed_reward.mean(),
            }
            for key, expected_value in expected_logs.items():
                torch.testing.assert_close(
                    log_infos["log"][key],
                    expected_value,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                )
            style_rewards.append(actual_style_reward)
    finally:
        discriminator.load_state_dict(saved_state)
        discriminator.task_reward_lerp = saved_lerp
        discriminator.train(was_training)

    style_rewards = torch.cat(style_rewards)
    if torch.any(style_rewards < 0.0) or torch.any(
        style_rewards > AMP_REWARD_COEF
    ):
        raise AssertionError("Recovery AMP style reward escaped [0, 0.1].")


def _write_default_standing(env):
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=env.device)
    env.scene.reset(env_ids)
    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()
    env.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    env.robot.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env.robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env.actor_obs_buffer.reset(env_ids)
    env.critic_obs_buffer.reset(env_ids)
    env.episode_length_buf[env_ids] = 0
    env.action[env_ids] = 0.0
    env.last_action[env_ids] = 0.0
    env.success_hold_counter[env_ids] = 0
    env.success_buf[env_ids] = False
    env.terminated_buf[env_ids] = False
    env.self_collision_hit_substeps[env_ids] = False


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
    if env.max_episode_length_s != 6.0:
        raise AssertionError(
            f"episode length must be 6.0 s, got {env.max_episode_length_s}."
        )
    if env.max_episode_length != 300:
        raise AssertionError(
            f"6.0 s must produce 300 policy steps, got {env.max_episode_length}."
        )
    if env.success_hold_steps != 25:
        raise AssertionError(
            f"0.5 s must produce 25 hold steps, got {env.success_hold_steps}."
        )
    if len(env.self_collision_sensors) != len(env.robot.body_names):
        raise AssertionError(
            "Recovery must create one filtered self-collision sensor per body."
        )
    for sensor in env.self_collision_sensors:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix.shape != (
            NUM_ENVS,
            1,
            len(env.robot.body_names),
            3,
        ):
            raise AssertionError(
                "Self-collision force matrix has invalid shape "
                f"{tuple(force_matrix.shape)}."
            )
        if any("ground" in path.lower() for path in sensor.cfg.filter_prim_paths_expr):
            raise AssertionError("Ground leaked into a self-collision filter.")
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

    # Fallen dataset resets always begin with a fresh hold counter and cannot
    # terminate immediately even if an individual sample is momentarily ready.
    env._update_recovery_state_metrics()
    if torch.any(env.success_buf) or torch.any(env.success_hold_counter):
        raise AssertionError("A fallen dataset reset started as successful.")
    reset_now, _ = env.check_reset()
    if torch.any(reset_now) or torch.any(env.success_buf):
        raise AssertionError("A fallen reset succeeded without a 25-step hold.")

    synthetic_hits = torch.zeros((3, 4), dtype=torch.bool, device=env.device)
    synthetic_hits[1, 2] = True
    synthetic_hits[2] = True
    synthetic_cost = env.self_collision_cost_from_hits(synthetic_hits)
    torch.testing.assert_close(
        synthetic_cost,
        torch.tensor([0.0, 1.0, 4.0], device=env.device),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.tensor(
            env.step_dt * env.cfg.reward.root_height_exp.weight,
            device=env.device,
        ),
        torch.tensor(0.07, device=env.device),
        rtol=0.0,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        env.step_dt
        * env.cfg.reward.self_collisions.weight
        * synthetic_cost[-1],
        torch.tensor(-0.008, device=env.device),
        rtol=0.0,
        atol=1.0e-8,
    )

    # The exact default standing pose satisfies both gates.  Success occurs on
    # the 25th consecutive policy evaluation, never earlier.
    _write_default_standing(env)
    env._update_recovery_state_metrics()
    if not torch.all(env.height_ok):
        raise AssertionError("Default standing state failed the height gate.")
    if not torch.all(env.upright_ok):
        raise AssertionError("Default standing state failed the torso upright gate.")
    for _ in range(env.success_hold_steps - 1):
        reset_now, time_out_now = env.check_reset()
        if torch.any(reset_now) or torch.any(time_out_now):
            raise AssertionError("Standing state succeeded before 25 hold steps.")
    reset_now, time_out_now = env.check_reset()
    if not torch.all(reset_now) or not torch.all(env.success_buf):
        raise AssertionError("Standing state did not succeed on hold step 25.")
    if torch.any(time_out_now) or not torch.all(env.terminated_buf):
        raise AssertionError("Recovery success must be a true termination.")

    # Exercise the real step/reset path from hold step 24 and verify success
    # terminal AMP states were captured before the new fallen reset.
    _write_default_standing(env)
    env._update_recovery_state_metrics()
    env.success_hold_counter.fill_(env.success_hold_steps - 1)
    zero_actions = torch.zeros(
        (NUM_ENVS, env.num_actions), device=env.device
    )
    _, _, success_done, success_infos = _check_step_output(
        env.step(zero_actions)
    )
    if not torch.all(success_done):
        raise AssertionError("Held standing state did not terminate as success.")
    if not torch.all(success_infos["terminated"]):
        raise AssertionError("Success infos did not mark terminated=True.")
    if torch.any(success_infos["time_outs"]):
        raise AssertionError("Success was incorrectly marked as timeout truncation.")
    torch.testing.assert_close(
        success_infos["terminal_amp_states"],
        env.pre_reset_amp_obs[env.reset_env_ids],
        rtol=0.0,
        atol=0.0,
    )
    _assert_dataset_reset(env, env.reset_env_ids)

    # Return to deterministic fallen states before action/reward checks.
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

    _, _, done, _ = _check_step_output(env.step(zero_actions))
    if torch.any(done):
        raise AssertionError("Recovery terminated before its timeout.")
    torch.testing.assert_close(
        env.joint_pos_target,
        env.robot.data.default_joint_pos,
        rtol=0.0,
        atol=1.0e-7,
    )
    observed_self_collision_cost = env.self_collision_cost_from_hits(
        env.self_collision_hit_substeps
    )
    torch.testing.assert_close(
        env.self_collision_cost,
        observed_self_collision_cost,
        rtol=0.0,
        atol=0.0,
    )
    if torch.any(env.self_collision_cost < 0.0) or torch.any(
        env.self_collision_cost > env.cfg.sim.decimation
    ):
        raise AssertionError("Observed self-collision cost escaped [0, 4].")
    h_default = env.robot.data.default_root_state[:, 2]
    expected_height_raw = torch.exp(
        -torch.square(h_default - env.robot.data.root_link_pos_w[:, 2]) / 0.09
    )
    expected_height_reward = env.step_dt * 3.5 * expected_height_raw
    expected_joint_acc_reward = (
        env.step_dt * -2.5e-7 * env.joint_acc_cost
    )
    expected_joint_pos_limits_reward = (
        env.step_dt * -10.0 * env.joint_pos_limits_cost
    )
    expected_action_rate_reward = (
        env.step_dt * -0.01 * env.action_rate_cost
    )
    expected_self_collision_reward = (
        env.step_dt * -0.1 * env.self_collision_cost
    )
    expected_task_reward = (
        expected_height_reward
        + expected_joint_acc_reward
        + expected_joint_pos_limits_reward
        + expected_action_rate_reward
        + expected_self_collision_reward
    )
    torch.testing.assert_close(
        env.height_reward_raw, expected_height_raw, rtol=1.0e-6, atol=1.0e-7
    )
    contribution_pairs = (
        (env.height_reward_scaled, expected_height_reward),
        (env.joint_acc_reward, expected_joint_acc_reward),
        (env.joint_pos_limits_reward, expected_joint_pos_limits_reward),
        (env.action_rate_reward, expected_action_rate_reward),
        (env.self_collision_reward, expected_self_collision_reward),
    )
    for actual_contribution, expected_contribution in contribution_pairs:
        torch.testing.assert_close(
            actual_contribution,
            expected_contribution,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    torch.testing.assert_close(
        env.task_reward, expected_task_reward, rtol=1.0e-6, atol=1.0e-7
    )
    torch.testing.assert_close(
        env.task_reward,
        env.height_reward_scaled
        + env.joint_acc_reward
        + env.joint_pos_limits_reward
        + env.action_rate_reward
        + env.self_collision_reward,
        rtol=1.0e-6,
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

    env.episode_length_buf.fill_(env.max_episode_length - 1)
    env.success_hold_counter.zero_()
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
        raise AssertionError("Horizon exits must be timeout truncations.")
    if torch.any(infos["terminated"]):
        raise AssertionError("Timeout truncation was incorrectly marked terminated.")
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

    # Initialize the real AMP runner without a preload cache to verify that the
    # standalone config selects RecoveryAMPData and applies both reward knobs.
    agent_cfg = G1RecoveryAgentCfg()
    if not agent_cfg.amp_preload_transitions:
        raise AssertionError("Recovery AMP transition preload must remain enabled.")
    if agent_cfg.amp_num_preload_transitions != AMP_PRELOAD_TRANSITIONS:
        raise AssertionError("Recovery AMP preload count must remain 200000.")
    if agent_cfg.amp_discr_hidden_dims != AMP_HIDDEN_DIMS:
        raise AssertionError(
            "Recovery discriminator hidden dimensions must remain "
            f"{AMP_HIDDEN_DIMS}."
        )
    if agent_cfg.min_normalized_std != [0.05] * 23:
        raise AssertionError("Recovery min_normalized_std must remain 23 x 0.05.")

    train_cfg = agent_cfg.to_dict()
    train_cfg["amp_preload_transitions"] = False
    train_cfg["amp_num_preload_transitions"] = 0
    runner = AmpOnPolicyRunner(
        env,
        train_cfg,
        log_dir=None,
        device=env.device,
    )
    if not isinstance(runner.alg.amp_data, RecoveryAMPData):
        raise AssertionError("Recovery runner did not select RecoveryAMPData.")
    if runner.alg.amp_data.transition_dt != EXPERT_TRANSITION_DT:
        raise AssertionError("Recovery runner AMP transition_dt is not 0.02 s.")
    if runner.alg.discriminator.amp_reward_coef != AMP_REWARD_COEF:
        raise AssertionError("Recovery amp_reward_coef is not 0.1.")
    if runner.alg.discriminator.task_reward_lerp != AMP_TASK_REWARD_LERP:
        raise AssertionError("Recovery amp_task_reward_lerp is not 0.75.")
    _assert_discriminator_architecture(runner.alg.discriminator)
    _assert_amp_reward_math(runner)

    print(
        "G1 Recovery V1 smoke test passed: "
        "physics_dt=0.005, decimation=4, step_dt=0.02, "
        "episode=300 steps, success_hold=25 steps, "
        "dt-scaled task reward and AMP 0.1/0.75 mixing checks passed, "
        "success/timeout/self-collision/terminal-AMP checks passed, "
        "actor=300D, critic=312D, AMP=195D, transition=390D.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        run_smoke_test()
    finally:
        # This one-shot smoke test creates no writers or persistent simulator state.
        simulation_app.close(skip_cleanup=True)
