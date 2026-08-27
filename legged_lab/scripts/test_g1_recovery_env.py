#!/usr/bin/env python3
"""Stage-4.1 smoke test for the standalone G1 Recovery environment."""

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
    description="Smoke-test the standalone Stage-4.1 G1 Recovery environment."
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


def run_smoke_test():
    cfg = G1RecoveryEnvCfg()
    cfg.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.max_episode_length_s = 0.08
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
    post_reset_amp = env.get_amp_obs_for_expert_trans()[env.reset_env_ids]
    if torch.allclose(terminal_amp_states, post_reset_amp):
        raise AssertionError(
            "terminal_amp_states unexpectedly match post-reset standing AMP states."
        )

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
        "G1 Recovery Stage 4.1 smoke test passed: "
        "physics_dt=0.005, decimation=4, step_dt=0.02, "
        "actor=300D, critic=312D, AMP=195D.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        run_smoke_test()
    finally:
        # This one-shot smoke test creates no writers or persistent simulator state.
        simulation_app.close(skip_cleanup=True)
