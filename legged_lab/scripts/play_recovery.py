#!/usr/bin/env python3
"""Play and export the standalone G1 Recovery policy in IsaacLab.

The recovery task is intentionally different from the locomotion tasks:

* it controls every joint of the 23-DoF G1 model;
* it has no velocity or height command;
* reset states come from the validated single-state Recovery NPZ dataset;
* the actor input is a 10-frame history of the 75-D proprioceptive state.

Examples
--------
Play random reset states with the newest checkpoint::

    python legged_lab/scripts/play_recovery.py --num_envs 16

Replay one reset state (1-based index) after every episode::

    python legged_lab/scripts/play_recovery.py --num_envs 1 --state 12

Skip TorchScript/ONNX export::

    python legged_lab/scripts/play_recovery.py --no_export
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="The function 'quat_rotate_inverse' will be deprecated",
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LOCAL_RSL_RL_PATH = REPO_ROOT / "rsl_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_RSL_RL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL_PATH))


# Isaac/Omniverse must be launched before importing the rest of IsaacLab.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Play a trained standalone g1_recovery policy."
)
parser.add_argument(
    "--task",
    type=str,
    default="g1_recovery",
    help="Registered task name. This script only supports g1_recovery.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Number of parallel Recovery environments.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--state",
    type=int,
    default=None,
    help=(
        "Optional 1-based Recovery reset-state index. When set, all completed "
        "episodes are reset to this same state."
    ),
)
parser.add_argument(
    "--no_export",
    action="store_true",
    help="Do not export policy.pt and policy.onnx after loading the checkpoint.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=50,
    help="Print Recovery metrics every N policy steps; 0 disables printing.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional policy-step limit, mainly useful for headless smoke tests.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task != "g1_recovery":
    parser.error("play_recovery.py only supports --task g1_recovery.")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive.")
if args_cli.state is not None and args_cli.state <= 0:
    parser.error("--state is 1-based and must be positive.")
if args_cli.print_every < 0:
    parser.error("--print_every must be non-negative.")
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import legged_lab.utils.cli_args as cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()


import torch  # noqa: E402
from export import export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import AmpOnPolicyRunner  # noqa: E402


def _set_fixed_reset_state(env, state_index: int, env_ids=None) -> None:
    """Reset selected environments to a user-facing 1-based dataset index."""
    state_id = state_index - 1
    num_states = env.reset_state_loader.num_states
    if state_id >= num_states:
        raise IndexError(
            f"--state {state_index} requested, but the dataset contains "
            f"{num_states} states."
        )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state_ids = torch.full_like(env_ids, state_id)
    env.reset_to_state_ids(env_ids, state_ids)


def _build_runner(env, agent_cfg, log_dir: str):
    if agent_cfg.runner_class_name != "AmpOnPolicyRunner":
        raise RuntimeError(
            "g1_recovery must use AmpOnPolicyRunner, got "
            f"{agent_cfg.runner_class_name!r}."
        )

    # Expert data is only needed to construct the AMP runner for checkpoint
    # compatibility. Playback never samples expert transitions, so avoid the
    # expensive 200k-transition preload used during training.
    train_cfg = agent_cfg.to_dict()
    train_cfg["amp_preload_transitions"] = False
    train_cfg["amp_num_preload_transitions"] = 0
    return AmpOnPolicyRunner(
        env,
        train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
    )


def _print_metrics(step: int, env, extras: dict) -> None:
    log = extras.get("log", {})

    def scalar(key: str) -> float:
        value = log.get(key, 0.0)
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu())
        return float(value)

    print(
        f"[INFO] step={step}, "
        f"pelvis_h={scalar('Recovery/pelvis_height_mean'):.3f}, "
        f"torso_up_z={scalar('Recovery/torso_up_z_mean'):.3f}, "
        f"ready={scalar('Recovery/ready_now_ratio'):.3f}, "
        f"episode_success={scalar('RecoveryEpisode/success_rate'):.3f}, "
        f"completed={int(scalar('RecoveryEpisode/completed_total'))}, "
        f"reset_state={int(env.last_reset_state_ids[0].item()) + 1}"
    )


def play() -> None:
    """Load the configured Recovery checkpoint and run IsaacLab playback."""
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    # Preserve the formal 6 s Recovery episode and flat terrain. Both are part
    # of the environment contract and are validated by G1RecoveryEnv.
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    print(f"[INFO] Recovery joint order ({env.num_actions} DoF):")
    print(tuple(env.robot.joint_names))
    print(
        "[INFO] Actor observation: "
        f"{env.ACTOR_CURRENT_DIM} x {env_cfg.observation.history_length} "
        f"= {env.ACTOR_CURRENT_DIM * env_cfg.observation.history_length}"
    )

    log_root_path = REPO_ROOT / "logs" / agent_cfg.experiment_name
    print(f"[INFO] Loading experiment from: {log_root_path}")
    resume_path = get_checkpoint_path(
        str(log_root_path),
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
    )
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    runner = _build_runner(env, agent_cfg, log_dir)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.no_export:
        export_dir = os.path.join(log_dir, "exported")
        export_policy_as_jit(
            runner.alg.policy,
            runner.obs_normalizer,
            path=export_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            runner.alg.policy,
            normalizer=runner.obs_normalizer,
            path=export_dir,
            filename="policy.onnx",
        )

    if args_cli.state is not None:
        _set_fixed_reset_state(env, args_cli.state)

    obs, _ = env.get_observations()
    expected_obs_dim = env.ACTOR_CURRENT_DIM * env_cfg.observation.history_length
    if obs.shape != (env.num_envs, expected_obs_dim):
        raise RuntimeError(
            f"Recovery actor observation must have shape "
            f"({env.num_envs}, {expected_obs_dim}), got {tuple(obs.shape)}."
        )

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            if actions.shape != (env.num_envs, env.num_actions):
                raise RuntimeError(
                    "Recovery policy action shape mismatch: expected "
                    f"({env.num_envs}, {env.num_actions}), got {tuple(actions.shape)}."
                )
            obs, _, dones, extras = env.step(actions)

            # G1RecoveryEnv has already applied a random dataset reset before
            # returning from step(). Replace it and refresh the observation
            # history when a deterministic state was requested.
            if args_cli.state is not None and torch.any(dones):
                done_ids = dones.nonzero(as_tuple=False).flatten()
                _set_fixed_reset_state(env, args_cli.state, done_ids)
                obs, _ = env.get_observations()

        step += 1
        if args_cli.print_every and step % args_cli.print_every == 0:
            _print_metrics(step, env, extras)
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            break


if __name__ == "__main__":
    try:
        play()
    finally:
        simulation_app.close()
