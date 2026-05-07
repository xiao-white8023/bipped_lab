

import argparse
import os
import sys

# --- 修复 1: 路径置顶 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
local_rsl_rl_path = os.path.abspath(os.path.join(current_dir, "../../rsl_rl"))
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

from isaaclab.app import AppLauncher

# --- 修复 2: 基础参数解析 (避开 __init__.py 陷阱) ---
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# 仅添加 AppLauncher 引擎启动参数并解析
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is not None and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# ==============================================================================
# 启动引擎！(此时没有任何底层依赖被触发)
# ==============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# App 启动完成！现在可以安全触发所有的 __init__.py 了
# ==============================================================================

# 导入 RL 相关的参数配置，并进行二次解析
import legged_lab.utils.cli_args as cli_args
cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from datetime import datetime
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner,MoeAmpOnPolicyRunner
from legged_lab.envs import * # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner_class: OnPolicyRunner | AmpOnPolicyRunner |MoeAmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    simulation_app.close()