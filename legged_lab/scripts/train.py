import argparse
import os
import sys

# --- 修复 1: 路径置顶 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
local_rsl_rl_path = os.path.abspath(os.path.join(current_dir, "../../rsl_rl"))
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

from isaaclab.app import AppLauncher


def parse_lp_height_source(value):
    if value != "pgtt_full_atten":
        raise argparse.ArgumentTypeError("g1_renet supports only pgtt_full_atten")
    return value


# --- 修复 2: 基础参数解析 (避开 __init__.py 陷阱) ---
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--lp_height_source", type=parse_lp_height_source,
                    metavar="{pgtt_full_atten}", default=None,
                    help="g1_renet LP input source (default from cfg)")
parser.add_argument(
    "--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", default=None,
    help="Direct checkpoint path; takes precedence over --load_run/--checkpoint and implies resume.",
)

# 仅添加 AppLauncher 引擎启动参数并解析
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is not None and ("sensor" in args_cli.task or args_cli.task in ("g1_depth", "g1_selector")):
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
from rsl_rl.runners import AmpOnPolicyRunner, ConstrainedOnPolicyRunner, OnPolicyRunner, MoeAmpOnPolicyRunner, DWAQAmpOnPolicyRunner, MoeOnPolicyRunner,G1OnPolicyRunner,FilmOnPolicyRunner, RENetAmpOnPolicyRunner
from rsl_rl.runners.g1_student_on_policy_runner import G1StudentOnPolicyRunner
from rsl_rl.runners.selector_on_policy_runner import SelectorOnPolicyRunner

from legged_lab.envs import * # noqa:F401, F403
from legged_lab.envs.g1.RENet_cfg import configure_agent_for_lp_source
from legged_lab.envs.g1.g1_selector_cfg import configure_selector_agent
from rsl_rl.runners import DepthAmpOnPolicyRunner
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner | DepthAmpOnPolicyRunner | SelectorOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if env_class_name == "g1_selector" and args_cli.distributed:
        raise ValueError("g1_selector currently supports single-process PPO training only.")
    if env_class_name in ("g1_depth", "g1_selector"):
        agent_cfg.device = args_cli.device
        env_cfg.device = args_cli.device
    if args_cli.lp_height_source is not None:
        if not hasattr(env_cfg, "lp_height"):
            raise ValueError("--lp_height_source is only supported by g1_renet")
        env_cfg.renet.lp_height_source = args_cli.lp_height_source
    if env_class_name == "g1_selector":
        # 高层独立同步观测尺寸、控制频率和 seed，不能套用底层 RENet 网络配置。
        configure_selector_agent(env_cfg, agent_cfg)
    elif hasattr(env_cfg, "lp_height"): # 判断lp_height是否存在于env_cfg中
        configure_agent_for_lp_source(env_cfg, agent_cfg)
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
    runner_class: FilmOnPolicyRunner|OnPolicyRunner | ConstrainedOnPolicyRunner | AmpOnPolicyRunner | RENetAmpOnPolicyRunner | MoeAmpOnPolicyRunner | DWAQAmpOnPolicyRunner | MoeOnPolicyRunner|G1OnPolicyRunner|G1StudentOnPolicyRunner = eval(agent_cfg.runner_class_name)
    train_cfg = agent_cfg.to_dict()
    if env_class_name == "g1_recovery":
        required_recovery_amp_fields = (
            "amp_data_class_name",
            "amp_motion_dir",
            "amp_transition_dt",
            "amp_expected_fps",
            "amp_preload_transitions",
            "amp_num_preload_transitions",
            "amp_reward_coef",
            "amp_task_reward_lerp",
            "amp_discr_hidden_dims",
            "min_normalized_std",
        )
        missing = [
            key for key in required_recovery_amp_fields if key not in train_cfg
        ]
        if missing:
            raise RuntimeError(
                "g1_recovery agent config is missing required Recovery AMP "
                f"fields after to_dict(): {missing}."
            )

    runner = runner_class(
        env,
        train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    if args_cli.checkpoint_path is not None:
        resume_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint_path))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")
        print(f"[INFO]: Loading direct model checkpoint from: {resume_path}")
        runner.load(resume_path)
    elif agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if hasattr(env, "lp_observation_contract"):
        dump_yaml(os.path.join(log_dir, "params", "lp_observation_contract.yaml"), env.lp_observation_contract)
    if hasattr(env, "depth_observation_contract"):
        dump_yaml(os.path.join(log_dir, "params", "depth_observation_contract.yaml"), env.depth_observation_contract)
    if hasattr(env, "foot_edge_monitor"):
        env.foot_edge_monitor.save_metadata(os.path.join(log_dir, "params", "foot_edge"))

    # 高层 episode、底层有效步数和感知时钟必须保持一致，不能单独随机化 episode 长度。
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=env_class_name != "g1_selector",
    )


if __name__ == "__main__":
    train()
    simulation_app.close(skip_cleanup=args_cli.task == "g1_depth" and args_cli.headless)
