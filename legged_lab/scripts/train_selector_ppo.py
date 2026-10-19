"""直接构造 G1 高层环境并训练离散 PPO，不经过旧任务注册或训练入口。

示例：python legged_lab/scripts/train_selector_ppo.py --headless --device cuda:0
恢复：追加 --resume /absolute/path/model_100.pt；max-iterations 表示本次新增更新。
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None, help="本次新增的 PPO 更新次数；默认沿用 cfg。")
    parser.add_argument("--num-steps-per-env", type=int, default=None, help="每次更新的高层 rollout 长度；默认 24。")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--logger", choices=("tensorboard", "jsonl"), default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", type=Path, default=None, help="新的空输出目录；默认 logs/g1_selector_ppo/时间戳。")
    parser.add_argument("--resume", type=Path, default=None, help="高层 PPO checkpoint，含优化器和已完成迭代数。")
    parser.add_argument("--low-level-checkpoint", type=Path, default=None, help="冻结 RENet checkpoint 覆盖路径。")
    parser.add_argument("--reference-policy", type=Path, default=None, help="冻结 RENet 的参考 policy.pt 覆盖路径。")
    parser.add_argument("--small-terrain", action="store_true", help="调试时只生成 2×3 地形块，仍保持固定等级。")
    parser.add_argument("--disable-faults", action="store_true", help="显式关闭统一感知故障；默认保持 cfg 设置。")
    return parser


def validate_args(args, parser):
    for name in ("num_envs", "max_iterations", "num_steps_per_env", "save_interval"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} 必须为正整数。")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("seed 必须在 [0,2**63) 内。")
    for name in ("resume", "low_level_checkpoint", "reference_policy"):
        value = getattr(args, name)
        if value is not None and not value.expanduser().is_file():
            parser.error(f"{name.replace('_', '-')} 文件不存在：{value}")


def parse_args(argv=None):
    # AppLauncher 的参数注册不会启动 Kit；环境和配置导入必须放在启动之后。
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    validate_args(args, parser)
    args.enable_cameras = True
    return args


def make_configs(args):
    from legged_lab.envs.g1.g1_selector_cfg import (
        G1SelectorAgentCfg, G1SelectorEnvCfg, configure_selector_agent,
    )

    env_cfg, agent_cfg = G1SelectorEnvCfg(), G1SelectorAgentCfg()
    env_cfg.device = agent_cfg.device = args.device
    for name in ("seed", "max_iterations", "num_steps_per_env", "save_interval", "logger", "run_name"):
        value = getattr(args, name)
        if value is not None:
            setattr(agent_cfg, name, value)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = agent_cfg.seed
    if args.low_level_checkpoint is not None:
        env_cfg.low_level.checkpoint_path = str(args.low_level_checkpoint.expanduser().resolve())
    if args.reference_policy is not None:
        env_cfg.low_level.reference_policy_path = str(args.reference_policy.expanduser().resolve())
    if args.disable_faults:
        env_cfg.faults.enable = False
    if args.small_terrain:
        generator = copy.deepcopy(env_cfg.scene.terrain_generator)
        if generator is None:
            raise ValueError("small-terrain requires the configured terrain generator.")
        generator.num_rows, generator.num_cols = 2, 3
        generator.seed, generator.use_cache = agent_cfg.seed, False
        env_cfg.scene.terrain_generator = generator
        level = env_cfg.scene.max_init_terrain_level
        env_cfg.scene.max_init_terrain_level = 1 if level is None else min(level, 1)
    agent_cfg.resume = args.resume is not None
    configure_selector_agent(env_cfg, agent_cfg)
    for path in (env_cfg.low_level.checkpoint_path, env_cfg.low_level.reference_policy_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Frozen RENet file does not exist: {path}")
    return env_cfg, agent_cfg


def json_value(value):
    """保存完整配置及审计；函数记录为限定名称，非有限诊断值记录为字符串。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if callable(value):
        return f"{value.__module__}:{getattr(value, '__qualname__', type(value).__name__)}"
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    paths = [
        "legged_lab/envs/g1/g1_selector_cfg.py", "legged_lab/envs/g1/g1_selector_env.py",
        "legged_lab/scripts/train_selector_ppo.py", "rsl_rl/rsl_rl/modules/selector_actor_critic.py",
        "rsl_rl/rsl_rl/storage/selector_rollout_storage.py", "rsl_rl/rsl_rl/algorithms/discrete_ppo.py",
        "rsl_rl/rsl_rl/runners/selector_on_policy_runner.py",
    ]
    paths.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "legged_lab/high_level").glob("*.py")))
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def make_log_dir(args):
    if args.log_dir is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        if args.run_name:
            # run-name 只作为目录名后缀，不能覆盖日志根目录。
            name += "_" + Path(args.run_name).name
        path = ROOT / "logs/g1_selector_ppo" / name
    else:
        path = args.log_dir.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new empty log directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    args = parse_args(argv)
    log_dir = make_log_dir(args)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    # AppLauncher 会改写传入 Namespace，先独立保存原始启动参数。
    report = {"status": "starting", "log_dir": str(log_dir), "arguments": copy.deepcopy(vars(args)), "command": command,
              "source_sha256": source_hashes(), "resume_environment_state": False}
    simulation_app = runner = None
    previous_sigint = None
    try:
        import torch
        from isaaclab.app import AppLauncher

        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run this simulation training entry on a host with an accessible NVIDIA GPU.")
        simulation_app = AppLauncher(args).app
        from legged_lab.envs.g1.g1_selector_env import G1SelectorEnv
        from rsl_rl.runners.selector_on_policy_runner import SelectorOnPolicyRunner

        env_cfg, agent_cfg = make_configs(args)
        report.update(seed=agent_cfg.seed, num_envs=env_cfg.scene.num_envs,
                      requested_updates=agent_cfg.max_iterations, faults_enabled=env_cfg.faults.enable)
        write_json(log_dir / "params/env_cfg.json", env_cfg)
        write_json(log_dir / "params/agent_cfg.json", agent_cfg)
        env = G1SelectorEnv(env_cfg, args.headless)
        runner = SelectorOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if args.resume is not None:
            runner.load(args.resume.expanduser().resolve())
        report.update(start_iteration=runner.current_learning_iteration, frozen_policy=env.frozen_policy_metadata,
                      observation_contract=runner.signature)
        write_json(log_dir / "run.json", report)
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            print("[Selector PPO] 收到停止请求，将完成当前 PPO 更新并保存。", flush=True)

        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        report["last_metrics"] = runner.learn(
            agent_cfg.max_iterations,
            should_continue=lambda: not stop_requested and simulation_app.is_running(),
        )
        assert not env.frozen_policy.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in env.frozen_policy.parameters())
        report.update(status="stopped" if runner.stopped else "completed", frozen_policy_remained_frozen=True,
                      physics_steps=int(env.sim_step_counter))
        write_json(log_dir / "fault_audit.json", env.faults.audit())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if runner is not None:
            report.update(completed_iteration=runner.current_learning_iteration, total_timesteps=runner.total_timesteps,
                          total_low_level_steps=runner.total_low_level_steps, checkpoint=runner.last_checkpoint)
            runner.close()
        write_json(log_dir / "report.json", report)
        print(json.dumps({"status": report["status"], "report": str(log_dir / "report.json")}), flush=True)
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
