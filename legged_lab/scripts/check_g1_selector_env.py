"""用真实 Isaac Sim 和冻结 checkpoint 检查高层选择器环境，不创建 PPO 或 runner。

默认 4 个环境、12 个高层步，依次执行固定 VP、固定 LP 和交替模式。
缩小地形生成规模以降低检查成本；报告完整配置、覆盖参数、种子和故障统计。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12, help="高层步数，分为 VP、LP、交替三段。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/g1_selector_env")
    parser.add_argument("--disable-faults", action="store_true", help="只关闭统一感知故障管线。")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 3:
        parser.error("num-envs 必须为正数，steps 至少为 3，以覆盖三种模式序列。")
    args.enable_cameras = True
    return args


def json_value(value):
    """将配置、张量和非有限诊断时间转成可移植的 JSON 值。"""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', type(value).__name__)}"
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    return repr(value)


def make_config(args):
    from legged_lab.envs.g1.g1_selector_cfg import G1SelectorEnvCfg

    cfg = G1SelectorEnvCfg()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = args.seed
    generator = copy.deepcopy(cfg.scene.terrain_generator)
    generator.num_rows, generator.num_cols = 2, 3
    generator.seed = args.seed
    generator.use_cache = False
    cfg.scene.terrain_generator = generator
    cfg.scene.max_init_terrain_level = 1
    cfg.faults.enable = not args.disable_faults
    cfg.resolve_observations()
    overrides = {
        "device": args.device, "seed": args.seed, "scene.seed": args.seed,
        "scene.num_envs": args.num_envs, "scene.max_init_terrain_level": 1,
        "scene.terrain_generator.num_rows": 2, "scene.terrain_generator.num_cols": 3,
        "scene.terrain_generator.seed": args.seed, "scene.terrain_generator.use_cache": False,
        "faults.enable": not args.disable_faults,
    }
    return cfg, overrides


def check_observations(env, obs, info):
    import torch

    assert obs.shape == (env.num_envs, 3466), obs.shape
    critic = info["observations"]["critic"]
    assert critic.shape == (env.num_envs, 3660), critic.shape
    assert env.low_level_obs.shape == (env.num_envs, 2120), env.low_level_obs.shape
    assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
    assert torch.isfinite(env.low_level_obs).all()


def check_read_only(env):
    """读取前后比较缓存、故障审计和仿真计数，读取不得产生新帧。"""
    def snapshot():
        return json_value({
            "actor": env.actor_obs_buffer.buffer,
            "actor_pushes": env.actor_obs_buffer.num_pushes,
            "actor_observation": env.actor_obs,
            "critic_observation": env.critic_obs,
            "low_level": env.low_level_obs,
            "action_history": env.action_buffer._circular_buffer.buffer,
            "captures": env.sensor_capture_count,
            "physics_steps": env.sim_step_counter,
            "fault_audit": env.faults.audit(),
        })

    before = snapshot()
    obs, info = env.get_observations()
    check_observations(env, obs, info)
    for _ in range(4):
        env.get_observations()
    assert snapshot() == before, "get_observations() 改变了缓存、故障或仿真状态。"


def mode_at(index, steps):
    if index < steps // 3:
        return "fixed_vp", 0
    if index < 2 * steps // 3:
        return "fixed_lp", 1
    return "alternating", (index - 2 * steps // 3) % 2


def run_check(args, simulation_app, report):
    import torch
    from isaaclab.utils.io import dump_yaml
    from legged_lab.envs.g1.g1_selector_env import G1SelectorEnv

    cfg, overrides = make_config(args)
    report.update(config=json_value(cfg), overrides=overrides)
    dump_yaml(str(args.output / "env_cfg.yaml"), cfg)
    G1SelectorEnv.seed(args.seed)
    env = G1SelectorEnv(cfg, args.headless)
    report["frozen_policy"] = json_value(env.frozen_policy_metadata)
    assert not env.frozen_policy.training
    assert all(not module.training for module in env.frozen_policy.modules())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in env.frozen_policy.parameters())
    assert env.num_actions == 1 and env.low_level_num_actions == 23
    assert math.isclose(env.step_dt, .2) and math.isclose(env.low_level_dt, .02)
    assert math.isclose(env.physics_dt, .005)
    terrain_levels = env.scene.terrain.terrain_levels.clone()
    terrain_origins = env.scene.env_origins.clone()

    def check_terrain():
        assert torch.equal(env.scene.terrain.terrain_levels, terrain_levels), "地形等级发生变化。"
        assert torch.equal(env.scene.env_origins, terrain_origins), "环境地形原点发生变化。"

    report["terrain"] = {
        "generator_curriculum": cfg.scene.terrain_generator.curriculum,
        "environment_curriculum": cfg.selector.terrain_curriculum,
        "initial_levels": json_value(terrain_levels), "initial_origins": json_value(terrain_origins),
        "coverage_note": "缩小地形数量，不保证覆盖全部地形；不评估摔倒率。",
    }
    report["steps"] = []
    report["checks"] = {"frozen_eval_no_grad": "passed"}
    check_read_only(env)
    with torch.no_grad():
        for index in range(args.steps):
            assert simulation_app.is_running(), "仿真应用在完成检查前关闭。"
            phase, mode = mode_at(index, args.steps)
            action = torch.full((env.num_envs,), mode, dtype=torch.long, device=env.device)
            capture_before = env.sensor_capture_count.clone()
            physics_before = int(env.sim_step_counter)
            obs, reward, done, info = env.step(action)
            check_observations(env, obs, info)
            assert reward.shape == done.shape == (env.num_envs,)
            assert done.dtype == torch.bool and torch.isfinite(reward).all()
            low_steps = info["low_level_steps"]
            assert low_steps.shape == (env.num_envs,)
            assert ((low_steps >= 1) & (low_steps <= 10)).all()
            assert (low_steps[~done] == 10).all()
            physics_delta = int(env.sim_step_counter) - physics_before
            assert physics_delta == 40, physics_delta
            capture_delta = env.sensor_capture_count - capture_before
            assert (capture_delta[~done] == 2).all(), capture_delta
            terminal_ids = info["terminal_env_ids"]
            assert torch.equal(terminal_ids, done.nonzero().flatten())
            assert info["terminal_actor_observations"].shape == (len(terminal_ids), 3466)
            assert info["terminal_critic_observations"].shape == (len(terminal_ids), 3660)
            assert not (info["time_outs"] & ~done).any()
            # 已终止的行已返回初始模式，其余行应立即反映本次请求。
            assert (env.low_level_obs[~done, -1] == mode).all()
            check_terrain()
            check_read_only(env)
            report["steps"].append(json_value({
                "step": index + 1, "phase": phase, "requested_mode": mode,
                "reward": reward, "done": done, "time_outs": info["time_outs"],
                "low_level_steps": low_steps, "physics_steps": physics_delta,
                "sensor_captures": capture_delta, "terminal_env_ids": terminal_ids,
                "log": info.get("log", {}),
            }))
        physics_before = int(env.sim_step_counter)
        env.reset(torch.tensor([0], device=env.device))
        assert int(env.sim_step_counter) == physics_before, "reset 额外推进了物理。"
        check_terrain()
        check_read_only(env)
    assert all(parameter.grad is None for parameter in env.frozen_policy.parameters())
    report["checks"].update({
        "observations_and_read_only": "passed", "forty_physics_steps_per_macro": "passed",
        "two_sensor_captures_for_unterminated_rows": "passed",
        "fixed_terrain_across_steps_and_partial_reset": "passed",
        "partial_reset_without_physics": "passed", "terminal_interface": "passed",
    })
    report["sensor_capture_count"] = json_value(env.sensor_capture_count)
    report["physics_step_count"] = int(env.sim_step_counter)
    report["fault_audit"] = json_value(env.faults.audit())


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting", "scope": "真实仿真环境接入检查，不进行 PPO 或安全性能评估。",
        "seed": args.seed, "num_envs": args.num_envs, "high_level_steps": args.steps,
        "faults_enabled": not args.disable_faults,
    }
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args).app
        run_check(args, simulation_app, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        path = args.output / "report.json"
        path.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "report": str(path.resolve())}), flush=True)
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
