from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "squat_stand_env.py"
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "squat_stand_cfg.py"
RUNNER_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"


def _load_env_methods(*method_names):
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    class_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SquatStandEnv"
    )
    selected = []
    for method_name in method_names:
        method = next(
            node
            for node in class_body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method.decorator_list = []
        selected.append(method)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _arm_cfg(curriculum_enable=True):
    return SimpleNamespace(
        enable=True,
        curriculum_enable=curriculum_enable,
        curriculum_end_iteration=8000,
        min_range_fraction=0.05,
        max_range_fraction=0.6,
        min_max_vel=0.1,
        max_max_vel=0.5,
        min_hold_time=0.3,
        max_hold_time=1.5,
        range_fraction=0.6,
        max_vel=[0.5, 0.4, 0.4, 0.5, 0.3, 0.25, 0.25],
        speed_scale_range=(0.4, 1.0),
        hold_time_range=(0.2, 1.0),
        min_duration=0.5,
        debug_checks=True,
    )


def _curriculum_env(num_envs=4096):
    update, apply_com_margin, set_iteration, sample = _load_env_methods(
        "update_arm_curriculum",
        "_apply_com_support_margin_curriculum",
        "set_training_iteration",
        "sample_new_right_arm_motion",
    )
    num_joints = 7
    data = SimpleNamespace(
        joint_pos=torch.zeros(num_envs, num_joints),
        default_joint_pos=torch.zeros(num_envs, num_joints),
        soft_joint_pos_limits=torch.tensor([-2.0, 2.0])
        .reshape(1, 1, 2)
        .expand(num_envs, num_joints, 2)
        .clone(),
    )
    com_margin_term_cfg = SimpleNamespace(weight=-2.0)

    class RewardManagerStub:
        def get_term_cfg(self, term_name):
            assert term_name == "com_support_margin"
            return com_margin_term_cfg

        def set_term_cfg(self, term_name, term_cfg):
            assert term_name == "com_support_margin"
            assert term_cfg is com_margin_term_cfg

    env = SimpleNamespace(
        cfg=SimpleNamespace(right_arm_motion=_arm_cfg()),
        device="cpu",
        current_iteration=0,
        com_margin_factor=0.0,
        _com_support_margin_weight=-2.0,
        reward_manager=RewardManagerStub(),
        arm_curriculum_factor=0.0,
        robot=SimpleNamespace(data=data),
        right_arm_ids=torch.arange(num_joints, dtype=torch.long),
        num_right_arm_joints=num_joints,
        _right_arm_max_vel=torch.tensor([0.5, 0.4, 0.4, 0.5, 0.3, 0.25, 0.25]),
        _right_arm_velocity_scale=torch.tensor([1.0, 0.8, 0.8, 1.0, 0.6, 0.5, 0.5]),
        arm_motion_start_q=torch.zeros(num_envs, num_joints),
        arm_motion_target_q=torch.zeros(num_envs, num_joints),
        arm_q_des=torch.zeros(num_envs, num_joints),
        arm_dq_des=torch.zeros(num_envs, num_joints),
        arm_motion_elapsed=torch.zeros(num_envs),
        arm_motion_duration=torch.zeros(num_envs),
        arm_motion_hold_duration=torch.zeros(num_envs),
    )
    env.update_arm_curriculum = MethodType(update, env)
    env._apply_com_support_margin_curriculum = MethodType(apply_com_margin, env)
    env.set_training_iteration = MethodType(set_iteration, env)
    env.sample_new_right_arm_motion = MethodType(sample, env)
    env._apply_com_support_margin_curriculum()
    return env


def test_right_arm_curriculum_config_defaults():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
    class_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RightArmMotionCfg"
    )
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in class_body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id
        in {
            "curriculum_enable",
            "curriculum_end_iteration",
            "min_range_fraction",
            "max_range_fraction",
            "min_max_vel",
            "max_max_vel",
            "min_hold_time",
            "max_hold_time",
        }
    }
    assert values == {
        "curriculum_enable": True,
        "curriculum_end_iteration": 8000,
        "min_range_fraction": 0.05,
        "max_range_fraction": 0.6,
        "min_max_vel": 0.1,
        "max_max_vel": 0.5,
        "min_hold_time": 0.3,
        "max_hold_time": 1.5,
    }


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(-1, 0.0), (0, 0.0), (4000, 0.5), (8000, 1.0), (9000, 1.0)],
)
def test_curriculum_factor_is_clamped_linear_training_progress(iteration, expected):
    env = _curriculum_env(num_envs=1)
    env.set_training_iteration(iteration)
    assert env.current_iteration == iteration
    assert env.arm_curriculum_factor == pytest.approx(expected)


@pytest.mark.parametrize(
    ("iteration", "factor", "weight"),
    [(0, 0.0, 0.0), (999, 0.0, 0.0), (1000, 1.0, -2.0), (8000, 1.0, -2.0)],
)
def test_com_support_margin_curriculum_updates_cached_reward_term(
    iteration, factor, weight
):
    env = _curriculum_env(num_envs=4096)
    env.set_training_iteration(iteration)

    assert env.com_margin_factor == factor
    assert env.reward_manager.get_term_cfg("com_support_margin").weight == weight


@pytest.mark.parametrize(
    ("iteration", "range_fraction", "max_vel", "hold_time"),
    [(0, 0.05, 0.1, 1.5), (4000, 0.325, 0.3, 0.9), (8000, 0.6, 0.5, 0.3)],
)
def test_sampling_scales_only_arm_motion_parameters_for_4096_envs(
    iteration, range_fraction, max_vel, hold_time
):
    torch.manual_seed(7)
    env = _curriculum_env()
    env.set_training_iteration(iteration)
    env_ids = torch.arange(4096, dtype=torch.long)

    env.sample_new_right_arm_motion(env_ids)

    assert env.arm_motion_target_q.shape == (4096, 7)
    assert env.arm_motion_target_q.device.type == "cpu"
    assert torch.all(env.arm_motion_target_q.abs() <= 2.0 * range_fraction + 1.0e-6)
    assert torch.allclose(
        env.arm_motion_hold_duration,
        torch.full((4096,), hold_time),
    )
    peak_velocity = (
        1.875
        * (env.arm_motion_target_q - env.arm_motion_start_q).abs()
        / env.arm_motion_duration.unsqueeze(1)
    )
    per_joint_max_vel = max_vel * env._right_arm_velocity_scale
    assert torch.all(peak_velocity <= per_joint_max_vel.unsqueeze(0) + 1.0e-6)
    assert torch.isfinite(env.arm_motion_duration).all()


def test_on_policy_runner_iteration_hook_is_optional_and_precedes_rollout():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    hook = 'if hasattr(self.env, "set_training_iteration"):'
    step = "obs, rewards, dones, infos = self.env.step"
    assert hook in source
    assert source.index(hook) < source.index(step)
