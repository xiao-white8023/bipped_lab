from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "squat_stand_env.py"
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "squat_stand_cfg.py"
REWARDS_PATH = REPO_ROOT / "legged_lab" / "mdp" / "rewards.py"
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
    namespace = {"torch": torch, "np": np}
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _load_reward_function(function_name):
    tree = ast.parse(
        REWARDS_PATH.read_text(encoding="utf-8"), filename=str(REWARDS_PATH)
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    selected = []
    if function_name == "track_com_forward_in_support_frame":
        selected.append(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "com_forward_target_from_height"
            )
        )
    selected.append(function)
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    namespace = {
        "torch": torch,
        "BaseEnv": object,
        "SceneEntityCfg": lambda *args, **kwargs: None,
    }
    exec(compile(module, str(REWARDS_PATH), "exec"), namespace)
    return namespace[function_name]


def _arm_cfg(curriculum_enable=True):
    return SimpleNamespace(
        enable=True,
        curriculum_enable=curriculum_enable,
        curriculum_start_iteration=2500,
        curriculum_end_iteration=12500,
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
    (
        update,
        linear_factor,
        validate_height,
        validate_com_forward,
        apply_height,
        apply_com_margin,
        hold_default,
        set_iteration,
        sample,
        advance,
    ) = _load_env_methods(
        "update_arm_curriculum",
        "_linear_curriculum_factor",
        "_validate_height_curriculum",
        "_validate_com_forward_reward",
        "_apply_height_curriculum",
        "_apply_com_support_margin_curriculum",
        "_hold_right_arm_at_default",
        "set_training_iteration",
        "sample_new_right_arm_motion",
        "update_right_arm_motion",
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
    term_cfgs = {
        "track_height_cmd": SimpleNamespace(weight=5.0, params={"std": 0.05}),
        "track_com_forward": SimpleNamespace(
            weight=0.5,
            params={
                "stand_target_x": 0.0,
                "squat_target_x": 0.02,
                "tracking_start_height": 0.65,
                "squat_height": 0.36,
                "std": 0.04,
            },
        ),
        "com_support_margin": SimpleNamespace(weight=-2.0),
    }

    class RewardManagerStub:
        def get_term_cfg(self, term_name):
            return term_cfgs[term_name]

        def set_term_cfg(self, term_name, term_cfg):
            assert term_cfg is term_cfgs[term_name]

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            right_arm_motion=_arm_cfg(),
            height_curriculum_start_iteration=0,
            height_curriculum_end_iteration=2500,
            height_curriculum_start_min=0.65,
            height_curriculum_final_min=0.36,
            height_tracking_start_std=0.08,
            height_tracking_final_std=0.05,
            com_support_margin_start_iteration=3500,
            com_support_margin_ramp_duration=1000,
        ),
        device="cpu",
        current_iteration=0,
        height_curriculum_factor=0.0,
        com_margin_factor=0.0,
        _height_command_final_range=(0.36, 0.77),
        _com_support_margin_weight=-2.0,
        command_generator=SimpleNamespace(
            command=torch.full((num_envs, 1), 0.7),
            cfg=SimpleNamespace(
                ranges=SimpleNamespace(height=(0.36, 0.77))
            )
        ),
        reward_manager=RewardManagerStub(),
        extras={"log": {}},
        arm_curriculum_factor=0.0,
        num_envs=num_envs,
        step_dt=0.02,
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
    env._linear_curriculum_factor = linear_factor
    env._validate_height_curriculum = MethodType(validate_height, env)
    env._validate_com_forward_reward = MethodType(
        validate_com_forward, env
    )
    env._apply_height_curriculum = MethodType(apply_height, env)
    env._apply_com_support_margin_curriculum = MethodType(apply_com_margin, env)
    env._hold_right_arm_at_default = MethodType(hold_default, env)
    env.set_training_iteration = MethodType(set_iteration, env)
    env.sample_new_right_arm_motion = MethodType(sample, env)
    env.update_right_arm_motion = MethodType(advance, env)
    env._validate_height_curriculum()
    env._validate_com_forward_reward()
    env._apply_height_curriculum()
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
            "curriculum_start_iteration",
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
        "curriculum_start_iteration": 2500,
        "curriculum_end_iteration": 12500,
        "min_range_fraction": 0.05,
        "max_range_fraction": 0.6,
        "min_max_vel": 0.1,
        "max_max_vel": 0.5,
        "min_hold_time": 0.3,
        "max_hold_time": 1.5,
    }


def test_com_support_margin_curriculum_config_defaults():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
    class_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SaquatStandENVCFG"
    )
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in class_body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id
        in {
            "com_support_margin_start_iteration",
            "com_support_margin_ramp_duration",
        }
    }
    assert values == {
        "com_support_margin_start_iteration": 3500,
        "com_support_margin_ramp_duration": 1000,
    }


def test_height_curriculum_config_defaults():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
    class_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SaquatStandENVCFG"
    )
    names = {
        "height_curriculum_start_iteration",
        "height_curriculum_end_iteration",
        "height_curriculum_start_min",
        "height_curriculum_final_min",
        "height_tracking_start_std",
        "height_tracking_final_std",
    }
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in class_body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in names
    }
    assert values == {
        "height_curriculum_start_iteration": 0,
        "height_curriculum_end_iteration": 2500,
        "height_curriculum_start_min": 0.65,
        "height_curriculum_final_min": 0.36,
        "height_tracking_start_std": 0.08,
        "height_tracking_final_std": 0.05,
    }


def test_com_forward_reward_has_no_iteration_curriculum():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")

    assert "com_forward_start_iteration" not in cfg_source
    assert "com_forward_ramp_duration" not in cfg_source
    assert "_apply_com_forward_curriculum" not in env_source
    assert "curriculum/com_forward" not in env_source


@pytest.mark.parametrize(
    ("iteration", "factor", "command_min", "height_std"),
    [
        (0, 0.0, 0.65, 0.08),
        (1250, 0.5, 0.505, 0.065),
        (2500, 1.0, 0.36, 0.05),
        (5000, 1.0, 0.36, 0.05),
    ],
)
def test_height_curriculum_updates_future_range_and_reward_std(
    iteration, factor, command_min, height_std
):
    env = _curriculum_env(num_envs=1)
    active_command_before = env.command_generator.command.clone()
    env.set_training_iteration(iteration)

    assert env.height_curriculum_factor == pytest.approx(factor)
    assert env.command_generator.cfg.ranges.height == pytest.approx(
        (command_min, 0.77)
    )
    assert env.reward_manager.get_term_cfg(
        "track_height_cmd"
    ).params["std"] == pytest.approx(height_std)
    torch.testing.assert_close(
        env.command_generator.command, active_command_before
    )


def test_curriculum_validation_rejects_bad_height_and_com_parameters():
    env = _curriculum_env(num_envs=1)
    env.reward_manager.get_term_cfg("track_height_cmd").params["std"] = 0.05
    env.cfg.height_curriculum_end_iteration = 0
    with pytest.raises(ValueError, match="height_curriculum_end_iteration"):
        env._validate_height_curriculum()

    env = _curriculum_env(num_envs=1)
    env.reward_manager.get_term_cfg("track_com_forward").params["std"] = 0.0
    with pytest.raises(ValueError, match="std must be positive"):
        env._validate_com_forward_reward()


def test_negative_training_iteration_is_rejected():
    env = _curriculum_env(num_envs=1)
    with pytest.raises(ValueError, match="non-negative"):
        env.set_training_iteration(-1)


@pytest.mark.parametrize("iteration", [0, 3000, 4000, 8000])
def test_com_forward_reward_weight_is_fixed_for_all_iterations(iteration):
    env = _curriculum_env(num_envs=1)
    env.set_training_iteration(iteration)

    assert env.reward_manager.get_term_cfg("track_com_forward").weight == pytest.approx(
        0.5
    )


def test_com_forward_target_interpolates_with_squat_depth():
    target_fn = _load_reward_function(
        "com_forward_target_from_height"
    )
    heights = torch.tensor(
        [0.77, 0.70, 0.65, (0.65 + 0.36) / 2.0, 0.36]
    )

    targets = target_fn(heights)

    torch.testing.assert_close(
        targets,
        torch.tensor([0.0, 0.0, 0.0, 0.01, 0.02]),
    )


def test_com_forward_reward_is_maximal_at_interpolated_target():
    reward_fn = _load_reward_function("track_com_forward_in_support_frame")
    height_command = torch.tensor(
        [[0.77], [0.70], [0.65], [0.505], [0.36], [0.36]]
    )
    com_x = torch.tensor([0.0, 0.0, 0.0, 0.01, 0.02, -0.02])
    env = SimpleNamespace(
        get_root_height_in_support_frame=lambda: height_command[:, 0],
        get_com_state_in_support_frame=lambda: (
            torch.stack((com_x, torch.zeros_like(com_x), torch.ones_like(com_x)), dim=-1),
            torch.zeros(6, 3),
        ),
    )

    reward = reward_fn(env, std=0.04)

    torch.testing.assert_close(reward[:5], torch.ones(5))
    assert reward[5] < reward[4]


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, 0.0),
        (1000, 0.0),
        (2500, 0.0),
        (5000, 0.25),
        (7500, 0.5),
        (12500, 1.0),
        (15000, 1.0),
    ],
)
def test_curriculum_factor_is_clamped_linear_training_progress(iteration, expected):
    env = _curriculum_env(num_envs=1)
    env.set_training_iteration(iteration)
    assert env.current_iteration == iteration
    assert env.arm_curriculum_factor == pytest.approx(expected)
    assert env.extras["log"]["curriculum/right_arm_factor"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("iteration", "factor", "weight"),
    [
        (0, 0.0, 0.0),
        (1000, 0.0, 0.0),
        (3499, 0.0, 0.0),
        (3500, 0.0, 0.0),
        (4000, 0.5, -1.0),
        (4499, 0.999, -1.998),
        (4500, 1.0, -2.0),
        (5000, 1.0, -2.0),
    ],
)
def test_com_support_margin_curriculum_updates_cached_reward_term(
    iteration, factor, weight
):
    env = _curriculum_env(num_envs=4096)
    env.set_training_iteration(iteration)

    assert env.com_margin_factor == pytest.approx(factor)
    assert env.reward_manager.get_term_cfg("com_support_margin").weight == pytest.approx(
        weight
    )
    assert env.extras["log"][
        "curriculum/com_support_margin_factor"
    ] == pytest.approx(factor)


def test_right_arm_is_held_at_default_before_curriculum_start():
    env = _curriculum_env(num_envs=4096)
    env.robot.data.default_joint_pos.fill_(0.2)
    env.arm_motion_start_q.fill_(1.0)
    env.arm_motion_target_q.fill_(1.0)
    env.arm_q_des.fill_(1.0)
    env.arm_dq_des.fill_(1.0)
    env.arm_motion_elapsed.fill_(1.0)
    env.arm_motion_duration.fill_(1.0)
    env.arm_motion_hold_duration.fill_(1.0)
    env.set_training_iteration(1000)

    env.sample_new_right_arm_motion(torch.arange(4096))
    env.update_right_arm_motion()

    expected_q = torch.full((4096, 7), 0.2)
    assert torch.allclose(env.arm_motion_start_q, expected_q)
    assert torch.allclose(env.arm_motion_target_q, expected_q)
    assert torch.allclose(env.arm_q_des, expected_q)
    assert torch.count_nonzero(env.arm_dq_des) == 0
    assert torch.count_nonzero(env.arm_motion_elapsed) == 0
    assert torch.count_nonzero(env.arm_motion_duration) == 0
    assert torch.count_nonzero(env.arm_motion_hold_duration) == 0


def test_com_support_margin_penalty_only_activates_outside_support_domain():
    reward_fn = _load_reward_function("com_support_margin_penalty")
    foot_corners = torch.tensor(
        [
            [[0.1, 0.2, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0], [-0.1, 0.2, 0.0]],
            [[0.1, -0.1, 0.0], [0.1, -0.2, 0.0], [-0.1, -0.2, 0.0], [-0.1, -0.1, 0.0]],
        ],
        dtype=torch.float,
    )
    com_xy = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.13, 0.0], [0.0, -0.26]],
        dtype=torch.float,
    )
    env = SimpleNamespace(
        foot_support_corners_w=foot_corners.unsqueeze(0).expand(4, -1, -1, -1),
        whole_body_com_pos_w=torch.cat([com_xy, torch.zeros(4, 1)], dim=1),
    )

    penalty = reward_fn(env, std=0.03, outside_scale=10.0)

    assert penalty[:2].tolist() == [0.0, 0.0]
    assert penalty[2] == pytest.approx(10.0)
    assert penalty[3] == pytest.approx(40.0)


def test_removed_com_reward_and_margin_shaping_are_absent_from_task_sources():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    reward_source = REWARDS_PATH.read_text(encoding="utf-8")
    reward_tree = ast.parse(reward_source, filename=str(REWARDS_PATH))
    reward_fn = next(
        node
        for node in reward_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "com_support_margin_penalty"
    )
    function_source = ast.get_source_segment(reward_source, reward_fn)

    assert "com_xy_velocity = RewTerm" not in cfg_source
    assert "ComXYVelocityCurriculumCfg" not in cfg_source
    assert "com_xy_velocity" not in env_source
    assert "rear_scale" not in function_source
    assert "edge_penalty" not in function_source
    assert "rear_edge_penalty" not in function_source
    assert "inside_distance" not in function_source
    assert '"curriculum/right_arm_factor"' in env_source
    assert '"curriculum/com_support_margin_factor"' in env_source


@pytest.mark.parametrize(
    ("iteration", "range_fraction", "max_vel", "hold_time"),
    [
        (5000, 0.1875, 0.2, 0.6),
        (7500, 0.325, 0.3, 0.9),
        (12500, 0.6, 0.5, 1.5),
    ],
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
