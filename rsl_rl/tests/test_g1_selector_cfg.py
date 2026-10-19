"""Selector configuration and reward integration checks without launching Kit."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import MISSING
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from legged_lab.high_level.high_level_fault_config import (
    FaultScheduleCfg, HighLevelFaultCfg, LpFaultCfg, VpFaultCfg,
)


ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "legged_lab/envs/g1/g1_selector_cfg.py"
REWARDS_PATH = ROOT / "legged_lab/mdp/rewards.py"


class SimulatorCfg:
    """Stand in only for simulator assets and sensor constructors."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


SimulatorCfg.OffsetCfg = SimulatorCfg


class EventFunctions(SimpleNamespace):
    def __getattr__(self, name):
        # Events are configured but never executed by these CPU-only checks.
        return _unused_event


def _unused_event(*args, **kwargs):
    raise AssertionError("Simulation events must not execute in configuration tests")


def _exec_nodes(path, scope, *, classes=(), functions=(), all_nodes=False):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if (
        all_nodes and not isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.ClassDef) and node.name in classes
        or isinstance(node, ast.FunctionDef) and node.name in functions
    )]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)


@pytest.fixture(scope="module")
def selector_module():
    # Load native configclass, term types, base configs, PPO config, math and
    # buffers. Only simulator assets/sensors are replaced; no pxr/Kit imports.
    isaac_spec = importlib.util.find_spec("isaaclab")
    rl_spec = importlib.util.find_spec("isaaclab_rl")
    if isaac_spec is None or rl_spec is None:
        pytest.skip("IsaacLab and isaaclab_rl are required for native configuration checks")
    isaac = Path(next(iter(isaac_spec.submodule_search_locations)))
    rl = Path(next(iter(rl_spec.submodule_search_locations)))
    with pytest.MonkeyPatch.context() as patch:
        utils = ModuleType("_selector_native_utils")
        utils.__path__ = [str(isaac / "utils")]
        patch.setitem(sys.modules, utils.__name__, utils)
        configclass = importlib.import_module(f"{utils.__name__}.configclass").configclass
        math_utils = importlib.import_module(f"{utils.__name__}.math")
        module = ModuleType("_selector_config_checks")
        patch.setitem(sys.modules, module.__name__, module)
        scope = module.__dict__
        scope.update(
            math=math, os=os, torch=torch, MISSING=MISSING, configclass=configclass,
            math_utils=math_utils, LEGGED_LAB_ROOT_DIR=str(ROOT), mdp=EventFunctions(),
            FaultScheduleCfg=FaultScheduleCfg, HighLevelFaultCfg=HighLevelFaultCfg,
            LpFaultCfg=LpFaultCfg, VpFaultCfg=VpFaultCfg,
            G1_23CFG=SimulatorCfg(), G1_23DOF_LINKS=(), ROUGH_TERRAINS_CFG=SimulatorCfg(curriculum=True),
            get_link_prim_targets=lambda links: [],
        )
        for name in (
            "ArticulationCfg", "TerrainGeneratorCfg", "TiledCameraCfg", "LidarCfg",
            "RayCasterCfg", "GroupedRayCasterCameraCfg", "PinholeCameraPatternCfg", "FootSolePatternCfg",
        ):
            scope[name] = SimulatorCfg
        _exec_nodes(isaac / "managers/scene_entity_cfg.py", scope, classes={"SceneEntityCfg"})
        _exec_nodes(isaac / "managers/manager_term_cfg.py", scope,
                    classes={"ManagerTermBaseCfg", "RewardTermCfg", "EventTermCfg"})
        scope.update(RewTerm=scope["RewardTermCfg"], EventTerm=scope["EventTermCfg"])
        _exec_nodes(ROOT / "legged_lab/envs/base/base_config.py", scope, classes={
            node.name for node in ast.parse((ROOT / "legged_lab/envs/base/base_config.py").read_text()).body
            if isinstance(node, ast.ClassDef)
        })
        _exec_nodes(rl / "rsl_rl/rl_cfg.py", scope, classes={"RslRlPpoAlgorithmCfg"})
        observation_noise_cfg = scope["NoiseCfg"]
        _exec_nodes(isaac / "utils/noise/noise_cfg.py", scope, classes={"NoiseCfg"})
        scope.update(ImageNoiseModel=SimulatorCfg, distance_dependent_gaussian_noise=_unused_event)
        _exec_nodes(ROOT / "legged_lab/sensors/camera_noise/camera_noise_cfg.py", scope,
                    classes={"ImageNoiseCfg", "DistanceDependentGaussianNoiseCfg"})
        scope["NoiseCfg"] = observation_noise_cfg
        functions = {
            "track_lin_vel_xy_yaw_frame_exp", "track_ang_vel_z_world_exp", "is_terminated",
            "alive", "action_rate_l2", "selector_mode_switch",
        }
        _exec_nodes(REWARDS_PATH, scope, functions=functions)
        scope["mdp"].__dict__.update({name: scope[name] for name in functions})
        for name, path in (
            ("lp", ROOT / "legged_lab/perception/lp_height.py"),
            ("fault", ROOT / "legged_lab/selector/config.py"),
        ):
            spec = importlib.util.spec_from_file_location(f"_selector_config_{name}", path)
            dependency = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, spec.name, dependency)
            spec.loader.exec_module(dependency)
            scope.update({key: getattr(dependency, key) for key in (
                ("LPHeightCfg", "PgttNoiseCfg", "resolve_pgtt_height_cfg") if name == "lp" else ("FaultConfig",)
            )})
        _exec_nodes(CFG_PATH, scope, all_nodes=True)
        _exec_nodes(isaac / "utils/buffers/circular_buffer.py", scope, classes={"CircularBuffer"})
        _exec_nodes(isaac / "utils/buffers/delay_buffer.py", scope, classes={"DelayBuffer"})
        # Execute the actual manager's compute method, not a duplicate reward sum.
        tree = ast.parse((isaac / "managers/reward_manager.py").read_text())
        manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardManager")
        compute = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "compute")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[compute], type_ignores=[])),
                     str(isaac / "managers/reward_manager.py"), "exec"), scope)
        yield module


def test_independent_configuration_constructs_and_serializes_exactly_six_rewards(selector_module):
    cfg = selector_module
    env, agent = cfg.G1SelectorEnvCfg(), cfg.G1SelectorAgentCfg()
    cfg.configure_selector_agent(env, agent)
    assert env.seed == agent.seed == 42
    env.validate()
    agent.validate()
    assert cfg.G1SelectorEnvCfg.__bases__ == (object,)
    tree = ast.parse(CFG_PATH.read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "legged_lab.envs.g1.RENet_cfg"
                   for node in tree.body)
    expected = {
        "track_lin_vel_xy_exp": (cfg.mdp.track_lin_vel_xy_yaw_frame_exp, 3.0),
        "track_ang_vel_z_exp": (cfg.mdp.track_ang_vel_z_world_exp, 2.0),
        "termination_penalty": (cfg.mdp.is_terminated, -200.0),
        "alive": (cfg.mdp.alive, 0.15),
        "mode_switch_penalty": (cfg.mdp.selector_mode_switch, -1.0),
        "action_rate_l2": (cfg.mdp.action_rate_l2, -0.01),
    }
    assert set(vars(env.reward)) == set(expected)
    for name, (function, weight) in expected.items():
        term = getattr(env.reward, name)
        assert isinstance(term, cfg.RewTerm)
        assert term.func is function and term.weight == weight
    assert not hasattr(env, "selector_reward")
    assert "SelectorRewardCfg" not in cfg.__all__
    encoded = json.loads(json.dumps({
        "reward": env.to_dict()["reward"], "selector": env.to_dict()["selector"],
        "policy": agent.to_dict()["policy"], "algorithm": agent.to_dict()["algorithm"],
    }))
    assert encoded["selector"]["aggregation"] == "sum"
    assert encoded["selector"]["stop_on_done"] and encoded["selector"]["bootstrap_timeouts"]
    assert encoded["selector"]["timeout_value_source"] == "terminal_observation"
    assert encoded["selector"]["terrain_curriculum"] is False
    assert "normal_blend_steps" not in encoded["selector"]
    assert encoded["policy"]["action_distribution"] == "categorical"
    assert encoded["algorithm"]["class_name"] == "DiscretePPO"
    assert env.selector.low_level_dt == 0.02 and env.selector.low_level_steps_per_decision == 10
    assert env.observation.actor_obs_dim == agent.policy.actor_obs_dim == 3466
    assert env.observation.critic_obs_dim == agent.policy.critic_obs_dim == 3660
    assert env.observation.state_dim == agent.policy.state_input_dim == 8
    assert not any(name.startswith("pending_") or name in ("confirmation_progress", "blend_progress")
                   for name in env.observation.state_names)
    assert env.selector.switch_confirmations == 1 and env.selector.min_dwell_s == 0.0
    # Terrain generation retains its difficulty rows; the future env must not
    # advance them at reset. Initial terrain coverage remains unchanged.
    assert env.selector.terrain_curriculum is False
    assert env.scene.terrain_generator.curriculum is True and env.scene.max_init_terrain_level == 5
    assert not env.lp_height.noise.enable and not env.scene.camera.add_camera_noise
    assert env.scene.camera.camera.depth_clipping_behavior == "none"
    assert env.depth_blur_probability == env.depth_noise.near_std == env.depth_noise.far_std == 0.0
    faults = HighLevelFaultCfg.from_config(env.faults)
    assert len(faults.lp.kinds) == 11 and len(faults.vp.modes) == 8
    assert env.lp_height.noise.start_iteration == env.depth_noise_start_iteration == 0


def test_physics_randomization_is_disabled_but_original_resets_and_action_history_remain(selector_module):
    cfg = selector_module
    env = cfg.G1SelectorEnvCfg()
    events = env.domain_rand.events
    active = {name: term for name, term in vars(events).items() if term is not None}
    # Includes the inherited arm reset; no inherited startup PD randomization may survive.
    assert set(active) == {"reset_base", "reset_robot_joints", "reset_arm_pose_and_hold"}
    assert all(term.mode == "reset" for term in active.values())
    defaults = cfg.EventCfg()
    for name, term in active.items():
        assert term.to_dict() == getattr(defaults, name).to_dict()
    assert events.reset_base.params["pose_range"] == {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
    }
    assert events.reset_base.params["velocity_range"] == {
        axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    assert events.reset_robot_joints.params == {
        "position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0),
    }
    assert env.domain_rand.action_delay.enable is False
    assert env.domain_rand.action_delay.params == {"max_delay": 5, "min_delay": 0}


def test_overrides_sync_dimensions_mode_and_do_not_share_mutable_defaults(selector_module):
    cfg = selector_module
    env, other = cfg.G1SelectorEnvCfg(), cfg.G1SelectorEnvCfg()
    agent, other_agent = cfg.G1SelectorAgentCfg(), cfg.G1SelectorAgentCfg()
    env.selector.initial_mode = 1
    agent.seed = 123
    env.observation.use_depth_valid_support = False
    env.observation.use_height_valid_mask = False
    env.observation.critic_use_privileged = False
    cfg.configure_selector_agent(env, agent)
    assert env.seed == 123 and other.seed == 42
    assert env.renet.mask_mode == "lp" and other.renet.mask_mode == "vp"
    assert agent.policy.actor_obs_dim == agent.policy.critic_obs_dim == 2127
    assert agent.policy.depth_encoder.input_channels == 2 and agent.policy.height_input_dim == 187
    agent.policy.actor_hidden_dims.append(99)
    env.scene.camera.depth_crop = (0, 0, 0, 0)
    env.reward.track_lin_vel_xy_exp.params["std"] = 1.0
    env.lp_height.noise.height_std_range = (0.0, 0.0)
    env.scene.terrain_generator.curriculum = False
    assert other_agent.policy.actor_hidden_dims == [256, 128]
    assert other.scene.camera.depth_crop == (18, 0, 16, 16)
    assert other.reward.track_lin_vel_xy_exp.params == {"std": 0.5}
    assert other.lp_height.noise.height_std_range == (0.01, 0.03)
    assert other.scene.terrain_generator.curriculum is cfg.ROUGH_TERRAINS_CFG.curriculum is True


@pytest.mark.parametrize("path,value", [
    ("selector.frequency", 0.0), ("selector.frequency", 7.0), ("sim.decimation", 8),
    ("selector.initial_mode", 2), ("observation.proprio_frame_dim", 77),
    ("robot.depth_update_interval", 4), ("scene.camera.add_camera", False),
    ("selector.stop_on_done", False), ("selector.bootstrap_timeouts", False), ("low_level.freeze", False),
    ("selector.switch_confirmations", 3), ("selector.min_dwell_s", 1.0),
    ("selector.terrain_curriculum", True),
    ("robot.limit_angle", 0.0), ("robot.limit_angle", float("nan")),
    ("domain_rand.action_delay.enable", True),
    ("domain_rand.action_delay.params", {"max_delay": 0, "min_delay": 0}),
    ("scene.camera.camera.depth_clipping_behavior", "max"),
    ("lp_height.noise.enable", True), ("depth_blur_probability", 0.5),
    ("scene.camera.add_camera_noise", True),
])
def test_invalid_control_and_observation_overrides_are_rejected(selector_module, path, value):
    env, agent = selector_module.G1SelectorEnvCfg(), selector_module.G1SelectorAgentCfg()
    parent = env
    for name in path.split(".")[:-1]:
        parent = getattr(parent, name)
    setattr(parent, path.split(".")[-1], value)
    with pytest.raises(ValueError):
        selector_module.configure_selector_agent(env, agent)


def _reward_environment(count):
    return SimpleNamespace(
        num_envs=count, device="cpu", reset_buf=torch.zeros(count, dtype=torch.bool),
        time_out_buf=torch.zeros(count, dtype=torch.bool), selector_switch_event=torch.zeros(count, dtype=torch.bool),
        command_generator=SimpleNamespace(command=torch.zeros(count, 3)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
            root_lin_vel_w=torch.zeros(count, 3), root_ang_vel_w=torch.zeros(count, 3),
        ))},
        action_buffer=SimpleNamespace(_circular_buffer=SimpleNamespace(buffer=torch.zeros(count, 2, 23))),
    )


def _reward_manager(cfg, env):
    terms = vars(cfg.Reward())
    return SimpleNamespace(
        _env=env, _term_names=list(terms), _term_cfgs=list(terms.values()),
        _reward_buf=torch.zeros(env.num_envs), _step_reward=torch.zeros(env.num_envs, len(terms)),
        _episode_sums={name: torch.zeros(env.num_envs) for name in terms},
    )


def test_native_reward_compute_includes_failure_excludes_timeout_and_scales_switch_once(selector_module):
    cfg = selector_module
    env = _reward_environment(4)
    env.reset_buf[:2] = True
    env.time_out_buf[1] = True
    env.selector_switch_event[2] = True
    manager = _reward_manager(cfg, env)
    torch.testing.assert_close(cfg.compute(manager, 0.02), torch.tensor([-3.897, .103, .083, .103]))
    torch.testing.assert_close(manager._episode_sums["termination_penalty"], torch.tensor([-4., 0., 0., 0.]))
    torch.testing.assert_close(manager._episode_sums["mode_switch_penalty"], torch.tensor([0., 0., -.02, 0.]))
    torch.testing.assert_close(cfg.mdp.selector_mode_switch(env), torch.tensor([0., 0., 1., 0.]))
    assert env.selector_switch_event.tolist() == [False, False, True, False]
    with pytest.raises(AttributeError, match="selector_switch_event"):
        cfg.mdp.selector_mode_switch(SimpleNamespace())


def test_one_step_switch_pulses_do_not_charge_remaining_macro_steps(selector_module):
    cfg = selector_module
    env = _reward_environment(3)
    manager = _reward_manager(cfg, env)
    total = torch.zeros(3)
    for step in range(10):
        env.selector_switch_event.zero_()
        if step in (0, 4):
            env.selector_switch_event[0 if step == 0 else 1] = True
        total += cfg.compute(manager, 0.02)
    torch.testing.assert_close(total, torch.tensor([1.01, 1.01, 1.03]))
    torch.testing.assert_close(manager._episode_sums["mode_switch_penalty"], torch.tensor([-.02, -.02, 0.]))


def test_action_rate_uses_23_joint_actions_and_native_reset_clears_episode_history(selector_module):
    cfg = selector_module
    buffer = cfg.DelayBuffer(5, 2, device="cpu")
    env = SimpleNamespace(action_buffer=buffer)
    buffer.compute(torch.zeros(2, 23))
    buffer.compute(torch.ones(2, 23))
    term = cfg.Reward().action_rate_l2
    torch.testing.assert_close(term.func(env) * term.weight * .02, torch.full((2,), -.0046))
    buffer.reset(torch.tensor([0]))
    buffer.compute(torch.tensor([[100.] * 23, [2.] * 23]))
    torch.testing.assert_close(term.func(env), torch.tensor([0., 23.]))
