from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

from legged_lab.perception.lp_height import LPHeightCfg, PgttNoiseCfg
from test_renet_action_delay import _environment, native_delay_buffer
from test_renet_auxiliary_updates import ROOT, _extract_function


ENV_PATH = ROOT / "legged_lab/envs/g1/RENet_env.py"
CFG_PATH = ROOT / "legged_lab/envs/g1/RENet_cfg.py"


def _configured_stages():
    tree = ast.parse(CFG_PATH.read_text())
    train = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RENetTrainCfg")
    start = next(node.value for node in train.body if isinstance(node, ast.AnnAssign)
                 and node.target.id == "domain_rand_start_iteration")
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENETENVCFG")
    lp = next(node.value for node in cfg.body if isinstance(node, ast.Assign) and node.targets[0].id == "lp_height")
    return ast.literal_eval(start), eval(compile(ast.Expression(lp), str(CFG_PATH), "eval"),
                                        {"LPHeightCfg": LPHeightCfg, "PgttNoiseCfg": PgttNoiseCfg})


def _scheduled_env(buffer_class, *, iteration=0, enabled=True, start_iteration=None):
    start, lp_cfg = _configured_stages()
    if start_iteration is not None:
        start = start_iteration
    env = _environment(buffer_class, minimum=2, maximum=2, num_envs=2,
                       start_iteration=start, iteration=iteration, enabled=enabled)
    env._domain_rand_stage_active = False
    env._domain_rand_startup_applied = False
    env.lp_runtime_cfg = lp_cfg
    env.pgtt_noise = object()
    env.applied_events = []
    env.event_manager = SimpleNamespace(available_modes=["randomization_start" if start else "startup", "interval"],
                                       apply=lambda **kwargs: env.applied_events.append(kwargs))
    for name in ("_sync_domain_randomization_stage", "_pgtt_noise_is_active", "set_training_iteration"):
        method = _extract_function(ENV_PATH, name, "G1RENetEnv")
        setattr(env, name, lambda *args, method=method: method(env, *args))
    env._sync_pgtt_noise_stage = lambda: None
    env.sample_renet_estimator_mask = lambda: None
    env.resample(torch.arange(env.num_envs))
    return env


def test_config_restores_startup_randomization_and_keeps_lp_noise_threshold():
    start, lp_cfg = _configured_stages()
    assert start == 0 and lp_cfg.noise.start_iteration == 30000
    tree = ast.parse(CFG_PATH.read_text())
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENETENVCFG")
    domain = next(node.value for node in cfg.body if isinstance(node, ast.AnnAssign)
                  and node.target.id == "domain_rand")
    scope = {name: lambda **kwargs: SimpleNamespace(**kwargs)
             for name in ("DomainRandCfg", "EventCfg", "EventTerm", "ActionDelayCfg")}
    scope["SceneEntityCfg"] = lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    scope["mdp"] = SimpleNamespace(**{name: object() for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "reset_root_state_uniform",
        "reset_joints_by_scale", "push_by_setting_velocity")})
    domain = eval(compile(ast.Expression(domain), str(CFG_PATH), "eval"), scope)
    assert domain.events.physics_material.mode == domain.events.add_base_mass.mode == "startup"
    assert domain.events.physics_material.params['static_friction_range'] == (.6, 1.)
    assert domain.events.physics_material.params['dynamic_friction_range'] == (.4, .8)
    assert domain.events.physics_material.params['restitution_range'] == (0., .005)
    assert domain.events.physics_material.params['num_buckets'] == 64
    assert domain.events.add_base_mass.params['mass_distribution_params'] == (-5., 5.)
    assert domain.events.add_base_mass.params['operation'] == 'add'
    assert domain.events.push_robot.mode == "interval"
    assert domain.events.push_robot.interval_range_s == (10., 15.)
    assert domain.events.push_robot.params['velocity_range'] == {'x': (-1., 1.), 'y': (-1., 1.)}
    assert domain.events.reset_base.mode == domain.events.reset_robot_joints.mode == "reset"
    assert not domain.action_delay.enable
    assert domain.action_delay.params == {"min_delay": 0, "max_delay": 5}


def test_startup_push_and_explicitly_enabled_delay_do_not_wait_for_lp_noise(native_delay_buffer):
    env = _scheduled_env(native_delay_buffer)
    tree = ast.parse(ENV_PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'G1RENetEnv')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    startup = next(node for node in init.body if isinstance(node, ast.If)
                   and any(isinstance(value, ast.Constant) and value.value == 'startup'
                           for value in ast.walk(node.test)))
    exec(compile(ast.Module(body=[startup], type_ignores=[]), str(ENV_PATH), 'exec'), {'self': env})
    env.set_training_iteration(0)
    assert env.applied_events == [{'mode': 'startup'}]
    assert env._domain_rand_is_active() and env.action_buffer.time_lags.tolist() == [2, 2]
    lag = env.action_buffer.time_lags.clone()
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'step')
    interval = next(node for node in step.body if isinstance(node, ast.If)
                    and any(isinstance(value, ast.Constant) and value.value == 'interval'
                            for value in ast.walk(node.test)))
    exec(compile(ast.Module(body=[interval], type_ignores=[]), str(ENV_PATH), 'exec'), {'self': env})
    assert env.applied_events[-1] == {'mode': 'interval', 'dt': .02}
    for iteration in (0, 29999, 30000, 50000):
        env.set_training_iteration(iteration)
        assert env._domain_rand_is_active()
        assert env._pgtt_noise_is_active() == (iteration >= 30000)
        torch.testing.assert_close(env.action_buffer.time_lags, lag)
    assert env.applied_events == [{'mode': 'startup'}, {'mode': 'interval', 'dt': .02}]


def test_clean_stage_and_boundary_apply_properties_once_and_preserve_queued_actions(native_delay_buffer):
    env = _scheduled_env(native_delay_buffer, start_iteration=30000)
    ids = torch.arange(2)
    for iteration in (0, 100, 29999):
        env.set_training_iteration(iteration)
        env.resample(ids)
        assert not env._pgtt_noise_is_active()
        assert not env.action_buffer.time_lags.any()
        assert not env.applied_events
    for value in (1., 2., 3.):
        action = torch.full((2, 23), value)
        torch.testing.assert_close(env.action_buffer.compute(action), action)
    env.set_training_iteration(30000)
    assert env._pgtt_noise_is_active()
    assert env.action_buffer.time_lags.tolist() == [2, 2]
    assert env.applied_events == [{"mode": "randomization_start"}]
    # The boundary must not clear the history or execute zeros instead of a queued action.
    output = env.action_buffer.compute(torch.full((2, 23), 4.))
    torch.testing.assert_close(output, torch.full_like(output, 2.))
    lag = env.action_buffer.time_lags.clone()
    for iteration in (30000, 30001, 50000):
        env.set_training_iteration(iteration)
        torch.testing.assert_close(env.action_buffer.time_lags, lag)
    assert len(env.applied_events) == 1


@pytest.mark.parametrize("iteration,active", [(29999, False), (30000, True), (42000, True)])
def test_resume_iteration_controls_stage_immediately(native_delay_buffer, iteration, active):
    env = _scheduled_env(native_delay_buffer, start_iteration=30000)
    env.set_training_iteration(iteration)
    assert env._pgtt_noise_is_active() == active
    assert env._domain_rand_stage_active == active
    assert bool(env.action_buffer.time_lags.any()) == active
    assert len(env.applied_events) == int(active)


def test_disabled_delay_and_disabled_lp_noise_remain_disabled_after_boundary(native_delay_buffer):
    env = _scheduled_env(native_delay_buffer, enabled=False)
    env.lp_runtime_cfg.noise.enable = False
    env.set_training_iteration(30000)
    assert not env.action_buffer.time_lags.any()
    assert not env._pgtt_noise_is_active()
    assert env._domain_rand_stage_active


def test_push_countdown_only_advances_after_stage_activation(native_delay_buffer):
    env = _scheduled_env(native_delay_buffer, start_iteration=30000)
    tree = ast.parse(ENV_PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    block = next(node for node in step.body if isinstance(node, ast.If)
                 and any(isinstance(value, ast.Constant) and value.value == "interval"
                         for value in ast.walk(node.test)))
    code = compile(ast.Module(body=[block], type_ignores=[]), str(ENV_PATH), "exec")
    for iteration in (0, 29999):
        env.set_training_iteration(iteration)
        exec(code, {"self": env})
        assert not env.applied_events
    env.set_training_iteration(30000)
    exec(code, {"self": env})
    assert env.applied_events == [{"mode": "randomization_start"}, {"mode": "interval", "dt": .02}]
