"""CPU checks of the selector environment's real control flow, without Kit.

The source classes are executed after replacing simulation constructors only.
These tests exercise the environment implementation; they are not a physics or
checkpoint validation substitute for ``check_selector_env.py``.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from legged_lab.selector.randomness import EnvironmentRandomness
from legged_lab.perception.lp_height import LPHeightCfg, observation_contract


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "legged_lab/envs/g1/g1_selector_env.py"
REWARD_PATH = ROOT / "legged_lab/mdp/rewards.py"


def _exec_source(path, scope, names=None):
    nodes = [node for node in ast.parse(path.read_text()).body
             if not isinstance(node, (ast.Import, ast.ImportFrom))
             and (names is None or getattr(node, "name", None) in names)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), scope)


class _SceneEntity:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)

    def resolve(self, scene):
        if self.name == "contact_sensor":
            self.body_ids = [1, 2] if len(self.body_names) == 2 else [0]


class _RewardBase:
    """Simulation-free storage for the native manager extension under test."""

    def __init__(self, cfg, env):
        self._env = env
        self.num_envs, self.device = env.num_envs, env.device
        terms = vars(cfg) if not isinstance(cfg, dict) else cfg
        self._term_names = [name for name, term in terms.items() if term is not None]
        self._term_cfgs = [terms[name] for name in self._term_names]
        self._class_term_cfgs = []
        self._reward_buf = torch.zeros(self.num_envs)
        self._step_reward = torch.zeros(self.num_envs, len(self._term_names))
        self._episode_sums = {name: torch.zeros(self.num_envs) for name in self._term_names}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        result = {name: value[ids].mean().clone() for name, value in self._episode_sums.items()}
        for value in self._episode_sums.values():
            value[ids] = 0
        return result


class _CommandBase:
    @property
    def command(self):
        return self.vel_command_b


@pytest.fixture(scope="module")
def env_module():
    module = ModuleType("_selector_env_cpu_checks")
    sys.modules[module.__name__] = module
    scope = module.__dict__
    scope.update(torch=torch, math=math, VecEnv=object, RewardManager=_RewardBase,
                 UniformVelocityCommand=_CommandBase, SceneEntityCfg=_SceneEntity,
                 deepcopy=copy.deepcopy, copy=copy, LEGGED_LAB_ROOT_DIR=str(ROOT))
    # IsaacLab math has no simulation dependency, unlike the environment import.
    isaac_spec = importlib.util.find_spec("isaaclab")
    if isaac_spec is not None:
        isaac = Path(next(iter(isaac_spec.submodule_search_locations)))
        native = ModuleType("_selector_env_native_utils")
        native.__path__ = [str(isaac / "utils")]
        sys.modules[native.__name__] = native
        math_utils = importlib.import_module(f"{native.__name__}.math")
    else:
        pytest.skip("IsaacLab native math package is not installed")
    scope.update(math_utils=math_utils, quat_apply_inverse=math_utils.quat_apply_inverse)
    # Execute the real parent reset as well; only its simulation constructor is
    # replaced. This catches the extension's post-resampling command handling.
    command_tree = ast.parse((isaac / "managers/command_manager.py").read_text())
    native_command = next(node for node in command_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "CommandTerm")
    native_reset = next(node for node in native_command.body
                        if isinstance(node, ast.FunctionDef) and node.name == "reset")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, native_reset], type_ignores=[])),
                 str(isaac / "managers/command_manager.py"), "exec"), scope)
    _CommandBase.reset = scope.pop("reset")
    for module_name in ("high_level_buffers", "high_level_faults"):
        dependency = importlib.import_module(f"legged_lab.high_level.{module_name}")
        scope.update({name: getattr(dependency, name) for name in dir(dependency)
                      if not name.startswith("_")})
    _exec_source(REWARD_PATH, scope, {
        "track_lin_vel_xy_yaw_frame_exp", "track_ang_vel_z_world_exp", "is_terminated",
        "alive", "action_rate_l2", "selector_mode_switch",
    })
    scope["mdp"] = SimpleNamespace(**{name: scope[name] for name in (
        "track_lin_vel_xy_yaw_frame_exp", "track_ang_vel_z_world_exp", "is_terminated",
        "alive", "action_rate_l2", "selector_mode_switch",
    )})
    _exec_source(ENV_PATH, scope)
    return module


def _term(function, weight, **params):
    return SimpleNamespace(func=function, weight=weight, params=params)


def _reward_manager(env_module, env):
    mdp = env_module.mdp
    terms = {
        "track_lin_vel_xy_exp": _term(mdp.track_lin_vel_xy_yaw_frame_exp, 3.0, std=0.5),
        "track_ang_vel_z_exp": _term(mdp.track_ang_vel_z_world_exp, 2.0, std=0.5),
        "termination_penalty": _term(mdp.is_terminated, -200.0),
        "alive": _term(mdp.alive, 0.15),
        "mode_switch_penalty": _term(mdp.selector_mode_switch, -1.0),
        "action_rate_l2": _term(mdp.action_rate_l2, -0.01),
    }
    return env_module.SelectorRewardManager(terms, env)


def test_reward_manager_masks_individual_terms_and_episodic_sums(env_module):
    n = 3
    data = SimpleNamespace(root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                           root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3))
    env = SimpleNamespace(num_envs=n, device="cpu", scene={"robot": SimpleNamespace(data=data)},
                          command_generator=SimpleNamespace(command=torch.zeros(n, 3)),
                          reset_buf=torch.tensor([True, True, False]),
                          time_out_buf=torch.tensor([False, True, False]),
                          selector_switch_event=torch.tensor([True, True, True]),
                          action_buffer=SimpleNamespace(_circular_buffer=SimpleNamespace(
                              buffer=torch.zeros(n, 6, 23))))
    manager = _reward_manager(env_module, env)
    reward = manager.compute(0.02, torch.tensor([True, True, False])).clone()
    assert torch.allclose(reward, torch.tensor([-3.917, 0.083, 0.0]))
    assert manager._episode_sums["termination_penalty"].tolist() == [-4.0, 0.0, 0.0]
    assert manager._episode_sums["mode_switch_penalty"].tolist() == pytest.approx([-0.02, -0.02, 0])
    previous = {name: value.clone() for name, value in manager._episode_sums.items()}
    manager.compute(0.02, torch.tensor([False, False, False]))
    assert not manager._reward_buf.any() and not manager._step_reward.any()
    for name, value in manager._episode_sums.items():
        assert torch.equal(value, previous[name])


def test_inactive_rows_cannot_contaminate_reward_with_nonfinite_state(env_module):
    env = SimpleNamespace(num_envs=2, device="cpu")
    manager = env_module.SelectorRewardManager(
        {"probe": _term(lambda current: torch.tensor([2., float("nan")]), 3.0)}, env)
    result = manager.compute(0.02, torch.tensor([True, False]))
    assert result.tolist() == pytest.approx([0.12, 0.0])
    assert manager._episode_sums["probe"].tolist() == pytest.approx([0.12, 0.0])
    assert torch.isfinite(manager._step_reward).all()


def _command(env_module):
    command = env_module.SelectorVelocityCommand.__new__(env_module.SelectorVelocityCommand)
    command.num_envs, command.device = 3, "cpu"
    command.cfg = SimpleNamespace(
        resampling_time_range=(10., 10.), heading_command=True, heading_control_stiffness=0.5,
        rel_heading_envs=1., rel_standing_envs=0., ranges=SimpleNamespace(
            lin_vel_x=(-1., 1.), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-1., 1.), heading=(-math.pi, math.pi)))
    command._env = SimpleNamespace(low_level_dt=0.02, step_dt=0.2)
    command._env._env_ids = lambda ids=None: torch.arange(3) if ids is None else torch.as_tensor(ids)
    command._env.selector_randomness = EnvironmentRandomness(3, 42, "cpu")
    command.robot = SimpleNamespace(data=SimpleNamespace(
        root_lin_vel_b=torch.zeros(3, 3), root_ang_vel_b=torch.zeros(3, 3),
        heading_w=torch.tensor([0., 0., 0.])))
    command.vel_command_b = torch.tensor([[2., 0., 0.5], [1., 0., 0.5], [3., 1., -0.5]])
    command.heading_target = torch.tensor([0.4, -0.8, 1.2])
    command.is_heading_env = torch.tensor([True, True, True])
    command.is_standing_env = torch.tensor([True, False, False])
    command.time_left = torch.full((3,), 5.)
    command.command_counter = torch.zeros(3, dtype=torch.long)
    command.metrics = {name: torch.zeros(3) for name in ("error_vel_xy", "error_vel_yaw")}
    return command


def test_command_updates_only_active_rows_and_uses_low_level_time(env_module):
    command = _command(env_module)
    inactive_command = command.command[2].clone()
    command.compute(0.02, env_ids=torch.tensor([0, 1]))
    assert command.time_left.tolist() == pytest.approx([4.98, 4.98, 5.])
    assert torch.equal(command.command[2], inactive_command)
    assert command.command[0].tolist() == [0., 0., 0.]
    assert command.command[1, 2] == pytest.approx(-0.4)
    assert command.metrics["error_vel_xy"].tolist() == pytest.approx([0.004, 0.002, 0.])
    assert command.metrics["error_vel_yaw"].tolist() == pytest.approx([0.001, 0.001, 0.])


def test_command_zero_active_rows_is_a_pure_noop(env_module):
    command = _command(env_module)
    before = (command.command.clone(), command.time_left.clone(), command.command_counter.clone(),
              {key: value.clone() for key, value in command.metrics.items()})
    command.compute(0.02, env_ids=torch.empty(0, dtype=torch.long))
    assert torch.equal(command.command, before[0])
    assert torch.equal(command.time_left, before[1])
    assert torch.equal(command.command_counter, before[2])
    for key, value in command.metrics.items():
        assert torch.equal(value, before[3][key])


def test_command_reset_postprocesses_first_frame_without_advancing_time(env_module):
    command = _command(env_module)
    command.cfg.rel_standing_envs = 1.
    before = command.command[2].clone()
    command.reset(torch.tensor([0, 1]))
    assert not command.command[:2].any()
    assert command.time_left.tolist() == [10., 10., 5.]
    assert command.command_counter.tolist() == [1, 1, 0]
    assert torch.equal(command.command[2], before)


def test_command_resampling_random_streams_are_independent_between_rows(env_module):
    first, second = _command(env_module), _command(env_module)
    for _ in range(3):
        first.reset(torch.tensor([0]))
    first.reset(torch.tensor([1]))
    second.reset(torch.tensor([1]))
    assert torch.equal(first.command[1], second.command[1])
    assert torch.equal(first.heading_target[1], second.heading_target[1])


class _Robot:
    def __init__(self, num_envs):
        n = num_envs
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros(n, 23), default_joint_vel=torch.zeros(n, 23),
            joint_pos=torch.zeros(n, 23), joint_vel=torch.zeros(n, 23),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            root_ang_vel_b=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3),
            root_ang_vel_w=torch.zeros(n, 3), root_lin_vel_w=torch.zeros(n, 3),
            body_state_w=torch.zeros(n, 5, 13))
        self.data.body_state_w[..., 3] = 1.
        self.targets = []

    def find_bodies(self, names, preserve_order=True):
        return ([0], [names]) if isinstance(names, str) else ([1, 2], names)

    def find_joints(self, names, preserve_order=True):
        return list(range(len(names))), names

    def set_joint_position_target(self, targets):
        self.targets.append(targets.clone())


class _Scene(dict):
    def __init__(self, env):
        super().__init__(robot=env.robot)
        self.env = env
        self.sensors = {"contact_sensor": env.contact_sensor}
        self.reset_rows, self.write_count, self.update_count = [], 0, 0
        self.terrain = SimpleNamespace(terrain_levels=torch.tensor([2, 4]),
                                       env_origins=torch.tensor([[0., 0., 0.], [4., 0., 0.]]))

    def reset(self, ids):
        self.reset_rows.append(ids.clone())
        self.env.robot.data.root_ang_vel_b[ids] = 0.
        self.env.robot.data.projected_gravity_b[ids] = torch.tensor([0., 0., -1.])
        self.env.contact_sensor.data.net_forces_w_history[ids] = 0.

    def write_data_to_sim(self):
        self.write_count += 1

    def update(self, dt):
        self.update_count += 1
        self.env.lp_sensor.scan_counts += 1


class _Physics:
    def __init__(self, env, fail_at=None, fail_rows=(0,)):
        self.env, self.fail_at, self.fail_rows = env, fail_at, fail_rows
        self.steps, self.forwards = 0, 0

    def step(self, render=False):
        self.steps += 1
        self.env.robot.data.root_ang_vel_b[:, 0] = self.steps / 4
        if self.fail_at is not None and self.steps >= 4 * self.fail_at:
            self.env.contact_sensor.data.net_forces_w_history[list(self.fail_rows), :, 0, 0] = 2.

    def forward(self):
        self.forwards += 1
        self.env.lp_sensor.scan_counts += 1


class _Policy:
    def __init__(self):
        self.inputs = []

    def evaluate(self, obs):
        self.inputs.append(obs.clone())
        n = len(obs)
        return SimpleNamespace(lp_action=torch.full((n, 23), 0.8),
                               vp_action=torch.full((n, 23), -0.6))


def _make_env(env_module, *, fail_at=None, fail_rows=(0,)):
    """Use real init_buffers, sensors, observations, rewards and step methods."""
    env = env_module.G1SelectorEnv.__new__(env_module.G1SelectorEnv)
    env.device, env.num_envs, env.headless = "cpu", 2, True
    env.physics_dt, env.low_level_dt, env.high_level_dt, env.step_dt = 0.005, 0.02, 0.2, 0.2
    state_names = ("mode_vp", "mode_lp", "vp_old_valid_ratio", "vp_new_valid_ratio", "lp_valid_ratio",
                   "vp_source_age", "lp_source_age", "time_since_switch")
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(max_episode_length_s=20.), sim=SimpleNamespace(decimation=4),
        selector=SimpleNamespace(initial_mode=0, low_level_steps_per_decision=10),
        normalization=SimpleNamespace(clip_actions=100., clip_observations=100., height_scan_offset=0.5,
                                      obs_scales=SimpleNamespace(ang_vel=1., projected_gravity=1., commands=1.,
                                          joint_pos=1., joint_vel=1., actions=1., lin_vel=1., height_scan=1.)),
        robot=SimpleNamespace(action_scale=0.25, terminate_contacts_body_names=["torso_link"],
                              feet_body_names=["left", "right"], depth_crop=(0, 0, 0, 0), limit_angle=0.8),
        domain_rand=SimpleNamespace(action_delay=SimpleNamespace(params={"max_delay": 5})),
        observation=SimpleNamespace(actor_obs_dim=3466, critic_obs_dim=3660, state_names=state_names,
                                     sensor_age_clip_s=2., switch_time_clip_s=2.,
                                     use_depth_valid_support=True, use_height_valid_mask=True,
                                     critic_use_privileged=True, critic_use_clean_height=True),
        noise=SimpleNamespace(add_noise=False), amp_observation_dim=58)
    env.robot = _Robot(2)
    env.contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=torch.zeros(2, 3, 3, 3)))
    env.left_feet_ray_caster = env.right_feet_ray_caster = None
    hits = torch.zeros(2, 187, 3)
    pos = torch.tensor([[0., 0., 0.6]]).repeat(2, 1)
    starts = torch.zeros(2, 187, 3)
    starts[..., 2] = 0.6
    env.height_scanner = SimpleNamespace(data=SimpleNamespace(pos_w=pos.clone(), ray_hits_w=hits.clone()),
        ray_starts_w=starts.clone(), cfg=SimpleNamespace(max_distance=40.))
    env.lp_sensor = SimpleNamespace(data=SimpleNamespace(pos_w=pos.clone(), ray_hits_w=hits.clone()),
        ray_starts_w=starts.clone(), cfg=SimpleNamespace(min_distance=0., max_distance=40.),
        scan_counts=torch.zeros(2, dtype=torch.long))
    env.lp_runtime_cfg = SimpleNamespace(height_clip=(-1., 1.), invalid_height=-1.1)
    env.camera = SimpleNamespace(data=SimpleNamespace(output={
        "distance_to_image_plane": torch.ones(2, 18, 32, 1)}))
    env.scene = _Scene(env)
    env.sim = _Physics(env, fail_at, fail_rows)
    env.sensor_interval = 5
    env.init_buffers()
    env.frozen_policy = _Policy()
    env.faults = env_module.HighLevelSensorFaultPipeline(
        2, env_module.HighLevelFaultCfg(enable=False), seed=42)
    env.sensor_observation = env.faults.read()
    command = SimpleNamespace(command=torch.zeros(2, 3), updates=torch.zeros(2, dtype=torch.long))
    command.compute = lambda dt, ids: command.updates.index_add_(0, ids, torch.ones_like(ids))
    command.reset = lambda ids: {}
    env.command_generator = command
    env.reward_manager = _reward_manager(env_module, env)
    env.event_manager = SimpleNamespace(available_modes=[])
    env.reset()
    return env


@pytest.mark.parametrize("column,dtype", [(False, torch.int64), (True, torch.int32)])
def test_macro_step_runs_ten_controls_forty_physics_steps_and_switches_directly(env_module, column, dtype):
    env = _make_env(env_module)
    mode = torch.tensor([0, 1], dtype=dtype)
    mode = mode[:, None] if column else mode
    initial_captures = env.sensor_capture_count.clone()
    actor, reward, done, extras = env.step(mode)
    assert env.sim.steps == env.scene.update_count == 40
    assert env.sim_step_counter == 40
    assert extras["low_level_steps"].tolist() == [10, 10]
    assert env.sensor_capture_count.tolist() == (initial_captures + 2).tolist()
    assert env.command_generator.updates.tolist() == [10, 10]
    assert len(env.frozen_policy.inputs) == 10
    assert not done.any()
    assert reward.tolist() == pytest.approx([1.03, 1.01], abs=1e-6)
    assert actor.shape == (2, 3466) and extras["observations"]["critic"].shape == (2, 3660)
    for target in env.robot.targets:
        assert torch.allclose(target[0], torch.full((23,), -0.15))
        assert torch.allclose(target[1], torch.full((23,), 0.2))
    for frozen_input in env.frozen_policy.inputs:
        assert frozen_input.shape == (2, 2120)
        assert frozen_input[:, -1].tolist() == [0., 1.]
    assert env.reward_manager._episode_sums["mode_switch_penalty"].tolist() == pytest.approx([0., -0.02])
    env.step(mode)
    # Repeating an accepted mode is not another switch.
    assert env.reward_manager._episode_sums["mode_switch_penalty"].tolist() == pytest.approx([0., -0.02])


def test_first_done_is_latched_and_dead_rows_stop_reward_history_commands_and_fault_time(env_module):
    env = _make_env(env_module, fail_at=4)
    actor, reward, done, extras = env.step(torch.tensor([0, 1]))
    assert env.sim.steps == 40
    assert done.tolist() == [True, False]
    assert extras["time_outs"].tolist() == [False, False]
    assert extras["low_level_steps"].tolist() == [4, 10]
    assert env.command_generator.updates.tolist() == [4, 10]
    assert [len(obs) for obs in env.frozen_policy.inputs] == [2] * 4 + [1] * 6
    assert env.sensor_capture_count.tolist() == [2, 3]  # First frame + terminal reset / two updates.
    assert reward.tolist() == pytest.approx([-3.588, 1.01], abs=1e-6)
    assert extras["terminal_env_ids"].tolist() == [0]
    assert extras["terminal_actor_observations"].shape == (1, 3466)
    assert extras["terminal_critic_observations"].shape == (1, 3660)
    terminal_actor = extras["terminal_actor_observations"].clone()
    terminal_critic = extras["terminal_critic_observations"].clone()
    # The last history frame is the fourth effective control, not control ten.
    assert terminal_actor[0, 9 * 78] == 4.
    assert actor[0, 9 * 78] == 0.
    assert env.low_level_episode_length_buf.tolist() == [0, 10]
    assert env.sensor_control_ticks.tolist() == [0, 10]
    assert env.actor_obs_buffer.num_pushes.tolist() == [1, 11]
    assert env.action_buffer._circular_buffer.num_pushes.tolist() == [0, 10]
    assert env.reward_manager._episode_sums["termination_penalty"].tolist() == [0., 0.]
    # Reset and later writes cannot modify the independently copied terminal rows.
    env.actor_obs.fill_(17.)
    env.critic_obs.fill_(19.)
    assert torch.equal(extras["terminal_actor_observations"], terminal_actor)
    assert torch.equal(extras["terminal_critic_observations"], terminal_critic)


@pytest.mark.parametrize("actions", [
    torch.tensor([0., 1.]), torch.tensor([False, True]), torch.tensor([0, 2]),
    torch.tensor([-1, 0]), torch.tensor([[0, 1]]), torch.tensor([0]), [0, 1],
])
def test_invalid_mode_actions_do_not_advance_physics(env_module, actions):
    env = _make_env(env_module)
    before = env.renet_estimator_mask.clone()
    with pytest.raises(ValueError):
        env.step(actions)
    assert env.sim.steps == 0
    assert torch.equal(before, env.renet_estimator_mask)


def test_observations_share_exact_frozen_inputs_and_cached_reads_are_pure(env_module):
    env = _make_env(env_module)
    env.step(torch.tensor([0, 1]))
    actor, critic, low = env.actor_obs, env.critic_obs, env.low_level_obs
    assert torch.equal(actor[:, :780], low[:, :780])
    assert torch.equal(actor[:, 780:780 + 1152], low[:, 780 + 187:780 + 187 + 1152])
    assert torch.equal(actor[:, 780 + 2304:780 + 2304 + 187], low[:, 780:780 + 187])
    assert torch.equal(critic[:, :3466], actor)
    state = actor[:, -8:]
    assert state[:, :2].tolist() == [[1., 0.], [0., 1.]]
    assert torch.equal(state[:, 2:5], torch.ones(2, 3))
    assert state[:, 5:7].tolist() == [[0., 0.], [0., 0.]]
    before = dict(proprio=env.actor_obs_buffer.buffer.clone(),
                  push=env.actor_obs_buffer.num_pushes.clone(),
                  sequence=env.sensor_sequence.clone(), captures=env.sensor_capture_count.clone(),
                  clock=env.sensor_control_ticks.clone(), schedule=env.faults.schedule.frame.clone(),
                  actor=actor.clone(), critic=critic.clone())
    for _ in range(50):
        obs, extras = env.get_observations()
        assert torch.equal(obs, before["actor"])
        assert torch.equal(extras["observations"]["critic"], before["critic"])
    assert torch.equal(env.actor_obs_buffer.buffer, before["proprio"])
    assert torch.equal(env.actor_obs_buffer.num_pushes, before["push"])
    assert torch.equal(env.sensor_sequence, before["sequence"])
    assert torch.equal(env.sensor_capture_count, before["captures"])
    assert torch.equal(env.sensor_control_ticks, before["clock"])
    assert torch.equal(env.faults.schedule.frame, before["schedule"])


def test_sensor_capture_preserves_support_before_filling_and_uses_meter_depth(env_module):
    env = _make_env(env_module)
    env.camera.data.output["distance_to_image_plane"][0, 0, 0] = 0.
    env.camera.data.output["distance_to_image_plane"][0, 0, 1] = float("nan")
    env.lp_sensor.data.ray_hits_w[0, 0] = float("nan")
    env.sensor_control_ticks[:] = 5
    env.lp_sensor.scan_counts += 1
    env._capture_sensor_frames(torch.tensor([0, 1]))
    env.compute_observations()
    sensor = env.sensor_observation
    assert sensor.vp[0, -1, 0, 0] == pytest.approx(0.04)
    assert sensor.vp[0, -1, 0, 1] == 1.
    assert sensor.vp[1, -1, 0, 0] == pytest.approx(1. / 2.5)
    assert sensor.vp_valid_support[0, -1, 0, :2].tolist() == [0., 0.]
    assert sensor.vp_valid_support[0, 0, 0, :2].tolist() == [1., 1.]
    assert sensor.lp[0, 0, 0] == pytest.approx(-1.1)
    assert sensor.lp_valid_support[0, 0, 0] == 0.
    assert sensor.vp_source_stamp[0].tolist() == pytest.approx([0., 0.1])
    assert sensor.lp_source_stamp.tolist() == pytest.approx([0.1, 0.1])
    # Frozen inputs use the same filled values, without support channels.
    assert env.low_level_obs[0, 780] == pytest.approx(-1.1)
    assert env.low_level_obs[0, 780 + 187 + 576] == pytest.approx(0.04)


def test_actual_action_is_clipped_before_buffering_and_scaling(env_module):
    env = _make_env(env_module)
    env.frozen_policy.evaluate = lambda obs: SimpleNamespace(
        vp_action=torch.full((len(obs), 23), -1000.), lp_action=torch.full((len(obs), 23), 1000.))
    env.step(torch.tensor([0, 1]))
    assert env.action[:, 0].tolist() == [-100., 100.]
    assert env.action_buffer._circular_buffer.buffer[:, -1, 0].tolist() == [-100., 100.]
    assert env.processed_actions[:, 0].tolist() == [-25., 25.]


def test_action_rate_penalty_uses_actual_mode_transition_actions(env_module):
    env = _make_env(env_module)
    env.step(torch.tensor([0, 0]))
    _, reward, _, _ = env.step(torch.tensor([1, 0]))
    expected_rate = -0.01 * 0.02 * 23 * (0.8 - (-0.6)) ** 2
    assert env.reward_manager._episode_sums["action_rate_l2"].tolist() == pytest.approx([expected_rate, 0.])
    assert reward.tolist() == pytest.approx([1.03 - 0.02 + expected_rate, 1.03], abs=1e-6)


def test_partial_reset_preserves_other_rows_and_fixed_terrain(env_module):
    env = _make_env(env_module)
    env.step(torch.tensor([0, 1]))
    before = {
        "actor": env.actor_obs[1].clone(), "critic": env.critic_obs[1].clone(),
        "body": env.actor_obs_buffer.buffer[1].clone(),
        "actions": env.action_buffer._circular_buffer.buffer[1].clone(),
        "clock": env.sensor_control_ticks[1].clone(), "sequence": env.sensor_sequence[1].clone(),
        "lp": env.faults.lp[1].clone(), "vp": env.faults.vp[1].clone(),
        "schedule": env.faults.schedule.frame[1].clone(),
        "levels": env.scene.terrain.terrain_levels.clone(), "origins": env.scene.terrain.env_origins.clone(),
    }
    physics_steps = env.sim.steps
    env.reset(torch.tensor([0]))
    assert env.sim.steps == physics_steps
    assert torch.equal(env.actor_obs[1], before["actor"])
    assert torch.equal(env.critic_obs[1], before["critic"])
    assert torch.equal(env.actor_obs_buffer.buffer[1], before["body"])
    assert torch.equal(env.action_buffer._circular_buffer.buffer[1], before["actions"])
    assert torch.equal(env.sensor_control_ticks[1], before["clock"])
    assert torch.equal(env.sensor_sequence[1], before["sequence"])
    assert torch.equal(env.faults.lp[1], before["lp"])
    assert torch.equal(env.faults.vp[1], before["vp"])
    assert torch.equal(env.faults.schedule.frame[1], before["schedule"])
    assert torch.equal(env.scene.terrain.terrain_levels, before["levels"])
    assert torch.equal(env.scene.terrain.env_origins, before["origins"])
    assert not env.action_buffer._circular_buffer.buffer[0].any()
    assert env.actor_obs_buffer.num_pushes[0] == 1


@pytest.mark.parametrize("contact,tilt,timed_out,expected", [
    (0., 0., False, (False, False)),
    (1., 0., False, (False, False)),
    (1.001, 0., False, (True, False)),
    (0., 0.7999, False, (False, False)),
    (0., 0.8001, False, (True, False)),
    (0., 0., True, (True, True)),
    (1.001, 0., True, (True, False)),
    (0., 0.8001, True, (True, False)),
])
def test_contact_tilt_thresholds_and_failure_priority(env_module, contact, tilt, timed_out, expected):
    env = _make_env(env_module)
    env.contact_sensor.data.net_forces_w_history[0, 0, 0, 0] = contact
    # Large foot contacts must not trigger the torso termination condition.
    env.contact_sensor.data.net_forces_w_history[:, :, 1:, 2] = 100.
    env.robot.data.projected_gravity_b[0, 2] = -math.cos(tilt)
    if timed_out:
        env.low_level_episode_length_buf[0] = env.low_level_max_episode_length
    done, timeout = env.check_reset()
    assert (done[0].item(), timeout[0].item()) == expected
    assert (done[1].item(), timeout[1].item()) == (False, False)


def test_tilt_acos_input_is_clamped_against_numerical_roundoff(env_module):
    env = _make_env(env_module)
    env.robot.data.projected_gravity_b[:, 2] = torch.tensor([-1.00001, 1.00001])
    done, timeout = env.check_reset()
    assert done.tolist() == [False, True]
    assert not timeout.any()


def test_failure_and_timeout_capture_first_terminal_before_reset_and_finish_global_period(env_module):
    env = _make_env(env_module, fail_at=4)
    env.low_level_episode_length_buf[:] = env.low_level_max_episode_length - 4
    actor, reward, done, extras = env.step(torch.tensor([0, 0]))
    assert env.sim.steps == 40
    assert len(env.frozen_policy.inputs) == 4
    assert env.command_generator.updates.tolist() == [4, 4]
    assert done.tolist() == [True, True]
    assert extras["low_level_steps"].tolist() == [4, 4]
    assert extras["time_outs"].tolist() == [False, True]
    assert extras["terminal_env_ids"].tolist() == [0, 1]
    assert extras["terminal_actor_observations"][:, 9 * 78].tolist() == [4., 4.]
    assert extras["terminal_critic_observations"][:, 9 * 78].tolist() == [4., 4.]
    assert actor[:, 9 * 78].tolist() == [0., 0.]
    assert reward.tolist() == pytest.approx([-3.588, 0.412], abs=1e-6)
    assert env.sensor_capture_count.tolist() == [2, 2]


def _contract_env(env_module):
    env = env_module.G1SelectorEnv.__new__(env_module.G1SelectorEnv)
    env.lp_observation_contract = observation_contract(
        LPHeightCfg(), "pgtt_full_atten", height_offset=0.5, height_scale=1.)
    env.frozen_policy_metadata = {"lp_observation_contract": copy.deepcopy(env.lp_observation_contract)}
    return env


@pytest.mark.parametrize("source", ["pgtt_full", "pgtt_full_atten"])
def test_checkpoint_contract_accepts_only_known_name_alias_without_mutating_metadata(env_module, source):
    env = _contract_env(env_module)
    env.frozen_policy_metadata["lp_observation_contract"]["source"] = source
    before = copy.deepcopy(env.frozen_policy_metadata)
    env._validate_lp_observation_contract()
    assert env.frozen_policy_metadata == before


@pytest.mark.parametrize("field,value", [
    ("height_offset", 0.6), ("shape", [17, 11]), ("source", "pgtt_unknown"),
    ("version", 2), ("height_scale", 2.),
])
def test_checkpoint_contract_rejects_numerical_geometry_and_unknown_source_mismatches(env_module, field, value):
    env = _contract_env(env_module)
    env.frozen_policy_metadata["lp_observation_contract"].update(source="pgtt_full")
    env.frozen_policy_metadata["lp_observation_contract"][field] = value
    before = copy.deepcopy(env.frozen_policy_metadata)
    with pytest.raises(ValueError, match="contract"):
        env._validate_lp_observation_contract()
    assert env.frozen_policy_metadata == before


def test_checkpoint_contract_is_required(env_module):
    env = _contract_env(env_module)
    env.frozen_policy_metadata = {}
    with pytest.raises(ValueError, match="missing"):
        env._validate_lp_observation_contract()

