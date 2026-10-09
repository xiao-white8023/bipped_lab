# 中文导读
# 文件用途：回归验证 selector 标签修正、reset 前测量、超时处理和有效样本过滤。
# 使用位置：通过 pytest 执行；使用轻量环境替身及真实 G1 控制步源码，不启动 Isaac Sim。
# 说明：这些测试检查标签和采集接口，不能证明 PhysX 中的完整轨迹重放或实际控制效果。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - _AutoResetEnv：模拟 G1 控制步结束后自动 reset 的最小环境，用于检验标签读取时机。
# - _AutoResetEnv.__init__：建立本体、扫描、接触、命令及 episode 计数等测试状态。
# - _AutoResetEnv.reset：将指定机器人恢复为正常姿态、速度和接触状态，并清零 episode 计数。
# - _AutoResetEnv.step：推进计数、执行测试事件并自动 reset，返回 reset 后观测和终止信息。
# - _frame：生成指定倾角、高度、有效性和接触标记的 SafetyStepMetrics 测试帧。
# - _finish：用指定或正常安全帧补齐 rollout，并返回累计结果。
# - test_obstacle_ahead_is_not_ground_under_the_robot：确认前方高障碍不会替代机身附近地面而产生错误低高度标签。
# - test_nearest_ground_uses_world_root_position_and_actual_height：验证世界坐标地面定位和真实低高度失败检测。
# - test_missing_or_distant_ground_cannot_produce_a_safe_label：验证缺失、非有限或距离过远的地面测量被标记无效。
# - test_out_of_range_ray_return_is_an_unknown_ground_measurement：验证超出扫描最大距离的返回值不能充当有效地面。
# - test_nonfinite_simulator_state_is_rejected：验证机器人位置、重力或速度包含非有限值时拒绝标签。
# - _make_terminal：构造低高度、大倾角、高速度和躯干接触的终止状态。
# - test_terminal_metrics_are_from_before_reset_and_method_is_restored：验证终止指标来自 reset 前且不混入新 episode，并恢复原 reset 方法。
# - test_capture_boundary_in_the_actual_g1_control_step_source：执行真实 G1 step/check_reset 源码的替身版本，验证测量先于 reset 和命令更新。
# - test_hook_captures_even_when_no_environment_resets：验证 reset 环境集合为空时也捕获本控制步的安全指标。
# - test_hook_restores_existing_instance_override_on_exception：验证 step 抛异常时仍恢复此前实例上的 reset 覆盖。
# - test_hook_refuses_post_reset_fallback_when_boundary_is_missing：验证缺少 reset 前边界时抛错，不使用 reset 后指标替代。
# - test_early_timeout_is_censored_and_final_step_timeout_is_complete：验证提前超时剔除，而最后一步超时保留完整观测。
# - test_torso_contact_on_a_timeout_step_is_still_a_physical_failure：验证同时发生超时和真实躯干接触时仍记录物理失败。
# - test_late_failure_preserves_short_safe_long_failed_order：验证只在长时间窗出现失败时短标签仍安全、长标签为失败。
# - test_replay_filter_checks_label_validity_and_censoring：验证重放过滤也检查标签有效性与超时事件。
# - test_collector_rollout_consumes_terminal_capture_not_returned_state：验证实际 collector rollout 使用 reset 前指标生成标签。
# - test_collector_persists_only_complete_valid_pairs：验证仅写入两分支有效且有完整剩余时间窗的样本，并记录过滤计数。
# - test_old_label_datasets_are_rejected_even_with_matching_policy：验证策略相同也不能放行缺失或过时的标签约定。
# 中文导读结束

"""Regression cases for pre-reset selector labels, censoring and persistence."""

from __future__ import annotations

import copy
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from legged_lab.selector.collector import CounterfactualCollector, outcome_replay_stability
from legged_lab.selector.config import CollectionConfig
from legged_lab.selector.labels import (
    FUTURE_METRIC_ORDER, SafetyOutcomeAccumulator, SafetyStepMetrics,
    capture_safety_step, dataset_safety_label_contract, local_clearance,
    safety_label_contract, step_with_safety_capture,
)


class _AutoResetEnv:
    """A simulator-free environment with G1's step -> reset -> return boundary."""

    def __init__(self, count=1):
        self.num_envs = count
        self.device = "cpu"
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[0., 0., .8]]).repeat(count, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            root_lin_vel_b=torch.zeros(count, 3),
            root_ang_vel_b=torch.zeros(count, 3),
            root_state_w=torch.zeros(count, 13),
            joint_pos=torch.zeros(count, 23), joint_vel=torch.zeros(count, 23),
        ))
        self.height_scanner = SimpleNamespace(data=SimpleNamespace(
            ray_hits_w=torch.tensor([[[0., 0., 0.], [.5, 0., .4]]]).repeat(count, 1, 1)))
        self.contact_sensor = SimpleNamespace(data=SimpleNamespace(
            net_forces_w_history=torch.zeros(count, 3, 1, 3)))
        self.termination_contact_cfg = SimpleNamespace(body_ids=[0])
        self.command_generator = SimpleNamespace(command=torch.zeros(count, 3))
        self.cfg = SimpleNamespace(renet=SimpleNamespace(mask_mode="vp"))
        self.renet_estimator_mask = torch.zeros(count, 1)
        self.episode_length_buf = torch.zeros(count, dtype=torch.long)
        self.max_episode_length = 1000
        self.dones = torch.zeros(count, dtype=torch.bool)
        self.time_outs = torch.zeros_like(self.dones)
        self.before_reset = None
        self.control_steps = 0

    def reset(self, ids):
        self.robot.data.root_pos_w[ids, 2] = .8
        self.robot.data.projected_gravity_b[ids] = torch.tensor([0., 0., -1.])
        self.robot.data.root_lin_vel_b[ids] = 0
        self.robot.data.root_ang_vel_b[ids] = 0
        self.contact_sensor.data.net_forces_w_history[ids] = 0
        self.episode_length_buf[ids] = 0

    def step(self, actions):
        self.control_steps += 1
        self.episode_length_buf += 1
        if self.before_reset is not None:
            self.before_reset(self)
        dones, timeouts = self.dones.clone(), self.time_outs.clone()
        self.reset(dones.nonzero().flatten())
        # Reset values are deliberately different from the true terminal ones.
        return torch.zeros(self.num_envs, 1), torch.zeros(self.num_envs), dones, {"time_outs": timeouts}


def _frame(count=1, *, tilt=0., clearance=.8, valid=True, torso=False):
    return SafetyStepMetrics(
        torch.full((count,), tilt), torch.full((count,), clearance),
        torch.zeros(count), torch.full((count,), torso), torch.full((count,), valid),
    )


def _finish(accumulator, frame=None):
    while accumulator.step_index < accumulator.long_horizon_steps:
        accumulator.update(frame or _frame(accumulator.num_envs),
                           torch.zeros(accumulator.num_envs).bool(),
                           torch.zeros(accumulator.num_envs).bool())
    return accumulator.outcome()


def test_obstacle_ahead_is_not_ground_under_the_robot():
    env = _AutoResetEnv()
    clearance, valid = local_clearance(env)
    torch.testing.assert_close(clearance, torch.tensor([.8]))
    assert valid.item()
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1)
    accumulator.update(capture_safety_step(env, env.command_generator.command), env.dones, env.time_outs)
    outcome = accumulator.outcome()
    assert outcome.label_valid.item() and not outcome.fail_long.item()


def test_nearest_ground_uses_world_root_position_and_actual_height():
    env = _AutoResetEnv()
    env.robot.data.root_pos_w[:] = torch.tensor([[12., -7., 1.1]])
    env.height_scanner.data.ray_hits_w = torch.tensor([[[12., -7., .3], [12.5, -7., .9]]])
    clearance, valid = local_clearance(env)
    torch.testing.assert_close(clearance, torch.tensor([.8]))
    assert valid.item()
    env.robot.data.root_pos_w[:, 2] = .6
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1)
    accumulator.update(capture_safety_step(env, env.command_generator.command), env.dones, env.time_outs)
    assert accumulator.outcome().fail_short.item()


@pytest.mark.parametrize("hits", [None, [[[float("nan"), 0., 0.]]], [[[.5, 0., 0.]]]])
def test_missing_or_distant_ground_cannot_produce_a_safe_label(hits):
    env = _AutoResetEnv()
    if hits is None:
        env.height_scanner = None
    else:
        env.height_scanner.data.ray_hits_w = torch.tensor(hits)
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=2)
    accumulator.update(capture_safety_step(env, env.command_generator.command), env.dones, env.time_outs)
    outcome = _finish(accumulator)
    assert not outcome.label_valid.item()
    assert outcome.first_invalid_step.item() == 1


def test_out_of_range_ray_return_is_an_unknown_ground_measurement():
    env = _AutoResetEnv()
    env.height_scanner.ray_starts_w = torch.tensor([[[0., 0., 20.], [.5, 0., 20.]]])
    env.height_scanner.cfg = SimpleNamespace(max_distance=5.)
    assert not capture_safety_step(env, env.command_generator.command).valid.item()


@pytest.mark.parametrize("field", ["root_pos_w", "projected_gravity_b", "root_lin_vel_b", "root_ang_vel_b"])
def test_nonfinite_simulator_state_is_rejected(field):
    env = _AutoResetEnv()
    getattr(env.robot.data, field)[0, 0] = float("nan")
    metrics = capture_safety_step(env, env.command_generator.command)
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1)
    accumulator.update(metrics, env.dones, env.time_outs)
    assert not accumulator.outcome().label_valid.item()


def _make_terminal(env):
    env.robot.data.root_pos_w[:, 2] = .3
    env.robot.data.projected_gravity_b[:] = torch.tensor([0., 0., -.5])
    env.robot.data.root_lin_vel_b[:, 0] = 3.
    env.contact_sensor.data.net_forces_w_history[:, 0, 0, 2] = 2.
    env.dones[:] = True


def test_terminal_metrics_are_from_before_reset_and_method_is_restored():
    env = _AutoResetEnv()
    env.before_reset = _make_terminal
    original_reset = env.reset
    result, metrics = step_with_safety_capture(env, torch.zeros(1, 23), torch.zeros(1, 3))
    assert env.reset == original_reset and "reset" not in vars(env)
    assert env.robot.data.root_pos_w[0, 2] == pytest.approx(.8)
    assert metrics.clearance.item() == pytest.approx(.3)
    assert metrics.tilt.item() == pytest.approx(torch.pi / 3)
    assert metrics.tracking_error.item() == 9.
    assert metrics.torso_contact.item()
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=2)
    accumulator.update(metrics, result[2], result[3]["time_outs"])
    # New-episode states must not contribute to the original candidate.
    outcome = _finish(accumulator, _frame(tilt=2., clearance=-1.))
    assert outcome.fail_short.item() and outcome.fail_long.item() and outcome.label_valid.item()
    assert outcome.max_tilt.item() == pytest.approx(torch.pi / 3)
    assert outcome.min_clearance.item() == pytest.approx(.3)
    assert outcome.max_tracking_error.item() == 9.
    assert outcome.first_failure_step.item() == outcome.first_reset_step.item() == 1
    assert outcome.observed_steps.item() == 1


def test_capture_boundary_in_the_actual_g1_control_step_source():
    """Execute the real control-step method with simulator APIs replaced by stubs."""
    source = Path(__file__).parents[2] / "legged_lab/envs/g1/RENet_env.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("step", "check_reset")]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    env = _AutoResetEnv()
    env.robot.data.default_joint_pos = torch.zeros(1, 23)
    env.robot.data.root_quat_w = torch.tensor([[1., 0., 0., 0.]])
    env.robot.data.root_lin_vel_w = torch.zeros(1, 3)
    env.robot.set_joint_position_target = lambda actions: None
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kw: None)
    env.sim = SimpleNamespace(step=lambda **kw: _make_terminal(env))
    env.cfg.sim = SimpleNamespace(decimation=1)
    env.cfg.renet.domain_rand_start_iteration = 0
    env.action_buffer = SimpleNamespace(compute=lambda actions: actions, time_lags=torch.zeros(1), max_time_lag=0)
    env.clip_actions, env.action_scale, env.step_dt, env.physics_dt = 100., .25, .02, .02
    env.sim_step_counter, env.headless, env.extras = 0, True, {}
    env._domain_rand_is_active = lambda: False
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(1))
    env.get_amp_obs_for_expert_trans = lambda: torch.zeros(1, 50)
    env.compute_observations = lambda: (torch.zeros(1, 780), torch.zeros(1, 1037))
    env.append_renet_training_inputs = lambda obs: obs
    env.command_generator.compute = lambda dt: env.command_generator.command.fill_(100.)
    env.check_reset = lambda: namespace["check_reset"](env)
    env.step = lambda actions: namespace["step"](env, actions)
    result, metrics = step_with_safety_capture(env, torch.zeros(1, 23), env.command_generator.command.clone())
    assert result[2].item() and not result[3]["time_outs"].item()
    assert metrics.clearance.item() == pytest.approx(.3)
    assert metrics.tracking_error.item() == 9.
    assert metrics.torso_contact.item()
    assert env.command_generator.command[0, 0] == 100.
    assert env.robot.data.root_pos_w[0, 2] == pytest.approx(.8)


def test_hook_captures_even_when_no_environment_resets():
    env = _AutoResetEnv()
    env.before_reset = lambda value: value.robot.data.root_pos_w[:, 2].fill_(.3)
    result, metrics = step_with_safety_capture(env, torch.zeros(1, 23), torch.zeros(1, 3))
    assert not result[2].any() and metrics.clearance.item() == pytest.approx(.3)


def test_hook_restores_existing_instance_override_on_exception():
    env = _AutoResetEnv()
    previous = env.reset
    env.reset = previous
    env.step = lambda actions: (_ for _ in ()).throw(RuntimeError("physics failed"))
    with pytest.raises(RuntimeError, match="physics failed"):
        step_with_safety_capture(env, torch.zeros(1, 23), torch.zeros(1, 3))
    assert vars(env)["reset"] is previous


def test_hook_refuses_post_reset_fallback_when_boundary_is_missing():
    env = _AutoResetEnv()
    env.step = lambda actions: (torch.zeros(1, 1), torch.zeros(1), env.dones, {})
    with pytest.raises(RuntimeError, match="pre-reset"):
        step_with_safety_capture(env, torch.zeros(1, 23), torch.zeros(1, 3))
    assert "reset" not in vars(env)


@pytest.mark.parametrize("timeout_step,valid", [(1, False), (2, False), (3, True)])
def test_early_timeout_is_censored_and_final_step_timeout_is_complete(timeout_step, valid):
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=3)
    for step in range(1, 4):
        done = torch.tensor([step == timeout_step])
        accumulator.update(_frame(), done, done)
    outcome = accumulator.outcome()
    assert outcome.label_valid.item() == valid
    assert not outcome.fail_long.item() and not outcome.terminated.item()
    assert outcome.first_timeout_step.item() == timeout_step
    assert outcome.observed_steps.item() == timeout_step


def test_torso_contact_on_a_timeout_step_is_still_a_physical_failure():
    env = _AutoResetEnv()
    env.before_reset = _make_terminal
    env.time_outs[:] = True
    result, metrics = step_with_safety_capture(env, torch.zeros(1, 23), torch.zeros(1, 3))
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1)
    accumulator.update(metrics, result[2], result[3]["time_outs"])
    outcome = accumulator.outcome()
    assert outcome.label_valid.item() and outcome.fail_short.item()
    assert outcome.torso_contact.item() and not outcome.terminated.item()


def test_late_failure_preserves_short_safe_long_failed_order():
    accumulator = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=3)
    accumulator.update(_frame(), torch.tensor([False]), torch.tensor([False]))
    accumulator.update(_frame(tilt=1.), torch.tensor([False]), torch.tensor([False]))
    outcome = _finish(accumulator)
    assert not outcome.fail_short.item() and outcome.fail_long.item() and outcome.label_valid.item()
    assert outcome.first_failure_step.item() == 2


def test_replay_filter_checks_label_validity_and_censoring():
    first = _finish(SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1))
    second = copy.deepcopy(first)
    second.label_valid[:] = False
    assert not outcome_replay_stability(first, second).item()
    second = copy.deepcopy(first)
    second.first_timeout_step[:] = 1
    assert not outcome_replay_stability(first, second).item()


def test_collector_rollout_consumes_terminal_capture_not_returned_state():
    env = _AutoResetEnv()
    env.before_reset = lambda value: _make_terminal(value) if value.control_steps == 1 else None
    collector = CounterfactualCollector.__new__(CounterfactualCollector)
    collector.env, collector.device = env, torch.device("cpu")
    collector.collection_cfg = CollectionConfig(num_envs=1, allow_nonproduction_num_envs=True)
    collector.observations = torch.zeros(1, 1)
    collector.adapter = SimpleNamespace(evaluate=lambda obs: SimpleNamespace(lp_action=torch.zeros(1, 23), vp_action=torch.zeros(1, 23)))
    collector.stream = SimpleNamespace(depth_end=0, reset_envs=lambda ids: None,
                                       process=lambda obs, **kwargs: (obs, False))
    collector.feature_history = SimpleNamespace(reset=lambda ids: None)
    collector._sensor_timestamps = lambda: (torch.zeros(1), torch.zeros(1))
    collector._time = lambda: env.control_steps * .02
    collector.episode_serial = torch.zeros(1).long()
    outcome, trajectory = collector._rollout(0., preserve_as_behavior=True)
    assert outcome.label_valid.item() and outcome.fail_short.item()
    assert outcome.min_clearance.item() == pytest.approx(.3)
    assert outcome.max_tracking_error.item() == 9.
    assert trajectory.shape == (50, 1, 59)
    assert "reset" not in vars(env)


def test_collector_persists_only_complete_valid_pairs():
    count = 4
    env = _AutoResetEnv(count)
    env.episode_length_buf[:] = torch.tensor([0, 0, 951, 950])
    collector = CounterfactualCollector.__new__(CounterfactualCollector)
    collector.env, collector.device, collector.seed = env, torch.device("cpu"), 42
    collector.collection_cfg = CollectionConfig(num_envs=count, target_samples=1, allow_nonproduction_num_envs=True)
    collector.observations = torch.zeros(count, 1)
    collector.feature_history = SimpleNamespace(buffer=torch.arange(count).float()[:, None, None].expand(-1, 5, 283), initialized=torch.ones(count).bool())
    collector.stream = SimpleNamespace(control_steps_per_sensor_update=5, faults=SimpleNamespace(
        audit_metadata=lambda: torch.zeros(count, 8).long(),
        unobservable_onset_within=lambda frames: torch.zeros(count).bool()))
    collector.validate_replay = lambda: {"passed": True}
    collector._terrain_metadata = lambda: (torch.zeros(count).long(), torch.zeros(count).long())
    collector.behavior_mask, collector.last_behavior_switch = torch.zeros(count), torch.zeros(count)
    collector.batch_index = 0
    collector._time = lambda: 1.
    collector._capture = lambda: SimpleNamespace(behavior_mask=torch.zeros(count))
    collector._restore = lambda snapshot: None
    lp = _finish(SafetyOutcomeAccumulator(count, device="cpu", short_horizon_steps=1, long_horizon_steps=1))
    vp = copy.deepcopy(lp)
    lp.label_valid[0], vp.label_valid[1] = False, False
    collector._repeated_branch_rollout = lambda snapshot, mask, **kwargs: (lp if mask else vp, torch.ones(count).bool())
    saved = []
    collector.writer = SimpleNamespace(metadata={}, append=saved.append)
    collector.collect()
    assert len(saved) == 1
    # Row 2 has only 49 steps remaining. Row 3 has exactly 50 and is valid.
    assert saved[0].features[0, 0, 0].item() == 3.
    assert saved[0].future_metrics.shape == (1, 2, len(FUTURE_METRIC_ORDER))
    assert collector.writer.metadata["safety_label_filter"] == {
        "insufficient_episode_horizon": 1, "invalid_lp_outcome": 1, "invalid_vp_outcome": 1}
    assert dataset_safety_label_contract({"metadata": collector.writer.metadata}) == safety_label_contract()


def test_old_label_datasets_are_rejected_even_with_matching_policy():
    for contract in (None, {"version": 1}, {**safety_label_contract(), "ground_reference": "whole_scan_max"}):
        with pytest.raises(ValueError, match="recollect"):
            dataset_safety_label_contract({"metadata": {"safety_label_contract": contract}})
    assert dataset_safety_label_contract({"metadata": {"safety_label_contract": safety_label_contract()}}) == safety_label_contract()
