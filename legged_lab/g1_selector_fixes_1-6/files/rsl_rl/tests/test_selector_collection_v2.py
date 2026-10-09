"""CPU regressions for the six collection fixes; no Isaac/ROS installation needed."""

from types import SimpleNamespace
import ast
from dataclasses import replace
from pathlib import Path
import copy
import json
import pytest
import torch
from torch import nn

from legged_lab.selector.collector import (
    CounterfactualCollector,
    FaultedObservationStream,
)
from legged_lab.selector.config import CollectionConfig, FaultConfig, RuntimeConfig
from legged_lab.selector.contracts import DualBranchOutput, SelectorInput
from legged_lab.selector.dataset import (
    SelectorBatch,
    SelectorDatasetWriter,
    SelectorShardDataset,
)
from legged_lab.selector.deployment import OnlineSelectorRuntime
from legged_lab.selector.feature_contract import (
    dataset_feature_contract,
    selector_feature_contract,
)
from legged_lab.selector.features import SensorHealthTracker
from legged_lab.selector.faults import FaultKind, SelectorFaultManager
from legged_lab.selector.randomness import (
    EnvironmentRandomness,
    PerEnvironmentRNG,
    bind_environment_randomness,
)
from legged_lab.selector.runtime import ActionBlender
from legged_lab.perception.lp_height import LPHeightCfg, PgttMapNoise
from test_selector_safety_labels import _AutoResetEnv


def test_health_polling_rate_does_not_erase_packet_statistics():
    slow, fast = (
        SensorHealthTracker(1, (2, 2), "cpu"),
        SensorHealthTracker(1, (2, 2), "cpu"),
    )
    origin = 1_790_000_000.0
    support = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    for index in range(12):
        data = torch.full((1, 2, 2), index * 0.1)
        stamp = torch.tensor([origin + index * 0.1], dtype=torch.float64)
        expected = slow.update(data, stamp, stamp, valid_mask=support)
        actual = fast.update(data, stamp, stamp, valid_mask=support)
        torch.testing.assert_close(expected.as_tensor(), actual.as_tensor())
        for poll in range(1, 5):
            actual = fast.update(data, stamp, stamp + poll * 0.02, valid_mask=support)
            torch.testing.assert_close(actual.temporal_l1, expected.temporal_l1)
            torch.testing.assert_close(actual.frozen_ratio, expected.frozen_ratio)
            assert actual.frame_age.item() == pytest.approx(poll * 0.02, abs=3e-7)
    assert actual.valid_ratio.item() == 0.75


def test_health_distinguishes_source_age_from_new_local_map_packet():
    tracker = SensorHealthTracker(1, (2,), "cpu")
    source = torch.tensor([100.0], dtype=torch.float64)
    tracker.update(torch.zeros(1, 2), source, 100.0, packet_timestamp=source)
    output = tracker.update(
        torch.ones(1, 2), source, 100.1, packet_timestamp=source + 0.1
    )
    assert output.temporal_l1.item() == 1.0
    assert output.frame_age.item() == pytest.approx(0.1)
    # Data in an invalid previous pixel cannot create a temporal-change spike.
    tracker.reset()
    tracker.update(
        torch.tensor([[99.0, 0.0]]),
        source,
        100.0,
        valid_mask=torch.tensor([[0.0, 1.0]]),
    )
    output = tracker.update(
        torch.ones(1, 2), source + 0.1, 100.1, valid_mask=torch.ones(1, 2)
    )
    assert output.temporal_l1.item() == 1.0


def test_named_rng_neighbor_resets_and_subset_order_are_independent():
    first, second = (
        EnvironmentRandomness(3, 42, "cpu"),
        EnvironmentRandomness(3, 42, "cpu"),
    )
    for _ in range(8):
        a = first.normal([0], (78,), "proprio")
        second.normal([1, 2], (78,), "proprio")
        b = second.normal([0], (78,), "proprio")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    state = first.snapshot()
    expected = first.normal([2, 0], (1152,), "camera")
    first.restore(state)
    actual = first.normal([0, 2], (1152,), "camera")
    torch.testing.assert_close(expected.flip(0), actual, rtol=0, atol=0)
    samples = PerEnvironmentRNG(1, 42).normal(torch.tensor([0]), (100000,))
    assert abs(samples.mean().item()) < 0.02 and abs(samples.std().item() - 1.0) < 0.02


def test_command_events_and_interval_timers_are_row_local():
    def make_env():
        env = SimpleNamespace(num_envs=3)
        env.command_generator = SimpleNamespace(command=torch.zeros(3, 3))
        env.command_generator._resample = lambda ids: (
            env.command_generator.command.__setitem__(ids, torch.rand(len(ids), 3))
        )
        env.push = torch.zeros(3, 3)
        cfg = SimpleNamespace(
            func=lambda e, ids: e.push.__setitem__(ids, torch.rand(len(ids), 3)),
            interval_range_s=(1.0, 2.0),
            is_global_time=False,
            params={},
        )
        manager = SimpleNamespace(
            _mode_term_cfgs={"interval": [cfg]},
            _interval_term_time_left=[torch.zeros(3)],
        )

        def apply(mode, *, dt):
            timer = manager._interval_term_time_left[0]
            timer -= dt
            ids = (timer < 1e-6).nonzero().flatten()
            timer[ids] = torch.rand(len(ids)) + 1.0
            cfg.func(env, ids)

        manager.apply = apply
        manager.reset = lambda ids: manager._interval_term_time_left[0].__setitem__(
            ids, torch.rand(len(ids)) + 1.0
        )
        env.event_manager = manager
        rng = EnvironmentRandomness(3, 42, "cpu")
        bind_environment_randomness(env, rng)
        return env, rng

    a, arng = make_env()
    b, brng = make_env()
    for _ in range(5):
        b.command_generator._resample(torch.tensor([1, 2]))
        b.event_manager.reset(torch.tensor([1, 2]))
        for env in (a, b):
            env.command_generator._resample(torch.tensor([0]))
            env.event_manager._interval_term_time_left[0][0] = 0.0
            env.event_manager.apply("interval", dt=0.02)
        torch.testing.assert_close(
            a.command_generator.command[0],
            b.command_generator.command[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(a.push[0], b.push[0], rtol=0, atol=0)
        torch.testing.assert_close(
            a.event_manager._interval_term_time_left[0][0],
            b.event_manager._interval_term_time_left[0][0],
            rtol=0,
            atol=0,
        )
    state = arng.snapshot()
    a.command_generator._resample(torch.tensor([0]))
    expected = a.command_generator.command[0].clone()
    arng.restore(state)
    a.command_generator._resample(torch.tensor([0]))
    torch.testing.assert_close(expected, a.command_generator.command[0], rtol=0, atol=0)


def test_pgtt_sensor_noise_survives_neighbor_resets_without_rng_coupling():
    a, b = PgttMapNoise(2, LPHeightCfg(), "cpu"), PgttMapNoise(2, LPHeightCfg(), "cpu")
    a.selector_randomness = EnvironmentRandomness(2, 123, "cpu")
    b.selector_randomness = EnvironmentRandomness(2, 123, "cpu")
    a.reset(torch.tensor([0, 1]))
    b.reset(torch.tensor([0, 1]))
    for index in range(6):
        b.reset(torch.tensor([1]))
        arguments = (
            torch.zeros(2, 187),
            torch.ones(2, 187).bool(),
            torch.ones(2),
            torch.full((2,), index).long(),
        )
        left = a.apply(*arguments, height_offset=0.0, height_scale=1.0)
        right = b.apply(*arguments, height_offset=0.0, height_scale=1.0)
        for x, y in zip(left, right):
            torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)


def test_camera_keeps_raw_valid_history_and_only_draws_for_updated_rows():
    # Execute the actual two environment methods without importing Isaac.
    source = Path(__file__).parents[2] / "legged_lab/envs/g1/RENet_env.py"
    klass = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    methods = [
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_processed_deepcamera", "get_deepcamera_history")
    ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))

    def noise(data, cfg, ids, **kwargs):
        return data + kwargs["standard_noise"] * 0.001

    namespace = {"torch": torch, "distance_dependent_gaussian_noise": noise}
    exec(compile(module, str(source), "exec"), namespace)

    def make_env():
        env = SimpleNamespace(
            num_envs=2, device="cpu", step_dt=0.02, depth_history_frames=2
        )
        image = torch.ones(2, 36, 64, 1)
        image[:, 18, 16, 0] = float("nan")
        image[:, 18, 17, 0] = 0.0
        env.camera = SimpleNamespace(
            data=SimpleNamespace(output={"distance_to_image_plane": image}),
            current_timestamps=torch.zeros(2, dtype=torch.float64),
        )
        env.cfg = SimpleNamespace(
            robot=SimpleNamespace(
                depth_crop=(18, 0, 16, 16),
                depth_min=0.1,
                depth_max=2.5,
                depth_update_interval=5,
            ),
            scene=SimpleNamespace(
                camera=SimpleNamespace(
                    camera=SimpleNamespace(
                        pattern_cfg=SimpleNamespace(height=36, width=64),
                        min_distance=0.1,
                    )
                )
            ),
            depth_blur_probability=0.0,
            depth_noise=None,
        )
        env.depth_buffer = torch.zeros(2, 2, 18, 32)
        env.depth_valid_buffer = torch.zeros_like(env.depth_buffer)
        env.depth_current_valid = torch.zeros(2, 18, 32)
        env.depth_initialized = torch.zeros(2).bool()
        env.depth_last_capture_time = torch.full((2,), -torch.inf, dtype=torch.float64)
        env.extras = {}
        env._depth_noise_is_active = lambda: True
        env.selector_randomness = EnvironmentRandomness(2, 321, "cpu")
        env.get_processed_deepcamera = lambda ids=None: namespace[
            "get_processed_deepcamera"
        ](env, ids)
        env.get_deepcamera_history = lambda: namespace["get_deepcamera_history"](env)
        return env

    a, b = make_env(), make_env()
    for env in (a, b):
        env.get_deepcamera_history()
    assert a.depth_valid_buffer[0, :, 0, :2].sum().item() == 0.0
    assert torch.isfinite(a.depth_buffer).all()
    b.camera.current_timestamps.fill_(0.02)
    b.depth_initialized[1] = False
    b.get_deepcamera_history()
    for env in (a, b):
        env.camera.current_timestamps.fill_(0.1)
        env.get_deepcamera_history()
    torch.testing.assert_close(a.depth_buffer[0], b.depth_buffer[0], rtol=0, atol=0)


def _sensor_inputs(count=2, timestamp=0.0):
    return (
        torch.zeros(count, 11, 17),
        torch.full((count, 2, 18, 32), 0.5),
        torch.full((count,), timestamp, dtype=torch.float64),
        torch.full((count,), timestamp, dtype=torch.float64),
    )


def test_fault_schedule_recurs_and_replays_after_other_row_reset():
    cfg = FaultConfig(
        max_start_delay_frames=0,
        min_duration_frames=3,
        max_duration_frames=8,
        min_recovery_frames=2,
        max_recovery_frames=3,
        scenario_probabilities=(0.0, 0.0, 1.0, 0.0),
    )
    left, right = SelectorFaultManager(2, cfg=cfg), SelectorFaultManager(2, cfg=cfg)
    for step in range(50):
        if step % 3 == 0:
            right.reset(torch.tensor([1]))
        a, b = (
            left.apply(*_sensor_inputs(timestamp=step * 0.1)),
            right.apply(*_sensor_inputs(timestamp=step * 0.1)),
        )
        torch.testing.assert_close(a.lp[0], b.lp[0], rtol=0, atol=0)
        torch.testing.assert_close(a.lp_valid[0], b.lp_valid[0], rtol=0, atol=0)
        assert left.lp_kind[0] == right.lp_kind[0]
    assert left.schedule_count[0] > 2
    snapshot = left.snapshot()
    expected = left.apply(*_sensor_inputs(timestamp=5.0))
    left.restore(snapshot)
    actual = left.apply(*_sensor_inputs(timestamp=5.0))
    for field in vars(expected):
        torch.testing.assert_close(
            getattr(expected, field), getattr(actual, field), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", [FaultKind.FREEZE, FaultKind.DELAY])
def test_fault_data_quality_source_and_packet_remain_atomic(kind):
    faults = SelectorFaultManager(1)
    faults.lp_kind.fill_(int(kind))
    faults.vp_kind.zero_()
    faults.lp_start_frame.zero_()
    faults.lp_duration_frames.fill_(100)
    faults.next_schedule_frame.fill_(1000)
    faults.lp_delay.fill_(1)
    lp, vp, source, stamp = _sensor_inputs(1)
    faults.apply(
        lp, vp, source, stamp, lp_valid=torch.ones_like(lp), lp_packet_timestamp=source
    )
    output = faults.apply(
        lp + 1,
        vp,
        source + 0.1,
        stamp + 0.1,
        lp_valid=torch.zeros_like(lp),
        lp_packet_timestamp=source + 0.1,
    )
    assert not output.lp.any() and output.lp_valid.all()
    assert (
        output.lp_timestamp.item() == 0.0 and output.lp_packet_timestamp.item() == 0.0
    )


def test_dual_faults_have_staggered_onsets_and_recoveries():
    faults = SelectorFaultManager(
        32, cfg=FaultConfig(scenario_probabilities=(0.0, 0.0, 0.0, 1.0))
    )
    assert (faults.lp_start_frame != faults.vp_start_frame).any()
    assert (
        faults.lp_start_frame + faults.lp_duration_frames
        != faults.vp_start_frame + faults.vp_duration_frames
    ).any()


@pytest.mark.parametrize("emergency,steps", [(False, 5), (True, 2)])
def test_action_transition_and_snapshot_match_deployment(emergency, steps):
    blender = ActionBlender(1, 23)
    blender.seed(torch.zeros(1, 23))
    flag = torch.ones(1).bool()
    outputs = [
        blender.step(
            torch.ones(1, 23), torch.zeros(1, 23), torch.ones(1), flag, flag & emergency
        )
    ]
    snapshot = blender.snapshot()
    for _ in range(1, steps):
        outputs.append(
            blender.step(
                torch.ones(1, 23), torch.zeros(1, 23), torch.ones(1), ~flag, ~flag
            )
        )
    assert [output[0, 0].item() for output in outputs] == pytest.approx(
        [(i + 1) / steps for i in range(steps)]
    )
    blender.restore(snapshot)
    if steps > 1:
        torch.testing.assert_close(
            blender.step(
                torch.ones(1, 23), torch.zeros(1, 23), torch.ones(1), ~flag, ~flag
            ),
            outputs[1],
        )
    assert torch.isfinite(
        blender.step(
            torch.ones(1, 23),
            torch.full((1, 23), float("nan")),
            torch.ones(1),
            ~flag,
            ~flag,
        )
    ).all()


class Adapter(nn.Module):
    policy = SimpleNamespace(
        proprio_actor_dim=780, lidar_flat_dim=187, depth_flat_dim=1152
    )

    def forward(self, observations):
        n = observations.shape[0]
        return (
            observations[:, 702:703].expand(n, 64),
            observations[:, 703:704].expand(n, 64),
            torch.zeros(n, 3),
            torch.zeros(n, 3),
            torch.full((n, 23), 0.8),
            torch.full((n, 23), -0.2),
        )

    def evaluate(self, observations):
        return DualBranchOutput(*self(observations))

    def normalize(self, observations):
        return observations


def test_stream_ticks_at_snapshot_boundary_and_reset_is_row_local():
    faults = SelectorFaultManager(
        2, cfg=FaultConfig(scenario_probabilities=(1.0, 0.0, 0.0, 0.0))
    )
    stream = FaultedObservationStream(Adapter(), faults)
    obs = torch.zeros(2, 2120)
    quality = dict(lp_valid=torch.ones(2, 11, 17), vp_valid=torch.ones(2, 2, 18, 32))
    clock = torch.zeros(2, dtype=torch.float64)
    stream.process(
        obs,
        lp_timestamp=clock,
        vp_timestamp=clock,
        now=0.0,
        force_update=True,
        advance_control_step=False,
        **quality,
    )
    ticks = []
    for step in range(1, 51):
        clock.fill_(step * 0.02)
        if step == 7:
            before = stream.faults.step_count[0].clone()
            stream.reset_envs(torch.tensor([1]))
        _, tick = stream.process(
            obs, lp_timestamp=clock, vp_timestamp=clock, now=step * 0.02, **quality
        )
        if step == 7:
            assert stream.faults.step_count[0] == before and not tick
        if tick:
            ticks.append(step)
    assert ticks == list(range(5, 51, 5))
    assert stream.lp_health_tracker.last_packet_timestamp.tolist() == [1.0, 1.0]


class FakeSimulator:
    def __init__(self):
        self._current_time = 400.0
        self._current_time_step_index = 0

    @property
    def current_time(self):
        return self._current_time

    def forward(self):
        pass


class FakeEnvironment(_AutoResetEnv):
    def __init__(self, count=8):
        super().__init__(count)
        self.num_actions, self.step_dt, self.physics_dt = 23, 0.02, 0.02
        self.sim_step_counter = 0
        self.sim = FakeSimulator()
        self.scene = SimpleNamespace(
            sensors={},
            write_data_to_sim=lambda: None,
            terrain=SimpleNamespace(
                terrain_levels=torch.zeros(count).long(),
                terrain_types=torch.arange(count),
            ),
        )
        self.actor_obs_buffer = SimpleNamespace(value=torch.zeros(count, 780))
        self.critic_obs_buffer = SimpleNamespace(value=torch.zeros(count, 1))
        self.action_buffer = SimpleNamespace(value=torch.zeros(count, 23))
        self.action = torch.zeros(count, 23)
        self.robot.data.root_state_w[:, 2] = 0.8
        self.robot.write_root_state_to_sim = lambda state: (
            self.robot.data.root_state_w.copy_(state)
        )
        self.robot.write_joint_state_to_sim = lambda p, v: (
            self.robot.data.joint_pos.copy_(p),
            self.robot.data.joint_vel.copy_(v),
        )
        self.command_generator._resample = lambda ids: None
        self.extras = {
            "observations": {
                "lp_valid_mask": torch.ones(count, 11, 17),
                "vp_valid_support": torch.ones(count, 2, 18, 32),
                "lp_frame_age": torch.zeros(count, 1),
            }
        }

    def get_observations(self):
        obs = torch.zeros(self.num_envs, 2120)
        obs[:, 702] = self.sim._current_time
        obs[:, 703:780] = self.action[:, 0:1]
        obs[:, 967:2119] = 0.5
        return obs, self.extras

    def step(self, actions):
        self.sim_step_counter += 1
        self.sim._current_time += 0.02
        self.sim._current_time_step_index += 1
        self.action.copy_(actions)
        self.episode_length_buf += 1
        self.robot.data.joint_pos += actions * 0.001
        self.reset(self.dones.nonzero().flatten())
        obs, _ = self.get_observations()
        return (
            obs,
            torch.zeros(self.num_envs),
            self.dones.clone(),
            {"time_outs": self.time_outs.clone()},
        )


def make_collector(root, **kwargs):
    env = FakeEnvironment()
    cfg = CollectionConfig(
        num_envs=8,
        target_samples=24,
        shard_size=8,
        allow_nonproduction_num_envs=True,
        balanced_scenarios=False,
        **kwargs,
    )
    writer = SelectorDatasetWriter(root, collection_cfg=cfg, fixed_split="test")
    faults = SelectorFaultManager(
        8, cfg=FaultConfig(scenario_probabilities=(1.0, 0.0, 0.0, 0.0))
    )
    return CounterfactualCollector(
        env, Adapter(), writer, faults=faults, collection_cfg=cfg
    )


def test_full_counterfactual_collection_aligns_time_state_and_replay(tmp_path):
    collector = make_collector(tmp_path)
    collector.collect()
    collector.writer.close()
    dataset = SelectorShardDataset(tmp_path, "test")
    assert len(dataset) == 24
    assert dataset_feature_contract(dataset.manifest) == selector_feature_contract()
    assert dataset.manifest["metadata"]["snapshot_replay_validation"]["passed"]
    assert dataset.manifest["metadata"]["coverage"]["accepted"] == [24, 0, 0, 0]
    times = []
    for row in dataset:
        torch.testing.assert_close(
            row["candidate_time"], row["feature_end_time"], rtol=0, atol=0
        )
        # Source features persisted in float16: use its quantization tolerance.
        assert row["features"][-1, 0].item() == pytest.approx(
            row["candidate_time"].item(), abs=0.13
        )
        times.append(row["candidate_time"].item())
    assert sorted(set(round(value, 5) for value in times)) == [400.0, 400.1, 400.2]


def test_transition_switch_time_is_recorded_before_first_action(tmp_path):
    collector = make_collector(tmp_path)
    collector.initialize()
    collector.behavior_mask.zero_()
    collector.blender.seed(torch.zeros(8, 23))
    old_clock = collector._time()
    collector._rollout(1.0, record_trajectory=False)
    assert collector.last_behavior_switch.tolist() == [old_clock] * 8
    assert collector.feature_history.buffer[0, -1, -1] == pytest.approx(1.0)


def test_incomplete_or_old_feature_datasets_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        dataset_feature_contract(
            {"metadata": {"selector_feature_contract": selector_feature_contract()}}
        )
    with pytest.raises(ValueError, match="recollect"):
        dataset_feature_contract({"metadata": {"collection_complete": True}})


def test_dataset_writer_rejects_misaligned_candidate(tmp_path):
    batch = SelectorBatch(
        torch.zeros(1, 5, 283),
        torch.zeros(1, 4),
        ["terrain"],
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        candidate_time=torch.tensor([1.0], dtype=torch.float64),
        feature_end_time=torch.tensor([0.92], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="aligned"):
        SelectorDatasetWriter(tmp_path).append(batch)


def test_online_mode_and_dwell_follow_executed_target_at_epoch_clock():
    class Risk(nn.Module):
        def forward(self, history):
            return torch.tensor([[0.01, 0.02, 0.9, 0.95]]).expand(history.shape[0], 4)

    runtime = OnlineSelectorRuntime(
        Adapter(), Risk(), runtime_cfg=RuntimeConfig(switch_confirmations=1)
    )
    origin = 1_790_000_000.0

    def value(now):
        stamp = torch.tensor([now], dtype=torch.float64)
        return SelectorInput(
            stamp,
            torch.zeros(1, 780),
            torch.zeros(1, 11, 17),
            stamp,
            torch.full((1, 2, 18, 32), 0.5),
            stamp,
            torch.zeros(1),
            torch.tensor([2.0]),
            lp_valid_mask=torch.ones(1, 11, 17),
            vp_valid_support=torch.ones(1, 2, 18, 32),
            lp_packet_timestamp=stamp,
        )

    first = runtime.step(value(origin))
    assert first.output.selected_mask.item() == 1.0
    runtime.step(value(origin + 0.1))
    assert runtime.history.buffer[0, -1, -2].item() == 1.0
    assert runtime.history.buffer[0, -1, -1].item() == pytest.approx(0.1, abs=3e-7)
    assert runtime.machine.last_switch_time.item() == origin


def test_shadow_features_follow_actual_mode_instead_of_recommendation():
    class Risk(nn.Module):
        def forward(self, history):
            return torch.tensor([[0.01, 0.02, 0.9, 0.95]])

    runtime = OnlineSelectorRuntime(
        Adapter(), Risk(), runtime_cfg=RuntimeConfig(switch_confirmations=1)
    )
    for now in (10.0, 10.1):
        stamp = torch.tensor([now], dtype=torch.float64)
        value = SelectorInput(
            stamp,
            torch.zeros(1, 780),
            torch.zeros(1, 11, 17),
            stamp,
            torch.full((1, 2, 18, 32), 0.5),
            stamp,
            torch.zeros(1),
            torch.tensor([2.0]),
            torch.ones(1, 11, 17),
            torch.ones(1, 2, 18, 32),
            stamp,
        )
        output = runtime.step(value, shadow=True)
        assert output.output.selected_mask.item() == 1.0
        assert output.control_action[0, 0].item() == pytest.approx(-0.2)
        assert runtime.history.buffer[0, -1, -2:].tolist() == [0.0, 2.0]


def test_accepted_scenario_quotas_are_enforced_after_filtering(tmp_path):
    collector = make_collector(tmp_path)
    collector.collection_cfg = replace(
        collector.collection_cfg, balanced_scenarios=True
    )
    faults = collector.stream.faults
    faults.cfg = replace(faults.cfg, scenario_probabilities=(0.25, 0.25, 0.25, 0.25))
    faults.lp_kind[:] = torch.tensor([0, 0, 0, 0, 10, 10, 10, 10])
    faults.vp_kind[:] = torch.tensor([0, 0, 4, 4, 0, 0, 4, 4])
    faults.lp_start_frame.zero_()
    faults.vp_start_frame.zero_()
    faults.start_frame.zero_()
    faults.lp_duration_frames.fill_(100)
    faults.vp_duration_frames.fill_(100)
    faults.duration_frames.fill_(100)
    faults.next_schedule_frame.fill_(200)
    collector.collect()
    collector.writer.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["metadata"]["coverage"]["quotas"] == [6] * 4
    assert manifest["metadata"]["coverage"]["accepted"] == [6] * 4


def test_candidate_limit_writes_partial_manifest_and_training_refuses_it(tmp_path):
    collector = make_collector(tmp_path, max_candidate_batches=1)
    with pytest.raises(RuntimeError, match="candidate limit"):
        collector.collect()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert sum(manifest["counts"].values()) == 8
    with pytest.raises(ValueError, match="incomplete"):
        dataset_feature_contract(manifest)


def test_shadow_to_auto_starts_blending_from_actual_executed_action():
    class Risk(nn.Module):
        def forward(self, history):
            return torch.tensor([[0.01, 0.02, 0.9, 0.95]])

    runtime = OnlineSelectorRuntime(
        Adapter(), Risk(), runtime_cfg=RuntimeConfig(switch_confirmations=1)
    )

    def value(now):
        stamp = torch.tensor([now], dtype=torch.float64)
        proprio = torch.zeros(1, 780)
        proprio[:, -23:] = -0.2
        return SelectorInput(
            stamp,
            proprio,
            torch.zeros(1, 11, 17),
            stamp,
            torch.full((1, 2, 18, 32), 0.5),
            stamp,
            torch.zeros(1),
            torch.tensor([2.0]),
            torch.ones(1, 11, 17),
            torch.ones(1, 2, 18, 32),
            stamp,
        )

    runtime.step(value(10.0), shadow=True)
    result = runtime.step(value(10.02), shadow=False)
    assert result.output.selected_mask.item() == 1.0
    assert result.control_action[0, 0].item() == pytest.approx(0.0, abs=1e-7)
