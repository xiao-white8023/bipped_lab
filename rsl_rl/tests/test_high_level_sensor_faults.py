"""Analytic sensor replay, scheduling and legacy Film compatibility checks."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch

from legged_lab.perception.structured_depth import StructuredDepthFailureModel
from legged_lab.selector.fault_kinds import FaultKind as K
from legged_lab.high_level.high_level_fault_config import HighLevelFaultCfg, LP_KINDS, VP_MODES
from legged_lab.high_level.high_level_faults import HighLevelSensorFaultPipeline, SensorFrame
from legged_lab.high_level.high_level_lp_faults import HighLevelLpFaults
from legged_lab.high_level.high_level_schedule import HighLevelFaultSchedule
from legged_lab.selector.randomness import PerEnvironmentRNG


def _film_cfg(**overrides):
    values = asdict(HighLevelFaultCfg().vp)
    values.update(failure_modes=values.pop("modes"), failure_mode_probabilities=None,
                  failure_probability=0., failure_duration_range=(2, 4),
                  consecutive_dropout_duration_range=(2, 4), freeze_duration_range=(2, 4),
                  distant_dust_duration_range=(2, 4))
    values.update(overrides)
    return SimpleNamespace(**values)


def _packets(pipeline, tick, *, ids=None, lp_value=0., vp_value=2.):
    ids = pipeline._ids(ids)
    n = len(ids)
    time = torch.full((n,), tick * .1, dtype=torch.float64)
    sequence = torch.full((n,), tick, dtype=torch.long)
    lp = SensorFrame(torch.full((n, 11, 17), float(lp_value)), torch.ones(n, 11, 17), time, time, sequence)
    vp = SensorFrame(torch.full((n, 18, 32), float(vp_value)), torch.ones(n, 18, 32), time, time, sequence)
    pose = torch.tensor([[0., 0., .8, 0.]]).repeat(n, 1)
    return lp, vp, pose


def _step(pipeline, tick, **kwargs):
    ids = kwargs.get("ids")
    return pipeline.update(*_packets(pipeline, tick, **kwargs), env_ids=ids)


def _force_plan(pipeline, *, lp_kind=0, vp_mode=-1, lp_duration=5, vp_duration=5, tick=1):
    s = pipeline.schedule
    s.next_plan.fill_(100000)
    s.lp_start.fill_(tick)
    s.vp_start.fill_(tick)
    s.lp_kind.fill_(int(lp_kind))
    s.vp_mode.fill_(vp_mode)
    s.lp_duration.fill_(lp_duration if lp_kind else 0)
    s.vp_duration.fill_(vp_duration if vp_mode >= 0 else 0)


@pytest.mark.parametrize("mode", range(8))
def test_all_film_modes_external_support_and_exact_last_frame(mode):
    model = StructuredDepthFailureModel(_film_cfg(), 2, "cpu")
    ids = torch.tensor([0])
    rng = PerEnvironmentRNG(2, 42)
    depth, support = torch.full((1, 18, 32, 1), 2.), torch.ones(1, 18, 32, 1)
    model.start_external(depth, support, ids, [mode], [1], rng=rng)
    output, valid = model.apply_external(depth + .1, support, ids, rng=rng)
    assert model.remaining_frames[0] == 0
    if mode == 6:
        torch.testing.assert_close(output, depth)
        torch.testing.assert_close(valid, support)
    else:
        missing = output == 0
        assert missing.any()
        assert (valid[missing] == 0).all()
        assert (valid[~missing] == 1).all()
    healthy, valid = model.apply_external(depth + .2, support, ids, rng=rng)
    torch.testing.assert_close(healthy, depth + .2)
    torch.testing.assert_close(valid, support)
    assert model.active_modes[1] == -1


def test_film_automatic_entry_remains_tensor_probability_driven_and_multiframe():
    random.seed(7)
    torch.manual_seed(7)
    cfg = _film_cfg(failure_probability=1., failure_modes=("full_dropout",), failure_duration_range=(2, 2))
    model = StructuredDepthFailureModel(cfg, 1)
    image = torch.ones(1, 18, 32, 1)
    assert isinstance(model(image, cfg, [0]), torch.Tensor)
    assert model.remaining_frames[0] == 1
    assert not model(image, cfg, [0]).any() and model.remaining_frames[0] == 0
    model.cfg.failure_probability = 0
    torch.testing.assert_close(model(image, cfg, [0]), image)
    with pytest.raises(RuntimeError, match="multi-frame"):
        StructuredDepthFailureModel(_film_cfg(failure_duration_range=(1, 3)), 1)


def test_film_external_never_uses_internal_probability_and_preserves_freeze_support():
    cfg = _film_cfg(failure_probability=1.)
    model = StructuredDepthFailureModel(cfg, 1)
    depth = torch.ones(1, 18, 32, 1)
    support = torch.zeros_like(depth)
    model.start_external(depth, support, [0], [6], [3])
    for _ in range(3):
        out, valid = model.apply_external(depth * 2, support + 1, [0])
        torch.testing.assert_close(out, depth)
        assert not valid.any()
    out, valid = model.apply_external(depth * 2, support + 1, [0])
    torch.testing.assert_close(out, depth * 2)
    assert valid.all()


def test_schedule_probabilities_duration_ranges_overlap_and_independent_normal_times():
    cfg = HighLevelFaultCfg()
    cfg.validate()
    n = 16000
    s = HighLevelFaultSchedule(n, cfg.schedule, seed=19)
    ids = torch.arange(n)
    s.frame.zero_()
    s.sample(ids)
    frequencies = s.scenario_counts.sum(0).float() / n
    torch.testing.assert_close(frequencies, torch.tensor([.30, .25, .25, .20]), atol=.015, rtol=0)
    dual = s.scenario == 3
    overlap = torch.minimum(s.lp_start + s.lp_duration, s.vp_start + s.vp_duration) - torch.maximum(
        s.lp_start, s.vp_start)
    assert (overlap[dual] >= 2).all()
    gap = (s.lp_start - s.vp_start).abs()
    assert (gap[dual] <= 5).all()
    simultaneous = (gap[dual] == 0).float().mean()
    assert .46 < simultaneous < .54
    assert ((s.next_plan - s.end >= 5) & (s.next_plan - s.end <= 30)).all()
    normal = s.scenario == 0
    assert ((s.end[normal] >= 10) & (s.end[normal] <= 50)).all()
    full = s.vp_mode == 4
    assert s.vp_duration[full].min() == 1
    assert ((s.vp_duration[full & dual] >= 2) & (s.vp_duration[full & dual] <= 3)).all()
    assert ((s.vp_duration[s.vp_mode == 5] >= 2) & (s.vp_duration[s.vp_mode == 5] <= 5)).all()
    for counts in (s.lp_counts.sum(0), s.vp_counts.sum(0)):
        expected = counts.sum().float() / len(counts)
        assert ((counts.float() - expected).abs() / expected < .15).all()


@pytest.mark.parametrize("scenario", range(4))
def test_schedule_uses_named_lp_vp_order_instead_of_legacy_reversed_ids(scenario):
    cfg = HighLevelFaultCfg()
    cfg.schedule.scenario_probabilities = tuple(float(i == scenario) for i in range(4))
    s = HighLevelFaultSchedule(32, cfg.schedule)
    s.sample(torch.arange(32))
    assert bool((s.lp_kind != 0).all()) == (scenario in (1, 3))
    assert bool((s.vp_mode >= 0).all()) == (scenario in (2, 3))


@pytest.mark.parametrize("kind", LP_KINDS)
def test_all_lp_kinds_accept_explicit_start_and_obey_quality_contract(kind):
    adapter = HighLevelLpFaults(1, HighLevelFaultCfg().lp)
    ids = torch.tensor([0])
    pose = torch.tensor([[0., 0., .8, 0.]])
    height, support = torch.zeros(1, 11, 17), torch.ones(1, 11, 17)
    adapter.begin(ids, torch.tensor([kind]), torch.tensor([15]), pose)
    for frame in range(17):
        result, valid, stamp = adapter.apply(height, support, torch.tensor([frame * .1]).double(), ids, pose)
        assert torch.isfinite(result).all() and torch.isfinite(valid).all()
        assert ((valid >= 0) & (valid <= 1)).all()
        assert (result[valid == 0] == -1.1).all()
        assert ((result[valid > 0] >= -1) & (result[valid > 0] <= 1)).all()
        if kind == int(K.LP_HEIGHT_BIAS):
            assert valid.all()
        if kind == int(K.LP_COLLAPSE) and frame < 15:
            assert not valid.any()
    assert adapter.remaining.item() == 0


def test_lp_freeze_and_delay_keep_matching_values_support_and_measurement_times():
    a = HighLevelLpFaults(2, HighLevelFaultCfg().lp)
    ids = torch.tensor([0, 1])
    pose = torch.tensor([[0., 0., .8, 0.]]).repeat(2, 1)
    h, valid = torch.zeros(2, 11, 17), torch.ones(2, 11, 17)
    for frame in range(6):
        a.apply(h + frame / 10, valid * (frame / 10 + .1), torch.full((2,), frame / 10).double(), ids, pose)
    a.begin(ids, torch.tensor([int(K.FREEZE), int(K.DELAY)]), torch.tensor([3, 3]), pose)
    a.delay[1] = 3
    outputs = []
    for frame in range(6, 9):
        outputs.append(a.apply(h + frame / 10, valid * .9, torch.full((2,), frame / 10).double(), ids, pose))
    for out, support, stamp in outputs:
        torch.testing.assert_close(out[0], h[0] + .6)
        torch.testing.assert_close(support[0], valid[0] * .9)
        assert stamp[0] == pytest.approx(.6)
    assert outputs[0][2][1] == pytest.approx(.3)
    torch.testing.assert_close(outputs[0][0][1], h[1] + .3)
    torch.testing.assert_close(outputs[0][1][1], valid[1] * .4)


def test_lp_delay_replays_the_previous_frames_corruption_without_reapplying_it():
    a = HighLevelLpFaults(1, HighLevelFaultCfg().lp)
    ids, pose = torch.tensor([0]), torch.tensor([[0., 0., .8, 0.]])
    height, support = torch.zeros(1, 11, 17), torch.ones(1, 11, 17)
    a.begin(ids, torch.tensor([int(K.LP_COLLAPSE)]), torch.tensor([1]), pose)
    a.apply(height, support, torch.tensor([0.]).double(), ids, pose)
    a.begin(ids, torch.tensor([int(K.DELAY)]), torch.tensor([1]), pose)
    a.delay[0] = 1
    out, valid, stamp = a.apply(height, support, torch.tensor([.1]).double(), ids, pose)
    assert (out == -1.1).all() and not valid.any() and stamp.item() == 0


def test_lp_parameter_ranges_are_uniform_and_independent_of_duration():
    n = 1600
    a = HighLevelLpFaults(n, HighLevelFaultCfg().lp, seed=89)
    ids = torch.arange(n)
    pose = torch.tensor([[0., 0., .8, 0.]]).repeat(n, 1)
    durations = PerEnvironmentRNG(n, 19).randint(10, 51, ids)
    a.begin(ids, torch.full((n,), int(K.LP_HEIGHT_BIAS)), durations, pose)
    magnitude = a.bias.abs()
    # The old 60% severe distribution fails this center/edge balance.
    assert .04 < magnitude.mean() < .11
    assert .28 < (magnitude < .05).float().mean() < .39
    correlation = torch.corrcoef(torch.stack((magnitude, durations.float())))[0, 1]
    assert abs(correlation) < .08


def test_persistent_map_recenter_retains_world_cell_height_and_freeze_validity():
    a = HighLevelLpFaults(1, HighLevelFaultCfg().lp)
    ids, pose = torch.tensor([0]), torch.tensor([[0., 0., .8, 0.]])
    s = a.map.state
    s["origin"][0] = pose[0, :3]
    s["ghost_valid"][0, 544] = True
    s["ghost_height"][0, 544] = .6
    a.begin(ids, torch.tensor([int(K.FREEZE)]), torch.tensor([3]), pose)
    for frame in range(3):
        out, valid, _ = a.apply(torch.zeros(1, 11, 17), torch.ones(1, 11, 17),
                                torch.tensor([frame * .1]).double(), ids, pose)
        assert valid.all() and s["ghost_valid"].any()
    pose[0, 0] = .1
    a.map.recenter_external(ids, pose)
    assert s["ghost_valid"][0, 543] and s["ghost_height"][0, 543] == pytest.approx(.6)
    assert not s["ghost_valid"][0, 544]


def test_pipeline_one_frame_failure_two_frame_history_and_read_only_cache():
    p = HighLevelSensorFaultPipeline(1)
    first = _step(p, 0)
    _force_plan(p, vp_mode=4, vp_duration=1)
    failed = _step(p, 1)
    assert not failed.vp_valid_support[:, -1].any()
    torch.testing.assert_close(failed.vp[:, -1], torch.full((1, 18, 32), .04))
    torch.testing.assert_close(failed.vp[:, 0], first.vp[:, -1])
    counters = p.vp_rng.get_state()
    frames = p.schedule.frame.clone()
    for _ in range(10):
        p.read().vp.zero_()  # Mutating a read must not corrupt the cache.
        duplicate = _step(p, 1)
        torch.testing.assert_close(duplicate.vp, failed.vp)
    torch.testing.assert_close(p.vp_rng.get_state(), counters)
    torch.testing.assert_close(p.schedule.frame, frames)
    recovered = _step(p, 2)
    assert not recovered.vp_valid_support[:, 0].any() and recovered.vp_valid_support[:, 1].all()
    assert _step(p, 3).vp_valid_support.all()


def test_pipeline_freeze_has_new_observable_times_but_hidden_content_age():
    p = HighLevelSensorFaultPipeline(1)
    _step(p, 0)
    _force_plan(p, vp_mode=6, vp_duration=3)
    _step(p, 1, vp_value=1.)
    result = _step(p, 2, vp_value=2.)
    torch.testing.assert_close(result.vp[:, -1], torch.full((1, 18, 32), .4))
    assert result.vp_source_stamp[0, -1] == pytest.approx(.2)
    assert p.audit()["vp_content_stamp"][0, -1] == pytest.approx(.1)
    assert "vp_content_stamp" not in vars(result)
    assert result.ages(.2)[1][0, -1] == 0


def test_drift_recovers_continuously_and_new_plan_does_not_clear_world_ghosts():
    a = HighLevelLpFaults(1, HighLevelFaultCfg().lp)
    ids = torch.tensor([0])
    pose = torch.tensor([[0., 0., .8, 0.]])
    a.map_drift[0] = torch.tensor([.1, .2, 2., .03])
    a.linear[0] = torch.tensor([.1, .1, 1., 0.])
    initial = a.map_drift.clone()
    a.apply(torch.zeros(1, 11, 17), torch.ones(1, 11, 17), torch.zeros(1).double(), ids, pose)
    torch.testing.assert_close(a.map_drift, initial * math.exp(-.1 / .5))
    a.map.state["ghost_valid"][0, 544] = True
    a.map.state["ghost_height"][0, 544] = .6
    a.map.state["origin"][0] = pose[0, :3]
    a.begin(ids, torch.tensor([int(K.LP_HEIGHT_BIAS)]), torch.tensor([10]), pose)
    assert a.map.state["ghost_valid"][0, 544]
    # Missing ground cannot clear accepted returns, despite elapsed time.
    for frame in range(4):
        a.apply(torch.full((1, 11, 17), -1.1), torch.zeros(1, 11, 17),
                torch.tensor([frame * .1]).double(), ids, pose)
    assert a.map.state["ghost_valid"][0, 544]
    # Fresh valid ground confirms replacement after three scans.
    a.kind.zero_()
    for frame in range(3):
        a.apply(torch.zeros(1, 11, 17), torch.ones(1, 11, 17),
                torch.tensor([frame * .1 + 1.]).double(), ids, pose)
    assert not a.map.state["ghost_valid"].any()
    for frame in range(55):
        a.apply(torch.zeros(1, 11, 17), torch.ones(1, 11, 17),
                torch.tensor([frame * .1 + 2.]).double(), ids, pose)
    assert not a.residual().any()


def test_resetting_or_updating_other_rows_preserves_random_streams_and_outputs():
    a, b = HighLevelSensorFaultPipeline(2, seed=91), HighLevelSensorFaultPipeline(2, seed=91)
    for tick in range(90):
        if tick == 10:
            b.reset([1])
        out_a = _step(a, tick)
        out_b = _step(b, tick, ids=[0])
        torch.testing.assert_close(out_a.lp[0], out_b.lp[0])
        torch.testing.assert_close(out_a.vp[0], out_b.vp[0])
        torch.testing.assert_close(out_a.vp_valid_support[0], out_b.vp_valid_support[0])
    torch.testing.assert_close(a.schedule.rng.get_state()[0], b.schedule.rng.get_state()[0])
    torch.testing.assert_close(a.vp_rng.get_state()[0], b.vp_rng.get_state()[0])


def test_disabled_faults_and_replay_metadata_are_serializable():
    cfg = HighLevelFaultCfg(enable=False)
    p = HighLevelSensorFaultPipeline(2, cfg=cfg, seed=123)
    out = _step(p, 0)
    assert out.initialized.all() and out.vp_valid_support.all()
    assert not p.schedule.plan_count.any()
    assert p.audit()["seed"] == 123
    assert HighLevelFaultCfg.from_config(json.loads(json.dumps(p.audit()["config"]))).to_dict()
    lp, vp, pose = _packets(p, 1)
    vp.delivery_stamp = vp.delivery_stamp + .1
    with pytest.raises(ValueError, match="common sensor-clock"):
        p.update(lp, vp, pose)


@pytest.mark.parametrize("edit", [
    lambda c: setattr(c.schedule, "scenario_probabilities", (.3, .3, .3, .3)),
    lambda c: setattr(c.schedule, "missing_duration_s", (.45, 2.)),
    lambda c: setattr(c.schedule, "dual_full_dropout_duration_s", (.1, .3)),
    lambda c: setattr(c.lp, "kinds", LP_KINDS[:-1]),
    lambda c: setattr(c.vp, "modes", VP_MODES[:-1]),
])
def test_invalid_fault_contract_is_rejected(edit):
    cfg = HighLevelFaultCfg()
    edit(cfg)
    with pytest.raises(ValueError):
        cfg.validate()
