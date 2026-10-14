"""Directed source admission, capture-once depth and snapshot regressions.

These run on CPU without Isaac Sim; they do not establish physical rescue rates.
"""

from dataclasses import replace
import argparse
import ast
from pathlib import Path

import pytest
import torch

from legged_lab.selector.config import CollectionConfig, FaultConfig
from legged_lab.selector.directed_audit import directed_audit_contract, SOURCE_AUDIT_ORDER, VP_AUDIT_ORDER
from legged_lab.selector.faults import SelectorFaultManager, FaultKind as K
from legged_lab.selector.fault_kinds import STRUCTURED_VP_FAULTS
from legged_lab.selector.supplement import fault_profile, SupplementConfig, OutcomeQuotaSampler, resolve_fault_source
from legged_lab.selector.dataset import SelectorBatch, SelectorDatasetWriter
from legged_lab.selector.analysis import analyze_selector_dataset
from legged_lab.selector.collection_audit import collection_audit_contract, CollectionAudit
from legged_lab.selector.labels import FUTURE_METRIC_ORDER, SafetyOutcomeAccumulator, SafetyStepMetrics


@pytest.mark.parametrize("name,scenario", [
    ("directed_lp", (0., 0., 1., 0.)), ("directed_vp", (0., 1., 0., 0.)),
    ("directed_dual", (0., 0., 0., 1.)),
])
def test_directed_profiles_have_exclusive_sources(name, scenario):
    cfg = fault_profile(name)
    assert cfg.scenario_probabilities == scenario
    assert cfg.structured_vp_faults and cfg.independent_dual_faults
    manager = SelectorFaultManager(128, cfg=cfg)
    assert bool((manager.lp_kind != 0).all()) == (scenario[2]+scenario[3] == 1)
    assert bool((manager.vp_kind != 0).all()) == (scenario[1]+scenario[3] == 1)
    assert set(manager.vp_kind.tolist()) <= {0, *map(int, STRUCTURED_VP_FAULTS)}
    assert manager.vp_buffer.numel() == 0  # no second copy of the VP capture ring


def test_kind_proposals_count_parent_events_once_outside_forks():
    manager = _manager(3, kind=K.VP_SPATIAL_BIAS)
    audit = CollectionAudit(3, "cpu")
    episodes = torch.zeros(3, dtype=torch.long)
    audit.observe_behavior_plans(episodes, manager)
    audit.observe_behavior_plans(episodes, manager)
    proposals = audit.summary()["observed_behavior_plans_by_kind_pair"]
    assert sum(proposals[f"0:{int(K.VP_SPATIAL_BIAS)}"]) == 3


def _admission(mode, labels, kinds, *, source=None, quotas=None):
    n = len(labels)
    cfg = SupplementConfig(fault_source=mode, pair_quotas=quotas)
    sampler = OutcomeQuotaSampler(cfg, sum(quotas) if quotas else n, seed=31)
    faults = torch.zeros(n, 8, dtype=torch.long)
    faults[:, :2] = torch.tensor(kinds)
    params = torch.zeros(n, 21)
    if source is None:
        source = torch.zeros(n, len(SOURCE_AUDIT_ORDER))
        source[:, :4] = 1
    events = torch.stack((torch.arange(n), torch.zeros(n), torch.ones(n)), -1).long()
    selected = sampler.select(torch.ones(n, dtype=torch.bool), torch.tensor(labels), events, 1.,
                              faults=faults, parameters=params, source=source)
    return selected, sampler


@pytest.mark.parametrize("mode,desired,opposite", [("lp_only", [1, 0], [0, 1]), ("vp_only", [0, 1], [1, 0])])
def test_single_banks_reject_reverse_and_dual_outcomes(mode, desired, opposite):
    correct = [int(K.LP_HEIGHT_BIAS), 0] if mode == "lp_only" else [0, int(K.VP_BIAS)]
    wrong = [0, int(K.VP_BIAS)] if mode == "lp_only" else [int(K.LP_HEIGHT_BIAS), 0]
    selected, sampler = _admission(mode, [desired, desired, desired, opposite, [1, 1]],
        [correct, wrong, [int(K.LP_HEIGHT_BIAS), int(K.VP_BIAS)], correct, correct])
    assert selected.tolist() == [0]
    assert sampler.rejections["wrong_source"] == 2
    assert sampler.rejections["source_outcome"] == 2


def test_dual_bank_needs_both_visible_effects_and_both_actual_failures():
    source = torch.ones(5, len(SOURCE_AUDIT_ORDER))
    source[1, 1] = 0
    kinds = [[int(K.LP_HEIGHT_BIAS), int(K.VP_SPATIAL_BIAS)]]*5
    quotas = (0, 0, 0, 1)
    ids, sampler = _admission("dual", [[1, 1], [1, 1], [1, 0], [0, 1], [0, 0]], kinds, source=source, quotas=quotas)
    assert ids.tolist() == [0]
    assert sampler.rejections["effect_not_observable"] == 1
    assert sampler.rejections["source_outcome"] == 2
    assert sampler.source_eligible_pairs[0] == 1
    assert sampler.counts[0] == 0
    detail = sampler.summary()["outcome_yield_by_kind_pair"][f"{int(K.LP_HEIGHT_BIAS)}:{int(K.VP_SPATIAL_BIAS)}"]
    assert detail["stable_attempt_1s"] == [1, 1, 1, 2]
    assert detail["source_eligible_1s"] == [1, 1, 1, 1]
    assert detail["written_1s"] == [0, 0, 0, 1]


def test_unavailable_healthy_side_is_rejected():
    source = torch.ones(2, len(SOURCE_AUDIT_ORDER))
    source[1, 3] = 0
    ids, sampler = _admission("lp_only", [[1, 0]]*2, [[int(K.LP_HEIGHT_BIAS), 0]]*2, source=source)
    assert ids.tolist() == [0]
    assert sampler.rejections["healthy_side_not_clean"] == 1


def test_both_safe_controls_keep_the_same_source_contract():
    ids, sampler = _admission("dual", [[0, 0], [0, 0]],
                             [[int(K.LP_HEIGHT_BIAS), int(K.VP_BIAS)], [0, int(K.VP_BIAS)]], quotas=(1, 0, 0, 0))
    assert ids.tolist() == [0]
    assert sampler.counts == [1, 0, 0, 0]


@pytest.mark.parametrize("cfg", [
    SupplementConfig(fault_source="lp_only", pair_quotas=(0, 0, 1, 0)),
    SupplementConfig(fault_source="dual", pair_quotas=(0, 1, 0, 0)),
    SupplementConfig(fault_source="vp_only", split="test"),
    SupplementConfig(fault_source="dual", require_initial_safe=False),
])
def test_invalid_source_configuration_is_not_silently_used(cfg):
    with pytest.raises(ValueError):
        cfg.validate(1)


def test_event_cap_and_real_time_spacing():
    sampler = OutcomeQuotaSampler(SupplementConfig(max_samples_per_event=2), 3, 42)
    labels, good, event = torch.tensor([[1, 0]]), torch.tensor([True]), torch.tensor([[2, 9, 1]])
    assert sampler.select(good, labels, event, 0.).tolist() == [0]
    assert sampler.select(good, labels, event, .1).tolist() == []
    assert sampler.select(good, labels, event, .2).tolist() == [0]
    assert sampler.select(good, labels, event, .4).tolist() == []
    assert sampler.rejections["event_spacing"] == 1 and sampler.rejections["event_cap"] == 1


def _manager(n=1, kind=K.NONE, profile="directed_vp", **overrides):
    cfg = replace(fault_profile(profile), max_start_delay_frames=0, **overrides)
    manager = SelectorFaultManager(n, cfg=cfg, seed=913)
    manager.lp_kind.zero_()
    manager.vp_kind.fill_(int(kind))
    manager.vp_start_frame.zero_()
    manager.vp_duration_frames.fill_(100)
    manager.next_schedule_frame.fill_(200)
    return manager


def _apply(manager, step, current=.4, *, previous=None, env_ids=None, same_source=None):
    n = manager.num_envs
    lp = torch.zeros(n, 11, 17)
    frame = torch.as_tensor(current).float().expand(n, 18, 32)
    previous = frame if previous is None else torch.as_tensor(previous).float().expand_as(frame)
    vp = torch.stack((previous, frame), 1)
    time = torch.full((n,), step*.1 if same_source is None else same_source, dtype=torch.float64)
    return manager.apply(lp, vp, time, time, lp_valid=torch.ones_like(lp), vp_valid=torch.ones_like(vp),
                         env_ids=env_ids, world_pose=torch.full((n, 4), step*.01), now=step*.1)


def test_old_depth_frame_keeps_its_capture_time_corruption():
    manager = _manager()
    first = _apply(manager, 0, .2)
    manager.vp_kind.fill_(int(K.VP_BIAS))
    manager.vp_bias.fill_(.1)
    second = _apply(manager, 1, .4, previous=.2)
    torch.testing.assert_close(second.vp[:, 0], first.vp[:, -1])
    torch.testing.assert_close(second.vp[:, -1], torch.full((1, 18, 32), .5))
    rng = manager.generator.get_state()
    repeated = _apply(manager, 2, .8, same_source=.1)
    torch.testing.assert_close(repeated.vp, second.vp)
    torch.testing.assert_close(rng, manager.generator.get_state())
    third = _apply(manager, 3, .6, previous=.4)
    torch.testing.assert_close(third.vp[:, 0], second.vp[:, -1])
    torch.testing.assert_close(third.vp[:, -1], torch.full((1, 18, 32), .7))


def test_freeze_survives_ring_wrap_and_does_not_append_history():
    manager = _manager(kind=K.FREEZE)
    first = _apply(manager, 0, .2)
    for step in range(1, 15):
        result = _apply(manager, step, .2+.02*step)
        torch.testing.assert_close(result.vp, first.vp)
        torch.testing.assert_close(result.vp_timestamp, first.vp_timestamp)
        assert not manager.vp_faults.state["history_append"].any()
    assert manager.audit_source()[0, 1] == 1


def test_delay_selects_stored_source_without_rewriting_the_previous_frame():
    manager = _manager(kind=K.DELAY)
    manager.vp_delay.fill_(2)
    results = [_apply(manager, step, .1+.1*step) for step in range(5)]
    torch.testing.assert_close(results[4].vp[:, -1], torch.full((1, 18, 32), .3))
    torch.testing.assert_close(results[4].vp[:, 0], results[3].vp[:, -1])
    assert results[4].vp_timestamp.item() == pytest.approx(.2)
    assert results[4].vp_packet_timestamp.item() == pytest.approx(.4)
    assert manager.vp_faults.audit(manager)[0, VP_AUDIT_ORDER.index("history_interval_s")] == pytest.approx(.1)


def test_missing_source_timestamp_does_not_create_a_captured_fault_frame():
    manager = _manager(kind=K.VP_SPATIAL_BIAS)
    rng = manager.generator.get_state()
    result = _apply(manager, 0, same_source=-float("inf"))
    assert not manager.vp_faults.state["initialized"].any()
    assert not manager.vp_observed_effect.any()
    assert manager.unobservable_onset_within(10).all()
    assert not torch.isfinite(result.vp_timestamp).any()
    torch.testing.assert_close(manager.generator.get_state(), rng)
    assert torch.isfinite(manager.vp_faults.audit(manager)).all()


def test_filter_trail_uses_fresh_timestamps_and_decays_after_fault():
    manager = _manager(kind=K.VP_FILTER_TRAIL)
    manager.vp_faults.state["alpha"].fill_(.8)
    _apply(manager, 0, .2)
    lagged = _apply(manager, 1, .8)
    torch.testing.assert_close(lagged.vp[:, -1], torch.full((1, 18, 32), .32))
    assert lagged.vp_timestamp.item() == pytest.approx(.1)
    assert lagged.vp_packet_timestamp.item() == pytest.approx(.1)
    manager.vp_duration_frames.fill_(2)
    recovery = _apply(manager, 2, .8)
    assert .32 < recovery.vp[0, -1].mean() < .8
    for step in range(3, 40):
        recovery = _apply(manager, step, .8)
    assert not manager.vp_faults.residual().any()
    torch.testing.assert_close(recovery.vp[:, -1], torch.full((1, 18, 32), .8))


def test_geometric_error_is_not_faked_as_invalid_or_stale_data():
    manager = _manager(kind=K.VP_SPATIAL_BIAS)
    result = _apply(manager, 0)
    assert not torch.allclose(result.vp[:, -1], torch.full((1, 18, 32), .4))
    assert (result.vp_valid == 1).all()
    assert result.vp_timestamp.item() == 0 and result.vp_packet_timestamp.item() == 0


def test_edge_fault_on_a_flat_image_has_no_effect_and_is_not_labelled_as_visible():
    manager = _manager(kind=K.VP_EDGE_MISMATCH)
    _apply(manager, 0)
    assert not manager.vp_observed_effect.any()
    assert manager.unobservable_onset_within(10).all()
    assert manager.audit_source()[0, 1] == 0


def test_local_missing_depth_has_raw_invalid_support():
    manager = _manager(kind=K.VP_LOCAL_DROPOUT)
    for step in range(5):
        result = _apply(manager, step)
    assert (result.vp_valid[:, -1] == 0).any()
    assert (result.vp_valid[:, -1] == 1).any()
    assert torch.isfinite(result.vp).all()


@pytest.mark.parametrize("kind", STRUCTURED_VP_FAULTS)
def test_each_vp_family_replays_identically_from_snapshot(kind):
    manager = _manager(kind=kind)
    plane = torch.linspace(.1, .9, 32)[None].expand(18, -1).clone()
    plane[:, 16:] += .08
    for step in range(8):
        _apply(manager, step, plane)
    snapshot = manager.snapshot()
    expected = [_apply(manager, step, plane+.001*step) for step in range(8, 15)]
    expected_rng = manager.generator.get_state()
    manager.restore(snapshot)
    actual = [_apply(manager, step, plane+.001*step) for step in range(8, 15)]
    for a, b in zip(actual, expected):
        for field in ("vp", "vp_valid", "vp_timestamp", "vp_packet_timestamp", "lp", "lp_valid"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    torch.testing.assert_close(manager.generator.get_state(), expected_rng)


def test_resetting_another_row_does_not_advance_this_rows_noise():
    a, b = _manager(2, kind=K.VP_SPATIAL_BIAS), _manager(2, kind=K.VP_SPATIAL_BIAS)
    for step in range(3):
        _apply(a, step)
        _apply(b, step)
    b.reset(torch.tensor([1]))
    for step in range(3, 8):
        expected, actual = _apply(a, step), _apply(b, step)
        torch.testing.assert_close(actual.vp[0], expected.vp[0], rtol=0, atol=0)
        assert b.generator.get_state()[0] == a.generator.get_state()[0]


def test_dual_sides_have_independent_severity_and_all_three_start_orders():
    manager = SelectorFaultManager(2048, cfg=fault_profile("directed_dual"))
    assert (manager.lp_severity != manager.vp_severity).any()
    assert (manager.lp_delay != manager.vp_delay).any()
    delta = manager.lp_start_frame-manager.vp_start_frame
    assert (delta == 0).any() and (delta < 0).any() and (delta > 0).any()


def test_pending_second_fault_is_not_used_to_label_the_current_dual_state():
    manager = _manager(profile="directed_dual", kind=K.VP_SPATIAL_BIAS)
    manager.lp_kind.fill_(int(K.LP_HEIGHT_BIAS))
    manager.lp_bias.fill_(.1)
    manager.lp_start_frame.zero_()
    manager.lp_duration_frames.fill_(100)
    manager.vp_start_frame.fill_(4)
    _apply(manager, 0)
    assert manager.audit_source()[0, 0] == 1
    assert manager.audit_source()[0, 1] == 0
    assert manager.unobservable_onset_within(10).all()
    for step in range(1, 5):
        _apply(manager, step)
    assert manager.audit_source()[0, :2].bool().all()
    assert not manager.unobservable_onset_within(10).any()


def test_old_parent_depth_cannot_prove_a_new_parents_unseen_effect():
    manager = _manager(kind=K.VP_BIAS)
    manager.vp_bias.fill_(.1)
    _apply(manager, 0)
    assert manager.vp_observed_effect.all()
    manager._schedule(torch.tensor([0]))
    manager.vp_kind.fill_(int(K.VP_EDGE_MISMATCH))
    _apply(manager, 1, same_source=0.)
    assert not manager.vp_observed_effect.any()
    assert manager.audit_source()[0, 1] == 0
    assert manager.unobservable_onset_within(10).all()


@pytest.mark.parametrize("profile,source", [("directed_lp", "lp_only"), ("directed_vp", "vp_only"), ("directed_dual", "dual")])
def test_directed_cli_source_cannot_be_bypassed(profile, source):
    assert resolve_fault_source(profile) == source
    assert resolve_fault_source(profile, source) == source
    with pytest.raises(ValueError):
        resolve_fault_source(profile, "any")


def test_collect_cli_accepts_new_profiles_and_preserves_pure_configuration():
    # Execute only argparse setup and the pure config function, never AppLauncher.
    path = Path(__file__).resolve().parents[2]/"legged_lab/scripts/collect_selector_data.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("parser", "quota_args", "phase_args") for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("parser", "quota_args", "phase_args"):
                nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "collection_fault_config":
            nodes.append(node)
    namespace = {"argparse": argparse, "Path": Path, "fault_profile": fault_profile, "replace": replace}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    args = namespace["parser"].parse_args(["--fault-profile", "directed_dual", "--fault-source", "dual", "--supplement-split", "train"])
    cfg = namespace["collection_fault_config"](args)
    assert cfg.scenario_probabilities == (0., 0., 0., 1.)
    assert cfg.structured_vp_faults


def test_legacy_short_delay_config_and_shape_still_work():
    cfg = replace(FaultConfig(), max_delay_frames=1)
    manager = SelectorFaultManager(2, cfg=cfg)
    assert manager.vp_faults is None
    result = _apply(manager, 0)
    assert result.vp.shape == (2, 2, 18, 32)


@pytest.mark.parametrize("timeout_step,valid", [(1, False), (20, False), (50, True)])
def test_directed_labels_still_use_the_full_one_second_timeout_contract(timeout_step, valid):
    accumulator = SafetyOutcomeAccumulator(1, device="cpu")
    frame = SafetyStepMetrics(torch.zeros(1), torch.full((1,), .8), torch.zeros(1),
                              torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    for step in range(1, 51):
        timeout = torch.tensor([step == timeout_step])
        accumulator.update(frame, timeout, timeout)
    outcome = accumulator.outcome()
    assert outcome.label_valid.item() is valid
    assert not outcome.fail_1s.item()
    assert outcome.first_timeout_step.item() == timeout_step


def _write_bank(root, *, missing_source=False, wrong_effect=False):
    n = 2
    metadata = {"collection_complete": True, "collection_audit_contract": collection_audit_contract(),
                "directed_audit_contract": directed_audit_contract(),
                "supplement_coverage": {"config": {"fault_source": "dual"}}}
    writer = SelectorDatasetWriter(root, collection_cfg=CollectionConfig(
        num_envs=n, target_samples=n, shard_size=1, allow_nonproduction_num_envs=True),
        metadata=metadata, fixed_split="train")
    faults = torch.zeros(n, 8, dtype=torch.long)
    faults[:, :2] = torch.tensor([int(K.LP_HEIGHT_BIAS), int(K.VP_SPATIAL_BIAS)])
    faults[:, 6:] = 1
    source = torch.ones(n, len(SOURCE_AUDIT_ORDER))
    if wrong_effect:
        source[1, 1] = 0
    future = torch.zeros(n, 2, len(FUTURE_METRIC_ORDER))
    future[:, :, 0] = torch.tensor([[5., 10.], [15., 30.]])
    params = torch.zeros(n, 21)
    params[:, :2] = 4.
    params[:, 4:6] = torch.tensor([[-.5, -.5], [-.8, -.5]])
    batch = SelectorBatch(torch.zeros(n, 5, 283), torch.ones(n, 2, dtype=torch.uint8),
        ["terrain_seed42:type1:level2"]*n, torch.zeros(n), torch.ones(n), torch.arange(n),
        future_metrics=future, fault_metadata=faults, fault_parameters=params,
        source_metadata=None if missing_source else source, vp_fault_metadata=torch.zeros(n, len(VP_AUDIT_ORDER)),
        event_identity=torch.tensor([[0, 1, 2], [1, 1, 2]]),
        candidate_safety=torch.tensor([[0., .8, 0., 1.]]*n))
    writer.append(batch)
    writer.close()
    return analyze_selector_dataset(root)


def test_persisted_bank_analysis_checks_source_and_joint_failure_coverage(tmp_path):
    report = _write_bank(tmp_path)
    assert report["directed_acceptance"]["verified"]
    assert report["overall"]["pair_counts"] == [[0, 0, 0, 2]]
    assert report["field_rows"]["source_metadata"] == 2
    assert report["by_group"]["dual_start_order"]["0"]["rows"] == 1
    assert report["by_group"]["dual_start_order"]["1"]["rows"] == 1
    assert report["by_group"]["both_failure_lead_time_bin"]["0"]["rows"] == 1
    assert report["by_group"]["both_failure_lead_time_bin"]["1"]["rows"] == 1
    payload = torch.load(next((tmp_path/"train").glob("shard_*.pt")), weights_only=False)
    assert payload["features"].shape == (1, 5, 283)
    assert payload["vp_fault_metadata"].shape == (1, len(VP_AUDIT_ORDER))


@pytest.mark.parametrize("missing_source,wrong_effect", [(True, False), (False, True)])
def test_audit_cannot_verify_missing_or_unobservable_source(tmp_path, missing_source, wrong_effect):
    report = _write_bank(tmp_path, missing_source=missing_source, wrong_effect=wrong_effect)
    assert not report["directed_acceptance"]["verified"]
