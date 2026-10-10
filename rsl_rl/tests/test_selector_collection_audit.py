"""Meaningful CPU regressions for audit collection, sharding and legacy analysis."""

from dataclasses import replace
import ast
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from legged_lab.selector.analysis import analyze_selector_dataset, write_collection_analysis
from legged_lab.selector.collection_audit import (
    CollectionAudit, FAULT_PARAMETER_ORDER, collection_audit_contract,
)
from legged_lab.selector.collector import CounterfactualCollector
from legged_lab.selector.config import CollectionConfig, FaultConfig
from legged_lab.selector.dataset import SelectorBatch, SelectorDatasetWriter, SelectorShardDataset, split_from_group
from legged_lab.selector.faults import SelectorFaultManager
from legged_lab.selector.feature_contract import dataset_feature_contract, selector_feature_contract
from legged_lab.selector.labels import SafetyOutcomeAccumulator, SafetyStepMetrics, safety_label_contract
from legged_lab.selector.model import SelectorNetwork
from test_selector_collection_v2 import make_collector


def frame(tilt=.0, clearance=.8, contact=False):
    return SafetyStepMetrics(torch.tensor([tilt]), torch.tensor([clearance]),
                             torch.zeros(1), torch.tensor([contact]), torch.ones(1).bool())


def finish(accumulator):
    while accumulator.step_index < accumulator.long_horizon_steps:
        accumulator.update(frame(), torch.zeros(1).bool(), torch.zeros(1).bool())
    return accumulator.outcome()


def test_first_failure_records_simultaneous_triggers_once():
    acc = SafetyOutcomeAccumulator(1, device="cpu")
    acc.update(frame(tilt=.9), torch.zeros(1).bool(), torch.zeros(1).bool())
    acc.update(frame(clearance=.3, contact=True), torch.ones(1).bool(), torch.zeros(1).bool())
    result = finish(acc)
    assert result.first_failure_step.item() == 1
    assert result.first_failure_flags.item() == 4  # later contact/termination cannot replace first tilt
    timeout = SafetyOutcomeAccumulator(1, device="cpu", short_horizon_steps=1, long_horizon_steps=1)
    timeout.update(frame(tilt=.9, clearance=.3, contact=True), torch.ones(1).bool(), torch.ones(1).bool())
    result = timeout.outcome()
    assert result.label_valid.item() and result.first_failure_flags.item() == 14


def test_disagreeing_first_cause_does_not_change_original_replay_acceptance():
    first = SafetyOutcomeAccumulator(1, device="cpu")
    first.update(frame(tilt=.9), torch.zeros(1).bool(), torch.zeros(1).bool())
    second = SafetyOutcomeAccumulator(1, device="cpu")
    second.update(frame(clearance=.3), torch.zeros(1).bool(), torch.zeros(1).bool())
    collector = CounterfactualCollector.__new__(CounterfactualCollector)
    outcomes = iter((finish(first), finish(second)))
    collector._restore = lambda snapshot: None
    collector._rollout = lambda *args, **kwargs: (next(outcomes), None)
    outcome, stable = collector._repeated_branch_rollout(None, 1.0)
    assert stable.item() and not outcome.first_failure_cause_stable.item()
    assert outcome.first_failure_flags.item() == 8


def synthetic_batch():
    labels = torch.tensor([[0,0,0,0], [1,1,0,0], [0,1,0,0], [0,0,1,1],
                           [0,0,0,1], [1,1,1,1], [0,1,0,1], [0,0,0,0]])
    metrics = torch.zeros(8, 2, 10)
    metrics[:, :, 0] = torch.tensor([[-1,-1], [5,-1], [30,-1], [-1,3],
                                    [-1,45], [10,10], [25,25], [-1,-1]])
    flags = torch.tensor([[0,0], [4,0], [8,0], [0,2], [0,1], [5,10], [8,8], [0,0]], dtype=torch.uint8)
    stable = torch.ones(8, 2).bool()
    stable[5, 1] = False
    faults = torch.tensor([[10,0,2,-1,0,10,0,0], [10,0,2,0,0,10,1,0],
                           [10,0,1,0,0,10,1,0], [0,4,2,0,0,10,0,1],
                           [0,4,2,0,0,10,0,1], [10,4,2,0,0,10,1,1],
                           [10,4,2,1,0,10,0,0], [0,0,0,0,0,10,0,0]])
    identity = torch.tensor([[0,0,1], [0,0,1], [0,0,2], [1,0,1],
                             [1,1,1], [2,0,1], [2,0,1], [3,0,1]])
    params = torch.ones(8, len(FAULT_PARAMETER_ORDER))
    params[:, FAULT_PARAMETER_ORDER.index("lp_height_bias_m")] = torch.arange(8) * .1
    initial = torch.tensor([[0.,.8,0.,1.]]).repeat(8, 1)
    initial[1, 0] = .9
    return SelectorBatch(
        torch.arange(8).float()[:, None, None].expand(-1, 5, 283),
        labels, ["terrain"] * 8, torch.arange(8) % 3, torch.arange(8) % 2,
        torch.arange(8), metrics, faults, execution_metadata=torch.zeros(8, 3),
        event_identity=identity, fault_parameters=params,
        first_failure_flags=flags, first_failure_cause_stable=stable, candidate_safety=initial,
    )


def writer(root, fixed_split="test"):
    return SelectorDatasetWriter(
        root, fixed_split=fixed_split,
        collection_cfg=CollectionConfig(shard_size=3),
        metadata={"collection_complete": True, "safety_label_contract": safety_label_contract(),
                  "selector_feature_contract": selector_feature_contract(),
                  "collection_audit_contract": collection_audit_contract()},
    )


def test_sharding_analysis_and_training_keep_audit_rows_aligned(tmp_path):
    batch = synthetic_batch()
    out = writer(tmp_path)
    # Two appends cross shard boundaries; row metadata must travel with features.
    for lo, hi in ((0, 2), (2, 8)):
        values = {name: (value[lo:hi] if value is not None else None) for name, value in vars(batch).items()}
        out.append(SelectorBatch(**values))
    out.close()
    data = SelectorShardDataset(tmp_path, "test")
    assert dataset_feature_contract(data.manifest) == selector_feature_contract()
    for i in range(8):
        assert int(data[i]["features"][0, 0]) == i
        assert torch.equal(data[i]["event_identity"], batch.event_identity[i])
        assert torch.equal(data[i]["first_failure_flags"], batch.first_failure_flags[i])
    report = analyze_selector_dataset(tmp_path)
    assert report["overall"]["positive"] == [2, 4, 2, 4]
    assert report["overall"]["pair_counts"] == [[5, 1, 1, 1], [2, 2, 2, 2]]
    assert report["events"]["episodes"]["unique"] == 5
    assert report["events"]["plans"]["unique"] == 6
    assert report["events"]["plans"]["with_counterfactual_failure_per_head"] == [2, 3, 2, 3]
    assert report["by_group"]["active_fault_severity"]["2"]["rows"] == 4
    assert report["first_failure_causes"]["ambiguous_failure_rows"] == [0, 1]
    assert report["first_failure_causes"]["stable_trigger_counts"]["low_clearance"] == [2, 1]
    assert report["first_failure_time"]["step_counts_1_to_50"][0][29] == 1
    assert report["candidate_safety"]["already_violated_rows"] == 1
    params = report["fault_parameter_summary"]["lp:lp_height_bias:lp_height_bias_m"]
    assert params["rows"] == 3 and params["mean"] == pytest.approx((.1 + .2 + .5) / 3)
    # Same training interface and finite gradients with all additional audit fields.
    model = SelectorNetwork()
    x = torch.stack([data[i]["features"] for i in range(8)])
    y = torch.stack([data[i]["labels"] for i in range(8)])
    prediction = model(x)
    loss = torch.nn.functional.binary_cross_entropy(prediction, y)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_legacy_v3_and_empty_partial_datasets_report_missing_fields(tmp_path):
    out = writer(tmp_path / "old")
    out.metadata.pop("collection_audit_contract")
    batch = synthetic_batch()
    for field in ("event_identity", "fault_parameters", "first_failure_flags", "first_failure_cause_stable", "candidate_safety"):
        setattr(batch, field, None)
    out.append(batch)
    out.close()
    report = analyze_selector_dataset(tmp_path / "old")
    assert report["overall"]["positive"] == [2,4,2,4]
    assert report["events"]["plans"] is None
    assert report["first_failure_causes"]["available_rows"] == 0
    assert report["first_failure_time"]["available_rows"] == 8
    empty = writer(tmp_path / "empty")
    empty.metadata["collection_complete"] = False
    empty.close()
    path = write_collection_analysis(tmp_path / "empty")
    report = json.loads(path.read_text())
    assert report["overall"]["positive_rate"] == [None] * 4
    assert not report["collection_complete"]


def test_real_behavior_events_exclude_forks_and_funnel_preserves_pairs(tmp_path):
    collector = make_collector(tmp_path)
    collector.collect()
    collector.writer.close()
    report = analyze_selector_dataset(tmp_path)
    audit = report["collection_audit"]
    assert torch.tensor(audit["observed_behavior_plans_by_planned_scenario_severity"]).sum() == 8
    assert report["events"]["plans"]["unique"] == 8
    stages = torch.tensor(audit["funnel_by_active_scenario_severity"]).sum((1, 2))
    assert stages.tolist() == [24] * 7
    assert torch.tensor(audit["pairs_by_stage_horizon_active_scenario_severity"])[2, 1].sum() == 24
    assert report["first_failure_causes"]["stable_bitmask_counts_0_to_15"][0][0] == 24


def test_observed_plan_counter_survives_reset_schedule_number_reuse():
    faults = SelectorFaultManager(2)
    episode = torch.zeros(2).long()
    audit = CollectionAudit(2, "cpu")
    audit.observe_behavior_plans(episode, faults)
    audit.observe_behavior_plans(episode, faults)
    faults.reset(torch.tensor([1]))
    episode[1] += 1
    audit.observe_behavior_plans(episode, faults)
    assert audit.observed_plans.sum() == 3
    snapshot = faults.snapshot()
    values = faults.audit_parameters()
    assert values.shape == (2, len(FAULT_PARAMETER_ORDER)) and torch.isfinite(values).all()
    assert torch.equal(snapshot.tensors["schedule_count"], faults.schedule_count)
    torch.testing.assert_close(values[:, FAULT_PARAMETER_ORDER.index("vp_bias_m")],
                               faults.vp_bias * faults.cfg.vp_depth_max_m)
    assert FaultConfig().severity_probabilities == (.1, .3, .6)
    replace(FaultConfig(), severity_probabilities=(.4, .4, .2)).validate()
    with pytest.raises(ValueError):
        replace(FaultConfig(), severity_probabilities=(.1, .3, .8)).validate()


def test_split_selection_preserves_identity_and_cli_runs_without_isaac(tmp_path):
    batch = synthetic_batch()
    names = {}
    for i in range(100):
        group = f"terrain_{i}"
        names.setdefault(split_from_group(group), group)
    batch.group_ids = [names[("train", "validation", "test")[i % 3]] for i in range(8)]
    out = writer(tmp_path, fixed_split=None)
    out.append(batch)
    out.close()
    for split in ("train", "validation", "test"):
        data = SelectorShardDataset(tmp_path, split)
        for row in data:
            i = int(row["features"][0,0])
            assert row["group_ids"] == batch.group_ids[i]
            assert torch.equal(row["event_identity"], batch.event_identity[i])
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[2] / "legged_lab/scripts/analyze_selector_data.py"),
         "--dataset", str(tmp_path)],
        capture_output=True, text=True, env=dict(os.environ), check=True,
    )
    assert "lp_1.0s: 4 failures" in result.stdout
    assert (tmp_path / "collection_analysis.json").is_file()
    report = json.loads((tmp_path / "collection_analysis.json").read_text())
    assert report["events"]["episodes"]["cross_split"] == 2
    assert report["events"]["by_split"]["train"]["episodes"]["with_counterfactual_failure_per_head"] == [0,1,1,2]


def test_replay_rejection_is_stratified_and_cause_contract_errors_are_explicit(tmp_path):
    batch = synthetic_batch()
    audit = CollectionAudit(8, "cpu")
    all_rows = torch.ones(8).bool()
    stable = all_rows.clone()
    stable[5] = False
    audit.record_candidates(batch.fault_metadata, batch.labels,
                            (all_rows, all_rows, all_rows, all_rows, stable), stable)
    result = audit.summary()
    funnel = torch.tensor(result["funnel_by_active_scenario_severity"])
    assert funnel[-2, 3, 2] == 0 and funnel[-3, 3, 2] == 1
    pairs = torch.tensor(result["pairs_by_stage_horizon_active_scenario_severity"])
    assert pairs[0, 1, 3, 2, 3] == 1 and pairs[1, 1, 3, 2, 3] == 0
    out = writer(tmp_path)
    out.append(batch)
    out.close()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"].pop("collection_audit_contract")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="require collection_audit_contract"):
        analyze_selector_dataset(tmp_path)


def test_cli_probability_override_reaches_real_fault_sampler_without_isaac():
    script = Path(__file__).parents[2] / "legged_lab/scripts/collect_selector_data.py"
    tree = ast.parse(script.read_text())
    parser_nodes = []
    collecting = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets):
            collecting = True
        if collecting and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "AppLauncher":
                break
        if collecting:
            parser_nodes.append(node)
    namespace = {"argparse": argparse, "Path": Path}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(script), "exec"), namespace)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    config_node = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fault_cfg" for t in n.targets))
    for arguments, expected in (([], (.1,.3,.6)), (["--fault-severity-probabilities", ".4", ".4", ".2"], (.4,.4,.2))):
        namespace["args_cli"] = namespace["parser"].parse_args(arguments)
        namespace["FaultConfig"] = FaultConfig
        exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(script), "exec"), namespace)
        cfg = namespace["fault_cfg"]
        cfg.validate()
        assert cfg.severity_probabilities == expected
        sampler = SelectorFaultManager(10000, cfg=cfg)
        proportions = torch.bincount(sampler.severity, minlength=3) / 10000
        torch.testing.assert_close(proportions, torch.tensor(expected), atol=.02, rtol=0)
