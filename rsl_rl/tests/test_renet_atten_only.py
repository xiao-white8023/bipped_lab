from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from legged_lab.perception.lp_height import LPHeightCfg, observation_contract, resolve_pgtt_height_cfg
from rsl_rl.runners import RENetAmpOnPolicyRunner
from test_renet_lp_vp_training import _make_policy


ROOT = Path(__file__).parents[2]
SOURCE = "pgtt_full_atten"
REMOVED = ("pgtt_front", "pgtt_full", "mid360")


def _function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("script", ["train.py", "play_new_renet.py", "check_renet_lp.py", "sim2sim_renet.py"])
def test_entry_points_accept_only_the_ame_source(script):
    parse = _function(f"legged_lab/scripts/{script}", "parse_lp_height_source", {"argparse": argparse})
    assert parse(SOURCE) == SOURCE
    for source in (*REMOVED, "pgtt", "unknown"):
        with pytest.raises(argparse.ArgumentTypeError, match=SOURCE):
            parse(source)


def test_task_configure_preserves_ame_dimensions_and_rejects_removed_sources():
    configure = _function("legged_lab/envs/g1/RENet_cfg.py", "configure_agent_for_lp_source",
                          {"resolve_pgtt_height_cfg": resolve_pgtt_height_cfg})
    env = SimpleNamespace(lp_height=LPHeightCfg(), renet=SimpleNamespace(lp_height_source=SOURCE))
    agent = SimpleNamespace(policy=SimpleNamespace(LidarCnnMlp=SimpleNamespace()))
    assert configure(env, agent) is agent
    assert agent.policy.LidarCnnMlp.input_dim == (11, 17)
    assert agent.policy.LidarCnnMlp.input_channels == agent.policy.lp_height_history_frames == 1
    assert agent.policy.lp_encoder_type == "ame_point_cross_attention"
    assert agent.policy.ame_lp_query_history_frames == 5
    assert agent.policy.ame_lp_attention_num_heads == 16
    assert not agent.policy.lp_terrain_recon_front_only
    for source in REMOVED:
        env.renet.lp_height_source = source
        with pytest.raises(ValueError, match=SOURCE):
            configure(env, agent)


@pytest.mark.parametrize("mismatch", ["shape", "target", "encoder", "missing_contract"])
def test_runner_rejects_incompatible_lp_metadata_before_loading_any_weights(monkeypatch, mismatch):
    import torch

    policy = _make_policy()
    encoder = policy.get_lp_encoder_contract()
    contract = observation_contract(resolve_pgtt_height_cfg(LPHeightCfg(), SOURCE), SOURCE,
                                    height_offset=.5, height_scale=1.)
    saved = {"lp_observation_contract": copy.deepcopy(contract),
             "lp_encoder_contract": copy.deepcopy(encoder)}
    if mismatch == "shape":
        saved["lp_observation_contract"]["shape"] = [11, 9]
    elif mismatch == "target":
        saved["lp_observation_contract"]["lp_target"] = "front_99"
    elif mismatch == "encoder":
        saved["lp_encoder_contract"] = None
    else:
        saved["lp_observation_contract"] = None
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: saved)
    runner = SimpleNamespace(device="cpu", env=SimpleNamespace(lp_observation_contract=contract,
                             lp_height_source=SOURCE), lp_encoder_contract=encoder,
                             vp_encoder_contract=policy.get_vp_encoder_contract(),
                             critic_observation_contract=None, alg=SimpleNamespace(policy=policy))
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    with pytest.raises(ValueError, match="Incompatible pgtt_full_atten checkpoint"):
        RENetAmpOnPolicyRunner.load(runner, "unused.pt", load_optimizer=False)
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_task_and_environment_have_no_removed_source_branches():
    for path in ("legged_lab/envs/g1/RENet_cfg.py", "legged_lab/envs/g1/RENet_env.py",
                 "legged_lab/scripts/sim2sim_renet.py"):
        tree = ast.parse((ROOT / path).read_text())
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        assert not literals.intersection(REMOVED)
    config = (ROOT / "legged_lab/envs/g1/RENet_cfg.py").read_text()
    assert "lidar=None" in config and "LidarElevationMapCfg" not in config
    environment = (ROOT / "legged_lab/envs/g1/RENet_env.py").read_text()
    assert "update_mid360" not in environment and "RollingElevationMap" not in environment
