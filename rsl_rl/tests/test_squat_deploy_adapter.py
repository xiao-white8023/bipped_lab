from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "leggeddeploy"))

from config_squart import Config  # noqa: E402
from squat_policy_adapter import resolve_squat_policy_layout  # noqa: E402


class _CurrentCENetExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cenet_module = torch.nn.Linear(800, 22)
        self.actor = torch.nn.Linear(102, 15)

    def forward(self, observation_history):
        context = self.cenet_module(observation_history)
        current_observation = observation_history[:, -80:]
        return self.actor(torch.cat((current_observation, context), dim=-1))


def test_deploy_adapter_detects_current_cenet_800_input():
    policy = torch.jit.trace(_CurrentCENetExport(), torch.zeros(1, 800))

    layout = resolve_squat_policy_layout(
        policy,
        history_length=10,
        num_actions=15,
        requested_layout="auto",
        configured_single_obs_dim=80,
    )

    assert layout.name == "current"
    assert layout.single_obs_dim == 80
    assert layout.policy_obs_dim == 800
    assert layout.has_cenet
    assert not layout.includes_root_height


def test_deploy_adapter_auto_detects_legacy_810_input_despite_current_yaml():
    policy = torch.jit.trace(torch.nn.Linear(810, 15), torch.zeros(1, 810))

    layout = resolve_squat_policy_layout(
        policy,
        history_length=10,
        num_actions=15,
        requested_layout="auto",
        configured_single_obs_dim=80,
    )

    assert layout.name == "legacy_root_height"
    assert layout.single_obs_dim == 81
    assert layout.policy_obs_dim == 810
    assert layout.includes_root_height
    assert not layout.has_cenet


def test_deploy_adapter_strict_current_mode_rejects_legacy_policy():
    policy = torch.jit.trace(torch.nn.Linear(810, 15), torch.zeros(1, 810))

    with pytest.raises(RuntimeError, match=r"must accept \[1, 800\]"):
        resolve_squat_policy_layout(
            policy,
            history_length=10,
            num_actions=15,
            requested_layout="current",
            configured_single_obs_dim=80,
        )


def test_squat_deploy_yaml_defaults_to_current_observation_contract():
    config_path = REPO_ROOT / "leggeddeploy" / "config" / "g1_squart.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["policy_obs_layout"] == "auto"
    assert config["num_obs"] == 80
    assert config["obs_history_length"] == 10
    assert config["num_policy_obs"] == 800
    assert config["action_filter"] == {
        "enabled": True,
        "cutoff_hz": 5.0,
    }


def test_legacy_yaml_without_action_filter_defaults_to_passthrough(tmp_path):
    config_path = REPO_ROOT / "leggeddeploy" / "config" / "g1_squart.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    config_data.pop("action_filter")
    legacy_config_path = tmp_path / "legacy_g1_squart.yaml"
    legacy_config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = Config(legacy_config_path)

    assert not config.action_filter_enabled
    assert config.action_filter_cutoff_hz == 5.0


def test_enabled_action_filter_rejects_cutoff_at_nyquist(tmp_path):
    config_path = REPO_ROOT / "leggeddeploy" / "config" / "g1_squart.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)
    config_data["action_filter"]["cutoff_hz"] = 25.0
    invalid_config_path = tmp_path / "invalid_g1_squart.yaml"
    invalid_config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="cutoff_hz"):
        Config(invalid_config_path)


def test_filtered_action_is_used_for_observation_and_leg_waist_targets_only():
    deploy_source = (
        REPO_ROOT / "leggeddeploy" / "deploy_squart.py"
    ).read_text(encoding="utf-8")

    assert "self.action = self.action_filter.update(raw_action)" in deploy_source
    assert "self.current_obs[idx : idx + self.config.num_actions] = self.action" in deploy_source
    assert "+ self.action * self.config.action_scale" in deploy_source
    assert "self.action_filter.update(right_arm" not in deploy_source
