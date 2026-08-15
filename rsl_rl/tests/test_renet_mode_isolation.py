from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
REWARDS_PATH = REPO_ROOT / "legged_lab" / "mdp" / "rewards.py"
MODE_MANAGER_PATH = REPO_ROOT / "legged_lab" / "utils" / "mode_aware_reward_manager.py"


def _load_functions(path: Path, class_name: str | None, *function_names):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    selected = []
    for function_name in function_names:
        function = next(
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        function.decorator_list = []
        selected.append(function)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "torch": torch,
        "BaseEnv": object,
        "G1ROUGHEnv": object,
        "G1Env": object,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in function_names)


def _load_mode_aware_reward_manager(monkeypatch):
    fake_isaaclab = ModuleType("isaaclab")
    fake_managers = ModuleType("isaaclab.managers")
    fake_managers.RewardManager = object
    fake_isaaclab.managers = fake_managers
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", fake_managers)

    spec = importlib.util.spec_from_file_location(
        "test_mode_aware_reward_manager",
        MODE_MANAGER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ModeAwareRewardManager


def _make_mode_manager(manager_class, raw_value, term_name="undesired_contacts", weight=1.0):
    manager = manager_class.__new__(manager_class)
    manager.num_envs = raw_value.shape[0]
    manager._env = SimpleNamespace(raw_value=raw_value)
    manager._term_names = [term_name]
    manager._term_cfgs = [
        SimpleNamespace(
            func=lambda env: env.raw_value,
            params={},
            weight=weight,
        )
    ]
    manager._reward_buf = torch.zeros(raw_value.shape[0])
    manager._episode_sums = {term_name: torch.zeros(raw_value.shape[0])}
    manager._step_reward = torch.zeros(raw_value.shape[0], 1)
    return manager


def test_mode_aware_reward_manager_masks_before_all_accumulation(monkeypatch):
    manager_class = _load_mode_aware_reward_manager(monkeypatch)
    raw_value = torch.tensor([1.0, 2.0, 100.0, 200.0])
    manager = _make_mode_manager(manager_class, raw_value)
    active_mask = torch.tensor([True, True, False, False])

    reward = manager.compute(1.0, active_mask=active_mask)

    expected = torch.tensor([1.0, 2.0, 0.0, 0.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(manager._episode_sums["undesired_contacts"], expected)
    torch.testing.assert_close(manager._step_reward[:, 0], expected)


def test_enter_recovery_penalty_keeps_action_time_locomotion_ownership(monkeypatch):
    manager_class = _load_mode_aware_reward_manager(monkeypatch)
    manager = _make_mode_manager(
        manager_class,
        raw_value=torch.tensor([1.0]),
        term_name="enter_recovery_penalty",
        weight=-200.0,
    )
    # check_reset() has already changed the new mode, but action_t was Locomotion.
    manager._env.recovery_mask = torch.tensor([True])
    recovery_mask_t = torch.tensor([False])

    reward = manager.compute(0.02, active_mask=~recovery_mask_t)

    torch.testing.assert_close(reward, torch.tensor([-4.0]))
    torch.testing.assert_close(
        manager._episode_sums["enter_recovery_penalty"],
        torch.tensor([-4.0]),
    )


def test_mode_aware_reward_manager_rejects_invalid_mask_and_dt(monkeypatch):
    manager_class = _load_mode_aware_reward_manager(monkeypatch)
    manager = _make_mode_manager(manager_class, torch.ones(4))
    with pytest.raises(TypeError, match="dtype bool"):
        manager.compute(1.0, active_mask=torch.ones(4))
    with pytest.raises(ValueError, match="shape"):
        manager.compute(1.0, active_mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="positive and finite"):
        manager.compute(0.0, active_mask=torch.ones(4, dtype=torch.bool))


def test_action_rate_histories_are_isolated_across_l_l_r_r_l_l():
    update_loco, update_recovery = _load_functions(
        ENV_PATH,
        "G1RENetEnv",
        "_update_locomotion_action_rate",
        "_update_recovery_action_rate",
    )
    env = SimpleNamespace(
        action=torch.zeros(1, 2),
        locomotion_prev_action=torch.zeros(1, 2),
        locomotion_prev_action_valid=torch.zeros(1, dtype=torch.bool),
        locomotion_action_rate_value=torch.zeros(1),
        locomotion_action_rate_valid_sample=torch.zeros(1, dtype=torch.bool),
        recovery_prev_action=torch.zeros(1, 2),
        recovery_prev_action_valid=torch.zeros(1, dtype=torch.bool),
        recovery_action_rate_value=torch.zeros(1),
        recovery_action_rate_valid_sample=torch.zeros(1, dtype=torch.bool),
    )
    recovery_modes = [False, False, True, True, False, False]
    actions = [
        [0.0, 0.0],
        [1.0, 2.0],
        [10.0, 10.0],
        [13.0, 14.0],
        [2.0, 4.0],
        [5.0, 8.0],
    ]
    loco_values = []
    recovery_values = []
    for is_recovery, action in zip(recovery_modes, actions):
        env.action = torch.tensor([action])
        mask = torch.tensor([is_recovery])
        update_loco(env, mask)
        update_recovery(env, mask)
        loco_values.append(float(env.locomotion_action_rate_value.item()))
        recovery_values.append(float(env.recovery_action_rate_value.item()))

    assert loco_values == [0.0, 5.0, 0.0, 0.0, 0.0, 25.0]
    assert recovery_values == [0.0, 0.0, 0.0, 25.0, 0.0, 0.0]
    assert env.recovery_prev_action_valid.item() is False


def test_locomotion_feet_timers_reset_through_recovery_and_restart_per_substep():
    (update_timers,) = _load_functions(
        ENV_PATH,
        "G1RENetEnv",
        "_update_locomotion_feet_timers",
    )
    (feet_reward,) = _load_functions(
        REWARDS_PATH,
        None,
        "locomotion_feet_air_time_positive_biped",
    )
    contact_time = torch.tensor([[0.01, 0.0]])
    env = SimpleNamespace(
        physics_dt=0.01,
        feet_cfg=SimpleNamespace(body_ids=[0, 1]),
        contact_sensor=SimpleNamespace(
            data=SimpleNamespace(current_contact_time=contact_time)
        ),
        locomotion_feet_air_time=torch.zeros(1, 2),
        locomotion_feet_contact_time=torch.zeros(1, 2),
        command_generator=SimpleNamespace(command=torch.tensor([[1.0, 0.0, 0.0]])),
    )

    update_timers(env, torch.tensor([False]))
    update_timers(env, torch.tensor([False]))
    torch.testing.assert_close(
        env.locomotion_feet_contact_time,
        torch.tensor([[0.02, 0.0]]),
    )
    torch.testing.assert_close(
        env.locomotion_feet_air_time,
        torch.tensor([[0.0, 0.02]]),
    )

    # Four seconds of Recovery substeps must never accumulate in Loco timers.
    for _ in range(400):
        update_timers(env, torch.tensor([True]))
    assert torch.count_nonzero(env.locomotion_feet_air_time) == 0
    assert torch.count_nonzero(env.locomotion_feet_contact_time) == 0

    update_timers(env, torch.tensor([False]))
    torch.testing.assert_close(
        env.locomotion_feet_contact_time,
        torch.tensor([[0.01, 0.0]]),
    )
    torch.testing.assert_close(
        env.locomotion_feet_air_time,
        torch.tensor([[0.0, 0.01]]),
    )
    torch.testing.assert_close(feet_reward(env, 0.4), torch.tensor([0.01]))

    update_timers(env, torch.tensor([False]))
    torch.testing.assert_close(
        env.locomotion_feet_contact_time,
        torch.tensor([[0.02, 0.0]]),
    )
    torch.testing.assert_close(
        env.locomotion_feet_air_time,
        torch.tensor([[0.0, 0.02]]),
    )
    torch.testing.assert_close(feet_reward(env, 0.4), torch.tensor([0.02]))

    env.command_generator.command.zero_()
    torch.testing.assert_close(feet_reward(env, 0.4), torch.tensor([0.0]))
