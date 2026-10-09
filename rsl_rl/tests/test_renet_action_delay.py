from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_renet_auxiliary_updates import ROOT, _extract_function


@pytest.fixture
def native_delay_buffer():
    # These two IsaacLab classes depend only on Torch. Compile their actual
    # class definitions to test the native buffer without launching Kit or
    # importing isaaclab.utils, whose other utilities require simulator plugins.
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        pytest.skip("IsaacLab is required for native delay-buffer behavior")
    directory = Path(next(iter(spec.submodule_search_locations))) / "utils/buffers"
    scope = {"torch": torch}
    for filename, name in (("circular_buffer.py", "CircularBuffer"), ("delay_buffer.py", "DelayBuffer")):
        path = directory / filename
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["DelayBuffer"]


def _environment(buffer_class, *, enabled=True, minimum=0, maximum=5, num_envs=8,
                 start_iteration=0, iteration=0):
    path = ROOT / "legged_lab/envs/g1/RENet_env.py"
    initialize = _extract_function(path, "_init_action_delay", "G1RENetEnv")
    initialize.__globals__["DelayBuffer"] = buffer_class
    resample = _extract_function(path, "_reset_action_delay", "G1RENetEnv")
    active = _extract_function(path, "_domain_rand_is_active", "G1RENetEnv")
    env = SimpleNamespace(num_envs=num_envs, num_actions=23, device="cpu", step_dt=.02, extras={},
        renet_training_iteration=iteration,
        cfg=SimpleNamespace(renet=SimpleNamespace(domain_rand_start_iteration=start_iteration),
                            domain_rand=SimpleNamespace(action_delay=SimpleNamespace(
            enable=enabled, params={"min_delay": minimum, "max_delay": maximum}))))
    env._domain_rand_is_active = lambda: active(env)
    initialize(env)
    env.resample = lambda ids: resample(env, ids)
    env._reset_action_delay = env.resample
    return env


def test_episode_sampling_includes_zero_and_five_and_does_not_change_other_environments(native_delay_buffer):
    torch.manual_seed(33)
    env = _environment(native_delay_buffer, num_envs=64)
    ids = torch.arange(64)
    env.resample(ids)
    original = env.action_buffer.time_lags.clone()
    assert set(original.tolist()) == set(range(6))
    # An episode's lag remains fixed while actions are queued.
    env.action_buffer.compute(torch.ones(64, 23))
    torch.testing.assert_close(env.action_buffer.time_lags, original)
    reset_ids = torch.tensor([2, 7, 20])
    env.action_delay_min_steps = env.action_delay_max_steps = 4
    env.action_buffer.reset(reset_ids)
    env.resample(reset_ids)
    remaining = torch.ones(64, dtype=torch.bool)
    remaining[reset_ids] = False
    torch.testing.assert_close(env.action_buffer.time_lags[remaining], original[remaining])
    assert (env.action_buffer.time_lags[reset_ids] == 4).all()
    assert env.extras["log"]["ActionDelay/mean_ms"].item() == pytest.approx(
        env.action_buffer.time_lags.float().mean().item() * 20.)


def test_native_buffer_executes_the_action_two_control_steps_ago_and_clears_reset_history(native_delay_buffer):
    env = _environment(native_delay_buffer, minimum=2, maximum=2, num_envs=2)
    ids = torch.arange(2)
    env.action_buffer.reset(ids)
    env.resample(ids)
    outputs = [env.action_buffer.compute(torch.full((2, 23), float(k)))[0, 0].item() for k in range(6)]
    assert outputs == [0., 0., 0., 1., 2., 3.]
    # Reset only environment 0: it must never execute the previous episode's queued action.
    reset_ids = torch.tensor([0])
    env.action_buffer.reset(reset_ids)
    env.resample(reset_ids)
    output = env.action_buffer.compute(torch.tensor([[100.] * 23, [6.] * 23]))
    assert output[0, 0].item() == 100.
    assert output[1, 0].item() == 4.


def test_disabled_delay_and_enabled_zero_delay_execute_current_action(native_delay_buffer):
    for enabled in (False, True):
        env = _environment(native_delay_buffer, enabled=enabled, maximum=0 if enabled else 5, num_envs=2)
        env.action_buffer.set_time_lag(0 if enabled else 5)
        env.resample(torch.arange(2))
        assert not env.action_buffer.time_lags.any()
        for k in range(3):
            action = torch.full((2, 23), float(k))
            torch.testing.assert_close(env.action_buffer.compute(action), action)


@pytest.mark.parametrize("minimum,maximum", [(-1, 5), (3, 2), (0, -1), (True, 5), (0, 2.5)])
def test_invalid_delay_range_is_rejected_before_buffer_creation(native_delay_buffer, minimum, maximum):
    with pytest.raises(ValueError, match="action_delay"):
        _environment(native_delay_buffer, minimum=minimum, maximum=maximum)
