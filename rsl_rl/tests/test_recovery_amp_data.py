from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import pytest
import torch

from rsl_rl.utils import RecoveryAMPData


MOTION_DIR = (
    REPO_ROOT
    / "legged_lab"
    / "envs"
    / "g1"
    / "datasets"
    / "npz"
    / "recovery_candidates_v3"
)


@pytest.fixture(scope="module")
def amp_data():
    return RecoveryAMPData(
        motion_dir=MOTION_DIR,
        device="cpu",
        transition_dt=0.01,
    )


def _assert_feature_batch(feature, batch_size):
    assert feature.shape == (batch_size, 195)
    assert feature.dtype == torch.float32
    assert feature.device.type == "cpu"
    assert torch.isfinite(feature).all()


def test_initialization_and_state_sampling(amp_data):
    assert amp_data.observation_dim == 195
    assert amp_data.transition_dt == 0.01
    assert amp_data.num_motions == 18
    assert amp_data.anchor_body_name == "torso_link"
    assert len(amp_data.target_body_names) == 13
    _assert_feature_batch(amp_data.sample_state_batch(512), 512)


def test_transition_sampling(amp_data):
    state, next_state = amp_data.sample_transition_batch(512)
    _assert_feature_batch(state, 512)
    _assert_feature_batch(next_state, 512)


def test_feed_forward_generator(amp_data):
    batches = list(
        amp_data.feed_forward_generator(num_mini_batch=4, mini_batch_size=256)
    )
    assert len(batches) == 4
    for state, next_state in batches:
        _assert_feature_batch(state, 256)
        _assert_feature_batch(next_state, 256)


def test_preloaded_transition_generator(monkeypatch):
    data = RecoveryAMPData(
        motion_dir=MOTION_DIR,
        device="cpu",
        transition_dt=0.01,
        preload_transitions=True,
        num_preload_transitions=2048,
    )
    _assert_feature_batch(data.preloaded_s, 2048)
    _assert_feature_batch(data.preloaded_s_next, 2048)

    selected_indices = torch.tensor([1, 7, 1, 13], dtype=torch.long)
    randint_calls = []

    def _fixed_randint(high, size, dtype, device):
        randint_calls.append((high, size, dtype, device))
        return selected_indices.to(device=device)

    monkeypatch.setattr(torch, "randint", _fixed_randint)
    state, next_state = next(
        data.feed_forward_generator(num_mini_batch=1, mini_batch_size=4)
    )
    _assert_feature_batch(state, 4)
    _assert_feature_batch(next_state, 4)
    assert len(randint_calls) == 1
    torch.testing.assert_close(state, data.preloaded_s[selected_indices])
    torch.testing.assert_close(next_state, data.preloaded_s_next[selected_indices])


@pytest.mark.parametrize("transition_dt", [0.01, 0.02])
def test_configurable_transition_dt(transition_dt):
    data = RecoveryAMPData(
        motion_dir=MOTION_DIR,
        device="cpu",
        transition_dt=transition_dt,
    )
    state, next_state = data.sample_transition_batch(512)
    _assert_feature_batch(state, 512)
    _assert_feature_batch(next_state, 512)


@pytest.mark.parametrize("transition_dt", [0.0, -0.01, float("inf")])
def test_rejects_invalid_transition_dt(transition_dt):
    with pytest.raises(ValueError, match="transition_dt"):
        RecoveryAMPData(motion_dir=MOTION_DIR, transition_dt=transition_dt)


def test_preload_requires_positive_count():
    with pytest.raises(ValueError, match="num_preload_transitions"):
        RecoveryAMPData(
            motion_dir=MOTION_DIR,
            preload_transitions=True,
            num_preload_transitions=0,
        )
