from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import pytest
import torch
import torch.nn as nn

from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.utils import Normalizer, RecoveryAMPData


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


def _make_discriminator(observation_dim):
    return Discriminator(
        input_dim=observation_dim * 2,
        amp_reward_coef=0.3,
        hidden_layer_sizes=[256, 128],
        device="cpu",
        task_reward_lerp=0.0,
    )


def test_existing_discriminator_forward_grad_penalty_and_reward(amp_data):
    batch_size = 128
    discriminator = _make_discriminator(amp_data.observation_dim)
    normalizer = Normalizer(amp_data.observation_dim)
    expert_state, expert_next_state = amp_data.sample_transition_batch(batch_size)
    policy_state = expert_state + 0.01 * torch.randn_like(expert_state)
    policy_next_state = expert_next_state + 0.01 * torch.randn_like(expert_next_state)

    assert discriminator.input_dim == 390
    expert_input = torch.cat((expert_state, expert_next_state), dim=-1)
    assert expert_input.shape == (batch_size, 390)
    expert_logits = discriminator(expert_input)
    assert expert_logits.shape == (batch_size, 1)
    assert torch.isfinite(expert_logits).all()

    grad_pen = discriminator.compute_grad_pen(expert_state, expert_next_state)
    assert grad_pen.ndim == 0
    assert torch.isfinite(grad_pen)

    reward, policy_logits = discriminator.predict_amp_reward(
        policy_state,
        policy_next_state,
        torch.zeros(batch_size),
        normalizer=normalizer,
    )
    assert reward.shape == (batch_size,)
    assert policy_logits.shape == (batch_size, 1)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(policy_logits).all()


def test_existing_ampppo_generator_loss_and_backward(amp_data):
    discriminator = _make_discriminator(amp_data.observation_dim)
    algorithm = AMPPPO(
        policy=nn.Linear(1, 1),
        discriminator=discriminator,
        amp_data=amp_data,
        amp_normalizer=Normalizer(amp_data.observation_dim),
        amp_replay_buffer_size=1024,
        device="cpu",
    )
    assert algorithm.amp_storage.states.shape == (1024, 195)
    assert algorithm.amp_storage.next_states.shape == (1024, 195)

    expert_state, expert_next_state = amp_data.sample_transition_batch(256)
    policy_state = expert_state + 0.01 * torch.randn_like(expert_state)
    policy_next_state = expert_next_state + 0.01 * torch.randn_like(expert_next_state)
    algorithm.amp_storage.insert(policy_state, policy_next_state)

    sample_amp = next(
        algorithm._amp_mini_batch_generator(num_updates=1, mini_batch_size=64)
    )
    sample_policy, sample_expert = sample_amp
    for state, next_state in (sample_policy, sample_expert):
        assert state.shape == (64, 195)
        assert next_state.shape == (64, 195)
        assert torch.isfinite(state).all()
        assert torch.isfinite(next_state).all()

    total_loss, metrics, normalizer_updates = algorithm._compute_amp_loss(sample_amp)
    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert len(normalizer_updates) == 1
    assert set(metrics) == {
        "amp",
        "amp_grad_pen",
        "amp_policy_pred",
        "amp_expert_pred",
    }
    assert all(torch.isfinite(value) for value in metrics.values())

    algorithm.optimizer.zero_grad()
    total_loss.backward()
    gradients = [
        parameter.grad
        for parameter in discriminator.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
