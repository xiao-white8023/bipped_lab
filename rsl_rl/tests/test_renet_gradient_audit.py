"""Gradient-audit mathematics and state preservation without Isaac Sim."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributions import Normal


PATH = Path(__file__).resolve().parents[2] / "legged_lab/utils/renet_gradient_audit.py"
SPEC = importlib.util.spec_from_file_location("renet_gradient_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ToyPolicy(nn.Module):
    single_proprio_dim = 2
    estimator_latent_dim = 1

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 1))
        self.std = nn.Parameter(torch.tensor([0.8]))
        self.critic = nn.Linear(4, 1)

    def update_distribution(self, observations):
        common = observations[:, :2]
        mode = observations[:, 2:3]
        self.distribution = Normal(self.actor(torch.cat((common, mode, 1 - mode), dim=1)), self.std.expand(observations.shape[0], 1))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


def batch(policy, mode):
    observations = torch.tensor([[0.2, -0.3, mode], [0.4, 0.1, mode], [-0.1, 0.5, mode], [0.8, -0.4, mode]])
    with torch.no_grad():
        policy.update_distribution(observations)
        actions = policy.distribution.mean + torch.tensor([[0.3], [-0.2], [0.1], [-0.4]])
        log_probs = policy.get_actions_log_prob(actions)
    return {"observations": observations, "actions": actions,
            "old_log_probs": log_probs, "advantages": torch.tensor([1., -0.4, 0.8, -0.9])}


@pytest.mark.parametrize("vp, expected", [([2., 4.], 1.), ([-2., -4.], -1.), ([0., 0.], None)])
def test_known_gradient_geometry(vp, expected):
    result = AUDIT.gradient_pair_metrics(torch.tensor([1., 2.]), torch.tensor(vp))
    assert result["cosine"] is None if expected is None else result["cosine"] == pytest.approx(expected)
    assert result["negative"] == (expected < 0 if expected is not None else None)
    assert result["vp_loss_change_per_unit_lp_gradient_step"] == -sum(a * b for a, b in zip([1., 2.], vp))


def test_mode_columns_are_separate_and_shared_block_matches_parameter_count():
    torch.manual_seed(7)
    policy = ToyPolicy()
    result = AUDIT.analyze_paired_ppo_gradients(policy, batch(policy, 1.), batch(policy, 0.))
    for key in ("surrogate_only", "surrogate_plus_entropy"):
        blocks = result[key]
        assert blocks["specific_columns"]["dot"] == 0
        assert blocks["lp_latent_columns"]["lp_norm"] > 0
        assert blocks["lp_latent_columns"]["vp_norm"] == 0
        assert blocks["vp_latent_columns"]["lp_norm"] == 0
        assert blocks["vp_latent_columns"]["vp_norm"] > 0
        assert blocks["shared_actor"]["num_parameters"] == 13
        assert blocks["whole_actor"]["num_parameters"] == 19
        assert blocks["std"]["num_parameters"] == 1
        assert set(blocks["linear_layers"]) == {"0", "2"}
    # A Normal policy's entropy does not depend on actor mean.
    assert result["surrogate_only"]["whole_actor"] == result["surrogate_plus_entropy"]["whole_actor"]


def test_opposing_advantages_produce_opposing_shared_ppo_gradients():
    torch.manual_seed(5)
    policy = ToyPolicy()
    # Equal latent input weights make LP/VP means identical. Reversing the
    # advantages then reverses every shared actor gradient at ratio=1.
    with torch.no_grad():
        policy.actor[0].weight[:, 3] = policy.actor[0].weight[:, 2]
    lp, vp = batch(policy, 1.), batch(policy, 0.)
    vp["advantages"] = -lp["advantages"]
    result = AUDIT.analyze_paired_ppo_gradients(policy, lp, vp)
    shared = result["surrogate_only"]["shared_actor"]
    assert shared["cosine"] == pytest.approx(-1.)
    assert shared["negative"]
    assert shared["vp_loss_change_per_unit_lp_gradient_step"] > 0
    assert result["surrogate_only"]["specific_columns"]["dot"] == 0


def test_zero_advantages_do_not_report_a_conflict():
    policy = ToyPolicy()
    lp, vp = batch(policy, 1.), batch(policy, 0.)
    lp["advantages"].zero_()
    vp["advantages"].zero_()
    result = AUDIT.analyze_paired_ppo_gradients(policy, lp, vp)
    shared = result["surrogate_only"]["shared_actor"]
    assert shared["lp_norm"] == shared["vp_norm"] == 0
    assert shared["cosine"] is None and shared["negative"] is None


def test_parameters_grad_flags_and_input_graphs_are_preserved():
    torch.manual_seed(3)
    policy = ToyPolicy()
    lp, vp = batch(policy, 1.), batch(policy, 0.)
    for data in (lp, vp):
        for value in data.values():
            value.requires_grad_(True)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    policy.actor[0].eval()  # Preserve heterogeneous module modes as well.
    flags = [module.training for module in policy.modules()]
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 7.)
        parameter.requires_grad_(False)
    result = AUDIT.analyze_paired_ppo_gradients(policy, lp, vp)
    assert result["losses"]["lp"]["ratio_mean"] == pytest.approx(1.)
    assert result["losses"]["vp"]["used_advantage_sample_std"] == pytest.approx(1.)
    assert [module.training for module in policy.modules()] == flags
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])
    for parameter in policy.parameters():
        assert not parameter.requires_grad
        assert torch.all(parameter.grad == 7.)
    for data in (lp, vp):
        assert all(value.grad is None for value in data.values())


def test_empty_and_nonfinite_batches_are_rejected_and_flags_restored():
    policy = ToyPolicy()
    lp, vp = batch(policy, 1.), batch(policy, 0.)
    lp["old_log_probs"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        AUDIT.analyze_paired_ppo_gradients(policy, lp, vp)
    lp = {key: value[:1] for key, value in batch(policy, 1.).items()}
    with pytest.raises(ValueError, match="at least two"):
        AUDIT.analyze_paired_ppo_gradients(policy, lp, vp)
    assert policy.training
    assert all(parameter.requires_grad for parameter in policy.parameters())
