"""高层 rollout 的解析 GAE、episode 隔离及批量采样检查，不启动仿真。"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from rsl_rl.storage.selector_rollout_storage import SelectorRolloutStorage, SelectorTransition


def _storage(n=3, steps=3, **kwargs):
    return SelectorRolloutStorage(n, steps, 2, 4, **kwargs)


def _transition(n=3, index=0, **kwargs):
    ids = torch.arange(n).float() + n * index
    actions = (ids.long() % 2).reshape(n, 1)
    logits = torch.tensor([[0.3, -0.5]]).repeat(n, 1)
    data = dict(
        observations=ids[:, None].repeat(1, 2), critic_observations=(ids + 100)[:, None].repeat(1, 4),
        actions=actions, rewards=torch.ones(n), dones=torch.zeros(n, dtype=torch.bool),
        values=torch.zeros(n, 1), actions_log_prob=F.log_softmax(logits, -1).gather(-1, actions),
        action_logits=logits,
    )
    data.update(kwargs)
    return SelectorTransition(**data)


def _analytic_rollout(*, reset_episode_reward=7.0):
    storage = _storage()
    rewards = [[1., 2., 3.], [4., 5., 6.], [reset_episode_reward, 8., 9.]]
    values = [[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]]
    for index in range(3):
        done = torch.tensor([index == 1, index == 1, False])
        timeout = torch.tensor([False, index == 1, False])
        storage.add_transitions(_transition(
            index=index, rewards=torch.tensor(rewards[index]), values=torch.tensor(values[index]),
            dones=done, time_outs=timeout, timeout_values=torch.tensor([999., 20., 999.]),
            low_level_steps=torch.tensor([4 if index == 1 else 10, 10, 10]),
        ))
    return storage


def test_gae_matches_hand_calculation_with_failure_timeout_and_rollout_cutoff():
    storage = _analytic_rollout()
    original_rewards = storage.rewards.clone()
    storage.compute_returns(torch.tensor([11., 12., 13.]), gamma=.5, lam=.8, normalize_advantage=False)
    # t=1 的 env0 真实失败，只取奖励4；env1 超时取5+0.5*20=15。
    # t=2 已是这两个环境的新 episode，不能进入 t<=1 的 GAE。
    expected_returns = torch.tensor([[3.1, 8.6, 8.98], [4., 15., 13.2], [12.5, 14., 15.5]])
    expected_advantages = torch.tensor([[1.1, 5.6, 4.98], [-1., 9., 6.2], [4.5, 5., 5.5]])
    torch.testing.assert_close(storage.returns[:, :, 0], expected_returns)
    torch.testing.assert_close(storage.advantages[:, :, 0], expected_advantages)
    torch.testing.assert_close(storage.rewards, original_rewards, rtol=0, atol=0)
    torch.testing.assert_close(storage.timeout_values[1, :, 0], torch.tensor([0., 20., 0.]))
    assert storage.low_level_steps[1, :, 0].tolist() == [4, 10, 10]


def test_reset_episode_rewards_cannot_cross_a_done_boundary():
    first, second = _analytic_rollout(), _analytic_rollout(reset_episode_reward=70000.)
    for storage in (first, second):
        storage.compute_returns(torch.tensor([11., 12., 13.]), .5, .8, normalize_advantage=False)
    torch.testing.assert_close(first.returns[:2], second.returns[:2], rtol=0, atol=0)
    assert first.returns[2, 0] != second.returns[2, 0]


def test_last_transition_bootstraps_only_nonterminal_or_timeout_rows():
    storage = _storage(n=3, steps=1)
    storage.add_transitions(_transition(
        rewards=torch.tensor([1., -4., 2.]), values=torch.tensor([2., 3., 4.]),
        dones=torch.tensor([False, True, True]), time_outs=torch.tensor([False, False, True]),
        timeout_values=torch.tensor([999., 999., 6.]), low_level_steps=torch.tensor([10, 4, 10]),
    ))
    storage.compute_returns(torch.tensor([8., 1000., 2000.]), .5, 1., normalize_advantage=False)
    torch.testing.assert_close(storage.returns[0, :, 0], torch.tensor([5., -4., 5.]))
    torch.testing.assert_close(storage.advantages[0, :, 0], torch.tensor([3., -7., 1.]))


def test_advantage_normalization_leaves_returns_and_rewards_unchanged():
    raw, normalized = _analytic_rollout(), _analytic_rollout()
    raw.compute_returns(torch.tensor([11., 12., 13.]), .5, .8, normalize_advantage=False)
    normalized.compute_returns(torch.tensor([11., 12., 13.]), .5, .8)
    torch.testing.assert_close(normalized.returns, raw.returns, rtol=0, atol=0)
    torch.testing.assert_close(normalized.rewards, raw.rewards, rtol=0, atol=0)
    assert normalized.advantages.mean().abs() < 1e-6
    torch.testing.assert_close(normalized.advantages.std(unbiased=False), torch.tensor(1.))
    single = _storage(n=1, steps=1)
    single.add_transitions(_transition(n=1))
    single.compute_returns(torch.zeros(1), .99, .95)
    assert single.advantages.item() == 0.0
    assert single.returns.item() == 1.0


def test_storage_owns_detached_snapshots_and_preserves_integer_modes():
    storage = _storage(steps=1)
    transition = _transition()
    transition.observations.requires_grad_()
    transition.values.requires_grad_()
    storage.add_transitions(transition)
    expected_obs = storage.observations.clone()
    with torch.no_grad():
        transition.observations.fill_(999.)
        transition.values.fill_(999.)
        transition.actions.fill_(1)
        transition.rewards.fill_(999.)
        transition.action_logits.zero_()
    torch.testing.assert_close(storage.observations, expected_obs, rtol=0, atol=0)
    assert storage.values.sum() == 0 and storage.rewards.sum() == 3
    assert storage.actions.dtype == torch.long and storage.actions[0, :, 0].tolist() == [0, 1, 0]
    assert all(not getattr(storage, name).requires_grad for name in (
        "observations", "critic_observations", "values", "action_logits", "actions_log_prob",
    ))


def test_minibatches_cover_every_sample_each_epoch_and_keep_fields_aligned():
    storage = _storage(n=3, steps=2)
    for index in range(2):
        storage.add_transitions(_transition(index=index))
    storage.compute_returns(torch.zeros(3), .99, .95)

    def collect():
        return list(storage.mini_batch_generator(3, 2, generator=torch.Generator().manual_seed(42)))

    first, second = collect(), collect()
    assert len(first) == 6
    for batch, repeated in zip(first, second):
        assert batch.observations.shape == (2, 2) and batch.old_action_logits.shape == (2, 2)
        ids = batch.observations[:, 0]
        torch.testing.assert_close(batch.critic_observations[:, 0], ids + 100)
        torch.testing.assert_close(batch.actions[:, 0], ids.long() % 2)
        torch.testing.assert_close(batch.observations, repeated.observations, rtol=0, atol=0)
        expected = F.log_softmax(batch.old_action_logits, -1).gather(-1, batch.actions)
        torch.testing.assert_close(batch.old_actions_log_prob, expected)
    orders = [torch.cat([b.observations[:, 0] for b in first[start:start + 3]]) for start in (0, 3)]
    for order in orders:
        assert order.sort().values.tolist() == list(range(6))
    assert not torch.equal(orders[0], orders[1])
    first[0].observations.fill_(999.)
    assert (storage.observations < 999).all()


@pytest.mark.parametrize("change,match", [
    ({"observations": torch.zeros(3, 3)}, "observations"),
    ({"critic_observations": torch.zeros(3, 4, dtype=torch.long)}, "floating"),
    ({"rewards": torch.full((3,), float("nan"))}, "finite"),
    ({"actions": torch.tensor([0., 1., 0.])}, "integer"),
    ({"actions": torch.tensor([0, 2, 0])}, "integer"),
    ({"actions_log_prob": torch.zeros(3)}, "log probabilities"),
    ({"dones": torch.zeros(3, dtype=torch.long)}, "boolean"),
    ({"time_outs": torch.tensor([True, False, False])}, "timeout"),
    ({"dones": torch.ones(3, dtype=torch.bool), "time_outs": torch.ones(3, dtype=torch.bool)}, "terminal critic"),
    ({"low_level_steps": torch.tensor([4, 10, 10])}, "complete high-level"),
    ({"low_level_steps": torch.tensor([0, 10, 10])}, "positive integer"),
    ({"low_level_steps": torch.tensor([11, 10, 10])}, "positive integer"),
    ({"low_level_steps": torch.tensor([10., 10., 10.])}, "positive integer"),
])
def test_rejected_transition_does_not_partially_overwrite_rollout(change, match):
    storage = _storage(steps=2)
    storage.add_transitions(_transition())
    before = {name: value.clone() for name, value in vars(storage).items() if isinstance(value, torch.Tensor)}
    with pytest.raises(ValueError, match=match):
        storage.add_transitions(replace(_transition(index=1), **change))
    assert storage.step == 1
    for name, value in before.items():
        torch.testing.assert_close(getattr(storage, name), value, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 0}, {"num_transitions_per_env": 0}, {"actor_obs_dim": 2.0},
    {"critic_obs_dim": -1}, {"num_modes": 3}, {"low_level_steps_per_decision": False},
])
def test_invalid_storage_configuration_is_rejected(kwargs):
    cfg = dict(num_envs=3, num_transitions_per_env=2, actor_obs_dim=2, critic_obs_dim=4)
    cfg.update(kwargs)
    with pytest.raises(ValueError):
        SelectorRolloutStorage(**cfg)


def test_lifecycle_guards_and_clear_prevent_stale_rollout_use():
    storage = _storage(steps=1)
    with pytest.raises(RuntimeError, match="complete"):
        storage.compute_returns(torch.zeros(3), .99, .95)
    with pytest.raises(RuntimeError, match="returns"):
        list(storage.mini_batch_generator(1))
    storage.add_transitions(_transition())
    with pytest.raises(OverflowError):
        storage.add_transitions(_transition())
    with pytest.raises(ValueError, match="gamma and lam"):
        storage.compute_returns(torch.zeros(3), .99, 1.1)
    storage.compute_returns(torch.zeros(3), .99, .95)
    with pytest.raises(ValueError, match="evenly"):
        list(storage.mini_batch_generator(2))
    storage.clear()
    assert storage.step == 0 and not storage.returns_ready
    with pytest.raises(RuntimeError, match="returns"):
        list(storage.mini_batch_generator(1))
    storage.add_transitions(_transition(rewards=torch.full((3,), 2.)))
    storage.compute_returns(torch.zeros(3), .99, .95)
    torch.testing.assert_close(storage.returns, torch.full((1, 3, 1), 2.))
