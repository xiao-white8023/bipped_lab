"""分类 PPO 的解析损失、超时接入和实际高层网络更新检查，不启动仿真。"""

import ast
import math
from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical

from rsl_rl.algorithms.discrete_ppo import DiscretePPO, categorical_kl
from rsl_rl.modules.selector_actor_critic import SelectorActorCritic
from rsl_rl.storage.selector_rollout_storage import SelectorMiniBatch


class TinyCategoricalPolicy(torch.nn.Module):
    """只用于解析样例：初始 P(VP)=3/4，V(s)=Critic 输入的第一维。"""

    is_recurrent = False
    action_distribution = "categorical"
    num_modes, num_actions, num_actor_obs, num_critic_obs = 2, 1, 2, 4

    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(4, 1)
        self.frozen_parameter = torch.nn.Parameter(torch.tensor(123.), requires_grad=False)
        with torch.no_grad():
            self.actor.weight.zero_()
            self.actor.bias.copy_(torch.tensor([math.log(3.), 0.]))
            self.critic.weight.copy_(torch.tensor([[1., 0., 0., 0.]]))
            self.critic.bias.zero_()
        self.evaluated_batches = []

    def act(self, obs):
        self.action_logits = self.actor(obs)
        self.distribution = Categorical(logits=self.action_logits)
        return self.distribution.sample().unsqueeze(-1)

    def evaluate(self, obs):
        self.evaluated_batches.append(obs.detach().clone())
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions[:, 0])

    def evaluate_actions(self, obs, critic_obs, actions):
        self.action_logits = self.actor(obs)
        self.distribution = Categorical(logits=self.action_logits)
        return self.evaluate(critic_obs), self.get_actions_log_prob(actions), self.distribution.entropy()

    def reset(self, dones):
        self.last_dones = dones.clone()


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _algorithm(n=3, steps=1, **kwargs):
    cfg = dict(num_learning_epochs=2, num_mini_batches=1, schedule="fixed")
    cfg.update(kwargs)
    algorithm = DiscretePPO(TinyCategoricalPolicy(), **cfg)
    algorithm.init_storage(n, steps)
    return algorithm


def _observations(n=3):
    return torch.randn(n, 2), torch.arange(1, n + 1).float()[:, None].repeat(1, 4)


def _finish_rollout(algorithm):
    for _ in range(algorithm.storage.num_transitions_per_env):
        actor, critic = _observations(algorithm.storage.num_envs)
        algorithm.act(actor, critic)
        algorithm.process_env_step(torch.ones(actor.shape[0]), torch.zeros(actor.shape[0], dtype=torch.bool), {})
    algorithm.compute_returns(critic + 3.)


def _loss_batch():
    n = 4
    critic = torch.zeros(n, 4)
    critic[:, 0] = torch.tensor([1., 0., 3., -1.])
    return SelectorMiniBatch(
        observations=torch.zeros(n, 2), critic_observations=critic,
        actions=torch.tensor([[0], [0], [1], [1]]), old_values=torch.zeros(n, 1), returns=torch.ones(n, 1),
        advantages=torch.tensor([[1.], [-1.], [2.], [-2.]]),
        old_actions_log_prob=torch.full((n, 1), math.log(.5)), old_action_logits=torch.zeros(n, 2),
        dones=torch.zeros(n, 1, dtype=torch.bool), time_outs=torch.zeros(n, 1, dtype=torch.bool),
        timeout_values=torch.zeros(n, 1), low_level_steps=torch.full((n, 1), 10, dtype=torch.long),
    )


def test_act_copies_observations_values_distribution_and_returned_actions():
    algorithm = _algorithm()
    actor, critic = _observations()
    original_actor, original_critic = actor.clone(), critic.clone()
    actions = algorithm.act(actor, critic)
    original_actions = actions.clone()
    old_logits = algorithm.policy.action_logits.detach().clone()
    old_log_prob = algorithm.policy.get_actions_log_prob(actions).detach().clone()
    # 模拟环境原地刷新缓存，以及调用者改写返回动作。
    actor.fill_(777.)
    critic.fill_(888.)
    actions.copy_(1 - actions)
    algorithm.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
    storage = algorithm.storage
    torch.testing.assert_close(storage.observations[0], original_actor, rtol=0, atol=0)
    torch.testing.assert_close(storage.critic_observations[0], original_critic, rtol=0, atol=0)
    torch.testing.assert_close(storage.values[0], original_critic[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(storage.actions[0], original_actions, rtol=0, atol=0)
    torch.testing.assert_close(storage.action_logits[0], old_logits, rtol=0, atol=0)
    torch.testing.assert_close(storage.actions_log_prob[0, :, 0], old_log_prob, rtol=0, atol=0)
    assert storage.actions.dtype == torch.int64 and not storage.values.requires_grad


def test_timeout_uses_unsorted_compact_terminal_rows_and_never_estimates_failure_rows():
    algorithm = _algorithm(gamma=.5)
    actor, critic = _observations()
    algorithm.act(actor, critic)
    terminal = torch.tensor([[7., 0., 0., 0.], [float("nan")] * 4])
    rewards = torch.tensor([-4., .5, 1.])
    algorithm.process_env_step(rewards, torch.tensor([True, False, True]), {
        "time_outs": torch.tensor([False, False, True]),
        "terminal_env_ids": torch.tensor([2, 0]), "terminal_critic_observations": terminal,
        "low_level_steps": torch.tensor([4, 10, 10]),
        "observations": {"critic": torch.full((3, 4), 9000.)},
    })
    torch.testing.assert_close(algorithm.policy.evaluated_batches[-1], terminal[:1])
    torch.testing.assert_close(algorithm.storage.timeout_values[0, :, 0], torch.tensor([0., 0., 7.]))
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], rewards, rtol=0, atol=0)
    torch.testing.assert_close(algorithm.policy.last_dones, torch.tensor([True, False, True]))
    last_critic = torch.tensor([[100., 0., 0., 0.], [101., 0., 0., 0.], [102., 0., 0., 0.]])
    algorithm.compute_returns(last_critic)
    torch.testing.assert_close(algorithm.storage.returns[0, :, 0], torch.tensor([-4., 51., 4.5]))


@pytest.mark.parametrize("bad", [
    {},
    {"terminal_env_ids": torch.tensor([2., 0.]), "terminal_critic_observations": torch.zeros(2, 4)},
    {"terminal_env_ids": torch.tensor([2, 2]), "terminal_critic_observations": torch.zeros(2, 4)},
    {"terminal_env_ids": torch.tensor([3, 0]), "terminal_critic_observations": torch.zeros(2, 4)},
    {"terminal_env_ids": torch.tensor([2, 1]), "terminal_critic_observations": torch.zeros(2, 4)},
    {"terminal_env_ids": torch.tensor([0]), "terminal_critic_observations": torch.zeros(1, 4)},
    {"terminal_env_ids": torch.tensor([2, 0]), "terminal_critic_observations": torch.zeros(2, 3)},
    {"terminal_env_ids": torch.tensor([2, 0]), "terminal_critic_observations": torch.full((2, 4), float("nan"))},
])
def test_invalid_timeout_metadata_preserves_pending_transition_for_corrected_retry(bad):
    algorithm = _algorithm()
    actor, critic = _observations()
    actions = algorithm.act(actor, critic)
    done = torch.tensor([True, False, True])
    infos = {"time_outs": torch.tensor([False, False, True]), **bad}
    with pytest.raises(ValueError):
        algorithm.process_env_step(torch.ones(3), done, infos)
    assert algorithm.storage.step == 0 and algorithm._pending is not None
    infos.update(terminal_env_ids=torch.tensor([2, 0]), terminal_critic_observations=torch.ones(2, 4))
    algorithm.process_env_step(torch.ones(3), done, infos)
    assert algorithm.storage.step == 1 and algorithm._pending is None
    torch.testing.assert_close(algorithm.storage.actions[0], actions)


def test_categorical_kl_matches_hand_calculation_and_ignores_logit_offsets():
    old = torch.tensor([[.75, .25], [.5, .5]]).log()
    new = torch.tensor([[.25, .75], [.8, .2]]).log()
    expected = torch.tensor([.5 * math.log(3.), .5 * math.log(.5 / .8) + .5 * math.log(.5 / .2)])
    torch.testing.assert_close(categorical_kl(old, new), expected)
    torch.testing.assert_close(categorical_kl(old + 5., new - 7.), expected)
    torch.testing.assert_close(categorical_kl(old, old), torch.zeros(2), rtol=0, atol=0)
    with pytest.raises(ValueError, match="matching"):
        categorical_kl(torch.zeros(2, 1), torch.zeros(2, 1))


@pytest.mark.parametrize("value_clipping,expected_value_loss", [(True, 2.41), (False, 2.25)])
def test_ppo_clipping_entropy_and_value_loss_match_independent_analytic_values(value_clipping, expected_value_loss):
    algorithm = _algorithm(value_loss_coef=.5, entropy_coef=.1, use_clipped_value_loss=value_clipping)
    with torch.no_grad():
        algorithm.policy.actor.bias.copy_(torch.tensor([math.log(4.), 0.]))
    loss, metrics = algorithm._compute_loss(_loss_batch())
    # 新概率 .8/.2，旧概率 .5/.5；比值为1.6/1.6/.4/.4。
    # 裁剪后四项为1.2/-1.6/.8/-1.6，负均值为0.3。
    entropy = -(.8 * math.log(.8) + .2 * math.log(.2))
    assert metrics["surrogate_loss"].item() == pytest.approx(.3, abs=1e-6)
    assert metrics["value_loss"].item() == pytest.approx(expected_value_loss, abs=1e-6)
    assert metrics["entropy"].item() == pytest.approx(entropy)
    assert metrics["kl"].item() == pytest.approx(math.log(1.25), abs=1e-6)
    assert metrics["clip_fraction"].item() == 1.0
    assert loss.item() == pytest.approx(.3 + .5 * expected_value_loss - .1 * entropy, abs=1e-6)
    loss.backward()
    assert algorithm.policy.actor.bias.grad.abs().sum() > 0
    assert algorithm.policy.critic.weight.grad.abs().sum() > 0


def test_per_minibatch_advantage_normalization_handles_singletons_and_matches_analytic_loss():
    algorithm = _algorithm(normalize_advantage_per_mini_batch=True)
    with torch.no_grad():
        algorithm.policy.actor.bias.copy_(torch.tensor([math.log(4.), 0.]))
    batch = _loss_batch()
    batch.advantages.copy_(torch.tensor([[1.], [3.], [5.], [7.]]))
    _, metrics = algorithm._compute_loss(batch)
    assert metrics["surrogate_loss"].item() == pytest.approx(1.2 / math.sqrt(5.), abs=1e-6)
    # 原始 rollout 优势不能被 minibatch 归一化原地改写。
    assert batch.advantages[:, 0].tolist() == [1., 3., 5., 7.]
    one = SelectorMiniBatch(**{key: value[:1] for key, value in vars(batch).items()})
    loss, metrics = algorithm._compute_loss(one)
    assert torch.isfinite(loss) and metrics["surrogate_loss"].item() == 0.0


def test_global_and_minibatch_normalization_are_applied_at_only_one_stage():
    raw, global_normalized = _algorithm(gamma=.5, normalize_advantage_per_mini_batch=True), _algorithm(gamma=.5)
    for algorithm in (raw, global_normalized):
        actor, critic = _observations()
        algorithm.act(actor, critic)
        algorithm.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
        algorithm.compute_returns(critic + 3.)
    torch.testing.assert_close(raw.storage.advantages[0, :, 0], torch.tensor([2., 1.5, 1.]))
    torch.testing.assert_close(raw.storage.returns, global_normalized.storage.returns, rtol=0, atol=0)
    assert global_normalized.storage.advantages.mean().abs() < 1e-6
    torch.testing.assert_close(global_normalized.storage.advantages.std(unbiased=False), torch.tensor(1.))


def test_adaptive_learning_rate_uses_categorical_kl_and_respects_bounds():
    algorithm = _algorithm(schedule="adaptive", desired_kl=.01)
    algorithm._adapt_learning_rate(.001)
    assert algorithm.learning_rate == pytest.approx(4.5e-4)
    algorithm._adapt_learning_rate(.03)
    assert algorithm.learning_rate == pytest.approx(3e-4)
    algorithm._adapt_learning_rate(0.)
    assert algorithm.learning_rate == pytest.approx(3e-4)
    for _ in range(100):
        algorithm._adapt_learning_rate(.03)
    assert algorithm.learning_rate == 1e-5
    for _ in range(100):
        algorithm._adapt_learning_rate(.001)
    assert algorithm.learning_rate == .01
    assert all(group["lr"] == .01 for group in algorithm.optimizer.param_groups)
    fixed = _algorithm(schedule="fixed", desired_kl=None)
    fixed._adapt_learning_rate(100.)
    assert fixed.learning_rate == 3e-4


@pytest.mark.parametrize("cfg", [
    {"class_name": "PPO"}, {"kl_method": "gaussian"}, {"symmetry_cfg": {}}, {"rnd_cfg": {}},
    {"num_learning_epochs": 0}, {"num_mini_batches": 1.5}, {"learning_rate": 0.},
    {"learning_rate": float("nan")}, {"gamma": 0.}, {"lam": 1.1}, {"max_grad_norm": -1.},
    {"clip_param": 1.}, {"entropy_coef": -.1}, {"value_loss_coef": -.1},
    {"schedule": "unknown"}, {"schedule": "adaptive", "desired_kl": None}, {"desired_kl": 0.},
    {"use_clipped_value_loss": 1}, {"normalize_advantage_per_mini_batch": 1},
])
def test_invalid_algorithm_configuration_is_rejected(cfg):
    with pytest.raises(ValueError):
        DiscretePPO(TinyCategoricalPolicy(), **cfg)


@pytest.mark.parametrize("attribute,value", [("is_recurrent", True), ("action_distribution", "gaussian"),
                                             ("num_modes", 3), ("num_actions", 2)])
def test_incompatible_policy_is_rejected(attribute, value):
    policy = TinyCategoricalPolicy()
    setattr(policy, attribute, value)
    with pytest.raises(ValueError, match="feedforward categorical"):
        DiscretePPO(policy)


def test_rollout_lifecycle_train_mode_gradient_clipping_and_frozen_parameter_isolation():
    policy = TinyCategoricalPolicy()
    algorithm = DiscretePPO(policy, num_mini_batches=1, num_learning_epochs=2, max_grad_norm=.05)
    actor, critic = _observations()
    with pytest.raises(RuntimeError, match="init_storage"):
        algorithm.act(actor, critic)
    algorithm.init_storage(3, 1)
    with pytest.raises(RuntimeError, match="[Cc]omplete"):
        algorithm.compute_returns(critic)
    with pytest.raises(RuntimeError, match="returns"):
        algorithm.update()
    with pytest.raises(RuntimeError, match="act"):
        algorithm.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
    algorithm.act(actor, critic)
    with pytest.raises(RuntimeError, match="previous"):
        algorithm.act(actor, critic)
    with pytest.raises(RuntimeError, match="unconsumed"):
        algorithm.init_storage(3, 1)
    algorithm.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
    with pytest.raises(RuntimeError, match="full"):
        algorithm.act(actor, critic)
    algorithm.compute_returns(critic)
    algorithm.test_mode()
    assert not policy.training
    actor_before, critic_before = policy.actor.bias.detach().clone(), policy.critic.bias.detach().clone()
    result = algorithm.update()
    assert policy.training and result["num_updates"] == 2
    assert all(math.isfinite(value) for value in result.values())
    assert not torch.equal(policy.actor.bias, actor_before) and not torch.equal(policy.critic.bias, critic_before)
    norm = torch.stack([parameter.grad.square().sum() for parameter in policy.parameters()
                        if parameter.grad is not None]).sum().sqrt()
    assert norm.item() <= .050001
    assert policy.frozen_parameter.item() == 123. and policy.frozen_parameter.grad is None
    assert all(policy.frozen_parameter is not parameter for group in algorithm.optimizer.param_groups
               for parameter in group["params"])
    assert algorithm.storage.step == 0 and not algorithm.storage.returns_ready
    with pytest.raises(RuntimeError, match="returns"):
        algorithm.update()
    algorithm.init_storage(3, 2)
    assert algorithm.storage.num_transitions_per_env == 2


def test_indivisible_batch_and_nonfinite_update_fail_without_consuming_rollout():
    algorithm = DiscretePPO(TinyCategoricalPolicy(), num_mini_batches=4)
    with pytest.raises(ValueError, match="evenly"):
        algorithm.init_storage(3, 1)
    assert algorithm.storage is None
    algorithm = _algorithm()
    _finish_rollout(algorithm)
    with torch.no_grad():
        algorithm.policy.critic.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        algorithm.update()
    assert algorithm.storage.step == 1 and algorithm.storage.returns_ready


def test_actual_algorithm_configuration_and_multimodal_network_complete_collect_update_cycle():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "legged_lab/envs/g1/g1_selector_cfg.py").read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1SelectorAgentCfg")
    algorithm_cfg = next(node.value for node in agent.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "algorithm" for target in node.targets))
    cfg = {item.arg: ast.literal_eval(item.value) for item in algorithm_cfg.keywords}
    # 保留实际 PPO 超参数；用4环境x4步执行完整采样和5轮x4批更新。
    torch.manual_seed(42)
    policy = SelectorActorCritic()
    algorithm = DiscretePPO(policy, **cfg)
    algorithm.init_storage(4, 4)
    actor_before = policy.actor[-1].weight.detach().clone()
    critic_before = policy.critic[-1].weight.detach().clone()
    actor, critic = torch.randn(4, 3466), torch.randn(4, 3660)
    for index in range(4):
        actions = algorithm.act(actor, critic)
        assert actions.shape == (4, 1) and actions.dtype == torch.long
        done, timeout = torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
        steps, extras = torch.full((4,), 10, dtype=torch.long), {}
        if index == 1:
            done[0], steps[0] = True, 4
        if index == 2:
            done[1], timeout[1] = True, True
            extras.update(terminal_env_ids=torch.tensor([1]), terminal_critic_observations=critic[1:2].clone())
        extras.update(time_outs=timeout, low_level_steps=steps)
        # 与真实环境相同：step 后原地刷新 actor/critic 缓存。
        actor.normal_()
        critic.normal_()
        rewards = torch.tensor([-.2, .1, .3, .5])
        if index == 1:
            rewards[0] = -4.
        algorithm.process_env_step(rewards, done, extras)
    algorithm.compute_returns(critic)
    result = algorithm.update()
    assert result["num_updates"] == 20
    assert result["entropy"] > 0. and result["kl"] >= 0.
    assert all(math.isfinite(value) for value in result.values())
    assert not torch.equal(policy.actor[-1].weight, actor_before)
    assert not torch.equal(policy.critic[-1].weight, critic_before)
    assert algorithm.storage.step == 0
