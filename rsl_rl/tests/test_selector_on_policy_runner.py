"""独立 runner 的环境协议、采样日志、checkpoint 和恢复更新检查，无物理仿真。"""

import ast
import copy
import json
from pathlib import Path
import random
import sys
from types import ModuleType

import pytest
import torch

from rsl_rl.runners.selector_on_policy_runner import SelectorOnPolicyRunner


class ProtocolEnv:
    """复现高层接口：原地刷新观测、失败提前结束、纯超时返回紧凑末端快照。"""

    device, num_actions, step_dt, low_level_dt = "cpu", 1, .2, .02

    def __init__(self, n=3, actor_dim=24, critic_dim=27):
        self.num_envs = n
        self.actor = torch.arange(n * actor_dim).float().reshape(n, actor_dim) / 100
        self.critic = torch.cat((self.actor.clone(), torch.ones(n, critic_dim - actor_dim)), dim=-1)
        self.mode = torch.zeros(n, 1, dtype=torch.long)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.low_level_episode_length_buf = torch.zeros_like(self.episode_length_buf)
        self.steps, self.resets = 0, 0
        self.actions, self.before_steps = [], []
        self.frozen_policy = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        self.frozen_policy_metadata = {"source_checkpoint_sha256": "checkpoint-hash", "reference_policy_sha256": "reference-hash"}

    def get_observations(self):
        return self.actor, {"observations": {"critic": self.critic}}

    def reset(self):
        self.resets += 1
        return self.get_observations()

    def step(self, actions):
        assert actions.dtype == torch.long and actions.shape == (self.num_envs, 1)
        assert ((actions == 0) | (actions == 1)).all()
        self.actions.append(actions.clone())
        self.before_steps.append((self.actor.clone(), self.critic.clone()))
        switched = (actions != self.mode).sum()
        self.mode.copy_(actions)
        self.steps += 1
        rewards = torch.arange(1, self.num_envs + 1).float()
        dones, timeouts = torch.zeros(self.num_envs, dtype=torch.bool), torch.zeros(self.num_envs, dtype=torch.bool)
        low_steps = torch.full((self.num_envs,), 10, dtype=torch.long)
        if self.steps % 3 == 2:
            dones[0], low_steps[0], rewards[0] = True, 4, -4.
        if self.steps % 3 == 0:
            dones[1], timeouts[1] = True, True
        self.actor.add_(.1)
        self.critic[:, :self.actor.shape[1]].copy_(self.actor)
        self.critic[:, self.actor.shape[1]:].add_(1.)
        self.episode_length_buf += 1
        self.low_level_episode_length_buf += low_steps
        ids = dones.nonzero().flatten()
        terminal_actor, terminal_critic = self.actor[ids].clone(), self.critic[ids].clone()
        self.actor[ids], self.critic[ids] = 0., 0.
        self.episode_length_buf[ids] = self.low_level_episode_length_buf[ids] = 0
        self.mode[ids] = 0
        logs = {"Selector/switches": switched, "Fault/observed_lp_only_fraction": torch.tensor(.25)}
        if self.steps % 3 == 0:
            logs["Sparse/only_at_third_step"] = torch.tensor(7.)
        return self.actor, rewards, dones, {
            "observations": {"critic": self.critic}, "time_outs": timeouts, "low_level_steps": low_steps,
            "terminal_env_ids": ids, "terminal_actor_observations": terminal_actor,
            "terminal_critic_observations": terminal_critic, "log": logs,
        }


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _config():
    return {
        "seed": 42, "num_steps_per_env": 3, "save_interval": 100, "logger": "jsonl",
        "runner_class_name": "SelectorOnPolicyRunner", "empirical_normalization": False, "clip_actions": None,
        "policy": {
            "class_name": "SelectorActorCritic", "proprio_input_dim": 4, "height_input_dim": 2,
            "state_input_dim": 2, "actor_obs_dim": 24, "critic_obs_dim": 27, "critic_privileged_dim": 3,
            "proprio_hidden_dims": [8, 4], "height_hidden_dims": [4], "state_hidden_dims": [4],
            "actor_hidden_dims": [8], "critic_hidden_dims": [8],
            "depth_encoder": {"input_dim": [2, 2], "input_channels": 4, "output_channels": [2],
                              "kernel_size": [1], "stride": [1], "padding": [0], "mlp_output_dim": 4},
        },
        "algorithm": {"class_name": "DiscretePPO", "num_learning_epochs": 2, "num_mini_batches": 1,
                      "schedule": "adaptive", "desired_kl": .01},
    }


def _runner(tmp_path=None, cfg=None, env=None):
    return SelectorOnPolicyRunner(ProtocolEnv() if env is None else env,
                                  _config() if cfg is None else cfg, log_dir=tmp_path)


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_constructor_preserves_config_and_environment_counters_without_reset():
    cfg, env = _config(), ProtocolEnv()
    original = copy.deepcopy(cfg)
    env.episode_length_buf.fill_(5)
    env.low_level_episode_length_buf.fill_(50)
    runner = _runner(cfg=cfg, env=env)
    assert cfg == original and env.steps == env.resets == 0
    assert env.episode_length_buf.tolist() == [5, 5, 5]
    assert env.low_level_episode_length_buf.tolist() == [50, 50, 50]
    assert runner.alg.storage.actor_obs_dim == 24 and runner.alg.storage.critic_obs_dim == 27
    frozen_ids = {id(parameter) for parameter in env.frozen_policy.parameters()}
    trainable_ids = {id(parameter) for group in runner.alg.optimizer.param_groups for parameter in group["params"]}
    assert not frozen_ids.intersection(trainable_ids)


def test_rollout_statistics_preserve_dt_reward_termination_and_observation_snapshots(tmp_path):
    runner = _runner(tmp_path)
    frozen_before = copy.deepcopy(runner.env.frozen_policy.state_dict())
    stats = runner.learn(1)
    assert runner.current_learning_iteration == 1
    assert runner.total_timesteps == 9 and runner.total_vector_steps == 3 and runner.total_low_level_steps == 84
    assert stats["Train/mean_transition_reward"] == pytest.approx(13 / 9)
    assert stats["Train/mean_reward"] == 1.5 and stats["Train/mean_episode_length"] == 2.5
    assert stats["Train/mean_episode_seconds"] == pytest.approx(.44)
    assert stats["Train/total_simulated_seconds"] == pytest.approx(.6)
    assert stats["Train/true_failures"] == stats["Train/time_outs"] == 1
    assert stats["Train/failures_per_effective_minute"] == pytest.approx(60 / (84 * .02))
    assert stats["Fault/observed_lp_only_fraction"] == .25 and stats["Sparse/only_at_third_step"] == 7.
    assert stats["Policy/lp_fraction"] + stats["Policy/vp_fraction"] == pytest.approx(1.)
    assert runner.env.resets == 0 and runner.alg.storage.step == 0
    for index, (actor, critic) in enumerate(runner.env.before_steps):
        torch.testing.assert_close(runner.alg.storage.observations[index], actor, rtol=0, atol=0)
        torch.testing.assert_close(runner.alg.storage.critic_observations[index], critic, rtol=0, atol=0)
    assert runner.alg.storage.rewards[1, 0, 0].item() == -4.
    assert runner.alg.storage.time_outs[2, 1, 0]
    assert not runner.alg.storage.time_outs[1, 0, 0]
    _assert_nested_equal(frozen_before, runner.env.frozen_policy.state_dict())
    assert all(parameter.grad is None for parameter in runner.env.frozen_policy.parameters())
    logged = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert logged == [stats] and (tmp_path / "model_1.pt").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_multiple_learn_calls_and_periodic_final_saves_have_no_repeated_iteration(tmp_path):
    cfg = _config()
    cfg["save_interval"] = 2
    runner = _runner(tmp_path, cfg=cfg)
    runner.learn(2)
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["model_2.pt"]
    runner.learn(1)
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["model_2.pt", "model_3.pt"]
    iterations = [json.loads(line)["iteration"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert iterations == [1, 2, 3] and runner.total_timesteps == 27


def test_stop_callback_finishes_full_rollout_and_saves_latest_completed_update(tmp_path):
    runner = _runner(tmp_path)
    decisions = iter([True, False])
    result = runner.learn(5, should_continue=lambda: next(decisions))
    assert runner.stopped and result["iteration"] == 1 and runner.env.steps == 3
    assert runner.alg.storage.step == 0 and (tmp_path / "model_1.pt").is_file()
    with pytest.raises(ValueError, match="clocks"):
        runner.learn(1, init_at_random_ep_len=True)
    assert runner.env.steps == 3


def test_checkpoint_restores_weights_adam_learning_rate_rng_and_counters_then_continues(tmp_path):
    first = _runner()
    first.learn(1)
    first.alg.learning_rate = 1.2e-4
    for group in first.alg.optimizer.param_groups:
        group["lr"] = first.alg.learning_rate
    checkpoint = tmp_path / "saved.pt"
    first.save(checkpoint, infos={"purpose": "resume-test"})
    expected_torch, expected_python = torch.rand(5), [random.random() for _ in range(3)]
    second = _runner(tmp_path / "continued", env=ProtocolEnv(n=4))
    assert second.load(checkpoint) == {"purpose": "resume-test"}
    torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)
    assert [random.random() for _ in range(3)] == expected_python
    _assert_nested_equal(first.policy.state_dict(), second.policy.state_dict())
    _assert_nested_equal(first.alg.optimizer.state_dict(), second.alg.optimizer.state_dict())
    assert second.alg.learning_rate == 1.2e-4 and second.current_learning_iteration == 1
    assert second.total_timesteps == 9 and second.total_low_level_steps == 84
    assert second.env.steps == 0 and second._episode_returns.sum() == 0 and not second._recent_episodes
    second.learn(1)
    assert second.current_learning_iteration == 2 and second.total_timesteps == 21
    assert second.total_vector_steps == 6 and (tmp_path / "continued/model_2.pt").is_file()


def test_weights_load_without_optimizer_or_rng_restore_preserves_the_new_optimizer(tmp_path):
    source = _runner()
    source.learn(1)
    source.alg.learning_rate = 9e-4
    path = tmp_path / "source.pt"
    source.save(path)
    target = _runner()
    before_rng, before_lr = torch.get_rng_state().clone(), target.alg.learning_rate
    target.load(path, load_optimizer=False, restore_rng=False)
    assert target.alg.learning_rate == before_lr and not target.alg.optimizer.state
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    _assert_nested_equal(source.policy.state_dict(), target.policy.state_dict())


@pytest.mark.parametrize("field,bad", [
    ("format", "old_amp"), ("version", 2), ("iter", -1), ("total_low_level_steps", 1.5),
    ("total_time", float("nan")), ("learning_rate", 0.),
])
def test_invalid_checkpoint_metadata_is_rejected_before_loading_weights(tmp_path, field, bad):
    runner = _runner()
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    data = torch.load(path, weights_only=True)
    data[field] = bad
    torch.save(data, path)
    before = copy.deepcopy(runner.policy.state_dict())
    with pytest.raises(ValueError):
        runner.load(path)
    _assert_nested_equal(runner.policy.state_dict(), before)


@pytest.mark.parametrize("contract", ["frozen_policy", "high_level_dt", "network"])
def test_incompatible_checkpoint_contract_is_rejected(tmp_path, contract):
    runner = _runner()
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    data = torch.load(path, weights_only=True)
    if contract == "network":
        data["train_cfg"]["policy"]["actor_hidden_dims"] = [64]
    elif contract == "frozen_policy":
        data["signature"]["frozen_policy"]["source_checkpoint_sha256"] = "another-low-level-policy"
    else:
        data["signature"]["high_level_dt"] = .1
    torch.save(data, path)
    with pytest.raises(ValueError, match="contract|configuration"):
        runner.load(path)


def test_checkpoint_and_learn_reject_unconsumed_rollout(tmp_path):
    runner = _runner()
    path = tmp_path / "initial.pt"
    runner.save(path)
    actor, extras = runner.env.get_observations()
    runner.alg.act(actor, extras["observations"]["critic"])
    for operation in (lambda: runner.save(path), lambda: runner.load(path), lambda: runner.learn(1)):
        with pytest.raises(RuntimeError, match="boundary"):
            operation()
    runner.alg.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {})
    with pytest.raises(RuntimeError, match="boundary"):
        runner.save(path)


def test_inference_copy_is_argmax_and_does_not_change_training_weights_mode_or_rng():
    runner = _runner()
    with torch.no_grad():
        runner.policy.actor[-1].weight.zero_()
        runner.policy.actor[-1].bias.copy_(torch.tensor([-2., 2.]))
    before_rng = torch.get_rng_state().clone()
    inference = runner.get_inference_policy()
    chosen = inference(runner.env.actor)
    assert chosen.shape == (3, 1) and chosen.dtype == torch.long and (chosen == 1).all()
    assert inference.__self__ is not runner.policy and not inference.__self__.training
    assert runner.policy.training and all(not parameter.requires_grad for parameter in inference.__self__.parameters())
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    with torch.no_grad():
        runner.policy.actor[-1].bias.copy_(torch.tensor([2., -2.]))
    assert (inference(runner.env.actor) == 1).all()
    runner.eval_mode()
    assert not runner.policy.training
    runner.train_mode()
    assert runner.policy.training


def test_tensorboard_is_lazy_local_and_closed_cleanly(tmp_path, monkeypatch):
    writers = []

    class Writer:
        def __init__(self, **kwargs):
            self.events, self.flushes, self.closed = [], 0, False
            writers.append(self)

        def add_scalar(self, *args):
            self.events.append(args)

        def flush(self):
            self.flushes += 1

        def close(self):
            self.closed = True

    module = ModuleType("torch.utils.tensorboard")
    module.SummaryWriter = Writer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = _config()
    cfg["logger"] = "tensorboard"
    runner = _runner(tmp_path, cfg=cfg)
    assert not writers
    runner.learn(1)
    runner.learn(1)
    assert len(writers) == 1
    assert {step for name, value, step in writers[0].events} == {1, 2}
    assert any(name == "Policy/entropy" for name, value, step in writers[0].events)
    runner.close()
    assert writers[0].closed and writers[0].flushes >= 3 and runner.writer is None


@pytest.mark.parametrize("change", [
    {"runner_class_name": "AmpOnPolicyRunner"}, {"empirical_normalization": True}, {"clip_actions": 1.},
    {"num_steps_per_env": 0}, {"save_interval": 0}, {"seed": -1}, {"logger": "wandb"},
])
def test_incompatible_runner_settings_are_rejected(change):
    cfg = _config()
    cfg.update(change)
    with pytest.raises(ValueError):
        _runner(cfg=cfg)


def test_environment_interface_and_fractional_control_ratio_are_rejected():
    env = ProtocolEnv()
    env.step_dt = .21
    with pytest.raises(ValueError, match="integer number"):
        _runner(env=env)
    env = ProtocolEnv()
    env.num_actions = 23
    with pytest.raises(ValueError, match="one VP"):
        _runner(env=env)
    env = ProtocolEnv()
    env.get_observations = lambda: (env.actor, {"observations": {}})
    with pytest.raises(ValueError, match="critic"):
        _runner(env=env)


def test_native_policy_and_ppo_configuration_train_together_through_runner():
    path = Path(__file__).resolve().parents[2] / "legged_lab/envs/g1/g1_selector_cfg.py"
    tree = ast.parse(path.read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1SelectorAgentCfg")

    def literals(node):
        if isinstance(node, ast.Call):
            return {item.arg: literals(item.value) for item in node.keywords}
        return ast.literal_eval(node)

    cfg = _config()
    for name in ("policy", "algorithm"):
        value = next(node.value for node in agent.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        cfg[name] = literals(value)
    cfg["num_steps_per_env"] = 2
    runner = _runner(cfg=cfg, env=ProtocolEnv(4, 3466, 3660))
    before = runner.policy.actor[-1].weight.detach().clone()
    stats = runner.learn(1)
    assert runner.total_timesteps == 8 and runner.total_low_level_steps == 74
    assert stats["Policy/entropy"] > 0 and stats["Policy/categorical_kl"] >= 0
    assert not torch.equal(before, runner.policy.actor[-1].weight)
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
