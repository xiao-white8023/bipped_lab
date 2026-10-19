"""独立高层 runner：分类策略、离散 PPO 和选择器环境之间的采样循环。

不继承现有 runner，不创建 AMP、观测归一化或底层优化器。
迭代编号表示已完成的 PPO 更新次数；learn(n) 执行 n 次新增更新。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch

from rsl_rl.algorithms.discrete_ppo import DiscretePPO
from rsl_rl.modules.selector_actor_critic import SelectorActorCritic


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class SelectorOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if hasattr(train_cfg, "to_dict"):
            train_cfg = train_cfg.to_dict()
        if not isinstance(train_cfg, Mapping):
            raise TypeError("Selector training configuration must be a mapping or expose to_dict().")
        self.cfg = copy.deepcopy(dict(train_cfg))
        self.policy_cfg = copy.deepcopy(self.cfg["policy"])
        self.alg_cfg = copy.deepcopy(self.cfg["algorithm"])
        if self.cfg.get("runner_class_name", "SelectorOnPolicyRunner") != "SelectorOnPolicyRunner":
            raise ValueError("Use the independent SelectorOnPolicyRunner.")
        if self.policy_cfg.get("class_name") != "SelectorActorCritic" or self.alg_cfg.get("class_name") != "DiscretePPO":
            raise ValueError("Selector runner requires SelectorActorCritic and DiscretePPO.")
        if self.cfg.get("empirical_normalization", False) or self.cfg.get("clip_actions") is not None:
            raise ValueError("Do not normalize the full multimodal observation or clip categorical mode IDs.")
        self.num_steps_per_env = _positive_int(self.cfg["num_steps_per_env"], "num_steps_per_env")
        self.save_interval = _positive_int(self.cfg.get("save_interval", 100), "save_interval")
        self.num_envs = _positive_int(env.num_envs, "num_envs")
        if env.num_actions != 1:
            raise ValueError("The selector environment must accept one VP=0 / LP=1 action.")
        self.step_dt, self.low_level_dt = float(env.step_dt), float(env.low_level_dt)
        if (not math.isfinite(self.step_dt) or not math.isfinite(self.low_level_dt)
                or min(self.step_dt, self.low_level_dt) <= 0):
            raise ValueError("Selector control periods must be finite and positive.")
        ratio = self.step_dt / self.low_level_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("A high-level step must contain an integer number of low-level steps.")
        self.low_level_steps_per_decision = round(ratio)
        self.device = torch.empty(0, device=device).device
        self.env_device = torch.empty(0, device=env.device).device
        self.env = env
        seed = self.cfg.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Selector seed must be a nonnegative integer below 2**63.")
        random.seed(seed)
        torch.manual_seed(seed)

        # ---------------------------------------------------------
        # 1. 读取环境已经准备好的缓存，不 reset，不改写 episode 或传感器时钟。
        # ---------------------------------------------------------
        actor, critic = self._observations(*env.get_observations())
        self.num_actor_obs, self.num_critic_obs = actor.shape[1], critic.shape[1]
        self.policy = SelectorActorCritic(
            num_actor_obs=self.num_actor_obs, num_critic_obs=self.num_critic_obs,
            num_actions=env.num_actions, **self.policy_cfg,
        )
        self.alg = DiscretePPO(self.policy, device=self.device, **self.alg_cfg)
        self.alg.init_storage(
            self.num_envs, self.num_steps_per_env,
            low_level_steps_per_decision=self.low_level_steps_per_decision,
        )

        # ---------------------------------------------------------
        # 2. 独立统计：奖励原值、高层步数，以及终止前真正有效的底层时长。
        # ---------------------------------------------------------
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_low_steps = torch.zeros_like(self._episode_lengths)
        self._recent_episodes = deque(maxlen=100)
        self.current_learning_iteration = 0
        self.total_timesteps = 0
        self.total_vector_steps = 0
        self.total_low_level_steps = 0
        self.total_time = 0.0
        self.last_log = {}
        self.stopped = False
        self.log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
        self.logger = self.cfg.get("logger", "tensorboard")
        if self.logger not in ("tensorboard", "jsonl"):
            raise ValueError("Selector runner supports local tensorboard or jsonl logging.")
        self.writer = None
        self.last_checkpoint = None
        self.signature = {
            "actor_obs_dim": self.num_actor_obs, "critic_obs_dim": self.num_critic_obs,
            "num_actions": 1, "mode_names": ["vp", "lp"],
            "high_level_dt": self.step_dt, "low_level_dt": self.low_level_dt,
        }
        metadata = getattr(env, "frozen_policy_metadata", {})
        self.signature["frozen_policy"] = {key: metadata.get(key) for key in (
            "source_checkpoint_sha256", "reference_policy_sha256",
        )}

    def _observations(self, actor, extras):
        if not isinstance(extras, Mapping) or not isinstance(extras.get("observations"), Mapping):
            raise ValueError("Selector extras must provide observations['critic'].")
        critic = extras["observations"].get("critic")
        for name, value in (("actor", actor), ("critic", critic)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[0] != self.num_envs or not value.is_floating_point()):
                raise ValueError(f"Selector {name} observations must have shape (num_envs,features).")
        return actor.to(self.device), critic.to(self.device)

    def _synchronize(self):
        for device in {self.device, self.env_device}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def _init_writer(self):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.logger == "tensorboard" and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)

    @torch.no_grad()
    def _record_step(self, rewards, dones, infos, actions, counters, env_logs):
        rewards = rewards.to(self.device).reshape(self.num_envs)
        dones = dones.to(self.device).reshape(self.num_envs)
        time_outs = infos["time_outs"].to(self.device).reshape(self.num_envs)
        steps = infos["low_level_steps"].to(self.device).reshape(self.num_envs)
        self._episode_returns += rewards
        self._episode_lengths += 1
        self._episode_low_steps += steps
        ended = dones.nonzero().flatten()
        if ended.numel():
            rows = zip(self._episode_returns[ended].tolist(), self._episode_lengths[ended].tolist(),
                       self._episode_low_steps[ended].tolist())
            self._recent_episodes.extend(rows)
            self._episode_returns[ended] = 0.0
            self._episode_lengths[ended] = 0
            self._episode_low_steps[ended] = 0
        low_steps = int(steps.sum())
        counters["reward"] += rewards.sum().item()
        counters["lp_actions"] += (actions == 1).sum().item()
        counters["episodes"] += dones.sum().item()
        counters["failures"] += (dones & ~time_outs).sum().item()
        counters["timeouts"] += time_outs.sum().item()
        counters["low_steps"] += low_steps
        self.total_timesteps += self.num_envs
        self.total_vector_steps += 1
        self.total_low_level_steps += low_steps
        for name, value in infos.get("log", {}).items():
            if isinstance(value, torch.Tensor):
                if not value.numel():
                    continue
                value = value.detach().float().mean().item()
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            total, count = env_logs.get(name, (0.0, 0))
            env_logs[name] = (total + float(value), count + 1)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, *, should_continue=None):
        """完成新增 PPO 更新；停止回调只在 rollout 边界检查，避免保存半次更新。"""
        _positive_int(num_learning_iterations, "num_learning_iterations")
        if init_at_random_ep_len:
            raise ValueError("Do not randomize only episode_length_buf; selector low-level clocks must remain consistent.")
        self._require_rollout_boundary()
        self._init_writer()
        actor, critic = self._observations(*self.env.get_observations())
        self.train_mode()
        self.stopped = False
        last_saved_iteration = None
        for _ in range(num_learning_iterations):
            if should_continue is not None and not should_continue():
                self.stopped = True
                break
            counters = dict.fromkeys(("reward", "lp_actions", "episodes", "failures", "timeouts", "low_steps"), 0.0)
            env_logs = {}
            self._synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(actor, critic)
                    actor_next, rewards, dones, infos = self.env.step(actions.to(self.env_device))
                    if "time_outs" not in infos or "low_level_steps" not in infos:
                        raise ValueError("Selector environment must return time_outs and low_level_steps.")
                    self.alg.process_env_step(rewards, dones, infos)
                    self._record_step(rewards, dones, infos, actions, counters, env_logs)
                    actor, critic = self._observations(actor_next, infos)
                self.alg.compute_returns(critic)
            self._synchronize()
            collected = time.perf_counter()
            losses = self.alg.update()
            self._synchronize()
            finished = time.perf_counter()
            self.current_learning_iteration += 1
            self.total_time += finished - start
            self.last_log = self.log(counters, env_logs, losses, collected - start, finished - collected)
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
                last_saved_iteration = self.current_learning_iteration
        if self.log_dir is not None and last_saved_iteration != self.current_learning_iteration:
            self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
        if self.writer is not None:
            self.writer.flush()
        return dict(self.last_log)

    def log(self, counters, env_logs, losses, collection_time, learning_time):
        samples = self.num_envs * self.num_steps_per_env
        elapsed = max(collection_time + learning_time, 1e-8)
        stats = {
            "iteration": self.current_learning_iteration,
            "Train/total_timesteps": self.total_timesteps,
            "Train/total_vector_steps": self.total_vector_steps,
            "Train/total_low_level_steps": self.total_low_level_steps,
            "Train/total_simulated_seconds": self.total_vector_steps * self.step_dt,
            "Train/mean_transition_reward": counters["reward"] / samples,
            "Train/completed_episodes": counters["episodes"], "Train/true_failures": counters["failures"],
            "Train/time_outs": counters["timeouts"],
            "Train/failures_per_effective_minute": 60.0 * counters["failures"] / (counters["low_steps"] * self.low_level_dt),
            "Policy/lp_fraction": counters["lp_actions"] / samples,
            "Policy/vp_fraction": 1.0 - counters["lp_actions"] / samples,
            "Policy/entropy": losses["entropy"], "Policy/categorical_kl": losses["kl"],
            "Policy/clip_fraction": losses["clip_fraction"], "Policy/grad_norm": losses["grad_norm"],
            "Loss/value_loss": losses["value_loss"], "Loss/surrogate_loss": losses["surrogate_loss"],
            "Loss/learning_rate": losses["learning_rate"],
            "Perf/high_level_samples_per_second": samples / elapsed,
            "Perf/effective_low_level_steps_per_second": counters["low_steps"] / elapsed,
            "Perf/collection_seconds": collection_time, "Perf/learning_seconds": learning_time,
            "Perf/total_seconds": self.total_time,
        }
        if self._recent_episodes:
            size = len(self._recent_episodes)
            stats["Train/mean_reward"] = sum(row[0] for row in self._recent_episodes) / size
            stats["Train/mean_episode_length"] = sum(row[1] for row in self._recent_episodes) / size
            stats["Train/mean_episode_seconds"] = sum(row[2] for row in self._recent_episodes) * self.low_level_dt / size
        # 环境日志按其出现次数取均值；episode 回报和失败总数由上面的逐环境统计给出。
        stats.update({name: total / count for name, (total, count) in env_logs.items() if name not in stats})
        if self.log_dir is not None:
            with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(stats, ensure_ascii=False, allow_nan=False) + "\n")
        if self.writer is not None:
            for name, value in stats.items():
                if name != "iteration":
                    self.writer.add_scalar(name, value, self.current_learning_iteration)
        print(f"[Selector PPO] iteration={self.current_learning_iteration} "
              f"reward={stats['Train/mean_transition_reward']:.4f} "
              f"failures={int(counters['failures'])} timeouts={int(counters['timeouts'])} "
              f"value_loss={losses['value_loss']:.4f} kl={losses['kl']:.5f}", flush=True)
        return stats

    def _require_rollout_boundary(self):
        if self.alg._pending is not None or self.alg.storage.step != 0:
            raise RuntimeError("Checkpoint and learn operations require a consumed rollout boundary.")

    def save(self, path, infos=None):
        self._require_rollout_boundary()
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "format": "g1_selector_discrete_ppo", "version": 1,
            "model_state_dict": self.policy.state_dict(), "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "learning_rate": self.alg.learning_rate, "iter": self.current_learning_iteration,
            "total_timesteps": self.total_timesteps, "total_vector_steps": self.total_vector_steps,
            "total_low_level_steps": self.total_low_level_steps, "total_time": self.total_time,
            "train_cfg": self.cfg, "signature": self.signature,
            "rng_state": {"torch": torch.get_rng_state(), "python": random.getstate(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            "infos": infos,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self.last_checkpoint = str(path)

    def load(self, path, load_optimizer=True, *, restore_rng=True):
        """恢复训练状态；仿真、故障随机流和半途 episode 不在 checkpoint 中。"""
        self._require_rollout_boundary()
        loaded = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=True)
        if loaded.get("format") != "g1_selector_discrete_ppo" or loaded.get("version") != 1:
            raise ValueError("Not an independent selector PPO checkpoint.")
        if loaded.get("signature") != self.signature:
            raise ValueError("Checkpoint observation, timing, mode or frozen-policy contract differs from this environment.")
        saved_policy_cfg = loaded.get("train_cfg", {}).get("policy")
        if saved_policy_cfg != self.policy_cfg:
            raise ValueError("Checkpoint selector network configuration differs from the current policy.")
        for name in ("iter", "total_timesteps", "total_vector_steps", "total_low_level_steps"):
            if type(loaded.get(name)) is not int or loaded[name] < 0:
                raise ValueError(f"Invalid checkpoint counter: {name}.")
        if not math.isfinite(loaded["total_time"]) or loaded["total_time"] < 0:
            raise ValueError("Invalid checkpoint total_time.")
        if not math.isfinite(loaded["learning_rate"]) or loaded["learning_rate"] <= 0:
            raise ValueError("Invalid checkpoint learning_rate.")
        self.policy.load_state_dict(loaded["model_state_dict"], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            self.alg.learning_rate = loaded["learning_rate"]
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
        self.current_learning_iteration = loaded["iter"]
        for name in ("total_timesteps", "total_vector_steps", "total_low_level_steps", "total_time"):
            setattr(self, name, loaded[name])
        # 环境未恢复到保存时的物理状态，因此不恢复未完成 episode 的统计。
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._episode_low_steps.zero_()
        self._recent_episodes.clear()
        self.last_log = {}
        if restore_rng:
            states = loaded["rng_state"]
            torch.set_rng_state(states["torch"].cpu())
            random.setstate(states["python"])
            if torch.cuda.is_available():
                for index, state in enumerate(states["cuda"][:torch.cuda.device_count()]):
                    torch.cuda.set_rng_state(state.cpu(), index)
        self.last_checkpoint = str(Path(path).expanduser().resolve())
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        """返回独立的 argmax 推理副本，不移动训练网络或更改其训练模式。"""
        # 新副本的随机初始化随后会被权重覆盖，不能消耗训练采样的随机流。
        with torch.random.fork_rng(devices=[]):
            model = SelectorActorCritic(
                num_actor_obs=self.num_actor_obs, num_critic_obs=self.num_critic_obs,
                num_actions=1, **self.policy_cfg,
            )
        model = model.to(self.device if device is None else device)
        model.load_state_dict(self.policy.state_dict(), strict=True)
        model.eval()
        model.requires_grad_(False)
        return model.act_inference

    def train_mode(self):
        self.alg.train_mode()

    def eval_mode(self):
        self.alg.test_mode()

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


__all__ = ["SelectorOnPolicyRunner"]
