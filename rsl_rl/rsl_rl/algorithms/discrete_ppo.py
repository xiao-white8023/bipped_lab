"""独立的高层分类 PPO；VP=0、LP=1，不继承连续动作 PPO 或 AMP。

采样顺序为 act() -> env.step() -> process_env_step()。填满 rollout 后调用
compute_returns()、update()。环境奖励已经积分，不在算法中再次乘 dt。
通过本文件的子模块路径导入，无需修改现有算法导出或任务注册。
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F

from rsl_rl.storage.selector_rollout_storage import SelectorRolloutStorage, SelectorTransition


def categorical_kl(old_logits, new_logits):
    """返回逐样本 KL(old || new)，与 logits 的共同平移无关。"""
    if (old_logits.ndim != 2 or old_logits.shape[-1] != 2
            or old_logits.shape != new_logits.shape):
        raise ValueError("Categorical KL requires matching (N,2) logits.")
    old_log_prob = F.log_softmax(old_logits, dim=-1)
    new_log_prob = F.log_softmax(new_logits, dim=-1)
    return (old_log_prob.exp() * (old_log_prob - new_log_prob)).sum(dim=-1).clamp_min(0.0)


class DiscretePPO:
    def __init__(
        self,
        policy,
        *,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
        device="cpu",
        class_name="DiscretePPO",
        kl_method="categorical",
    ):
        if class_name != "DiscretePPO" or kl_method != "categorical":
            raise ValueError("The selector requires DiscretePPO with categorical KL.")
        if (not isinstance(policy, torch.nn.Module) or getattr(policy, "is_recurrent", True)
                or getattr(policy, "action_distribution", None) != "categorical"
                or getattr(policy, "num_modes", None) != 2 or getattr(policy, "num_actions", None) != 1):
            raise ValueError("DiscretePPO requires a feedforward categorical VP/LP actor-critic.")
        for name in ("num_actor_obs", "num_critic_obs"):
            value = getattr(policy, name, None)
            if type(value) is not int or value < 1:
                raise ValueError(f"The selector policy must define a positive {name}.")
        if symmetry_cfg is not None or rnd_cfg is not None:
            raise ValueError("Selector PPO does not configure symmetry augmentation or RND rewards.")
        for name, value in (("num_learning_epochs", num_learning_epochs), ("num_mini_batches", num_mini_batches)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        scalars = {
            "value_loss_coef": value_loss_coef, "clip_param": clip_param, "entropy_coef": entropy_coef,
            "learning_rate": learning_rate, "gamma": gamma, "lam": lam, "max_grad_norm": max_grad_norm,
        }
        for name, value in scalars.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0.0 < clip_param < 1.0 or learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("clip_param must be in (0,1); learning_rate and max_grad_norm must be positive.")
        if value_loss_coef < 0 or entropy_coef < 0 or not 0.0 < gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("Loss coefficients must be nonnegative; gamma and lam must be in their probability ranges.")
        if type(use_clipped_value_loss) is not bool or type(normalize_advantage_per_mini_batch) is not bool:
            raise ValueError("Value clipping and advantage normalization flags must be boolean.")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be 'fixed' or 'adaptive'.")
        if desired_kl is not None and (isinstance(desired_kl, bool) or not isinstance(desired_kl, (int, float))
                                       or not math.isfinite(desired_kl) or desired_kl <= 0):
            raise ValueError("desired_kl must be positive and finite, or None with a fixed schedule.")
        if schedule == "adaptive" and desired_kl is None:
            raise ValueError("Adaptive learning rates require desired_kl.")

        self.device = torch.empty(0, device=device).device
        self.policy = policy.to(self.device)
        self._parameters = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not self._parameters:
            raise ValueError("Selector PPO requires trainable high-level parameters.")
        self.optimizer = torch.optim.Adam(self._parameters, lr=learning_rate)
        self.learning_rate = float(learning_rate)
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.use_clipped_value_loss, self.clip_param = use_clipped_value_loss, clip_param
        self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
        self.schedule, self.desired_kl = schedule, desired_kl
        self.gamma, self.lam, self.max_grad_norm = gamma, lam, max_grad_norm
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.storage: SelectorRolloutStorage | None = None
        self._pending = None

    def init_storage(self, num_envs, num_transitions_per_env, *, low_level_steps_per_decision=10):
        if self._pending is not None or (self.storage is not None and self.storage.step):
            raise RuntimeError("Cannot replace an unconsumed selector rollout.")
        storage = SelectorRolloutStorage(
            num_envs, num_transitions_per_env, self.policy.num_actor_obs, self.policy.num_critic_obs,
            num_modes=self.policy.num_modes, low_level_steps_per_decision=low_level_steps_per_decision,
            device=self.device,
        )
        if num_envs * num_transitions_per_env % self.num_mini_batches:
            raise ValueError("The rollout batch must divide evenly into num_mini_batches.")
        self.storage = storage

    def _storage(self):
        if self.storage is None:
            raise RuntimeError("Call init_storage() before collecting selector transitions.")
        return self.storage

    def _float(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point()):
            raise ValueError(f"{name} must be floating point with shape {shape}.")
        value = value.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
        return value

    @torch.no_grad()
    def act(self, observations, critic_observations):
        storage = self._storage()
        if self._pending is not None:
            raise RuntimeError("Process the previous environment step before sampling another action.")
        if storage.step >= storage.num_transitions_per_env:
            raise RuntimeError("The selector rollout is full; compute returns and update first.")
        n = storage.num_envs
        # env.step() 会原地更新观测；必须在物理推进前保存当前状态及旧策略量。
        observations = self._float(observations, (n, self.policy.num_actor_obs), "observations").clone()
        critic_observations = self._float(
            critic_observations, (n, self.policy.num_critic_obs), "critic_observations"
        ).clone()
        actions = self.policy.act(observations)
        if (not isinstance(actions, torch.Tensor) or actions.shape not in ((n,), (n, 1))
                or actions.dtype not in (torch.int32, torch.int64)
                or ((actions < 0) | (actions > 1)).any()):
            raise ValueError("The selector policy must sample integer VP=0 / LP=1 modes.")
        actions = actions.to(self.device, dtype=torch.long).reshape(n, 1).clone()
        self._pending = {
            "observations": observations, "critic_observations": critic_observations, "actions": actions,
            "values": self._float(self.policy.evaluate(critic_observations), (n, 1), "values").clone(),
            "actions_log_prob": self._float(
                self.policy.get_actions_log_prob(actions), (n,), "actions_log_prob"
            ).clone(),
            "action_logits": self._float(self.policy.action_logits, (n, 2), "action_logits").clone(),
        }
        # 返回动作与待写入的动作也不共享存储，避免外部改写破坏旧 log_prob 的对应关系。
        return actions.clone()

    @torch.no_grad()
    def process_env_step(self, rewards, dones, infos):
        storage = self._storage()
        if self._pending is None:
            raise RuntimeError("Call act() before processing an environment step.")
        if not isinstance(infos, Mapping):
            raise TypeError("Environment extras must be a mapping.")
        done = storage._bool(dones, "dones")
        time_outs = storage._bool(infos.get("time_outs", torch.zeros_like(done)), "time_outs")
        if (time_outs & ~done).any():
            raise ValueError("A timeout must also end the transition's episode.")
        timeout_values = torch.zeros(storage.num_envs, 1, device=self.device)
        if time_outs.any():
            ids = infos.get("terminal_env_ids")
            terminal = infos.get("terminal_critic_observations")
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 1
                    or ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Timeouts require integer terminal_env_ids and pre-reset terminal critic observations.")
            ids = ids.to(self.device, dtype=torch.long)
            if ((ids < 0) | (ids >= storage.num_envs)).any() or ids.unique().numel() != ids.numel():
                raise ValueError("terminal_env_ids must be unique, valid environment indices.")
            if not done[ids].all():
                raise ValueError("terminal_env_ids may only refer to terminated environments.")
            if (not isinstance(terminal, torch.Tensor)
                    or terminal.shape != (ids.numel(), self.policy.num_critic_obs)
                    or not terminal.is_floating_point()):
                raise ValueError("Terminal critic rows must match terminal_env_ids and the critic observation layout.")
            terminal = terminal.to(self.device)
            selected = time_outs[ids, 0]
            timeout_ids = ids[selected]
            if timeout_ids.numel() != int(time_outs.sum()):
                raise ValueError("Terminal critic observations must cover every timeout environment.")
            # 只估值纯超时的首次末端快照；不使用返回的 reset 观测或当前状态价值。
            # 真实失败行不需要 bootstrap，即使其末端观测非有限也不送入网络。
            timeout_obs = self._float(
                terminal[selected], (timeout_ids.numel(), self.policy.num_critic_obs), "timeout critic observations"
            )
            timeout_values[timeout_ids] = self._float(
                self.policy.evaluate(timeout_obs), (timeout_ids.numel(), 1), "timeout values"
            )
        storage.add_transitions(SelectorTransition(
            **self._pending, rewards=rewards, dones=done, time_outs=time_outs, timeout_values=timeout_values,
            low_level_steps=infos.get("low_level_steps"),
        ))
        self._pending = None
        self.policy.reset(done[:, 0])

    @torch.no_grad()
    def compute_returns(self, last_critic_observations):
        storage = self._storage()
        if self._pending is not None or storage.step != storage.num_transitions_per_env:
            raise RuntimeError("Complete the selector rollout before computing returns.")
        observations = self._float(
            last_critic_observations, (storage.num_envs, self.policy.num_critic_obs), "last critic observations"
        )
        storage.compute_returns(
            self.policy.evaluate(observations), self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def _compute_loss(self, batch):
        values, log_prob, entropy = self.policy.evaluate_actions(
            batch.observations, batch.critic_observations, batch.actions
        )
        n = batch.actions.shape[0]
        if values.shape != (n, 1) or log_prob.shape != (n,) or entropy.shape != (n,):
            raise RuntimeError("Policy outputs must be values (N,1), log probabilities (N,) and entropy (N,).")
        advantages = batch.advantages[:, 0]
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        # log_prob 与优势都为 (N,)，避免 (N,1) 和 (N,) 广播成 (N,N)。
        ratio = (log_prob - batch.old_actions_log_prob[:, 0]).exp()
        surrogate_loss = -torch.minimum(
            ratio * advantages, ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        ).mean()
        value_errors = (values - batch.returns).square()
        if self.use_clipped_value_loss:
            clipped_values = batch.old_values + (values - batch.old_values).clamp(-self.clip_param, self.clip_param)
            value_errors = torch.maximum(value_errors, (clipped_values - batch.returns).square())
        value_loss, mean_entropy = value_errors.mean(), entropy.mean()
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * mean_entropy
        with torch.no_grad():
            kl = categorical_kl(batch.old_action_logits, self.policy.action_logits).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
        return loss, {"surrogate_loss": surrogate_loss, "value_loss": value_loss, "entropy": mean_entropy,
                      "kl": kl, "clip_fraction": clip_fraction}

    def _adapt_learning_rate(self, kl):
        if self.schedule != "adaptive":
            return
        if kl > 2.0 * self.desired_kl:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self):
        storage = self._storage()
        if self._pending is not None or storage.step != storage.num_transitions_per_env or not storage.returns_ready:
            raise RuntimeError("Compute returns for a complete selector rollout before updating.")
        self.train_mode()
        totals = dict.fromkeys(("value_loss", "surrogate_loss", "entropy", "kl", "clip_fraction", "grad_norm"), 0.0)
        count = 0
        for batch in storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            loss, metrics = self._compute_loss(batch)
            if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Selector PPO loss or categorical KL is non-finite.")
            self._adapt_learning_rate(metrics["kl"].item())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            for name, value in metrics.items():
                totals[name] += value.detach().item()
            totals["grad_norm"] += grad_norm.item()
            count += 1
        result = {name: value / count for name, value in totals.items()}
        result.update(learning_rate=self.learning_rate, num_updates=count)
        storage.clear()
        return result

    def train_mode(self):
        self.policy.train()

    def test_mode(self):
        self.policy.eval()


__all__ = ["DiscretePPO", "categorical_kl"]
