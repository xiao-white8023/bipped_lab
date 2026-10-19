"""高层离散 PPO 的独立 rollout 存储，不继承现有 RolloutStorage。

一条 transition 对应一次高层决策。奖励已经由环境按底层 dt 积分，只保存原值。
超时保存末端 Critic 估值；GAE 在 done 处停止，不能进入重置后的 episode。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectorTransition:
    observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    actions_log_prob: torch.Tensor
    action_logits: torch.Tensor
    time_outs: torch.Tensor | None = None
    timeout_values: torch.Tensor | None = None
    low_level_steps: torch.Tensor | None = None


@dataclass(frozen=True)
class SelectorMiniBatch:
    observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_actions_log_prob: torch.Tensor
    old_action_logits: torch.Tensor
    dones: torch.Tensor
    time_outs: torch.Tensor
    timeout_values: torch.Tensor
    low_level_steps: torch.Tensor


class SelectorRolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_dim, critic_obs_dim, *,
                 num_modes=2, low_level_steps_per_decision=10, device="cpu"):
        for name, value in (("num_envs", num_envs), ("num_transitions_per_env", num_transitions_per_env),
                            ("actor_obs_dim", actor_obs_dim), ("critic_obs_dim", critic_obs_dim),
                            ("low_level_steps_per_decision", low_level_steps_per_decision)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(num_modes) is not int or num_modes != 2:
            raise ValueError("Selector storage requires VP=0 and LP=1.")
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actor_obs_dim, self.critic_obs_dim = actor_obs_dim, critic_obs_dim
        self.num_modes = num_modes
        self.low_level_steps_per_decision = low_level_steps_per_decision
        self.device = torch.empty(0, device=device).device
        shape = (num_transitions_per_env, num_envs)
        self.observations = torch.zeros(*shape, actor_obs_dim, device=self.device)
        self.critic_observations = torch.zeros(*shape, critic_obs_dim, device=self.device)
        self.actions = torch.zeros(*shape, 1, dtype=torch.long, device=self.device)
        self.action_logits = torch.zeros(*shape, num_modes, device=self.device)
        for name in ("rewards", "values", "actions_log_prob", "timeout_values", "returns", "advantages"):
            setattr(self, name, torch.zeros(*shape, 1, device=self.device))
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=self.device)
        self.time_outs = torch.zeros_like(self.dones)
        self.low_level_steps = torch.zeros_like(self.actions)
        self.step = 0
        self.returns_ready = False

    def _tensor(self, value, shape, name, *, vector=False):
        allowed = (shape, (self.num_envs,)) if vector else (shape,)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) not in allowed:
            raise ValueError(f"{name} must have shape {shape}.")
        return value.detach().reshape(shape).to(self.device)

    def _float(self, value, shape, name, *, vector=False):
        value = self._tensor(value, shape, name, vector=vector)
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point.")
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
        return value

    def _bool(self, value, name):
        value = self._tensor(value, (self.num_envs, 1), name, vector=True)
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean.")
        return value

    @torch.no_grad()
    def add_transitions(self, transition: SelectorTransition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Selector rollout is full; update and clear before collecting again.")
        if not isinstance(transition, SelectorTransition):
            raise TypeError("Expected a SelectorTransition.")
        n, column = self.num_envs, (self.num_envs, 1)
        data = {
            "observations": self._float(transition.observations, (n, self.actor_obs_dim), "observations"),
            "critic_observations": self._float(transition.critic_observations, (n, self.critic_obs_dim),
                                                "critic_observations"),
        }
        for name in ("rewards", "values", "actions_log_prob"):
            data[name] = self._float(getattr(transition, name), column, name, vector=True)
        data["action_logits"] = self._float(transition.action_logits, (n, self.num_modes), "action_logits")
        actions = self._tensor(transition.actions, column, "actions", vector=True)
        if actions.dtype not in (torch.int32, torch.int64) or ((actions < 0) | (actions >= self.num_modes)).any():
            raise ValueError("actions must be integer VP=0 / LP=1 modes.")
        data["actions"] = actions.long()
        expected_log_prob = F.log_softmax(data["action_logits"], dim=-1).gather(-1, data["actions"])
        if not torch.allclose(data["actions_log_prob"], expected_log_prob, atol=1e-5, rtol=1e-4):
            raise ValueError("Old action log probabilities do not match the stored categorical logits/actions.")
        data["dones"] = self._bool(transition.dones, "dones")
        data["time_outs"] = (torch.zeros_like(data["dones"]) if transition.time_outs is None
                            else self._bool(transition.time_outs, "time_outs"))
        if (data["time_outs"] & ~data["dones"]).any():
            raise ValueError("A timeout must also end the transition's episode.")
        if transition.timeout_values is None:
            if data["time_outs"].any():
                raise ValueError("Timeouts require pre-reset terminal critic values.")
            timeout_values = torch.zeros(n, 1, device=self.device)
        else:
            timeout_values = self._float(transition.timeout_values, column, "timeout_values", vector=True)
        data["timeout_values"] = torch.where(data["time_outs"], timeout_values, 0.0)
        steps = (torch.full(column, self.low_level_steps_per_decision, dtype=torch.long, device=self.device)
                 if transition.low_level_steps is None else
                 self._tensor(transition.low_level_steps, column, "low_level_steps", vector=True))
        if (steps.dtype not in (torch.int32, torch.int64)
                or ((steps < 1) | (steps > self.low_level_steps_per_decision)).any()):
            raise ValueError("low_level_steps must be positive integer counts within one high-level decision.")
        if ((~data["dones"]) & (steps != self.low_level_steps_per_decision)).any():
            raise ValueError("Unterminated transitions must contain a complete high-level control period.")
        data["low_level_steps"] = steps.long()
        # 全部验证完成后再写入，避免错误输入留下半条 transition。
        for name, value in data.items():
            getattr(self, name)[self.step].copy_(value)
        self.step += 1
        self.returns_ready = False

    @torch.no_grad()
    def compute_returns(self, last_values, gamma, lam, *, normalize_advantage=True):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Fill the complete selector rollout before computing returns.")
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0 or not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma and lam must lie in their probability ranges.")
        last_values = self._float(last_values, (self.num_envs, 1), "last_values", vector=True)
        advantage = torch.zeros_like(last_values)
        for index in reversed(range(self.num_transitions_per_env)):
            next_value = last_values if index == self.num_transitions_per_env - 1 else self.values[index + 1]
            # 普通步取下一状态价值；真实失败为零；超时取首次终止时的末端价值。
            bootstrap = torch.where(self.dones[index], 0.0, next_value)
            bootstrap = torch.where(self.time_outs[index], self.timeout_values[index], bootstrap)
            delta = self.rewards[index] + gamma * bootstrap - self.values[index]
            # 两种 done 都阻断 GAE，重置后 episode 的优势不能传回来。
            advantage = delta + gamma * lam * (~self.dones[index]).float() * advantage
            self.advantages[index].copy_(advantage)
            self.returns[index].copy_(advantage + self.values[index])
        if normalize_advantage:
            mean = self.advantages.mean()
            std = self.advantages.std(unbiased=False)
            self.advantages.sub_(mean).div_(std + 1e-8)
        # gamma/lam 按高层 transition 计；low_level_steps 不再引入额外时间缩放。
        self.returns_ready = True

    def mini_batch_generator(self, num_mini_batches, num_epochs=1, *, generator=None):
        if self.step != self.num_transitions_per_env or not self.returns_ready:
            raise RuntimeError("Compute returns for a full rollout before requesting minibatches.")
        for name, value in (("num_mini_batches", num_mini_batches), ("num_epochs", num_epochs)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        size = self.num_envs * self.num_transitions_per_env
        if size % num_mini_batches:
            raise ValueError("The rollout batch must divide evenly into minibatches; no samples may be dropped.")
        names = {
            "observations": "observations", "critic_observations": "critic_observations", "actions": "actions",
            "old_values": "values", "returns": "returns", "advantages": "advantages",
            "old_actions_log_prob": "actions_log_prob", "old_action_logits": "action_logits",
            "dones": "dones", "time_outs": "time_outs", "timeout_values": "timeout_values",
            "low_level_steps": "low_level_steps",
        }
        flat = {key: getattr(self, name).flatten(0, 1) for key, name in names.items()}
        width = size // num_mini_batches
        for _ in range(num_epochs):
            order = torch.randperm(size, device=self.device, generator=generator)
            for start in range(0, size, width):
                indices = order[start:start + width]
                yield SelectorMiniBatch(**{name: values[indices] for name, values in flat.items()})

    def clear(self):
        self.step = 0
        self.returns_ready = False


__all__ = ["SelectorTransition", "SelectorMiniBatch", "SelectorRolloutStorage"]
