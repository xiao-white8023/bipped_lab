"""独立的高层离散 Actor–Critic，仅依赖 PyTorch 和 Python 标准库。

默认 Actor 输入为本体历史 780、VP 四通道 2304、LP 两通道 374、状态 8。
Critic 在相同布局之后增加 194 维特权信息。VP=0、LP=1，动作形状为 (N,1)。
通过子模块直接导入本类；不需要修改现有 modules/__init__.py 或任务注册。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _layer_widths(values, name):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a nonempty sequence of positive integers.")
    return tuple(_positive_int(value, name) for value in values)


def _activation(name):
    activations = {
        "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
        "silu": nn.SiLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported selector activation: {name!r}.")
    return activations[name]()


def _build_mlp(input_dim, widths, activation, *, activate_last):
    layers = []
    for index, width in enumerate(widths):
        layers.append(nn.Linear(input_dim, width))
        if activate_last or index < len(widths) - 1:
            layers.append(_activation(activation))
        input_dim = width
    return nn.Sequential(*layers)


def _depth_config(value):
    if value is None:
        value = {
            "input_dim": (18, 32), "input_channels": 4, "output_channels": (16, 32),
            "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "mlp_output_dim": 128,
        }
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("depth_encoder must be a configuration mapping or expose to_dict().")
    expected = {"input_dim", "input_channels", "output_channels", "kernel_size", "stride", "padding",
                "mlp_output_dim"}
    if set(value) != expected:
        raise ValueError(f"Depth encoder configuration keys must be {sorted(expected)}.")
    cfg = dict(value)
    cfg["input_dim"] = _layer_widths(cfg["input_dim"], "depth input dimensions")
    if len(cfg["input_dim"]) != 2:
        raise ValueError("Depth input dimensions must contain height and width.")
    for name in ("input_channels", "mlp_output_dim"):
        cfg[name] = _positive_int(cfg[name], name)
    for name in ("output_channels", "kernel_size", "stride"):
        cfg[name] = _layer_widths(cfg[name], name)
    if (not isinstance(cfg["padding"], Sequence) or isinstance(cfg["padding"], (str, bytes))
            or any(type(value) is not int or value < 0 for value in cfg["padding"])):
        raise ValueError("Depth CNN padding must contain nonnegative integers.")
    cfg["padding"] = tuple(cfg["padding"])
    if len({len(cfg[name]) for name in ("output_channels", "kernel_size", "stride", "padding")}) != 1:
        raise ValueError("Depth CNN layer settings must have equal lengths.")
    return cfg


class _SelectorObservationEncoder(nn.Module):
    """按环境观测顺序切片；四路编码拼接，默认输出 352 维。"""

    def __init__(self, proprio_dim, height_dim, state_dim, proprio_dims, height_dims, state_dims,
                 depth_cfg, activation):
        super().__init__()
        self.depth_shape = (depth_cfg["input_channels"], *depth_cfg["input_dim"])
        self.depth_input_dim = self.depth_shape[0] * self.depth_shape[1] * self.depth_shape[2]
        self.input_dims = (proprio_dim, self.depth_input_dim, height_dim, state_dim)
        self.input_dim = sum(self.input_dims)
        self.proprio_encoder = _build_mlp(proprio_dim, proprio_dims, activation, activate_last=True)
        self.height_encoder = _build_mlp(height_dim, height_dims, activation, activate_last=True)
        self.state_encoder = _build_mlp(state_dim, state_dims, activation, activate_last=True)

        channels, rows, cols = self.depth_shape
        depth_layers = []
        for output, kernel, stride, padding in zip(
            depth_cfg["output_channels"], depth_cfg["kernel_size"], depth_cfg["stride"], depth_cfg["padding"]
        ):
            rows = (rows + 2 * padding - kernel) // stride + 1
            cols = (cols + 2 * padding - kernel) // stride + 1
            if min(rows, cols) < 1:
                raise ValueError("Depth CNN collapses the configured image dimensions.")
            depth_layers.extend((nn.Conv2d(channels, output, kernel, stride, padding), _activation(activation)))
            channels = output
        depth_layers.extend((nn.Flatten(start_dim=1),
                             nn.Linear(channels * rows * cols, depth_cfg["mlp_output_dim"]),
                             _activation(activation)))
        self.depth_encoder = nn.Sequential(*depth_layers)
        self.output_dim = proprio_dims[-1] + depth_cfg["mlp_output_dim"] + height_dims[-1] + state_dims[-1]

    def forward(self, observations):
        if (not isinstance(observations, torch.Tensor) or observations.ndim != 2
                or observations.shape[1] != self.input_dim or not observations.is_floating_point()):
            raise ValueError(f"Selector actor observations must be floating point with shape (N,{self.input_dim}).")
        proprio, depth, height, state = observations.split(self.input_dims, dim=-1)
        depth = depth.reshape(observations.shape[0], *self.depth_shape)
        return torch.cat((self.proprio_encoder(proprio), self.depth_encoder(depth),
                          self.height_encoder(height), self.state_encoder(state)), dim=-1)


class SelectorActorCritic(nn.Module):
    """固定历史、独立编码器的分类策略，不持有冻结底层网络或环境状态。

    act() 训练采样返回 int64 (N,1)；act_inference() 部署时返回 argmax 模式。
    forward() 返回两个原始 logits；evaluate() 返回 (N,1) 的状态价值。
    每次 PPO 重新计算分布后，log_prob 和 entropy 均为 (N,)。
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs=None,
        num_critic_obs=None,
        num_actions=1,
        *,
        action_distribution="categorical",
        num_modes=2,
        critic_output_dim=1,
        activation="elu",
        share_encoders=False,
        proprio_hidden_dims=(256, 128),
        depth_encoder=None,
        height_hidden_dims=(128, 64),
        state_hidden_dims=(64, 32),
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        proprio_input_dim=780,
        height_input_dim=374,
        state_input_dim=8,
        actor_obs_dim=None,
        critic_obs_dim=None,
        critic_privileged_dim=194,
        class_name="SelectorActorCritic",
    ):
        super().__init__()
        if (action_distribution != "categorical" or type(num_modes) is not int or num_modes != 2
                or type(num_actions) is not int or num_actions != 1):
            raise ValueError("Selector requires one categorical action with VP=0 and LP=1.")
        if type(critic_output_dim) is not int or critic_output_dim != 1:
            raise ValueError("Selector critic must output one state value.")
        if share_encoders is not False:
            raise ValueError("Selector actor and critic must use independent encoders.")
        if class_name != "SelectorActorCritic":
            raise ValueError("Unexpected selector policy class_name.")
        if type(critic_privileged_dim) is not int or critic_privileged_dim < 0:
            raise ValueError("critic_privileged_dim must be a nonnegative integer.")
        self.proprio_input_dim = _positive_int(proprio_input_dim, "proprio_input_dim")
        self.height_input_dim = _positive_int(height_input_dim, "height_input_dim")
        self.state_input_dim = _positive_int(state_input_dim, "state_input_dim")
        proprio_dims = _layer_widths(proprio_hidden_dims, "proprio_hidden_dims")
        height_dims = _layer_widths(height_hidden_dims, "height_hidden_dims")
        state_dims = _layer_widths(state_hidden_dims, "state_hidden_dims")
        actor_dims = _layer_widths(actor_hidden_dims, "actor_hidden_dims")
        critic_dims = _layer_widths(critic_hidden_dims, "critic_hidden_dims")
        depth_cfg = _depth_config(depth_encoder)
        self.depth_input_dim = depth_cfg["input_channels"] * depth_cfg["input_dim"][0] * depth_cfg["input_dim"][1]
        expected_actor = self.proprio_input_dim + self.depth_input_dim + self.height_input_dim + self.state_input_dim
        expected_critic = expected_actor + critic_privileged_dim
        for name, value, expected in (
            ("num_actor_obs", num_actor_obs, expected_actor), ("actor_obs_dim", actor_obs_dim, expected_actor),
            ("num_critic_obs", num_critic_obs, expected_critic), ("critic_obs_dim", critic_obs_dim, expected_critic),
        ):
            if value is not None and (type(value) is not int or value != expected):
                raise ValueError(f"{name} must match the configured observation layout: {expected}.")
        self.num_actor_obs = self.actor_obs_dim = expected_actor
        self.num_critic_obs = self.critic_obs_dim = expected_critic
        self.num_actions, self.num_modes = num_actions, num_modes
        self.critic_privileged_dim = critic_privileged_dim
        self.action_distribution = action_distribution
        self.share_encoders = share_encoders

        # ---------------------------------------------------------
        # Actor 与 Critic 各自建立完整编码器；两者不共享参数。
        # ---------------------------------------------------------
        encoder_args = (self.proprio_input_dim, self.height_input_dim, self.state_input_dim,
                        proprio_dims, height_dims, state_dims, depth_cfg, activation)
        self.actor_encoder = _SelectorObservationEncoder(*encoder_args)
        self.critic_encoder = _SelectorObservationEncoder(*encoder_args)
        self.actor = _build_mlp(self.actor_encoder.output_dim, (*actor_dims, num_modes), activation,
                                activate_last=False)
        self.critic = _build_mlp(self.critic_encoder.output_dim + critic_privileged_dim,
                                 (*critic_dims, critic_output_dim), activation, activate_last=False)
        self.distribution: Categorical | None = None
        self._action_logits: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Actor 只接收可观测部分，返回 (N,2) 原始 logits。"""
        return self.actor(self.actor_encoder(observations))

    def update_distribution(self, observations: torch.Tensor):
        logits = self(observations)
        self.distribution = Categorical(logits=logits, validate_args=True)
        self._action_logits = logits

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample().unsqueeze(-1)

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """部署选择概率最大的模式；不改变训练分布缓存。"""
        return self(observations).argmax(dim=-1, keepdim=True)

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        if (not isinstance(critic_observations, torch.Tensor) or critic_observations.ndim != 2
                or critic_observations.shape[1] != self.num_critic_obs
                or not critic_observations.is_floating_point()):
            raise ValueError(f"Selector critic observations must be floating point with shape (N,{self.num_critic_obs}).")
        observable = critic_observations[:, :self.num_actor_obs]
        privileged = critic_observations[:, self.num_actor_obs:]
        features = self.critic_encoder(observable)
        return self.critic(torch.cat((features, privileged), dim=-1))

    def _current_distribution(self) -> Categorical:
        if self.distribution is None:
            raise RuntimeError("Call update_distribution() or act() before reading the action distribution.")
        return self.distribution

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        distribution = self._current_distribution()
        batch = distribution.logits.shape[0]
        if (not isinstance(actions, torch.Tensor) or actions.dtype not in (torch.int32, torch.int64)
                or actions.shape not in ((batch,), (batch, 1))):
            raise ValueError("Selector actions must be integer tensors with shape (N,) or (N,1).")
        actions = actions.reshape(batch).to(device=distribution.logits.device, dtype=torch.long)
        return distribution.log_prob(actions)

    def evaluate_actions(self, observations, critic_observations, actions):
        """返回 values (N,1)、log_prob (N,) 和 entropy (N,)，供离散 PPO 更新。"""
        self.update_distribution(observations)
        values = self.evaluate(critic_observations)
        if values.shape[0] != observations.shape[0]:
            raise ValueError("Actor and critic observations must have the same batch size.")
        return values, self.get_actions_log_prob(actions), self.entropy

    @property
    def action_logits(self) -> torch.Tensor:
        """当前分布对应的原始 logits，可 detach 后保存以计算分类 KL。"""
        self._current_distribution()
        return self._action_logits

    @property
    def action_probs(self) -> torch.Tensor:
        return self._current_distribution().probs

    @property
    def entropy(self) -> torch.Tensor:
        return self._current_distribution().entropy()

    def reset(self, dones=None):
        # 本体/图像历史由环境管理；前馈网络没有跨 episode 的隐藏状态。
        return None


__all__ = ["SelectorActorCritic"]
