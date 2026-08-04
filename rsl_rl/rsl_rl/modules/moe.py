import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation

# from .actor_critic import get_activation  # Assuming get_activation is needed


class GateFiLM(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int, activation="elu", hidden_dim: int | None = None):
        super().__init__()
        if isinstance(activation, str):
            act_fn = resolve_nn_activation(activation)
        elif isinstance(activation, nn.Module):
            act_fn = activation
        else:
            raise ValueError(f"activation must be a string or nn.Module, got: {type(activation)}")

        hidden_dim = feature_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 2 * feature_dim),
        )

        # Start from identity modulation: gamma = 1, beta = 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        gamma = 1.0 + gamma
        return gamma, beta


class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        output_dim=None,
        activation="elu",
        expert_hidden_dims=[],
        gate_hidden_dims=[],
        use_gate_film=False,
        gate_film_condition_dim=None,
        gate_input_dim=None,
        use_top_k=False,
        top_k=2,
        top_k_start_iter=2000,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_gate_film = use_gate_film
        self.gate_input_dim = input_dim if gate_input_dim is None else gate_input_dim
        self.use_separate_gate_input = gate_input_dim is not None
        self.use_top_k = use_top_k
        self.top_k = top_k
        self.top_k_start_iter = top_k_start_iter
        self.current_iteration = 0
        self.last_raw_gate_scores = None
        self.last_gate_scores = None
        self.last_gate_entropy = None
        self.last_balance_loss = None
        self.last_expert_usage = None
        self.last_expert_top1_freq = None
        self.last_expert_action_disagreement = None
        self.last_action_disagreement = None
        # 激活函数
        if isinstance(activation, str):
            # 如果是字符串，就用工具函数转换
            self.act_fn = resolve_nn_activation(activation)
        elif isinstance(activation, nn.Module):
            # 如果已经是 nn.Module 对象 (比如 ELU())，直接使用
            self.act_fn = activation
        else:
            raise ValueError(f"activation 必须是字符串 (如 'elu') 或 nn.Module 对象，当前类型: {type(activation)}")

        # 创建门控
        self.gate = self._build_gate(self.gate_input_dim, num_experts, gate_hidden_dims)
        self.gate_feature_dim = gate_hidden_dims[-1] if len(gate_hidden_dims) > 0 else self.gate_input_dim
        if self.use_gate_film:
            if gate_film_condition_dim is None:
                raise ValueError("gate_film_condition_dim must be set when use_gate_film=True.")
            self.gate_film = GateFiLM(gate_film_condition_dim, self.gate_feature_dim, activation=self.act_fn)
        else:
            self.gate_film = None
        # 创建专家网络
        self.experts = nn.ModuleList(
            [self._build_expert(input_dim, output_dim, expert_hidden_dims) for _ in range(num_experts)] # 建立四个专家网络
        )

    def _build_gate(self, input_dim, num_experts, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        layers.append(nn.Linear(curr_dim, num_experts)) # 这一步就是说门控网络输出了4个权重数字
        return nn.Sequential(*layers)

    def _build_expert(self, input_dim, output_dim, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(curr_dim, output_dim))  # no activation for the last layer
        return nn.Sequential(*layers)

    def set_iteration(self, iteration: int):
        self.current_iteration = int(iteration)

    def _forward_gate(self, gate_input, gate_condition=None):
        if not self.use_gate_film:
            return self.gate(gate_input)

        if gate_condition is None:
            raise ValueError("gate_condition must be provided when use_gate_film=True.")
        # 执行到这里说明是使用了film调制
        gate_layers = list(self.gate.children()) # self.gate是由_build_gate函数建立起来的，假设hidden_dims是[64],output_dims是4,则gate_layer=[Linear(input_dim, 64),ELU(),Linear(64, 4)]
        h = gate_input
        for layer in gate_layers[:-1]: # 则[Linear(input_dim, 64),ELU()]
            h = layer(h) # h=linear(h) h=ELU(h) h就是隐藏特征

        gamma, beta = self.gate_film(gate_condition)
        h = gamma * h + beta
        return gate_layers[-1](h)  # 这一步把调制后的隐藏特征 h 输入最后一层线性层。

    def _apply_top_k(self, raw_gate_scores):
        if not self.use_top_k or self.current_iteration < self.top_k_start_iter:
            return raw_gate_scores

        k = max(1, min(int(self.top_k), self.num_experts))
        topk_values, topk_indices = torch.topk(raw_gate_scores, k=k, dim=-1)
        sparse_scores = torch.zeros_like(raw_gate_scores)
        sparse_scores.scatter_(dim=-1, index=topk_indices, src=topk_values)
        return sparse_scores / sparse_scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _update_gate_stats(self, raw_gate_scores, gate_scores):
        mean_usage = raw_gate_scores.mean(dim=0)
        target_usage = torch.full_like(mean_usage, 1.0 / self.num_experts)

        self.last_raw_gate_scores = raw_gate_scores.detach()
        self.last_gate_scores = gate_scores.detach()
        self.last_balance_loss = torch.mean((mean_usage - target_usage) ** 2)
        self.last_gate_entropy = -(
            raw_gate_scores * torch.log(raw_gate_scores + 1.0e-8)
        ).sum(dim=-1).mean()
        self.last_expert_usage = gate_scores.detach().mean(dim=0)
        top1_indices = torch.argmax(raw_gate_scores.detach(), dim=-1)
        self.last_expert_top1_freq = F.one_hot(
            top1_indices,
            num_classes=self.num_experts,
        ).float().mean(dim=0)

    def _update_expert_stats(self, expert_outputs):
        expert_mean_output = expert_outputs.detach().mean(dim=1, keepdim=True)
        per_expert_disagreement = torch.square(
            expert_outputs.detach() - expert_mean_output
        ).mean(dim=(0, 2))
        self.last_expert_action_disagreement = per_expert_disagreement
        self.last_action_disagreement = per_expert_disagreement.mean()

    def get_balance_loss(self, device=None):
        if self.last_balance_loss is None:
            return torch.zeros(()) if device is None else torch.zeros((), device=device)
        return self.last_balance_loss

    def get_gate_entropy(self, device=None):
        if self.last_gate_entropy is None:
            return torch.zeros(()) if device is None else torch.zeros((), device=device)
        return self.last_gate_entropy

    def get_routing_stats(self, device=None):
        if device is None:
            if self.last_expert_usage is not None:
                device = self.last_expert_usage.device
            else:
                device = next(self.parameters()).device

        zeros = torch.zeros(self.num_experts, device=device)
        scalar_zero = torch.zeros((), device=device)
        return {
            "expert_usage": self.last_expert_usage if self.last_expert_usage is not None else zeros,
            "expert_top1_freq": self.last_expert_top1_freq if self.last_expert_top1_freq is not None else zeros,
            "expert_action_disagreement": (
                self.last_expert_action_disagreement
                if self.last_expert_action_disagreement is not None
                else zeros
            ),
            "action_disagreement": (
                self.last_action_disagreement
                if self.last_action_disagreement is not None
                else scalar_zero
            ),
        }

    def forward(self, x, gate_input=None, gate_condition=None):
        if gate_input is None:
            gate_input = x
        elif gate_input.shape[-1] != self.gate_input_dim:
            raise ValueError(
                f"gate_input dim mismatch: expected {self.gate_input_dim}, got {gate_input.shape[-1]}"
            )

        gate_logits = self._forward_gate(gate_input, gate_condition)
        raw_gate_scores = F.softmax(gate_logits, dim=-1)  # [batch, num_experts]
        gate_scores = self._apply_top_k(raw_gate_scores)
        self._update_gate_stats(raw_gate_scores, gate_scores)

        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        self._update_expert_stats(expert_outputs)
        output = torch.einsum("be,beo->bo", gate_scores, expert_outputs)  # mix the expert outputs
        return output
