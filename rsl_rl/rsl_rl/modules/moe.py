from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation

# from .actor_critic import get_activation  # Assuming get_activation is needed


class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        output_dim=None,
        activation="elu",
        expert_hidden_dims=[],
        gate_hidden_dims=[],
    ):
        super().__init__()
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
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
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

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [batch, num_experts] # gate the expert outputs
        # === 新增：保存当前门控权重，供 play.py 读取 ===
        #self.current_gating_weights = gate_scores.detach()
        # ===============================================
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        output = torch.einsum("be,beo->bo", gate_scores, expert_outputs)  # mix the expert outputs
        return output
