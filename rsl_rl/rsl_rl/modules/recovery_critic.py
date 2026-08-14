from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class RecoveryCritic(nn.Module):
    """Three independent value functions for future Recovery reward streams."""

    def __init__(
        self,
        num_critic_obs: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        activation: str = "elu",
    ):
        super().__init__()
        if num_critic_obs <= 0:
            raise ValueError(f"num_critic_obs must be positive, got {num_critic_obs}.")
        if list(hidden_dims) != [512, 256]:
            raise ValueError(
                "RecoveryCritic V1 requires hidden_dims=[512, 256], got "
                f"{list(hidden_dims)}."
            )

        # Each call constructs new Linear and activation modules. There is no
        # shared backbone and no shared nn.Parameter between the three critics.
        self.task_critic = self._build_mlp(num_critic_obs, hidden_dims, activation)
        self.amp_critic = self._build_mlp(num_critic_obs, hidden_dims, activation)
        self.reg_critic = self._build_mlp(num_critic_obs, hidden_dims, activation)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims, activation: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, critic_observations: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "task": self.task_critic(critic_observations),
            "amp": self.amp_critic(critic_observations),
            "reg": self.reg_critic(critic_observations),
        }

