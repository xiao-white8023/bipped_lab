# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class HwcRecoveryActor(nn.Module):
    """HWC-Loco mimic actor: per-frame encoder, temporal merger, actor backbone."""

    def __init__(
        self,
        num_prop: int,
        num_demo: int,
        text_feat_input_dim: int,
        text_feat_output_dim: int,
        feat_hist_len: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        n_decoder_out: int,
        activation: nn.Module,
        tanh_encoder_output: bool = False,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.num_demo = num_demo
        self.text_feat_input_dim = text_feat_input_dim
        self.text_feat_output_dim = text_feat_output_dim
        self.feat_hist_len = feat_hist_len
        self.n_decoder_out = n_decoder_out

        self.text_feat_encoder = nn.Sequential(
            nn.Linear(num_prop, 128),
            activation,
            nn.Linear(128, text_feat_output_dim),
            activation,
        )
        self.text_feat_merger = nn.Sequential(
            nn.Linear(text_feat_output_dim * feat_hist_len, text_feat_output_dim),
            activation,
        )

        actor_input_dim = text_feat_output_dim + num_prop + num_demo + n_decoder_out
        actor_layers = [nn.Linear(actor_input_dim, actor_hidden_dims[0]), activation]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    @property
    def actor_input_dim(self) -> int:
        return self.text_feat_output_dim + self.num_prop + self.num_demo + self.n_decoder_out

    def forward(self, obs_all: torch.Tensor, hist_encoding: bool = False, **kwargs) -> torch.Tensor:
        # obs_all = [feature_history, current_proprio, demo, estimated_latent_and_labels]
        text_feat = obs_all[:, : self.text_feat_input_dim].reshape(-1, self.num_prop)
        text_feat_latent = self.text_feat_encoder(text_feat).view(obs_all.shape[0], -1)
        text_feat_latent = self.text_feat_merger(text_feat_latent)

        obs = obs_all[:, self.text_feat_input_dim :]
        obs_prop = obs[:, : self.num_prop]
        obs_demo = obs[:, self.num_prop : self.num_prop + self.num_demo]
        obs_priv_explicit = obs[:, -self.n_decoder_out :]
        backbone_input = torch.cat([text_feat_latent, obs_prop, obs_demo, obs_priv_explicit], dim=1)
        return self.actor_backbone(backbone_input)


class ActorCriticCost(nn.Module):
    """HWC-Loco actor-critic with independent reward and ZMP cost critics."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type

        self.num_prop = kwargs.pop("num_prop")
        self.num_demo = kwargs.pop("num_demo", 0)
        self.text_feat_input_dim = kwargs.pop("text_feat_input_dim")
        self.text_feat_output_dim = kwargs.pop("text_feat_output_dim", 16)
        self.feat_hist_len = kwargs.pop("feat_hist_len", 5)
        self.n_decoder_out = kwargs.pop("n_decoder_out")
        self.num_priv_explicit = kwargs.pop("num_priv_explicit", self.n_decoder_out)
        self.num_hist = kwargs.pop("num_hist", 6)
        tanh_encoder_output = kwargs.pop("tanh_encoder_output", False)
        if kwargs:
            print("ActorCriticCost.__init__ ignored arguments: " + str([key for key in kwargs.keys()]))

        self.actor = HwcRecoveryActor(
            num_prop=self.num_prop,
            num_demo=self.num_demo,
            text_feat_input_dim=self.text_feat_input_dim,
            text_feat_output_dim=self.text_feat_output_dim,
            feat_hist_len=self.feat_hist_len,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            n_decoder_out=self.n_decoder_out,
            activation=activation_fn,
            tanh_encoder_output=tanh_encoder_output,
        )
        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, activation_fn)
        self.critic_cost = self._build_critic(num_critic_obs, critic_hidden_dims, activation_fn)

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}.")
        self.distribution = None

        Normal.set_default_validate_args(False)

        print(f"HWC Recovery Actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Cost Critic MLP: {self.critic_cost}")
        print(
            "HWC Recovery dims: "
            f"raw_actor_obs={num_actor_obs}, actor_estimated_obs={self.text_feat_input_dim + self.num_prop + self.num_demo + self.n_decoder_out}, "
            f"actor_backbone_input={self.actor.actor_input_dim}, critic_obs={num_critic_obs}, actions={num_actions}"
        )

    def _build_critic(self, input_dim, hidden_dims, activation_fn):
        critic_layers = [nn.Linear(input_dim, hidden_dims[0]), activation_fn]
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        return nn.Sequential(*critic_layers)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self.act_inference(observations)

    def update_distribution(self, observations, hist_encoding: bool = False):
        mean = self.actor(observations, hist_encoding=hist_encoding)
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std, min=1.0e-6, max=1.0e3).expand_as(mean)
        else:
            std = torch.exp(torch.clamp(self.log_std, min=-10.0, max=10.0)).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, hist_encoding: bool = False, **kwargs):
        self.update_distribution(observations, hist_encoding=hist_encoding)
        return self.distribution.sample()

    def act_inference(self, observations, hist_encoding: bool = False, **kwargs):
        return self.actor(observations, hist_encoding=hist_encoding)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def evaluate_cost(self, critic_observations, **kwargs):
        return self.critic_cost(critic_observations)

    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        if self.noise_std_type == "scalar":
            self.std.data = new_std.data
        else:
            self.log_std.data = torch.log(new_std).data
