from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _cfg_get(cfg, key):
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


class MhaMapEncoder(nn.Module):
    """AME-style point-wise map encoder for xyz image observations."""

    def __init__(
        self,
        input_channels: int,
        input_dim: tuple[int, int],
        query_dim: int,
        embed_dim: int = 64,
        num_heads: int = 16,
        point_feature_dim: int = 3,
        conv_hidden_channels: int = 16,
        kernel_size: int = 5,
        activation: str = "elu",
    ):
        super().__init__()
        if embed_dim <= point_feature_dim:
            raise ValueError("embed_dim must be larger than point_feature_dim.")
        if input_channels % point_feature_dim != 0:
            raise ValueError(
                f"input_channels ({input_channels}) must be divisible by point_feature_dim ({point_feature_dim})."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.input_channels = input_channels
        self.input_h, self.input_w = input_dim
        self.point_feature_dim = point_feature_dim
        self.history_frames = input_channels // point_feature_dim
        self.embed_dim = embed_dim

        act_fn = resolve_nn_activation(activation)
        padding = kernel_size // 2
        self.local_cnn = nn.Sequential(
            nn.Conv2d(input_channels, conv_hidden_channels, kernel_size=kernel_size, padding=padding),
            act_fn,
            nn.Conv2d(conv_hidden_channels, embed_dim - point_feature_dim, kernel_size=kernel_size, padding=padding),
            act_fn,
        )
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, visual_flat: torch.Tensor, query_context: torch.Tensor) -> torch.Tensor:
        batch_size = visual_flat.shape[0]
        visual = visual_flat.reshape(batch_size, self.input_channels, self.input_h, self.input_w)
        point_history = visual.reshape(
            batch_size,
            self.history_frames,
            self.point_feature_dim,
            self.input_h,
            self.input_w,
        )
        current_points = point_history[:, -1]

        local_features = self.local_cnn(visual)
        point_tokens = torch.cat([current_points, local_features], dim=1)
        point_tokens = point_tokens.flatten(2).transpose(1, 2)

        query = self.query_proj(query_context).unsqueeze(1)
        attended, attn_weights = self.attention(
            query,
            point_tokens,
            point_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = attn_weights.detach()
        return self.output_norm(attended.squeeze(1) + query.squeeze(1))


class MhaActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        CnnMlp=None,
        **kwargs,
    ):
        super().__init__()
        if CnnMlp is None:
            raise ValueError("MhaActorCritic requires CnnMlp input_dim/input_channels metadata.")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        self.single_proprio_dim = kwargs.pop("single_proprio_dim", 96)
        his_encoder_dims = kwargs.pop("his_encoder_dims", [256, 128])
        self.his_latent_dim = kwargs.pop("his_latent_dim", 64)

        self.use_vel_estimation = kwargs.pop("use_vel_estimation", False)
        vel_hidden_dim = kwargs.pop("vel_hidden_dim", [32])
        vel_activation_str = kwargs.pop("vel_activation", "elu")
        vel_dim = kwargs.pop("vel_dim", 3)

        self.use_terrain_recon = kwargs.pop("use_terrain_recon", False)
        terrain_hidden_dim = kwargs.pop("terrain_hidden_dim", [256, 128])
        terrain_activation_str = kwargs.pop("terrain_activation", "elu")
        terrain_scan_dim = kwargs.pop("terrain_scan_dim", 187)
        terrain_recon_front_only = kwargs.pop("terrain_recon_front_only", False)
        terrain_recon_grid_cols = kwargs.pop("terrain_recon_grid_cols", 17)
        terrain_recon_grid_rows = kwargs.pop("terrain_recon_grid_rows", 11)
        terrain_recon_x_min = kwargs.pop("terrain_recon_x_min", 0.0)

        self.mha_embed_dim = kwargs.pop("mha_embed_dim", 64)
        self.mha_num_heads = kwargs.pop("mha_num_heads", 16)
        self.mha_conv_hidden_channels = kwargs.pop("mha_conv_hidden_channels", 16)
        self.mha_kernel_size = kwargs.pop("mha_kernel_size", 5)
        self.point_feature_dim = kwargs.pop("point_feature_dim", 3)

        # Legacy keys from MoeActorCritic/CNN configs are harmless for this policy.
        ignored_keys = [
            "use_gru",
            "use_film_cnn",
            "use_film_moe_gate",
            "use_separate_moe_gate_input",
            "moe_gate_command_start_idx",
            "moe_gate_command_dim",
            "use_moe_topk",
            "moe_topk",
            "moe_topk_start_iter",
            "num_experts",
            "gate_hidden_dim",
        ]
        for key in ignored_keys:
            kwargs.pop(key, None)
        if kwargs:
            print("MhaActorCritic.__init__ ignored unexpected arguments: " + str(sorted(kwargs.keys())))

        act_fn = resolve_nn_activation(activation)
        self.cnn_channels = _cfg_get(CnnMlp, "input_channels")
        input_dim = _cfg_get(CnnMlp, "input_dim")
        self.cnn_h = input_dim[0]
        self.cnn_w = input_dim[1]
        self.depth_flat_dim = self.cnn_channels * self.cnn_h * self.cnn_w
        self.depth_history_frames = self.cnn_channels // self.point_feature_dim
        self.visual_latent_dim = self.mha_embed_dim
        self.has_cnn = True

        self.proprio_actor_dim = num_actor_obs - self.depth_flat_dim
        self.proprio_critic_dim = num_critic_obs
        if self.proprio_actor_dim <= 0:
            raise ValueError(
                f"Invalid proprio_actor_dim={self.proprio_actor_dim}; "
                f"num_actor_obs={num_actor_obs}, depth_flat_dim={self.depth_flat_dim}."
            )

        self.history_encoder = self._build_mlp(
            input_dim=self.proprio_actor_dim,
            hidden_dims=his_encoder_dims,
            activation_fn=act_fn,
            output_dim=self.his_latent_dim,
        )
        query_dim = self.single_proprio_dim + self.his_latent_dim
        self.map_encoder = MhaMapEncoder(
            input_channels=self.cnn_channels,
            input_dim=(self.cnn_h, self.cnn_w),
            query_dim=query_dim,
            embed_dim=self.mha_embed_dim,
            num_heads=self.mha_num_heads,
            point_feature_dim=self.point_feature_dim,
            conv_hidden_channels=self.mha_conv_hidden_channels,
            kernel_size=self.mha_kernel_size,
            activation=activation,
        )

        if self.use_vel_estimation:
            self.vel_estimator = self._build_mlp(
                input_dim=self.his_latent_dim,
                hidden_dims=vel_hidden_dim,
                activation_fn=resolve_nn_activation(vel_activation_str),
                output_dim=vel_dim,
            )

        actor_input_dim = self.single_proprio_dim + self.his_latent_dim + self.visual_latent_dim
        if self.use_terrain_recon:
            if terrain_recon_front_only:
                resolution = 0.1
                size_x = (terrain_recon_grid_cols - 1) * resolution
                x_min_idx = max(0, int((terrain_recon_x_min + size_x / 2) / resolution))
                front_indices = []
                for y_idx in range(terrain_recon_grid_rows):
                    for x_idx in range(x_min_idx, terrain_recon_grid_cols):
                        front_indices.append(y_idx * terrain_recon_grid_cols + x_idx)
                self.terrain_output_dim = len(front_indices)
                self.register_buffer("terrain_front_indices", torch.tensor(front_indices, dtype=torch.long))
            else:
                self.terrain_output_dim = terrain_scan_dim
                self.terrain_front_indices = None

            self.terrain_decoder = self._build_mlp(
                input_dim=actor_input_dim,
                hidden_dims=terrain_hidden_dim,
                activation_fn=resolve_nn_activation(terrain_activation_str),
                output_dim=self.terrain_output_dim,
            )

        self.actor = self._build_actor(actor_input_dim, actor_hidden_dims, act_fn, num_actions)
        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, act_fn)

        print(f"MHA Map Encoder: {self.map_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        self.distribution = None
        Normal.set_default_validate_args(False)
        self._last_his_latent: Optional[torch.Tensor] = None
        self._last_actor_input: Optional[torch.Tensor] = None

    def _build_mlp(self, input_dim, hidden_dims, activation_fn, output_dim):
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(activation_fn)
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_actor(self, input_dim, hidden_dims, activation_fn, num_actions):
        return self._build_mlp(input_dim, hidden_dims, activation_fn, num_actions)

    def _build_critic(self, input_dim, hidden_dims, activation_fn):
        return self._build_mlp(input_dim, hidden_dims, activation_fn, 1)

    def _process_obs(self, obs, is_actor: bool = True):
        if not is_actor:
            return obs

        proprio_history = obs[:, : self.proprio_actor_dim]
        visual_flat = obs[:, self.proprio_actor_dim :]
        current_proprio = proprio_history[:, -self.single_proprio_dim :]
        his_latent = self.history_encoder(proprio_history)
        query_context = torch.cat([current_proprio, his_latent], dim=-1)
        map_latent = self.map_encoder(visual_flat, query_context)

        actor_input = torch.cat([current_proprio, his_latent, map_latent], dim=-1)
        if not torch.jit.is_scripting() and self.training:
            self._last_his_latent = his_latent
            self._last_actor_input = actor_input
        return actor_input

    def update_distribution(self, observations):
        actor_input = self._process_obs(observations, is_actor=True)
        mean = self.actor(actor_input)
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std, min=1e-6, max=1e3).expand_as(mean)
        else:
            std = torch.exp(torch.clamp(self.log_std, min=-10.0, max=10.0)).expand_as(mean)
        if not torch.isfinite(mean).all():
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        if not torch.isfinite(std).all():
            std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-3))
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actor_input = self._process_obs(observations, is_actor=True)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(self._process_obs(critic_observations, is_actor=False))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.jit.unused
    @property
    def action_mean(self):
        return self.distribution.mean

    @torch.jit.unused
    @property
    def action_std(self):
        return self.distribution.stddev

    @torch.jit.unused
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def forward(self, observations):
        return self.act_inference(observations)

    def reset(self, dones=None):
        pass

    def predict_velocity(self):
        if self.use_vel_estimation and self._last_his_latent is not None:
            return self.vel_estimator(self._last_his_latent)
        return None

    def predict_terrain(self):
        if self.use_terrain_recon and self._last_actor_input is not None:
            return self.terrain_decoder(self._last_actor_input)
        return None

    def set_moe_iteration(self, iteration: int):
        pass

    def get_moe_balance_loss(self):
        return torch.zeros((), device=next(self.parameters()).device)

    def get_moe_gate_entropy(self):
        return torch.zeros((), device=next(self.parameters()).device)

    def get_moe_routing_stats(self):
        return {}

    def get_attention_weights(self):
        return self.map_encoder.last_attention_weights

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
