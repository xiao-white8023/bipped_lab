from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .cnn_mlp import CnnMlp as CnnMlpModule


class G1DepthActorCritic(nn.Module):
    """Independent four-head VP: proprio_history | depth_history, without routing."""

    is_recurrent = False

    def __init__(
        self, num_actor_obs, num_critic_obs, num_actions,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", init_noise_std=1.0, noise_std_type="scalar", state_dependent_std=False,
        actor_obs_normalization=False, critic_obs_normalization=False, CnnMlp=None,
        single_proprio_dim=78, estimator_latent_dim=64, vp_output_type="multi_head",
        vp_context_dim=32, vp_terrain_latent_dim=32, vp_head_hidden_dims=[64],
        vp_vae_decoder_dims=[128, 256], proprio_embed_dim=64,
        proprio_embed_dims=[256, 128], fusion_type="attention", attention_num_heads=1,
        vp_proprio_history_frames=10, single_critic_dim=85, critic_history_frames=10,
        vp_encoder_dims=[128], use_vel_estimation=True, vel_activation="elu",
        vel_hidden_dim=[32], vel_dim=3, use_terrain_recon=True,
        terrain_activation="elu", terrain_hidden_dim=[256, 128], terrain_scan_dim=187,
        terrain_recon_front_only=True, terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11, terrain_recon_x_min=0.0,
        use_feet_height_prediction=True, feet_height_activation="elu",
        feet_height_hidden_dim=[64], feet_height_dim=2,
    ):
        super().__init__()
        if CnnMlp is None:
            raise ValueError("g1_depth requires a depth CNN configuration.")
        expected = (num_actor_obs, num_critic_obs, num_actions, single_proprio_dim,
                    vp_proprio_history_frames, single_critic_dim, critic_history_frames,
                    estimator_latent_dim, proprio_embed_dim, vp_context_dim,
                    vp_terrain_latent_dim, vel_dim, feet_height_dim, terrain_scan_dim)
        if expected != (1932, 1037, 23, 78, 10, 85, 10, 64, 64, 32, 32, 3, 2, 187):
            raise ValueError(f"Incompatible g1_depth observation/network dimensions: {expected}")
        if vp_output_type != "multi_head" or fusion_type != "attention":
            raise ValueError("g1_depth uses the four-head attention VP encoder.")
        if not (use_vel_estimation and use_terrain_recon and use_feet_height_prediction):
            raise ValueError("The four-head VP requires velocity, terrain and feet-height supervision.")
        if attention_num_heads != 1:
            raise ValueError("g1_depth VP attention uses one head.")
        if tuple(CnnMlp["input_dim"]) != (18, 32) or CnnMlp["input_channels"] != 2:
            raise ValueError("g1_depth consumes two 18x32 depth frames.")
        if not init_noise_std > 0:
            raise ValueError("init_noise_std must be positive.")
        if state_dependent_std:
            raise ValueError("g1_depth uses one learned action standard deviation per joint.")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("g1_depth observation normalization is owned by its AMP runner.")

        self.num_actor_obs, self.num_critic_obs, self.num_actions = num_actor_obs, num_critic_obs, num_actions
        self.single_proprio_dim = single_proprio_dim
        self.proprio_actor_dim = 10 * single_proprio_dim
        self.history_len = self.vp_proprio_history_frames = vp_proprio_history_frames
        self.single_critic_dim, self.critic_history_frames = single_critic_dim, critic_history_frames
        self.critic_terrain_dim = terrain_scan_dim
        self.estimator_latent_dim, self.proprio_embed_dim = estimator_latent_dim, proprio_embed_dim
        self.vp_output_type, self.fusion_type = vp_output_type, fusion_type
        self.vp_context_dim, self.vp_terrain_latent_dim = vp_context_dim, vp_terrain_latent_dim
        self.vel_dim, self.feet_height_dim = vel_dim, feet_height_dim
        self.use_vel_estimation = use_vel_estimation
        self.use_terrain_recon = use_terrain_recon
        self.use_feet_height_prediction = use_feet_height_prediction
        self.use_vp_vae = True
        self.vp_feature_dim = vel_dim + feet_height_dim + vp_context_dim + vp_terrain_latent_dim
        self.depth_history_frames = 2
        self.cnn_h, self.cnn_w = 18, 32
        self.depth_flat_dim = 2 * 18 * 32

        activation_fn = resolve_nn_activation(activation)
        self.proprio_embedding = self._build_mlp(
            single_proprio_dim, proprio_embed_dims, activation_fn, proprio_embed_dim,
        )
        cnn_cfg = dict(CnnMlp)
        cnn_cfg["input_channels"] = 1  # 两帧共享同一个逐帧 CNN。
        self.cnn = CnnMlpModule(**cnn_cfg)
        self.visual_latent_dim = self.cnn.output_dim
        if self.visual_latent_dim != proprio_embed_dim:
            raise ValueError("Depth and proprioception tokens must both have dimension 64.")
        self.vp_attention = nn.MultiheadAttention(proprio_embed_dim, attention_num_heads, batch_first=True)
        self.vp_encoder = self._build_mlp(proprio_embed_dim, vp_encoder_dims, activation_fn, estimator_latent_dim)
        self.vp_gru = nn.GRU(estimator_latent_dim, estimator_latent_dim, num_layers=1, batch_first=True)

        self.vp_velocity_head = self._build_mlp(
            estimator_latent_dim, vel_hidden_dim, resolve_nn_activation(vel_activation), vel_dim,
        )
        self.vp_feet_height_head = self._build_mlp(
            estimator_latent_dim, feet_height_hidden_dim, resolve_nn_activation(feet_height_activation), feet_height_dim,
        )
        self.vp_context_head = self._build_mlp(estimator_latent_dim, vp_head_hidden_dims, activation_fn, 2 * vp_context_dim)
        self.vp_terrain_latent_head = self._build_mlp(estimator_latent_dim, vp_head_hidden_dims, activation_fn, vp_terrain_latent_dim)
        self.terrain_output_dim, indices = self._resolve_terrain_output(
            terrain_scan_dim, terrain_recon_front_only, terrain_recon_grid_cols,
            terrain_recon_grid_rows, terrain_recon_x_min,
        )
        self.register_buffer("terrain_front_indices", indices)
        self.vp_terrain_decoder = self._build_mlp(
            vp_terrain_latent_dim, terrain_hidden_dim, resolve_nn_activation(terrain_activation), self.terrain_output_dim,
        )
        self.vp_next_proprio_decoder = self._build_mlp(vp_context_dim, vp_vae_decoder_dims, activation_fn, single_proprio_dim)
        self.actor = self._build_mlp(single_proprio_dim + self.vp_feature_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = self._build_mlp(num_critic_obs, critic_hidden_dims, activation_fn, 1)

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}.")
        self.distribution = None
        Normal.set_default_validate_args(False)
        self._last_current_proprio: Optional[torch.Tensor] = None
        self._last_vp_latent: Optional[torch.Tensor] = None
        self._last_vp_velocity: Optional[torch.Tensor] = None
        self._last_vp_feet_height: Optional[torch.Tensor] = None
        self._last_vp_context_mu: Optional[torch.Tensor] = None
        self._last_vp_context_logvar: Optional[torch.Tensor] = None
        self._last_vp_terrain_latent: Optional[torch.Tensor] = None
        print(f"G1Depth Actor MLP: {self.actor}")
        print(f"G1Depth Critic MLP: {self.critic}")

    def _build_mlp(self, input_dim, hidden_dims, activation_fn, output_dim):
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation_fn))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_terrain_output(terrain_scan_dim, front_only, grid_cols, grid_rows, x_min):
        if grid_cols * grid_rows != terrain_scan_dim:
            raise ValueError("Terrain target geometry does not match the 187-cell scan.")
        if not front_only:
            return terrain_scan_dim, None
        x_min_idx = max(0, int((x_min + (grid_cols - 1) * 0.1 / 2) / 0.1))
        indices = [y * grid_cols + x for y in range(grid_rows) for x in range(x_min_idx, grid_cols)]
        if not indices:
            raise ValueError("Terrain reconstruction region is empty.")
        return len(indices), torch.tensor(indices, dtype=torch.long)

    def _split_actor_obs(self, observations):
        if observations.ndim != 2 or observations.shape[1] != self.num_actor_obs:
            raise ValueError("g1_depth observations must have shape [batch, 1932].")
        proprio = observations[:, :self.proprio_actor_dim]
        return proprio, observations[:, self.proprio_actor_dim:], proprio[:, -self.single_proprio_dim:]

    def _encode_vp_latent(self, proprio_history, depth_flat):
        batch = proprio_history.shape[0]
        proprio = self.proprio_embedding(proprio_history.reshape(-1, self.single_proprio_dim))
        proprio = proprio.reshape(batch, self.history_len, self.proprio_embed_dim)
        depth = self.cnn(depth_flat.reshape(-1, 1, self.cnn_h, self.cnn_w))
        depth = depth.reshape(batch, self.depth_history_frames, self.visual_latent_dim)
        tokens = torch.cat((proprio, depth), dim=1)
        fused, _ = self.vp_attention(tokens, tokens, tokens, need_weights=False)
        encoded = self.vp_encoder(fused[:, :self.history_len].reshape(-1, self.proprio_embed_dim))
        _, hidden = self.vp_gru(encoded.reshape(batch, self.history_len, self.estimator_latent_dim))
        return hidden[-1]

    def _process_actor_obs(self, observations):
        proprio, depth, current = self._split_actor_obs(observations)
        latent = self._encode_vp_latent(proprio, depth)
        velocity = self.vp_velocity_head(latent)
        feet_height = self.vp_feet_height_head(latent)
        mu, logvar = self.vp_context_head(latent).split(self.vp_context_dim, dim=-1)
        terrain = self.vp_terrain_latent_head(latent)
        self._last_current_proprio, self._last_vp_latent = current, latent
        self._last_vp_velocity, self._last_vp_feet_height = velocity, feet_height
        self._last_vp_context_mu, self._last_vp_context_logvar = mu, logvar.clamp(-10.0, 10.0)
        self._last_vp_terrain_latent = terrain
        return torch.cat((current, velocity, feet_height, mu, terrain), dim=-1)

    def reset(self, dones=None):
        pass  # GRU state is recomputed from each complete observation window.

    def forward(self, observations):
        return self.act_inference(observations)

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

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        if self.noise_std_type == "scalar":
            std = self.std.clamp(1e-6, 1e3).expand_as(mean)
        else:
            std = self.log_std.clamp(-10.0, 10.0).exp().expand_as(mean)
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-3))
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(self._process_actor_obs(observations))

    def evaluate(self, critic_observations, actor_observations=None, **kwargs):
        return self.critic(critic_observations)

    def predict_velocity(self):
        return self._last_vp_velocity

    def predict_feet_height(self):
        return self._last_vp_feet_height

    def predict_terrain(self):
        return None if self._last_vp_terrain_latent is None else self.vp_terrain_decoder(self._last_vp_terrain_latent)

    @torch.jit.unused
    def predict_next_proprio(self):
        if self._last_vp_context_mu is None:
            return None
        mu, logvar = self._last_vp_context_mu, self._last_vp_context_logvar
        sampled = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.vp_next_proprio_decoder(sampled), mu, logvar

    def get_vp_encoder_contract(self):
        return {
            "version": 2, "output_type": "multi_head", "proprio_history_frames": self.history_len,
            "single_proprio_dim": self.single_proprio_dim, "depth_history_frames": self.depth_history_frames,
            "fusion_type": self.fusion_type, "sequence_order": "oldest_to_current",
            "recurrent_state": "reset_each_observation_window", "gru_hidden_dim": self.estimator_latent_dim,
            "feature_order": ["velocity", "feet_height", "context_mu", "terrain_latent"],
            "feature_dims": [3, 2, 32, 32], "feature_dim": 69,
            "context_distribution": "diagonal_gaussian", "actor_context": "posterior_mean",
            "terrain_decoder_input": "terrain_latent_only",
        }

    def get_critic_observation_contract(self):
        return {"version": 1, "single_frame_dim": 85, "history_frames": 10, "terrain_dim": 187}

    def get_network_contract(self):
        return {
            "version": 1, "task": "g1_depth", "actor_observation_dim": 1932,
            "critic_observation_dim": 1037, "action_dim": 23, "actor_input_dim": 147,
            "state_shapes": {name: list(value.shape) for name, value in self.state_dict().items()},
            "module_types": {name: type(module).__name__ for name, module in self.named_modules()},
        }
