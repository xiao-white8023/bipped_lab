from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from .cnn_mlp import CnnMlp as CnnMlpModule


class RENetActorCritic(nn.Module):
    """Training-side RENet actor-critic.

    Actor observations are expected to be:
        proprio_history | depth_history_flat | estimator_mask

    estimator_mask follows the paper convention used in the environment:
        1 -> OP estimator, 0 -> VP estimator
    """

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
            raise ValueError("RENetActorCritic requires a CnnMlp config for the VP estimator.")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        self.single_proprio_dim = kwargs.pop("single_proprio_dim", 78) # 单个actor的维度78维
        self.estimator_mask_dim = kwargs.pop("estimator_mask_dim", 1) # 
        self.estimator_latent_dim = kwargs.pop("estimator_latent_dim", 64)
        self.proprio_embed_dim = kwargs.pop("proprio_embed_dim", 64)
        proprio_embed_dims = kwargs.pop("proprio_embed_dims", [256, 128])
        op_encoder_dims = kwargs.pop("op_encoder_dims", [128])
        vp_encoder_dims = kwargs.pop("vp_encoder_dims", [128])
        self.fusion_type = kwargs.pop("fusion_type", "attention")
        attention_num_heads = kwargs.pop("attention_num_heads", 1)

        self.use_vel_estimation = kwargs.pop("use_vel_estimation", True)
        vel_hidden_dim = kwargs.pop("vel_hidden_dim", [32])
        vel_activation_str = kwargs.pop("vel_activation", "elu")
        self.vel_dim = kwargs.pop("vel_dim", 3)

        self.use_terrain_recon = kwargs.pop("use_terrain_recon", False)
        terrain_hidden_dim = kwargs.pop("terrain_hidden_dim", [256, 128])
        terrain_activation_str = kwargs.pop("terrain_activation", "elu")
        terrain_scan_dim = kwargs.pop("terrain_scan_dim", 187)
        terrain_recon_front_only = kwargs.pop("terrain_recon_front_only", False)
        terrain_recon_grid_cols = kwargs.pop("terrain_recon_grid_cols", 17)
        terrain_recon_grid_rows = kwargs.pop("terrain_recon_grid_rows", 11)
        terrain_recon_x_min = kwargs.pop("terrain_recon_x_min", 0.0)

        self.use_feet_height_prediction = kwargs.pop("use_feet_height_prediction", False)
        feet_height_hidden_dim = kwargs.pop("feet_height_hidden_dim", [64])
        feet_height_activation_str = kwargs.pop("feet_height_activation", "elu")
        self.feet_height_dim = kwargs.pop("feet_height_dim", 2)

        # Accept legacy visual-policy keys so configs can evolve from g1_film cleanly.
        for key in (
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
            "his_encoder_dims",
            "his_latent_dim",
        ):
            kwargs.pop(key, None)
        if kwargs:
            print("RENetActorCritic ignored unexpected arguments: " + str(sorted(kwargs.keys())))

        activation_fn = resolve_nn_activation(activation)

        # In the env, depth history is flattened as H2 frames. Each frame is
        # embedded by the same lightweight CNN, matching the paper's image Emb.
        self.depth_history_frames = CnnMlp["input_channels"]
        self.cnn_h = CnnMlp["input_dim"][0]
        self.cnn_w = CnnMlp["input_dim"][1]
        self.depth_flat_dim = self.depth_history_frames * self.cnn_h * self.cnn_w
        self.proprio_actor_dim = num_actor_obs - self.depth_flat_dim - self.estimator_mask_dim # 本体感知观测
        if self.proprio_actor_dim <= 0:
            raise ValueError(
                "Invalid RENet actor observation layout: "
                f"num_actor_obs={num_actor_obs}, depth_flat_dim={self.depth_flat_dim}, "
                f"estimator_mask_dim={self.estimator_mask_dim}."
            )
        if self.proprio_actor_dim % self.single_proprio_dim != 0:
            raise ValueError(
                "proprio history dimension is not divisible by single_proprio_dim: "
                f"{self.proprio_actor_dim} vs {self.single_proprio_dim}."
            )

        self.history_len = self.proprio_actor_dim // self.single_proprio_dim

        self.proprio_embedding = self._build_mlp(
            self.single_proprio_dim, # 78
            proprio_embed_dims, # 
            activation_fn,
            self.proprio_embed_dim, # 64
        )

        cnn_cfg = dict(CnnMlp)
        cnn_cfg["input_channels"] = 1
        self.cnn = CnnMlpModule(**cnn_cfg)
        self.visual_latent_dim = self.cnn.output_dim

        if self.fusion_type not in ("mlp", "attention"):
            raise ValueError(f"Unsupported RENet fusion_type: {self.fusion_type}.")
        if self.fusion_type == "attention":
            if self.visual_latent_dim != self.proprio_embed_dim:
                raise ValueError(
                    "attention fusion expects visual_latent_dim == proprio_embed_dim, "
                    f"got {self.visual_latent_dim} and {self.proprio_embed_dim}."
                )
            self.op_attention = nn.MultiheadAttention(
                embed_dim=self.proprio_embed_dim,
                num_heads=attention_num_heads,
                batch_first=True,
            )
            self.vp_attention = nn.MultiheadAttention(
                embed_dim=self.proprio_embed_dim,
                num_heads=attention_num_heads,
                batch_first=True,
            )
            self.op_encoder = self._build_mlp(
                self.proprio_embed_dim,
                op_encoder_dims,
                activation_fn,
                self.estimator_latent_dim,
            )
            self.vp_encoder = self._build_mlp(
                self.proprio_embed_dim,
                vp_encoder_dims,
                activation_fn,
                self.estimator_latent_dim,
            )
        else:
            self.op_encoder = self._build_mlp(
                self.proprio_embed_dim,
                op_encoder_dims,
                activation_fn,
                self.estimator_latent_dim,
            )
            self.vp_encoder = self._build_mlp(
                self.proprio_embed_dim + self.visual_latent_dim,
                vp_encoder_dims,
                activation_fn,
                self.estimator_latent_dim,
            )

        self.op_gru = nn.GRU(
            input_size=self.estimator_latent_dim,
            hidden_size=self.estimator_latent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.vp_gru = nn.GRU(
            input_size=self.estimator_latent_dim,
            hidden_size=self.estimator_latent_dim,
            num_layers=1,
            batch_first=True,
        )

        vel_activation_fn = resolve_nn_activation(vel_activation_str)
        if self.use_vel_estimation:
            self.op_vel_estimator = self._build_mlp(
                self.estimator_latent_dim,
                vel_hidden_dim,
                vel_activation_fn,
                self.vel_dim,
            )
            self.vp_vel_estimator = self._build_mlp(
                self.estimator_latent_dim,
                vel_hidden_dim,
                vel_activation_fn,
                self.vel_dim,
            )

        if self.use_terrain_recon:
            self.terrain_output_dim, terrain_front_indices = self._resolve_terrain_output(
                terrain_scan_dim,
                terrain_recon_front_only,
                terrain_recon_grid_cols,
                terrain_recon_grid_rows,
                terrain_recon_x_min,
            )
            if terrain_front_indices is None:
                self.terrain_front_indices = None
            else:
                self.register_buffer("terrain_front_indices", terrain_front_indices)
            terrain_activation_fn = resolve_nn_activation(terrain_activation_str)
            self.terrain_decoder = self._build_mlp(
                self.single_proprio_dim + self.estimator_latent_dim,
                terrain_hidden_dim,
                terrain_activation_fn,
                self.terrain_output_dim,
            )

        if self.use_feet_height_prediction:
            feet_height_activation_fn = resolve_nn_activation(feet_height_activation_str)
            self.feet_height_decoder = self._build_mlp(
                self.single_proprio_dim + self.estimator_latent_dim,
                feet_height_hidden_dim,
                feet_height_activation_fn,
                self.feet_height_dim,
            )

        actor_input_dim = self.single_proprio_dim + 2 * self.estimator_latent_dim
        self.actor = self._build_actor(actor_input_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, activation_fn)

        print(f"RENet proprio embedding: {self.proprio_embedding}")
        print(f"RENet fusion type: {self.fusion_type}")
        print(f"RENet OP encoder: {self.op_encoder}")
        print(f"RENet VP encoder: {self.vp_encoder}")
        print(f"RENet Actor MLP: {self.actor}")
        print(f"RENet Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self._last_current_proprio: Optional[torch.Tensor] = None
        self._last_op_latent: Optional[torch.Tensor] = None
        self._last_vp_latent: Optional[torch.Tensor] = None
        self._last_estimator_mask: Optional[torch.Tensor] = None

    def _build_mlp(self, input_dim, hidden_dims, activation_fn, output_dim):
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_actor(self, input_dim, hidden_dims, activation_fn, num_actions):
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_actions))
        return nn.Sequential(*layers)

    def _build_critic(self, input_dim, hidden_dims, activation_fn):
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_terrain_output(
        terrain_scan_dim,
        front_only,
        grid_cols,
        grid_rows,
        x_min,
    ):
        if not front_only:
            return terrain_scan_dim, None
        resolution = 0.1
        size_x = (grid_cols - 1) * resolution
        x_min_idx = max(0, int((x_min + size_x / 2) / resolution))
        front_indices = []
        for y_idx in range(grid_rows):
            for x_idx in range(x_min_idx, grid_cols):
                front_indices.append(y_idx * grid_cols + x_idx)
        return len(front_indices), torch.tensor(front_indices, dtype=torch.long)

    def _split_actor_obs(self, observations: torch.Tensor):
        proprio_history = observations[:, : self.proprio_actor_dim]
        depth_start = self.proprio_actor_dim
        depth_end = depth_start + self.depth_flat_dim
        depth_flat = observations[:, depth_start:depth_end]
        if self.estimator_mask_dim > 0:
            estimator_mask = observations[:, depth_end : depth_end + self.estimator_mask_dim]
        else:
            estimator_mask = torch.zeros(observations.shape[0], 1, device=observations.device)
        estimator_mask = estimator_mask[:, :1].clamp(0.0, 1.0)
        current_proprio = proprio_history[:, -self.single_proprio_dim :]
        return proprio_history, depth_flat, estimator_mask, current_proprio

    def _embed_proprio_history(self, proprio_history: torch.Tensor):
        batch_size = proprio_history.shape[0]
        proprio_seq = proprio_history.view(batch_size, self.history_len, self.single_proprio_dim)
        proprio_embed = self.proprio_embedding(proprio_seq.reshape(batch_size * self.history_len, -1))
        return proprio_embed.view(batch_size, self.history_len, self.proprio_embed_dim)

    def _embed_depth_history(self, depth_flat: torch.Tensor):
        batch_size = depth_flat.shape[0]
        depth_seq = depth_flat.view(batch_size, self.depth_history_frames, self.cnn_h, self.cnn_w)
        depth_img = depth_seq.reshape(batch_size * self.depth_history_frames, 1, self.cnn_h, self.cnn_w)
        depth_embed = self.cnn(depth_img)
        return depth_embed.view(batch_size, self.depth_history_frames, self.visual_latent_dim)

    def _align_depth_to_proprio_history(self, depth_embed: torch.Tensor):
        batch_size = depth_embed.shape[0]
        depth_context = torch.zeros(
            batch_size,
            self.history_len,
            self.visual_latent_dim,
            dtype=depth_embed.dtype,
            device=depth_embed.device,
        )
        frames = min(self.depth_history_frames, self.history_len)
        depth_context[:, -frames:, :] = depth_embed[:, -frames:, :]
        return depth_context

    def _fuse_op_features(self, proprio_embed: torch.Tensor):
        batch_size = proprio_embed.shape[0]
        if self.fusion_type == "attention":
            fused_tokens, _ = self.op_attention(proprio_embed, proprio_embed, proprio_embed, need_weights=False)
            fused = fused_tokens.reshape(batch_size * self.history_len, self.proprio_embed_dim)
        else:
            fused = proprio_embed.reshape(batch_size * self.history_len, self.proprio_embed_dim)
        op_features = self.op_encoder(fused)
        return op_features.view(batch_size, self.history_len, self.estimator_latent_dim)

    def _fuse_vp_features(
        self,
        proprio_embed: torch.Tensor,
        depth_embed: torch.Tensor,
        depth_context: torch.Tensor,
    ):
        batch_size = proprio_embed.shape[0]
        if self.fusion_type == "attention":
            tokens = torch.cat([proprio_embed, depth_embed], dim=1)
            fused_tokens, _ = self.vp_attention(tokens, tokens, tokens, need_weights=False)
            fused = fused_tokens[:, : self.history_len, :]
            fused = fused.reshape(batch_size * self.history_len, self.proprio_embed_dim)
        else:
            fused = torch.cat([proprio_embed, depth_context], dim=-1)
            fused = fused.reshape(batch_size * self.history_len, self.proprio_embed_dim + self.visual_latent_dim)
        vp_features = self.vp_encoder(fused)
        return vp_features.view(batch_size, self.history_len, self.estimator_latent_dim)

    def _process_actor_obs(self, observations: torch.Tensor):
        proprio_history, depth_flat, estimator_mask, current_proprio = self._split_actor_obs(observations)

        proprio_embed = self._embed_proprio_history(proprio_history)
        depth_embed = self._embed_depth_history(depth_flat)
        depth_context = self._align_depth_to_proprio_history(depth_embed)

        op_features = self._fuse_op_features(proprio_embed)
        vp_features = self._fuse_vp_features(proprio_embed, depth_embed, depth_context)
        _, op_hidden = self.op_gru(op_features)
        _, vp_hidden = self.vp_gru(vp_features)
        op_latent = op_hidden[-1]
        vp_latent = vp_hidden[-1]

        renet_latent = torch.cat(
            [
                op_latent * estimator_mask,
                vp_latent * (1.0 - estimator_mask),
            ],
            dim=-1,
        )
        actor_input = torch.cat([current_proprio, renet_latent], dim=-1)

        self._last_current_proprio = current_proprio
        self._last_op_latent = op_latent
        self._last_vp_latent = vp_latent
        self._last_estimator_mask = estimator_mask
        return actor_input

    def reset(self, dones=None):
        pass

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
        actor_input = self._process_actor_obs(observations)
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

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_input = self._process_actor_obs(observations)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def predict_velocity(self):
        if not self.use_vel_estimation:
            return None
        if self._last_op_latent is None or self._last_vp_latent is None or self._last_estimator_mask is None:
            return None
        op_vel = self.op_vel_estimator(self._last_op_latent)
        vp_vel = self.vp_vel_estimator(self._last_vp_latent)
        active_vel = op_vel * self._last_estimator_mask + vp_vel * (1.0 - self._last_estimator_mask)
        return {"op": op_vel, "vp": vp_vel, "active": active_vel}

    def predict_terrain(self):
        if not self.use_terrain_recon:
            return None
        if self._last_current_proprio is None or self._last_vp_latent is None:
            return None
        terrain_input = torch.cat([self._last_current_proprio, self._last_vp_latent], dim=-1)
        return self.terrain_decoder(terrain_input)

    def predict_feet_height(self):
        if not self.use_feet_height_prediction:
            return None
        if self._last_current_proprio is None or self._last_vp_latent is None:
            return None
        feet_height_input = torch.cat([self._last_current_proprio, self._last_vp_latent], dim=-1)
        return self.feet_height_decoder(feet_height_input)
