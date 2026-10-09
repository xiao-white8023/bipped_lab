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
        proprio_history | lidar_height_flat | depth_history_flat | estimator_mask

    estimator_mask follows the paper convention used in the environment:
        1 -> LP estimator, 0 -> VP estimator
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
        LidarCnnMlp=None,
        **kwargs,
    ):
        super().__init__()

        if CnnMlp is None:
            raise ValueError("RENetActorCritic requires a CnnMlp config for the VP estimator.")
        if LidarCnnMlp is None:
            raise ValueError("RENetActorCritic requires a LidarCnnMlp config for the LP estimator.")
        if LidarCnnMlp.get("input_channels", 1) != 1:
            raise ValueError("LP takes one single-channel height snapshot: input_channels=1")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        self.single_proprio_dim = kwargs.pop("single_proprio_dim", 78)
        self.estimator_mask_dim = kwargs.pop("estimator_mask_dim", 1)
        self.estimator_latent_dim = kwargs.pop("estimator_latent_dim", 64)
        # Keep the default exact legacy architecture for archived deployment.
        self.vp_output_type = kwargs.pop("vp_output_type", "legacy_latent")
        self.vp_context_dim = kwargs.pop("vp_context_dim", 32)
        self.vp_terrain_latent_dim = kwargs.pop("vp_terrain_latent_dim", 32)
        vp_head_hidden_dims = kwargs.pop("vp_head_hidden_dims", [64])
        vp_vae_decoder_dims = kwargs.pop("vp_vae_decoder_dims", [128, 256])
        if self.vp_output_type not in ("legacy_latent", "multi_head"):
            raise ValueError("vp_output_type must be 'legacy_latent' or 'multi_head'.")
        if any(type(dim) is not int or dim < 1 for dim in (self.vp_context_dim, self.vp_terrain_latent_dim)):
            raise ValueError("VP context and terrain latent dimensions must be positive integers.")
        self.use_vp_vae = self.vp_output_type == "multi_head"
        self.proprio_embed_dim = kwargs.pop("proprio_embed_dim", 64)
        proprio_embed_dims = kwargs.pop("proprio_embed_dims", [256, 128])
        kwargs.pop("lp_encoder_dims", None)  # Accepted for archived cfg compatibility.
        self.lp_height_history_frames = kwargs.pop("lp_height_history_frames", 1)
        if self.lp_height_history_frames != 1:
            raise ValueError("LP height input is a single snapshot; packet delay is handled before the policy")
        vp_encoder_dims = kwargs.pop("vp_encoder_dims", [128])
        self.fusion_type = kwargs.pop("fusion_type", "attention")
        attention_num_heads = kwargs.pop("attention_num_heads", 1)
        self.lp_encoder_type = kwargs.pop(
            "lp_encoder_type", "ame_point_cross_attention"
        )
        self.ame_lp_attention_num_heads = kwargs.pop("ame_lp_attention_num_heads", 16)
        self.ame_lp_query_history_frames = kwargs.pop("ame_lp_query_history_frames", 5)
        vp_proprio_history_frames = kwargs.pop("vp_proprio_history_frames", None)
        self.single_critic_dim = kwargs.pop("single_critic_dim", 85)
        critic_history_frames = kwargs.pop("critic_history_frames", None)
        vp_critic_history_frames = kwargs.pop("vp_critic_history_frames", None)

        self.use_vel_estimation = kwargs.pop("use_vel_estimation", True)
        vel_hidden_dim = kwargs.pop("vel_hidden_dim", [32])
        vel_activation_str = kwargs.pop("vel_activation", "elu")
        self.vel_dim = kwargs.pop("vel_dim", 3)

        self.use_terrain_recon = kwargs.pop("use_terrain_recon", False)
        terrain_hidden_dim = kwargs.pop("terrain_hidden_dim", [256, 128])
        lp_terrain_hidden_dim = kwargs.pop("lp_terrain_hidden_dim", [256, 128])
        terrain_activation_str = kwargs.pop("terrain_activation", "elu")
        terrain_scan_dim = kwargs.pop("terrain_scan_dim", 187)
        self.critic_terrain_dim = terrain_scan_dim
        terrain_recon_front_only = kwargs.pop("terrain_recon_front_only", False)
        lp_terrain_recon_front_only = kwargs.pop("lp_terrain_recon_front_only", False)
        if lp_terrain_recon_front_only:
            raise ValueError("pgtt_full_atten reconstructs the full 187-cell LP terrain target.")
        terrain_recon_grid_cols = kwargs.pop("terrain_recon_grid_cols", 17)
        terrain_recon_grid_rows = kwargs.pop("terrain_recon_grid_rows", 11)
        terrain_recon_x_min = kwargs.pop("terrain_recon_x_min", 0.0)

        self.use_feet_height_prediction = kwargs.pop("use_feet_height_prediction", False)
        self.use_lp_feet_height_prediction = kwargs.pop("use_lp_feet_height_prediction", False)
        feet_height_hidden_dim = kwargs.pop("feet_height_hidden_dim", [64])
        feet_height_activation_str = kwargs.pop("feet_height_activation", "elu")
        self.feet_height_dim = kwargs.pop("feet_height_dim", 2)
        if self.use_vp_vae and not (self.use_vel_estimation and self.use_terrain_recon and self.use_feet_height_prediction):
            raise ValueError("The four-head VP requires velocity, feet-height and terrain prediction.")
        self.vp_feature_dim = (
            self.vel_dim + self.feet_height_dim + self.vp_context_dim + self.vp_terrain_latent_dim
            if self.use_vp_vae else self.estimator_latent_dim
        )

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
        self.lidar_h = LidarCnnMlp["input_dim"][0]
        self.lidar_w = LidarCnnMlp["input_dim"][1]
        self.lidar_flat_dim = self.lp_height_history_frames * self.lidar_h * self.lidar_w
        self.proprio_actor_dim = (
            num_actor_obs - self.lidar_flat_dim - self.depth_flat_dim - self.estimator_mask_dim
        )
        if self.proprio_actor_dim <= 0:
            raise ValueError(
                "Invalid RENet actor observation layout: "
                f"num_actor_obs={num_actor_obs}, lidar_flat_dim={self.lidar_flat_dim}, "
                f"depth_flat_dim={self.depth_flat_dim}, "
                f"estimator_mask_dim={self.estimator_mask_dim}."
            )
        if self.proprio_actor_dim % self.single_proprio_dim != 0:
            raise ValueError(
                "proprio history dimension is not divisible by single_proprio_dim: "
                f"{self.proprio_actor_dim} vs {self.single_proprio_dim}."
            )

        self.history_len = self.proprio_actor_dim // self.single_proprio_dim
        self.vp_proprio_history_frames = self.history_len if vp_proprio_history_frames is None else vp_proprio_history_frames
        if (isinstance(self.vp_proprio_history_frames, bool)
                or not isinstance(self.vp_proprio_history_frames, int)
                or not 1 <= self.vp_proprio_history_frames <= self.history_len):
            raise ValueError("vp_proprio_history_frames must be an integer in [1, actor history length].")

        # Mixed LP/VP batches keep the full LP layout. VP value estimation must
        # ignore the older frames, just as the VP actor/auxiliary encoder does.
        critic_history_dim = num_critic_obs - self.critic_terrain_dim
        inferred_critic_frames = (
            critic_history_dim // self.single_critic_dim
            if critic_history_dim > 0 and critic_history_dim % self.single_critic_dim == 0
            else None
        )
        self.critic_history_frames = (
            inferred_critic_frames if critic_history_frames is None else critic_history_frames
        )
        self.vp_critic_history_frames = (
            self.critic_history_frames if vp_critic_history_frames is None else vp_critic_history_frames
        )
        if critic_history_frames is not None or vp_critic_history_frames is not None:
            if (type(self.critic_history_frames) is not int
                    or self.critic_history_frames < 1
                    or self.critic_history_frames != inferred_critic_frames):
                raise ValueError("critic_history_frames does not match the critic observation layout.")
            if (type(self.vp_critic_history_frames) is not int
                    or not 1 <= self.vp_critic_history_frames <= self.critic_history_frames):
                raise ValueError("vp_critic_history_frames must be an integer in [1, critic history length].")

        self.proprio_embedding = self._build_mlp(
            self.single_proprio_dim,
            proprio_embed_dims,
            activation_fn,
            self.proprio_embed_dim,
        )

        cnn_cfg = dict(CnnMlp)
        cnn_cfg["input_channels"] = 1
        self.cnn = CnnMlpModule(**cnn_cfg)
        self.visual_latent_dim = self.cnn.output_dim

        if self.fusion_type not in ("mlp", "attention"):
            raise ValueError(f"Unsupported RENet fusion_type: {self.fusion_type}.")
        if self.lp_encoder_type != "ame_point_cross_attention":
            raise ValueError("g1_renet requires lp_encoder_type='ame_point_cross_attention'.")
        self.uses_ame_lp = True
        self.lp_frame_embedding = None
        if self.fusion_type != "attention":
            raise ValueError("AME LP requires fusion_type='attention'.")
        if (self.lidar_h, self.lidar_w) != (11, 17):
            raise ValueError(
                "AME LP requires the pgtt_full_atten 11x17 map, got "
                f"{(self.lidar_h, self.lidar_w)}."
            )
        if self.history_len < self.ame_lp_query_history_frames:
            raise ValueError(
                "AME LP query history cannot exceed the actor proprio history: "
                f"{self.ame_lp_query_history_frames} > {self.history_len}."
            )
        if self.proprio_embed_dim != 64 or self.estimator_latent_dim != 64:
            raise ValueError("AME LP requires 64-dimensional query, tokens, and latent.")
        if self.ame_lp_attention_num_heads != 16:
            raise ValueError("The pgtt_full_atten contract requires exactly 16 attention heads.")

        # Preserve the archived AME module names and registration order so
        # model_25000.pt (including Adam state) can be migrated strictly.
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 61, kernel_size=5, padding=2),
            nn.ELU(),
        )
        self.lidar_latent_dim = 64
        x = torch.linspace(-0.8, 0.8, self.lidar_w)
        y = torch.linspace(-0.5, 0.5, self.lidar_h)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("lp_grid_xy", torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.lp_query_mlp = self._build_mlp(
            self.ame_lp_query_history_frames * self.single_proprio_dim,
            [256, 128],
            activation_fn,
            self.proprio_embed_dim,
        )
        self.lp_attention = nn.MultiheadAttention(
            embed_dim=self.proprio_embed_dim,
            num_heads=self.ame_lp_attention_num_heads,
            batch_first=True,
        )
        self.lp_encoder = None
        if self.visual_latent_dim != self.proprio_embed_dim:
            raise ValueError(
                "attention fusion expects visual_latent_dim == proprio_embed_dim, "
                f"got {self.visual_latent_dim} and {self.proprio_embed_dim}."
            )
        self.vp_attention = nn.MultiheadAttention(
            embed_dim=self.proprio_embed_dim,
            num_heads=attention_num_heads,
            batch_first=True,
        )
        self.vp_encoder = self._build_mlp(
            self.proprio_embed_dim, vp_encoder_dims, activation_fn, self.estimator_latent_dim,
        )
        self.lp_gru = None
        self.vp_gru = nn.GRU(
            input_size=self.estimator_latent_dim,
            hidden_size=self.estimator_latent_dim,
            num_layers=1,
            batch_first=True,
        )

        vel_activation_fn = resolve_nn_activation(vel_activation_str)
        if self.use_vel_estimation:
            self.lp_vel_estimator = self._build_mlp(
                self.estimator_latent_dim,
                vel_hidden_dim,
                vel_activation_fn,
                self.vel_dim,
            )
            velocity_head = self._build_mlp(
                self.estimator_latent_dim,
                vel_hidden_dim,
                vel_activation_fn,
                self.vel_dim,
            )
            if self.use_vp_vae:
                self.vp_velocity_head = velocity_head
            else:
                self.vp_vel_estimator = velocity_head

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
            terrain_decoder = self._build_mlp(
                self.vp_terrain_latent_dim if self.use_vp_vae else self.single_proprio_dim + self.estimator_latent_dim,
                terrain_hidden_dim,
                terrain_activation_fn,
                self.terrain_output_dim,
            )
            if self.use_vp_vae:
                self.vp_terrain_decoder = terrain_decoder
            else:
                self.terrain_decoder = terrain_decoder
            self.lp_terrain_output_dim, lp_indices = self._resolve_terrain_output(
                terrain_scan_dim, lp_terrain_recon_front_only,
                terrain_recon_grid_cols, terrain_recon_grid_rows, terrain_recon_x_min)
            self.register_buffer("lp_terrain_front_indices", lp_indices)
            self.lp_terrain_decoder = self._build_mlp(
                self.estimator_latent_dim,
                lp_terrain_hidden_dim,
                terrain_activation_fn,
                self.lp_terrain_output_dim,
            )

        if self.use_feet_height_prediction:
            feet_height_activation_fn = resolve_nn_activation(feet_height_activation_str)
            feet_height_head = self._build_mlp(
                self.estimator_latent_dim if self.use_vp_vae else self.single_proprio_dim + self.estimator_latent_dim,
                feet_height_hidden_dim,
                feet_height_activation_fn,
                self.feet_height_dim,
            )
            if self.use_vp_vae:
                self.vp_feet_height_head = feet_height_head
            else:
                self.feet_height_decoder = feet_height_head

        actor_input_dim = self.single_proprio_dim + self.estimator_latent_dim + self.vp_feature_dim
        self.actor = self._build_actor(actor_input_dim, actor_hidden_dims, activation_fn, num_actions)
        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, activation_fn)

        print(f"RENet proprio embedding: {self.proprio_embedding}")
        print(f"RENet fusion type: {self.fusion_type}, LP encoder: {self.lp_encoder_type}")
        print(f"RENet LP height encoder: {self.lidar_cnn}")
        print(f"RENet LP encoder: {self.lp_encoder}")
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
        self._last_lp_query: Optional[torch.Tensor] = None
        self._last_lp_latent: Optional[torch.Tensor] = None
        self._last_vp_latent: Optional[torch.Tensor] = None
        self._last_estimator_mask: Optional[torch.Tensor] = None
        self._last_vp_feature: Optional[torch.Tensor] = None
        self._last_vp_velocity: Optional[torch.Tensor] = None
        self._last_vp_feet_height: Optional[torch.Tensor] = None
        self._last_vp_context_mu: Optional[torch.Tensor] = None
        self._last_vp_context_logvar: Optional[torch.Tensor] = None
        self._last_vp_terrain_latent: Optional[torch.Tensor] = None

        # Legacy mode preserves module registration order for archived Adam state.
        if self.use_lp_feet_height_prediction:
            self.lp_feet_height_decoder = self._build_mlp(
                self.single_proprio_dim + self.estimator_latent_dim,
                feet_height_hidden_dim,
                resolve_nn_activation(feet_height_activation_str),
                self.feet_height_dim,
            )
        if self.use_vp_vae:
            # One logical z_o head supplies posterior mean and log variance.
            self.vp_context_head = self._build_mlp(
                self.estimator_latent_dim, vp_head_hidden_dims, activation_fn, 2 * self.vp_context_dim,
            )
            self.vp_terrain_latent_head = self._build_mlp(
                self.estimator_latent_dim, vp_head_hidden_dims, activation_fn, self.vp_terrain_latent_dim,
            )
            self.vp_next_proprio_decoder = self._build_mlp(
                self.vp_context_dim, vp_vae_decoder_dims, activation_fn, self.single_proprio_dim,
            )

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
        exporting = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if not exporting and (
            observations.ndim != 2 or observations.shape[1] != self.num_actor_obs
        ):
            raise ValueError(
                "RENet actor observations must have shape "
                f"(batch, {self.num_actor_obs}), got {tuple(observations.shape)}."
            )
        proprio_history = observations[:, : self.proprio_actor_dim]
        lidar_start = self.proprio_actor_dim
        lidar_end = lidar_start + self.lidar_flat_dim
        lidar_flat = observations[:, lidar_start:lidar_end]
        depth_start = lidar_end
        depth_end = depth_start + self.depth_flat_dim
        depth_flat = observations[:, depth_start:depth_end]
        if self.estimator_mask_dim > 0:
            estimator_mask = observations[:, depth_end : depth_end + self.estimator_mask_dim]
        else:
            estimator_mask = torch.zeros(observations.shape[0], 1, device=observations.device)
        estimator_mask = estimator_mask[:, :1].clamp(0.0, 1.0)
        current_proprio = proprio_history[:, -self.single_proprio_dim :]
        return proprio_history, lidar_flat, depth_flat, estimator_mask, current_proprio

    def _embed_proprio_history(self, proprio_history: torch.Tensor):
        batch_size = proprio_history.shape[0]
        frames = proprio_history.shape[1] // self.single_proprio_dim
        proprio_seq = proprio_history.reshape(batch_size, frames, self.single_proprio_dim)
        proprio_embed = self.proprio_embedding(proprio_seq.reshape(batch_size * frames, -1))
        return proprio_embed.view(batch_size, frames, self.proprio_embed_dim)

    def _embed_depth_history(self, depth_flat: torch.Tensor):
        batch_size = depth_flat.shape[0]
        depth_seq = depth_flat.view(batch_size, self.depth_history_frames, self.cnn_h, self.cnn_w)
        depth_img = depth_seq.reshape(batch_size * self.depth_history_frames, 1, self.cnn_h, self.cnn_w)
        depth_embed = self.cnn(depth_img)
        return depth_embed.view(batch_size, self.depth_history_frames, self.visual_latent_dim)

    def _embed_lidar_height(self, lidar_flat: torch.Tensor):
        return self._embed_ame_lidar_tokens(lidar_flat)

    def _encode_ame_lp_query(self, proprio_history: torch.Tensor):
        batch_size = proprio_history.shape[0]
        proprio_seq = proprio_history.view(
            batch_size, self.history_len, self.single_proprio_dim
        )
        query_history = proprio_seq[:, -self.ame_lp_query_history_frames :]
        query = self.lp_query_mlp(query_history.reshape(batch_size, -1))
        return query.unsqueeze(1)

    def _embed_ame_lidar_tokens(self, lidar_flat: torch.Tensor):
        batch_size = lidar_flat.shape[0]
        lidar_height = lidar_flat.reshape(batch_size, 1, self.lidar_h, self.lidar_w)
        spatial = self.lidar_cnn(lidar_height)
        spatial = spatial.permute(0, 2, 3, 1).reshape(batch_size, -1, 61)
        coordinates = self.lp_grid_xy.to(dtype=lidar_flat.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        encoded_height = lidar_flat.reshape(batch_size, -1, 1)
        return torch.cat((spatial, coordinates, encoded_height), dim=-1)

    @torch.jit.unused
    def get_lp_attention_weights(
        self, proprio_history: torch.Tensor, lidar_flat: torch.Tensor
    ) -> torch.Tensor:
        """Return per-head AME weights for explicit offline diagnostics only."""
        query = self._encode_ame_lp_query(proprio_history)
        tokens = self._embed_ame_lidar_tokens(lidar_flat)
        _, weights = self.lp_attention(
            query,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        return weights

    def _align_depth_to_proprio_history(self, depth_embed: torch.Tensor):
        batch_size = depth_embed.shape[0]
        depth_context = torch.zeros(
            batch_size,
            self.vp_proprio_history_frames,
            self.visual_latent_dim,
            dtype=depth_embed.dtype,
            device=depth_embed.device,
        )
        frames = min(self.depth_history_frames, self.vp_proprio_history_frames)
        depth_context[:, -frames:, :] = depth_embed[:, -frames:, :]
        return depth_context

    def _encode_lp_latent(
        self,
        proprio_history: torch.Tensor,
        lidar_flat: torch.Tensor,
        proprio_embed: Optional[torch.Tensor] = None,
    ):
        query = self._encode_ame_lp_query(proprio_history)
        lidar_tokens = self._embed_ame_lidar_tokens(lidar_flat)
        lp_latent, _ = self.lp_attention(query, lidar_tokens, lidar_tokens, need_weights=False)
        self._last_lp_query = query[:, 0]
        return lp_latent[:, 0]

    def _encode_vp_latent(
        self,
        proprio_history: torch.Tensor,
        depth_flat: torch.Tensor,
        proprio_embed: Optional[torch.Tensor] = None,
    ):
        if proprio_embed is None:
            recent_history = proprio_history[:, -self.vp_proprio_history_frames * self.single_proprio_dim :]
            proprio_embed = self._embed_proprio_history(recent_history)
        else:
            proprio_embed = proprio_embed[:, -self.vp_proprio_history_frames :]
        depth_embed = self._embed_depth_history(depth_flat)
        depth_context = self._align_depth_to_proprio_history(depth_embed)
        vp_features = self._fuse_vp_features(proprio_embed, depth_embed, depth_context)
        _, vp_hidden = self.vp_gru(vp_features)
        return vp_hidden[-1]

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
            fused = fused_tokens[:, : self.vp_proprio_history_frames, :]
            fused = fused.reshape(batch_size * self.vp_proprio_history_frames, self.proprio_embed_dim)
        else:
            fused = torch.cat([proprio_embed, depth_context], dim=-1)
            fused = fused.reshape(batch_size * self.vp_proprio_history_frames, self.proprio_embed_dim + self.visual_latent_dim)
        vp_features = self.vp_encoder(fused)
        return vp_features.view(batch_size, self.vp_proprio_history_frames, self.estimator_latent_dim)

    def _process_actor_obs(self, observations: torch.Tensor):
        proprio_history, lidar_flat, depth_flat, estimator_mask, current_proprio = self._split_actor_obs(
            observations
        )

        # AME LP embeds its five-frame query; VP embeds its configured window.
        proprio_embed = None
        lp_latent = self._encode_lp_latent(
            proprio_history, lidar_flat, proprio_embed=proprio_embed
        )
        vp_latent = self._encode_vp_latent(
            proprio_history, depth_flat, proprio_embed=proprio_embed
        )

        vp_feature = vp_latent
        if self.use_vp_vae:
            vp_velocity = self.vp_velocity_head(vp_latent)
            vp_feet_height = self.vp_feet_height_head(vp_latent)
            context_parameters = self.vp_context_head(vp_latent)
            context_mu, context_logvar = context_parameters.split(self.vp_context_dim, dim=-1)
            context_logvar = context_logvar.clamp(-10.0, 10.0)
            terrain_latent = self.vp_terrain_latent_head(vp_latent)
            # Sampling is restricted to the VAE loss. PPO and deployment both
            # use mu, so re-evaluating an action does not resample its context.
            vp_feature = torch.cat((vp_velocity, vp_feet_height, context_mu, terrain_latent), dim=-1)
            self._last_vp_velocity = vp_velocity
            self._last_vp_feet_height = vp_feet_height
            self._last_vp_context_mu = context_mu
            self._last_vp_context_logvar = context_logvar
            self._last_vp_terrain_latent = terrain_latent

        renet_latent = torch.cat(
            [
                lp_latent * estimator_mask,
                vp_feature * (1.0 - estimator_mask),
            ],
            dim=-1,
        )
        actor_input = torch.cat([current_proprio, renet_latent], dim=-1)

        self._last_current_proprio = current_proprio
        self._last_lp_latent = lp_latent
        self._last_vp_latent = vp_latent
        self._last_vp_feature = vp_feature
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

    def evaluate(self, critic_observations, actor_observations=None, **kwargs):
        if self.vp_critic_history_frames == self.critic_history_frames:
            return self.critic(critic_observations)
        if (actor_observations is None or actor_observations.ndim != 2
                or actor_observations.shape != (critic_observations.shape[0], self.num_actor_obs)):
            raise ValueError("VP critic history selection requires the matching actor observations.")
        if critic_observations.ndim != 2 or critic_observations.shape[1] != self.num_critic_obs:
            raise ValueError("RENet critic observations have an incompatible shape.")
        older_dim = (self.critic_history_frames - self.vp_critic_history_frames) * self.single_critic_dim
        lp_mode = actor_observations[:, -1:] >= 0.5
        older_history = torch.where(lp_mode, critic_observations[:, :older_dim], 0.0)
        selected = torch.cat((older_history, critic_observations[:, older_dim:]), dim=-1)
        return self.critic(selected)

    def get_critic_observation_contract(self):
        if self.critic_history_frames is None:
            return None
        return {
            "version": 1,
            "single_frame_dim": self.single_critic_dim,
            "history_frames": self.critic_history_frames,
            "vp_history_frames": self.vp_critic_history_frames,
            "terrain_dim": self.critic_terrain_dim,
            "mode_source": "matching_actor_estimator_mask",
        }

    def predict_velocity(self):
        if not self.use_vel_estimation:
            return None
        if self._last_lp_latent is None or self._last_vp_latent is None or self._last_estimator_mask is None:
            return None
        lp_velocity_input = self._last_lp_query
        if lp_velocity_input is None:
            return None
        lp_vel = self.lp_vel_estimator(lp_velocity_input)
        vp_vel = self._last_vp_velocity if self.use_vp_vae else self.vp_vel_estimator(self._last_vp_latent)
        active_vel = lp_vel * self._last_estimator_mask + vp_vel * (1.0 - self._last_estimator_mask)
        return {"lp": lp_vel, "vp": vp_vel, "active": active_vel}

    def predict_lp_terrain(self):
        if not self.use_terrain_recon:
            return None
        if self._last_current_proprio is None or self._last_lp_latent is None:
            return None
        terrain_input = self._last_lp_latent
        return self.lp_terrain_decoder(terrain_input)

    @torch.jit.unused
    def get_lp_encoder_contract(self):
        return {
            "version": 1,
            "encoder_type": "ame_point_cross_attention",
            "proprio_history_frames": self.ame_lp_query_history_frames,
            "height_history_frames": self.lp_height_history_frames,
            "map_shape": [self.lidar_h, self.lidar_w],
            "map_token_dim": self.proprio_embed_dim,
            "attention_heads": self.ame_lp_attention_num_heads,
            "sample_order": "y_ascending/x_ascending",
            "grid_x_range": [-0.8, 0.8],
            "grid_y_range": [-0.5, 0.5],
            "coordinate_features": "x_y_encoded_height",
            "uses_lp_gru": False,
            "reconstruction_region": "full_187",
        }

    @torch.jit.unused
    def get_vp_encoder_contract(self):
        contract = {
            "version": 1,
            "proprio_history_frames": self.vp_proprio_history_frames,
            "single_proprio_dim": self.single_proprio_dim,
            "depth_history_frames": self.depth_history_frames,
            "fusion_type": self.fusion_type,
            "sequence_order": "oldest_to_current",
            "recurrent_state": "reset_each_observation_window",
        }
        if self.use_vp_vae:
            contract.update({
                "version": 2,
                "output_type": self.vp_output_type,
                "gru_hidden_dim": self.estimator_latent_dim,
                "feature_order": ["velocity", "feet_height", "context_mu", "terrain_latent"],
                "feature_dims": [self.vel_dim, self.feet_height_dim, self.vp_context_dim, self.vp_terrain_latent_dim],
                "feature_dim": self.vp_feature_dim,
                "context_distribution": "diagonal_gaussian",
                "actor_context": "posterior_mean",
                "terrain_decoder_input": "terrain_latent_only",
            })
        return contract

    def predict_terrain(self):
        if not self.use_terrain_recon:
            return None
        if self._last_current_proprio is None or self._last_vp_latent is None:
            return None
        terrain_input = (self._last_vp_terrain_latent if self.use_vp_vae else
                         torch.cat([self._last_current_proprio, self._last_vp_latent], dim=-1))
        if self.use_vp_vae:
            return self.vp_terrain_decoder(terrain_input)
        return self.terrain_decoder(terrain_input)

    def predict_feet_height(self):
        if not self.use_feet_height_prediction:
            return None
        if self._last_current_proprio is None or self._last_vp_latent is None:
            return None
        if self.use_vp_vae:
            return self._last_vp_feet_height
        feet_height_input = torch.cat([self._last_current_proprio, self._last_vp_latent], dim=-1)
        return self.feet_height_decoder(feet_height_input)

    @torch.jit.unused
    def predict_next_proprio(self):
        """Training-only stochastic reconstruction from q(z_o | history)."""
        if not self.use_vp_vae or self._last_vp_context_mu is None or self._last_vp_context_logvar is None:
            return None
        mu, logvar = self._last_vp_context_mu, self._last_vp_context_logvar
        sampled_context = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.vp_next_proprio_decoder(sampled_context), mu, logvar

    def predict_lp_feet_height(self):
        if not self.use_lp_feet_height_prediction:
            return None
        if self._last_current_proprio is None or self._last_lp_latent is None:
            return None
        features = torch.cat([self._last_current_proprio, self._last_lp_latent], dim=-1)
        return self.lp_feet_height_decoder(features)
