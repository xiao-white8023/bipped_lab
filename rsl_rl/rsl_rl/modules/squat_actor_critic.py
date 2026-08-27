from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from .actor_critic_rough import ActorCritic


class SquatCENet(nn.Module):
    """DreamWaQ-style context encoder used by the squat policy."""

    def __init__(
        self,
        history_obs_dim: int,
        single_obs_dim: int,
        encoder_hidden_dims: list[int],
        com_position_dim: int,
        com_velocity_dim: int,
        com_position_scale: list[float],
        com_velocity_scale: list[float],
        latent_dim: int,
        decoder_hidden_dims: list[int],
        activation: str,
        logvar_min: float,
        logvar_max: float,
    ):
        super().__init__()
        if not encoder_hidden_dims:
            raise ValueError("cenet.encoder_hidden_dims must not be empty.")
        if not decoder_hidden_dims:
            raise ValueError("cenet.decoder_hidden_dims must not be empty.")
        if com_position_dim != 3 or com_velocity_dim != 3:
            raise ValueError(
                "g1_squart CENet requires three-dimensional COM position and "
                "velocity heads."
            )
        if len(com_position_scale) != 3 or any(
            scale <= 0.0 for scale in com_position_scale
        ):
            raise ValueError(
                "cenet.com_position_scale must contain three positive values."
            )
        if len(com_velocity_scale) != 3 or any(
            scale <= 0.0 for scale in com_velocity_scale
        ):
            raise ValueError(
                "cenet.com_velocity_scale must contain three positive values."
            )
        if logvar_min >= logvar_max:
            raise ValueError("cenet.logvar_min must be smaller than cenet.logvar_max.")

        self.history_obs_dim = history_obs_dim
        self.single_obs_dim = single_obs_dim
        self.com_position_dim = com_position_dim
        self.com_velocity_dim = com_velocity_dim
        self.latent_dim = latent_dim
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        activation_fn = resolve_nn_activation(activation)
        encoder_layers: list[nn.Module] = []
        input_dim = history_obs_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_fn))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.com_position_head = nn.Linear(input_dim, com_position_dim)
        self.com_velocity_head = nn.Linear(input_dim, com_velocity_dim)
        self.latent_mu_head = nn.Linear(input_dim, latent_dim)
        self.latent_logvar_head = nn.Linear(input_dim, latent_dim)
        self.register_buffer(
            "com_position_scale",
            torch.tensor(com_position_scale, dtype=torch.float),
        )
        self.register_buffer(
            "com_velocity_scale",
            torch.tensor(com_velocity_scale, dtype=torch.float),
        )

        decoder_layers: list[nn.Module] = []
        input_dim = com_position_dim + com_velocity_dim + latent_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_fn))
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, single_obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(
        self,
        observation_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not torch.jit.is_tracing()
            and observation_history.shape[-1] != self.history_obs_dim
        ):
            raise ValueError(
                "CENet history dimension mismatch: "
                f"expected {self.history_obs_dim}, got {observation_history.shape[-1]}."
            )
        features = self.encoder(observation_history)
        com_position = self.com_position_head(features)
        com_velocity = self.com_velocity_head(features)
        latent_mu = self.latent_mu_head(features)
        latent_logvar = self.latent_logvar_head(features).clamp(
            self.logvar_min, self.logvar_max
        )
        return com_position, com_velocity, latent_mu, latent_logvar

    @staticmethod
    def reparameterize(latent_mu: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * latent_logvar)
        return latent_mu + torch.randn_like(std) * std

    def decode(
        self,
        com_position: torch.Tensor,
        com_velocity: torch.Tensor,
        latent_mu: torch.Tensor,
        latent_logvar: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.reparameterize(latent_mu, latent_logvar)
        return self.decoder(
            torch.cat((com_position, com_velocity, latent), dim=-1)
        )


class SquatActorCritic(ActorCritic):
    """Switchable squat policy.

    With CENet disabled this class delegates to :class:`ActorCritic` without
    registering any additional parameters, preserving the legacy state dict.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        cenet: dict[str, Any] | None = None,
        **kwargs,
    ):
        cenet_cfg = dict(cenet or {})
        self.cenet_enabled = bool(cenet_cfg.pop("enabled", False))
        self.raw_num_actor_obs = int(num_actor_obs)

        if not self.cenet_enabled:
            super().__init__(
                num_actor_obs,
                num_critic_obs,
                num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                **kwargs,
            )
            return

        if "velocity_dim" in cenet_cfg:
            raise ValueError(
                "cenet.velocity_dim belongs to the obsolete scalar-v_z CENet "
                "configuration. Use the 3D com_position_dim/com_velocity_dim "
                "architecture and retrain the policy."
            )
        single_obs_dim = int(cenet_cfg.pop("single_obs_dim", 80))
        history_length = int(cenet_cfg.pop("history_length", 10))
        encoder_hidden_dims = list(cenet_cfg.pop("encoder_hidden_dims", [128, 64]))
        com_position_dim = int(cenet_cfg.pop("com_position_dim", 3))
        com_velocity_dim = int(cenet_cfg.pop("com_velocity_dim", 3))
        com_position_scale = list(
            cenet_cfg.pop("com_position_scale", [10.0, 10.0, 2.0])
        )
        com_velocity_scale = list(
            cenet_cfg.pop("com_velocity_scale", [2.0, 2.0, 2.0])
        )
        latent_dim = int(cenet_cfg.pop("latent_dim", 16))
        decoder_hidden_dims = list(cenet_cfg.pop("decoder_hidden_dims", [64, 128]))
        logvar_min = float(cenet_cfg.pop("logvar_min", -10.0))
        logvar_max = float(cenet_cfg.pop("logvar_max", 2.0))
        if cenet_cfg:
            raise ValueError(f"Unknown g1_squart CENet configuration keys: {sorted(cenet_cfg)}")

        expected_actor_obs = single_obs_dim * history_length
        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                "g1_squart actor observation dimension mismatch: "
                f"expected {history_length} x {single_obs_dim} = {expected_actor_obs}, got {num_actor_obs}."
            )

        actor_input_dim = (
            single_obs_dim
            + com_position_dim
            + com_velocity_dim
            + latent_dim
        )
        super().__init__(
            actor_input_dim,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        # The policy receives raw history observations even though its internal
        # actor MLP consumes only the compact CENet representation.
        self.num_actor_obs = num_actor_obs
        self.single_obs_dim = single_obs_dim
        self.history_length = history_length
        self.actor_input_dim = actor_input_dim
        self.com_position_dim = com_position_dim
        self.com_velocity_dim = com_velocity_dim
        self.latent_dim = latent_dim
        self.cenet_module = SquatCENet(
            history_obs_dim=num_actor_obs,
            single_obs_dim=single_obs_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            com_position_dim=com_position_dim,
            com_velocity_dim=com_velocity_dim,
            com_position_scale=com_position_scale,
            com_velocity_scale=com_velocity_scale,
            latent_dim=latent_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        self._last_cenet_outputs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

    def _process_obs(self, obs: torch.Tensor, is_actor: bool = True) -> torch.Tensor:
        if not is_actor or not self.cenet_enabled:
            return super()._process_obs(obs, is_actor=is_actor)
        if (
            not torch.jit.is_tracing()
            and obs.shape[-1] != self.raw_num_actor_obs
        ):
            raise ValueError(
                "g1_squart policy expected raw actor history with dimension "
                f"{self.raw_num_actor_obs}, got {obs.shape[-1]}."
            )

        com_position, com_velocity, latent_mu, latent_logvar = (
            self.cenet_module.encode(obs)
        )
        current_obs = obs[:, -self.single_obs_dim :]
        self._last_cenet_outputs = (
            com_position,
            com_velocity,
            latent_mu,
            latent_logvar,
        )
        return torch.cat(
            (current_obs, com_position, com_velocity, latent_mu), dim=-1
        )

    def get_cenet_auxiliary_outputs(self) -> dict[str, torch.Tensor]:
        if not self.cenet_enabled or self._last_cenet_outputs is None:
            raise RuntimeError("CENet outputs are unavailable; call policy.act() first with CENet enabled.")
        com_position, com_velocity, latent_mu, latent_logvar = (
            self._last_cenet_outputs
        )
        reconstruction = self.cenet_module.decode(
            com_position,
            com_velocity,
            latent_mu,
            latent_logvar,
        )
        return {
            "com_position": com_position,
            "com_velocity": com_velocity,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "reconstruction": reconstruction,
        }

    def cenet_encoder_parameters(self):
        if not self.cenet_enabled:
            return []
        return list(self.cenet_module.encoder.parameters())

    def load_state_dict(self, state_dict, strict=True):
        checkpoint_has_cenet = any(key.startswith("cenet_module.") for key in state_dict)
        checkpoint_has_legacy_velocity_head = any(
            key.startswith("cenet_module.velocity_head.") for key in state_dict
        )
        if self.cenet_enabled and checkpoint_has_legacy_velocity_head:
            raise RuntimeError(
                "This g1_squart checkpoint uses the obsolete scalar-v_z CENet "
                "architecture. The new policy predicts 3D support-frame COM "
                "position and velocity and must be trained from scratch."
            )
        if self.cenet_enabled != checkpoint_has_cenet:
            requested = "enabled" if self.cenet_enabled else "disabled"
            checkpoint = "CENet" if checkpoint_has_cenet else "legacy"
            raise RuntimeError(
                f"Cannot load a {checkpoint} g1_squart checkpoint while policy.cenet is {requested}. "
                "Use the matching policy.cenet.enabled setting."
            )
        return super().load_state_dict(state_dict, strict=strict)
