from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils import resolve_nn_activation


class MLPHistoryEncoder(nn.Module):
    """HWC-Loco VAE history encoder."""

    def __init__(
        self,
        num_obs: int,
        num_history: int,
        num_latent: int,
        activation: str = "elu",
        adaptation_module_branch_hidden_dims: list[int] = [256, 128],
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        activation_fn = resolve_nn_activation(activation)
        input_size = num_obs * num_history
        layers = [nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]), activation_fn]
        for layer_index in range(len(adaptation_module_branch_hidden_dims)):
            if layer_index == len(adaptation_module_branch_hidden_dims) - 1:
                layers.append(nn.Linear(adaptation_module_branch_hidden_dims[layer_index], num_latent))
            else:
                layers.append(
                    nn.Linear(
                        adaptation_module_branch_hidden_dims[layer_index],
                        adaptation_module_branch_hidden_dims[layer_index + 1],
                    )
                )
                layers.append(activation_fn)
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor):
        batch_size = obs_history.shape[0]
        return self.encoder(obs_history.reshape(batch_size, -1))


class Estimator(nn.Module):
    """HWC-Loco VAE estimator for context latent and explicit policy labels."""

    def __init__(
        self,
        num_obs: int,
        num_history: int,
        num_latent: int,
        num_labels: int,
        activation: str = "elu",
        decoder_hidden_dims: list[int] = [256, 128, 64],
        encoder_hidden_dims: list[int] = [256, 128, 64],
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent
        self.num_labels = num_labels

        self.encoder = MLPHistoryEncoder(
            num_obs=num_obs,
            num_history=num_history + 1,
            num_latent=num_latent * 4,
            activation=activation,
            adaptation_module_branch_hidden_dims=encoder_hidden_dims,
        )
        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)
        self.label_mu = nn.Linear(num_latent * 4, num_labels)

        self.prior_mu = nn.Linear(num_obs * num_history, num_latent)
        self.prior_var = nn.Linear(num_obs * num_history, num_latent)

        activation_fn = resolve_nn_activation(activation)
        decoder_input_dim = num_latent + num_labels
        decoder_layers = [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]), activation_fn]
        for layer_index in range(len(decoder_hidden_dims)):
            if layer_index == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], num_obs))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_hidden_dims[layer_index + 1]))
                decoder_layers.append(activation_fn)
        self.decoder = nn.Sequential(*decoder_layers)

    def _reshape_history(self, obs_history):
        if obs_history.dim() == 2:
            return obs_history.reshape(-1, self.num_history, self.num_obs)
        return obs_history

    def _reshape_current(self, current_obs):
        if current_obs.dim() == 1:
            return current_obs.unsqueeze(0)
        return current_obs

    def _prepare_inputs(self, obs_history, current_obs=None):
        if current_obs is None and obs_history.dim() == 2 and obs_history.shape[1] == self.num_obs * (
            self.num_history + 1
        ):
            full_context = obs_history.reshape(-1, self.num_history + 1, self.num_obs)
            return full_context[:, : self.num_history, :], full_context[:, -1, :]

        obs_history = self._reshape_history(obs_history)
        if current_obs is None:
            current_obs = obs_history[:, -1, :]
        else:
            current_obs = self._reshape_current(current_obs)
        return obs_history, current_obs

    def encode(self, obs_history, current_obs=None):
        obs_history, current_obs = self._prepare_inputs(obs_history, current_obs)
        posterior_context = torch.cat((obs_history, current_obs.unsqueeze(1)), dim=1)
        encoded = self.encoder(posterior_context)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        label_mu = self.label_mu(encoded)
        return latent_mu, latent_var, label_mu

    def compute_prior(self, obs_history):
        obs_history, _ = self._prepare_inputs(obs_history, None)
        prior_input = obs_history.reshape(obs_history.shape[0], -1)
        return self.prior_mu(prior_input), self.prior_var(prior_input)

    def decode(self, z, labels):
        return self.decoder(torch.cat([z, labels], dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs_history, current_obs=None):
        latent_mu, latent_var, label_mu = self.encode(obs_history, current_obs)
        z = self.reparameterize(latent_mu, latent_var)
        prior_mu, prior_var = self.compute_prior(obs_history)
        return [z, label_mu], [latent_mu, latent_var, label_mu, prior_mu, prior_var]

    def loss_fn(self, obs_history, current_obs, future_obs, future_labels, dones=None, kld_weight=1.0):
        estimation, latent_params = self.forward(obs_history, current_obs)
        z, labels = estimation
        latent_mu, latent_var, label_mu, prior_mu, prior_var = latent_params

        recons = self.decode(z, labels)
        recons_loss = F.mse_loss(recons, future_obs, reduction="none").mean(-1)
        label_loss = F.mse_loss(label_mu, future_labels, reduction="none").mean(-1)
        kld_loss = 0.5 * torch.sum(
            prior_var - latent_var + (latent_var.exp() + (latent_mu - prior_mu).pow(2)) / prior_var.exp() - 1,
            dim=1,
        )

        loss = recons_loss + label_loss + kld_weight * kld_loss
        if dones is not None:
            valid_mask = (~dones.view(-1).bool()).float()
            valid_scale = valid_mask.numel() / valid_mask.sum().clamp(min=1.0)
            loss = loss * valid_mask * valid_scale
            recons_loss = recons_loss * valid_mask * valid_scale
            label_loss = label_loss * valid_mask * valid_scale
            kld_loss = kld_loss * valid_mask * valid_scale
        return {
            "loss": loss,
            "recons_loss": recons_loss,
            "label_loss": label_loss,
            "kld_loss": kld_loss,
        }

    def sample(self, obs_history, current_obs=None):
        estimation, _ = self.forward(obs_history, current_obs)
        return estimation

    def inference(self, obs_history, current_obs=None):
        _, latent_params = self.forward(obs_history, current_obs)
        latent_mu, _latent_var, label_mu, _prior_mu, _prior_var = latent_params
        return [latent_mu, label_mu]
