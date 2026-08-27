from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.storage import SquatRolloutStorage

from .ppo import PPO


class SquatPPO(PPO):
    """PPO with an optional DreamWaQ-style CENet objective for g1_squart."""

    def __init__(
        self,
        policy,
        com_position_loss_coef: float = 1.0,
        com_velocity_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 0.5,
        reconstruction_warmup_iters: int = 500,
        kl_loss_coef: float = 0.01,
        kl_delay_iters: int = 500,
        kl_warmup_end_iter: int = 2000,
        critic_single_obs_dim: int = 106,
        critic_history_length: int = 10,
        com_position_target_index: int = 86,
        com_velocity_target_index: int = 89,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.cenet_enabled = bool(getattr(policy, "cenet_enabled", False))
        self.current_iteration = 0

        self.com_position_loss_coef = float(com_position_loss_coef)
        self.com_velocity_loss_coef = float(com_velocity_loss_coef)
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)
        self.reconstruction_warmup_iters = int(reconstruction_warmup_iters)
        self.kl_loss_coef = float(kl_loss_coef)
        self.kl_delay_iters = int(kl_delay_iters)
        self.kl_warmup_end_iter = int(kl_warmup_end_iter)
        self.critic_single_obs_dim = int(critic_single_obs_dim)
        self.critic_history_length = int(critic_history_length)
        self.com_position_target_index = int(com_position_target_index)
        self.com_velocity_target_index = int(com_velocity_target_index)
        self._warned_missing_reconstruction_target = False

        if self.cenet_enabled:
            if self.rnd is not None or self.symmetry is not None:
                raise ValueError(
                    "g1_squart CENet currently requires rnd_cfg=None and "
                    "symmetry_cfg=None."
                )
            if self.reconstruction_warmup_iters < 0:
                raise ValueError("reconstruction_warmup_iters must be non-negative.")
            if (
                self.com_position_loss_coef < 0.0
                or self.com_velocity_loss_coef < 0.0
            ):
                raise ValueError("CENet COM loss coefficients must be non-negative.")
            if (
                self.kl_delay_iters < 0
                or self.kl_warmup_end_iter <= self.kl_delay_iters
            ):
                raise ValueError(
                    "kl_warmup_end_iter must be greater than the non-negative "
                    "kl_delay_iters."
                )
            if not (
                0
                <= self.com_position_target_index
                <= self.critic_single_obs_dim - 3
            ):
                raise ValueError(
                    "com_position_target_index must index three values in one "
                    "critic observation."
                )
            if not (
                0
                <= self.com_velocity_target_index
                <= self.critic_single_obs_dim - 3
            ):
                raise ValueError(
                    "com_velocity_target_index must index three values in one "
                    "critic observation."
                )
            self.transition = SquatRolloutStorage.Transition()

    def set_training_iteration(self, iteration: int):
        if int(iteration) < 0:
            raise ValueError("training iteration must be non-negative.")
        self.current_iteration = int(iteration)

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        if not self.cenet_enabled:
            return super().init_storage(
                training_type,
                num_envs,
                num_transitions_per_env,
                actor_obs_shape,
                critic_obs_shape,
                actions_shape,
            )
        self.storage = SquatRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            reconstruction_target_dim=self.policy.single_obs_dim,
            com_position_target_dim=self.policy.com_position_dim,
            com_velocity_target_dim=self.policy.com_velocity_dim,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        actions = super().act(obs, critic_obs)
        if not self.cenet_enabled:
            return actions

        required_critic_dim = self.critic_single_obs_dim * self.critic_history_length
        if critic_obs.shape[-1] < required_critic_dim:
            raise ValueError(
                "g1_squart critic observation is too small for the configured history: "
                f"expected at least {required_critic_dim}, got {critic_obs.shape[-1]}."
            )
        current_frame_start = (self.critic_history_length - 1) * self.critic_single_obs_dim
        position_index = current_frame_start + self.com_position_target_index
        velocity_index = current_frame_start + self.com_velocity_target_index
        self.transition.com_position_target = critic_obs[
            :, position_index : position_index + self.policy.com_position_dim
        ].detach()
        self.transition.com_velocity_target = critic_obs[
            :, velocity_index : velocity_index + self.policy.com_velocity_dim
        ].detach()
        return actions

    def process_env_step(self, rewards, dones, infos, next_actor_obs=None):
        if not self.cenet_enabled:
            return super().process_env_step(rewards, dones, infos)

        observations = infos.get("observations", {})
        reconstruction_target = observations.get("cenet_reconstruction_target")
        if reconstruction_target is None:
            if next_actor_obs is None:
                raise KeyError(
                    "CENet reconstruction target is missing and next_actor_obs was not provided by the runner."
                )
            if next_actor_obs.shape[-1] < self.policy.single_obs_dim:
                raise ValueError(
                    "next_actor_obs is too small for the CENet reconstruction target: "
                    f"expected at least {self.policy.single_obs_dim}, got {next_actor_obs.shape[-1]}."
                )
            reconstruction_target = next_actor_obs[:, -self.policy.single_obs_dim :]
            if not self._warned_missing_reconstruction_target:
                print(
                    "[WARN] g1_squart did not provide a clean CENet reconstruction target; "
                    "falling back to the last single observation in next_actor_obs."
                )
                self._warned_missing_reconstruction_target = True
        self.transition.reconstruction_target = reconstruction_target.to(self.device).detach()
        return super().process_env_step(rewards, dones, infos)

    def _reconstruction_weight(self) -> float:
        if self.reconstruction_warmup_iters == 0:
            return self.reconstruction_loss_coef
        progress = min(max(self.current_iteration, 0) / self.reconstruction_warmup_iters, 1.0)
        return self.reconstruction_loss_coef * progress

    def _kl_weight(self) -> float:
        if self.current_iteration <= self.kl_delay_iters:
            return 0.0
        progress = (self.current_iteration - self.kl_delay_iters) / (
            self.kl_warmup_end_iter - self.kl_delay_iters
        )
        return self.kl_loss_coef * min(max(progress, 0.0), 1.0)

    @staticmethod
    def _masked_reconstruction_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1, keepdim=True)
        valid = (~dones.view(-1, 1).bool()).to(per_sample_loss.dtype)
        return (per_sample_loss * valid).sum() / valid.sum().clamp_min(1.0)

    @staticmethod
    def _kl_loss(latent_mu: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(
            latent_mu.square() + latent_logvar.exp() - 1.0 - latent_logvar,
            dim=-1,
        ).mean()

    @staticmethod
    def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
        if not loss.requires_grad or not parameters:
            return 0.0
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared_norm = sum(
            gradient.detach().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        if isinstance(squared_norm, int):
            return 0.0
        return math.sqrt(squared_norm.item())

    def update(self):  # noqa: C901
        if not self.cenet_enabled:
            return super().update()
        if self.policy.is_recurrent:
            raise ValueError("SquatActorCritic with CENet is feed-forward and does not support recurrent storage.")

        reconstruction_weight = self._reconstruction_weight()
        kl_weight = self._kl_weight()
        totals = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "cenet/com_position": 0.0,
            "cenet/com_velocity": 0.0,
            "cenet/com_position_rmse_x_m": 0.0,
            "cenet/com_position_rmse_y_m": 0.0,
            "cenet/com_position_rmse_z_m": 0.0,
            "cenet/com_velocity_rmse_x_mps": 0.0,
            "cenet/com_velocity_rmse_y_mps": 0.0,
            "cenet/com_velocity_rmse_z_mps": 0.0,
            "cenet/reconstruction": 0.0,
            "cenet/kl": 0.0,
            "cenet/mu_std": 0.0,
            "cenet/posterior_std": 0.0,
        }
        gradient_norms = None
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for update_index, (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            _rnd_state_batch,
            dones_batch,
            com_position_target_batch,
            com_velocity_target_batch,
            reconstruction_target_batch,
        ) in enumerate(generator):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy
            cenet_outputs = self.policy.get_cenet_auxiliary_outputs()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                        / (2.0 * sigma_batch.square())
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).square(),
                    (value_clipped - returns_batch).square(),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).square().mean()

            position_scale = self.policy.cenet_module.com_position_scale
            velocity_scale = self.policy.cenet_module.com_velocity_scale
            normalized_position_target = (
                com_position_target_batch * position_scale
            )
            normalized_velocity_target = (
                com_velocity_target_batch * velocity_scale
            )
            com_position_loss = F.mse_loss(
                cenet_outputs["com_position"],
                normalized_position_target,
            )
            com_velocity_loss = F.mse_loss(
                cenet_outputs["com_velocity"],
                normalized_velocity_target,
            )
            reconstruction_loss = self._masked_reconstruction_loss(
                cenet_outputs["reconstruction"],
                reconstruction_target_batch,
                dones_batch,
            )
            cenet_kl_loss = self._kl_loss(
                cenet_outputs["latent_mu"],
                cenet_outputs["latent_logvar"],
            )
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            weighted_com_position_loss = (
                self.com_position_loss_coef * com_position_loss
            )
            weighted_com_velocity_loss = (
                self.com_velocity_loss_coef * com_velocity_loss
            )
            weighted_reconstruction_loss = reconstruction_weight * reconstruction_loss
            weighted_kl_loss = kl_weight * cenet_kl_loss
            loss = (
                ppo_loss
                + weighted_com_position_loss
                + weighted_com_velocity_loss
                + weighted_reconstruction_loss
                + weighted_kl_loss
            )

            if self.current_iteration % 100 == 0 and update_index == 0:
                encoder_parameters = self.policy.cenet_encoder_parameters()
                gradient_norms = {
                    "cenet/grad_ppo": self._gradient_norm(ppo_loss, encoder_parameters),
                    "cenet/grad_com_position": self._gradient_norm(
                        weighted_com_position_loss, encoder_parameters
                    ),
                    "cenet/grad_com_velocity": self._gradient_norm(
                        weighted_com_velocity_loss, encoder_parameters
                    ),
                    "cenet/grad_reconstruction": self._gradient_norm(
                        weighted_reconstruction_loss, encoder_parameters
                    ),
                    "cenet/grad_kl": self._gradient_norm(weighted_kl_loss, encoder_parameters),
                }

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            totals["value_function"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy_batch.mean().item()
            totals["cenet/com_position"] += com_position_loss.item()
            totals["cenet/com_velocity"] += com_velocity_loss.item()
            position_prediction_m = (
                cenet_outputs["com_position"].detach() / position_scale
            )
            velocity_prediction_mps = (
                cenet_outputs["com_velocity"].detach() / velocity_scale
            )
            position_rmse = torch.sqrt(
                (position_prediction_m - com_position_target_batch)
                .square()
                .mean(dim=0)
            )
            velocity_rmse = torch.sqrt(
                (velocity_prediction_mps - com_velocity_target_batch)
                .square()
                .mean(dim=0)
            )
            for axis, axis_index in zip(("x", "y", "z"), range(3)):
                totals[f"cenet/com_position_rmse_{axis}_m"] += (
                    position_rmse[axis_index].item()
                )
                totals[f"cenet/com_velocity_rmse_{axis}_mps"] += (
                    velocity_rmse[axis_index].item()
                )
            totals["cenet/reconstruction"] += reconstruction_loss.item()
            totals["cenet/kl"] += cenet_kl_loss.item()
            totals["cenet/mu_std"] += cenet_outputs["latent_mu"].detach().std(unbiased=False).item()
            totals["cenet/posterior_std"] += torch.exp(
                0.5 * cenet_outputs["latent_logvar"].detach()
            ).mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {key: value / num_updates for key, value in totals.items()}
        loss_dict["cenet/com_position_weight"] = self.com_position_loss_coef
        loss_dict["cenet/com_velocity_weight"] = self.com_velocity_loss_coef
        loss_dict["cenet/reconstruction_weight"] = reconstruction_weight
        loss_dict["cenet/kl_weight"] = kl_weight
        if gradient_norms is not None:
            loss_dict.update(gradient_norms)
        self.storage.clear()
        return loss_dict
