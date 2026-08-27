from __future__ import annotations

import torch

from .rollout_storage import RolloutStorage


class SquatRolloutStorage(RolloutStorage):
    """Rollout storage for the g1_squart CENet auxiliary targets."""

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.com_position_target = None
            self.com_velocity_target = None
            self.reconstruction_target = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        reconstruction_target_dim: int,
        com_position_target_dim: int = 3,
        com_velocity_target_dim: int = 3,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape=None,
            device=device,
        )
        if training_type != "rl":
            raise ValueError("SquatRolloutStorage is only available for reinforcement learning training.")
        self.com_position_targets = torch.zeros(
            num_transitions_per_env,
            num_envs,
            com_position_target_dim,
            device=self.device,
        )
        self.com_velocity_targets = torch.zeros(
            num_transitions_per_env,
            num_envs,
            com_velocity_target_dim,
            device=self.device,
        )
        self.reconstruction_targets = torch.zeros(
            num_transitions_per_env,
            num_envs,
            reconstruction_target_dim,
            device=self.device,
        )

    def add_transitions(self, transition: Transition):
        if transition.com_position_target is None:
            raise ValueError(
                "Missing g1_squart COM position target for rollout transition."
            )
        if transition.com_velocity_target is None:
            raise ValueError(
                "Missing g1_squart COM velocity target for rollout transition."
            )
        if transition.reconstruction_target is None:
            raise ValueError("Missing g1_squart reconstruction target for rollout transition.")
        storage_step = self.step
        expected_position_shape = self.com_position_targets[storage_step].shape
        expected_velocity_shape = self.com_velocity_targets[storage_step].shape
        expected_reconstruction_shape = self.reconstruction_targets[
            storage_step
        ].shape
        if transition.com_position_target.shape != expected_position_shape:
            raise ValueError(
                "g1_squart COM position target shape mismatch: expected "
                f"{tuple(expected_position_shape)}, got "
                f"{tuple(transition.com_position_target.shape)}."
            )
        if transition.com_velocity_target.shape != expected_velocity_shape:
            raise ValueError(
                "g1_squart COM velocity target shape mismatch: expected "
                f"{tuple(expected_velocity_shape)}, got "
                f"{tuple(transition.com_velocity_target.shape)}."
            )
        if transition.reconstruction_target.shape != expected_reconstruction_shape:
            raise ValueError(
                "g1_squart reconstruction target shape mismatch: expected "
                f"{tuple(expected_reconstruction_shape)}, got "
                f"{tuple(transition.reconstruction_target.shape)}."
            )
        super().add_transitions(transition)
        self.com_position_targets[storage_step].copy_(
            transition.com_position_target
        )
        self.com_velocity_targets[storage_step].copy_(
            transition.com_velocity_target
        )
        self.reconstruction_targets[storage_step].copy_(transition.reconstruction_target)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        privileged_observations = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else observations
        )
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        com_position_targets = self.com_position_targets.flatten(0, 1)
        com_velocity_targets = self.com_velocity_targets.flatten(0, 1)
        reconstruction_targets = self.reconstruction_targets.flatten(0, 1)

        for _ in range(num_epochs):
            for mini_batch_index in range(num_mini_batches):
                start = mini_batch_index * mini_batch_size
                end = (mini_batch_index + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    None,
                    dones[batch_idx],
                    com_position_targets[batch_idx],
                    com_velocity_targets[batch_idx],
                    reconstruction_targets[batch_idx],
                )
