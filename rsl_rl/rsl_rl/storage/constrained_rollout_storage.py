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
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories

from .rollout_storage import RolloutStorage


class ConstrainedRolloutStorage(RolloutStorage):
    """Rollout storage with step-level cost buffers for constrained PPO."""

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.next_observations = None
            self.costs = None
            self.cost_values = None

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape,
            device,
        )

        if training_type == "rl":
            self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
            self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def add_transitions(self, transition: Transition):
        storage_step = self.step
        super().add_transitions(transition)

        if self.training_type != "rl":
            return

        if transition.next_observations is not None:
            self.next_observations[storage_step].copy_(transition.next_observations)
        else:
            self.next_observations[storage_step].copy_(transition.observations)

        if transition.costs is not None:
            self.costs[storage_step].copy_(transition.costs.view(-1, 1))
        else:
            self.costs[storage_step].zero_()

        if transition.cost_values is not None:
            self.cost_values[storage_step].copy_(transition.cost_values)
        else:
            self.cost_values[storage_step].zero_()

    def compute_cost_returns(self, last_cost_values, gamma, lam, normalize_advantages: bool = False):
        cost_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_cost_values = last_cost_values
            else:
                next_cost_values = self.cost_values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_cost_values - self.cost_values[step]
            cost_advantage = delta + next_is_not_terminal * gamma * lam * cost_advantage
            self.cost_returns[step] = cost_advantage + self.cost_values[step]

        self.cost_advantages = self.cost_returns - self.cost_values
        if normalize_advantages:
            self.cost_advantages = (self.cost_advantages - self.cost_advantages.mean()) / (
                self.cost_advantages.std() + 1e-8
            )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, include_residual_gate=False):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        residual_gates = self.residual_gates.flatten(0, 1) if include_residual_gate else None

        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                rnd_state_batch = rnd_state[batch_idx] if self.rnd_state_shape is not None else None
                residual_gate_batch = residual_gates[batch_idx] if include_residual_gate else None

                batch = (
                    observations[batch_idx],
                    next_observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    self.dones.flatten(0, 1)[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    cost_values[batch_idx],
                    cost_advantages[batch_idx],
                    cost_returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (
                        None,
                        None,
                    ),
                    None,
                    rnd_state_batch,
                )
                if include_residual_gate:
                    batch = (*batch, residual_gate_batch)
                yield batch

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8, include_residual_gate=False):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_next_obs_trajectories, _ = split_and_pad_trajectories(self.next_observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                cost_values_batch = self.cost_values[:, start:stop]
                cost_returns_batch = self.cost_returns[:, start:stop]
                cost_advantages_batch = self.cost_advantages[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                residual_gate_batch = self.residual_gates[:, start:stop] if include_residual_gate else None

                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                batch = (
                    obs_batch,
                    padded_next_obs_trajectories[:, first_traj:last_traj],
                    privileged_obs_batch,
                    actions_batch,
                    dones[:, start:stop],
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    cost_values_batch,
                    cost_advantages_batch,
                    cost_returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hid_a_batch,
                        hid_c_batch,
                    ),
                    masks_batch,
                    rnd_state_batch,
                )
                if include_residual_gate:
                    batch = (*batch, residual_gate_batch)
                yield batch

                first_traj = last_traj
