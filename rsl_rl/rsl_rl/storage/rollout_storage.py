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


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None
            self.residual_gate = None
            # RENet Recovery metadata. ``recovery_mask_t`` is the mode that
            # produced action_t; the event flags describe the resulting
            # transition and must be captured explicitly before env reset.
            self.recovery_mask_t = None
            self.enter_recovery = None
            self.exit_recovery = None
            self.recovery_failed = None
            self.time_outs = None
            self.recovery_task_reward = None
            self.recovery_amp_reward = None
            self.recovery_reg_reward = None
            self.recovery_rewards_valid = None
            self.recovery_task_value = None
            self.recovery_amp_value = None
            self.recovery_reg_value = None
            self.timeout_loco_value = None
            self.timeout_rec_task_value = None
            self.timeout_rec_amp_value = None
            self.timeout_rec_reg_value = None

        def clear(self):
            self.__init__()

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
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # Core 
        # [24,4096,960]
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.residual_gates = torch.ones(num_transitions_per_env, num_envs, 1, device=self.device)

            # These buffers are inert for algorithms that do not provide the
            # corresponding Transition fields. Keeping them in the base
            # storage avoids a second, subtly incompatible rollout format.
            bool_shape = (num_transitions_per_env, num_envs, 1)
            self.recovery_masks = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
            self.enter_recovery = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
            self.exit_recovery = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
            self.recovery_failed = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
            self.time_outs = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)
            self.recovery_rewards_valid = torch.zeros(bool_shape, dtype=torch.bool, device=self.device)

            value_shape = (num_transitions_per_env, num_envs, 1)
            self.recovery_task_rewards = torch.zeros(value_shape, device=self.device)
            self.recovery_amp_rewards = torch.zeros(value_shape, device=self.device)
            self.recovery_reg_rewards = torch.zeros(value_shape, device=self.device)
            self.recovery_task_values = torch.zeros(value_shape, device=self.device)
            self.recovery_amp_values = torch.zeros(value_shape, device=self.device)
            self.recovery_reg_values = torch.zeros(value_shape, device=self.device)
            self.recovery_task_returns = torch.zeros(value_shape, device=self.device)
            self.recovery_amp_returns = torch.zeros(value_shape, device=self.device)
            self.recovery_reg_returns = torch.zeros(value_shape, device=self.device)
            self.recovery_task_advantages = torch.zeros(value_shape, device=self.device)
            self.recovery_amp_advantages = torch.zeros(value_shape, device=self.device)
            self.recovery_reg_advantages = torch.zeros(value_shape, device=self.device)
            # True pre-reset V(s_terminal) values. They remain zero for normal
            # transitions and true terminations, including Recovery failure.
            self.timeout_loco_values = torch.zeros(value_shape, device=self.device)
            self.timeout_rec_task_values = torch.zeros(value_shape, device=self.device)
            self.timeout_rec_amp_values = torch.zeros(value_shape, device=self.device)
            self.timeout_rec_reg_values = torch.zeros(value_shape, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
            if transition.residual_gate is not None:
                self.residual_gates[self.step].copy_(transition.residual_gate.view(-1, 1))
            else:
                self.residual_gates[self.step].fill_(1.0)

            self._copy_optional_bool(self.recovery_masks[self.step], transition.recovery_mask_t)
            self._copy_optional_bool(self.enter_recovery[self.step], transition.enter_recovery)
            self._copy_optional_bool(self.exit_recovery[self.step], transition.exit_recovery)
            self._copy_optional_bool(self.recovery_failed[self.step], transition.recovery_failed)
            self._copy_optional_bool(self.time_outs[self.step], transition.time_outs)
            self._copy_optional_bool(
                self.recovery_rewards_valid[self.step], transition.recovery_rewards_valid
            )
            self._copy_optional_float(
                self.recovery_task_rewards[self.step], transition.recovery_task_reward
            )
            self._copy_optional_float(
                self.recovery_amp_rewards[self.step], transition.recovery_amp_reward
            )
            self._copy_optional_float(
                self.recovery_reg_rewards[self.step], transition.recovery_reg_reward
            )
            self._copy_optional_float(
                self.recovery_task_values[self.step], transition.recovery_task_value
            )
            self._copy_optional_float(
                self.recovery_amp_values[self.step], transition.recovery_amp_value
            )
            self._copy_optional_float(
                self.recovery_reg_values[self.step], transition.recovery_reg_value
            )
            self._copy_optional_float(
                self.timeout_loco_values[self.step], transition.timeout_loco_value
            )
            self._copy_optional_float(
                self.timeout_rec_task_values[self.step], transition.timeout_rec_task_value
            )
            self._copy_optional_float(
                self.timeout_rec_amp_values[self.step], transition.timeout_rec_amp_value
            )
            self._copy_optional_float(
                self.timeout_rec_reg_values[self.step], transition.timeout_rec_reg_value
            )

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    @staticmethod
    def _copy_optional_bool(destination: torch.Tensor, source):
        if source is None:
            destination.zero_()
        else:
            destination.copy_(source.view(-1, 1).bool())

    @staticmethod
    def _copy_optional_float(destination: torch.Tensor, source):
        if source is None:
            destination.zero_()
        else:
            destination.copy_(source.view(-1, 1))

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    # 这个函数就是 GAE（Generalized Advantage Estimation）真正发生的地方
    '''
    假设 num_steps_per_env = 24，那么调用这个函数之前，storage 大致已经有：
    step       reward       value       done
    -----------------------------------------
    0           r0           V0          d0
    1           r1           V1          d1
    2           r2           V2          d2
    ...
    22          r22          V22         d22
    23          r23          V23         d23
    还有V(S_24)

    A_t=siga_t + gamma*lam*A_{t+1} 
    '''
    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):  # 倒着算，因为当前时刻的优势函数需要下一时刻的优势函数进行计算。例如我想计算A_3 就必须要先知道A_4
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()  # 为什么不是step+1：因为这个down就是S_{t+1}的状态了
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t) 
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    @staticmethod
    def normalize_masked_advantage(
        advantages: torch.Tensor,
        sample_mask: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        """Normalize only selected samples, safely handling zero or one sample."""
        mask = sample_mask.bool().expand_as(advantages)
        normalized = torch.zeros_like(advantages)
        if not torch.any(mask):
            return normalized

        selected = advantages[mask]
        mean = selected.mean()
        # unbiased=False is finite for a one-sample Recovery segment.
        variance = torch.mean(torch.square(selected - mean))
        normalized[mask] = (selected - mean) / torch.sqrt(variance + eps)
        return normalized

    @staticmethod
    def compute_segmented_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        sample_mask: torch.Tensor,
        trace_end: torch.Tensor,
        env_terminal: torch.Tensor,
        time_outs: torch.Tensor,
        gamma: float,
        lam: float,
        timeout_bootstrap_values: torch.Tensor | None = None,
        normalize_advantage: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE within explicit value-function segments.

        ``trace_end`` is a value boundary such as enter/exit Recovery and is
        deliberately independent of ``env_terminal``. Timeouts stop the trace
        but retain truncation bootstrap through ``timeout_bootstrap_values``.
        ``timeout_bootstrap_values`` contains V(s_terminal), evaluated on the
        true post-physics/pre-reset terminal observation. The post-reset state
        belongs to a new episode and is never used here.
        """
        expected_shape = values.shape
        tensors = {
            "rewards": rewards,
            "sample_mask": sample_mask,
            "trace_end": trace_end,
            "env_terminal": env_terminal,
            "time_outs": time_outs,
        }
        for name, tensor in tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}."
                )
        if last_values.shape != values.shape[1:]:
            raise ValueError(
                "last_values must match one rollout step: "
                f"expected {tuple(values.shape[1:])}, got {tuple(last_values.shape)}."
            )

        active = sample_mask.bool()
        trace_end = trace_end.bool()
        env_terminal = env_terminal.bool()
        time_outs = time_outs.bool()
        if timeout_bootstrap_values is None:
            # Never silently substitute V(s_t) for a missing terminal value.
            timeout_bootstrap_values = torch.zeros_like(values)
        elif timeout_bootstrap_values.shape != values.shape:
            raise ValueError(
                "timeout_bootstrap_values must match values: "
                f"{tuple(timeout_bootstrap_values.shape)} != {tuple(values.shape)}."
            )

        returns = values.clone()
        advantages = torch.zeros_like(values)
        next_advantage = torch.zeros_like(last_values)
        num_steps = values.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
                next_active = active[step]
            else:
                next_values = values[step + 1]
                next_active = active[step + 1]

            boundary = trace_end[step] | env_terminal[step] | time_outs[step]
            continues = active[step] & ~boundary & next_active
            # A truncation bootstraps its terminal value for the TD residual,
            # but the reset remains a hard GAE boundary. True terminations
            # (including Recovery failure) always override timeout bootstrap.
            timeout_bootstrap = active[step] & time_outs[step] & ~env_terminal[step]
            timeout_correction = gamma * timeout_bootstrap_values[step] * timeout_bootstrap.float()
            delta = (
                rewards[step]
                + timeout_correction
                + gamma * next_values * continues.float()
                - values[step]
            )
            step_advantage = delta + gamma * lam * next_advantage * continues.float()
            next_advantage = torch.where(active[step], step_advantage, torch.zeros_like(step_advantage))
            advantages[step] = next_advantage
            returns[step] = torch.where(active[step], next_advantage + values[step], values[step])

        if normalize_advantage:
            advantages = RolloutStorage.normalize_masked_advantage(advantages, active)
        return returns, advantages

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs=8,
        include_residual_gate=False,
        include_recovery_data=False,
    ):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env   ## 也就是说，我们现在手里有 98304 条 (s_t, a_t, r_t, ...) 数据。
        mini_batch_size = batch_size // num_mini_batches # 把数据平均分成num_mini_batches个小批次
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device) # 生成了一个长度为batch_size的乱序的张量索引[10,2,5,123,1,97,56,2312,3545,345,123....]

        # Core
        # 原来的维度是[24,4096,观测维度]  flatten(0, 1) 让第0维第一维合并维一个维度  变成了[24*4096 观测维度]
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        residual_gates = self.residual_gates.flatten(0, 1) if include_residual_gate else None
        if include_recovery_data:
            recovery_data = {
                "recovery_mask_t": self.recovery_masks.flatten(0, 1),
                "enter_recovery_t": self.enter_recovery.flatten(0, 1),
                "exit_recovery_t": self.exit_recovery.flatten(0, 1),
                "recovery_failed_t": self.recovery_failed.flatten(0, 1),
                "time_out_t": self.time_outs.flatten(0, 1),
                "recovery_rewards_valid": self.recovery_rewards_valid.flatten(0, 1),
                "recovery_task_values": self.recovery_task_values.flatten(0, 1),
                "recovery_amp_values": self.recovery_amp_values.flatten(0, 1),
                "recovery_reg_values": self.recovery_reg_values.flatten(0, 1),
                "recovery_task_returns": self.recovery_task_returns.flatten(0, 1),
                "recovery_amp_returns": self.recovery_amp_returns.flatten(0, 1),
                "recovery_reg_returns": self.recovery_reg_returns.flatten(0, 1),
                "recovery_task_advantages": self.recovery_task_advantages.flatten(0, 1),
                "recovery_amp_advantages": self.recovery_amp_advantages.flatten(0, 1),
                "recovery_reg_advantages": self.recovery_reg_advantages.flatten(0, 1),
            }

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size  # 第一次循环是0
                end = (i + 1) * mini_batch_size 
                batch_idx = indices[start:end] # 这个就是从乱序索引中索引

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx] # 得到第n个小批次的观测值
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                residual_gate_batch = residual_gates[batch_idx] if include_residual_gate else None

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                batch = (
                    obs_batch,
                    privileged_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                    None,
                    None,
                    ),
                    None,
                    rnd_state_batch,
                )
                if include_residual_gate:
                    batch = (*batch, residual_gate_batch)
                if include_recovery_data:
                    batch = (*batch, {key: value[batch_idx] for key, value in recovery_data.items()})
                yield batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs=8,
        include_residual_gate=False,
        include_recovery_data=False,
    ):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
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
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                residual_gate_batch = self.residual_gates[:, start:stop] if include_residual_gate else None

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
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
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                batch = (
                    obs_batch,
                    privileged_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
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
                if include_recovery_data:
                    recovery_data_batch = {
                        "recovery_mask_t": self.recovery_masks[:, start:stop],
                        "enter_recovery_t": self.enter_recovery[:, start:stop],
                        "exit_recovery_t": self.exit_recovery[:, start:stop],
                        "recovery_failed_t": self.recovery_failed[:, start:stop],
                        "time_out_t": self.time_outs[:, start:stop],
                        "recovery_rewards_valid": self.recovery_rewards_valid[:, start:stop],
                        "recovery_task_values": self.recovery_task_values[:, start:stop],
                        "recovery_amp_values": self.recovery_amp_values[:, start:stop],
                        "recovery_reg_values": self.recovery_reg_values[:, start:stop],
                        "recovery_task_returns": self.recovery_task_returns[:, start:stop],
                        "recovery_amp_returns": self.recovery_amp_returns[:, start:stop],
                        "recovery_reg_returns": self.recovery_reg_returns[:, start:stop],
                        "recovery_task_advantages": self.recovery_task_advantages[:, start:stop],
                        "recovery_amp_advantages": self.recovery_amp_advantages[:, start:stop],
                        "recovery_reg_advantages": self.recovery_reg_advantages[:, start:stop],
                    }
                    batch = (*batch, recovery_data_batch)
                yield batch

                first_traj = last_traj
