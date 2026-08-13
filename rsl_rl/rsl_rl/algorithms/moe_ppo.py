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

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class MoePPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # ========== Auxiliary task parameters (added for PIE alignment) ==========
        vel_estimation_coef: float = 0.0,
        vel_estimation_warmup_iters: int = 0,
        terrain_recon_coef: float = 0.0,
        terrain_recon_warmup_iters: int = 0,
        terrain_recon_target_clip: float = 1.0,
        obs_dim: int = 102,                 # actor observation dimension
        terrain_scan_dim: int = 187,        # height scan dimension in critic_obs
        vel_dim: int = 3,
        terrain_recon_front_only: bool = True,
        terrain_recon_grid_cols: int = 17,
        terrain_recon_grid_rows: int = 11,
        terrain_recon_x_min: float = 0.0,
        single_critic_dim: int = 107,
        critic_history_len: int = 10,
        vel_in_critic_offset: int = 104,
        feet_height_coef: float = 0.0,
        feet_height_warmup_iters: int = 0,
        feet_height_dim: int = 2,
        feet_height_in_critic_offset: int = 0,
        use_moe_balance_loss: bool = False,
        moe_balance_coef: float = 0.0,
        use_moe_gate_entropy_loss: bool = False,
        moe_gate_entropy_coef: float = 0.0,

    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch


        # ========== Auxiliary task parameters ==========
        self.vel_estimation_coef = vel_estimation_coef
        self.vel_estimation_warmup_iters = vel_estimation_warmup_iters
        self.terrain_recon_coef = terrain_recon_coef
        self.terrain_recon_warmup_iters = terrain_recon_warmup_iters
        self.terrain_recon_target_clip = terrain_recon_target_clip
        self.obs_dim = obs_dim
        self.terrain_scan_dim = terrain_scan_dim
        #self.vel_obs_start_idx = vel_obs_start_idx
        self.vel_dim = vel_dim
        self.current_iteration = 0
        self.single_critic_dim = single_critic_dim
        self.critic_history_len = critic_history_len
        self.vel_in_critic_offset = vel_in_critic_offset
        self.vel_obs_start_idx = (self.critic_history_len - 1) * self.single_critic_dim + self.vel_in_critic_offset
        self.feet_height_coef = feet_height_coef
        self.feet_height_warmup_iters = feet_height_warmup_iters
        self.feet_height_dim = feet_height_dim
        self.feet_height_in_critic_offset = feet_height_in_critic_offset
        self.feet_height_obs_start_idx = (
            (self.critic_history_len - 1) * self.single_critic_dim + self.feet_height_in_critic_offset
        )
        print(f"[MoeAmpPpO] Computed vel_obs_start_idx = {self.vel_obs_start_idx}")
        print(f"[MoeAmpPpO] Computed feet_height_obs_start_idx = {self.feet_height_obs_start_idx}")
        self.use_moe_balance_loss = use_moe_balance_loss
        self.moe_balance_coef = moe_balance_coef
        self.use_moe_gate_entropy_loss = use_moe_gate_entropy_loss
        self.moe_gate_entropy_coef = moe_gate_entropy_coef
        
        # Precompute front terrain indices (if needed)
        self.terrain_front_indices = None
        if terrain_recon_front_only and terrain_recon_coef > 0:
            resolution = 0.1
            size_x = (terrain_recon_grid_cols - 1) * resolution
            x_min_idx = max(0, int((terrain_recon_x_min + size_x / 2) / resolution))
            front_indices = []
            for y_idx in range(terrain_recon_grid_rows):
                for x_idx in range(x_min_idx, terrain_recon_grid_cols):
                    front_indices.append(y_idx * terrain_recon_grid_cols + x_idx)
            self.terrain_front_indices = torch.tensor(front_indices, dtype=torch.long, device=device)
            self.terrain_recon_target_dim = len(front_indices)
        else:
            self.terrain_recon_target_dim = terrain_scan_dim

        # Print auxiliary task info
        if vel_estimation_coef > 0:
            print(f"[MoeAMPPPO] Velocity estimation AUX TASK enabled:")
            print(f"  - coef: {vel_estimation_coef}, warmup: {vel_estimation_warmup_iters} iters")
        if terrain_recon_coef > 0:
            print(f"[MoeAMPPPO] Terrain reconstruction AUX TASK enabled:")
            print(f"  - coef: {terrain_recon_coef}, warmup: {terrain_recon_warmup_iters} iters")
            print(f"  - front_only: {terrain_recon_front_only}, predict {self.terrain_recon_target_dim} points")
        if feet_height_coef > 0:
            print(f"[MoeAMPPPO] Feet height AUX TASK enabled:")
            print(f"  - coef: {feet_height_coef}, warmup: {feet_height_warmup_iters} iters")
        print(
            "[MoePPO] MoE load-balance loss: "
            f"enabled={self.use_moe_balance_loss}, coef={self.moe_balance_coef}"
        )
        print(
            "[MoePPO] MoE gate entropy loss: "
            f"enabled={self.use_moe_gate_entropy_loss}, coef={self.moe_gate_entropy_coef}"
        )
    
    
    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        self._set_policy_moe_iteration()
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs  如果 episode 只是因为时间限制结束，不应该把该环境的未来价值直接砍成 0。而要补一个 value estimate：r_t​⟶r_t​+γV(⋅)
        # Values的形状是[4096,1]
        '''
        假设有4个并行环境：
        env0   env1   env2   env3
          0      1      0      1
        env0：没 timeout
        env1：因为时间限制结束
        env2：没 timeout
        env3：因为时间限制结束
        infos["time_outs"].unsqueeze(1)： 把[4]变成[4,1]
        values * time_outs 相当于 mask:
        values:
            [[ 5],
            [ 8],
            [ 3],
            [10]]

            timeouts:
            [[0],
            [1],
            [0],
            [1]]
        values * timeouts

        [[ 0],
        [ 8],
        [ 0],
        [10]]

        torch.squeeze(..., 1):变成[0, 8, 0, 10] 方便与reward相加。这样就做到了只有 timeout 的环境 reward 被补偿

        '''
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
    '''
    last_values = self.policy.evaluate(obs).detach() 算最后一个状态的 value

    假设这次 rollout 一共收集 24 步：
                                t = 0
                                t = 1
                                ...
                                t = 23
    在 storage 里已经保存了：V(s0​),V(s1​),...,V(s23​)
    但是为了算最后一步的 TD error：siga_23=r_23+gamma*V(S_24)-V(S_23)
    还缺：V(S_24)
    所以 rollout 结束后，当前的 obs 已经是S_24
    于是self.policy.evaluate(obs) 再计算一次V(S_24)
    '''
    def compute_returns(self, last_critic_obs):
        self._set_policy_moe_iteration()
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _get_warmup_weight(self, iteration: int, warmup_iters: int, target_weight: float) -> float:
        """Get warmup-adjusted weight for auxiliary losses."""
        if warmup_iters <= 0:
            return target_weight
        if iteration >= warmup_iters:
            return target_weight
        return target_weight * (iteration / warmup_iters)


    def update(self):  # noqa: C901
        self._set_policy_moe_iteration()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # ========== Auxiliary losses ==========
        mean_vel_loss = 0
        mean_op_vel_loss = 0
        mean_vp_vel_loss = 0
        mean_terrain_recon_loss = 0
        mean_feet_height_loss = 0
        mean_moe_balance_loss = 0
        mean_moe_gate_entropy = 0
        mean_moe_gate_entropy_loss = 0
        mean_moe_action_disagreement = 0
        sum_moe_expert_usage = None
        sum_moe_expert_top1_freq = None
        sum_moe_expert_action_disagreement = None
        vel_weight = self._get_warmup_weight(
            self.current_iteration, self.vel_estimation_warmup_iters, self.vel_estimation_coef
        )
        terrain_weight = self._get_warmup_weight(
            self.current_iteration, self.terrain_recon_warmup_iters, self.terrain_recon_coef
        )
        feet_height_weight = self._get_warmup_weight(
            self.current_iteration, self.feet_height_warmup_iters, self.feet_height_coef
        )


        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
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
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            moe_balance_loss = (
                self.policy.get_moe_balance_loss()
                if hasattr(self.policy, "get_moe_balance_loss")
                else torch.zeros((), device=self.device)
            )
            moe_gate_entropy = (
                self.policy.get_moe_gate_entropy()
                if hasattr(self.policy, "get_moe_gate_entropy")
                else torch.zeros((), device=self.device)
            )
            moe_gate_entropy_loss = -moe_gate_entropy
            moe_routing_stats = (
                self.policy.get_moe_routing_stats()
                if hasattr(self.policy, "get_moe_routing_stats")
                else {}
            )
            moe_expert_usage = moe_routing_stats.get("expert_usage")
            moe_expert_top1_freq = moe_routing_stats.get("expert_top1_freq")
            moe_expert_action_disagreement = moe_routing_stats.get("expert_action_disagreement")
            moe_action_disagreement = moe_routing_stats.get(
                "action_disagreement",
                torch.zeros((), device=self.device),
            )

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ========== Auxiliary Losses ==========
            # 1. Velocity estimation loss
            vel_loss = torch.tensor(0.0, device=self.device)
            op_vel_loss = torch.tensor(0.0, device=self.device)
            vp_vel_loss = torch.tensor(0.0, device=self.device)
            if vel_weight > 0 and hasattr(self.policy, 'use_vel_estimation') and self.policy.use_vel_estimation:
                vel_estimate = self.policy.predict_velocity()
                if vel_estimate is not None and self.vel_obs_start_idx is not None:
                    # Extract velocity ground truth from critic_obs_batch
                    vel_target = critic_obs_batch[:, self.vel_obs_start_idx:self.vel_obs_start_idx + self.vel_dim].detach()
                    if isinstance(vel_estimate, dict):
                        vel_losses = []
                        if "op" in vel_estimate:
                            op_vel_loss = nn.functional.mse_loss(vel_estimate["op"], vel_target)
                            vel_losses.append(op_vel_loss)
                        if "vp" in vel_estimate:
                            vp_vel_loss = nn.functional.mse_loss(vel_estimate["vp"], vel_target)
                            vel_losses.append(vp_vel_loss)
                        if vel_losses:
                            vel_loss = torch.stack(vel_losses).mean()
                    else:
                        vel_loss = nn.functional.mse_loss(vel_estimate, vel_target)

            # 2. Terrain reconstruction loss
            terrain_recon_loss = torch.tensor(0.0, device=self.device)
            if terrain_weight > 0 and hasattr(self.policy, 'use_terrain_recon') and self.policy.use_terrain_recon:
                terrain_pred = self.policy.predict_terrain()
                if terrain_pred is not None:
                    # Get terrain target from critic_obs_batch (last terrain_scan_dim dims)
                    terrain_target_full = critic_obs_batch[:, -self.terrain_scan_dim:].detach()
                    # Take front region only if configured
                    if self.terrain_front_indices is not None:
                        terrain_target = terrain_target_full[:, self.terrain_front_indices]
                    else:
                        terrain_target = terrain_target_full
                    # Clip target values
                    if self.terrain_recon_target_clip > 0:
                        terrain_target = terrain_target.clamp(
                            -self.terrain_recon_target_clip,
                            self.terrain_recon_target_clip
                        )
                    # Per-batch normalize (same as reference project)
                    target_mean = terrain_target.mean()
                    target_std = terrain_target.std().clamp(min=1e-6)
                    terrain_target_norm = (terrain_target - target_mean) / target_std
                    terrain_recon_loss = nn.functional.mse_loss(terrain_pred, terrain_target_norm)

            # 3. Feet height prediction loss
            feet_height_loss = torch.tensor(0.0, device=self.device)
            if (
                feet_height_weight > 0
                and hasattr(self.policy, "use_feet_height_prediction")
                and self.policy.use_feet_height_prediction
            ):
                feet_height_pred = self.policy.predict_feet_height()
                if feet_height_pred is not None:
                    feet_height_target = critic_obs_batch[
                        :,
                        self.feet_height_obs_start_idx : self.feet_height_obs_start_idx + self.feet_height_dim,
                    ].detach()
                    feet_height_loss = nn.functional.mse_loss(feet_height_pred, feet_height_target)

            # Add auxiliary losses to total loss
            loss += vel_weight * vel_loss + terrain_weight * terrain_recon_loss
            loss += feet_height_weight * feet_height_loss
            if self.use_moe_balance_loss and self.moe_balance_coef > 0:
                loss += self.moe_balance_coef * moe_balance_loss
            if self.use_moe_gate_entropy_loss and self.moe_gate_entropy_coef != 0:
                loss += self.moe_gate_entropy_coef * moe_gate_entropy_loss


            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # -- Auxiliary losses
            mean_vel_loss += vel_loss.item()
            mean_op_vel_loss += op_vel_loss.item()
            mean_vp_vel_loss += vp_vel_loss.item()
            mean_terrain_recon_loss += terrain_recon_loss.item()
            mean_feet_height_loss += feet_height_loss.item()
            mean_moe_balance_loss += moe_balance_loss.item()
            mean_moe_gate_entropy += moe_gate_entropy.item()
            mean_moe_gate_entropy_loss += moe_gate_entropy_loss.item()
            mean_moe_action_disagreement += moe_action_disagreement.item()

            if moe_expert_usage is not None:
                if sum_moe_expert_usage is None:
                    sum_moe_expert_usage = torch.zeros_like(moe_expert_usage)
                sum_moe_expert_usage += moe_expert_usage
            if moe_expert_top1_freq is not None:
                if sum_moe_expert_top1_freq is None:
                    sum_moe_expert_top1_freq = torch.zeros_like(moe_expert_top1_freq)
                sum_moe_expert_top1_freq += moe_expert_top1_freq
            if moe_expert_action_disagreement is not None:
                if sum_moe_expert_action_disagreement is None:
                    sum_moe_expert_action_disagreement = torch.zeros_like(moe_expert_action_disagreement)
                sum_moe_expert_action_disagreement += moe_expert_action_disagreement


            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # -- For Auxiliary tasks
        mean_vel_loss /= num_updates
        mean_op_vel_loss /= num_updates
        mean_vp_vel_loss /= num_updates
        mean_terrain_recon_loss /= num_updates
        mean_feet_height_loss /= num_updates
        mean_moe_balance_loss /= num_updates
        mean_moe_gate_entropy /= num_updates
        mean_moe_gate_entropy_loss /= num_updates
        mean_moe_action_disagreement /= num_updates
        mean_moe_expert_usage = (
            sum_moe_expert_usage / num_updates if sum_moe_expert_usage is not None else None
        )
        mean_moe_expert_top1_freq = (
            sum_moe_expert_top1_freq / num_updates if sum_moe_expert_top1_freq is not None else None
        )
        mean_moe_expert_action_disagreement = (
            sum_moe_expert_action_disagreement / num_updates
            if sum_moe_expert_action_disagreement is not None
            else None
        )


        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        
        if self.vel_estimation_coef > 0:
            loss_dict["vel_estimation"] = mean_vel_loss
            loss_dict["vel_estimation_coef_current"] = vel_weight
            loss_dict["vel_estimation/op"] = mean_op_vel_loss
            loss_dict["vel_estimation/vp"] = mean_vp_vel_loss
        if self.terrain_recon_coef > 0:
            loss_dict["terrain_recon"] = mean_terrain_recon_loss
            loss_dict["terrain_recon_coef_current"] = terrain_weight
        if self.feet_height_coef > 0:
            loss_dict["feet_height"] = mean_feet_height_loss
            loss_dict["feet_height_coef_current"] = feet_height_weight
        loss_dict["moe_gate_entropy"] = mean_moe_gate_entropy
        loss_dict["moe/action_disagreement"] = mean_moe_action_disagreement
        if mean_moe_expert_usage is not None:
            for expert_idx, value in enumerate(mean_moe_expert_usage.detach().cpu().tolist()):
                loss_dict[f"moe/expert_{expert_idx}_usage"] = value
        if mean_moe_expert_top1_freq is not None:
            for expert_idx, value in enumerate(mean_moe_expert_top1_freq.detach().cpu().tolist()):
                loss_dict[f"moe/expert_{expert_idx}_top1_freq"] = value
        if mean_moe_expert_action_disagreement is not None:
            for expert_idx, value in enumerate(mean_moe_expert_action_disagreement.detach().cpu().tolist()):
                loss_dict[f"moe/expert_{expert_idx}_action_disagreement"] = value
        if self.use_moe_balance_loss and self.moe_balance_coef > 0:
            loss_dict["moe_balance"] = mean_moe_balance_loss
            loss_dict["moe_balance_coef"] = self.moe_balance_coef
        if self.use_moe_gate_entropy_loss and self.moe_gate_entropy_coef != 0:
            loss_dict["moe_gate_entropy_loss"] = mean_moe_gate_entropy_loss
            loss_dict["moe_gate_entropy_coef"] = self.moe_gate_entropy_coef
        self.current_iteration += 1
        self._set_policy_moe_iteration()
        return loss_dict

    """
    Helper functions
    """

    def _set_policy_moe_iteration(self):
        if hasattr(self.policy, "set_moe_iteration"):
            self.policy.set_moe_iteration(self.current_iteration)

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
