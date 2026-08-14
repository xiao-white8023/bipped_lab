from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import string_to_callable


class AMPPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
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
        # Auxiliary task parameters
        vel_estimation_coef: float = 0.0,
        vel_estimation_warmup_iters: int = 0,
        terrain_recon_coef: float = 0.0,
        terrain_recon_warmup_iters: int = 0,
        terrain_recon_target_clip: float = 1.0,
        terrain_scan_dim: int = 187,
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

        # Discriminator components
        '''
        含义：AMP 损失（判别器的 Loss）在总损失中的权重。
        背景：在 AMPPPO 中，我们通常用同一个优化器（Optimizer）同时更新 Policy（机器人）和 Discriminator（判别器）。
        作用：
        Ltotal​=LPPO​+amploss_coef×LDiscriminator​
        这决定了在一次反向传播中，我们是更在乎“优化机器人的策略”，还是更在乎“提升判别器的眼力”。
        设为 1.0 表示两者同等重要。
        '''
        self.amploss_coef = 1.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        '''
        这两行代码是在为视 AMP (Adversarial Motion Priors) 算法构建**“假数据”的存储系统**。
        在 AMP 训练中，我们需要一个地方来存放机器人产生的动作数据（这些数据被为“假”数据），以便后续和专家数据（“真”数据）一起喂给判别器进行训练。这两行代码分别建立了短期缓存和长期经验池。
        '''
        self.amp_transition = RolloutStorage.Transition() # 这是一个微型的临时容器，通常只用来存放**“当前这一个时间步 (timestep)”** 产生的数据。
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device) # 但是我们在 Buffer 里只需要存单个状态 st​。取用的时候，我们只要拿出 st​ 和它后面的 st+1​ 拼起来就行了。  amp_replay_buffer_size如果存满了，新来的数据会把最旧的数据挤出去（FIFO）。
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer  # amp归一化

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer  创建优化器
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
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

        self.vel_estimation_coef = vel_estimation_coef
        self.vel_estimation_warmup_iters = vel_estimation_warmup_iters
        self.terrain_recon_coef = terrain_recon_coef
        self.terrain_recon_warmup_iters = terrain_recon_warmup_iters
        self.terrain_recon_target_clip = terrain_recon_target_clip
        self.terrain_scan_dim = terrain_scan_dim
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

        if vel_estimation_coef > 0:
            print("[AMPPPO] Velocity estimation AUX TASK enabled:")
            print(f"  - coef: {vel_estimation_coef}, warmup: {vel_estimation_warmup_iters} iters")
        if terrain_recon_coef > 0:
            print("[AMPPPO] Terrain reconstruction AUX TASK enabled:")
            print(f"  - coef: {terrain_recon_coef}, warmup: {terrain_recon_warmup_iters} iters")
            print(f"  - front_only: {terrain_recon_front_only}, predict {self.terrain_recon_target_dim} points")
        if feet_height_coef > 0:
            print("[AMPPPO] Feet height AUX TASK enabled:")
            print(f"  - coef: {feet_height_coef}, warmup: {feet_height_warmup_iters} iters")

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

    def act(self, obs, critic_obs, amp_obs):
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
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, recovery_mask_t=None):
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

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.time_outs = infos["time_outs"].to(self.device).clone().bool()
        else:
            self.transition.time_outs = torch.zeros_like(dones, dtype=torch.bool, device=self.device)
        self._apply_timeout_bootstrap()

        # record the transition
        self._store_amp_transition(self.amp_transition.observations, amp_obs, recovery_mask_t)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def _apply_timeout_bootstrap(self):
        """Apply the legacy PPO truncation correction using action-time V(s_t)."""
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * self.transition.time_outs.unsqueeze(1), 1
        )

    def _store_amp_transition(self, amp_obs, next_amp_obs, recovery_mask_t=None):
        """Store one AMP transition batch.

        ``recovery_mask_t`` is accepted so specialized algorithms can route
        transitions without changing the legacy single-discriminator API.
        """
        del recovery_mask_t
        self.amp_storage.insert(amp_obs, next_amp_obs)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _get_warmup_weight(self, iteration: int, warmup_iters: int, target_weight: float) -> float:
        if warmup_iters <= 0:
            return target_weight
        if iteration >= warmup_iters:
            return target_weight
        return target_weight * (iteration / warmup_iters)

    def _amp_mini_batch_generator(self, num_updates: int, mini_batch_size: int):
        """Yield policy/expert AMP pairs for the legacy discriminator."""
        policy_generator = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
        expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)
        yield from zip(policy_generator, expert_generator)

    def _compute_single_amp_discriminator_loss(
        self,
        discriminator,
        normalizer,
        sample_amp_policy,
        sample_amp_expert,
    ):
        """Compute one discriminator branch without coupling it to PPO."""
        policy_state, policy_next_state = sample_amp_policy
        expert_state, expert_next_state = sample_amp_expert
        if normalizer is not None:
            with torch.no_grad():
                policy_state = normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = normalizer.normalize_torch(expert_next_state, self.device)

        policy_d = discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(policy_d, -torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        # Preserve the existing AMP behavior: gradient penalty is evaluated on
        # the raw expert batch rather than the normalized local tensors above.
        grad_pen_loss = discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
        total_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss
        metrics = {
            "loss": amp_loss,
            "grad_pen": grad_pen_loss,
            "policy_pred": policy_d.mean(),
            "expert_pred": expert_d.mean(),
        }
        normalizer_update = None
        if normalizer is not None:
            normalizer_update = (normalizer, policy_state, expert_state)
        return total_loss, metrics, normalizer_update

    def _compute_amp_loss(self, sample_amp):
        """Compute the legacy discriminator loss and legacy log fields."""
        sample_amp_policy, sample_amp_expert = sample_amp
        total_loss, branch_metrics, normalizer_update = self._compute_single_amp_discriminator_loss(
            self.discriminator,
            self.amp_normalizer,
            sample_amp_policy,
            sample_amp_expert,
        )
        metrics = {
            "amp": branch_metrics["loss"],
            "amp_grad_pen": branch_metrics["grad_pen"],
            "amp_policy_pred": branch_metrics["policy_pred"],
            "amp_expert_pred": branch_metrics["expert_pred"],
        }
        normalizer_updates = [] if normalizer_update is None else [normalizer_update]
        return total_loss, metrics, normalizer_updates

    def _rollout_mini_batch_generator(self):
        if self.policy.is_recurrent:
            return self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        return self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    def _get_auxiliary_sample_mask(self, rollout_data, reference_tensor):
        """Return an optional batch mask for estimator auxiliary supervision.

        Ordinary AMP tasks retain their historical full-mini-batch behavior.
        Specialized algorithms may return a boolean ``[batch_size]`` mask.
        """
        del rollout_data, reference_tensor
        return None

    @staticmethod
    def _select_auxiliary_samples(prediction, target, sample_mask):
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "Auxiliary prediction and target batch sizes must match: "
                f"{prediction.shape[0]} != {target.shape[0]}."
            )
        if sample_mask is None:
            return prediction, target
        if not isinstance(sample_mask, torch.Tensor):
            raise TypeError("Auxiliary sample mask must be a torch.Tensor or None.")
        if sample_mask.dtype != torch.bool:
            raise TypeError(f"Auxiliary sample mask must have dtype bool, got {sample_mask.dtype}.")
        if sample_mask.device != prediction.device:
            raise ValueError(
                "Auxiliary sample mask and prediction must be on the same device: "
                f"{sample_mask.device} != {prediction.device}."
            )
        if sample_mask.shape != (prediction.shape[0],):
            raise ValueError(
                "Auxiliary sample mask must have shape "
                f"({prediction.shape[0]},), got {tuple(sample_mask.shape)}."
            )
        if not torch.any(sample_mask):
            return None, None
        return prediction[sample_mask], target[sample_mask]

    @classmethod
    def _masked_auxiliary_mse(cls, prediction, target, sample_mask):
        selected_prediction, selected_target = cls._select_auxiliary_samples(
            prediction, target, sample_mask
        )
        if selected_prediction is None:
            return prediction.sum() * 0.0
        return nn.functional.mse_loss(selected_prediction, selected_target)

    @classmethod
    def _masked_terrain_reconstruction_loss(cls, prediction, target, sample_mask):
        selected_prediction, selected_target = cls._select_auxiliary_samples(
            prediction, target, sample_mask
        )
        if selected_prediction is None:
            return prediction.sum() * 0.0
        # Preserve the existing global target normalization, but compute its
        # statistics only from samples that actually receive supervision.
        target_mean = selected_target.mean()
        target_std = selected_target.std().clamp(min=1e-6)
        normalized_target = (selected_target - target_mean) / target_std
        return nn.functional.mse_loss(selected_prediction, normalized_target)

    @staticmethod
    def _compute_surrogate_loss(surrogate, surrogate_clipped, rollout_data=None, ratio=None):
        del rollout_data, ratio
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_losses(
        self,
        value_batch,
        target_values_batch,
        returns_batch,
        critic_obs_batch,
        rollout_data=None,
    ):
        del critic_obs_batch, rollout_data
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss, value_batch.sum() * 0.0, {}

    def _parameters_for_gradient_clipping(self):
        return self.policy.parameters()

    @staticmethod
    def _update_amp_normalizers(normalizer_updates):
        for normalizer, policy_state, expert_state in normalizer_updates:
            normalizer.update(policy_state.detach().cpu().numpy())
            normalizer.update(expert_state.detach().cpu().numpy())

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        amp_metric_sums = {}
        value_metric_sums = {}
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
        mean_vel_loss = 0
        mean_op_vel_loss = 0
        mean_vp_vel_loss = 0
        mean_op_supervised_loss = 0
        mean_vp_supervised_loss = 0
        uses_renet_separate_supervision = False
        mean_terrain_recon_loss = 0
        mean_feet_height_loss = 0
        auxiliary_mask_seen = False
        auxiliary_loco_samples = 0.0
        auxiliary_recovery_samples = 0.0
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
        generator = self._rollout_mini_batch_generator()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        amp_mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_generator = self._amp_mini_batch_generator(num_updates, amp_mini_batch_size)

        # iterate over batches
        for sample, sample_amp in zip(generator, amp_generator):
            rollout_data = None
            if isinstance(sample[-1], dict):
                rollout_data = sample[-1]
                sample = sample[:-1]
            (
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
            ) = sample

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
                if rollout_data is not None:
                    rollout_data = {
                        key: value.repeat(num_aug, *([1] * (value.ndim - 1)))
                        for key, value in rollout_data.items()
                    }

            # Resolve this once and reuse it for every estimator auxiliary
            # loss. RENet supplies the action-time locomotion mask; ordinary
            # AMP tasks return None and keep full-batch supervision.
            auxiliary_sample_mask = self._get_auxiliary_sample_mask(rollout_data, obs_batch)
            if auxiliary_sample_mask is not None:
                auxiliary_mask_seen = True
                num_loco_samples = float(auxiliary_sample_mask.sum().item())
                auxiliary_loco_samples += num_loco_samples
                auxiliary_recovery_samples += float(auxiliary_sample_mask.numel()) - num_loco_samples

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
            surrogate_loss = self._compute_surrogate_loss(
                surrogate, surrogate_clipped, rollout_data, ratio
            )

            # Value function loss
            value_loss, auxiliary_value_loss, value_metrics = self._compute_value_losses(
                value_batch,
                target_values_batch,
                returns_batch,
                critic_obs_batch,
                rollout_data,
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * (value_loss + auxiliary_value_loss)
                - self.entropy_coef * entropy_batch.mean()
            )

            vel_loss = torch.tensor(0.0, device=self.device)
            op_vel_loss = torch.tensor(0.0, device=self.device)
            vp_vel_loss = torch.tensor(0.0, device=self.device)
            use_renet_separate_supervision = False
            if vel_weight > 0 and hasattr(self.policy, "use_vel_estimation") and self.policy.use_vel_estimation:
                vel_estimate = self.policy.predict_velocity()
                if vel_estimate is not None:
                    vel_target = critic_obs_batch[
                        :, self.vel_obs_start_idx : self.vel_obs_start_idx + self.vel_dim
                    ].detach()
                    if isinstance(vel_estimate, dict):
                        use_renet_separate_supervision = True
                        uses_renet_separate_supervision = True
                        if "op" in vel_estimate:
                            op_vel_loss = self._masked_auxiliary_mse(
                                vel_estimate["op"], vel_target, auxiliary_sample_mask
                            )
                        if "vp" in vel_estimate:
                            vp_vel_loss = self._masked_auxiliary_mse(
                                vel_estimate["vp"], vel_target, auxiliary_sample_mask
                            )
                        vel_loss = op_vel_loss + vp_vel_loss
                    else:
                        vel_loss = self._masked_auxiliary_mse(
                            vel_estimate, vel_target, auxiliary_sample_mask
                        )

            terrain_recon_loss = torch.tensor(0.0, device=self.device)
            if terrain_weight > 0 and hasattr(self.policy, "use_terrain_recon") and self.policy.use_terrain_recon:
                terrain_pred = self.policy.predict_terrain()
                if terrain_pred is not None:
                    terrain_target_full = critic_obs_batch[:, -self.terrain_scan_dim :].detach()
                    if self.terrain_front_indices is not None:
                        terrain_target = terrain_target_full[:, self.terrain_front_indices]
                    else:
                        terrain_target = terrain_target_full
                    if self.terrain_recon_target_clip > 0:
                        terrain_target = terrain_target.clamp(
                            -self.terrain_recon_target_clip,
                            self.terrain_recon_target_clip,
                        )
                    terrain_recon_loss = self._masked_terrain_reconstruction_loss(
                        terrain_pred,
                        terrain_target,
                        auxiliary_sample_mask,
                    )

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
                    feet_height_loss = self._masked_auxiliary_mse(
                        feet_height_pred,
                        feet_height_target,
                        auxiliary_sample_mask,
                    )

            op_supervised_loss = torch.tensor(0.0, device=self.device)
            vp_supervised_loss = torch.tensor(0.0, device=self.device)
            if use_renet_separate_supervision:
                op_supervised_loss = vel_weight * op_vel_loss
                vp_supervised_loss = (
                    vel_weight * vp_vel_loss
                    + terrain_weight * terrain_recon_loss
                    + feet_height_weight * feet_height_loss
                )
                loss += op_supervised_loss + vp_supervised_loss
            else:
                loss += vel_weight * vel_loss + terrain_weight * terrain_recon_loss
                loss += feet_height_weight * feet_height_loss

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

            amp_loss, amp_metrics, normalizer_updates = self._compute_amp_loss(sample_amp)
            loss += amp_loss

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
            nn.utils.clip_grad_norm_(self._parameters_for_gradient_clipping(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            self._update_amp_normalizers(normalizer_updates)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            for key, value in amp_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                amp_metric_sums[key] = amp_metric_sums.get(key, 0.0) + value
            for key, value in value_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                value_metric_sums[key] = value_metric_sums.get(key, 0.0) + value
            mean_vel_loss += vel_loss.item()
            mean_op_vel_loss += op_vel_loss.item()
            mean_vp_vel_loss += vp_vel_loss.item()
            mean_op_supervised_loss += op_supervised_loss.item()
            mean_vp_supervised_loss += vp_supervised_loss.item()
            mean_terrain_recon_loss += terrain_recon_loss.item()
            mean_feet_height_loss += feet_height_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        amp_metrics = {key: value / num_updates for key, value in amp_metric_sums.items()}
        value_metrics = {key: value / num_updates for key, value in value_metric_sums.items()}
        mean_vel_loss /= num_updates
        mean_op_vel_loss /= num_updates
        mean_vp_vel_loss /= num_updates
        mean_op_supervised_loss /= num_updates
        mean_vp_supervised_loss /= num_updates
        mean_terrain_recon_loss /= num_updates
        mean_feet_height_loss /= num_updates
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        loss_dict.update(amp_metrics)
        loss_dict.update(value_metrics)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.vel_estimation_coef > 0:
            loss_dict["vel_estimation"] = mean_vel_loss
            loss_dict["vel_estimation_coef_current"] = vel_weight
            if uses_renet_separate_supervision:
                loss_dict["vel_estimation/op"] = mean_op_vel_loss
                loss_dict["vel_estimation/vp"] = mean_vp_vel_loss
                loss_dict["renet/op_supervised"] = mean_op_supervised_loss
                loss_dict["renet/vp_supervised"] = mean_vp_supervised_loss
        if self.terrain_recon_coef > 0:
            loss_dict["terrain_recon"] = mean_terrain_recon_loss
            loss_dict["terrain_recon_coef_current"] = terrain_weight
        if self.feet_height_coef > 0:
            loss_dict["feet_height"] = mean_feet_height_loss
            loss_dict["feet_height_coef_current"] = feet_height_weight
        if auxiliary_mask_seen:
            loss_dict["Auxiliary/loco_samples"] = auxiliary_loco_samples / num_updates
            loss_dict["Auxiliary/recovery_samples"] = auxiliary_recovery_samples / num_updates

        self.current_iteration += 1
        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        modules = self._modules_for_parameter_sync()
        model_params = [module.state_dict() for module in modules]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        for module, state_dict in zip(modules, model_params):
            module.load_state_dict(state_dict)

    def _modules_for_parameter_sync(self):
        modules = [self.policy, self.discriminator]
        if self.rnd:
            modules.append(self.rnd.predictor)
        return modules

    def _parameters_for_gradient_reduction(self):
        parameters = chain(self.policy.parameters(), self.discriminator.parameters())
        if self.rnd:
            parameters = chain(parameters, self.rnd.parameters())
        return parameters

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        all_params = list(self._parameters_for_gradient_reduction())
        # Ranks may observe different routed modes. Use zero placeholders for
        # locally unused branches so every rank reduces an identical vector.
        grad_flags = torch.tensor(
            [param.grad is not None for param in all_params],
            dtype=torch.int32,
            device=self.device,
        )
        grads = [
            param.grad.view(-1) if param.grad is not None else torch.zeros_like(param).view(-1)
            for param in all_params
        ]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(grad_flags, op=torch.distributed.ReduceOp.MAX)
        all_grads /= self.gpu_world_size

        offset = 0
        for param, has_global_grad in zip(all_params, grad_flags):
            numel = param.numel()
            if has_global_grad:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel
