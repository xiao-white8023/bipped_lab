from __future__ import annotations

import inspect
from itertools import chain

import torch

from rsl_rl.storage import ReplayBuffer

from .amp_ppo import AMPPPO


class RENetAMPPPO(AMPPPO):
    """RENet PPO with independently routed locomotion and recovery AMP."""

    AMP_BUNDLE_KEYS = frozenset(("loco", "recovery"))

    def __init__(
        self,
        *args,
        recovery_discriminator=None,
        recovery_amp_data=None,
        recovery_amp_normalizer=None,
        recovery_critic=None,
        recovery_state_machine_enabled: bool = False,
        enable_recovery_learning: bool = False,
        recovery_task_adv_weight: float = 2.5,
        recovery_amp_adv_weight: float = 1.0,
        recovery_reg_adv_weight: float = 0.1,
        recovery_ppo_min_rollout_samples: int = 2048,
        recovery_drec_min_replay_samples: int = 2048,
        **kwargs,
    ):
        # RENet grew out of the visual/MoE configs; these keys are irrelevant
        # for the AMP optimizer and are kept here only for config compatibility.
        for key in (
            "obs_dim",
            "use_moe_balance_loss",
            "moe_balance_coef",
            "use_moe_gate_entropy_loss",
            "moe_gate_entropy_coef",
        ):
            kwargs.pop(key, None)

        renet_aux_keys = (
            "feet_height_coef",
            "feet_height_warmup_iters",
            "feet_height_dim",
            "feet_height_in_critic_offset",
        )
        renet_aux_cfg = {key: kwargs.get(key) for key in renet_aux_keys if key in kwargs}

        parent_params = inspect.signature(AMPPPO.__init__).parameters
        parent_accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parent_params.values())
        if not parent_accepts_kwargs:
            for key in renet_aux_keys:
                if key not in parent_params:
                    kwargs.pop(key, None)

        if recovery_discriminator is None or recovery_amp_data is None or recovery_amp_normalizer is None:
            raise ValueError(
                "RENetAMPPPO requires recovery_discriminator, recovery_amp_data, "
                "and recovery_amp_normalizer."
            )
        if recovery_critic is None:
            raise ValueError("RENetAMPPPO requires a RecoveryCritic instance.")
        if enable_recovery_learning and not recovery_state_machine_enabled:
            raise ValueError(
                "enable_recovery_learning=True requires the environment Recovery state machine to be enabled."
            )
        for name, value in (
            ("recovery_ppo_min_rollout_samples", recovery_ppo_min_rollout_samples),
            ("recovery_drec_min_replay_samples", recovery_drec_min_replay_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}.")

        super().__init__(*args, **kwargs)

        self.recovery_state_machine_enabled = bool(recovery_state_machine_enabled)
        self.enable_recovery_learning = bool(enable_recovery_learning)
        self.recovery_advantage_weights = (
            float(recovery_task_adv_weight),
            float(recovery_amp_adv_weight),
            float(recovery_reg_adv_weight),
        )
        self.recovery_ppo_min_rollout_samples = recovery_ppo_min_rollout_samples
        self.recovery_drec_min_replay_samples = recovery_drec_min_replay_samples
        self._current_recovery_rollout_samples = 0
        self._recovery_ppo_ready = False
        self._drec_update_ready = False
        self.drec_reward_ready = False
        self.recovery_discriminator_updated_this_update = False
        self._recovery_discriminator_loss_computed_this_update = False
        self.recovery_critic = recovery_critic.to(self.device)
        self._assert_independent_recovery_critics()

        # Explicit locomotion names plus legacy aliases kept by AMPPPO.
        self.discriminator_loco = self.discriminator
        self.amp_data_loco = self.amp_data
        self.amp_normalizer_loco = self.amp_normalizer
        self.amp_storage_loco = self.amp_storage

        self.discriminator_recovery = recovery_discriminator.to(self.device)
        self.amp_data_recovery = recovery_amp_data
        self.amp_normalizer_recovery = recovery_amp_normalizer
        self.amp_storage_recovery = ReplayBuffer(
            self.discriminator_recovery.input_dim // 2,
            self.amp_storage_loco.buffer_size,
            self.device,
        )

        for name, discriminator in (
            ("locomotion", self.discriminator_loco),
            ("recovery", self.discriminator_recovery),
        ):
            if discriminator.input_dim <= 0 or discriminator.input_dim % 2 != 0:
                raise ValueError(
                    f"{name} discriminator input_dim must be a positive even number, "
                    f"got {discriminator.input_dim}."
                )

        # The shared optimizer has explicit, independently named groups for
        # both discriminator trunks and heads.
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "amp_trunk":
                param_group["name"] = "loco_amp_trunk"
            elif param_group.get("name") == "amp_head":
                param_group["name"] = "loco_amp_head"
        self.optimizer.add_param_group(
            {
                "params": self.discriminator_recovery.trunk.parameters(),
                "weight_decay": 10e-4,
                "name": "recovery_amp_trunk",
            }
        )
        self._last_recovery_learning_diagnostics = {
            "RecoveryLearning/enabled": float(self.enable_recovery_learning),
            "Rollout/loco_samples": 0.0,
            "Rollout/recovery_samples": 0.0,
            "Rollout/enter_recovery_count": 0.0,
            "Rollout/exit_recovery_count": 0.0,
            "Rollout/recovery_failed_count": 0.0,
            "RecoveryValue/task_loss": 0.0,
            "RecoveryValue/amp_loss": 0.0,
            "RecoveryValue/reg_loss": 0.0,
            "RecoveryWarmup/ppo_min_samples": float(self.recovery_ppo_min_rollout_samples),
            "RecoveryWarmup/ppo_rollout_samples": 0.0,
            "RecoveryWarmup/ppo_ready": 0.0,
            "RecoveryWarmup/drec_min_replay_samples": float(self.recovery_drec_min_replay_samples),
            "RecoveryWarmup/drec_replay_samples": 0.0,
            "RecoveryWarmup/drec_update_ready": 0.0,
            "RecoveryWarmup/drec_reward_ready": 0.0,
            "RecoveryWarmup/drec_updated_this_update": 0.0,
            "RecoveryWarmup/recovery_update_skipped": 1.0,
        }

        # V_rec_amp has no separate delay or ramp. It follows the same on-policy
        # Recovery PPO gate as the other two Recovery value heads.
        self.optimizer.add_param_group(
            {
                "params": self.discriminator_recovery.amp_linear.parameters(),
                "weight_decay": 10e-2,
                "name": "recovery_amp_head",
            }
        )
        self.optimizer.add_param_group(
            {
                "params": self.recovery_critic.parameters(),
                "name": "recovery_critic",
            }
        )

        # Keep RENet-specific attributes available even when running against an
        # older AMPPPO implementation that does not declare these parameters.
        if renet_aux_cfg:
            self.feet_height_coef = renet_aux_cfg.get("feet_height_coef", 0.0)
            self.feet_height_warmup_iters = renet_aux_cfg.get("feet_height_warmup_iters", 0)
            self.feet_height_dim = renet_aux_cfg.get("feet_height_dim", 2)
            self.feet_height_in_critic_offset = renet_aux_cfg.get("feet_height_in_critic_offset", 0)
            self.feet_height_obs_start_idx = (
                (self.critic_history_len - 1) * self.single_critic_dim + self.feet_height_in_critic_offset
            )

    def _assert_independent_recovery_critics(self):
        parameter_ids = {
            "task": {id(param) for param in self.recovery_critic.task_critic.parameters()},
            "amp": {id(param) for param in self.recovery_critic.amp_critic.parameters()},
            "reg": {id(param) for param in self.recovery_critic.reg_critic.parameters()},
        }
        if parameter_ids["task"] & parameter_ids["amp"]:
            raise RuntimeError("Recovery task_critic and amp_critic share parameters.")
        if parameter_ids["task"] & parameter_ids["reg"]:
            raise RuntimeError("Recovery task_critic and reg_critic share parameters.")
        if parameter_ids["amp"] & parameter_ids["reg"]:
            raise RuntimeError("Recovery amp_critic and reg_critic share parameters.")

    def _validate_recovery_mask(self, recovery_mask_t, batch_size: int) -> torch.Tensor:
        if recovery_mask_t is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if not isinstance(recovery_mask_t, torch.Tensor):
            raise TypeError("recovery_mask_t must be a torch.Tensor or None.")
        recovery_mask_t = recovery_mask_t.to(device=self.device)
        if recovery_mask_t.dtype != torch.bool:
            raise TypeError(f"recovery_mask_t must have dtype bool, got {recovery_mask_t.dtype}.")
        if recovery_mask_t.shape != (batch_size,):
            raise ValueError(
                f"recovery_mask_t must have shape ({batch_size},), got {tuple(recovery_mask_t.shape)}."
            )
        return recovery_mask_t

    def _validate_amp_obs_bundle(self, bundle, batch_size: int | None = None):
        """Validate the two independent AMP observation tensors."""
        if not isinstance(bundle, dict):
            raise TypeError("RENet AMP observations must be a dict bundle.")
        if set(bundle) != self.AMP_BUNDLE_KEYS:
            raise KeyError(
                "RENet AMP observation bundle must have exact keys "
                f"{sorted(self.AMP_BUNDLE_KEYS)}, got {sorted(bundle)}."
            )

        expected_widths = {
            "loco": self.discriminator_loco.input_dim // 2,
            "recovery": self.discriminator_recovery.input_dim // 2,
        }
        inferred_batch_size = batch_size
        reference_device = None
        for name in ("loco", "recovery"):
            amp_obs = bundle[name]
            if not isinstance(amp_obs, torch.Tensor):
                raise TypeError(f"AMP bundle['{name}'] must be a torch.Tensor.")
            if amp_obs.ndim != 2:
                raise ValueError(
                    f"AMP bundle['{name}'] must be 2-D, got {tuple(amp_obs.shape)}."
                )
            if not torch.is_floating_point(amp_obs):
                raise TypeError(
                    f"AMP bundle['{name}'] must be floating point, got {amp_obs.dtype}."
                )
            if amp_obs.shape[1] != expected_widths[name]:
                raise ValueError(
                    f"AMP bundle['{name}'] width must be {expected_widths[name]}, "
                    f"got {amp_obs.shape[1]}."
                )
            if inferred_batch_size is None:
                inferred_batch_size = amp_obs.shape[0]
            elif amp_obs.shape[0] != inferred_batch_size:
                raise ValueError(
                    f"AMP bundle['{name}'] batch size must be {inferred_batch_size}, "
                    f"got {amp_obs.shape[0]}."
                )
            if reference_device is None:
                reference_device = amp_obs.device
            elif amp_obs.device != reference_device:
                raise ValueError(
                    "Locomotion and Recovery AMP observations must share a device, got "
                    f"{reference_device} and {amp_obs.device}."
                )
        return inferred_batch_size

    def act(self, obs, critic_obs, amp_obs):
        self._validate_amp_obs_bundle(amp_obs, batch_size=obs.shape[0])
        actions = super().act(obs, critic_obs, amp_obs)
        if self.recovery_state_machine_enabled and self.enable_recovery_learning:
            recovery_values = self.recovery_critic(critic_obs)
            self.transition.recovery_task_value = recovery_values["task"].detach()
            self.transition.recovery_amp_value = recovery_values["amp"].detach()
            self.transition.recovery_reg_value = recovery_values["reg"].detach()
        return actions

    def _read_bool_transition_flag(self, infos, key: str, batch_size: int) -> torch.Tensor:
        value = infos.get(key)
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"infos['{key}'] must be a torch.Tensor.")
        value = value.to(self.device)
        if value.shape != (batch_size,):
            raise ValueError(
                f"infos['{key}'] must have shape ({batch_size},), got {tuple(value.shape)}."
            )
        return value.bool().clone()

    def _read_recovery_reward_interface(self, infos, batch_size: int):
        keys = (
            "recovery_task_reward",
            "recovery_amp_reward",
            "recovery_reg_reward",
        )
        present = [key in infos for key in keys]
        if any(present) and not all(present):
            missing = [key for key, is_present in zip(keys, present) if not is_present]
            raise RuntimeError(
                "Recovery reward interface is partial; provide all three streams or none. Missing: "
                + ", ".join(missing)
            )
        if not all(present):
            zeros = torch.zeros(batch_size, dtype=torch.float, device=self.device)
            return (zeros, zeros.clone(), zeros.clone()), torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )

        rewards = []
        for key in keys:
            reward = infos[key]
            if not isinstance(reward, torch.Tensor):
                raise TypeError(f"infos['{key}'] must be a torch.Tensor.")
            reward = reward.to(self.device)
            if reward.shape != (batch_size,):
                raise ValueError(
                    f"infos['{key}'] must have shape ({batch_size},), got {tuple(reward.shape)}."
                )
            rewards.append(reward.clone())
        return tuple(rewards), torch.ones(batch_size, dtype=torch.bool, device=self.device)

    def _read_timeout_bootstrap_values(self, timeout_bootstrap_values, batch_size: int):
        field_map = {
            "timeout_loco_values": "timeout_loco_value",
            "timeout_rec_task_values": "timeout_rec_task_value",
            "timeout_rec_amp_values": "timeout_rec_amp_value",
            "timeout_rec_reg_values": "timeout_rec_reg_value",
        }
        if timeout_bootstrap_values is None:
            timeout_bootstrap_values = {}
        if not isinstance(timeout_bootstrap_values, dict):
            raise TypeError("timeout_bootstrap_values must be a dict or None.")
        unexpected = set(timeout_bootstrap_values) - set(field_map)
        if unexpected:
            raise KeyError(f"Unexpected timeout bootstrap fields: {sorted(unexpected)}.")

        reference = self.transition.values
        expected_shape = (batch_size, 1)
        for key, transition_field in field_map.items():
            value = timeout_bootstrap_values.get(key)
            if value is None:
                value = torch.zeros(expected_shape, dtype=reference.dtype, device=self.device)
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{key} must be a torch.Tensor.")
                if value.device != reference.device:
                    raise ValueError(
                        f"{key} must be on {reference.device}, got {value.device}."
                    )
                if value.dtype != reference.dtype:
                    raise TypeError(
                        f"{key} must have dtype {reference.dtype}, got {value.dtype}."
                    )
                if value.shape != expected_shape:
                    raise ValueError(
                        f"{key} must have shape {expected_shape}, got {tuple(value.shape)}."
                    )
                value = value.clone()
            setattr(self.transition, transition_field, value)

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        amp_obs,
        recovery_mask_t=None,
        timeout_bootstrap_values=None,
    ):
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("rewards must be a torch.Tensor.")
        if rewards.ndim != 1:
            raise ValueError(f"rewards must have shape [num_envs], got {tuple(rewards.shape)}.")
        if not torch.is_floating_point(rewards):
            raise TypeError(f"rewards must be floating point, got {rewards.dtype}.")
        expected_device = torch.device(self.device)
        if rewards.device != expected_device:
            raise ValueError(f"rewards must be on {expected_device}, got {rewards.device}.")
        batch_size = rewards.shape[0]
        recovery_mask_t = self._validate_recovery_mask(recovery_mask_t, batch_size)
        recovery_rewards, rewards_valid = self._read_recovery_reward_interface(infos, batch_size)
        for name, reward in zip(("task", "amp", "reg"), recovery_rewards):
            if reward.dtype != rewards.dtype:
                raise TypeError(
                    f"Recovery {name} reward must have dtype {rewards.dtype}, got {reward.dtype}."
                )

        # Final no-leakage guard at the rollout boundary. This uses the mask
        # captured before env.step(), never the environment's updated mode.
        locomotion_mask_t = ~recovery_mask_t
        rewards = rewards * locomotion_mask_t.to(dtype=rewards.dtype)
        recovery_rewards = tuple(
            reward * recovery_mask_t.to(dtype=reward.dtype) for reward in recovery_rewards
        )
        rewards_valid = rewards_valid & recovery_mask_t

        self.transition.recovery_mask_t = recovery_mask_t.clone()
        self.transition.enter_recovery = self._read_bool_transition_flag(
            infos, "enter_recovery", batch_size
        )
        self.transition.exit_recovery = self._read_bool_transition_flag(
            infos, "exit_recovery", batch_size
        )
        self.transition.recovery_failed = self._read_bool_transition_flag(
            infos, "recovery_failed", batch_size
        )
        self.transition.recovery_task_reward = recovery_rewards[0]
        self.transition.recovery_amp_reward = recovery_rewards[1]
        self.transition.recovery_reg_reward = recovery_rewards[2]
        self.transition.recovery_rewards_valid = rewards_valid
        self._read_timeout_bootstrap_values(timeout_bootstrap_values, batch_size)

        super().process_env_step(rewards, dones, infos, amp_obs, recovery_mask_t)

    def _apply_timeout_bootstrap(self):
        if not self.recovery_state_machine_enabled:
            # Preserve the recovery-disabled baseline path, including its
            # historical action-time V(s_t) timeout correction.
            return super()._apply_timeout_bootstrap()
        if self.enable_recovery_learning:
            # Segmented GAE below is solely responsible for mode-specific
            # terminal bootstrap. Applying a reward correction here as well
            # would double-bootstrap timeout transitions.
            return

        # While Recovery learning is disabled, the legacy non-segmented GAE is
        # still used. Correct only locomotion-owned timeout transitions once,
        # using V_loco(s_terminal); Recovery terminal values remain safely
        # stored for the future segmented path.
        locomotion_timeout = (
            self.transition.time_outs
            & ~self.transition.recovery_mask_t
            & ~self.transition.recovery_failed
        )
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.timeout_loco_value * locomotion_timeout.unsqueeze(1),
            1,
        )

    def compute_returns(self, last_critic_obs):
        if not (self.recovery_state_machine_enabled and self.enable_recovery_learning):
            return super().compute_returns(last_critic_obs)

        with torch.no_grad():
            last_loco_values = self.policy.evaluate(last_critic_obs).detach()
            last_recovery_values = self.recovery_critic(last_critic_obs)

        recovery_mask = self.storage.recovery_masks
        if torch.any(recovery_mask & ~self.storage.recovery_rewards_valid):
            raise RuntimeError(
                "Recovery learning is enabled, but one or more Recovery transitions have no explicit "
                "task/AMP/regularization reward streams. Reward composition is intentionally not inferred."
            )

        time_outs = self.storage.time_outs
        env_terminal = (
            (self.storage.dones.bool() & ~time_outs)
            | self.storage.recovery_failed
        )
        locomotion_mask = ~recovery_mask
        loco_returns, loco_advantages = self.storage.compute_segmented_gae(
            rewards=self.storage.rewards,
            values=self.storage.values,
            last_values=last_loco_values,
            sample_mask=locomotion_mask,
            trace_end=self.storage.enter_recovery,
            env_terminal=env_terminal,
            time_outs=time_outs,
            gamma=self.gamma,
            lam=self.lam,
            timeout_bootstrap_values=self.storage.timeout_loco_values,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )
        self.storage.returns.copy_(loco_returns)
        self.storage.advantages.copy_(loco_advantages)

        recovery_trace_end = self.storage.exit_recovery | self.storage.recovery_failed
        recovery_specs = (
            (
                "task",
                self.storage.recovery_task_rewards,
                self.storage.recovery_task_values,
                self.storage.recovery_task_returns,
                self.storage.recovery_task_advantages,
                self.storage.timeout_rec_task_values,
            ),
            (
                "amp",
                self.storage.recovery_amp_rewards,
                self.storage.recovery_amp_values,
                self.storage.recovery_amp_returns,
                self.storage.recovery_amp_advantages,
                self.storage.timeout_rec_amp_values,
            ),
            (
                "reg",
                self.storage.recovery_reg_rewards,
                self.storage.recovery_reg_values,
                self.storage.recovery_reg_returns,
                self.storage.recovery_reg_advantages,
                self.storage.timeout_rec_reg_values,
            ),
        )
        for name, rewards, values, returns_buffer, advantages_buffer, timeout_values in recovery_specs:
            returns, advantages = self.storage.compute_segmented_gae(
                rewards=rewards,
                values=values,
                last_values=last_recovery_values[name].detach(),
                sample_mask=recovery_mask,
                trace_end=recovery_trace_end,
                env_terminal=env_terminal,
                time_outs=time_outs,
                gamma=self.gamma,
                lam=self.lam,
                timeout_bootstrap_values=timeout_values,
                # Recovery advantages are always normalized independently over
                # Recovery samples from this rollout.
                normalize_advantage=True,
            )
            returns_buffer.copy_(returns)
            advantages_buffer.copy_(advantages)

    def _store_amp_transition(self, amp_obs, next_amp_obs, recovery_mask_t=None):
        batch_size = self._validate_amp_obs_bundle(amp_obs)
        self._validate_amp_obs_bundle(next_amp_obs, batch_size=batch_size)
        recovery_mask_t = self._validate_recovery_mask(recovery_mask_t, batch_size)
        locomotion_mask_t = ~recovery_mask_t
        if locomotion_mask_t.any():
            self.amp_storage_loco.insert(
                amp_obs["loco"][locomotion_mask_t],
                next_amp_obs["loco"][locomotion_mask_t],
            )
        if recovery_mask_t.any():
            self.amp_storage_recovery.insert(
                amp_obs["recovery"][recovery_mask_t],
                next_amp_obs["recovery"][recovery_mask_t],
            )

    def predict_routed_amp_reward(
        self,
        amp_obs,
        next_amp_obs,
        task_reward,
        recovery_mask_t=None,
    ):
        """Route each ``s_t -> s_t+1`` reward by the mode that produced ``a_t``."""
        batch_size = self._validate_amp_obs_bundle(amp_obs)
        self._validate_amp_obs_bundle(next_amp_obs, batch_size=batch_size)
        if task_reward.shape != (batch_size,):
            raise ValueError(
                f"task_reward must have shape ({batch_size},), got {tuple(task_reward.shape)}."
            )

        recovery_mask_t = self._validate_recovery_mask(recovery_mask_t, batch_size)
        locomotion_mask_t = ~recovery_mask_t
        routed_reward = torch.zeros_like(task_reward)
        routed_logits = torch.zeros(
            (batch_size, 1),
            dtype=amp_obs["loco"].dtype,
            device=self.device,
        )

        if locomotion_mask_t.any():
            loco_reward, loco_logits = self.discriminator_loco.predict_amp_reward(
                amp_obs["loco"][locomotion_mask_t],
                next_amp_obs["loco"][locomotion_mask_t],
                task_reward[locomotion_mask_t],
                normalizer=self.amp_normalizer_loco,
            )
            routed_reward[locomotion_mask_t] = loco_reward
            routed_logits[locomotion_mask_t] = loco_logits
        if recovery_mask_t.any() and self.drec_reward_ready:
            recovery_reward, recovery_logits = self.discriminator_recovery.predict_amp_reward(
                amp_obs["recovery"][recovery_mask_t],
                next_amp_obs["recovery"][recovery_mask_t],
                task_reward[recovery_mask_t],
                normalizer=self.amp_normalizer_recovery,
            )
            routed_reward[recovery_mask_t] = recovery_reward
            routed_logits[recovery_mask_t] = recovery_logits
        return routed_reward, routed_logits

    def _rollout_mini_batch_generator(self):
        if self.policy.is_recurrent:
            return self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                include_recovery_data=True,
            )
        return self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            include_recovery_data=True,
        )

    def _get_auxiliary_sample_mask(self, rollout_data, reference_tensor):
        """Exclude action-time Recovery samples from OP/VP estimator supervision."""
        if rollout_data is None or "recovery_mask_t" not in rollout_data:
            raise RuntimeError("RENet auxiliary supervision requires action-time recovery_mask_t metadata.")
        recovery_mask = rollout_data["recovery_mask_t"]
        if not isinstance(recovery_mask, torch.Tensor):
            raise TypeError("rollout_data['recovery_mask_t'] must be a torch.Tensor.")
        if recovery_mask.dtype != torch.bool:
            raise TypeError(
                "rollout_data['recovery_mask_t'] must have dtype bool, "
                f"got {recovery_mask.dtype}."
            )
        if recovery_mask.shape == (reference_tensor.shape[0], 1):
            recovery_mask = recovery_mask.squeeze(-1)
        elif recovery_mask.shape != (reference_tensor.shape[0],):
            raise ValueError(
                "rollout_data['recovery_mask_t'] must have shape "
                f"({reference_tensor.shape[0]}, 1) or ({reference_tensor.shape[0]},), "
                f"got {tuple(recovery_mask.shape)}."
            )
        return (~recovery_mask).to(device=reference_tensor.device)

    @staticmethod
    def combine_recovery_advantages(
        task_advantage,
        amp_advantage,
        reg_advantage,
        weights: tuple[float, float, float] | None = None,
    ):
        """Combine already independently normalized Recovery group advantages once."""
        if weights is None:
            return None
        if len(weights) != 3:
            raise ValueError("Recovery advantage weights must contain task, AMP, and regularization weights.")
        return (
            weights[0] * task_advantage
            + weights[1] * amp_advantage
            + weights[2] * reg_advantage
        )

    def _compute_surrogate_loss(self, surrogate, surrogate_clipped, rollout_data=None, ratio=None):
        terms = torch.max(surrogate, surrogate_clipped)
        if not (self.recovery_state_machine_enabled and self.enable_recovery_learning):
            # Identical to the existing PPO reduction when Recovery learning is off.
            return terms.mean()
        if rollout_data is None:
            raise RuntimeError("Mode-aware PPO requires Recovery rollout metadata.")

        recovery_mask = rollout_data["recovery_mask_t"].squeeze(-1).bool()
        locomotion_mask = ~recovery_mask
        if torch.any(locomotion_mask):
            locomotion_term = terms[locomotion_mask].mean()
        else:
            locomotion_term = terms.sum() * 0.0

        if not self._recovery_ppo_ready:
            return locomotion_term

        recovery_advantage = self.combine_recovery_advantages(
            rollout_data["recovery_task_advantages"].squeeze(-1),
            rollout_data["recovery_amp_advantages"].squeeze(-1),
            rollout_data["recovery_reg_advantages"].squeeze(-1),
            weights=self.recovery_advantage_weights,
        )
        if recovery_advantage is None:
            return locomotion_term
        if ratio is None:
            raise RuntimeError("Recovery Actor PPO term requires the current/old policy ratio.")

        recovery_surrogate = -recovery_advantage * ratio
        recovery_surrogate_clipped = -recovery_advantage * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        recovery_terms = torch.max(recovery_surrogate, recovery_surrogate_clipped)
        recovery_term = (
            recovery_terms[recovery_mask].mean()
            if torch.any(recovery_mask)
            else recovery_terms.sum() * 0.0
        )
        return locomotion_term + recovery_term

    def _masked_value_loss(self, prediction, old_prediction, returns, mask):
        mask = mask.bool().expand_as(prediction)
        if not torch.any(mask):
            return prediction.sum() * 0.0
        if self.use_clipped_value_loss:
            prediction_clipped = old_prediction + (prediction - old_prediction).clamp(
                -self.clip_param, self.clip_param
            )
            losses = torch.square(prediction - returns)
            losses_clipped = torch.square(prediction_clipped - returns)
            return torch.max(losses, losses_clipped)[mask].mean()
        return torch.square(returns - prediction)[mask].mean()

    def _compute_value_losses(
        self,
        value_batch,
        target_values_batch,
        returns_batch,
        critic_obs_batch,
        rollout_data=None,
    ):
        if not (self.recovery_state_machine_enabled and self.enable_recovery_learning):
            return super()._compute_value_losses(
                value_batch,
                target_values_batch,
                returns_batch,
                critic_obs_batch,
                rollout_data,
            )
        if rollout_data is None:
            raise RuntimeError("Recovery value losses require Recovery rollout metadata.")

        recovery_mask = rollout_data["recovery_mask_t"].bool()
        locomotion_mask = ~recovery_mask
        locomotion_value_loss = self._masked_value_loss(
            value_batch,
            target_values_batch,
            returns_batch,
            locomotion_mask,
        )

        if not self._recovery_ppo_ready:
            differentiable_zero = value_batch.sum() * 0.0
            metrics = {
                "RecoveryValue/task_loss": differentiable_zero,
                "RecoveryValue/amp_loss": differentiable_zero,
                "RecoveryValue/reg_loss": differentiable_zero,
            }
            return locomotion_value_loss, differentiable_zero, metrics

        recovery_predictions = self.recovery_critic(critic_obs_batch)
        recovery_losses = {}
        for name in ("task", "amp", "reg"):
            recovery_losses[name] = self._masked_value_loss(
                recovery_predictions[name],
                rollout_data[f"recovery_{name}_values"],
                rollout_data[f"recovery_{name}_returns"],
                recovery_mask,
            )
        auxiliary_value_loss = sum(recovery_losses.values())
        metrics = {
            "RecoveryValue/task_loss": recovery_losses["task"],
            "RecoveryValue/amp_loss": recovery_losses["amp"],
            "RecoveryValue/reg_loss": recovery_losses["reg"],
        }
        return locomotion_value_loss, auxiliary_value_loss, metrics

    def _parameters_for_gradient_clipping(self):
        return chain(self.policy.parameters(), self.recovery_critic.parameters())

    def _prepare_recovery_warmup_for_update(self):
        """Snapshot the two independent Recovery sample gates for this update."""
        self._current_recovery_rollout_samples = int(self.storage.recovery_masks.sum().item())
        self._recovery_ppo_ready = (
            self._current_recovery_rollout_samples >= self.recovery_ppo_min_rollout_samples
        )
        self._drec_update_ready = (
            int(self.amp_storage_recovery.num_samples) >= self.recovery_drec_min_replay_samples
        )
        self.recovery_discriminator_updated_this_update = False
        self._recovery_discriminator_loss_computed_this_update = False

    def _capture_recovery_rollout_diagnostics(self):
        recovery_mask = self.storage.recovery_masks
        recovery_samples = self._current_recovery_rollout_samples
        drec_replay_samples = int(self.amp_storage_recovery.num_samples)
        diagnostics = {
            "RecoveryLearning/enabled": float(self.enable_recovery_learning),
            "Rollout/loco_samples": float(recovery_mask.numel() - recovery_samples),
            "Rollout/recovery_samples": float(recovery_samples),
            "Rollout/enter_recovery_count": float(self.storage.enter_recovery.sum().item()),
            "Rollout/exit_recovery_count": float(self.storage.exit_recovery.sum().item()),
            "Rollout/recovery_failed_count": float(self.storage.recovery_failed.sum().item()),
            "RecoveryValue/task_loss": 0.0,
            "RecoveryValue/amp_loss": 0.0,
            "RecoveryValue/reg_loss": 0.0,
            "RecoveryWarmup/ppo_min_samples": float(self.recovery_ppo_min_rollout_samples),
            "RecoveryWarmup/ppo_rollout_samples": float(recovery_samples),
            "RecoveryWarmup/ppo_ready": float(self._recovery_ppo_ready),
            "RecoveryWarmup/drec_min_replay_samples": float(self.recovery_drec_min_replay_samples),
            "RecoveryWarmup/drec_replay_samples": float(drec_replay_samples),
            "RecoveryWarmup/drec_update_ready": float(self._drec_update_ready),
            "RecoveryWarmup/drec_reward_ready": float(self.drec_reward_ready),
            "RecoveryWarmup/drec_updated_this_update": 0.0,
            "RecoveryWarmup/recovery_update_skipped": float(not self._recovery_ppo_ready),
        }
        if self.enable_recovery_learning and torch.any(recovery_mask):
            for name in ("task", "amp", "reg"):
                advantage = getattr(self.storage, f"recovery_{name}_advantages")[recovery_mask]
                diagnostics[f"RecoveryAdv/{name}_mean"] = float(advantage.mean().item())
                diagnostics[f"RecoveryAdv/{name}_std"] = float(
                    advantage.std(unbiased=False).item()
                )
        self._last_recovery_learning_diagnostics = diagnostics

    def update(self):
        self._prepare_recovery_warmup_for_update()
        self._capture_recovery_rollout_diagnostics()
        loss_dict = super().update()
        self.recovery_discriminator_updated_this_update = (
            self._recovery_discriminator_loss_computed_this_update
        )
        if self.recovery_discriminator_updated_this_update:
            self.drec_reward_ready = True
        for name in ("task", "amp", "reg"):
            key = f"RecoveryValue/{name}_loss"
            if key in loss_dict:
                self._last_recovery_learning_diagnostics[key] = float(loss_dict[key])
        self._last_recovery_learning_diagnostics.update(
            {
                "RecoveryWarmup/drec_reward_ready": float(self.drec_reward_ready),
                "RecoveryWarmup/drec_updated_this_update": float(
                    self.recovery_discriminator_updated_this_update
                ),
            }
        )
        return loss_dict

    def get_recovery_learning_diagnostics(self):
        return dict(self._last_recovery_learning_diagnostics)

    def get_recovery_warmup_state(self):
        """Return only the persistent D_REC reward trust state."""
        return {"drec_reward_ready": bool(self.drec_reward_ready)}

    def load_recovery_warmup_state(self, state):
        """Strictly restore the persistent D_REC reward trust state."""
        if not isinstance(state, dict):
            raise TypeError("Recovery warm-up state must be a dict.")
        if set(state) != {"drec_reward_ready"}:
            raise ValueError(
                "Recovery warm-up state must contain exactly the field 'drec_reward_ready'."
            )
        if not isinstance(state["drec_reward_ready"], bool):
            raise TypeError("Recovery warm-up state 'drec_reward_ready' must be a bool.")
        self.drec_reward_ready = state["drec_reward_ready"]

    def _amp_mini_batch_generator(self, num_updates: int, mini_batch_size: int):
        loco_policy_generator = self.amp_storage_loco.feed_forward_generator(num_updates, mini_batch_size)
        loco_expert_generator = self.amp_data_loco.feed_forward_generator(num_updates, mini_batch_size)

        recovery_num_samples = self.amp_storage_recovery.num_samples
        use_recovery = recovery_num_samples >= self.recovery_drec_min_replay_samples
        self._drec_update_ready = use_recovery
        if use_recovery:
            recovery_policy_generator = self.amp_storage_recovery.feed_forward_generator(
                num_updates, mini_batch_size
            )
            recovery_expert_generator = self.amp_data_recovery.feed_forward_generator(num_updates, mini_batch_size)

        for _ in range(num_updates):
            batch = {
                "loco_policy": next(loco_policy_generator),
                "loco_expert": next(loco_expert_generator),
                "recovery_policy": None,
                "recovery_expert": None,
                "recovery_num_samples": recovery_num_samples,
            }
            if use_recovery:
                batch["recovery_policy"] = next(recovery_policy_generator)
                batch["recovery_expert"] = next(recovery_expert_generator)
            yield batch

    def _compute_amp_loss(self, sample_amp):
        loco_loss, loco_metrics, loco_normalizer_update = self._compute_single_amp_discriminator_loss(
            self.discriminator_loco,
            self.amp_normalizer_loco,
            sample_amp["loco_policy"],
            sample_amp["loco_expert"],
        )
        total_loss = loco_loss
        normalizer_updates = [] if loco_normalizer_update is None else [loco_normalizer_update]
        metrics = {
            # Legacy fields continue to mean locomotion AMP.
            "amp": loco_metrics["loss"],
            "amp_grad_pen": loco_metrics["grad_pen"],
            "amp_policy_pred": loco_metrics["policy_pred"],
            "amp_expert_pred": loco_metrics["expert_pred"],
            "amp/loco_loss": loco_metrics["loss"],
            "amp/loco_grad_pen": loco_metrics["grad_pen"],
            "amp/loco_policy_pred": loco_metrics["policy_pred"],
            "amp/loco_expert_pred": loco_metrics["expert_pred"],
            "amp/recovery_loss": 0.0,
            "amp/recovery_grad_pen": 0.0,
            "amp/recovery_policy_pred": 0.0,
            "amp/recovery_expert_pred": 0.0,
            "amp/recovery_num_samples": float(sample_amp["recovery_num_samples"]),
        }

        if sample_amp["recovery_policy"] is not None:
            recovery_loss, recovery_metrics, recovery_normalizer_update = (
                self._compute_single_amp_discriminator_loss(
                    self.discriminator_recovery,
                    self.amp_normalizer_recovery,
                    sample_amp["recovery_policy"],
                    sample_amp["recovery_expert"],
                )
            )
            self._recovery_discriminator_loss_computed_this_update = True
            total_loss = total_loss + recovery_loss
            metrics.update(
                {
                    "amp/recovery_loss": recovery_metrics["loss"],
                    "amp/recovery_grad_pen": recovery_metrics["grad_pen"],
                    "amp/recovery_policy_pred": recovery_metrics["policy_pred"],
                    "amp/recovery_expert_pred": recovery_metrics["expert_pred"],
                }
            )
            if recovery_normalizer_update is not None:
                normalizer_updates.append(recovery_normalizer_update)

        return total_loss, metrics, normalizer_updates

    def _modules_for_parameter_sync(self):
        modules = [
            self.policy,
            self.recovery_critic,
            self.discriminator_loco,
            self.discriminator_recovery,
        ]
        if self.rnd:
            modules.append(self.rnd.predictor)
        return modules

    def _parameters_for_gradient_reduction(self):
        parameters = chain(
            self.policy.parameters(),
            self.recovery_critic.parameters(),
            self.discriminator_loco.parameters(),
            self.discriminator_recovery.parameters(),
        )
        if self.rnd:
            parameters = chain(parameters, self.rnd.parameters())
        return parameters
