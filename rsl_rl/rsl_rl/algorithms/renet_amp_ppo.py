from __future__ import annotations

import inspect
from itertools import chain

import torch

from rsl_rl.storage import ReplayBuffer

from .amp_ppo import AMPPPO


class RENetAMPPPO(AMPPPO):
    """RENet PPO with independently routed locomotion and recovery AMP."""

    def __init__(
        self,
        *args,
        recovery_discriminator=None,
        recovery_amp_data=None,
        recovery_amp_normalizer=None,
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

        super().__init__(*args, **kwargs)

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

        if self.discriminator_loco.input_dim != self.discriminator_recovery.input_dim:
            raise ValueError(
                "Locomotion and recovery discriminator input dimensions must match: "
                f"{self.discriminator_loco.input_dim} != {self.discriminator_recovery.input_dim}."
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
        self.optimizer.add_param_group(
            {
                "params": self.discriminator_recovery.amp_linear.parameters(),
                "weight_decay": 10e-2,
                "name": "recovery_amp_head",
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

    def _store_amp_transition(self, amp_obs, next_amp_obs, recovery_mask_t=None):
        if amp_obs.ndim != 2 or next_amp_obs.shape != amp_obs.shape:
            raise ValueError(
                "AMP transition states must be matching 2-D tensors, got "
                f"{tuple(amp_obs.shape)} and {tuple(next_amp_obs.shape)}."
            )
        recovery_mask_t = self._validate_recovery_mask(recovery_mask_t, amp_obs.shape[0])
        locomotion_mask_t = ~recovery_mask_t
        if locomotion_mask_t.any():
            self.amp_storage_loco.insert(amp_obs[locomotion_mask_t], next_amp_obs[locomotion_mask_t])
        if recovery_mask_t.any():
            self.amp_storage_recovery.insert(amp_obs[recovery_mask_t], next_amp_obs[recovery_mask_t])

    def predict_routed_amp_reward(
        self,
        amp_obs,
        next_amp_obs,
        task_reward,
        recovery_mask_t=None,
    ):
        """Route each ``s_t -> s_t+1`` reward by the mode that produced ``a_t``."""
        if amp_obs.ndim != 2 or next_amp_obs.shape != amp_obs.shape:
            raise ValueError(
                "AMP reward states must be matching 2-D tensors, got "
                f"{tuple(amp_obs.shape)} and {tuple(next_amp_obs.shape)}."
            )
        if task_reward.shape != (amp_obs.shape[0],):
            raise ValueError(
                f"task_reward must have shape ({amp_obs.shape[0]},), got {tuple(task_reward.shape)}."
            )

        recovery_mask_t = self._validate_recovery_mask(recovery_mask_t, amp_obs.shape[0])
        locomotion_mask_t = ~recovery_mask_t
        routed_reward = torch.empty_like(task_reward)
        routed_logits = torch.empty((amp_obs.shape[0], 1), dtype=amp_obs.dtype, device=self.device)

        if locomotion_mask_t.any():
            loco_reward, loco_logits = self.discriminator_loco.predict_amp_reward(
                amp_obs[locomotion_mask_t],
                next_amp_obs[locomotion_mask_t],
                task_reward[locomotion_mask_t],
                normalizer=self.amp_normalizer_loco,
            )
            routed_reward[locomotion_mask_t] = loco_reward
            routed_logits[locomotion_mask_t] = loco_logits
        if recovery_mask_t.any():
            recovery_reward, recovery_logits = self.discriminator_recovery.predict_amp_reward(
                amp_obs[recovery_mask_t],
                next_amp_obs[recovery_mask_t],
                task_reward[recovery_mask_t],
                normalizer=self.amp_normalizer_recovery,
            )
            routed_reward[recovery_mask_t] = recovery_reward
            routed_logits[recovery_mask_t] = recovery_logits
        return routed_reward, routed_logits

    def _amp_mini_batch_generator(self, num_updates: int, mini_batch_size: int):
        loco_policy_generator = self.amp_storage_loco.feed_forward_generator(num_updates, mini_batch_size)
        loco_expert_generator = self.amp_data_loco.feed_forward_generator(num_updates, mini_batch_size)

        recovery_num_samples = self.amp_storage_recovery.num_samples
        use_recovery = recovery_num_samples > 0
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
        modules = [self.policy, self.discriminator_loco, self.discriminator_recovery]
        if self.rnd:
            modules.append(self.rnd.predictor)
        return modules

    def _parameters_for_gradient_reduction(self):
        parameters = chain(
            self.policy.parameters(),
            self.discriminator_loco.parameters(),
            self.discriminator_recovery.parameters(),
        )
        if self.rnd:
            parameters = chain(parameters, self.rnd.parameters())
        return parameters
