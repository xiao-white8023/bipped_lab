from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticCost, Estimator
from rsl_rl.storage import ConstrainedRolloutStorage, ReplayBuffer

from .ppo import PPO


class ConstrainedPPO(PPO):
    """HWC-Loco style PPO with VAE estimated states and ZMP constrained cost."""

    policy: ActorCriticCost

    def __init__(
        self,
        policy,
        *args,
        estimator: Estimator | None = None,
        estimator_paras: dict | None = None,
        use_zmp_cost=True,
        zmp_cost_limit=0.004,
        zmp_lambda_init=0.0,
        zmp_lambda_lr=1.0e-2,
        zmp_lambda_max=0.1,
        zmp_cost_value_loss_coef=0.1,
        normalize_cost_advantages=False,
        use_amp=False,
        discriminator=None,
        amp_data=None,
        amp_normalizer=None,
        amp_replay_buffer_size=100000,
        amp_loss_coef=1.0,
        amp_grad_penalty_coef=10.0,
        amp_walk_only=True,
        **kwargs,
    ):
        super().__init__(policy, *args, **kwargs)

        if use_zmp_cost and not hasattr(self.policy, "evaluate_cost"):
            raise AttributeError("ConstrainedPPO requires a policy with evaluate_cost() when use_zmp_cost=True.")
        if estimator is None:
            raise ValueError("HWC recovery ConstrainedPPO requires a VAE estimator.")

        self.estimator = estimator.to(self.device)
        self.estimator_paras = estimator_paras or {}
        self.estimator_optimizer = optim.Adam(
            self.estimator.parameters(),
            lr=self.estimator_paras.get("learning_rate", 1.0e-4),
        )
        self.train_with_estimated_states = self.estimator_paras.get("train_with_estimated_states", True)
        self.est_start = self.estimator_paras["priv_start"]
        self.num_prop = self.estimator_paras["prop_dim"]
        self.prop_start = self.estimator_paras["prop_start"]
        self.history_len = self.estimator_paras["history_len"]
        self.priv_states_dim = self.estimator_paras["priv_states_dim"]

        self.use_zmp_cost = use_zmp_cost
        self.zmp_cost_limit = zmp_cost_limit
        self.zmp_lambda = torch.tensor(zmp_lambda_init, device=self.device, dtype=torch.float32)
        self.zmp_lambda_lr = zmp_lambda_lr
        self.zmp_lambda_max = zmp_lambda_max
        self.zmp_cost_value_loss_coef = zmp_cost_value_loss_coef
        self.normalize_cost_advantages = normalize_cost_advantages
        self.last_mean_rollout_cost = 0.0
        self._rollout_cost_sum = 0.0
        self._rollout_cost_count = 0

        self.use_amp = bool(use_amp)
        self.amp_walk_only = bool(amp_walk_only)
        self.discriminator = discriminator
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amp_loss_coef = amp_loss_coef
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.amp_transition_observations = None
        self.amp_storage = None
        if self.use_amp:
            if self.discriminator is None or self.amp_data is None:
                raise ValueError("ConstrainedPPO use_amp=True requires discriminator and amp_data.")
            self.discriminator.to(self.device)
            self.amp_storage = ReplayBuffer(self.discriminator.input_dim // 2, amp_replay_buffer_size, self.device)
            params = [
                {"params": self.policy.parameters(), "name": "policy"},
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
            ]
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

        self.transition = ConstrainedRolloutStorage.Transition()

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        rnd_state_shape = [self.rnd.num_states] if self.rnd else None
        self.storage = ConstrainedRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def _split_estimator_inputs(self, obs):
        hist_obs = obs[:, self.prop_start - self.history_len * self.num_prop : self.prop_start]
        current_obs = obs[:, self.prop_start : self.prop_start + self.num_prop]
        return hist_obs, current_obs

    def _estimate_actor_observations(self, obs, detach_latent: bool):
        if not self.train_with_estimated_states:
            return obs
        hist_obs, current_obs = self._split_estimator_inputs(obs)
        z, labels = self.estimator.sample(hist_obs, current_obs)
        latent = torch.cat([z, labels], dim=1)
        if detach_latent:
            latent = latent.detach()
        return torch.cat([obs[:, : self.est_start], latent], dim=1)

    def act(self, obs, critic_obs, amp_obs=None):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        actor_obs = self._estimate_actor_observations(obs, detach_latent=False)
        self.transition.actions = self.policy.act(actor_obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        if self.use_zmp_cost:
            self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        else:
            self.transition.cost_values = torch.zeros_like(self.transition.values)
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        if self.use_amp and amp_obs is not None:
            self.amp_transition_observations = amp_obs.detach()
        return self.transition.actions

    def process_env_step(self, next_obs, rewards, dones, infos, amp_obs=None, amp_mask=None):
        self.transition.next_observations = next_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.use_zmp_cost:
            zmp_cost = infos.get("zmp_cost", torch.zeros_like(rewards))
            if not isinstance(zmp_cost, torch.Tensor):
                zmp_cost = torch.tensor(zmp_cost, device=self.device, dtype=rewards.dtype)
            zmp_cost = zmp_cost.to(device=self.device, dtype=rewards.dtype).view(-1)
            self.transition.costs = zmp_cost.clone()
            self._rollout_cost_sum += zmp_cost.sum().item()
            self._rollout_cost_count += zmp_cost.numel()
        else:
            self.transition.costs = torch.zeros_like(rewards).view(-1)

        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            if self.use_zmp_cost:
                self.transition.costs += self.gamma * torch.squeeze(self.transition.cost_values * time_outs, 1)

        if self.use_amp and amp_obs is not None and self.amp_transition_observations is not None:
            amp_states = self.amp_transition_observations
            amp_next_states = amp_obs.detach()
            if self.amp_walk_only and amp_mask is not None:
                amp_mask = amp_mask.to(device=self.device, dtype=torch.bool).view(-1)
                amp_states = amp_states[amp_mask]
                amp_next_states = amp_next_states[amp_mask]
            if amp_states.numel() > 0:
                self.amp_storage.insert(amp_states, amp_next_states)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition_observations = None
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
        if self.use_zmp_cost:
            last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
            self.storage.compute_cost_returns(
                last_cost_values,
                self.gamma,
                self.lam,
                normalize_advantages=self.normalize_cost_advantages,
            )

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0
        mean_recon_loss = 0.0
        mean_predict_loss = 0.0
        mean_kld_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_cost_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_amp_grad_pen_loss = 0.0
        mean_amp_policy_pred = 0.0
        mean_amp_expert_pred = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        use_amp_update = self.use_amp and self.amp_storage is not None and self.amp_storage.num_samples > 0
        if use_amp_update:
            amp_policy_generator = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
            amp_expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)

        for (
            obs_batch,
            next_obs_batch,
            critic_obs_batch,
            actions_batch,
            dones_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            target_cost_values_batch,
            cost_advantages_batch,
            cost_returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            _rnd_state_batch,
        ) in generator:
            obs_est_batch = self._estimate_actor_observations(obs_batch, detach_latent=True)

            self.policy.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            if self.use_zmp_cost:
                cost_value_batch = self.policy.evaluate_cost(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
            else:
                cost_value_batch = torch.zeros_like(value_batch)

            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            hist_obs_batch, current_obs_batch = self._split_estimator_inputs(obs_batch)
            future_obs_batch = next_obs_batch[:, self.prop_start : self.prop_start + self.num_prop]
            future_label_batch = next_obs_batch[:, self.est_start : self.est_start + self.priv_states_dim]
            loss_dict = self.estimator.loss_fn(
                hist_obs_batch,
                current_obs_batch,
                future_obs_batch,
                future_label_batch,
                dones=dones_batch,
                kld_weight=1.0,
            )
            estimator_loss = torch.mean(loss_dict["loss"])
            recon_loss = torch.mean(loss_dict["recons_loss"])
            predict_loss = torch.mean(loss_dict["label_loss"])
            kld_loss = torch.mean(loss_dict["kld_loss"])

            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

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
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_zmp_cost:
                cost_surrogate = torch.squeeze(cost_advantages_batch) * ratio
                cost_surrogate_clipped = torch.squeeze(cost_advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                cost_surrogate_loss = torch.max(cost_surrogate, cost_surrogate_clipped).mean()
            else:
                cost_surrogate_loss = torch.zeros((), device=self.device)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.use_zmp_cost:
                if self.use_clipped_value_loss:
                    cost_value_clipped = target_cost_values_batch + (
                        cost_value_batch - target_cost_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                    cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                    cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
                else:
                    cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()
            else:
                cost_value_loss = torch.zeros((), device=self.device)

            loss = (
                surrogate_loss
                + self.zmp_lambda.detach() * cost_surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            if self.use_zmp_cost:
                loss = loss + self.zmp_cost_value_loss_coef * cost_value_loss

            amp_loss = torch.zeros((), device=self.device)
            amp_grad_pen_loss = torch.zeros((), device=self.device)
            amp_policy_pred = torch.zeros((), device=self.device)
            amp_expert_pred = torch.zeros((), device=self.device)
            if use_amp_update:
                policy_state, policy_next_state = next(amp_policy_generator)
                expert_state, expert_next_state = next(amp_expert_generator)
                sample_amp_expert = (expert_state, expert_next_state)
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                amp_grad_pen_loss = self.discriminator.compute_grad_pen(
                    *sample_amp_expert,
                    lambda_=self.amp_grad_penalty_coef,
                )
                loss = loss + self.amp_loss_coef * amp_loss + self.amp_loss_coef * amp_grad_pen_loss
                amp_policy_pred = policy_d.mean()
                amp_expert_pred = expert_d.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if use_amp_update and self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.detach().cpu().numpy())
                self.amp_normalizer.update(expert_state.detach().cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_predict_loss += predict_loss.item()
            mean_kld_loss += kld_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_cost_surrogate_loss += cost_surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_amp_grad_pen_loss += amp_grad_pen_loss.item()
            mean_amp_policy_pred += amp_policy_pred.item()
            mean_amp_expert_pred += amp_expert_pred.item()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_predict_loss /= num_updates
        mean_kld_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_cost_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_amp_grad_pen_loss /= num_updates
        mean_amp_policy_pred /= num_updates
        mean_amp_expert_pred /= num_updates

        self._update_zmp_lambda()
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
            "estimator/reconstruction": mean_recon_loss,
            "estimator/prediction": mean_predict_loss,
            "estimator/kld": mean_kld_loss,
        }
        if self.use_zmp_cost:
            loss_dict["cost_value"] = mean_cost_value_loss
            loss_dict["cost_surrogate"] = mean_cost_surrogate_loss
        if self.use_amp:
            loss_dict["amp"] = mean_amp_loss
            loss_dict["amp_grad_pen"] = mean_amp_grad_pen_loss
            loss_dict["amp_policy_pred"] = mean_amp_policy_pred
            loss_dict["amp_expert_pred"] = mean_amp_expert_pred
        return loss_dict

    def _update_zmp_lambda(self):
        if self.use_zmp_cost:
            if self.is_multi_gpu:
                cost_stats = torch.tensor(
                    [self._rollout_cost_sum, float(self._rollout_cost_count)],
                    device=self.device,
                    dtype=torch.float32,
                )
                torch.distributed.all_reduce(cost_stats, op=torch.distributed.ReduceOp.SUM)
                rollout_cost_sum = cost_stats[0].item()
                rollout_cost_count = int(cost_stats[1].item())
            else:
                rollout_cost_sum = self._rollout_cost_sum
                rollout_cost_count = self._rollout_cost_count

            self.last_mean_rollout_cost = rollout_cost_sum / max(rollout_cost_count, 1)
            self.zmp_lambda = torch.clamp(
                self.zmp_lambda + self.zmp_lambda_lr * (self.last_mean_rollout_cost - self.zmp_cost_limit),
                min=0.0,
                max=self.zmp_lambda_max,
            )
        else:
            self.last_mean_rollout_cost = 0.0
        self._rollout_cost_sum = 0.0
        self._rollout_cost_count = 0
