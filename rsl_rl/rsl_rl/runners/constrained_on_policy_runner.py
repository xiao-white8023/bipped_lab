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

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import ConstrainedPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCost,
    Discriminator,
    EmpiricalNormalization,
    Estimator,
    MhaActorCritic,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


class ConstrainedOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for recovery tasks trained with constrained PPO."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] != "ConstrainedPPO":
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")
        self.training_type = "rl"
        self.use_amp = bool(train_cfg.get("use_amp", self.alg_cfg.get("use_amp", False)))
        print("使用ConstrainedPPO + HWC estimator + ZMP cost" + (" + walking AMP" if self.use_amp else ""))

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        policy_class = eval(self.policy_cfg.pop("class_name"))
        if "CnnMlp" in self.policy_cfg and isinstance(self.policy_cfg["CnnMlp"], dict):
            self.policy_cfg["CnnMlp"].pop("num_heads", None)
            self.policy_cfg["CnnMlp"].pop("embed_dim", None)
        policy: ActorCritic | ActorCriticCost | MhaActorCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        estimator = Estimator(
            num_obs=self.estimator_cfg["prop_dim"],
            num_history=self.estimator_cfg["history_len"],
            num_latent=self.estimator_cfg["priv_latent_dim"],
            num_labels=self.estimator_cfg["priv_states_dim"],
            activation="elu",
            decoder_hidden_dims=self.estimator_cfg["decoder_hidden_dims"],
            encoder_hidden_dims=self.estimator_cfg["encoder_hidden_dims"],
        ).to(self.device)

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            num_rnd_state = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        discriminator = None
        amp_data = None
        amp_normalizer = None
        if self.use_amp:
            amp_data = AMPLoader(
                device,
                time_between_frames=self.env.step_dt,
                preload_transitions=True,
                num_preload_transitions=train_cfg["amp_num_preload_transitions"],
                motion_files=train_cfg["amp_motion_files"],
            )
            amp_obs_demo = self.env.get_amp_obs_for_expert_trans()
            print(f"DEBUG: recovery AMP dataset dim: {amp_data.observation_dim}")
            print(f"DEBUG: recovery AMP env obs dim: {amp_obs_demo.shape[1]}")
            if amp_data.observation_dim != amp_obs_demo.shape[1]:
                raise ValueError(
                    "AMP observation dimension mismatch: "
                    f"dataset={amp_data.observation_dim}, policy={amp_obs_demo.shape[1]}."
                )
            amp_normalizer = Normalizer(amp_data.observation_dim)
            discriminator = Discriminator(
                amp_data.observation_dim * 2,
                train_cfg["amp_reward_coef"],
                train_cfg["amp_discr_hidden_dims"],
                device,
                train_cfg["amp_task_reward_lerp"],
            ).to(self.device)

        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg_cfg.pop("optimizer", None)
        self.alg_cfg.pop("share_cnn_encoders", None)
        self.alg: ConstrainedPPO = alg_class(
            policy,
            estimator=estimator,
            estimator_paras=self.estimator_cfg,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        if self.use_amp:
            amp_obs = self.env.get_amp_obs_for_expert_trans().to(self.device)
        else:
            amp_obs = None
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        costbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    if self.use_amp:
                        next_amp_obs = self.env.get_amp_obs_for_expert_trans().to(self.device)
                    else:
                        next_amp_obs = None
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    if self.use_amp:
                        next_amp_obs_with_term = next_amp_obs.clone()
                        reset_env_ids = self.env.reset_env_ids.to(self.device)
                        if reset_env_ids.numel() > 0:
                            if "terminal_amp_states" not in infos:
                                raise RuntimeError(
                                    "The environment reset one or more environments, but infos does not contain "
                                    "'terminal_amp_states'."
                                )
                            terminal_amp_states = infos["terminal_amp_states"].to(self.device)
                            if terminal_amp_states.ndim != 2:
                                raise RuntimeError(
                                    "terminal_amp_states must be a 2-D tensor, got "
                                    f"shape={tuple(terminal_amp_states.shape)}."
                                )
                            if terminal_amp_states.shape[0] != reset_env_ids.numel():
                                raise RuntimeError(
                                    "The number of terminal AMP states does not match reset envs: "
                                    f"{terminal_amp_states.shape[0]} != {reset_env_ids.numel()}."
                                )
                            if terminal_amp_states.shape[1] != next_amp_obs.shape[1]:
                                raise RuntimeError(
                                    "Terminal AMP state dimension does not match AMP obs: "
                                    f"{terminal_amp_states.shape[1]} != {next_amp_obs.shape[1]}."
                                )
                            next_amp_obs_with_term.index_copy_(0, reset_env_ids, terminal_amp_states)

                        amp_rewards = self.alg.discriminator.predict_amp_reward(
                            amp_obs,
                            next_amp_obs_with_term,
                            rewards,
                            normalizer=self.alg.amp_normalizer,
                        )[0]
                        amp_mask = infos.get("walk_amp_mask")
                        if amp_mask is None:
                            amp_mask = torch.ones_like(rewards, dtype=torch.bool)
                        amp_mask = amp_mask.to(self.device, dtype=torch.bool).view(-1)
                        rewards = torch.where(amp_mask, amp_rewards, rewards)
                        amp_obs = next_amp_obs.clone()
                        self.alg.process_env_step(obs, rewards, dones, infos, next_amp_obs_with_term, amp_mask)
                    else:
                        self.alg.process_env_step(obs, rewards, dones, infos)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards

                        if self.alg.use_zmp_cost and "zmp_cost" in infos:
                            cur_cost_sum += infos["zmp_cost"].to(self.device).view(-1)

                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if self.alg.use_zmp_cost:
                            costbuffer.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_cost_sum[new_ids] = 0

                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and self.log_dir is not None and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)

        if not self.alg.use_zmp_cost:
            return

        zmp_lambda = float(self.alg.zmp_lambda.item())
        if self.writer is not None:
            self.writer.add_scalar("Constraint/zmp_cost", self.alg.last_mean_rollout_cost, locs["it"])
            self.writer.add_scalar("Constraint/zmp_cost_limit", self.alg.zmp_cost_limit, locs["it"])
            self.writer.add_scalar("Constraint/zmp_lambda", zmp_lambda, locs["it"])
            if len(locs["costbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_episode_cost_zmp", statistics.mean(locs["costbuffer"]), locs["it"]
                )

        print(
            f"{'ZMP cost / limit / lambda:':>{pad}} "
            f"{self.alg.last_mean_rollout_cost:.4f} / {self.alg.zmp_cost_limit:.4f} / {zmp_lambda:.4f}"
        )

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "zmp_lambda": float(self.alg.zmp_lambda.item()) if hasattr(self.alg, "zmp_lambda") else 0.0,
        }
        if self.use_amp:
            saved_dict["discriminator_state_dict"] = self.alg.discriminator.state_dict()
            saved_dict["amp_normalizer"] = self.alg.amp_normalizer
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)

        model_state_dict = loaded_dict["model_state_dict"]
        if not any(key.startswith("critic_cost.") for key in model_state_dict):
            for key, value in list(model_state_dict.items()):
                if key.startswith("critic."):
                    model_state_dict["critic_cost." + key[len("critic.") :]] = value.clone()

        resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=False)
        if "estimator_state_dict" in loaded_dict:
            self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=False)
        if self.use_amp and "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
        if self.use_amp and "amp_normalizer" in loaded_dict:
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        if "zmp_lambda" in loaded_dict and hasattr(self.alg, "zmp_lambda"):
            self.alg.zmp_lambda = torch.tensor(float(loaded_dict["zmp_lambda"]), device=self.device)

        if load_optimizer and resumed_training and "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[WARN] Skipping optimizer state load for ConstrainedPPO: {exc}")
            if "estimator_optimizer_state_dict" in loaded_dict:
                try:
                    self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
                except ValueError as exc:
                    print(f"[WARN] Skipping estimator optimizer state load for ConstrainedPPO: {exc}")
            if self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        self.alg.policy.train()
        self.alg.estimator.train()
        if self.use_amp:
            self.alg.discriminator.train()
        if self.alg.rnd:
            self.alg.rnd.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        self.alg.policy.eval()
        self.alg.estimator.eval()
        if self.use_amp:
            self.alg.discriminator.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()
