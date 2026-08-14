from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import AMPPPO, RENetAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    Discriminator,
    EmpiricalNormalization,
    MhaActorCritic,
    RENetActorCritic,
    RecoveryCritic,
)
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


class RENetAmpOnPolicyRunner:
    """AMP runner for RENet training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  #  算法参数
        self.policy_cfg = train_cfg["policy"]  #  策略参数
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] != "RENetAMPPPO":
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")
        self.training_type = "rl"

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]  # actor的输入维度

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1] # critic的观测维度
        else:
            num_privileged_obs = num_obs
        self.num_privileged_obs = num_privileged_obs

        '''
        policy中有actor critic网络
        '''
        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy_activation = self.policy_cfg.get("activation", "elu")
        if "CnnMlp" in self.policy_cfg and isinstance(self.policy_cfg["CnnMlp"], dict):
            self.policy_cfg["CnnMlp"].pop("num_heads", None)
            self.policy_cfg["CnnMlp"].pop("embed_dim", None)
        policy: ActorCritic | MhaActorCritic | RENetActorCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        recovery_critic_hidden_dims = self.alg_cfg.pop("recovery_critic_hidden_dims", [512, 256])
        recovery_critic = RecoveryCritic(
            num_critic_obs=num_privileged_obs,
            hidden_dims=recovery_critic_hidden_dims,
            activation=policy_activation,
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt
        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # Initialize fully independent locomotion/recovery AMP data paths.
        amp_data_loco = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        amp_data_recovery = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["recovery_amp_num_preload_transitions"],
            motion_files=train_cfg["recovery_amp_motion_files"],
        )

        amp_obs_demo = self.env.get_amp_obs_for_expert_trans()
        if amp_obs_demo.ndim != 2:
            raise ValueError(f"Environment AMP observations must be 2-D, got {tuple(amp_obs_demo.shape)}.")
        if amp_data_loco.observation_dim != 50:
            raise ValueError(
                f"Locomotion AMP expert observation dimension must be 50, got {amp_data_loco.observation_dim}."
            )
        if amp_data_recovery.observation_dim != 50:
            raise ValueError(
                f"Recovery AMP expert observation dimension must be 50, got {amp_data_recovery.observation_dim}."
            )
        if amp_obs_demo.shape[1] != 50:
            raise ValueError(f"Environment AMP observation dimension must be 50, got {amp_obs_demo.shape[1]}.")

        amp_normalizer_loco = Normalizer(amp_data_loco.observation_dim)
        amp_normalizer_recovery = Normalizer(amp_data_recovery.observation_dim)
        discriminator_loco = Discriminator(
            amp_data_loco.observation_dim * 2,
            train_cfg["amp_reward_coef"],      # 风格奖励系数
            train_cfg["amp_discr_hidden_dims"],
            device, 
            train_cfg["amp_task_reward_lerp"], # 惩罚系数 只注重任务奖励 不重视模仿
        ).to(self.device)
        discriminator_recovery = Discriminator(
            amp_data_recovery.observation_dim * 2,
            train_cfg["recovery_amp_reward_coef"],
            train_cfg["recovery_amp_discr_hidden_dims"],
            device,
            train_cfg["recovery_amp_task_reward_lerp"],
        ).to(self.device)
        
        min_std = torch.tensor(
            train_cfg["min_normalized_std"],
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        if min_std.numel() != self.env.num_actions:
            raise ValueError(
                "min_normalized_std length must match the action dimension: "
                f"{min_std.numel()} != {self.env.num_actions}."
            )

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg_cfg.pop("optimizer", None)
        self.alg_cfg.pop("share_cnn_encoders", None)
        recovery_state_machine_enabled = bool(getattr(self.env.cfg.recovery, "enable", False))
        if not recovery_state_machine_enabled:
            # One environment switch is sufficient to recover the exact
            # locomotion-only baseline path.
            self.alg_cfg["enable_recovery_learning"] = False
        self.alg: AMPPPO | RENetAMPPPO = alg_class(
            policy,
            discriminator_loco,
            amp_data_loco,
            amp_normalizer_loco,
            recovery_discriminator=discriminator_recovery,
            recovery_amp_data=amp_data_recovery,
            recovery_amp_normalizer=amp_normalizer_recovery,
            recovery_critic=recovery_critic,
            recovery_state_machine_enabled=recovery_state_machine_enabled,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],            # actor观测
            [num_privileged_obs], # critic 观测维度
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
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

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = self.env.get_amp_obs_for_expert_trans()
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example) 

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Capture mode before the action. The transition produced
                    # by this action must never be classified using a mode that
                    # may change inside env.step().
                    recovery_mask_t = self._get_recovery_mask_t()
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    # Move to device
                    obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # Account for terminal state transitions.
                    # env.step() has already reset terminated environments, so next_amp_obs
                    # contains reset states for those environments. The environment must cache
                    # the true pre-reset AMP states in infos["terminal_amp_states"].
                    next_amp_obs_with_term = next_amp_obs.clone()
                    reset_env_ids = self.env.reset_env_ids.to(self.device)

                    if reset_env_ids.numel() > 0:
                        if "terminal_amp_states" not in infos:
                            raise RuntimeError(
                                "The environment reset one or more environments, but infos does not contain "
                                "'terminal_amp_states'. Cache the AMP observations before env.reset() inside env.step()."
                            )

                        terminal_amp_states = infos["terminal_amp_states"].to(self.device)

                        if terminal_amp_states.ndim != 2:
                            raise RuntimeError(
                                "terminal_amp_states must be a 2-D tensor, got "
                                f"shape={tuple(terminal_amp_states.shape)}."
                            )
                        if terminal_amp_states.shape[0] != reset_env_ids.numel():
                            raise RuntimeError(
                                "The number of terminal AMP states does not match the number of reset environments: "
                                f"{terminal_amp_states.shape[0]} != {reset_env_ids.numel()}."
                            )
                        if terminal_amp_states.shape[1] != next_amp_obs.shape[1]:
                            raise RuntimeError(
                                "Terminal AMP state dimension does not match the normal AMP observation dimension: "
                                f"{terminal_amp_states.shape[1]} != {next_amp_obs.shape[1]}."
                            )

                        # terminal_amp_states[i] corresponds to reset_env_ids[i].
                        next_amp_obs_with_term.index_copy_(
                            0,
                            reset_env_ids,
                            terminal_amp_states,
                        )

                    timeout_bootstrap_values = self._compute_timeout_bootstrap_values(
                        infos,
                        reset_env_ids,
                        recovery_mask_t,
                        privileged_obs,
                    )

                    rewards = self.alg.predict_routed_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        recovery_mask_t,
                    )[0]
                    infos["recovery_amp_reward"] = torch.where(
                        recovery_mask_t,
                        rewards,
                        torch.zeros_like(rewards),
                    )
                    recovery_count = recovery_mask_t.float().sum().clamp(min=1.0)
                    recovery_amp_mean = (
                        infos["recovery_amp_reward"].sum() / recovery_count
                        if torch.any(recovery_mask_t)
                        else rewards.sum() * 0.0
                    )
                    infos.setdefault("log", {})["RecoveryAMP/reward_mean"] = recovery_amp_mean

                    # Start the next rollout transition from the post-reset states.
                    amp_obs = next_amp_obs.clone()

                    # Store current -> true terminal transitions for reset environments.
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        next_amp_obs_with_term,
                        recovery_mask_t,
                        timeout_bootstrap_values=timeout_bootstrap_values,
                    )

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                if hasattr(self.env, "update_depth_noise_curriculum_once"):
                    self.env.update_depth_noise_curriculum_once()
                    if self.log_dir is not None and hasattr(self.env, "get_depth_noise_curriculum_log"):
                        depth_noise_log = self.env.get_depth_noise_curriculum_log()
                        if depth_noise_log:
                            ep_infos.append(depth_noise_log)

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if hasattr(self.alg, "get_recovery_learning_diagnostics"):
            for key, value in self.alg.get_recovery_learning_diagnostics().items():
                self.writer.add_scalar(key, value, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            # Legacy keys continue to represent locomotion AMP.
            "discriminator_state_dict": self.alg.discriminator_loco.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer_loco,
            "recovery_discriminator_state_dict": self.alg.discriminator_recovery.state_dict(),
            "recovery_amp_normalizer": self.alg.amp_normalizer_recovery,
            "recovery_critic_state_dict": self.alg.recovery_critic.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        model_state_dict, actor_input_migrated = self._migrate_legacy_actor_input(
            loaded_dict["model_state_dict"],
            self.alg.policy.state_dict(),
        )
        if actor_input_migrated:
            warnings.warn(
                "Migrated legacy RENet Actor input 206 -> 209: copied the original 206 columns and "
                "zero-initialized is_op, is_recovery, and beta columns. Optimizer state will be reinitialized.",
                stacklevel=2,
            )
        resumed_training = self.alg.policy.load_state_dict(model_state_dict)
        self.alg.discriminator_loco.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer_loco = loaded_dict["amp_normalizer"]
        self.alg.amp_normalizer = self.alg.amp_normalizer_loco
        if "recovery_discriminator_state_dict" in loaded_dict:
            self.alg.discriminator_recovery.load_state_dict(loaded_dict["recovery_discriminator_state_dict"])
        else:
            warnings.warn(
                "Checkpoint has no recovery_discriminator_state_dict; recovery discriminator remains newly initialized.",
                stacklevel=2,
            )
        if "recovery_amp_normalizer" in loaded_dict:
            self.alg.amp_normalizer_recovery = loaded_dict["recovery_amp_normalizer"]
        else:
            warnings.warn(
                "Checkpoint has no recovery_amp_normalizer; recovery AMP normalizer remains newly initialized.",
                stacklevel=2,
            )
        if "recovery_critic_state_dict" in loaded_dict:
            self.alg.recovery_critic.load_state_dict(loaded_dict["recovery_critic_state_dict"])
        else:
            warnings.warn(
                "Checkpoint has no recovery_critic_state_dict; all three Recovery critics remain newly initialized.",
                stacklevel=2,
            )
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        checkpoint_has_recovery_amp = "recovery_discriminator_state_dict" in loaded_dict
        checkpoint_has_recovery_critic = "recovery_critic_state_dict" in loaded_dict
        if (
            load_optimizer
            and resumed_training
            and not actor_input_migrated
            and checkpoint_has_recovery_amp
            and checkpoint_has_recovery_critic
        ):
            # -- algorithm optimizer
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except ValueError as error:
                warnings.warn(
                    f"Could not restore optimizer state ({error}); using the newly initialized optimizer.",
                    stacklevel=2,
                )
        elif load_optimizer and resumed_training and not actor_input_migrated:
            warnings.warn(
                "Legacy checkpoint optimizer state does not contain all Recovery AMP/critic parameter groups; "
                "using the newly initialized optimizer.",
                stacklevel=2,
            )
        # -- RND optimizer is independent of the AMP optimizer groups.
        if load_optimizer and resumed_training and self.alg.rnd:
            self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            if hasattr(self.alg, "current_iteration"):
                self.alg.current_iteration = self.current_learning_iteration
        return loaded_dict["infos"]

    @staticmethod
    def _migrate_legacy_actor_input(model_state_dict, target_state_dict):
        """Expand only the legacy 206-column RENet Actor first layer to 209."""
        actor_weight_key = "actor.0.weight"
        if actor_weight_key not in model_state_dict or actor_weight_key not in target_state_dict:
            return model_state_dict, False

        old_weight = model_state_dict[actor_weight_key]
        target_weight = target_state_dict[actor_weight_key]
        if old_weight.shape == target_weight.shape:
            return model_state_dict, False
        if (
            old_weight.ndim != 2
            or target_weight.ndim != 2
            or old_weight.shape[0] != target_weight.shape[0]
            or old_weight.shape[1] != 206
            or target_weight.shape[1] != 209
        ):
            return model_state_dict, False

        migrated_state_dict = model_state_dict.copy()
        migrated_weight = torch.zeros_like(target_weight)
        migrated_weight[:, :206].copy_(old_weight.to(device=target_weight.device, dtype=target_weight.dtype))
        migrated_state_dict[actor_weight_key] = migrated_weight
        return migrated_state_dict, True

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        self.alg.discriminator_loco.train()
        self.alg.discriminator_recovery.train()
        self.alg.recovery_critic.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        self.alg.discriminator_loco.eval()
        self.alg.discriminator_recovery.eval()
        self.alg.recovery_critic.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _get_recovery_mask_t(self):
        """Read the action-time recovery mode, defaulting to all locomotion."""
        mask = None
        if hasattr(self.env, "get_recovery_mask"):
            mask = self.env.get_recovery_mask()
        elif hasattr(self.env, "recovery_mask"):
            mask = self.env.recovery_mask
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "get_recovery_mask"):
            mask = self.env.unwrapped.get_recovery_mask()
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "recovery_mask"):
            mask = self.env.unwrapped.recovery_mask

        if mask is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Environment recovery mask must be a torch.Tensor.")
        mask = mask.to(device=self.device)
        if mask.dtype != torch.bool:
            raise TypeError(f"Environment recovery mask must have dtype bool, got {mask.dtype}.")
        if mask.shape != (self.env.num_envs,):
            raise ValueError(
                f"Environment recovery mask must have shape ({self.env.num_envs},), got {tuple(mask.shape)}."
            )
        return mask.clone()

    def _normalize_terminal_critic_obs(self, terminal_critic_obs):
        """Apply the critic transform without updating empirical statistics a second time."""
        if not self.empirical_normalization:
            return self.privileged_obs_normalizer(terminal_critic_obs)

        was_training = self.privileged_obs_normalizer.training
        self.privileged_obs_normalizer.eval()
        try:
            return self.privileged_obs_normalizer(terminal_critic_obs)
        finally:
            self.privileged_obs_normalizer.train(was_training)

    def _compute_timeout_bootstrap_values(
        self,
        infos,
        reset_env_ids,
        recovery_mask_t,
        critic_obs_reference,
    ):
        """Evaluate mode-routed critics on true pre-reset timeout observations."""
        value_shape = (self.env.num_envs, 1)
        values = {
            "timeout_loco_values": torch.zeros(
                value_shape, dtype=critic_obs_reference.dtype, device=self.device
            ),
            "timeout_rec_task_values": torch.zeros(
                value_shape, dtype=critic_obs_reference.dtype, device=self.device
            ),
            "timeout_rec_amp_values": torch.zeros(
                value_shape, dtype=critic_obs_reference.dtype, device=self.device
            ),
            "timeout_rec_reg_values": torch.zeros(
                value_shape, dtype=critic_obs_reference.dtype, device=self.device
            ),
        }
        if reset_env_ids.numel() == 0:
            return values
        if reset_env_ids.ndim != 1 or reset_env_ids.dtype != torch.long:
            raise RuntimeError(
                "reset_env_ids must be a one-dimensional torch.long tensor, got "
                f"shape={tuple(reset_env_ids.shape)}, dtype={reset_env_ids.dtype}."
            )
        if torch.any((reset_env_ids < 0) | (reset_env_ids >= self.env.num_envs)):
            raise RuntimeError("reset_env_ids contains an out-of-range environment index.")

        if "terminal_critic_obs" not in infos:
            raise RuntimeError(
                "The environment reset one or more environments, but infos does not contain "
                "'terminal_critic_obs'. Build the critic snapshot before env.reset()."
            )
        terminal_critic_obs = infos["terminal_critic_obs"]
        if not isinstance(terminal_critic_obs, torch.Tensor):
            raise TypeError("infos['terminal_critic_obs'] must be a torch.Tensor.")

        raw_critic_obs = infos["observations"].get(self.privileged_obs_type)
        if not isinstance(raw_critic_obs, torch.Tensor):
            raise RuntimeError("The environment did not provide the normal critic observation tensor.")
        if terminal_critic_obs.device != raw_critic_obs.device:
            raise RuntimeError(
                "terminal_critic_obs device must match the environment critic observation device: "
                f"{terminal_critic_obs.device} != {raw_critic_obs.device}."
            )
        if terminal_critic_obs.dtype != raw_critic_obs.dtype:
            raise TypeError(
                "terminal_critic_obs dtype must match the environment critic observation dtype: "
                f"{terminal_critic_obs.dtype} != {raw_critic_obs.dtype}."
            )
        if not terminal_critic_obs.is_floating_point():
            raise TypeError(f"terminal_critic_obs must be floating point, got {terminal_critic_obs.dtype}.")
        if terminal_critic_obs.ndim != 2:
            raise RuntimeError(
                "terminal_critic_obs must be a 2-D tensor, got "
                f"shape={tuple(terminal_critic_obs.shape)}."
            )
        if terminal_critic_obs.shape[0] != reset_env_ids.numel():
            raise RuntimeError(
                "The number of terminal critic observations does not match reset_env_ids: "
                f"{terminal_critic_obs.shape[0]} != {reset_env_ids.numel()}."
            )
        if terminal_critic_obs.shape[1] != self.num_privileged_obs:
            raise RuntimeError(
                "Terminal critic observation dimension does not match the critic input dimension: "
                f"{terminal_critic_obs.shape[1]} != {self.num_privileged_obs}."
            )
        if not torch.isfinite(terminal_critic_obs).all():
            raise RuntimeError("terminal_critic_obs contains NaN or infinity.")

        terminal_critic_obs = terminal_critic_obs.to(self.device)
        if terminal_critic_obs.device != critic_obs_reference.device:
            raise RuntimeError(
                "terminal_critic_obs could not be moved to the critic device: "
                f"{terminal_critic_obs.device} != {critic_obs_reference.device}."
            )
        terminal_critic_obs = self._normalize_terminal_critic_obs(terminal_critic_obs)

        time_outs = infos.get("time_outs")
        if not isinstance(time_outs, torch.Tensor):
            raise TypeError("infos['time_outs'] must be a torch.Tensor.")
        if time_outs.shape != (self.env.num_envs,):
            raise RuntimeError(
                f"infos['time_outs'] must have shape ({self.env.num_envs},), got {tuple(time_outs.shape)}."
            )
        if time_outs.dtype != torch.bool:
            raise TypeError(f"infos['time_outs'] must have dtype bool, got {time_outs.dtype}.")
        time_outs = time_outs.to(self.device)

        recovery_failed = infos.get("recovery_failed")
        if not isinstance(recovery_failed, torch.Tensor):
            raise TypeError("infos['recovery_failed'] must be a torch.Tensor.")
        if recovery_failed.shape != (self.env.num_envs,):
            raise RuntimeError(
                "infos['recovery_failed'] must have shape "
                f"({self.env.num_envs},), got {tuple(recovery_failed.shape)}."
            )
        recovery_failed = recovery_failed.to(device=self.device, dtype=torch.bool)

        reset_time_outs = time_outs.index_select(0, reset_env_ids)
        reset_recovery_failed = recovery_failed.index_select(0, reset_env_ids)
        reset_recovery_mask = recovery_mask_t.index_select(0, reset_env_ids)
        effective_time_outs = reset_time_outs & ~reset_recovery_failed
        loco_rows = effective_time_outs & ~reset_recovery_mask
        recovery_rows = effective_time_outs & reset_recovery_mask

        if torch.any(loco_rows):
            loco_values = self.alg.policy.evaluate(terminal_critic_obs[loco_rows]).detach()
            values["timeout_loco_values"].index_copy_(
                0,
                reset_env_ids[loco_rows],
                loco_values,
            )
        if torch.any(recovery_rows):
            recovery_values = self.alg.recovery_critic(terminal_critic_obs[recovery_rows])
            recovery_env_ids = reset_env_ids[recovery_rows]
            values["timeout_rec_task_values"].index_copy_(
                0, recovery_env_ids, recovery_values["task"].detach()
            )
            values["timeout_rec_amp_values"].index_copy_(
                0, recovery_env_ids, recovery_values["amp"].detach()
            )
            values["timeout_rec_reg_values"].index_copy_(
                0, recovery_env_ids, recovery_values["reg"].detach()
            )
        return values

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
