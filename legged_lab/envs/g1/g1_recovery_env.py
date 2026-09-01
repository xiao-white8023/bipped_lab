# ================================================
# G1 23dof recovery
# ================================================
import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import RewardManager
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse

from legged_lab.envs.g1.g1_recovery_cfg import (
    G1_RECOVERY_BODY_NAMES,
    G1RecoveryEnvCfg,
)
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import RecoveryAMPFeatureBuilder, RecoveryNPZMotionLoader
from rsl_rl.utils.recovery_amp_features import BODY_ORI_6D_SLICE, BODY_POS_SLICE


class G1RecoveryEnv(VecEnv):

    ACTOR_CURRENT_DIM = 75
    CRITIC_CURRENT_DIM = 195
    AMP_OBSERVATION_DIM = 195
    FORMAL_EXPERT_TRANSITION_DT = 0.02
    SELF_COLLISION_SENSOR_PREFIX = "recovery_self_collision_"

    def __init__(self, cfg: G1RecoveryEnvCfg, headless):
        self.cfg: G1RecoveryEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs

        if abs(self.step_dt - 0.02) >= 1.0e-8:
            raise RuntimeError(
                f"G1 Recovery policy step_dt must be 0.02 s, got {self.step_dt}."
            )
        if abs(self.cfg.scene.max_episode_length_s - 6.0) >= 1.0e-8:
            raise RuntimeError(
                "G1 Recovery max_episode_length_s must remain 6.0 s, got "
                f"{self.cfg.scene.max_episode_length_s}."
            )
        if not self.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions:
            raise RuntimeError(
                "Standalone G1 Recovery articulation must enable self-collisions."
            )

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(
            config=cfg.scene,
            physics_dt=self.physics_dt,
            step_dt=self.step_dt,
        )
        # IsaacLab filtered contact reporting is one-to-many only.  Use one
        # sensor body per sensor and filter exclusively against robot links;
        # ground forces therefore never enter the self-collision metric.
        for body_index, body_name in enumerate(G1_RECOVERY_BODY_NAMES):
            setattr(
                scene_cfg,
                f"{self.SELF_COLLISION_SENSOR_PREFIX}{body_index:02d}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                    # Each filter expression must resolve to exactly one body
                    # per environment in the PhysX tensor API.  One wildcard
                    # expression over all links is rejected by IsaacLab/PhysX.
                    filter_prim_paths_expr=[
                        f"{{ENV_REGEX_NS}}/Robot/{filter_body_name}"
                        for filter_body_name in G1_RECOVERY_BODY_NAMES
                    ],
                    history_length=0,
                    update_period=self.physics_dt,
                    max_contact_data_count_per_prim=32,
                ),
            )
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.self_collision_sensors = tuple(
            self.scene.sensors[
                f"{self.SELF_COLLISION_SENSOR_PREFIX}{body_index:02d}"
            ]
            for body_index in range(len(G1_RECOVERY_BODY_NAMES))
        )

        if self.robot.num_joints != 23:
            raise RuntimeError(
                f"G1 Recovery requires exactly 23 joints, got {self.robot.num_joints}."
            )
        if tuple(self.robot.body_names) != G1_RECOVERY_BODY_NAMES:
            mismatches = self._first_order_mismatches(
                tuple(self.robot.body_names), G1_RECOVERY_BODY_NAMES
            )
            raise RuntimeError(
                "G1 Recovery self-collision sensor body order does not match the "
                f"articulation: {'; '.join(mismatches)}"
            )
        if self.cfg.reset.random_yaw:
            raise RuntimeError("Recovery random_yaw must remain disabled.")

        try:
            self.torso_body_index = self.robot.body_names.index("torso_link")
        except ValueError as exc:
            raise RuntimeError(
                "G1 Recovery articulation is missing required torso_link."
            ) from exc

        self.reset_motion_loader = RecoveryNPZMotionLoader(
            motion_dir=self.cfg.reset.motion_dir,
            device=self.device,
            expected_fps=50.0,
            sampling="uniform_frame",
        )
        self._validate_reset_motion_ordering()

        # Expert 与 policy 共用 Stage 3.2 的 whole-body AMP feature builder。
        self.amp_feature_builder = RecoveryAMPFeatureBuilder(self.robot.body_names)
        if self.amp_feature_builder.feature_dim != self.AMP_OBSERVATION_DIM:
            raise RuntimeError(
                "RecoveryAMPFeatureBuilder must produce exactly 195 features, got "
                f"{self.amp_feature_builder.feature_dim}."
            )

        # 与 G1 Film 一致，由 RewardManager 解析并管理奖励项。
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.self_collision_force_threshold = float(
            self.reward_manager.get_term_cfg("self_collisions").params[
                "force_threshold"
            ]
        )

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(env_ids)

        print(f"G1 recovery policy dt: {self.step_dt:.8f} s")
        print(
            "G1 recovery reset motions/frames: "
            f"{self.reset_motion_loader.num_motions}/"
            f"{self.reset_motion_loader.total_frames}"
        )
        print("G1 recovery joint names:", self.robot.joint_names)
        print("G1 recovery body names:", self.robot.body_names)

    @staticmethod
    def _first_order_mismatches(actual, expected, limit=5):
        mismatches = []
        for index in range(max(len(actual), len(expected))):
            actual_name = actual[index] if index < len(actual) else "<missing>"
            expected_name = expected[index] if index < len(expected) else "<missing>"
            if actual_name != expected_name:
                mismatches.append(
                    f"index {index}: robot={actual_name!r}, dataset={expected_name!r}"
                )
                if len(mismatches) == limit:
                    break
        return mismatches

    def _validate_reset_motion_ordering(self):
        ordering_pairs = (
            (
                "joint_names",
                tuple(self.robot.joint_names),
                tuple(self.reset_motion_loader.joint_names),
            ),
            (
                "body_names",
                tuple(self.robot.body_names),
                tuple(self.reset_motion_loader.body_names),
            ),
        )
        for label, robot_order, dataset_order in ordering_pairs:
            if robot_order != dataset_order:
                first_mismatches = self._first_order_mismatches(
                    robot_order, dataset_order
                )
                message = (
                    f"G1 Recovery robot/dataset {label} ordering mismatch; "
                    f"first mismatches: {'; '.join(first_mismatches)}"
                )
                print(message)
                raise RuntimeError(message)
        if (
            self.reset_motion_loader.root_body_name != "pelvis"
            or self.reset_motion_loader.root_body_index
            != self.reset_motion_loader.body_index("pelvis")
        ):
            raise RuntimeError(
                "G1 Recovery reset trajectory root body must be pelvis, got "
                f"{self.reset_motion_loader.root_body_name!r} at index "
                f"{self.reset_motion_loader.root_body_index}."
            )
        if self.reset_motion_loader.fps != 50.0:
            raise RuntimeError(
                "G1 Recovery reset trajectories must use exactly 50 Hz, got "
                f"{self.reset_motion_loader.fps}."
            )

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = math.ceil(
            self.max_episode_length_s / self.step_dt
        )
        if self.max_episode_length != 300:
            raise RuntimeError(
                "G1 Recovery 6.0 s episode must contain exactly 300 policy "
                f"steps, got {self.max_episode_length}."
            )
        hold_steps = self.cfg.success.hold_s / self.step_dt
        self.success_hold_steps = round(hold_steps)
        if (
            abs(hold_steps - self.success_hold_steps) >= 1.0e-8
            or self.success_hold_steps != 25
        ):
            raise RuntimeError(
                "G1 Recovery 0.5 s success hold must contain exactly 25 policy "
                f"steps, got {hold_steps}."
            )
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.control.clip_actions
        self.clip_obs = self.cfg.observation.clip_observations
        self.action_scale = self.cfg.control.action_scale

        # 上一个已经执行的 policy action。
        self.action = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_action = torch.zeros_like(self.action)
        self.joint_pos_target = self.robot.data.default_joint_pos.clone()

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.sim_step_counter = 0
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.terminated_buf = torch.zeros_like(self.reset_buf)
        self.success_buf = torch.zeros_like(self.reset_buf)
        self.success_hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Lifetime episode statistics for this environment instance. Individual
        # dataset resets must not clear these cumulative counters.
        self.recovery_episode_success_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.recovery_episode_timeout_total = torch.zeros_like(
            self.recovery_episode_success_total
        )
        self.recovery_episode_completed_total = torch.zeros_like(
            self.recovery_episode_success_total
        )
        self.recovery_episode_duration_sum_s = torch.zeros(
            (), dtype=torch.float, device=self.device
        )
        self.recovery_episode_success_this_step = torch.zeros_like(
            self.recovery_episode_success_total
        )
        self.recovery_episode_timeout_this_step = torch.zeros_like(
            self.recovery_episode_success_total
        )
        self.recovery_episode_completed_this_step = torch.zeros_like(
            self.recovery_episode_success_total
        )

        self.reset_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.pre_reset_amp_obs = torch.empty(
            self.num_envs,
            self.AMP_OBSERVATION_DIM,
            dtype=torch.float,
            device=self.device,
        )
        self.last_reset_motion_ids = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.last_reset_frame_ids = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.last_reset_joint_pos = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
        )

        # IsaacLab's filtered contact force history cannot aggregate multiple
        # sensor bodies correctly.  Record the exact per-physics-substep hit
        # semantics explicitly for each 0.02 s policy step.
        self.self_collision_hit_substeps = torch.zeros(
            self.num_envs,
            self.cfg.sim.decimation,
            dtype=torch.bool,
            device=self.device,
        )

        float_buffers = (
            "pelvis_height",
            "torso_up_z",
            "height_reward_raw",
            "height_reward_scaled",
            "upright_reward_raw",
            "upright_reward_scaled",
            "joint_acc_cost",
            "joint_pos_limits_cost",
            "action_rate_cost",
            "self_collision_cost",
            "joint_acc_reward",
            "joint_pos_limits_reward",
            "action_rate_reward",
            "self_collision_reward",
            "task_reward",
        )
        for buffer_name in float_buffers:
            setattr(
                self,
                buffer_name,
                torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                ),
            )
        self.height_ok = torch.zeros_like(self.reset_buf)
        self.upright_ok = torch.zeros_like(self.reset_buf)
        self.ready_now = torch.zeros_like(self.reset_buf)
        self.torso_projected_gravity_b = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self.init_obs_buffer()

    def init_obs_buffer(self):
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.observation.history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.observation.history_length,
            batch_size=self.num_envs,
            device=self.device,
        )

    @staticmethod
    def _require_finite_shape(tensor, expected_shape, label):
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"{label} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{label} contains NaN or Inf values.")

    def compute_current_observations(self):
        robot = self.robot

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel
        action = self.action

        current_actor_obs = torch.cat(
            [
                ang_vel,  # 3
                projected_gravity,  # 3
                joint_pos,  # 23
                joint_vel,  # 23
                action,  # 23
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        whole_body_amp_features = self.amp_feature_builder.build(
            robot.data.body_link_pos_w,
            robot.data.body_link_quat_w,
            robot.data.body_link_lin_vel_w,
            robot.data.body_link_ang_vel_w,
        )
        body_pos_b = whole_body_amp_features[:, BODY_POS_SLICE]
        body_ori_6d = whole_body_amp_features[:, BODY_ORI_6D_SLICE]
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel, body_pos_b, body_ori_6d], dim=-1
        )

        self._require_finite_shape(
            current_actor_obs,
            (self.num_envs, self.ACTOR_CURRENT_DIM),
            "Recovery actor current observation",
        )
        self._require_finite_shape(
            current_critic_obs,
            (self.num_envs, self.CRITIC_CURRENT_DIM),
            "Recovery critic current observation",
        )
        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        self.actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        self.critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        self.actor_obs = torch.clip(self.actor_obs, -self.clip_obs, self.clip_obs)
        self.critic_obs = torch.clip(self.critic_obs, -self.clip_obs, self.clip_obs)

        history_length = self.cfg.observation.history_length
        self._require_finite_shape(
            self.actor_obs,
            (self.num_envs, self.ACTOR_CURRENT_DIM * history_length),
            "Recovery actor history observation",
        )
        self._require_finite_shape(
            self.critic_obs,
            (self.num_envs, self.CRITIC_CURRENT_DIM * history_length),
            "Recovery critic history observation",
        )
        return self.actor_obs, self.critic_obs

    def get_amp_obs_for_expert_trans(self):
        amp_obs = self.amp_feature_builder.build(
            self.robot.data.body_link_pos_w,
            self.robot.data.body_link_quat_w,
            self.robot.data.body_link_lin_vel_w,
            self.robot.data.body_link_ang_vel_w,
        )
        self._require_finite_shape(
            amp_obs,
            (self.num_envs, self.AMP_OBSERVATION_DIM),
            "Recovery AMP observation",
        )
        return amp_obs

    def _compute_joint_position_target(self, clipped_actions):
        processed_actions = (
            clipped_actions * self.action_scale
            + self.robot.data.default_joint_pos
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
        processed_actions = torch.clip(
            processed_actions,
            min=soft_joint_pos_limits[..., 0],
            max=soft_joint_pos_limits[..., 1],
        )
        return processed_actions

    def _normalize_env_ids(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        env_ids = env_ids.flatten()
        if torch.any(env_ids < 0) or torch.any(env_ids >= self.num_envs):
            raise IndexError(
                f"Recovery env_ids must be in [0, {self.num_envs - 1}]."
            )
        return env_ids

    @torch.no_grad()
    def _apply_reset_motion_sample(self, env_ids, sample):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() != sample["motion_ids"].numel():
            raise ValueError(
                "Recovery env_ids and sampled motion frames must have the same "
                f"length, got {env_ids.numel()} and "
                f"{sample['motion_ids'].numel()}."
            )

        if env_ids.numel() == 0:
            return

        self.scene.reset(env_ids)
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras.setdefault("log", {}).update(reward_extras)

        root_index = self.reset_motion_loader.root_body_index
        source_root_pos = sample["body_pos_w"][:, root_index]
        root_pos = source_root_pos.clone()
        env_origins = self.scene.env_origins[env_ids]
        root_pos[:, :2] = env_origins[:, :2]
        root_pos[:, 2] = env_origins[:, 2] + source_root_pos[:, 2]

        root_quat_w = sample["body_quat_w"][:, root_index]
        root_lin_vel_w = sample["body_lin_vel_w"][:, root_index]
        root_ang_vel_w = sample["body_ang_vel_w"][:, root_index]
        root_pose = torch.cat([root_pos, root_quat_w], dim=-1)
        root_velocity = torch.cat(
            [root_lin_vel_w, root_ang_vel_w], dim=-1
        )
        joint_pos = sample["joint_pos"]
        joint_vel = sample["joint_vel"]

        self.robot.write_root_link_pose_to_sim(
            root_pose, env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim(
            root_velocity, env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids
        )
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(
            torch.zeros_like(joint_vel), env_ids=env_ids
        )

        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.joint_pos_target[env_ids] = joint_pos
        self.last_reset_motion_ids[env_ids] = sample["motion_ids"]
        self.last_reset_frame_ids[env_ids] = sample["frame_ids"]
        self.last_reset_joint_pos[env_ids] = joint_pos
        self.success_hold_counter[env_ids] = 0
        self.success_buf[env_ids] = False
        self.terminated_buf[env_ids] = False
        self.height_ok[env_ids] = False
        self.upright_ok[env_ids] = False
        self.ready_now[env_ids] = False
        self.self_collision_hit_substeps[env_ids] = False

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

    def reset(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        sample = self.reset_motion_loader.sample_frame_batch(env_ids.numel())
        self._apply_reset_motion_sample(env_ids, sample)

    @torch.no_grad()
    def reset_to_motion_frames(self, env_ids, motion_ids, frame_ids):
        env_ids = self._normalize_env_ids(env_ids)
        sample = self.reset_motion_loader.get_frame_batch(motion_ids, frame_ids)
        if env_ids.numel() != sample["motion_ids"].numel():
            raise ValueError(
                "Recovery env_ids, motion_ids, and frame_ids must have the same "
                f"length, got {env_ids.numel()}, {sample['motion_ids'].numel()}, "
                f"and {sample['frame_ids'].numel()}."
            )
        self._apply_reset_motion_sample(env_ids, sample)

    def _read_self_collision_hits(self):
        """Return whether any robot-vs-robot force exceeds the V1 threshold."""

        threshold = self.self_collision_force_threshold
        hit = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for sensor in self.self_collision_sensors:
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    "Recovery self-collision sensor has no filtered force matrix."
                )
            if force_matrix_w.shape[0] != self.num_envs:
                raise RuntimeError(
                    "Recovery self-collision force matrix has invalid batch shape "
                    f"{tuple(force_matrix_w.shape)}."
                )
            force_magnitude = torch.linalg.vector_norm(
                force_matrix_w, dim=-1
            )
            hit |= (force_magnitude > threshold).reshape(
                self.num_envs, -1
            ).any(dim=-1)
        return hit

    @staticmethod
    def self_collision_cost_from_hits(hit_substeps):
        """Count physics substeps containing a hit; input entries are booleans."""

        if not isinstance(hit_substeps, torch.Tensor):
            raise TypeError("self-collision hit history must be a torch.Tensor.")
        if hit_substeps.ndim != 2:
            raise ValueError(
                "self-collision hit history must have shape [num_envs, substeps], "
                f"got {tuple(hit_substeps.shape)}."
            )
        if hit_substeps.dtype != torch.bool:
            raise ValueError("self-collision hit history must have bool dtype.")
        return hit_substeps.sum(dim=-1).float()

    def _compute_torso_upright(self):
        torso_quat_w = torch.nn.functional.normalize(
            self.robot.data.body_link_quat_w[:, self.torso_body_index], dim=-1
        )
        gravity_direction_w = torch.tensor(
            (0.0, 0.0, -1.0),
            dtype=torso_quat_w.dtype,
            device=torso_quat_w.device,
        ).expand(self.num_envs, -1)
        self.torso_projected_gravity_b = quat_apply_inverse(
            torso_quat_w, gravity_direction_w
        )
        self.torso_up_z = -self.torso_projected_gravity_b[:, 2]
        return self.torso_projected_gravity_b, self.torso_up_z

    def _update_recovery_state_metrics(self):
        self.pelvis_height = self.robot.data.root_link_pos_w[:, 2]
        h_default = self.robot.data.default_root_state[:, 2]
        self.height_ok = (
            self.pelvis_height >= self.cfg.success.height_ratio * h_default
        )

        _, self.torso_up_z = self._compute_torso_upright()
        self.upright_ok = (
            self.torso_up_z >= self.cfg.success.upright_threshold
        )
        self.ready_now = self.height_ok & self.upright_ok

    def compute_reward(self):
        """Compute the policy-dt-scaled pre-AMP Recovery task reward."""

        self._update_recovery_state_metrics()
        self.task_reward = self.reward_manager.compute(self.step_dt)

        reward_cfg = self.cfg.reward
        self.height_reward_scaled = (
            self.step_dt
            * reward_cfg.root_height_exp.weight
            * self.height_reward_raw
        )
        self.upright_reward_scaled = (
            self.step_dt * reward_cfg.upright.weight * self.upright_reward_raw
        )
        self.joint_acc_reward = (
            self.step_dt * reward_cfg.dof_acc_l2.weight * self.joint_acc_cost
        )
        self.joint_pos_limits_reward = (
            self.step_dt
            * reward_cfg.dof_pos_limits.weight
            * self.joint_pos_limits_cost
        )
        self.action_rate_reward = (
            self.step_dt
            * reward_cfg.action_rate_l2.weight
            * self.action_rate_cost
        )
        self.self_collision_reward = (
            self.step_dt
            * reward_cfg.self_collisions.weight
            * self.self_collision_cost
        )
        if not torch.isfinite(self.task_reward).all():
            raise RuntimeError("G1 Recovery task reward contains NaN or Inf.")
        return self.task_reward

    def _update_recovery_episode_statistics(self):
        """Accumulate completed-episode outcomes before done environments reset."""

        success_this_step = self.success_buf.sum()
        timeout_this_step = self.time_out_buf.sum()
        completed_this_step = success_this_step + timeout_this_step

        self.recovery_episode_success_this_step.copy_(success_this_step)
        self.recovery_episode_timeout_this_step.copy_(timeout_this_step)
        self.recovery_episode_completed_this_step.copy_(completed_this_step)

        self.recovery_episode_success_total.add_(success_this_step)
        self.recovery_episode_timeout_total.add_(timeout_this_step)
        self.recovery_episode_completed_total.add_(completed_this_step)

        ended_mask = self.success_buf | self.time_out_buf
        duration_sum_this_step = (
            self.episode_length_buf[ended_mask].float() * self.step_dt
        ).sum()
        self.recovery_episode_duration_sum_s.add_(duration_sum_this_step)

    def _build_recovery_episode_log(self):
        """Build lifetime completed-episode statistics for this env instance."""

        completed_denominator = (
            self.recovery_episode_completed_total.float().clamp_min(1.0)
        )
        success_rate = (
            self.recovery_episode_success_total.float()
            / completed_denominator
        )
        timeout_rate = (
            self.recovery_episode_timeout_total.float()
            / completed_denominator
        )
        mean_duration_s = (
            self.recovery_episode_duration_sum_s / completed_denominator
        )
        return {
            "RecoveryEpisode/completed_total": (
                self.recovery_episode_completed_total.clone()
            ),
            "RecoveryEpisode/success_total": (
                self.recovery_episode_success_total.clone()
            ),
            "RecoveryEpisode/timeout_total": (
                self.recovery_episode_timeout_total.clone()
            ),
            "RecoveryEpisode/success_rate": success_rate,
            "RecoveryEpisode/timeout_rate": timeout_rate,
            "RecoveryEpisode/mean_duration_s": mean_duration_s,
            "RecoveryEpisode/completed_this_step": (
                self.recovery_episode_completed_this_step.clone()
            ),
            "RecoveryEpisode/success_this_step": (
                self.recovery_episode_success_this_step.clone()
            ),
            "RecoveryEpisode/timeout_this_step": (
                self.recovery_episode_timeout_this_step.clone()
            ),
        }

    def _build_step_log(self):
        log = {
            "RecoveryReward/height_raw": self.height_reward_raw.mean(),
            "RecoveryReward/height_scaled": self.height_reward_scaled.mean(),
            "RecoveryReward/upright_raw": self.upright_reward_raw.mean(),
            "RecoveryReward/upright_scaled": self.upright_reward_scaled.mean(),
            "RecoveryReward/joint_acc": self.joint_acc_reward.mean(),
            "RecoveryReward/joint_pos_limits": self.joint_pos_limits_reward.mean(),
            "RecoveryReward/action_rate": self.action_rate_reward.mean(),
            "RecoveryReward/self_collision": self.self_collision_reward.mean(),
            "RecoveryReward/task_total": self.task_reward.mean(),
            "Recovery/height_ok_ratio": self.height_ok.float().mean(),
            "Recovery/upright_ok_ratio": self.upright_ok.float().mean(),
            "Recovery/ready_now_ratio": self.ready_now.float().mean(),
            "Recovery/success_count": self.success_buf.float().sum(),
            # Backward-compatible step-level event ratio; this is not an
            # episode-level Recovery success rate.
            "Recovery/success_ratio": self.success_buf.float().mean(),
            "Recovery/step_success_event_ratio": self.success_buf.float().mean(),
            "Recovery/timeout_count": self.time_out_buf.float().sum(),
            "Recovery/success_hold_counter_mean": self.success_hold_counter.float().mean(),
            "Recovery/pelvis_height_mean": self.pelvis_height.mean(),
            "Recovery/torso_up_z_mean": self.torso_up_z.mean(),
            "RecoverySelfCollision/cost_mean": self.self_collision_cost.mean(),
            "RecoverySelfCollision/hit_env_ratio": (
                self.self_collision_cost > 0.0
            ).float().mean(),
        }
        log.update(self._build_recovery_episode_log())
        return log

    def step(self, actions: torch.Tensor):
        expected_shape = (self.num_envs, self.num_actions)
        if not isinstance(actions, torch.Tensor):
            raise TypeError("Recovery actions must be a torch.Tensor.")
        if actions.shape != expected_shape:
            raise ValueError(
                f"Recovery actions must have shape {expected_shape}, "
                f"got {tuple(actions.shape)}."
            )

        actions = actions.to(device=self.device, dtype=torch.float)
        if not torch.isfinite(actions).all():
            raise ValueError("Recovery actions contain NaN or Inf values.")

        self.action = torch.clip(
            actions, -self.clip_actions, self.clip_actions
        )
        self.joint_pos_target = self._compute_joint_position_target(self.action)
        self.self_collision_hit_substeps.zero_()

        for substep_index in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(self.joint_pos_target)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.self_collision_hit_substeps[:, substep_index] = (
                self._read_self_collision_hits()
            )

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        reward_buf = self.compute_reward()
        self.reset_buf, self.time_out_buf = self.check_reset()
        self._update_recovery_episode_statistics()
        self.last_action.copy_(self.action)

        self.extras = {
            "time_outs": self.time_out_buf.clone(),
            "terminated": self.terminated_buf.clone(),
            "log": self._build_step_log(),
        }

        # Terminal AMP 必须在 dataset-based fallen reset 之前采集。
        self.pre_reset_amp_obs = self.get_amp_obs_for_expert_trans().detach()
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.reset_env_ids.numel() > 0:
            self.extras["terminal_amp_states"] = self.pre_reset_amp_obs[
                self.reset_env_ids
            ].clone()
            self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras.setdefault("observations", {})["critic"] = critic_obs
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        self.success_hold_counter = torch.where(
            self.ready_now,
            self.success_hold_counter + 1,
            torch.zeros_like(self.success_hold_counter),
        )
        self.success_buf = self.success_hold_counter >= self.success_hold_steps
        self.terminated_buf = self.success_buf.clone()

        raw_time_out = self.episode_length_buf >= self.max_episode_length
        # If success and the horizon coincide, true termination takes priority;
        # only non-success horizon exits receive PPO timeout bootstrapping.
        time_out_buf = raw_time_out & ~self.success_buf
        reset_buf = self.success_buf | time_out_buf
        return reset_buf, time_out_buf

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        if "observations" not in self.extras:
            self.extras["observations"] = {}
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, self.extras


__all__ = ["G1RecoveryEnv"]
