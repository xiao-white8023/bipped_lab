from __future__ import annotations

import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.g1.g1_recovery_cfg import G123RECOVERYENVCFG
from legged_lab.utils.env_utils.scene import SceneCfg

from rsl_rl.env import VecEnv


class G1RecoveryEnv(VecEnv):

    def __init__(
        self,
        cfg: (
            G123RECOVERYENVCFG
        ),
        headless,
    ):
        self.cfg: (
            G123RECOVERYENVCFG
        )

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()
        # 为每一个环境创建一个id
        env_ids = torch.arange(self.num_envs, device=self.device)

        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self._validate_robot_asset()
        print("G1 recovery joint order:", self.robot.joint_names)
        print("G1 recovery body names:", self.robot.body_names)
        print("G1 recovery foot body names:", self.support_foot_body_names)
        print("G1 recovery support points local:", self.cfg.support_polygon.support_points_local)
        print("G1 recovery collision sphere radius:", self.cfg.support_polygon.collision_sphere_radius)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            self.noisy = self.cfg.noise.noise_scales
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.action = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.ankle_link_ids, self.support_foot_body_names = self.robot.find_bodies(
            name_keys=self.cfg.support_polygon.foot_body_names,
            preserve_order=True,
        )
        if len(self.ankle_link_ids) != 2:
            raise RuntimeError(f"G1 recovery expected two ankle roll links, got {self.support_foot_body_names}.")

        self.support_contact_cfg = SceneEntityCfg(
            name="contact_sensor",
            body_names=self.cfg.support_polygon.foot_body_names,
        )
        self.support_contact_cfg.resolve(self.scene)

        self.pelvis_contact_cfg = SceneEntityCfg(name="contact_sensor", body_names=["pelvis"])
        self.pelvis_contact_cfg.resolve(self.scene)

        self._initialize_joint_groups()
        self._initialize_extreme_reset()
        self._initialize_command_curriculum()
        self._initialize_zmp_buffers()
        self.obs_noisy_vec_and_buffer()
        self._validate_hwc_observation_dims()

    def _initialize_joint_groups(self):
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.waist_ids, _ = self.robot.find_joints(name_keys=["waist_yaw_joint"], preserve_order=True)
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
            ],
            preserve_order=True,
        )
        amp_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]
        self.amp_joint_ids, self.amp_joint_names = self.robot.find_joints(amp_joint_names, preserve_order=True)
        if len(self.amp_joint_ids) != 19:
            raise RuntimeError(f"G1 recovery AMP expected 19 non-ankle joints, got {self.amp_joint_names}.")

        self.wrist_link_ids, self.wrist_link_names = self.robot.find_bodies(
            name_keys=["left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"],
            preserve_order=True,
        )
        if len(self.wrist_link_ids) != 2:
            raise RuntimeError(f"G1 recovery AMP expected two wrist links, got {self.wrist_link_names}.")

    def _initialize_extreme_reset(self):
        reset_cfg = self.cfg.recovery_reset
        self.extreme_data = None
        self.extreme_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not reset_cfg.extreme_reset_enable:
            return

        extreme_data = np.load(reset_cfg.extreme_data_path, allow_pickle=True)
        if extreme_data.ndim != 2 or extreme_data.shape[1] != 57:
            raise RuntimeError(
                "G1 recovery extreme data should have shape [N, 57], "
                f"got {tuple(extreme_data.shape)} from {reset_cfg.extreme_data_path}."
            )
        if not np.isfinite(extreme_data).all():
            raise RuntimeError(f"G1 recovery extreme data contains NaN or Inf: {reset_cfg.extreme_data_path}.")

        self.extreme_data = torch.as_tensor(extreme_data, dtype=torch.float, device=self.device)
        print(f"G1 recovery loaded extreme data: {reset_cfg.extreme_data_path}, shape={tuple(extreme_data.shape)}")

    def _initialize_command_curriculum(self):
        curriculum_cfg = self.cfg.command_curriculum
        self._command_start_ranges = {
            "lin_vel_x": tuple(curriculum_cfg.start_ranges.lin_vel_x),
            "lin_vel_y": tuple(curriculum_cfg.start_ranges.lin_vel_y),
            "ang_vel_z": tuple(curriculum_cfg.start_ranges.ang_vel_z),
        }
        self._command_target_ranges = {
            "lin_vel_x": tuple(curriculum_cfg.target_ranges.lin_vel_x),
            "lin_vel_y": tuple(curriculum_cfg.target_ranges.lin_vel_y),
            "ang_vel_z": tuple(curriculum_cfg.target_ranges.ang_vel_z),
        }
        self._update_command_curriculum()

    def _initialize_zmp_buffers(self):
        support_cfg = self.cfg.support_polygon
        local_points = torch.tensor(support_cfg.support_points_local, dtype=torch.float, device=self.device)
        if local_points.shape != (4, 3):
            raise RuntimeError(f"support_points_local should have shape [4, 3], got {tuple(local_points.shape)}.")

        self.support_points_local = local_points
        self.num_feet = 2
        self.num_points_per_foot = local_points.shape[0]
        self.num_support_points = self.num_feet * self.num_points_per_foot
        self.collision_sphere_radius = float(support_cfg.collision_sphere_radius)

        # [num_envs, 2, 4, 3]
        self.support_points_world = torch.zeros(
            self.num_envs,
            self.num_feet,
            self.num_points_per_foot,
            3,
            dtype=torch.float,
            device=self.device,
        )
        # [num_envs, 2, 4]
        self.active_support_point_mask = torch.zeros(
            self.num_envs,
            self.num_feet,
            self.num_points_per_foot,
            dtype=torch.bool,
            device=self.device,
        )
        self.support_plane_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.support_is_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_support_feet = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # [num_envs, num_bodies]
        self.body_masses = torch.zeros(
            self.num_envs,
            self.robot.num_bodies,
            dtype=torch.float,
            device=self.device,
        )
        self.total_body_mass = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.com_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.com_vel_world = torch.zeros_like(self.com_pos_world)
        self.filtered_com_vel_world = torch.zeros_like(self.com_pos_world)
        self.last_filtered_com_vel_world = torch.zeros_like(self.com_pos_world)
        self.com_acc_world = torch.zeros_like(self.com_pos_world)
        self.zmp_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.stability_margin = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.zmp_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.zmp_cost = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_zmp_cost_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.reset_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_pelvis_contact_buf = torch.zeros_like(self.reset_contact_buf)
        self.reset_roll_buf = torch.zeros_like(self.reset_contact_buf)
        self.reset_pitch_buf = torch.zeros_like(self.reset_contact_buf)

        self._support_pair_ids = torch.combinations(torch.arange(self.num_support_points), r=2).to(self.device)
        gravity = torch.tensor(self.sim.cfg.gravity, dtype=torch.float, device=self.device)
        self.gravity_magnitude = torch.linalg.vector_norm(gravity).clamp_min(1.0e-6)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras.setdefault("episode", {})
        episode_extras = self._get_episode_extras(env_ids)

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        self._update_command_curriculum()
        self.command_generator.reset(env_ids)
        self._apply_recovery_reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        episode_extras.update(reward_extras)
        self.extras["episode"] = episode_extras
        self.extras["time_outs"] = self.time_out_buf.clone()

        self.proprio_history_buf[env_ids] = 0.0
        self.privileged_history_buf[env_ids] = 0.0
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()
        self._reset_recovery_state(env_ids)

    def _apply_recovery_reset(self, env_ids: torch.Tensor):
        reset_cfg = self.cfg.recovery_reset
        self.extreme_reset_buf[env_ids] = False
        if reset_cfg.extreme_reset_enable and self.extreme_data is not None and reset_cfg.extreme_reset_prob > 0.0:
            use_extreme = torch.rand(len(env_ids), device=self.device) < reset_cfg.extreme_reset_prob
        else:
            use_extreme = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        normal_env_ids = env_ids[~use_extreme]
        extreme_env_ids = env_ids[use_extreme]
        if normal_env_ids.numel() > 0:
            self._apply_random_dangerous_reset(normal_env_ids)
        if extreme_env_ids.numel() > 0:
            self._apply_extreme_replay_reset(extreme_env_ids)
            self.extreme_reset_buf[extreme_env_ids] = True

    def _sample_root_positions(self, env_ids: torch.Tensor, root_states: torch.Tensor, z_samples: torch.Tensor):
        reset_cfg = self.cfg.recovery_reset
        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids]
        if reset_cfg.randomize_terrain_xy:
            x_range = reset_cfg.terrain_x_range
            y_range = reset_cfg.terrain_y_range
        else:
            x_range = reset_cfg.pose_range.get("x", (0.0, 0.0))
            y_range = reset_cfg.pose_range.get("y", (0.0, 0.0))

        positions[:, 0] += math_utils.sample_uniform(x_range[0], x_range[1], (len(env_ids),), device=self.device)
        positions[:, 1] += math_utils.sample_uniform(y_range[0], y_range[1], (len(env_ids),), device=self.device)
        positions[:, 2] += z_samples
        return positions

    def _apply_random_dangerous_reset(self, env_ids: torch.Tensor):
        reset_cfg = self.cfg.recovery_reset
        root_states = self.robot.data.default_root_state[env_ids].clone()

        pose_keys = ["z", "roll", "pitch", "yaw"]
        pose_ranges = torch.tensor(
            [reset_cfg.pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
            dtype=torch.float,
            device=self.device,
        )
        pose_samples = math_utils.sample_uniform(
            pose_ranges[:, 0],
            pose_ranges[:, 1],
            (len(env_ids), len(pose_keys)),
            device=self.device,
        )

        positions = self._sample_root_positions(env_ids, root_states, pose_samples[:, 0])
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_samples[:, 1],
            pose_samples[:, 2],
            pose_samples[:, 3],
        )
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

        velocity_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        velocity_ranges = torch.tensor(
            [reset_cfg.velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
            dtype=torch.float,
            device=self.device,
        )
        velocity_samples = math_utils.sample_uniform(
            velocity_ranges[:, 0],
            velocity_ranges[:, 1],
            (len(env_ids), len(velocity_keys)),
            device=self.device,
        )
        velocities = root_states[:, 7:13] + velocity_samples

        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos += math_utils.sample_uniform(
            reset_cfg.joint_pos_offset_range[0],
            reset_cfg.joint_pos_offset_range[1],
            joint_pos.shape,
            device=self.device,
        )
        joint_vel += math_utils.sample_uniform(
            reset_cfg.joint_vel_range[0],
            reset_cfg.joint_vel_range[1],
            joint_vel.shape,
            device=self.device,
        )

        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids]
        joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _apply_extreme_replay_reset(self, env_ids: torch.Tensor):
        reset_cfg = self.cfg.recovery_reset
        if self.extreme_data is None:
            raise RuntimeError("G1 recovery extreme reset requested, but extreme data is not loaded.")

        root_states = self.robot.data.default_root_state[env_ids].clone()
        z_range = reset_cfg.pose_range.get("z", (0.0, 0.0))
        z_samples = math_utils.sample_uniform(z_range[0], z_range[1], (len(env_ids),), device=self.device)
        positions = self._sample_root_positions(env_ids, root_states, z_samples)

        data_ids = torch.randint(self.extreme_data.shape[0], (len(env_ids),), device=self.device)
        sampled_data = self.extreme_data[data_ids]
        yaw_range = reset_cfg.extreme_yaw_range
        yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids),), device=self.device)
        orientations = math_utils.quat_from_euler_xyz(sampled_data[:, 6], sampled_data[:, 7], yaw)

        velocities = root_states[:, 7:13].clone()
        velocities[:, 0:3] = sampled_data[:, 0:3]
        velocities[:, 3:6] = sampled_data[:, 3:6]

        joint_pos = sampled_data[:, 8:31].clone()
        joint_vel = sampled_data[:, 31:54].clone()
        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids]
        joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        if reset_cfg.extreme_use_data_commands:
            self.command_generator.command[env_ids, :3] = sampled_data[:, 54:57]

    def _reset_recovery_state(self, env_ids: torch.Tensor):
        self.update_body_mass_cache(env_ids)
        self.support_points_world[env_ids] = self._get_foot_support_points_world()[env_ids]
        self._get_active_support_geometry()
        self._compute_com(env_ids)

        self.filtered_com_vel_world[env_ids] = self.com_vel_world[env_ids]
        self.last_filtered_com_vel_world[env_ids] = self.com_vel_world[env_ids]
        self.com_acc_world[env_ids] = 0.0
        self.zmp_xy[env_ids] = self.com_pos_world[env_ids, :2]
        self.stability_margin[env_ids] = 0.0
        self.zmp_distance[env_ids] = 0.0
        self.zmp_cost[env_ids] = 0.0
        self.episode_zmp_cost_sum[env_ids] = 0.0
        self.action[env_ids] = 0.0

    def compute_current_observations(self):
        robot = self.robot

        ang_vel = robot.data.root_ang_vel_b
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                torch.stack((roll, pitch), dim=1),
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
            ],
            dim=-1,
        )
        return current_actor_obs

    def compute_policy_labels(self):
        root_lin_vel = self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel
        zmp = self.frequency_encoding(self.zmp_distance.unsqueeze(-1), self.cfg.hwc_observation.zmp_frequency_count)
        foot_height = self.robot.data.body_link_pos_w[:, self.ankle_link_ids, 2].mean(dim=1)
        base_height = self.robot.data.root_pos_w[:, 2] - (foot_height - 0.05)
        return torch.cat([root_lin_vel, zmp, base_height.unsqueeze(1)], dim=-1)

    def compute_domain_randomization_labels(self):
        return torch.cat(
            [
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1.0,
                self.motor_strength[1] - 1.0,
                self._kp_scale,
                self._kd_scale,
                self.rand_push_force,
                self.rand_push_torque,
                self.height_context,
            ],
            dim=-1,
        )

    def compute_current_privileged_observations(self, current_actor_obs, policy_labels):
        dr_labels = self.compute_domain_randomization_labels()
        return torch.cat([current_actor_obs, dr_labels, policy_labels], dim=-1)

    def frequency_encoding(self, zmp_feature, num_frequencies: int):
        encoding = []
        for i in range(num_frequencies):
            freq = 2 ** i
            encoding.append(torch.sin(freq * torch.pi * zmp_feature))
            encoding.append(torch.cos(freq * torch.pi * zmp_feature))
        return torch.cat(encoding, dim=-1)

    def _update_hwc_history(self, current_actor_obs, current_privileged_obs):
        reset_like_mask = (self.episode_length_buf <= 1).view(-1, 1, 1)
        actor_stack = current_actor_obs.unsqueeze(1).expand(-1, self.hwc_history_buffer_len, -1)
        privileged_stack = current_privileged_obs.unsqueeze(1).expand(-1, self.hwc_history_buffer_len, -1)

        self.proprio_history_buf = torch.where(
            reset_like_mask,
            actor_stack,
            torch.cat([self.proprio_history_buf[:, 1:], current_actor_obs.unsqueeze(1)], dim=1),
        )
        self.privileged_history_buf = torch.where(
            reset_like_mask,
            privileged_stack,
            torch.cat([self.privileged_history_buf[:, 1:], current_privileged_obs.unsqueeze(1)], dim=1),
        )

    def obs_noisy_vec_and_buffer(self):
        hwc_cfg = self.cfg.hwc_observation
        self.num_proprio = hwc_cfg.num_proprio
        self.hwc_prop_hist_len = hwc_cfg.prop_history_len
        self.hwc_history_buffer_len = hwc_cfg.history_buffer_len
        self.policy_label_dim = hwc_cfg.policy_label_dim
        self.dr_label_dim = hwc_cfg.dr_label_dim
        self.privileged_proprio_dim = hwc_cfg.privileged_proprio_dim
        self.actor_obs_dim = hwc_cfg.actor_obs_dim
        self.actor_estimated_obs_dim = hwc_cfg.actor_estimated_obs_dim
        self.critic_obs_dim = hwc_cfg.critic_obs_dim

        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.friction_coeffs_tensor = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.motor_strength = torch.ones(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self._kp_scale = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self._kd_scale = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.rand_push_force = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.height_context = torch.zeros(
            self.num_envs,
            hwc_cfg.height_context_dim,
            dtype=torch.float,
            device=self.device,
        )

        self.proprio_history_buf = torch.zeros(
            self.num_envs,
            self.hwc_history_buffer_len,
            self.num_proprio,
            dtype=torch.float,
            device=self.device,
        )
        self.privileged_history_buf = torch.zeros(
            self.num_envs,
            self.hwc_history_buffer_len,
            self.privileged_proprio_dim,
            dtype=torch.float,
            device=self.device,
        )

        if self.add_noise:
            noise_obs_vec = torch.zeros(self.num_proprio, dtype=torch.float, device=self.device)
            noise_obs_vec[0:3] = self.obs_scales.ang_vel * self.noisy.ang_vel
            noise_obs_vec[3:5] = getattr(self.noisy, "imu", self.noisy.projected_gravity)
            noise_obs_vec[5 : 5 + self.num_actions] = self.obs_scales.joint_pos * self.noisy.joint_pos
            noise_obs_vec[5 + self.num_actions : 5 + self.num_actions * 2] = (
                self.obs_scales.joint_vel * self.noisy.joint_vel
            )
            action_start = 5 + self.num_actions * 2
            gravity_start = action_start + self.num_actions
            noise_obs_vec[action_start:gravity_start] = 0.0
            noise_obs_vec[gravity_start : gravity_start + 3] = (
                self.obs_scales.projected_gravity * self.noisy.projected_gravity
            )
            noise_obs_vec[gravity_start + 3 : gravity_start + 6] = 0.0
            self.noise_obs_vec = noise_obs_vec

    def compute_observations(self):
        self.compute_zmp()
        current_actor_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_obs_vec

        policy_labels = self.compute_policy_labels()
        current_privileged_obs = self.compute_current_privileged_observations(current_actor_obs, policy_labels)

        motion_features = self.proprio_history_buf[:, -self.hwc_prop_hist_len :].reshape(self.num_envs, -1)
        priv_motion_features = self.privileged_history_buf[:, -self.hwc_prop_hist_len :].reshape(self.num_envs, -1)

        self.actor_obs = torch.cat([motion_features, current_actor_obs, policy_labels], dim=-1)
        self.critic_obs = torch.cat([priv_motion_features, current_privileged_obs], dim=-1)
        self._update_hwc_history(current_actor_obs, current_privileged_obs)

        if self.actor_obs.shape[1] != self.actor_obs_dim:
            raise RuntimeError(f"G1 recovery actor obs dim mismatch: {self.actor_obs.shape[1]} != {self.actor_obs_dim}.")
        if self.critic_obs.shape[1] != self.critic_obs_dim:
            raise RuntimeError(f"G1 recovery critic obs dim mismatch: {self.critic_obs.shape[1]} != {self.critic_obs_dim}.")

        self.actor_obs = torch.clip(self.actor_obs, -self.clip_obs, self.clip_obs)
        self.critic_obs = torch.clip(self.critic_obs, -self.clip_obs, self.clip_obs)

        return self.actor_obs, self.critic_obs

    def step(self, actions: torch.Tensor):
        self.extras = {}
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._update_command_curriculum()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.compute_zmp() # 计算ZMP
        self.zmp_cost.copy_(self._compute_zmp_cost()) # 从当前ZMP中加入一个稳定性cost
        self.episode_zmp_cost_sum += self.zmp_cost

        step_zmp_cost = self.zmp_cost.clone()
        step_zmp_distance = self.zmp_distance.clone()
        step_stability_margin = self.stability_margin.clone()
        step_zmp_valid = self.support_is_valid.clone()

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        pre_reset_amp_obs = self.get_amp_obs_for_expert_trans().detach()
        terminal_amp_states = pre_reset_amp_obs[self.reset_env_ids].clone()
        step_walk_amp_mask = self._compute_walk_amp_mask()

        if "observations" not in self.extras:
            self.extras["observations"] = {}
        self.extras["terminal_amp_states"] = terminal_amp_states
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["terminal_amp_states"] = terminal_amp_states
        self.extras["walk_amp_mask"] = step_walk_amp_mask
        self.extras["zmp_cost"] = step_zmp_cost
        self.extras["zmp_distance"] = step_zmp_distance
        self.extras["stability_margin"] = step_stability_margin
        self.extras["zmp_valid"] = step_zmp_valid
        self.extras["time_outs"] = self.time_out_buf.clone()

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        termination_forces = torch.norm(
            net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
            dim=-1,
        )
        self.reset_contact_buf = torch.any(
            torch.max(termination_forces, dim=1)[0] > self.cfg.termination.contact_force_threshold,
            dim=1,
        )

        pelvis_forces = torch.norm(
            net_contact_forces[:, :, self.pelvis_contact_cfg.body_ids],
            dim=-1,
        )
        self.reset_pelvis_contact_buf = torch.any(
            torch.max(pelvis_forces, dim=1)[0] > self.cfg.termination.pelvis_contact_force_threshold,
            dim=1,
        )

        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.reset_roll_buf = torch.abs(roll) > self.cfg.termination.roll_threshold
        self.reset_pitch_buf = torch.abs(pitch) > self.cfg.termination.pitch_threshold
        time_out_buf = self.episode_length_buf >= self.max_episode_length

        reset_buf = (
            self.reset_contact_buf
            | self.reset_pelvis_contact_buf
            | self.reset_roll_buf
            | self.reset_pitch_buf
            | time_out_buf
        )
        return reset_buf, time_out_buf

    def _compute_walk_amp_mask(self) -> torch.Tensor:
        command = self.command_generator.command
        command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        return command_norm > self.cfg.amp_walk_command_threshold

    def update_body_mass_cache(self, env_ids: torch.Tensor | None = None):
        current_masses = self.robot.root_physx_view.get_masses().to(device=self.device, dtype=torch.float)
        if env_ids is None:
            self.body_masses.copy_(current_masses)
        else:
            self.body_masses[env_ids] = current_masses[env_ids]
        self.total_body_mass[:] = self.body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def _get_foot_support_points_world(self) -> torch.Tensor:
        # [num_envs, 2, 3]
        foot_pos_w = self.robot.data.body_link_pos_w[:, self.ankle_link_ids, :]
        # [num_envs, 2, 4]
        foot_quat_w = self.robot.data.body_link_quat_w[:, self.ankle_link_ids, :]

        local_points = self.support_points_local.view(1, 1, self.num_points_per_foot, 3)
        local_points = local_points.expand(self.num_envs, self.num_feet, self.num_points_per_foot, 3)
        foot_quat = foot_quat_w.unsqueeze(2).expand(self.num_envs, self.num_feet, self.num_points_per_foot, 4)

        rotated_points = quat_apply(
            foot_quat.reshape(-1, 4),
            local_points.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_feet, self.num_points_per_foot, 3)

        return rotated_points + foot_pos_w.unsqueeze(2)

    def _get_active_support_geometry(self):
        support_cfg = self.cfg.support_polygon
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        contact_force = torch.norm(
            net_contact_forces[:, :, self.support_contact_cfg.body_ids, :],
            dim=-1,
        )
        foot_contact = torch.max(contact_force, dim=1)[0] > support_cfg.contact_force_threshold

        # sphere centers are stored in support_points_world; contact surface is lower by radius.
        surface_z = self.support_points_world[..., 2] - self.collision_sphere_radius
        inf = torch.full_like(surface_z, float("inf"))
        contact_surface_z = torch.where(foot_contact.unsqueeze(-1), surface_z, inf)
        min_surface_z = contact_surface_z.reshape(self.num_envs, -1).amin(dim=1)
        self.support_is_valid = torch.isfinite(min_surface_z)
        self.support_plane_height = torch.where(
            self.support_is_valid,
            min_surface_z,
            torch.zeros_like(min_surface_z),
        )

        tolerance = support_cfg.support_point_contact_tolerance
        self.active_support_point_mask = (
            foot_contact.unsqueeze(-1)
            & self.support_is_valid.view(-1, 1, 1)
            & (surface_z <= self.support_plane_height.view(-1, 1, 1) + tolerance)
        )
        self.num_support_feet = self.active_support_point_mask.any(dim=2).sum(dim=1)

    def _compute_com(self, env_ids: torch.Tensor | None = None):
        body_com_pos_w = self.robot.data.body_com_pos_w
        body_com_vel_w = self.robot.data.body_com_lin_vel_w
        body_masses = self.body_masses

        if env_ids is not None:
            body_com_pos_w = body_com_pos_w[env_ids]
            body_com_vel_w = body_com_vel_w[env_ids]
            body_masses = body_masses[env_ids]

        total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        com_pos = (body_com_pos_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass
        com_vel = (body_com_vel_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass

        if env_ids is None:
            self.com_pos_world.copy_(com_pos)
            self.com_vel_world.copy_(com_vel)
        else:
            self.com_pos_world[env_ids] = com_pos
            self.com_vel_world[env_ids] = com_vel

    def compute_zmp(self):
        self.support_points_world.copy_(self._get_foot_support_points_world())
        self._get_active_support_geometry()
        self._compute_com()

        alpha = self.cfg.zmp.zmp_com_vel_filter_alpha
        self.last_filtered_com_vel_world.copy_(self.filtered_com_vel_world)
        self.filtered_com_vel_world.mul_(1.0 - alpha).add_(self.com_vel_world, alpha=alpha)
        self.com_acc_world.copy_((self.filtered_com_vel_world - self.last_filtered_com_vel_world) / self.step_dt)

        com_height = (self.com_pos_world[:, 2] - self.support_plane_height).clamp_min(1.0e-4)
        self.zmp_xy[:, 0] = self.com_pos_world[:, 0] - (com_height / self.gravity_magnitude) * self.com_acc_world[:, 0]
        self.zmp_xy[:, 1] = self.com_pos_world[:, 1] - (com_height / self.gravity_magnitude) * self.com_acc_world[:, 1]

        flat_points = self.support_points_world.reshape(self.num_envs, self.num_support_points, 3)
        flat_mask = self.active_support_point_mask.reshape(self.num_envs, self.num_support_points)
        self.stability_margin = self._signed_margin_to_support_patch(self.zmp_xy, flat_points[..., :2], flat_mask)
        self.stability_margin = torch.where(
            self.support_is_valid,
            self.stability_margin,
            torch.full_like(self.stability_margin, -1.0),
        )
        self.zmp_distance = torch.relu(-self.stability_margin)

    def _signed_margin_to_support_patch(
        self,
        query_xy: torch.Tensor,
        support_points_xy: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        # query_xy: [num_envs, 2], support_points_xy: [num_envs, num_support_points, 2]
        eps = 1.0e-6
        num_active = active_mask.sum(dim=1)

        point_distance = torch.linalg.vector_norm(support_points_xy - query_xy.unsqueeze(1), dim=-1)
        point_distance = torch.where(active_mask, point_distance, torch.full_like(point_distance, float("inf")))
        min_point_distance = point_distance.amin(dim=1)

        pair_i = self._support_pair_ids[:, 0]
        pair_j = self._support_pair_ids[:, 1]
        point_i = support_points_xy[:, pair_i, :]
        point_j = support_points_xy[:, pair_j, :]
        edge = point_j - point_i
        edge_len = torch.linalg.vector_norm(edge, dim=-1).clamp_min(eps)
        pair_active = active_mask[:, pair_i] & active_mask[:, pair_j] & (edge_len > eps)

        rel_points = support_points_xy.unsqueeze(1) - point_i.unsqueeze(2)
        cross_points = edge[:, :, 0:1] * rel_points[..., 1] - edge[:, :, 1:2] * rel_points[..., 0]
        masked_left = (cross_points >= -eps) | (~active_mask.unsqueeze(1))
        masked_right = (cross_points <= eps) | (~active_mask.unsqueeze(1))
        valid_left = pair_active & masked_left.all(dim=2)
        valid_right = pair_active & masked_right.all(dim=2)

        rel_query = query_xy.unsqueeze(1) - point_i
        cross_query = edge[..., 0] * rel_query[..., 1] - edge[..., 1] * rel_query[..., 0]
        signed_left = cross_query / edge_len
        signed_right = -cross_query / edge_len
        hull_edge_distance = torch.full_like(signed_left, float("inf"))
        hull_edge_distance = torch.where(valid_left, signed_left, hull_edge_distance)
        hull_edge_distance = torch.where(valid_right, signed_right, hull_edge_distance)
        polygon_margin = hull_edge_distance.amin(dim=1)

        segment_t = torch.sum((query_xy.unsqueeze(1) - point_i) * edge, dim=-1) / torch.square(edge_len)
        segment_t = segment_t.clamp(0.0, 1.0)
        closest_on_segment = point_i + segment_t.unsqueeze(-1) * edge
        segment_distance = torch.linalg.vector_norm(query_xy.unsqueeze(1) - closest_on_segment, dim=-1)
        segment_distance = torch.where(pair_active, segment_distance, torch.full_like(segment_distance, float("inf")))
        min_segment_distance = segment_distance.amin(dim=1)

        degenerate_margin = torch.where(
            num_active >= 2,
            -min_segment_distance,
            -min_point_distance,
        )
        margin = torch.where(
            (num_active >= 3) & torch.isfinite(polygon_margin),
            polygon_margin,
            degenerate_margin,
        )
        margin = torch.where(torch.isfinite(margin), margin, torch.full_like(margin, -1.0))
        return margin

    def _compute_zmp_cost(self) -> torch.Tensor:
        zmp_cfg = self.cfg.zmp
        if not zmp_cfg.use_zmp_cost:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        valid_support = self.support_is_valid
        if zmp_cfg.zmp_cost_type == "indicator":
            cost = (self.stability_margin < 0.0).float()
        elif zmp_cfg.zmp_cost_type == "margin":
            outside = torch.relu(-self.stability_margin - zmp_cfg.zmp_margin_slack)
            outside = torch.clamp(outside, max=zmp_cfg.zmp_cost_clip)
            support_weight = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            support_weight = torch.where(
                self.num_support_feet <= 1,
                torch.full_like(support_weight, zmp_cfg.zmp_single_support_weight),
                torch.full_like(support_weight, zmp_cfg.zmp_double_support_weight),
            )
            cost = outside * support_weight
        else:
            raise ValueError(f"Unsupported zmp_cost_type: {zmp_cfg.zmp_cost_type}")

        no_contact_cost = torch.full_like(cost, zmp_cfg.zmp_no_contact_cost)
        return torch.where(valid_support, cost, no_contact_cost)

    def _update_command_curriculum(self):
        curriculum_cfg = self.cfg.command_curriculum
        if not curriculum_cfg.enable:
            return

        global_step = self.sim_step_counter // self.cfg.sim.decimation
        progress = min(float(global_step) / max(float(curriculum_cfg.steps), 1.0), 1.0)
        ranges = self.command_generator.cfg.ranges
        ranges.lin_vel_x = self._lerp_range(self._command_start_ranges["lin_vel_x"], self._command_target_ranges["lin_vel_x"], progress)
        ranges.lin_vel_y = self._lerp_range(self._command_start_ranges["lin_vel_y"], self._command_target_ranges["lin_vel_y"], progress)
        ranges.ang_vel_z = self._lerp_range(self._command_start_ranges["ang_vel_z"], self._command_target_ranges["ang_vel_z"], progress)

    @staticmethod
    def _lerp_range(start_range: tuple[float, float], target_range: tuple[float, float], progress: float) -> tuple[float, float]:
        return (
            start_range[0] + (target_range[0] - start_range[0]) * progress,
            start_range[1] + (target_range[1] - start_range[1]) * progress,
        )

    # 记录此次reset的cost以及是因为什么原因而重置的
    def _get_episode_extras(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if env_ids.numel() == 0:
            return {}

        episode_len = self.episode_length_buf[env_ids].float().clamp_min(1.0)
        zmp_cost_mean = self.episode_zmp_cost_sum[env_ids] / episode_len
        extras = {
            "cost_zmp": zmp_cost_mean.mean(),
            "Episode_Cost/zmp_sum": self.episode_zmp_cost_sum[env_ids].mean(),
            "Episode_Cost/zmp_mean": zmp_cost_mean.mean(),
        }
        if hasattr(self, "reset_contact_buf"):
            extras.update(
                {
                    "reset_contact": self.reset_contact_buf[env_ids].float().mean(),
                    "reset_pelvis_contact": self.reset_pelvis_contact_buf[env_ids].float().mean(),
                    "reset_roll": self.reset_roll_buf[env_ids].float().mean(),
                    "reset_pitch": self.reset_pitch_buf[env_ids].float().mean(),
                    "reset_timeout": self.time_out_buf[env_ids].float().mean(),
                    "reset_extreme": self.extreme_reset_buf[env_ids].float().mean(),
                }
            )
        return extras

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance
            < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        if "observations" not in self.extras:
            self.extras["observations"] = {}
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        joint_pos = self.robot.data.joint_pos[:, self.amp_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.amp_joint_ids]

        left_hand_pos = self.robot.data.body_state_w[:, self.wrist_link_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        right_hand_pos = self.robot.data.body_state_w[:, self.wrist_link_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        left_foot_pos = self.robot.data.body_state_w[:, self.ankle_link_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        right_foot_pos = self.robot.data.body_state_w[:, self.ankle_link_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]

        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])
        left_hand_pos = quat_apply(root_quat_inv, left_hand_pos)
        right_hand_pos = quat_apply(root_quat_inv, right_hand_pos)
        left_foot_pos = quat_apply(root_quat_inv, left_foot_pos)
        right_foot_pos = quat_apply(root_quat_inv, right_foot_pos)

        amp_obs = torch.cat(
            [
                joint_pos,
                joint_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ],
            dim=-1,
        )
        if amp_obs.shape[1] != 50:
            raise RuntimeError(f"G1 recovery AMP obs should have 50 dims, got {tuple(amp_obs.shape)}.")
        return amp_obs
    '''
    在 Recovery 环境正式训练之前，对你加载的 G1 URDF 做一次“结构一致性检查”，确保后面 ZMP / support polygon 的计算建立在正确的机器人模型上
    而它主要检查两件事情：
        1.这个 URDF 是不是你预期的 23-DoF G1。
        2.你代码里人为配置的 脚底四个支撑点，是不是和 URDF 里脚底实际的四个球形 collision 完全一致。
    '''
    def _validate_robot_asset(self):
        asset_path = self.cfg.scene.robot.spawn.asset_path
        root = ET.parse(asset_path).getroot()
        joint_names = [joint.get("name") for joint in root.findall("joint") if joint.get("type") != "fixed"]
        if len(joint_names) != 23:
            raise RuntimeError(f"G1 recovery URDF should contain 23 non-fixed joints, got {len(joint_names)}.")

        cfg_points = sorted(tuple(round(float(value), 6) for value in point) for point in self.cfg.support_polygon.support_points_local)
        expected_radius = round(float(self.cfg.support_polygon.collision_sphere_radius), 6)
        for foot_name in self.cfg.support_polygon.foot_body_names:
            link = root.find(f"link[@name='{foot_name}']")
            if link is None:
                raise RuntimeError(f"Could not find foot link {foot_name} in {asset_path}.")
            actual_points = []
            actual_radii = []
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                geometry = collision.find("geometry")
                sphere = geometry.find("sphere") if geometry is not None else None
                if origin is None or sphere is None:
                    continue
                actual_points.append(tuple(round(float(value), 6) for value in origin.get("xyz").split()))
                actual_radii.append(round(float(sphere.get("radius")), 6))
            if sorted(actual_points) != cfg_points:
                raise RuntimeError(
                    f"Configured support points do not match {foot_name} sphere collision origins: {actual_points}."
                )
            if any(radius != expected_radius for radius in actual_radii):
                raise RuntimeError(f"Configured sphere radius does not match {foot_name}: {actual_radii}.")

    def _validate_hwc_observation_dims(self):
        if self.num_actions != 23:
            raise RuntimeError(f"G1 recovery HWC migration expects 23 actions, got {self.num_actions}.")
        expected_proprio = 3 + 2 + self.num_actions * 3 + 3 + 3
        if self.num_proprio != expected_proprio:
            raise RuntimeError(f"G1 recovery proprio dim mismatch: {self.num_proprio} != {expected_proprio}.")
        expected_policy_labels = 3 + self.cfg.hwc_observation.zmp_frequency_count * 2 + 1
        if self.policy_label_dim != expected_policy_labels:
            raise RuntimeError(
                f"G1 recovery policy label dim mismatch: {self.policy_label_dim} != {expected_policy_labels}."
            )
        print("G1 recovery HWC tensor dims:")
        print(f"  single proprio: {self.num_proprio}")
        print(f"  actor raw obs: {self.actor_obs_dim}")
        print(f"  actor estimated obs: {self.actor_estimated_obs_dim}")
        print(f"  policy labels: {self.policy_label_dim}")
        print(f"  DR/context labels: {self.dr_label_dim}")
        print(f"  critic obs: {self.critic_obs_dim}")

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
