"""G1 5 Hz 高层模式选择环境；冻结 RENet 在内部按 50 Hz 执行。"""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, patterns
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.math import quat_apply, quat_conjugate

from legged_lab.envs.g1.g1_selector_cfg import G1SelectorEnvCfg
from legged_lab.perception.lp_height import observation_contract, resolve_pgtt_height_cfg
from legged_lab.selector.frozen_policy import load_frozen_adapter
from legged_lab.high_level.high_level_buffers import SelectorActionBuffer, SelectorHistoryBuffer
from legged_lab.high_level.high_level_faults import HighLevelSensorFaultPipeline, SensorFrame
from legged_lab.selector.randomness import EnvironmentRandomness
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg
from legged_lab.utils.env_utils.scene import SceneCfg

from rsl_rl.env import VecEnv


class SelectorRewardManager(RewardManager):
    """保持原奖励管理器语义，仅在逐项累计前屏蔽已终止环境。"""

    def compute(self, dt: float, active_mask: torch.Tensor) -> torch.Tensor:
        self._reward_buf.zero_()
        self._step_reward.zero_()
        for index, (name, term) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term.weight == 0.0:
                continue
            value = term.func(self._env, **term.params) * term.weight * dt
            # 不能只屏蔽最终总奖励，也不能用 NaN * 0 处理停止行。
            value = torch.where(active_mask, value, 0.0)
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, index] = value / dt
        return self._reward_buf


class SelectorVelocityCommand(UniformVelocityCommand):
    """按底层秒数和活动行推进命令，不依赖高层 env.step_dt。"""

    def _resample(self, env_ids):
        if len(env_ids) == 0:
            return
        rng = self._env.selector_randomness
        low, high = self.cfg.resampling_time_range
        self.time_left[env_ids] = low + (high - low) * rng.uniform(env_ids, (), "command_duration")
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        rng = self._env.selector_randomness
        samples = rng.uniform(env_ids, (6,), "command_values")
        for column, bounds in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y,
                                         self.cfg.ranges.ang_vel_z)):
            self.vel_command_b[env_ids, column] = bounds[0] + (bounds[1] - bounds[0]) * samples[:, column]
        if self.cfg.heading_command:
            low, high = self.cfg.ranges.heading
            self.heading_target[env_ids] = low + (high - low) * samples[:, 3]
            self.is_heading_env[env_ids] = samples[:, 4] < self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = samples[:, 5] < self.cfg.rel_standing_envs

    def _update_command(self, env_ids=None):
        ids = self._env._env_ids(env_ids)
        if self.cfg.heading_command:
            heading_ids = ids[self.is_heading_env[ids]]
            error = self.heading_target[heading_ids] - self.robot.data.heading_w[heading_ids]
            error = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
            self.vel_command_b[heading_ids, 2] = (
                self.cfg.heading_control_stiffness * error
            ).clamp(*self.cfg.ranges.ang_vel_z)
        self.vel_command_b[ids[self.is_standing_env[ids]]] = 0.0

    def reset(self, env_ids=None):
        ids = self._env._env_ids(env_ids)
        if not len(ids):
            return {}
        metrics = super().reset(ids)
        # 父类 reset 只采样；首帧也必须满足 heading/站立命令约定。
        self._update_command(ids)
        return metrics

    def compute(self, dt: float, env_ids=None):
        ids = self._env._env_ids(env_ids)
        if not len(ids):
            return
        scale = dt / self.cfg.resampling_time_range[1]
        self.metrics["error_vel_xy"][ids] += scale * torch.linalg.vector_norm(
            self.vel_command_b[ids, :2] - self.robot.data.root_lin_vel_b[ids, :2], dim=-1)
        self.metrics["error_vel_yaw"][ids] += scale * (
            self.vel_command_b[ids, 2] - self.robot.data.root_ang_vel_b[ids, 2]).abs()
        self.time_left[ids] -= dt
        self._resample(ids[self.time_left[ids] <= 0.0])
        self._update_command(ids)


class G1SelectorEnv(VecEnv):
    def __init__(self, cfg: G1SelectorEnvCfg, headless):
        self.cfg = cfg
        self.cfg.resolve_observations()
        self.headless = headless
        self.device = self.cfg.device
        self.num_envs = self.cfg.scene.num_envs
        self.physics_dt = self.cfg.sim.dt
        self.low_level_dt = self.cfg.selector.low_level_dt
        self.high_level_dt = 1.0 / self.cfg.selector.frequency
        self.step_dt = self.high_level_dt
        self.seed(self.cfg.seed)
        self.selector_randomness = EnvironmentRandomness(self.num_envs, self.cfg.seed, self.device)

        # ---------------------------------------------------------
        # 1. 仿真与场景：场景传感器使用底层 0.02 s，而非高层 0.2 s。
        # ---------------------------------------------------------
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device, dt=self.physics_dt, render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        self.lp_height_source = cfg.renet.lp_height_source
        self.lp_runtime_cfg = resolve_pgtt_height_cfg(cfg.lp_height, self.lp_height_source)
        self.sensor_interval = round(self.lp_runtime_cfg.update_period / self.low_level_dt)
        scene_config = copy.deepcopy(cfg.scene)
        scene_config.lidar = None
        if scene_config.terrain_generator is not None and scene_config.terrain_generator.seed is None:
            scene_config.terrain_generator.seed = cfg.seed
        scene_cfg = SceneCfg(scene_config, physics_dt=self.physics_dt, step_dt=self.low_level_dt)
        scene_cfg.contact_sensor.track_air_time = False
        lp = self.lp_runtime_cfg
        scene_cfg.pgtt_height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.5 * sum(lp.x_range), 0.0, 20.0)),
            ray_alignment="yaw", update_period=lp.update_period, max_distance=40.0, debug_vis=False,
            pattern_cfg=patterns.GridPatternCfg(
                resolution=lp.resolution,
                size=(lp.x_range[1] - lp.x_range[0], lp.y_range[1] - lp.y_range[0]),
                direction=(0.0, 0.0, -1.0), ordering="xy",
            ),
            mesh_prim_paths=["/World/ground"], drift_range=(0.0, 0.0),
        )
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.height_scanner = self.scene.sensors["height_scanner"]
        self.lp_sensor = self.scene.sensors["pgtt_height_scanner"]
        self.camera: GroupedRayCasterCamera = self.scene.sensors["camera"]
        self.left_feet_ray_caster = self.scene.sensors.get("left_feet_ray_caster")
        self.right_feet_ray_caster = self.scene.sensors.get("right_feet_ray_caster")

        # ---------------------------------------------------------
        # 2. 冻结策略、缓存和管理器；不创建高层网络或 AMP 判别器。
        # ---------------------------------------------------------
        self.frozen_policy, reference, self.frozen_policy_metadata = load_frozen_adapter(
            cfg.low_level.checkpoint_path, cfg.low_level.reference_policy_path, device=self.device)
        del reference
        self.lp_observation_contract = observation_contract(
            lp, self.lp_height_source, height_offset=cfg.normalization.height_scan_offset,
            height_scale=cfg.normalization.obs_scales.height_scan)
        self._validate_lp_observation_contract()
        self.init_buffers()
        self.faults = HighLevelSensorFaultPipeline(
            self.num_envs, cfg.faults, seed=cfg.seed, device=self.device)
        self.sensor_observation = self.faults.read()
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot", resampling_time_range=cfg.commands.resampling_time_range,
            rel_standing_envs=cfg.commands.rel_standing_envs, rel_heading_envs=cfg.commands.rel_heading_envs,
            heading_command=cfg.commands.heading_command,
            heading_control_stiffness=cfg.commands.heading_control_stiffness,
            debug_vis=cfg.commands.debug_vis, ranges=cfg.commands.ranges,
        )
        self.command_generator = SelectorVelocityCommand(command_cfg, self)
        self.reward_manager = SelectorRewardManager(cfg.reward, self)
        self.event_manager = EventManager(cfg.domain_rand.events, self)
        self.amp_loader_display = None
        self.reset()
        print("训练时的真实关节顺序:", self.robot.joint_names)

    def _validate_lp_observation_contract(self):
        expected = copy.deepcopy(self.frozen_policy_metadata.get("lp_observation_contract"))
        if expected is None:
            raise ValueError("Frozen checkpoint is missing its LP observation contract.")
        # 归档 checkpoint 使用重命名前的 pgtt_full；只兼容名称，几何和数值契约逐项严格比较。
        if expected.get("source") == "pgtt_full":
            expected["source"] = "pgtt_full_atten"
        if expected != self.lp_observation_contract:
            raise ValueError("Environment LP observations differ from the frozen checkpoint contract.")

    def init_buffers(self):
        self.extras = {"log": {}, "observations": {}}
        self.num_actions = 1
        self.low_level_num_actions = self.robot.data.default_joint_pos.shape[1]
        if self.low_level_num_actions != 23:
            raise ValueError("Frozen RENet requires the original 23-joint action layout.")
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.high_level_dt)
        self.low_level_max_episode_length = math.ceil(self.max_episode_length_s / self.low_level_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.low_level_episode_length_buf = torch.zeros_like(self.episode_length_buf)
        # 独立整数传感器时钟，不使用 runner 可能改写的 episode_length_buf。
        self.sensor_control_ticks = torch.zeros_like(self.episode_length_buf)
        self.sensor_sequence = torch.full_like(self.episode_length_buf, -1)
        self.sensor_capture_count = torch.zeros_like(self.episode_length_buf)
        self.lp_last_scan_count = torch.full_like(self.episode_length_buf, -1)
        self.sim_step_counter = 0
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = self.cfg.robot.action_scale
        self.obs_scales = self.cfg.normalization.obs_scales
        self.action = torch.zeros(self.num_envs, 23, device=self.device)
        self.processed_actions = self.robot.data.default_joint_pos.clone()
        self.action_buffer = SelectorActionBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, 23, self.device)
        self.actor_obs_buffer = SelectorHistoryBuffer(10, self.num_envs, 78, self.device)
        self.renet_estimator_mask = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_switch_time = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self.selector_switch_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf = torch.zeros_like(self.selector_switch_event)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.contact_failure_buf = torch.zeros_like(self.reset_buf)
        self.tilt_failure_buf = torch.zeros_like(self.reset_buf)
        self.active_mask = torch.ones_like(self.reset_buf)
        self.actor_obs = torch.zeros(self.num_envs, self.cfg.observation.actor_obs_dim, device=self.device)
        self.critic_obs = torch.zeros(self.num_envs, self.cfg.observation.critic_obs_dim, device=self.device)
        self.low_level_obs = torch.zeros(self.num_envs, 2120, device=self.device)
        self.feet_height_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.terrain_height_valid = torch.zeros(self.num_envs, 187, dtype=torch.bool, device=self.device)

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names)
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        if len(self.feet_cfg.body_ids) != 2:
            raise ValueError("Selector critic requires exactly two feet contact bodies.")
        self.torso_link_ids, _ = self.robot.find_bodies("torso_link", preserve_order=True)
        self.ankle_link_ids, _ = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
        self.wrist_link_ids, _ = self.robot.find_bodies(
            ["left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"], preserve_order=True)
        if len(self.torso_link_ids) != 1:
            raise ValueError("Selector height sampling requires exactly one torso_link.")
        joint_groups = (
            ("left_leg", ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                          "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]),
            ("right_leg", ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                           "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]),
            ("waist", ["waist_yaw_joint"]),
            ("left_arm", ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                          "left_elbow_joint", "left_wrist_roll_joint"]),
            ("right_arm", ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                           "right_elbow_joint", "right_wrist_roll_joint"]),
        )
        amp_names = []
        for group, names in joint_groups:
            indices, found = self.robot.find_joints(names, preserve_order=True)
            if found != names:
                raise ValueError(f"Unexpected G1 joint order for {group}: {found}")
            setattr(self, group + "_ids", indices)
            amp_names.extend(names)
        self.amp_observation_dim = self.cfg.amp_observation_dim
        if self.amp_observation_dim == 50:
            amp_names = [name for name in amp_names if "ankle" not in name]
        elif self.amp_observation_dim != 58:
            raise ValueError("AMP compatibility observations must have 50 or 58 dimensions.")
        self.amp_joint_ids, self.amp_joint_names = self.robot.find_joints(amp_names, preserve_order=True)

        self.noise_obs_vec = torch.zeros(78, device=self.device)
        if self.cfg.noise.add_noise:
            noise = self.cfg.noise.noise_scales
            self.noise_obs_vec[:3] = self.obs_scales.ang_vel * noise.ang_vel
            self.noise_obs_vec[3:6] = self.obs_scales.projected_gravity * noise.projected_gravity
            self.noise_obs_vec[9:32] = self.obs_scales.joint_pos * noise.joint_pos
            self.noise_obs_vec[32:55] = self.obs_scales.joint_vel * noise.joint_vel

    def _env_ids(self, env_ids=None):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("env_ids must be a one-dimensional integer tensor.")
        ids = ids.long()
        if ((ids < 0) | (ids >= self.num_envs)).any() or len(ids.unique()) != len(ids):
            raise ValueError("env_ids must be unique valid environment indices.")
        return ids

    def compute_current_observations(self, env_ids=None):
        ids = self._env_ids(env_ids)
        robot = self.robot.data
        previous_action = self.action_buffer._circular_buffer.buffer[ids, -1]
        current_actor_obs = torch.cat((
            robot.root_ang_vel_b[ids] * self.obs_scales.ang_vel,
            robot.projected_gravity_b[ids] * self.obs_scales.projected_gravity,
            self.command_generator.command[ids] * self.obs_scales.commands,
            (robot.joint_pos[ids] - robot.default_joint_pos[ids]) * self.obs_scales.joint_pos,
            (robot.joint_vel[ids] - robot.default_joint_vel[ids]) * self.obs_scales.joint_vel,
            previous_action * self.obs_scales.actions,
        ), dim=-1)
        forces = self.contact_sensor.data.net_forces_w_history[ids][:, :, self.feet_cfg.body_ids]
        contact = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > 0.5
        privileged = torch.cat((robot.root_lin_vel_b[ids] * self.obs_scales.lin_vel,
                                contact.float(), self.compute_feet_height_target(ids)), dim=-1)
        return current_actor_obs, privileged

    def compute_feet_height_target(self, env_ids=None):
        ids = self._env_ids(env_ids)
        targets, masks = [], []
        for sensor in (self.left_feet_ray_caster, self.right_feet_ray_caster):
            if sensor is None:
                targets.append(torch.zeros(len(ids), 1, device=self.device))
                masks.append(torch.zeros(len(ids), 1, dtype=torch.bool, device=self.device))
                continue
            data = sensor.data
            sensor_z, hit_z = data.pos_w[ids, 2:3], data.ray_hits_w[ids, :, 2]
            valid_hit = torch.isfinite(hit_z)
            count = valid_hit.sum(-1, keepdim=True)
            valid = (count > 0) & torch.isfinite(sensor_z)
            ground_z = torch.where(valid_hit, hit_z, 0.0).sum(-1, keepdim=True) / count.clamp_min(1)
            targets.append(torch.where(valid, sensor_z - ground_z, 0.0).clamp(-1.0, 1.0))
            masks.append(valid)
        self.feet_height_valid[ids] = torch.cat(masks, dim=-1)
        return torch.cat(targets, dim=-1)

    def compute_terrain_height_target(self, env_ids=None):
        ids = self._env_ids(env_ids)
        data = self.height_scanner.data
        hits, sensor_z = data.ray_hits_w[ids], data.pos_w[ids, 2:3]
        distance = (hits - self.height_scanner.ray_starts_w[ids]).norm(dim=-1)
        height = (sensor_z - hits[..., 2] - self.cfg.normalization.height_scan_offset) * self.obs_scales.height_scan
        valid = torch.isfinite(hits).all(-1) & torch.isfinite(height)
        valid &= torch.isfinite(distance) & (distance <= self.height_scanner.cfg.max_distance)
        self.terrain_height_valid[ids] = valid
        return torch.where(valid, height, 0.0)

    def _update_proprio_history(self, env_ids):
        current, _ = self.compute_current_observations(env_ids)
        if self.cfg.noise.add_noise:
            uniform = self.selector_randomness.uniform(env_ids, (78,), "proprio_noise")
            current += (2.0 * uniform - 1.0) * self.noise_obs_vec
        self.actor_obs_buffer.append(current.clamp(-self.clip_obs, self.clip_obs), env_ids)

    def _capture_sensor_frames(self, env_ids):
        if not len(env_ids):
            return
        data = self.lp_sensor.data
        if (self.lp_sensor.scan_counts[env_ids] <= self.lp_last_scan_count[env_ids]).any():
            raise RuntimeError("The paired 10 Hz tick did not produce a new LP scan.")
        hits = data.ray_hits_w[env_ids]
        distance = (hits - self.lp_sensor.ray_starts_w[env_ids]).norm(dim=-1)
        valid = torch.isfinite(hits).all(-1)
        valid &= ~torch.isclose(hits, torch.full_like(hits, self.lp_sensor.cfg.max_distance)).all(-1)
        valid &= (distance >= getattr(self.lp_sensor.cfg, "min_distance", 0.0))
        valid &= distance <= self.lp_sensor.cfg.max_distance
        height = ((data.pos_w[env_ids, 2:3] - hits[..., 2] - self.cfg.normalization.height_scan_offset)
                  * self.obs_scales.height_scan).clamp(*self.lp_runtime_cfg.height_clip)
        valid &= torch.isfinite(height)
        height = torch.where(valid, height, self.lp_runtime_cfg.invalid_height).reshape(-1, 11, 17)
        lp_support = valid.float().reshape(-1, 11, 17)
        depth = self.camera.data.output["distance_to_image_plane"][env_ids].squeeze(-1)
        top, bottom, left, right = self.cfg.robot.depth_crop
        depth = depth[:, top:depth.shape[1] - bottom, left:depth.shape[2] - right]
        vp_support = (torch.isfinite(depth) & (depth > 0)).float()
        clock = self.sensor_control_ticks[env_ids].double() * self.low_level_dt
        sequence = self.sensor_sequence[env_ids] + 1
        lp = SensorFrame(height, lp_support, clock, clock, sequence)
        vp = SensorFrame(depth, vp_support, clock, clock, sequence)
        torso = self.robot.data.body_state_w[env_ids, self.torso_link_ids[0], :7]
        w, x, y, z = torso[:, 3:7].unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        pose = torch.cat((torso[:, :3], yaw[:, None]), dim=-1)
        self.sensor_observation = self.faults.update(lp, vp, pose, env_ids=env_ids)
        self.sensor_sequence[env_ids] = sequence
        self.sensor_capture_count[env_ids] += 1
        self.lp_last_scan_count[env_ids] = self.lp_sensor.scan_counts[env_ids]

    def compute_observations(self, env_ids=None):
        ids = self._env_ids(env_ids)
        sensor = self.sensor_observation
        if not sensor.initialized[ids].all():
            raise RuntimeError("Initialize real LP/VP sensor frames before building observations.")
        obs_cfg = self.cfg.observation
        proprio = self.actor_obs_buffer.buffer[ids].flatten(1)
        depth, lp = sensor.vp[ids], sensor.lp[ids]
        vp_support, lp_support = sensor.vp_valid_support[ids], sensor.lp_valid_support[ids]
        now = self.sensor_control_ticks[ids].double() * self.low_level_dt
        lp_age = (now - sensor.lp_source_stamp[ids]).clamp(0, obs_cfg.sensor_age_clip_s)
        vp_age = (now - sensor.vp_source_stamp[ids, -1]).clamp(0, obs_cfg.sensor_age_clip_s)
        mode = self.renet_estimator_mask[ids, 0]
        states = {
            "mode_vp": 1.0 - mode, "mode_lp": mode,
            "vp_old_valid_ratio": vp_support[:, 0].mean((1, 2)),
            "vp_new_valid_ratio": vp_support[:, 1].mean((1, 2)),
            "lp_valid_ratio": lp_support.mean((1, 2)),
            "vp_source_age": (vp_age / obs_cfg.sensor_age_clip_s).float(),
            "lp_source_age": (lp_age / obs_cfg.sensor_age_clip_s).float(),
            "time_since_switch": ((now - self.last_switch_time[ids]).clamp(0, obs_cfg.switch_time_clip_s)
                                  / obs_cfg.switch_time_clip_s).float(),
        }
        depth_input = torch.cat((depth, vp_support), dim=1) if obs_cfg.use_depth_valid_support else depth
        height_input = torch.stack((lp, lp_support), dim=1) if obs_cfg.use_height_valid_mask else lp[:, None]
        actor = torch.cat((proprio, depth_input.flatten(1), height_input.flatten(1),
                           torch.stack([states[name] for name in obs_cfg.state_names], dim=-1)), dim=-1)
        self.actor_obs[ids] = actor.clamp(-self.clip_obs, self.clip_obs)
        critic = [self.actor_obs[ids]]
        if obs_cfg.critic_use_privileged:
            _, privileged = self.compute_current_observations(ids)
            critic.append(privileged)
            if obs_cfg.critic_use_clean_height:
                critic.append(self.compute_terrain_height_target(ids))
        self.critic_obs[ids] = torch.cat(critic, dim=-1).clamp(-self.clip_obs, self.clip_obs)
        self.low_level_obs[ids] = torch.cat((proprio, lp.flatten(1), depth.flatten(1), mode[:, None]), dim=-1)
        self.extras.setdefault("observations", {})["critic"] = self.critic_obs
        return self.actor_obs, self.critic_obs

    @torch.no_grad()
    def reset(self, env_ids=None):
        ids = self._env_ids(env_ids)
        if not len(ids):
            return self.get_observations()
        # ---------------------------------------------------------
        # 固定地形：仅重置机器人，不改变 terrain_levels 或 env_origins。
        # ---------------------------------------------------------
        self.extras.setdefault("log", {}).update(self.reward_manager.reset(ids))
        self.scene.reset(ids)
        if "reset" in self.event_manager.available_modes:
            # 每行独立的重置随机流；其他行 reset 不改变本行未来的抽样。
            for env_id in ids.tolist():
                row = torch.tensor([env_id], device=self.device, dtype=torch.long)
                with self.selector_randomness.single_env_seed(env_id, "episode_reset"):
                    self.event_manager.apply(
                        mode="reset", env_ids=row, dt=self.low_level_dt,
                        global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation)
        self.extras["log"].update(self.command_generator.reset(ids))
        self.actor_obs_buffer.reset(ids)
        self.action_buffer.reset(ids)
        self.action[ids] = 0.0
        self.processed_actions[ids] = self.robot.data.default_joint_pos[ids]
        self.episode_length_buf[ids] = 0
        self.low_level_episode_length_buf[ids] = 0
        self.sensor_control_ticks[ids] = 0
        self.sensor_sequence[ids] = -1
        self.lp_last_scan_count[ids] = -1
        self.renet_estimator_mask[ids] = self.cfg.selector.initial_mode
        self.last_switch_time[ids] = 0.0
        self.selector_switch_event[ids] = False
        self.reset_buf[ids] = self.time_out_buf[ids] = False
        self.contact_failure_buf[ids] = self.tilt_failure_buf[ids] = False
        self.active_mask[ids] = True
        self.faults.reset(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        # SensorBase.reset 将对应行置为过期；新位姿首帧不需要额外物理步。
        self._capture_sensor_frames(ids)
        self._update_proprio_history(ids)
        self.compute_observations(ids)
        return self.get_observations()

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor) or actions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Selector actions must be integer VP=0 / LP=1 mode tensors.")
        if actions.shape not in ((self.num_envs,), (self.num_envs, 1)):
            raise ValueError("Selector actions must have shape (num_envs,) or (num_envs, 1).")
        modes = actions.to(self.device).reshape(-1)
        if ((modes < 0) | (modes > 1)).any():
            raise ValueError("Selector mode must be VP=0 or LP=1.")
        switched = modes != self.renet_estimator_mask[:, 0]
        self.renet_estimator_mask[:, 0] = modes
        self.last_switch_time[switched] = self.sensor_control_ticks[switched].double() * self.low_level_dt
        self.active_mask.fill_(True)
        self.extras = {"log": {}, "observations": {}}
        done = torch.zeros_like(self.reset_buf)
        time_outs = torch.zeros_like(done)
        contact = torch.zeros_like(done)
        tilt = torch.zeros_like(done)
        reward = torch.zeros(self.num_envs, device=self.device)
        counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terminal_actor = torch.zeros_like(self.actor_obs)
        terminal_critic = torch.zeros_like(self.critic_obs)
        self.episode_length_buf += 1

        # ---------------------------------------------------------
        # 一个高层决策固定推进十个底层步；首次 done 后该行停止有效累计。
        # ---------------------------------------------------------
        for index in range(self.cfg.selector.low_level_steps_per_decision):
            start_active = self.active_mask.clone()
            self.selector_switch_event.zero_()
            if index == 0:
                self.selector_switch_event.copy_(switched)
            step_reward = self._step_low_level(start_active)
            reward += step_reward
            counts += start_active.long()
            first_done = start_active & self.reset_buf
            if first_done.any():
                terminal_actor[first_done] = self.actor_obs[first_done]
                terminal_critic[first_done] = self.critic_obs[first_done]
                time_outs[first_done] = self.time_out_buf[first_done]
                contact[first_done] = self.contact_failure_buf[first_done]
                tilt[first_done] = self.tilt_failure_buf[first_done]
                done |= first_done
                self.active_mask[first_done] = False

        terminal_ids = done.nonzero().flatten()
        terminal_actor = terminal_actor[terminal_ids].clone()
        terminal_critic = terminal_critic[terminal_ids].clone()
        self.reset(terminal_ids)
        self.reset_buf.copy_(done)
        self.time_out_buf.copy_(time_outs)
        self.selector_switch_event.zero_()
        self.extras.update(
            time_outs=time_outs, terminal_env_ids=terminal_ids,
            terminal_actor_observations=terminal_actor, terminal_critic_observations=terminal_critic,
            low_level_steps=counts,
        )
        self.extras["log"].update({
            "Selector/switches": switched.sum(), "Selector/lp_fraction": modes.float().mean(),
            "Termination/contact": contact.sum(), "Termination/tilt": tilt.sum(),
            "Termination/time_out": time_outs.sum(), "Selector/effective_low_level_steps": counts.float().mean(),
            "Sensors/captures_total": self.sensor_capture_count.sum(),
        })
        observed = self.faults.actual_counts.sum(dim=0).float()
        total_frames = observed.sum().clamp_min(1.0)
        for index, name in enumerate(("normal", "lp_only", "vp_only", "dual")):
            self.extras["log"][f"Fault/observed_{name}_fraction"] = observed[index] / total_frames
        self.extras["observations"]["critic"] = self.critic_obs
        return self.actor_obs, reward, done, self.extras

    def _step_low_level(self, active_mask):
        ids = active_mask.nonzero().flatten()
        if len(ids):
            # mask 已由高层接管；不调用旧模式抽样或任何动作混合逻辑。
            self.low_level_obs[ids, -1] = self.renet_estimator_mask[ids, 0]
            output = self.frozen_policy.evaluate(self.low_level_obs[ids])
            actions = torch.where(self.renet_estimator_mask[ids] > 0.5, output.lp_action, output.vp_action)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Frozen RENet produced non-finite selected actions.")
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
            self.action[ids] = self.action_buffer.compute(actions, ids)
            self.processed_actions[ids] = self.action[ids] * self.action_scale + self.robot.data.default_joint_pos[ids]

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(self.processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            _ = self.contact_sensor.data
        if not self.headless:
            self.sim.render()
        self.low_level_episode_length_buf[ids] += 1
        self.sensor_control_ticks[ids] += 1
        self.reset_buf, self.time_out_buf = self.check_reset()
        self.reset_buf &= active_mask
        self.time_out_buf &= active_mask
        reward = self.reward_manager.compute(self.low_level_dt, active_mask)
        # 奖励先使用产生动作的命令；更新后的命令属于下一观测。
        self.command_generator.compute(self.low_level_dt, ids)
        fresh = ids[self.sensor_control_ticks[ids] % self.sensor_interval == 0]
        self._capture_sensor_frames(fresh)
        if len(ids):
            self._update_proprio_history(ids)
            self.compute_observations(ids)
        return reward

    def check_reset(self):
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.termination_contact_cfg.body_ids]
        self.contact_failure_buf = (torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > 1.0).any(dim=1)
        # 真实重力投影，不使用 Actor 中加噪后的姿态；clamp 防止 acos 舍入越界。
        tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        self.tilt_failure_buf = tilt > self.cfg.robot.limit_angle
        failure = self.contact_failure_buf | self.tilt_failure_buf
        timeout = (self.low_level_episode_length_buf >= self.low_level_max_episode_length) & ~failure
        return failure | timeout, timeout

    def get_observations(self):
        self.extras.setdefault("observations", {})["critic"] = self.critic_obs
        return self.actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """兼容现有专家数据接口；不用于高层奖励或训练。"""
        joint_pos = self.robot.data.joint_pos[:, self.amp_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.amp_joint_ids]
        root_position = self.robot.data.root_state_w[:, :3]
        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])
        link_ids = (*self.wrist_link_ids, *self.ankle_link_ids)
        local_positions = [
            quat_apply(root_quat_inv, self.robot.data.body_state_w[:, body_id, :3] - root_position)
            for body_id in link_ids
        ]
        amp_obs = torch.cat([joint_pos, joint_vel, *local_positions], dim=-1)
        expected_dim = 2 * len(self.amp_joint_ids) + 12
        if amp_obs.shape[1] != expected_dim:
            raise RuntimeError(f"AMP obs 应为{expected_dim}维，实际为 {amp_obs.shape}")
        return amp_obs

    def visualize_motion(self, time):
        """懒加载展示数据，兼容 23/29 关节动作；不推进训练或感知缓存。"""
        from scipy.spatial.transform import Rotation

        if self.amp_loader_display is None:
            from rsl_rl.utils import AMPLoaderDisplay

            self.amp_loader_display = AMPLoaderDisplay(
                motion_files=self.cfg.amp_motion_files_display, device=self.device,
                time_between_frames=self.physics_dt,
            )
            self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        root_pose_size, root_vel_size = 6, 6
        motion_frame_size = visual_motion_frame.shape[0]
        motion_joint_size = (motion_frame_size - root_pose_size - root_vel_size) // 2
        if motion_frame_size != root_pose_size + root_vel_size + 2 * motion_joint_size:
            raise ValueError(f"Unsupported motion frame size {motion_frame_size}")
        if motion_joint_size == 29:
            motion_group_sizes = (6, 6, 3, 7, 7)
        elif motion_joint_size == 23:
            motion_group_sizes = (6, 6, 1, 5, 5)
        else:
            raise ValueError(f"Unsupported G1 motion joint size {motion_joint_size}. Expected 29 or 23.")

        dof_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        dof_vel = torch.zeros_like(dof_pos)
        joint_groups = (
            ("left_leg", self.left_leg_ids), ("right_leg", self.right_leg_ids),
            ("waist", self.waist_ids), ("left_arm", self.left_arm_ids), ("right_arm", self.right_arm_ids),
        )
        pos_src_idx = root_pose_size
        vel_src_idx = root_pose_size + motion_joint_size + root_vel_size
        for (_, joint_ids), motion_group_size in zip(joint_groups, motion_group_sizes):
            copy_size = min(len(joint_ids), motion_group_size)
            dof_pos[:, joint_ids[:copy_size]] = visual_motion_frame[pos_src_idx:pos_src_idx + copy_size]
            dof_vel[:, joint_ids[:copy_size]] = visual_motion_frame[vel_src_idx:vel_src_idx + copy_size]
            pos_src_idx += motion_group_size
            vel_src_idx += motion_group_size
        self.robot.write_joint_position_to_sim(dof_pos)
        # 展示只设置姿态，保留原接口的零速度写入，避免物理积分引起抖动。
        self.robot.write_joint_velocity_to_sim(torch.zeros_like(dof_vel))

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += float(getattr(self, "visualization_root_height_offset", 0.1))
        quat_xyzw = Rotation.from_euler("XYZ", visual_motion_frame[3:6].cpu().numpy()).as_quat()
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=self.device, dtype=torch.float32,
        )
        root_state = torch.zeros(self.num_envs, 13, device=self.device)
        root_state[:, :3] = root_pos
        root_state[:, 3:7] = quat_wxyz
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        if not self.headless:
            self.sim.render()

        root_quat_inv = quat_conjugate(self.robot.data.root_state_w[:, 3:7])
        local_positions = [
            quat_apply(root_quat_inv, self.robot.data.body_state_w[:, body_id, :3]
                       - self.robot.data.root_state_w[:, :3])
            for body_id in (*self.wrist_link_ids, *self.ankle_link_ids)
        ]
        joint_positions, joint_velocities = [], []
        for name, joint_ids in joint_groups:
            position, velocity = dof_pos[:, joint_ids], dof_vel[:, joint_ids]
            setattr(self, f"{name}_dof_pos", position)
            setattr(self, f"{name}_dof_vel", velocity)
            joint_positions.append(position)
            joint_velocities.append(velocity)
        return torch.cat([*joint_positions, *joint_velocities, *local_positions], dim=-1)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
