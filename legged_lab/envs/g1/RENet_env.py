import math

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
import torchvision.transforms as T

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.g1.RENet_cfg import G1RENETENVCFG
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.utils.env_utils.scene import SceneCfg

from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay

class G1RENetEnv(VecEnv):
    
    def __init__(
        self,
        cfg: (
            G1RENETENVCFG
        ),
        headless,
    ):
        self.cfg: (
            G1RENETENVCFG
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
        '''
        核心行：创建 Isaac Sim 的仿真上下文实例，是整个仿真的 “总控台”—— 你附带的SimulationContext类的__init__会执行以下关键操作：
        验证仿真配置（cfg.validate()）；
        配置 PhysX 物理引擎（碰撞迭代次数、GPU 设备、CCD 连续碰撞检测）；
        配置渲染模式（无头模式禁用 GUI，有头模式启用视口）；
        初始化仿真的时间步、设备、物理场景；
        绑定默认物理材质（如上面配置的摩擦系数）；
        最终，self.sim成为控制仿真启动、暂停、步进的核心对象（后续会调用self.sim.reset()启动仿真，self.sim.step()步进物理）。
        '''
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        '''
        cfg.scene（BaseSceneCfg）只包含 “静态的场景参数”，但缺少仿真步长（物理 / 控制）这一动态参数；
        而SceneCfg（你导入的legged_lab.utils.env_utils.scene.SceneCfg）是对 Isaac Lab 原生配置的封装适配，
        专门用于整合 “静态场景参数 + 动态步长参数”，输出一个能被InteractiveScene直接使用的完整配置。
        '''
        # scene_cfg.camera = cfg.scene.camera
        # scene_cfg.left_feet_ray_caster=cfg.scene.left_feet_ray_caster
        # scene_cfg.right_feet_ray_caster=cfg.scene.right_feet_ray_caster

        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()  # 重置一次
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        if "camera" in self.scene.sensors:
            self.camera: GroupedRayCasterCamera = self.scene.sensors["camera"]
        else:
            self.camera=None

        self.left_feet_ray_caster = (
            self.scene.sensors["left_feet_ray_caster"] if "left_feet_ray_caster" in self.scene.sensors else None
        )
        self.right_feet_ray_caster = (
            self.scene.sensors["right_feet_ray_caster"] if "right_feet_ray_caster" in self.scene.sensors else None
        )

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
        # 设置了 速度命令生成器 奖励管理器
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        
        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 设置了 事件管理器
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        # 数据的总的帧数
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]
        print("训练时的真实关节顺序:", self.robot.joint_names)
    def visualize_motion(self, time):
        # 返回的是当前时间下的 数据中对应的帧
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        root_pose_size = 6
        root_vel_size = 6
        motion_frame_size = visual_motion_frame.shape[0]
        motion_joint_size = (motion_frame_size - root_pose_size - root_vel_size) // 2
        if motion_frame_size != root_pose_size + root_vel_size + 2 * motion_joint_size:
            raise ValueError(
                f"Unsupported motion frame size {motion_frame_size}. Expected root(6) + joints + root_vel(6) + joint_vel."
            )

        joint_pos_start_idx = root_pose_size
        root_vel_start_idx = joint_pos_start_idx + motion_joint_size
        joint_vel_start_idx = root_vel_start_idx + root_vel_size

        if motion_joint_size == 29:
            motion_group_sizes = {
                "left_leg": 6,
                "right_leg": 6,
                "waist": 3,
                "left_arm": 7,
                "right_arm": 7,
            }
        elif motion_joint_size == 23:
            motion_group_sizes = {
                "left_leg": 6,
                "right_leg": 6,
                "waist": 1,
                "left_arm": 5,
                "right_arm": 5,
            }
        else:
            raise ValueError(f"Unsupported G1 motion joint size {motion_joint_size}. Expected 29 or 23.")
 
        # 关节位置
        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        # 关节速度
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        joint_groups = (
            (self.left_leg_ids, motion_group_sizes["left_leg"]),
            (self.right_leg_ids, motion_group_sizes["right_leg"]),
            (self.waist_ids, motion_group_sizes["waist"]),
            (self.left_arm_ids, motion_group_sizes["left_arm"]),
            (self.right_arm_ids, motion_group_sizes["right_arm"]),
        )

        pos_src_idx = joint_pos_start_idx
        vel_src_idx = joint_vel_start_idx
        for joint_ids, motion_group_size in joint_groups:
            copy_size = min(len(joint_ids), motion_group_size)
            if copy_size > 0:
                dof_pos[:, joint_ids[:copy_size]] = visual_motion_frame[pos_src_idx : pos_src_idx + copy_size]
                dof_vel[:, joint_ids[:copy_size]] = visual_motion_frame[vel_src_idx : vel_src_idx + copy_size]
            pos_src_idx += motion_group_size
            vel_src_idx += motion_group_size
 
        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(torch.zeros_like(dof_vel))

                # 生成环境ID张量：维度 [num_envs]，值为0,1,...,num_envs-1，用于批量更新所有并行环境的机器人状态
        env_ids = torch.arange(self.num_envs, device=device)
 
        root_pos = visual_motion_frame[:3].clone()
        # 抬高根节点z轴（+0.3米）：避免机器人初始位置过低，与地面碰撞导致物理仿真异常
        root_pos[2] += 0.1
 
        # 提取根节点欧拉角（AMP帧3:6维：roll/pitch/yaw），转numpy用于四元数转换
        euler = visual_motion_frame[3:6].cpu().numpy()
        # 欧拉角转四元数：scipy的Rotation.from_euler默认输出 [x,y,z,w] 格式（XYZW）
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        # 转换四元数格式为 [w,x,y,z]（WXYZ）：Isaac Sim/PhysX物理引擎的标准四元数格式
        # 并转换为torch张量，与仿真设备一致
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )
        # 可视化时只摆姿态，避免速度尖峰被物理积分放大成抖动。
        lin_vel = torch.zeros(3, dtype=torch.float32, device=device)
        # 设置根节点的角速度 初始化一个 3 维全 0 张量，作为根节点的角速度 ang_vel
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))
 
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        if not self.headless:
            self.sim.render()

        # 左手的位置
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_link_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        # 右手的位置
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.wrist_link_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        # 
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)

        # 左脚的位置
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.ankle_link_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        # 右脚的位置
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.ankle_link_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids] 
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids] 
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        self.waist_dof_pos = dof_pos[:,self.waist_ids]
        self.waist_dof_vel = dof_vel[:,self.waist_ids]

        # 专家数据 标签数据
        return torch.cat(
            (
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.waist_dof_pos,
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                self.waist_dof_vel,
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos
            ),
            dim=-1,
        )

    def init_buffers(self):
        self.extras = {}

        self.recovery_state_machine_enabled = bool(self.cfg.recovery.enable)
        self.baseline_max_episode_length_s = float(self.cfg.scene.max_episode_length_s)
        recovery_positive_params = {
            "max_duration_s": self.cfg.recovery.max_duration_s,
            "absolute_episode_timeout_s": self.cfg.recovery.absolute_episode_timeout_s,
            "ready_hold_s": self.cfg.recovery.ready_hold_s,
            "upright_threshold": self.cfg.recovery.upright_threshold,
            "max_ang_vel": self.cfg.recovery.max_ang_vel,
            "max_vertical_vel": self.cfg.recovery.max_vertical_vel,
            "torso_force_threshold": self.cfg.recovery.torso_force_threshold,
            "foot_force_threshold": self.cfg.recovery.foot_force_threshold,
        }
        for name, value in recovery_positive_params.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"recovery.{name} must be positive and finite, got {value}.")
        if not math.isfinite(self.baseline_max_episode_length_s) or self.baseline_max_episode_length_s <= 0.0:
            raise ValueError("scene.max_episode_length_s must be positive and finite.")
        if not math.isfinite(float(self.step_dt)) or float(self.step_dt) <= 0.0:
            raise ValueError("The control step_dt must be positive and finite.")
        if float(self.cfg.recovery.upright_threshold) > 1.0:
            raise ValueError("recovery.upright_threshold cannot exceed 1.0.")
        if not 0.0 < float(self.cfg.recovery.height_ratio) <= 1.0:
            raise ValueError("recovery.height_ratio must be in (0, 1].")

        self.max_episode_length_s = (
            float(self.cfg.recovery.absolute_episode_timeout_s)
            if self.recovery_state_machine_enabled
            else self.baseline_max_episode_length_s
        )
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.step_dt)
        self.recovery_max_steps = math.ceil(float(self.cfg.recovery.max_duration_s) / self.step_dt)
        self.recovery_ready_hold_steps = math.ceil(float(self.cfg.recovery.ready_hold_s) / self.step_dt)

        # G1_23CFG's default root z is the nominal base height relative to the
        # terrain origin. Unlike a guessed constant, it follows robot config changes.
        nominal_root_pos = getattr(self.cfg.scene.robot.init_state, "pos", None)
        if nominal_root_pos is None or len(nominal_root_pos) < 3 or not math.isfinite(float(nominal_root_pos[2])):
            raise ValueError("Recovery state machine requires a finite default robot root height.")
        self.nominal_base_height = float(nominal_root_pos[2])
        if self.nominal_base_height <= 0.0:
            raise ValueError(f"Nominal robot root height must be positive, got {self.nominal_base_height}.")
        self.recovery_ready_height_threshold = float(self.cfg.recovery.height_ratio) * self.nominal_base_height

        self.num_actions = self.robot.data.default_joint_pos.shape[1]  # 获取机器人的动作维度（关节数量），这是策略网络输出层的维度。
        self.clip_actions = self.cfg.normalization.clip_actions   # 读取 “动作裁剪阈值”，限制策略网络输出的动作范围，避免动作过大导致机器人关节损坏 / 物理仿真崩溃。
        self.clip_obs = self.cfg.normalization.clip_observations  # 读取 “观测裁剪阈值”，限制机器人观测数据的范围，避免异常值（如传感器故障、物理抖动）导致网络训练不稳定

        self.action_scale = self.cfg.robot.action_scale  # 把策略网络输出的 “归一化动作” 映射到机器人实际能执行的控制范围；
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        
        '''
        通过SceneEntityCfg（场景实体配置类）精准定位场景中的关键实体 / 刚体（机器人本体、触发终止的刚体、脚部刚体）
        调用resolve()方法将 “抽象的名称配置” 转换为 “场景中可直接访问的索引 / 引用”
        SceneEntityCfg(name="robot") resolve 可以定位到整个机器人 也可以定位到这个机器人中的某些刚体
        Self.scene['robot']只能定位到整个机器人 而不能定位到某些刚体
        '''
        self.robot_cfg = SceneEntityCfg(name="robot")  # 此时self.robot_cfg 还只是创建一个定位规则 
        self.robot_cfg.resolve(self.scene) # 这一步是从场景中 定位机器人刚体
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)  # 从场景中定位到 在walk_CFG.py的配置的接触终止的关节
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene) # 从场景中定位到创建「脚部接触传感器的定位配置」，指定 “哪些刚体是机器人的脚部”，用于检测脚部是否落地  # 用于判断脚是否接触地面了
        if len(self.feet_cfg.body_ids) != 2:
            raise RuntimeError(
                "Recovery support-height routing requires exactly two feet, got "
                f"{self.feet_cfg.body_names}."
            )
        resolved_foot_names = [self.contact_sensor.body_names[index] for index in self.feet_cfg.body_ids]
        left_matches = [index for index, name in enumerate(resolved_foot_names) if "left" in name]
        right_matches = [index for index, name in enumerate(resolved_foot_names) if "right" in name]
        if len(left_matches) != 1 or len(right_matches) != 1:
            raise RuntimeError(f"Could not identify left/right foot contact bodies: {resolved_foot_names}.")
        self.left_foot_contact_index = left_matches[0]
        self.right_foot_contact_index = right_matches[0]
        
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            self.noisy=self.cfg.noise.noise_scales
        # 创建「episode 步数缓冲区」，记录每个环境当前 episode 已经运行的步数
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0 # 创建「全局仿真步数计数器」，记录整个仿真运行的总步数（所有环境共享），而非单个环境的 episode 步数
        # 创建「超时标记缓冲区」，标记哪些环境因达到最大 episode 长度需要重置
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Recovery state is per environment. Mode switches do not set PPO done;
        # only recovery/absolute timeouts do so in the enabled state machine.
        self.recovery_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.recovery_ready_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.recovery_trigger_armed = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.enter_recovery_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.exit_recovery_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_failed_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_ready_now_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_upright_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_height_ok_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_foot_support_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.recovery_torso_clear_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._recovery_diagnostics = self._empty_recovery_diagnostics()

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 为每一个环境创建一个mask
        self.renet_estimator_mask = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.renet_training_iteration = 0
        self.renet_hard_terrain_type_ids = self.resolve_renet_hard_terrain_type_ids()
    
        #
        if self.camera is not None:
            self.depth_history_frames = self.cfg.robot.depth_history_frames
            self.camera_height = self.cfg.scene.camera.camera.pattern_cfg.height-self.cfg.robot.depth_crop[0]-self.cfg.robot.depth_crop[1]
            self.camera_width = self.cfg.scene.camera.camera.pattern_cfg.width-self.cfg.robot.depth_crop[2]-self.cfg.robot.depth_crop[3]
            # 缓冲区形状: [num_envs, history_frames, height, width]
            self.depth_buffer = torch.zeros(
                (self.num_envs, self.depth_history_frames, self.camera_height, self.camera_width),
                device=self.device, dtype=torch.float
            )
        #
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

        self.amp_joint_ids, self.amp_joint_names = (
            self.robot.find_joints(
                amp_joint_names,
                preserve_order=True,
            )
        )

        if len(self.amp_joint_ids) != 19:
            raise RuntimeError(
                "AMP应找到19个非脚踝关节，"
                f"实际找到 {len(self.amp_joint_ids)}："
                f"{self.amp_joint_names}"
            )
        # 脚连杆的索引
        self.ankle_link_ids,_ = self.robot.find_bodies(
            name_keys=['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True,
        )
        # 手腕处的连杆
        self.wrist_link_ids,_ =self.robot.find_bodies(
            name_keys=['left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand'],
            preserve_order=True,
        )
        # 左腿关节索引
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                    'left_hip_pitch_joint',
                    'left_hip_roll_joint',
                    'left_hip_yaw_joint',
                    'left_knee_joint',
                    'left_ankle_pitch_joint',
                    'left_ankle_roll_joint'  
            ],
            preserve_order=True,
        )
        # 右腿关节索引
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                    'right_hip_pitch_joint',
                    'right_hip_roll_joint',
                    'right_hip_yaw_joint',
                    'right_knee_joint',
                    'right_ankle_pitch_joint', 
                    'right_ankle_roll_joint'
            ],
            preserve_order=True,
        )
        # 左胳膊关节索引
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                    'left_shoulder_pitch_joint', 
                    'left_shoulder_roll_joint',
                    'left_shoulder_yaw_joint',
                    'left_elbow_joint',
                    'left_wrist_roll_joint',
            ],
            preserve_order=True,
        )
        # 右胳膊关节索引
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                    'right_shoulder_pitch_joint',
                    'right_shoulder_roll_joint',
                    'right_shoulder_yaw_joint',
                    'right_elbow_joint',
                    'right_wrist_roll_joint',
            ],
            preserve_order=True,
        )
        # 腰部关节索引
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
            ],
            preserve_order=True,
        )
        # 脚的平均接触力
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        # 统计每个并行仿真环境下机器人每只脚的平均地面滑动速度
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.obs_noisy_vec_and_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b    # 获取根节点自身坐标系下的角速度
        projected_gravity = robot.data.projected_gravity_b  # 获取机器人自身坐标系的重力投影
        command = self.command_generator.command # 速度命令
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos  # 关节的位相对默认角度的位置
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel  # 关节速度相对于默认关节速度的速度
        '''
        self.action_buffer：类实例中保存的动作缓冲区实例，用于缓存智能体之前输出的动作（强化学习中，智能体通常需要参考近期的动作来做决策，保持动作的连续性）。
        _circular_buffer：循环缓冲区（环形缓冲区），一种高效的缓存数据结构，当缓冲区满时，新数据会覆盖最旧的数据，无需频繁移动数据。
        buffer[:, -1, :]：对循环缓冲区的张量进行索引：
        第一个维度:：取所有并行环境（num_envs）。
        第二个维度-1：取最后一个（最新的）时刻的动作（循环缓冲区保存了近期多个时刻的动作，-1表示当前时刻的前一个动作，即上一步输出的动作）。
        第三个维度:：取动作的所有维度（形状为(num_envs, action_dim)，action_dim是动作空间维度，通常和关节数一致）。
        整体作用：提取智能体上一步输出的最新动作，作为观测的一部分，让智能体知道自己之前做了什么动作，保持决策的连续性。
        '''
        action = self.action_buffer._circular_buffer.buffer[:, -1, :] # 前一帧的动作
        '''
        当前的ACTOR观测
            角速度 
            投影重力 
            速度命令 
            关节位置 
            关节速度 
            上一步的动作 
            步态的 
            步态相位比率，描述当前步态周期内的进度比例
        当前的CRITIC:
            当前的观测   根节点的自身坐标系下的线速度(3维)   脚接触力(4维)
        '''
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 23
                joint_vel * self.obs_scales.joint_vel,  # 23
                action * self.obs_scales.actions,  # 23
            ],
            dim=-1,
        )
        
        root_lin_vel = robot.data.root_lin_vel_b # 根节点的线速度
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        feet_height = self.compute_feet_height_target()
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_height],
            dim=-1,
        )
        return current_actor_obs, current_critic_obs

    def compute_feet_height_target(self):
        if self.left_feet_ray_caster is None or self.right_feet_ray_caster is None:
            return torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        targets = []
        for ray_sensor in (self.left_feet_ray_caster, self.right_feet_ray_caster):
            sensor_z = ray_sensor.data.pos_w[:, 2:3]
            hit_z = ray_sensor.data.ray_hits_w[..., 2]
            valid_hit = torch.isfinite(hit_z)
            safe_hit_z = torch.where(valid_hit, hit_z, sensor_z.expand_as(hit_z))
            ground_z = safe_hit_z.mean(dim=-1, keepdim=True)
            targets.append(torch.clamp(sensor_z - ground_z, min=-1.0, max=1.0))
        return torch.cat(targets, dim=-1)

    def set_training_iteration(self, iteration: int):
        self.renet_training_iteration = int(iteration)

    def resolve_renet_hard_terrain_type_ids(self):
        terrain_generator = getattr(self.cfg.scene, "terrain_generator", None) # 拿到ROUGH_TERRAINS_CFG
        hard_names = set(getattr(self.cfg.renet, "force_vp_terrain_names", [])) # 拿到困难地形的名字
        if terrain_generator is None or not hard_names or not getattr(terrain_generator, "curriculum", False):
            return torch.empty(0, dtype=torch.long, device=self.device)

        sub_terrains = getattr(terrain_generator, "sub_terrains", {})  # 拿到所有的地形
        if not sub_terrains:
            return torch.empty(0, dtype=torch.long, device=self.device)

        sub_terrain_names = list(sub_terrains.keys()) # 地形的名字
        proportions = np.array([sub_terrains[name].proportion for name in sub_terrain_names], dtype=np.float64) # 拿到每一个地形的比例
        if proportions.sum() <= 0.0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        proportions /= proportions.sum() # 进行比例归一化
        cumulative = np.cumsum(proportions)

        hard_type_ids = []
        num_cols = int(getattr(terrain_generator, "num_cols", 0))
        for col_idx in range(num_cols):
            matches = np.where(col_idx / num_cols + 0.001 < cumulative)[0]
            if len(matches) == 0:
                continue
            sub_index = int(np.min(matches))
            if sub_terrain_names[sub_index] in hard_names:
                hard_type_ids.append(col_idx)

        return torch.tensor(hard_type_ids, dtype=torch.long, device=self.device)
        # 
    # 当前这 4096 个并行环境里，哪些环境现在正处在这些困难地形上，因此必须强制使用 VP。
    def get_renet_force_vp_envs(self):
        force_vp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terrain = getattr(self.scene, "terrain", None)
        if terrain is None:
            return force_vp

        if self.renet_hard_terrain_type_ids.numel() > 0 and hasattr(terrain, "terrain_types"):
            terrain_types = terrain.terrain_types.to(self.device)
            force_vp |= (terrain_types.unsqueeze(1) == self.renet_hard_terrain_type_ids.unsqueeze(0)).any(dim=1)

        level_threshold = int(getattr(self.cfg.renet, "force_vp_terrain_level", -1))
        if level_threshold >= 0 and hasattr(terrain, "terrain_levels"):
            force_vp |= terrain.terrain_levels.to(self.device) >= level_threshold

        return force_vp

    def sample_renet_estimator_mask(self):
        mask_mode = getattr(self.cfg.renet, "mask_mode", "alternate")
        if mask_mode == "alternate":
            interval = max(1, int(getattr(self.cfg.renet, "alternate_interval_iters", 20)))
            use_op = (self.renet_training_iteration // interval) % 2 == 0
            self.renet_estimator_mask[:] = 1.0 if use_op else 0.0
        elif mask_mode == "random":
            op_probability = float(getattr(self.cfg.renet, "op_probability", 0.5))
            op_probability = max(0.0, min(1.0, op_probability))
            self.renet_estimator_mask = (
                torch.rand(self.num_envs, 1, device=self.device) < op_probability
            ).float()
        elif mask_mode == "op":
            self.renet_estimator_mask[:] = 1.0
        elif mask_mode == "vp":
            self.renet_estimator_mask[:] = 0.0
        else:
            raise ValueError(f"Unsupported RENet mask_mode: {mask_mode}")

        force_vp_envs = self.get_renet_force_vp_envs()
        self.renet_estimator_mask[force_vp_envs] = 0.0 # 强制这些环境使用VP
        return self.renet_estimator_mask

    def append_renet_training_inputs(self, actor_obs):
        if self.camera is not None:
            depth_obs = self.get_deepcamera_history()
            flat_depth = depth_obs.view(self.num_envs, -1)
            self.extras["observations"]["depth"] = flat_depth
            actor_obs = torch.cat([actor_obs, flat_depth], dim=-1)

        estimator_mask = self.sample_renet_estimator_mask()
        self.extras["observations"]["renet_mask"] = estimator_mask
        actor_obs = torch.cat([actor_obs, estimator_mask], dim=-1)
        return actor_obs
    
    def obs_noisy_vec_and_buffer(self):
        if self.add_noise:
        
            current_actor_obs, _ = self.compute_current_observations()
        
            noise_obs_vec = torch.zeros_like(current_actor_obs[0]) # 定义一个和current_actor_obs维度一样的向量
            noise_obs_vec[0:3] = self.obs_scales.ang_vel*self.noisy.ang_vel
            noise_obs_vec[3:6] = self.obs_scales.projected_gravity*self.noisy.projected_gravity
            noise_obs_vec[6:9] = 0
            noise_obs_vec[9:9+self.num_actions]=self.obs_scales.joint_pos*self.noisy.joint_pos
            noise_obs_vec[9 + self.num_actions : 9 + self.num_actions * 2] = self.obs_scales.joint_vel*self.noisy.joint_vel 
            noise_obs_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_obs_vec=noise_obs_vec
        
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
            height_scan_noise_vec = torch.zeros_like(height_scan[0])
            if self.add_noise:
                height_scan_noise_vec[:] = self.noisy.height_scan * self.obs_scales.height_scan
            self.height_scan_noise_vec = height_scan_noise_vec
        
        # 定义actor缓存区 存入单帧观测数据后，self.actor_obs_buffer 的核心存储张量（buffer）是一个 3 维张量，维度为 [num_envs, max_len, single_actor_obs_dim]
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        # 定义critic缓存区
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
    
    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        '''
        torch.rand_like(current_actor_obs)：生成一个和current_actor_obs形状完全相同的张量，元素值服从[0, 1]之间的均匀分布。
        `2 * torch.rand_like(...) - 1：将均匀分布从[0, 1]映射到[-1, 1]。
        * self.noise_scale_vec：乘以噪声缩放系数（self.noise_scale_vec是与观测特征维度匹配的张量），控制噪声的大小，避免噪声过大覆盖有效观测。
        ''' 
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_obs_vec
        
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        self.actor_obs=self.actor_obs_buffer.buffer.reshape(self.num_envs,-1)
        self.critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            self.critic_obs = torch.cat([self.critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            
        self.actor_obs = torch.clip(self.actor_obs, -self.clip_obs, self.clip_obs)
        self.critic_obs = torch.clip(self.critic_obs, -self.clip_obs, self.clip_obs)

        return self.actor_obs, self.critic_obs

    @staticmethod
    def _finite_ray_median(hit_z: torch.Tensor):
        """Return per-environment medians over finite ray hits without CPU copies."""
        if hit_z.ndim < 2:
            raise ValueError(f"Ray hit z must have shape [num_envs, ...], got {tuple(hit_z.shape)}.")
        flat_hit_z = hit_z.flatten(start_dim=1)
        if flat_hit_z.shape[1] == 0:
            raise ValueError("Ray hit z must contain at least one ray per environment.")
        finite = torch.isfinite(flat_hit_z)
        valid_count = finite.sum(dim=1)
        sorted_hit_z = torch.sort(
            torch.where(finite, flat_hit_z, torch.full_like(flat_hit_z, torch.inf)),
            dim=1,
        ).values
        lower_idx = torch.clamp((valid_count - 1) // 2, min=0)
        upper_idx = torch.clamp(valid_count // 2, max=flat_hit_z.shape[1] - 1)
        lower = sorted_hit_z.gather(1, lower_idx.unsqueeze(1)).squeeze(1)
        upper = sorted_hit_z.gather(1, upper_idx.unsqueeze(1)).squeeze(1)
        valid = valid_count > 0
        median = torch.where(valid, 0.5 * (lower + upper), torch.zeros_like(lower))
        return median, valid

    @staticmethod
    def _route_support_height(
        left_ground_z: torch.Tensor,
        left_valid: torch.Tensor,
        right_ground_z: torch.Tensor,
        right_valid: torch.Tensor,
        left_support: torch.Tensor,
        right_support: torch.Tensor,
    ):
        """Select terrain height only from feet with contact and valid ray hits."""
        left_source = left_support & left_valid
        right_source = right_support & right_valid
        both = left_source & right_source
        left_only = left_source & ~right_source
        right_only = right_source & ~left_source
        support_height_valid = left_source | right_source
        support_height = torch.zeros_like(left_ground_z)
        support_height = torch.where(both, 0.5 * (left_ground_z + right_ground_z), support_height)
        support_height = torch.where(left_only, left_ground_z, support_height)
        support_height = torch.where(right_only, right_ground_z, support_height)
        return support_height, support_height_valid

    def _get_current_foot_support(self):
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3],
            dim=-1,
        )
        threshold = float(self.cfg.recovery.foot_force_threshold)
        return (
            foot_forces[:, self.left_foot_contact_index] > threshold,
            foot_forces[:, self.right_foot_contact_index] > threshold,
        )

    def compute_local_support_height(self, left_support=None, right_support=None):
        """Compute contact-selected terrain height from each foot's finite ray median."""
        if left_support is None or right_support is None:
            left_support, right_support = self._get_current_foot_support()

        if self.left_feet_ray_caster is None or self.right_feet_ray_caster is None:
            zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, invalid, zeros.clone(), zeros.clone()

        left_ground_z, left_ray_valid = self._finite_ray_median(
            self.left_feet_ray_caster.data.ray_hits_w[..., 2]
        )
        right_ground_z, right_ray_valid = self._finite_ray_median(
            self.right_feet_ray_caster.data.ray_hits_w[..., 2]
        )
        support_height, support_height_valid = self._route_support_height(
            left_ground_z,
            left_ray_valid,
            right_ground_z,
            right_ray_valid,
            left_support,
            right_support,
        )
        return support_height, support_height_valid, left_ground_z, right_ground_z

    def _compute_current_torso_clear(self):
        torso_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.termination_contact_cfg.body_ids, :3],
            dim=-1,
        )
        return torch.all(torso_forces < float(self.cfg.recovery.torso_force_threshold), dim=1)

    def compute_recovery_ready_now(self):
        """Evaluate the instantaneous V1 Recovery Ready conditions."""
        left_support, right_support = self._get_current_foot_support()
        support_height, support_height_valid, _, _ = self.compute_local_support_height(
            left_support,
            right_support,
        )
        terrain_relative_base_height = self.robot.data.root_pos_w[:, 2] - support_height

        upright = -self.robot.data.projected_gravity_b[:, 2] > float(self.cfg.recovery.upright_threshold)
        height_ok = support_height_valid & (
            terrain_relative_base_height > self.recovery_ready_height_threshold
        )
        low_ang_vel = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b, dim=1) < float(
            self.cfg.recovery.max_ang_vel
        )
        low_vertical_vel = torch.abs(self.robot.data.root_lin_vel_b[:, 2]) < float(
            self.cfg.recovery.max_vertical_vel
        )
        foot_support = left_support | right_support
        torso_clear = self._compute_current_torso_clear()

        active = self.recovery_mask
        self.recovery_upright_buf = active & upright
        self.recovery_height_ok_buf = active & height_ok
        self.recovery_foot_support_buf = active & foot_support
        self.recovery_torso_clear_buf = active & torso_clear
        self.recovery_ready_now_buf = (
            active
            & upright
            & height_ok
            & low_ang_vel
            & low_vertical_vel
            & foot_support
            & torso_clear
        )
        return self.recovery_ready_now_buf

    def compute_locomotion_failure(self):
        """The original RENet torso-contact termination condition, unchanged."""
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        return torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

    def get_recovery_mask(self):
        """Return action-time mode routing state for the AMP runner."""
        return self.recovery_mask

    def _empty_recovery_diagnostics(self):
        zero = torch.zeros((), dtype=torch.float, device=self.device)
        return {
            "Recovery/active_ratio": zero,
            "Recovery/enter_count": zero.clone(),
            "Recovery/success_count": zero.clone(),
            "Recovery/failure_count": zero.clone(),
            "Recovery/ready_now_ratio": zero.clone(),
            "Recovery/upright_ratio": zero.clone(),
            "Recovery/height_ok_ratio": zero.clone(),
            "Recovery/foot_support_ratio": zero.clone(),
            "Recovery/torso_clear_ratio": zero.clone(),
        }

    def _update_recovery_diagnostics(self, evaluated_recovery_mask):
        evaluated_count = evaluated_recovery_mask.float().sum().clamp(min=1.0)
        self._recovery_diagnostics = {
            "Recovery/active_ratio": self.recovery_mask.float().mean(),
            "Recovery/enter_count": self.enter_recovery_buf.float().sum(),
            "Recovery/success_count": self.exit_recovery_buf.float().sum(),
            "Recovery/failure_count": self.recovery_failed_buf.float().sum(),
            "Recovery/ready_now_ratio": self.recovery_ready_now_buf.float().sum() / evaluated_count,
            "Recovery/upright_ratio": self.recovery_upright_buf.float().sum() / evaluated_count,
            "Recovery/height_ok_ratio": self.recovery_height_ok_buf.float().sum() / evaluated_count,
            "Recovery/foot_support_ratio": self.recovery_foot_support_buf.float().sum() / evaluated_count,
            "Recovery/torso_clear_ratio": self.recovery_torso_clear_buf.float().sum() / evaluated_count,
        }

    def get_recovery_diagnostics(self):
        """Expose the most recent pre-reset Recovery diagnostics."""
        return dict(self._recovery_diagnostics)

    def _reset_recovery_buffers(self, env_ids):
        self.recovery_mask[env_ids] = False
        self.recovery_timer[env_ids] = 0
        self.recovery_ready_counter[env_ids] = 0
        self.recovery_trigger_armed[env_ids] = True
        self.enter_recovery_buf[env_ids] = False
        self.exit_recovery_buf[env_ids] = False
        self.recovery_failed_buf[env_ids] = False
        self.recovery_ready_now_buf[env_ids] = False
        self.recovery_upright_buf[env_ids] = False
        self.recovery_height_ok_buf[env_ids] = False
        self.recovery_foot_support_buf[env_ids] = False
        self.recovery_torso_clear_buf[env_ids] = False

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        # A Recovery failure uses this exact existing reset path. Only the
        # state-machine bookkeeping is additionally cleared.
        self._reset_recovery_buffers(env_ids)

        #
        if self.camera is not None:
            self.depth_buffer[env_ids]=0
        #
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        # ---------------------------------------------------------
        # 1. 处理动作
        # ---------------------------------------------------------
        delayed_actions = self.action_buffer.compute(actions)

        self.action = torch.clip(
            delayed_actions,
            -self.clip_actions,
            self.clip_actions,
        ).to(self.device)

        processed_actions = (
            self.action * self.action_scale
            + self.robot.data.default_joint_pos
        )

        # ---------------------------------------------------------
        # 2. 物理仿真
        # ---------------------------------------------------------
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1

            self.robot.set_joint_position_target(
                processed_actions
            )

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[
                    :, self.feet_cfg.body_ids, :3
                ],
                dim=-1,
            )

            self.avg_feet_speed_per_step += torch.norm(
                self.robot.data.body_lin_vel_w[
                    :, self.ankle_link_ids, :
                ],
                dim=-1,
            )

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        # ---------------------------------------------------------
        # 3. 更新episode、命令和随机事件
        # ---------------------------------------------------------
        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="interval",
                dt=self.step_dt,
            )

        # ---------------------------------------------------------
        # 4. 判断终止和计算任务奖励
        # ---------------------------------------------------------
        self.reset_buf, self.time_out_buf = self.check_reset()
        # Snapshot before true reset clears per-environment Recovery buffers.
        recovery_diagnostics = self.get_recovery_diagnostics()

        reward_buf = self.reward_manager.compute(
            self.step_dt
        )

        self.reset_env_ids = self.reset_buf.nonzero(
            as_tuple=False
        ).flatten()

        # ---------------------------------------------------------
        # 5. 关键：必须在reset前保存真实终止AMP状态
        # ---------------------------------------------------------
        pre_reset_amp_obs = (
            self.get_amp_obs_for_expert_trans()
            .detach()
        )

        # 只保存需要reset的环境，避免保存整个4096×50张量
        terminal_amp_states = (
            pre_reset_amp_obs[self.reset_env_ids]
            .clone()
        )

        # 确保extras结构存在
        if "observations" not in self.extras:
            self.extras["observations"] = {}

        # terminal_amp_states的第i行对应reset_env_ids的第i个环境
        self.extras["terminal_amp_states"] = (
            terminal_amp_states
        )

        # 不要只在reset()内部设置，否则没有reset时可能残留旧值
        self.extras["time_outs"] = (
            self.time_out_buf.clone()
        )

        # ---------------------------------------------------------
        # 6. reset终止环境
        # ---------------------------------------------------------
        self.reset(self.reset_env_ids)
        # Use a fresh dictionary every step so the runner does not retain
        # multiple references to one subsequently mutated diagnostics object.
        step_log = dict(self.extras.get("log", {})) if self.reset_env_ids.numel() > 0 else {}
        step_log.update(recovery_diagnostics)
        self.extras["log"] = step_log

        # ---------------------------------------------------------
        # 7. reset后计算下一时刻观测
        # ---------------------------------------------------------
        actor_obs, critic_obs = self.compute_observations()

        actor_obs = self.append_renet_training_inputs(actor_obs)

        self.extras["observations"]["critic"] = (
            critic_obs
        )

        return (
            actor_obs,
            reward_buf,
            self.reset_buf,
            self.extras,
        )

    def check_reset(self):
        locomotion_failure = self.compute_locomotion_failure()

        self.enter_recovery_buf.zero_()
        self.exit_recovery_buf.zero_()
        self.recovery_failed_buf.zero_()
        self.recovery_ready_now_buf.zero_()
        self.recovery_upright_buf.zero_()
        self.recovery_height_ok_buf.zero_()
        self.recovery_foot_support_buf.zero_()
        self.recovery_torso_clear_buf.zero_()

        if not self.recovery_state_machine_enabled:
            # Strict baseline compatibility: original torso failure and the
            # configured 20-second scene horizon both remain true resets.
            self.recovery_mask.zero_()
            self.recovery_timer.zero_()
            self.recovery_ready_counter.zero_()
            self.recovery_trigger_armed.fill_(True)
            time_out_buf = self.episode_length_buf >= self.max_episode_length
            reset_buf = locomotion_failure | time_out_buf
            self._update_recovery_diagnostics(torch.zeros_like(self.recovery_mask))
            return reset_buf, time_out_buf

        absolute_timeout = self.episode_length_buf >= self.max_episode_length
        was_recovery = self.recovery_mask.clone()

        # A trigger disarmed by entry/success may only re-arm on a later step
        # that started in NORMAL and has a genuinely clear history signal.
        rearm = ~was_recovery & ~locomotion_failure
        self.recovery_trigger_armed[rearm] = True

        self.recovery_timer = torch.where(
            was_recovery,
            self.recovery_timer + 1,
            torch.zeros_like(self.recovery_timer),
        )
        ready_now = self.compute_recovery_ready_now()
        self.recovery_ready_counter = torch.where(
            was_recovery & ready_now,
            self.recovery_ready_counter + 1,
            torch.zeros_like(self.recovery_ready_counter),
        )
        exit_recovery = was_recovery & (
            self.recovery_ready_counter >= self.recovery_ready_hold_steps
        )
        recovery_failed = (
            was_recovery
            & (self.recovery_timer >= self.recovery_max_steps)
            & ~exit_recovery
        )

        # Absolute timeout wins over a simultaneous NORMAL locomotion failure:
        # that environment resets directly and never enters Recovery.
        enter_recovery = (
            ~was_recovery
            & self.recovery_trigger_armed
            & locomotion_failure
            & ~absolute_timeout
        )

        self.enter_recovery_buf.copy_(enter_recovery)
        self.exit_recovery_buf.copy_(exit_recovery)
        self.recovery_failed_buf.copy_(recovery_failed)

        self.recovery_mask[enter_recovery] = True
        self.recovery_timer[enter_recovery] = 0
        self.recovery_ready_counter[enter_recovery] = 0
        self.recovery_trigger_armed[enter_recovery] = False

        self.recovery_mask[exit_recovery] = False
        self.recovery_timer[exit_recovery] = 0
        self.recovery_ready_counter[exit_recovery] = 0
        # Deliberately keep recovery_trigger_armed=False on success. A later
        # NORMAL step with locomotion_failure=False performs the re-arm.

        # Recovery torso contact is ignored. Only its own timeout and the
        # whole-episode absolute horizon are true terminations.
        reset_buf = recovery_failed | absolute_timeout
        time_out_buf = absolute_timeout
        self._update_recovery_diagnostics(was_recovery)
        return reset_buf, time_out_buf
    
    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance
            < torch.norm(self.command_generator.command[env_ids, :2], dim=1)
            * self.baseline_max_episode_length_s
            * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        # 核心修复：确保 extras 中存在 "observations" 这个嵌套字典
        if "observations" not in self.extras:
            self.extras["observations"] = {}

        actor_obs = self.append_renet_training_inputs(actor_obs)
            
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, self.extras
    
    def get_amp_obs_for_expert_trans(self):

        joint_pos = self.robot.data.joint_pos[
            :, self.amp_joint_ids
        ]

        joint_vel = self.robot.data.joint_vel[
            :, self.amp_joint_ids
        ]

        left_hand_pos = (
            self.robot.data.body_state_w[
                :, self.wrist_link_ids[0], :3
            ]
            - self.robot.data.root_state_w[:, 0:3]
        )

        right_hand_pos = (
            self.robot.data.body_state_w[
                :, self.wrist_link_ids[1], :3
            ]
            - self.robot.data.root_state_w[:, 0:3]
        )

        left_foot_pos = (
            self.robot.data.body_state_w[
                :, self.ankle_link_ids[0], :3
            ]
            - self.robot.data.root_state_w[:, 0:3]
        )

        right_foot_pos = (
            self.robot.data.body_state_w[
                :, self.ankle_link_ids[1], :3
            ]
            - self.robot.data.root_state_w[:, 0:3]
        )

        root_quat_inv = quat_conjugate(
            self.robot.data.root_state_w[:, 3:7]
        )

        left_hand_pos = quat_apply(
            root_quat_inv,
            left_hand_pos,
        )
        right_hand_pos = quat_apply(
            root_quat_inv,
            right_hand_pos,
        )
        left_foot_pos = quat_apply(
            root_quat_inv,
            left_foot_pos,
        )
        right_foot_pos = quat_apply(
            root_quat_inv,
            right_foot_pos,
        )

        amp_obs = torch.cat(
            [
                joint_pos,       # 19
                joint_vel,       # 19
                left_hand_pos,   # 3
                right_hand_pos,  # 3
                left_foot_pos,   # 3
                right_foot_pos,  # 3
            ],
            dim=-1,
        )

        if amp_obs.shape[1] != 50:
            raise RuntimeError(
                f"AMP obs 应为50维，实际为 {amp_obs.shape}"
            )

        return amp_obs
    
    def get_processed_deepcamera(self):
        # 获取底层输出: (num_envs, H=36, W=64)
        depth = self.camera.data.output["distance_to_image_plane"].clone()
        depth = depth.squeeze(-1)

        # 1. CropAndResize: 裁剪掉无用视野 (up=18, down=0, left=16, right=16)
        # 结果尺寸: (num_envs, 18, 32)
        depth_cropped = depth[:, self.cfg.robot.depth_crop[0]:self.cfg.scene.camera.camera.pattern_cfg.height-self.cfg.robot.depth_crop[1], self.cfg.robot.depth_crop[2]:self.cfg.scene.camera.camera.pattern_cfg.width-self.cfg.robot.depth_crop[3]]
        
        # 截断与归一化
        depth_cropped[torch.isinf(depth_cropped)] = self.cfg.robot.depth_max
        depth_cropped[torch.isnan(depth_cropped)] = self.cfg.robot.depth_max
        
        # 高斯模糊 (kernel_size=3, sigma=1)
        depth_cropped = depth_cropped.unsqueeze(1) 
        blur_transform = T.GaussianBlur(kernel_size=3, sigma=1.0)
        depth_blurred = blur_transform(depth_cropped)
        depth_blurred = depth_blurred.squeeze(1) # 恢复为 (N, H, W)
        
        # 截断到 2.5 米
        depth_clipped = torch.clip(depth_blurred, min=self.cfg.scene.camera.camera.min_distance, max=self.cfg.robot.depth_max)
        
        # 线性映射并归一化到 [0, 1]
        depth_normalized = depth_clipped / self.cfg.robot.depth_max
        
        return depth_normalized  # （N，H，W）

    def get_deepcamera_history(self):
        current_depth = self.get_processed_deepcamera()

        update_mask = (self.episode_length_buf % self.cfg.robot.depth_update_interval == 0)
        reset_mask = (self.episode_length_buf == 0)

        # 3. 缓冲区移位
        shifted_buffer = torch.roll(self.depth_buffer, shifts=-1, dims=1)
        shifted_buffer[:, -1, :, :] = current_depth
        
        # 4. 向量化条件写入
        update_mask_4d = update_mask.view(-1, 1, 1, 1).expand_as(self.depth_buffer)
        self.depth_buffer = torch.where(update_mask_4d, shifted_buffer, self.depth_buffer)
        
        # 5. 处理刚重置的环境
        if reset_mask.any():
            reset_frames = current_depth[reset_mask].unsqueeze(1).repeat(1, self.depth_history_frames, 1, 1)
            self.depth_buffer[reset_mask] = reset_frames
            
        return self.depth_buffer

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore
 
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        gait_cfg = self.cfg.gait
        if not gait_cfg.enable:
            return

        period = gait_cfg.period
        offset = gait_cfg.offset
        t = self.episode_length_buf.float() * self.step_dt

        self.phase = (t % period) / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1.0
        self.leg_phase[:, 0] = self.phase_left
        self.leg_phase[:, 1] = self.phase_right
    
