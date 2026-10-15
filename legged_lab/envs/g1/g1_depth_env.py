import copy

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
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

from legged_lab.envs.g1.g1_depth_cfg import G1DEPTHENVCFG
from legged_lab.sensors.camera_noise import distance_dependent_gaussian_noise
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.utils.env_utils.scene import SceneCfg

from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay

class G1DepthEnv(VecEnv):
    
    def __init__(
        self,
        cfg: (
            G1DEPTHENVCFG
        ),
        headless,
    ):
        self.cfg: (
            G1DEPTHENVCFG
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
        scene_config = copy.deepcopy(cfg.scene)
        scene_config.lidar = None
        scene_cfg = SceneCfg(config=scene_config, physics_dt=self.physics_dt, step_dt=self.step_dt)
        cfg.foot_edge.validate()
        # 此处仅配置奖励开关；奖励数值仍由RewardManager统一计算。
        reward_cfg = copy.deepcopy(cfg.reward)
        if not cfg.foot_edge.enabled:
            reward_cfg.foot_edge_penalty = None
        self._foot_edge_reward_weight = (
            float(reward_cfg.foot_edge_penalty.weight)
            if reward_cfg.foot_edge_penalty is not None else 0.0
        )
        if not np.isfinite(self._foot_edge_reward_weight) or self._foot_edge_reward_weight > 0:
            raise ValueError("foot_edge_penalty weight must be finite and nonpositive.")
        if self._foot_edge_reward_weight == 0.0:
            reward_cfg.foot_edge_penalty = None
        needs_edge_geometry = self._foot_edge_reward_weight != 0.0 or cfg.foot_edge.debug_vis
        if needs_edge_geometry:
            from legged_lab.terrains.foot_edge_terrain import with_foot_edge_geometry

            scene_cfg.terrain = with_foot_edge_geometry(scene_cfg.terrain, cfg.foot_edge.geometry)
        # Air-time rewards require contact timers. Follow the task's enabled
        # reward so disabling it also avoids computing unused timer buffers.
        air_time_reward = getattr(cfg.reward, "feet_air_time", None)
        scene_cfg.contact_sensor.track_air_time = (
            air_time_reward is not None and air_time_reward.weight != 0.0
        )
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
            raise ValueError("g1_depth requires the depth camera.")

        self.left_feet_ray_caster = (
            self.scene.sensors["left_feet_ray_caster"] if "left_feet_ray_caster" in self.scene.sensors else None
        )
        self.right_feet_ray_caster = (
            self.scene.sensors["right_feet_ray_caster"] if "right_feet_ray_caster" in self.scene.sensors else None
        )

        from legged_lab.mdp.foot_edge import FootEdgeMonitor

        self.foot_edge_monitor = FootEdgeMonitor(self)
        if needs_edge_geometry and getattr(self.scene.terrain, "foot_edge_geometry", None) is None:
            raise RuntimeError("Foot edge geometry was requested but not initialized.")

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
        self.reward_manager = RewardManager(reward_cfg, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 设置了 事件管理器
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._sync_domain_randomization_stage()
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
        # Keep the historical +0.1 m default, while allowing the playback tool
        # to disable it for trajectories already grounded against this URDF.
        root_pos[2] += float(
            getattr(self, "visualization_root_height_offset", 0.1)
        )
 
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

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s  # 定义的智能体最多可以存活的时间 20s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt) # 将智能体最大存活时间 转化为最大的步数 为1000步 用于后续判断是否大于最大步数了 大于就重置环境
        self.num_actions = self.robot.data.default_joint_pos.shape[1]  # 获取机器人的动作维度（关节数量），这是策略网络输出层的维度。
        self.clip_actions = self.cfg.normalization.clip_actions   # 读取 “动作裁剪阈值”，限制策略网络输出的动作范围，避免动作过大导致机器人关节损坏 / 物理仿真崩溃。
        self.clip_obs = self.cfg.normalization.clip_observations  # 读取 “观测裁剪阈值”，限制机器人观测数据的范围，避免异常值（如传感器故障、物理抖动）导致网络训练不稳定

        self.action_scale = self.cfg.robot.action_scale  # 把策略网络输出的 “归一化动作” 映射到机器人实际能执行的控制范围；
        for name, start_iteration in (
            ("depth_training.domain_rand_start_iteration", self.cfg.depth_training.domain_rand_start_iteration),
            ("depth_noise_start_iteration", self.cfg.depth_noise_start_iteration),
        ):
            if isinstance(start_iteration, bool) or not isinstance(start_iteration, int) or start_iteration < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        self._init_action_delay()
        
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
        
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            self.noisy=self.cfg.noise.noise_scales
        # 创建「episode 步数缓冲区」，记录每个环境当前 episode 已经运行的步数
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0 # 创建「全局仿真步数计数器」，记录整个仿真运行的总步数（所有环境共享），而非单个环境的 episode 步数
        # 创建「超时标记缓冲区」，标记哪些环境因达到最大 episode 长度需要重置
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.training_iteration = 0
        self._domain_rand_stage_active = False
        self._domain_rand_startup_applied = False
    
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
            self.depth_last_capture_time = torch.full((self.num_envs,), -torch.inf, device=self.device)
            self.depth_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.depth_valid_buffer = torch.zeros_like(self.depth_buffer)
            self.depth_current_valid = torch.zeros(
                self.num_envs, self.camera_height, self.camera_width, device=self.device)
        #
        self.amp_observation_dim = self.cfg.amp_observation_dim
        if type(self.amp_observation_dim) is not int or self.amp_observation_dim not in (50, 58):
            raise ValueError("g1_depth amp_observation_dim must be 50 or 58.")
        amp_joint_count = (self.amp_observation_dim - 12) // 2
        amp_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",

            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",

            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",

            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",

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

        if self.amp_observation_dim == 50:
            amp_joint_names = [name for name in amp_joint_names if "ankle" not in name]

        self.amp_joint_ids, self.amp_joint_names = (
            self.robot.find_joints(
                amp_joint_names,
                preserve_order=True,
            )
        )

        if len(self.amp_joint_ids) != amp_joint_count or self.amp_joint_names != amp_joint_names:
            raise RuntimeError(
                f"AMP应按指定顺序找到{amp_joint_count}个关节，"
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
        self.obs_noisy_vec_and_buffer()
        if self.num_actions != 23 or self.cfg.robot.actor_obs_history_length != 10 or self.cfg.robot.critic_obs_history_length != 10:
            raise ValueError("g1_depth requires 23 actions and ten-frame actor/critic histories.")
        if (self.depth_history_frames, self.camera_height, self.camera_width) != (2, 18, 32):
            raise ValueError("g1_depth requires two 18x32 depth frames.")
        camera_cfg = self.cfg.scene.camera.camera
        pattern = camera_cfg.pattern_cfg
        self.depth_observation_contract = {
            "version": 1, "task": "g1_depth", "actor_observation_dim": 1932,
            "critic_observation_dim": 1037, "action_dim": 23,
            "proprio_history_frames": 10, "proprio_frame_dim": 78,
            "proprio_order": ["angular_velocity", "projected_gravity", "command", "joint_position_offset", "joint_velocity", "previous_action"],
            "joint_names": list(self.robot.joint_names),
            "obs_scales": {name: getattr(self.obs_scales, name) for name in ("ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions")},
            "depth_shape": [2, 18, 32], "depth_capture_period_s": self.cfg.robot.depth_update_interval * self.step_dt,
            "sequence_order": "oldest_to_current", "depth_crop": list(self.cfg.robot.depth_crop),
            "depth_min_m": camera_cfg.min_distance, "depth_max_m": self.cfg.robot.depth_max,
            "depth_normalization": "clip_metres_divide_depth_max", "depth_invalid_fill": "nonfinite_to_depth_max",
            "camera_size": [pattern.height, pattern.width], "camera_focal_length": pattern.focal_length,
            "camera_horizontal_aperture": pattern.horizontal_aperture, "camera_vertical_aperture": pattern.vertical_aperture,
            "camera_position": list(camera_cfg.offset.pos), "camera_rotation": list(camera_cfg.offset.rot),
            "camera_convention": camera_cfg.offset.convention, "camera_ray_alignment": camera_cfg.ray_alignment,
            "control_dt_s": self.step_dt, "action_scale": self.action_scale,
            "clip_observations": self.clip_obs, "clip_actions": self.clip_actions,
        }


    def _init_action_delay(self):
        """Allocate the action history and validate delays in control steps."""
        params = self.cfg.domain_rand.action_delay.params
        min_steps = params.get("min_delay", 0)
        max_steps = params.get("max_delay", 0)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (min_steps, max_steps)):
            raise ValueError("action_delay min_delay/max_delay must be integers in control steps.")
        if not 0 <= min_steps <= max_steps:
            raise ValueError("action_delay requires 0 <= min_delay <= max_delay.")
        self.action_delay_min_steps = min_steps
        self.action_delay_max_steps = max_steps
        self.action_buffer = DelayBuffer(max_steps, self.num_envs, device=self.device)
        self.action_buffer.compute(torch.zeros(self.num_envs, self.num_actions, device=self.device))

    def _reset_action_delay(self, env_ids):
        """Resample only reset environments; keep the lag fixed within an episode."""
        if self.cfg.domain_rand.action_delay.enable and self._domain_rand_is_active():
            time_lags = torch.randint(
                self.action_delay_min_steps, self.action_delay_max_steps + 1,
                (len(env_ids),), dtype=torch.int, device=self.device,
            )
        else:
            time_lags = torch.zeros(len(env_ids), dtype=torch.int, device=self.device)
        self.action_buffer.set_time_lag(time_lags, batch_ids=env_ids)
        mean_steps = self.action_buffer.time_lags.float().mean()
        self.extras.setdefault("log", {}).update({
            "ActionDelay/mean_steps": mean_steps,
            "ActionDelay/mean_ms": mean_steps * self.step_dt * 1000.0,
            "ActionDelay/max_steps": self.action_buffer.max_time_lag,
        })

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
        当前的CRITIC:
            当前的观测(78维)、根节点自身坐标系下线速度(3维)、脚接触状态(2维)、脚高度(2维)
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
            self.feet_height_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
            return torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        targets = []
        validity = []
        for ray_sensor in (self.left_feet_ray_caster, self.right_feet_ray_caster):
            sensor_z = ray_sensor.data.pos_w[:, 2:3]
            hit_z = ray_sensor.data.ray_hits_w[..., 2]
            valid_hit = torch.isfinite(hit_z)
            count = valid_hit.sum(dim=-1, keepdim=True)
            valid = (count > 0) & torch.isfinite(sensor_z)
            ground_z = torch.where(valid_hit, hit_z, 0.0).sum(dim=-1, keepdim=True) / count.clamp_min(1)
            height = torch.where(valid, sensor_z - ground_z, 0.0)
            targets.append(height.clamp(min=-1.0, max=1.0))
            validity.append(valid)
        self.feet_height_valid = torch.cat(validity, dim=-1)
        return torch.cat(targets, dim=-1)

    def compute_terrain_height_target(self):
        data = self.height_scanner.data
        hits = data.ray_hits_w
        sensor_z = data.pos_w[:, 2:3]
        valid = torch.isfinite(hits).all(dim=-1) & torch.isfinite(sensor_z)
        # Use the actual elevated ray origins, not the tracked body's position.
        distance = (hits - self.height_scanner.ray_starts_w).norm(dim=-1)
        valid &= torch.isfinite(distance) & (distance <= self.height_scanner.cfg.max_distance)
        height = (sensor_z - hits[..., 2] - self.cfg.normalization.height_scan_offset) * self.obs_scales.height_scan
        valid &= torch.isfinite(height)
        self.terrain_height_valid = valid
        return torch.where(valid, height, 0.0)

    def set_training_iteration(self, iteration: int):
        if type(iteration) is not int or iteration < 0:
            raise ValueError("training iteration must be a nonnegative integer.")
        self.training_iteration = iteration
        self._sync_domain_randomization_stage()

    def _domain_rand_is_active(self):
        return self.training_iteration >= self.cfg.depth_training.domain_rand_start_iteration

    def _sync_domain_randomization_stage(self):
        """Apply deferred startup properties once, then enable push/lag sampling."""
        active = self._domain_rand_is_active()
        if active == self._domain_rand_stage_active:
            return
        if active and not self._domain_rand_startup_applied:
            if "randomization_start" in self.event_manager.available_modes:
                self.event_manager.apply(mode="randomization_start")
            self._domain_rand_startup_applied = True
        self._domain_rand_stage_active = active
        # Enable delays at the iteration boundary, including ongoing episodes.
        # Keep the queued actions, and keep each sampled lag until episode reset.
        self._reset_action_delay(torch.arange(self.num_envs, device=self.device))


    def _depth_noise_is_active(self):
        return bool(
            self.cfg.scene.camera.add_camera_noise
            and (
                self.cfg.depth_noise_force_active
                or self.training_iteration >= self.cfg.depth_noise_start_iteration
            )
        )






    def append_depth_training_inputs(self, actor_obs):
        depth = self.get_deepcamera_history().reshape(self.num_envs, -1)
        self.extras.setdefault("observations", {})["depth"] = depth
        self.extras.setdefault("log", {}).update({
            "Depth/noise_enabled": float(self._depth_noise_is_active()),
            "Depth/noise_stage_start_iteration": float(self.cfg.depth_noise_start_iteration),
        })
        result = torch.cat((actor_obs, depth), dim=-1)
        if result.shape != (self.num_envs, 1932):
            raise RuntimeError(f"g1_depth actor observation mismatch: {tuple(result.shape)}")
        return result

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
            height_scan_noise_vec = torch.zeros_like(self.height_scanner.data.ray_hits_w[0, :, 2])
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
            uniform = torch.rand_like(current_actor_obs)
            current_actor_obs += (2 * uniform - 1) * self.noise_obs_vec
        
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        self.actor_obs=self.actor_obs_buffer.buffer.reshape(self.num_envs,-1)
        self.critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = self.compute_terrain_height_target()
            self.critic_obs = torch.cat([self.critic_obs, height_scan], dim=-1)
            self.extras.setdefault("observations", {})["terrain_height_valid"] = self.terrain_height_valid

        self.actor_obs = torch.clip(self.actor_obs, -self.clip_obs, self.clip_obs)
        self.critic_obs = torch.clip(self.critic_obs, -self.clip_obs, self.clip_obs)
        # Keep label validity separate from the 85-D critic and actor contracts.
        self.extras.setdefault("observations", {})["feet_height_valid"] = self.feet_height_valid

        return self.actor_obs, self.critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        self.foot_edge_monitor.reset(env_ids)
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
        self._reset_action_delay(env_ids)
        self.episode_length_buf[env_ids] = 0

        #
        if self.camera is not None:
            self.depth_buffer[env_ids]=0
            self.depth_valid_buffer[env_ids] = 0
            self.depth_current_valid[env_ids] = 0
            self.depth_last_capture_time[env_ids] = -torch.inf
            self.depth_initialized[env_ids] = False
        #
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        action_command = self.command_generator.command.clone()
        self.extras["log"] = {}
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
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1

            self.robot.set_joint_position_target(
                processed_actions
            )

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            # Keep sampling contacts at the physics rate (200 Hz), even though
            # periodic gait force/speed averages are no longer computed.
            # Scene sensors update lazily, so consuming data here is required.
            _ = self.contact_sensor.data

        if not self.headless:
            self.sim.render()

        # ---------------------------------------------------------
        # 3. 更新episode和随机事件
        # ---------------------------------------------------------
        self.episode_length_buf += 1

        # With the default start_iteration=0, push countdown advances from the
        # first control step. An explicitly deferred schedule can still pause it.
        if self._domain_rand_is_active() and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="interval",
                dt=self.step_dt,
            )

        # ---------------------------------------------------------
        # 4. 判断终止和计算任务奖励
        # ---------------------------------------------------------
        self.reset_buf, self.time_out_buf = self.check_reset()

        reward_buf = self.reward_manager.compute(
            self.step_dt
        )
        # 方向跟踪日志使用生成本次动作的命令，先于下一命令更新。
        yaw = torch.atan2(
            2 * (self.robot.data.root_quat_w[:, 0] * self.robot.data.root_quat_w[:, 3]
                 + self.robot.data.root_quat_w[:, 1] * self.robot.data.root_quat_w[:, 2]),
            1 - 2 * (self.robot.data.root_quat_w[:, 2].square()
                     + self.robot.data.root_quat_w[:, 3].square()),
        )
        vx, vy = self.robot.data.root_lin_vel_w[:, :2].unbind(-1)
        actual_xy = torch.stack((vx*yaw.cos()+vy*yaw.sin(), -vx*yaw.sin()+vy*yaw.cos()), -1)
        xy_error = (actual_xy-action_command[:, :2]).square().sum(-1)
        directions = {
            "forward": action_command[:, 0] > 0.1,
            "backward": action_command[:, 0] < -0.1,
            "lateral": action_command[:, 1].abs() > 0.1,
            "standing": action_command.norm(dim=-1) < 0.1,
        }
        step_metrics = {
            "DomainRand/enabled": float(self._domain_rand_is_active()),
            "DomainRand/start_iteration": float(self.cfg.depth_training.domain_rand_start_iteration),
            "ActionDelay/mean_steps": self.action_buffer.time_lags.float().mean(),
            "ActionDelay/mean_ms": self.action_buffer.time_lags.float().mean() * self.step_dt * 1000.0,
            "ActionDelay/max_steps": self.action_buffer.max_time_lag,
        }
        step_metrics["Depth/reward"] = reward_buf.mean()
        # 必须在reset前记录；诊断函数不添加任何奖励。
        step_metrics.update(self.foot_edge_monitor.diagnostics())
        for direction, direction_mask in directions.items():
            if direction_mask.any():
                step_metrics[f"Depth/{direction}_tracking_xy_mse"] = xy_error[direction_mask].mean()

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

        # 只保存需要reset的环境，避免保存所有环境的AMP状态。
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
        self.extras.setdefault("log", {}).update(step_metrics)

        # ---------------------------------------------------------
        # 7. 奖励使用产生本次动作的命令，更新后的命令供下一时刻使用
        # ---------------------------------------------------------
        self.command_generator.compute(self.step_dt)
        actor_obs, critic_obs = self.compute_observations()

        actor_obs = self.append_depth_training_inputs(actor_obs)

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
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
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
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf
    
    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
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

        actor_obs = self.append_depth_training_inputs(actor_obs)
            
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
                joint_pos,       # 19 or 23, fixed AMP joint order
                joint_vel,       # same joint order as positions
                left_hand_pos,   # 3
                right_hand_pos,  # 3
                left_foot_pos,   # 3
                right_foot_pos,  # 3
            ],
            dim=-1,
        )

        expected_dim = 2 * len(self.amp_joint_ids) + 12
        if amp_obs.shape[1] != expected_dim:
            raise RuntimeError(
                f"AMP obs 应为{expected_dim}维，实际为 {amp_obs.shape}"
            )

        return amp_obs
    
    def get_processed_deepcamera(self, env_ids=None):
        # 获取底层输出: (num_envs, H=36, W=64)
        depth = self.camera.data.output["distance_to_image_plane"].clone()
        depth = depth.squeeze(-1)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        depth = depth[env_ids]

        # 1. CropAndResize: 裁剪掉无用视野 (up=18, down=0, left=16, right=16)
        # 结果尺寸: (num_envs, 18, 32)
        depth_cropped = depth[:, self.cfg.robot.depth_crop[0]:self.cfg.scene.camera.camera.pattern_cfg.height-self.cfg.robot.depth_crop[1], self.cfg.robot.depth_crop[2]:self.cfg.scene.camera.camera.pattern_cfg.width-self.cfg.robot.depth_crop[3]]
        self.depth_current_valid[env_ids] = (torch.isfinite(depth_cropped) & (depth_cropped > 0)).float()

        # 截断与归一化
        depth_cropped[torch.isinf(depth_cropped)] = self.cfg.robot.depth_max
        depth_cropped[torch.isnan(depth_cropped)] = self.cfg.robot.depth_max
        
        # Independently select whole frames per environment at capture time.
        # Stored history frames retain this decision until the next capture.
        blur_probability = self.cfg.depth_blur_probability
        if not 0.0 <= blur_probability <= 1.0:
            raise ValueError("depth_blur_probability must be in [0, 1].")
        noise_active = self._depth_noise_is_active()
        depth_blurred = depth_cropped
        if noise_active and blur_probability > 0.0:
            uniform = torch.rand(depth_cropped.shape[0], device=depth_cropped.device)
            blur_mask = uniform < blur_probability
            if blur_mask.any():
                depth_blurred = depth_cropped.clone()
                depth_blurred[blur_mask] = T.GaussianBlur(kernel_size=3, sigma=1.0)(
                    depth_cropped[blur_mask].unsqueeze(1)
                ).squeeze(1)

        # Apply stochastic noise while depth is still in metres. Keeping this
        # after the optional blur preserves the configured per-pixel std.
        if noise_active:
            camera_env_ids = env_ids
            depth_blurred = distance_dependent_gaussian_noise(
                depth_blurred,
                self.cfg.depth_noise,
                camera_env_ids,
                min_distance=self.cfg.scene.camera.camera.min_distance,
                max_distance=self.cfg.robot.depth_max,
            )
        
        # 截断到 2.5 米
        depth_clipped = torch.clip(depth_blurred, min=self.cfg.scene.camera.camera.min_distance, max=self.cfg.robot.depth_max)
        
        # 线性映射并归一化到 [0, 1]
        depth_normalized = depth_clipped / self.cfg.robot.depth_max
        
        return depth_normalized  # （N，H，W）

    def get_deepcamera_history(self):
        # Same 2-frame VP input and image processing, but packet timing must
        # not depend on episode_length_buf (randomized by the training runner).
        now = self.camera.current_timestamps
        period = self.cfg.robot.depth_update_interval * self.step_dt
        update_mask = (now-self.depth_last_capture_time >= period-1e-6) | ~self.depth_initialized
        if update_mask.any():
            ids = update_mask.nonzero().flatten()
            current_depth = self.get_processed_deepcamera()[ids]
            self.depth_buffer[update_mask] = torch.roll(self.depth_buffer[update_mask], shifts=-1, dims=1)
            self.depth_buffer[update_mask, -1] = current_depth
            self.depth_valid_buffer[update_mask] = torch.roll(self.depth_valid_buffer[update_mask], shifts=-1, dims=1)
            self.depth_valid_buffer[update_mask, -1] = self.depth_current_valid[update_mask]
            reset_mask = ~self.depth_initialized
            first_local = reset_mask[ids]
            self.depth_buffer[reset_mask] = current_depth[first_local, None].expand(-1, self.depth_history_frames, -1, -1)
            self.depth_valid_buffer[reset_mask] = self.depth_current_valid[reset_mask, None].expand(-1, self.depth_history_frames, -1, -1)
            self.depth_initialized[update_mask] = True
            self.depth_last_capture_time[update_mask] = now[update_mask]
        self.extras.setdefault("observations", {})["depth_valid_support"] = self.depth_valid_buffer
        return self.depth_buffer

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore
 
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
