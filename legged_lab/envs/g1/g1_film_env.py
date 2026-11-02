import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
import torchvision.transforms as T
from collections import deque

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.utils import math as math_utils
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.g1.g1_film_cfg import G129MOE_FILMENVCFG
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.utils.camera_noise import camera_noise_cfg
from legged_lab.utils.env_utils.scene import SceneCfg
from legged_lab.utils.camera_noise.camera_noise import range_based_gaussian_noise
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay

class G1MOEFILMEnv(VecEnv):
    
    def __init__(
        self,
        cfg: (
            G129MOE_FILMENVCFG
        ),
        headless,
    ):
        self.cfg: (
            G129MOE_FILMENVCFG
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
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()  # 重置一次
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        if "camera" in self.scene.sensors:
            self.camera: GroupedRayCasterCamera = self.scene.sensors["camera"]
        else:
            raise Exception("找不到camera")

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

    def init_buffers(self):
        self.extras = {}
        
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s  # 定义的智能体最多可以存活的时间 20s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt) # 将智能体最大存活时间 转化为最大的步数 为1000步 用于后续判断是否大于最大步数了 大于就重置环境
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
        self.feet_cfg.resolve(self.scene) # 从场景中定位到创建「脚部接触传感器的定位配置」，指定 “哪些刚体是机器人的脚部”，用于检测脚部是否落地
                                        # 用于判断脚是否接触地面了
        
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
        self.depth_noise_curriculum_cfg = getattr(self.cfg, "depth_noise_curriculum", None)
        self.use_depth_noise_curriculum = bool(
            self.depth_noise_curriculum_cfg is not None and self.depth_noise_curriculum_cfg.enable
        )
        self.use_depth_camera_noise = bool(self.cfg.scene.camera.add_camera_noise or self.use_depth_noise_curriculum)
        self._init_depth_noise_curriculum()

        #
        if self.use_depth_camera_noise:
            initial_failure_probability = self.cfg.depth_noise_curriculum.failure_probability_range[1]
            initial_gaussian_std = self.cfg.depth_noise_curriculum.gaussian_std_range[1]
            if self.use_depth_noise_curriculum:
                initial_failure_probability = self._lerp_cfg_range(
                    self.depth_noise_curriculum_cfg.failure_probability_range,
                    self.depth_noise_strength,
                )
                initial_gaussian_std = self._lerp_cfg_range(
                    self.depth_noise_curriculum_cfg.gaussian_std_range,
                    self.depth_noise_strength,
                )
            # 以 initial_failure_probability 概率触发一次 failure，然后从 failure_modes 中随机选一种故障模式，并持续对应帧数。
            self.structured_depth_failure_model = camera_noise_cfg.StructuredDepthFailureModel(
                cfg=camera_noise_cfg.StructuredDepthFailureCfg(
                    failure_probability=initial_failure_probability,
                    failure_duration_range=(2, 4),
                    consecutive_dropout_duration_range=(2,4),
                    freeze_duration_range=(2, 4),
                    distant_dust_duration_range=(2, 4), # 远距离区域被污染，类似灰尘、雾、远处深度失效
                    distant_dust_start=1.5,
                    device=self.device,                    
                ),
                num_envs=self.num_envs,
                device=self.device,
            )
            self.gaussian_cfg = camera_noise_cfg.RangeBasedGaussianNoiseCfg(
                noise_std=initial_gaussian_std,
                min_value=0.0,
                max_value=self.cfg.scene.camera.depth_max,
                device=self.device,
            )

        # 脚连杆的索引
        self.ankle_link_ids,_ = self.robot.find_bodies(
            name_keys=['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True,
        )
        # 手腕处的连杆
        self.wrist_link_ids,_ =self.robot.find_bodies(
            name_keys=['left_wrist_yaw_link', 'right_wrist_yaw_link'],
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
                    'left_wrist_pitch_joint', 
                    'left_wrist_yaw_joint',
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
                    'right_wrist_pitch_joint',
                    'right_wrist_yaw_joint'
            ],
            preserve_order=True,
        )
        # 腰部关节索引
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
            preserve_order=True,
        )

        # Initialize feet state buffers for privileged info
        num_feet = len(self.feet_cfg.body_ids)
        self.feet_pos_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        self.feet_vel_in_body = torch.zeros(self.num_envs, num_feet, 3, device=self.device)

        # Initialize gait phase buffers for bipedal walking
        # phase: normalized gait phase [0, 1)
        # phase_left/phase_right: phase for each leg (offset by gait_phase.offset)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)


        # 脚的平均接触力
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        # 统计每个并行仿真环境下机器人每只脚的平均地面滑动速度
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )

        self.obs_noisy_vec_and_buffer()

    def _init_depth_noise_curriculum(self):
        window_size = 1
        if self.use_depth_noise_curriculum:
            window_size = max(1, int(self.depth_noise_curriculum_cfg.return_window_size))
        self.depth_noise_episode_returns = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # 每一个环境的回报
        self.depth_noise_return_window = deque(maxlen=window_size)
        self.depth_noise_strength = torch.zeros((), dtype=torch.float, device=self.device)
        self.depth_noise_return_cv = torch.zeros((), dtype=torch.float, device=self.device)
        self.depth_noise_return_mean = torch.zeros((), dtype=torch.float, device=self.device)
        self.depth_noise_return_std = torch.zeros((), dtype=torch.float, device=self.device)

    @staticmethod
    def _lerp_cfg_range(value_range, alpha) -> float:
        if isinstance(alpha, torch.Tensor):
            alpha = float(alpha.detach().cpu().item())
        alpha = max(0.0, min(1.0, float(alpha)))
        return float(value_range[0] + (value_range[1] - value_range[0]) * alpha)

    def _accumulate_depth_noise_returns(self, reward_buf: torch.Tensor): # 获取回报
        if not self.use_depth_noise_curriculum:
            return
        rewards = reward_buf.detach().reshape(self.num_envs, -1).sum(dim=1)
        self.depth_noise_episode_returns += rewards

    def _collect_depth_noise_completed_returns(self, env_ids: torch.Tensor):
        if torch.is_tensor(env_ids):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if not self.use_depth_noise_curriculum or env_ids.numel() == 0:
            return

        completed_returns = self.depth_noise_episode_returns[env_ids].detach().cpu().tolist()
        self.depth_noise_return_window.extend(completed_returns)

    def update_depth_noise_curriculum_once(self):
        if not self.use_depth_noise_curriculum:
            return

        returns = torch.tensor(list(self.depth_noise_return_window), dtype=torch.float, device=self.device)
        self.depth_noise_return_mean = returns.mean() if returns.numel() > 0 else torch.zeros((), device=self.device)
        self.depth_noise_return_std = (
            returns.std(unbiased=False) if returns.numel() > 1 else torch.zeros((), device=self.device)
        )

        if returns.numel() < 2:
            self.depth_noise_return_cv = torch.zeros((), dtype=torch.float, device=self.device)
            self.depth_noise_strength = torch.zeros((), dtype=torch.float, device=self.device)
            self._apply_depth_noise_curriculum_strength()
            self._add_depth_noise_curriculum_logs()
            return

        eps = self.depth_noise_curriculum_cfg.cv_epsilon
        return_cv = self.depth_noise_return_std / (self.depth_noise_return_mean.abs() + eps)
        raw_strength = torch.clamp(1.0 - torch.tanh(return_cv), min=0.0, max=1.0)
        ema_beta = max(0.0, min(1.0, float(self.depth_noise_curriculum_cfg.ema_beta)))

        self.depth_noise_return_cv = return_cv.detach()
        self.depth_noise_strength = torch.clamp(
            ema_beta * self.depth_noise_strength + (1.0 - ema_beta) * raw_strength, # self.depth_noise_strength是上一回合的噪声强度，raw_strength是这一回合的噪声强度，这样做是为了使噪声不要突变。
            min=0.0,
            max=1.0,
        ).detach()
        self._apply_depth_noise_curriculum_strength()
        self._add_depth_noise_curriculum_logs()

    def _reset_depth_noise_returns(self, env_ids: torch.Tensor):
        if torch.is_tensor(env_ids):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self.use_depth_noise_curriculum and env_ids.numel() > 0:
            self.depth_noise_episode_returns[env_ids] = 0.0

    def _apply_depth_noise_curriculum_strength(self):
        if not self.use_depth_noise_curriculum:
            return
        self.gaussian_cfg.noise_std = self._lerp_cfg_range(
            self.depth_noise_curriculum_cfg.gaussian_std_range,
            self.depth_noise_strength
        )
        self.structured_depth_failure_model.cfg.failure_probability = self._lerp_cfg_range(
            self.depth_noise_curriculum_cfg.failure_probability_range,
            self.depth_noise_strength,
        )

    def get_depth_noise_curriculum_log(self):
        if not self.use_depth_camera_noise:
            return {}
        self._apply_depth_noise_curriculum_strength()
        log = {}
        if self.use_depth_noise_curriculum:
            log["Curriculum/depth_noise_strength"] = self.depth_noise_strength
            log["Curriculum/depth_noise_return_cv"] = self.depth_noise_return_cv
            log["Curriculum/depth_noise_return_mean"] = self.depth_noise_return_mean
            log["Curriculum/depth_noise_return_std"] = self.depth_noise_return_std
        log["Curriculum/depth_noise_std"] = torch.tensor(self.gaussian_cfg.noise_std, device=self.device)
        log["Curriculum/depth_noise_failure_probability"] = torch.tensor(
            self.structured_depth_failure_model.cfg.failure_probability,
            device=self.device,
        )
        return log

    def _add_depth_noise_curriculum_logs(self):
        if not self.use_depth_camera_noise:
            return
        self.extras.setdefault("log", {}).update(self.get_depth_noise_curriculum_log())

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
                joint_pos * self.obs_scales.joint_pos,  # 29
                joint_vel * self.obs_scales.joint_vel,  # 29
                action * self.obs_scales.actions,  # 29
                torch.sin(2 * torch.pi * self.leg_phase),
                torch.cos(2 * torch.pi * self.leg_phase)
            ],
            dim=-1,
        )
        
        root_lin_vel = robot.data.root_lin_vel_b # 根节点的线速度
        
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs_list = [current_actor_obs,  feet_contact,root_lin_vel * self.obs_scales.lin_vel]

        # Compute feet info for privileged observations
        priv_cfg = self.cfg.scene.privileged_info
        if priv_cfg.enable_feet_info or priv_cfg.enable_feet_contact_force:
            self._compute_feet_state()
        
        # Add feet position and velocity in body frame (12 dim)
        if priv_cfg.enable_feet_info:
            current_critic_obs_list.append(self.feet_pos_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_pos)
            current_critic_obs_list.append(self.feet_vel_in_body.reshape(self.num_envs, -1) * self.obs_scales.feet_vel)
        
        # Add feet contact force 3D (6 dim for 2 feet)
        if priv_cfg.enable_feet_contact_force:
            feet_force = net_contact_forces[:, -1, self.feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)
            current_critic_obs_list.append(feet_force.reshape(self.num_envs, -1) * self.obs_scales.contact_force)
        
        # Add root height (1 dim)
        if priv_cfg.enable_root_height:
            root_height = robot.data.root_pos_w[:, 2:3]  # (num_envs, 1)
            current_critic_obs_list.append(root_height)
        
        current_critic_obs = torch.cat(current_critic_obs_list, dim=-1)
        
        return current_actor_obs, current_critic_obs
    
    def _compute_feet_state(self):
        """Compute feet position and velocity in body frame."""
        robot = self.robot
        
        # Get feet body IDs
        feet_body_ids = self.feet_cfg.body_ids
        
        # Get feet positions in world frame
        feet_pos_w = robot.data.body_pos_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        feet_vel_w = robot.data.body_lin_vel_w[:, feet_body_ids, :]  # (num_envs, num_feet, 3)
        
        # Get root state
        root_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        root_vel_w = robot.data.root_lin_vel_w  # (num_envs, 3)
        root_quat_w = robot.data.root_quat_w  # (num_envs, 4) in (w, x, y, z) format
        
        # Translate to root frame
        feet_pos_translated = feet_pos_w - root_pos_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        feet_vel_translated = feet_vel_w - root_vel_w.unsqueeze(1)  # (num_envs, num_feet, 3)
        
        # Rotate to body frame using Isaac Lab's quat_apply_inverse
        num_feet = feet_pos_translated.shape[1]
        for i in range(num_feet):
            self.feet_pos_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_pos_translated[:, i, :])
            self.feet_vel_in_body[:, i, :] = math_utils.quat_apply_inverse(root_quat_w, feet_vel_translated[:, i, :])
        
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

        self.actor_obs = torch.clip(self.actor_obs, -self.clip_obs, self.clip_obs)
        self.critic_obs = torch.clip(self.critic_obs, -self.clip_obs, self.clip_obs)

        return self.actor_obs, self.critic_obs
    
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
        self._add_depth_noise_curriculum_logs()

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._reset_depth_noise_returns(env_ids)

        #
        if self.camera is not None:
            self.depth_buffer[env_ids]=0
        if self.use_depth_camera_noise:
            self.structured_depth_failure_model.reset(env_ids)
        
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.ankle_link_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self._accumulate_depth_noise_returns(reward_buf)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._collect_depth_noise_completed_returns(self.reset_env_ids)
        self.reset(self.reset_env_ids)
        self._add_depth_noise_curriculum_logs()

        actor_obs, critic_obs = self.compute_observations()
        if self.camera is not None:
            depth_obs = self.get_deepcamera_history()
            flat_depth = depth_obs.view(self.num_envs, -1)
            self.extras["observations"]["depth"] = flat_depth
            
            # 核心修改：同理，在 step 中也进行拼接
            actor_obs = torch.cat([actor_obs, flat_depth], dim=-1)
            critic_obs = torch.cat([critic_obs], dim=-1)
            
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, reward_buf, self.reset_buf, self.extras

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

        orientation_reset_buf = torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs() > self.cfg.robot.limit_angle

        reset_buf|=orientation_reset_buf

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

        if self.camera is not None:
            depth_obs = self.get_deepcamera_history()
            # 将 (num_envs, 8, 18, 32) 展平为 (num_envs, 4608)
            flat_depth = depth_obs.view(self.num_envs, -1)
            self.extras["observations"]["depth"] = flat_depth
            
            # 核心修改：将深度图直接拼接到本体观测的末尾
            actor_obs = torch.cat([actor_obs, flat_depth], dim=-1)
            critic_obs = torch.cat([critic_obs], dim=-1)
            
        self.extras["observations"]["critic"] = critic_obs
        return actor_obs, self.extras
    
    def add_camera_noise(self,depth,env_ids):
        depth = depth.unsqueeze(-1)

        self._apply_depth_noise_curriculum_strength()
        depth = range_based_gaussian_noise(depth, self.gaussian_cfg, env_ids)
        depth = self.structured_depth_failure_model(depth, self.structured_depth_failure_model.cfg, env_ids)
        return depth.squeeze(-1)

    def get_processed_deepcamera(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 获取底层输出: [num_envs, H, W, 1] 或 [num_envs, H, W]
        depth = self.camera.data.output["distance_to_image_plane"][env_ids].clone()
        depth = depth.squeeze(-1)

        # Crop
        crop_up, crop_down, crop_left, crop_right = self.cfg.scene.camera.depth_crop

        H_full = self.cfg.scene.camera.camera.pattern_cfg.height
        W_full = self.cfg.scene.camera.camera.pattern_cfg.width

        depth_cropped = depth[
            :,
            crop_up : H_full - crop_down,
            crop_left : W_full - crop_right,
        ]

        # 处理 inf / nan
        depth_cropped[torch.isinf(depth_cropped)] = self.cfg.scene.camera.depth_max
        depth_cropped[torch.isnan(depth_cropped)] = self.cfg.scene.camera.depth_max

        # Clip
        depth_clipped = torch.clip(
            depth_cropped,
            min=self.cfg.scene.camera.camera.min_distance,
            max=self.cfg.scene.camera.depth_max,
        )

        # 加噪声，只对这些 env_ids 加
        if self.use_depth_camera_noise:
            depth_clipped = self.add_camera_noise(depth_clipped, env_ids)
            depth_clipped = torch.clip(
                depth_clipped,
                min=0.0,
                max=self.cfg.scene.camera.depth_max,
            )

        # 归一化到 [0, 1]
        depth_normalized = depth_clipped / self.cfg.scene.camera.depth_max
        if getattr(self, "debug_depth", True):
            zero_ratio = (depth_normalized < 0.03).float().mean().item()
            print(
                "[DEPTH DEBUG] "
                f"min={depth_normalized.min().item():.3f}, "
                f"max={depth_normalized.max().item():.3f}, "
                f"mean={depth_normalized.mean().item():.3f}, "
                f"zero_ratio={zero_ratio:.3f}"
            )
        return depth_normalized

    def get_deepcamera_history(self):
        update_mask = (
            self.episode_length_buf % self.cfg.scene.camera.depth_update_interval == 0
        )
        reset_mask = self.episode_length_buf == 0

        need_depth_mask = update_mask | reset_mask

        # 如果当前 step 没有任何环境需要更新深度，直接返回旧 buffer
        if not need_depth_mask.any():
            return self.depth_buffer

        env_ids = need_depth_mask.nonzero(as_tuple=False).flatten()

        # 只处理需要更新的 env
        current_depth = self.get_processed_deepcamera(env_ids=env_ids)
        # current_depth: [num_need_envs, H, W]

        # 1. 对需要正常更新的环境，滚动 buffer 并写入最新深度
        update_in_subset = update_mask[env_ids]
        if update_in_subset.any():
            update_env_ids = env_ids[update_in_subset]
            update_depth = current_depth[update_in_subset]

            shifted_buffer = torch.roll(
                self.depth_buffer[update_env_ids],
                shifts=-1,
                dims=1,
            )
            shifted_buffer[:, -1, :, :] = update_depth
            self.depth_buffer[update_env_ids] = shifted_buffer

        # 2. 对刚 reset 的环境，用当前深度填满全部历史帧
        reset_in_subset = reset_mask[env_ids]
        if reset_in_subset.any():
            reset_env_ids = env_ids[reset_in_subset]
            reset_depth = current_depth[reset_in_subset]

            reset_frames = reset_depth.unsqueeze(1).repeat(
                1,
                self.depth_history_frames,
                1,
                1,
            )
            self.depth_buffer[reset_env_ids] = reset_frames

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
        """Update gait phase for bipedal walking.
        
        Computes the normalized gait phase [0, 1) based on episode time.
        Left and right legs have a phase offset (default 0.5 = alternating gait).
        
        The gait phase is used by gait_phase_contact reward to encourage
        proper stance/swing timing for each leg.
        
        Reference: DreamWaQ _post_physics_step_callback()
        """
        gait_cfg = self.cfg.gait
        if not gait_cfg.enable:
            return
            
        period = gait_cfg.period  # Gait cycle period in seconds (e.g., 0.8s)
        offset = gait_cfg.offset  # Phase offset between legs (e.g., 0.5 = 50%)
        
        # Compute normalized phase from episode time
        # t = episode_length * step_dt, phase = (t % period) / period
        t = self.episode_length_buf.float() * self.step_dt
        self.phase = (t % period) / period
        
        # Left leg uses base phase, right leg is offset
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1.0
        
        # Stack for convenience (used by some reward functions)
        self.leg_phase[:, 0] = self.phase_left
        self.leg_phase[:, 1] = self.phase_right
