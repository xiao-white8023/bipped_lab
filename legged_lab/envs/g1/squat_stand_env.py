import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
import torchvision.transforms as T

from isaaclab.assets.articulation import Articulation

from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.utils.height_cammand.height_cammand import UniformHeightCommand
from legged_lab.utils.height_cammand.height_cammand_cfg import UniformHeightCommandCfg

from legged_lab.envs.g1.squat_stand_cfg import SaquatStandENVCFG
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.utils.env_utils.scene import SceneCfg

from rsl_rl.env import VecEnv


class SquatStandEnv(VecEnv):
    
    def __init__(
        self,
        cfg: (
            SaquatStandENVCFG
        ),
        headless,
    ):
        self.cfg: (
            SaquatStandENVCFG
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


        command_cfg = UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            stand_height=self.cfg.commands.stand_height,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        # 设置了 速度命令生成器 奖励管理器
        self.command_generator = UniformHeightCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.current_iteration = 0
        self.com_margin_factor = 0.0
        self._com_support_margin_weight = float(
            self.reward_manager.get_term_cfg("com_support_margin").weight
        )
        self._validate_com_support_margin_curriculum()
        
        self.init_buffers()
        self._apply_com_support_margin_curriculum()

        env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 设置了 事件管理器
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.update_body_mass_cache()
        self.reset(env_ids)

    def init_buffers(self):
        self.extras = {}
        
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s  # 定义的智能体最多可以存活的时间 20s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt) # 将智能体最大存活时间 转化为最大的步数 为1000步 用于后续判断是否大于最大步数了 大于就重置环境

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
        
        # 机器人所有刚体的质量
        # shape: (num_envs, num_bodies)
        self.body_masses = torch.zeros(
            self.num_envs,
            self.robot.num_bodies,
            dtype=torch.float,
            device=self.device,
        )
        # 每个环境下机器人的总质量
        # shape: (num_envs, 1)
        self.total_body_mass = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
        )
        # 整台机器人质心在世界坐标系下的位置
        # shape: (num_envs, 3)
        self.whole_body_com_pos_w = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
        )

        self.whole_body_com_vel_w = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
        )
        # 连杆坐标系下的脚的4个角点
        self.foot_support_corners_b = torch.tensor(
            [
                [0.127,  0.030, -0.0354],   # 前端，局部 +y 侧
                [0.127, -0.030, -0.0354],   # 前端，局部 -y 侧
                [-0.051, -0.030, -0.0354],  # 后端，局部 -y 侧
                [-0.051,  0.030, -0.0354],  # 后端，局部 +y 侧
            ],
            dtype=torch.float,
            device=self.device,
        )
    
        # 脚连杆的索引
        self.ankle_link_ids,_ = self.robot.find_bodies(
            name_keys=['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True,
        )
        self.foot_support_corners_w = torch.zeros(
            self.num_envs,
            len(self.ankle_link_ids),              # 2 只脚
            self.foot_support_corners_b.shape[0],  # 4 个角点
            3,                                     # xyz
            dtype=torch.float,
            device=self.device,
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
        self.left_arm_ids = torch.as_tensor(
            self.left_arm_ids, device=self.device, dtype=torch.long
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
        self.right_arm_ids = torch.as_tensor(
            self.right_arm_ids, device=self.device, dtype=torch.long
        )
        self.num_right_arm_joints = len(self.right_arm_ids)
        arm_buffer_shape = (self.num_envs, self.num_right_arm_joints) # (4096,7) 4096个机器人，每一个的右手臂都是7个关节
        self.arm_motion_start_q = torch.zeros(
            arm_buffer_shape, dtype=torch.float, device=self.device
        ) # 这一次右臂运动开始时的关节角度
        '''
        这次右臂随机运动的目标姿态,例如当前：肩pitch=0  随机采样：肩pitch=0.8rad  那么：arm_motion_target_q保存[0.8, ...] 之后：minimum jerk：从start_q移动到：target_q
        '''
        self.arm_motion_target_q = torch.zeros(
            arm_buffer_shape, dtype=torch.float, device=self.device
        )
        # 当前时刻，右臂控制器希望机器人达到的目标关节位置
        self.arm_q_des = torch.zeros(
            arm_buffer_shape, dtype=torch.float, device=self.device
        )
        # 当前期望速度。
        self.arm_dq_des = torch.zeros(
            arm_buffer_shape, dtype=torch.float, device=self.device
        )
        # 当前这一次右臂动作已经执行了多久。
        self.arm_motion_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # 从 start 到 target 需要移动多久
        self.arm_motion_duration = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # 到达目标后要保持多久
        self.arm_motion_hold_duration = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # 每一个关节的最大运动速度
        self._right_arm_max_vel = torch.tensor(
            self.cfg.right_arm_motion.max_vel,
            dtype=torch.float,
            device=self.device,
        )
        # The runner updates this scalar once per training iteration. Keeping
        # it as a Python scalar avoids introducing a CPU tensor into the
        # batched CUDA trajectory calculations below.
        self.arm_curriculum_factor = 0.0
        # 腰部关节索引
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
            preserve_order=True,
        )

        # 策略真正控制的关节：腿 + 腰
        self.control_joint_ids = torch.cat(
            [
                torch.as_tensor(self.left_leg_ids, device=self.device),
                torch.as_tensor(self.right_leg_ids, device=self.device),
                torch.as_tensor(self.waist_ids, device=self.device),
            ],
            dim=0,
        ).long()

        # self.num_actions = self.robot.data.default_joint_pos.shape[1]  # 获取机器人的动作维度（关节数量），这是策略网络输出层的维度。
        self.num_actions = len(self.control_joint_ids)  # 获取机器人的动作维度（关节数量），这是策略网络输出层的维度。 包括两条腿，一个腰部
        self._validate_right_arm_motion_setup()
        self._right_arm_velocity_scale = (
            self._right_arm_max_vel / self._right_arm_max_vel.max()
        )
        self.update_arm_curriculum()
        
        self.clip_actions = self.cfg.normalization.clip_actions   # 读取 “动作裁剪阈值”，限制策略网络输出的动作范围，避免动作过大导致机器人关节损坏 / 物理仿真崩溃。
        self.clip_obs = self.cfg.normalization.clip_observations  # 读取 “观测裁剪阈值”，限制机器人观测数据的范围，避免异常值（如传感器故障、物理抖动）导致网络训练不稳定

        self.action_scale = self.cfg.robot.action_scale  # 把策略网络输出的 “归一化动作” 映射到机器人实际能执行的控制范围；
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
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

    def _validate_right_arm_motion_setup(self):
        """Validate task-local action partitioning and trajectory settings."""
        if self.num_actions != 15:
            raise RuntimeError(
                f"g1_squart policy must control exactly 15 joints, got {self.num_actions}."
            )
        if self.num_right_arm_joints != 7:
            raise RuntimeError(
                "g1_squart dynamic right arm requires exactly 7 joints, "
                f"got {self.num_right_arm_joints}."
            )
        if torch.isin(self.control_joint_ids, self.right_arm_ids).any().item():
            raise RuntimeError(
                "g1_squart control_joint_ids and right_arm_ids must not overlap."
            )

        arm_cfg = self.cfg.right_arm_motion
        if self._right_arm_max_vel.shape != (self.num_right_arm_joints,):
            raise ValueError(
                "right_arm_motion.max_vel must contain one value for each of "
                f"the 7 right-arm joints, got {len(arm_cfg.max_vel)}."
            )
        if not torch.all(self._right_arm_max_vel > 0.0).item():
            raise ValueError("right_arm_motion.max_vel values must be positive.")
        if torch.unique(self._right_arm_max_vel).numel() == 1:
            raise ValueError(
                "right_arm_motion.max_vel must define different velocities "
                "for the right-arm joints."
            )
        if not 0.0 <= arm_cfg.range_fraction <= 1.0:
            raise ValueError("right_arm_motion.range_fraction must be in [0, 1].")

        if arm_cfg.curriculum_end_iteration <= 0:
            raise ValueError(
                "right_arm_motion.curriculum_end_iteration must be positive."
            )
        if not (
            0.0
            <= arm_cfg.min_range_fraction
            <= arm_cfg.max_range_fraction
            <= 1.0
        ):
            raise ValueError(
                "right_arm_motion curriculum range fractions must be ordered "
                "within [0, 1]."
            )
        if (
            arm_cfg.min_max_vel <= 0.0
            or arm_cfg.min_max_vel > arm_cfg.max_max_vel
        ):
            raise ValueError(
                "right_arm_motion curriculum max velocities must be positive "
                "and ordered."
            )
        if (
            arm_cfg.min_hold_time < 0.0
            or arm_cfg.min_hold_time > arm_cfg.max_hold_time
        ):
            raise ValueError(
                "right_arm_motion curriculum hold times must be non-negative "
                "and ordered."
            )

        speed_lower, speed_upper = arm_cfg.speed_scale_range
        if speed_lower <= 0.0 or speed_lower > speed_upper:
            raise ValueError(
                "right_arm_motion.speed_scale_range must be positive and ordered."
            )
        hold_lower, hold_upper = arm_cfg.hold_time_range
        if hold_lower < 0.0 or hold_lower > hold_upper:
            raise ValueError(
                "right_arm_motion.hold_time_range must be non-negative and ordered."
            )
        if arm_cfg.min_duration <= 0.0:
            raise ValueError("right_arm_motion.min_duration must be positive.")

    def set_training_iteration(self, iteration: int):
        """Receive the current PPO iteration from the task's runner."""
        self.current_iteration = int(iteration)
        self._apply_com_support_margin_curriculum()
        self.update_arm_curriculum()

    def _validate_com_support_margin_curriculum(self):
        """Validate the task-local COM support-margin curriculum settings."""
        if self.cfg.com_support_margin_start_iteration < 0:
            raise ValueError(
                "com_support_margin_start_iteration must be non-negative."
            )
        if self.cfg.com_support_margin_ramp_duration <= 0:
            raise ValueError(
                "com_support_margin_ramp_duration must be positive."
            )

    def _apply_com_support_margin_curriculum(self):
        """Update the RewardManager-cached COM-margin term weight."""
        start_iteration = self.cfg.com_support_margin_start_iteration
        ramp_duration = self.cfg.com_support_margin_ramp_duration
        if self.current_iteration < start_iteration:
            factor = 0.0
        elif self.current_iteration < start_iteration + ramp_duration:
            factor = (self.current_iteration - start_iteration) / ramp_duration
        else:
            factor = 1.0

        self.com_margin_factor = float(factor)
        term_cfg = self.reward_manager.get_term_cfg("com_support_margin")
        term_cfg.weight = self._com_support_margin_weight * self.com_margin_factor
        self.reward_manager.set_term_cfg("com_support_margin", term_cfg)

    def update_arm_curriculum(self):
        """Update the task-local right-arm difficulty from training progress."""
        arm_cfg = self.cfg.right_arm_motion
        if not arm_cfg.curriculum_enable:
            self.arm_curriculum_factor = 1.0
            return

        difficulty = float(self.current_iteration) / float(
            arm_cfg.curriculum_end_iteration
        )
        self.arm_curriculum_factor = max(0.0, min(difficulty, 1.0))

    def sample_new_right_arm_motion(self, env_ids: torch.Tensor):
        """Sample independent, velocity-limited right-arm motions for env_ids."""
        if env_ids.numel() == 0:
            return
        if not self.cfg.right_arm_motion.enable:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        current_q = self.robot.data.joint_pos[env_ids[:, None], self.right_arm_ids]
        default_q = self.robot.data.default_joint_pos[
            env_ids[:, None], self.right_arm_ids
        ]
        soft_limits = self.robot.data.soft_joint_pos_limits[
            env_ids[:, None], self.right_arm_ids
        ]

        arm_cfg = self.cfg.right_arm_motion
        difficulty = self.arm_curriculum_factor
        if arm_cfg.curriculum_enable:
            range_fraction = (
                arm_cfg.min_range_fraction
                + difficulty
                * (arm_cfg.max_range_fraction - arm_cfg.min_range_fraction)
            )
        else:
            range_fraction = arm_cfg.range_fraction

        sample_lower = default_q + range_fraction * (
            soft_limits[..., 0] - default_q
        )
        sample_upper = default_q + range_fraction * (
            soft_limits[..., 1] - default_q
        )
        random_values = torch.rand(
            (env_ids.numel(), self.num_right_arm_joints),
            dtype=torch.float,
            device=self.device,
        )
        target_q = sample_lower + random_values * (sample_upper - sample_lower)

        speed_lower, speed_upper = arm_cfg.speed_scale_range
        speed_scale = speed_lower + torch.rand(
            (env_ids.numel(), 1), dtype=torch.float, device=self.device
        ) * (speed_upper - speed_lower)
        if arm_cfg.curriculum_enable:
            max_vel = (
                arm_cfg.min_max_vel
                + difficulty * (arm_cfg.max_max_vel - arm_cfg.min_max_vel)
            )
            scheduled_max_vel = (
                max_vel * self._right_arm_velocity_scale.unsqueeze(0)
            )
        else:
            scheduled_max_vel = self._right_arm_max_vel.unsqueeze(0)
        effective_max_vel = scheduled_max_vel * speed_scale

        delta_q = torch.abs(target_q - current_q)
        required_duration_per_joint = 1.875 * delta_q / effective_max_vel
        duration = required_duration_per_joint.max(dim=1).values.clamp_min(
            arm_cfg.min_duration
        )

        if arm_cfg.curriculum_enable:
            hold_time = (
                arm_cfg.max_hold_time
                - difficulty * (arm_cfg.max_hold_time - arm_cfg.min_hold_time)
            )
            hold_duration = torch.full(
                (env_ids.numel(),),
                hold_time,
                dtype=torch.float,
                device=self.device,
            )
        else:
            hold_lower, hold_upper = arm_cfg.hold_time_range
            hold_duration = hold_lower + torch.rand(
                env_ids.numel(), dtype=torch.float, device=self.device
            ) * (hold_upper - hold_lower)

        self.arm_motion_start_q[env_ids] = current_q
        self.arm_motion_target_q[env_ids] = target_q
        self.arm_q_des[env_ids] = current_q
        self.arm_dq_des[env_ids] = 0.0
        self.arm_motion_elapsed[env_ids] = 0.0
        self.arm_motion_duration[env_ids] = duration
        self.arm_motion_hold_duration[env_ids] = hold_duration

        if self.cfg.right_arm_motion.debug_checks:
            peak_velocity = 1.875 * delta_q / duration.unsqueeze(1)
            tolerance = 1.0e-6
            assert torch.all(peak_velocity <= effective_max_vel + tolerance), (
                "Minimum-jerk right-arm duration violates configured max velocity."
            )

    def update_right_arm_motion(self):
        """Advance right-arm minimum-jerk commands at the RL control rate."""
        if not self.cfg.right_arm_motion.enable:
            return

        finished = self.arm_motion_elapsed >= (
            self.arm_motion_duration + self.arm_motion_hold_duration
        )
        finished_env_ids = finished.nonzero(as_tuple=False).flatten()
        self.sample_new_right_arm_motion(finished_env_ids)

        moving = self.arm_motion_elapsed < self.arm_motion_duration
        duration = self.arm_motion_duration.clamp_min(
            self.cfg.right_arm_motion.min_duration
        )
        u = torch.clamp(self.arm_motion_elapsed / duration, 0.0, 1.0)
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        ds_du = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
        delta_q = self.arm_motion_target_q - self.arm_motion_start_q
        moving_q_des = self.arm_motion_start_q + s.unsqueeze(1) * delta_q
        moving_dq_des = ds_du.unsqueeze(1) * delta_q / duration.unsqueeze(1)

        self.arm_q_des.copy_(
            torch.where(moving.unsqueeze(1), moving_q_des, self.arm_motion_target_q)
        )
        self.arm_dq_des.copy_(
            torch.where(
                moving.unsqueeze(1),
                moving_dq_des,
                torch.zeros_like(moving_dq_des),
            )
        )
        self.arm_motion_elapsed.add_(self.step_dt)

        if self.cfg.right_arm_motion.debug_checks:
            assert torch.isfinite(self.arm_q_des).all(), (
                "arm_q_des contains NaN or Inf."
            )
            assert torch.isfinite(self.arm_dq_des).all(), (
                "arm_dq_des contains NaN or Inf."
            )

    def _get_right_arm_statistics(self) -> dict[str, torch.Tensor]:
        """Compute lightweight step statistics without changing rewards."""
        actual_qd = self.robot.data.joint_vel[:, self.right_arm_ids]
        return {
            "right_arm/mean_abs_qd": actual_qd.abs().mean(),
            "right_arm/max_abs_qd": actual_qd.abs().max(),
            "right_arm/mean_command_qd": self.arm_dq_des.abs().mean(),
            "right_arm/max_command_qd": self.arm_dq_des.abs().max(),
        }

    def compute_current_observations(self):
        """Compute one-frame actor and critic observations.

        Actor observation stays deployment-friendly:
            root angular velocity                  3
            projected gravity                     3
            height command                        1
            joint position error                 29
            joint velocity error                 29
            previous lower-body action           15
                                                  --
                                                  80

        Critic receives additional privileged information:
            root linear velocity                  3
            left/right foot contact               2
            root height above support plane       1
            whole-body COM in support frame       3
            whole-body COM velocity               3
            right-arm desired joint position      7
            right-arm desired joint velocity      7

        Therefore critic single-frame dimension is 106.
        """
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # =========================================================
        # Actor observations
        # =========================================================
        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command

        joint_pos = (
            robot.data.joint_pos
            - robot.data.default_joint_pos
        )
        joint_vel = (
            robot.data.joint_vel
            - robot.data.default_joint_vel
        )

        action = (
            self.action_buffer
            ._circular_buffer
            .buffer[:, -1, :]
        )

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,                       # 3
                projected_gravity * self.obs_scales.projected_gravity, # 3
                command * self.obs_scales.commands,                     # 1
                joint_pos * self.obs_scales.joint_pos,                  # 29
                joint_vel * self.obs_scales.joint_vel,                  # 29
                action * self.obs_scales.actions,                       # 15
            ],
            dim=-1,
        )

        # =========================================================
        # Critic privileged observations
        # =========================================================
        root_lin_vel = robot.data.root_lin_vel_b

        feet_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces[
                        :,
                        :,
                        self.feet_cfg.body_ids,
                    ],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 0.5
        ).float()

        # Whole-body COM position and velocity expressed in the
        # current double-support frame.
        com_pos_support, com_vel_support = (
            self.get_com_state_in_support_frame()
        )

        # Root height relative to the current support plane rather than
        # an absolute world-z coordinate.
        support_height = (
            self.foot_support_corners_w[..., 2]
            .mean(dim=(1, 2))
        )
        root_height_support = (
            robot.data.root_pose_w[:, 2]
            - support_height
        ).unsqueeze(-1)

        # Give the critic the upcoming right-arm disturbance command.
        # Keep these quantities out of the actor for now.
        right_arm_default_q = robot.data.default_joint_pos[
            :,
            self.right_arm_ids,
        ]
        right_arm_q_des_rel = (
            self.arm_q_des - right_arm_default_q
        ) * self.obs_scales.joint_pos
        right_arm_dq_des = (
            self.arm_dq_des * self.obs_scales.joint_vel
        )

        current_critic_obs = torch.cat(
            [
                current_actor_obs,                         # 80
                root_lin_vel * self.obs_scales.lin_vel,   # 3
                feet_contact,                             # 2
                root_height_support,                      # 1
                com_pos_support,                          # 3
                com_vel_support,                          # 3
                right_arm_q_des_rel,                      # 7
                right_arm_dq_des,                         # 7
            ],
            dim=-1,
        )

        if self.cfg.right_arm_motion.debug_checks:
            assert current_actor_obs.shape[-1] == 80, (
                f"Expected actor single-frame dim 80, got "
                f"{current_actor_obs.shape[-1]}."
            )
            assert current_critic_obs.shape[-1] == 106, (
                f"Expected critic single-frame dim 106, got "
                f"{current_critic_obs.shape[-1]}."
            )

        return current_actor_obs, current_critic_obs

    def get_com_state_in_support_frame(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return whole-body COM position/velocity in the support frame.

        The support frame is defined from the current two foot support
        rectangles:
            x: average forward direction of the two feet
            y: horizontal axis perpendicular to x
            z: world vertical

        Returns:
            com_pos_support:
                shape = (num_envs, 3)
                [com_x, com_y, com_z]

            com_vel_support:
                shape = (num_envs, 3)
                [com_vx, com_vy, com_vz]
        """

        # shape: (num_envs, 2, 4, 3)
        foot_support_points_w = self.foot_support_corners_w
        foot_support_xy = foot_support_points_w[..., :2]

        # Each foot's front/rear center in world XY.
        foot_front_center_xy = (
            foot_support_xy[:, :, 0:2, :]
            .mean(dim=2)
        )
        foot_rear_center_xy = (
            foot_support_xy[:, :, 2:4, :]
            .mean(dim=2)
        )

        # Average foot-forward direction.
        support_x_axis_xy = (
            foot_front_center_xy
            - foot_rear_center_xy
        ).mean(dim=1)
        support_x_axis_xy = (
            support_x_axis_xy
            / torch.linalg.vector_norm(
                support_x_axis_xy,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-6)
        )

        # Horizontal lateral axis.
        support_y_axis_xy = torch.stack(
            [
                -support_x_axis_xy[:, 1],
                support_x_axis_xy[:, 0],
            ],
            dim=-1,
        )

        # Support origin: mean of all eight foot support points.
        support_origin_xy = foot_support_xy.mean(dim=(1, 2))
        support_origin_z = (
            foot_support_points_w[..., 2]
            .mean(dim=(1, 2))
        )

        # =========================================================
        # COM position
        # =========================================================
        com_xy_relative = (
            self.whole_body_com_pos_w[:, :2]
            - support_origin_xy
        )

        com_x = torch.sum(
            com_xy_relative * support_x_axis_xy,
            dim=-1,
        )
        com_y = torch.sum(
            com_xy_relative * support_y_axis_xy,
            dim=-1,
        )
        com_z = (
            self.whole_body_com_pos_w[:, 2]
            - support_origin_z
        )

        com_pos_support = torch.stack(
            [com_x, com_y, com_z],
            dim=-1,
        )

        # =========================================================
        # COM velocity
        # =========================================================
        com_vel_xy_w = self.whole_body_com_vel_w[:, :2]

        com_vx = torch.sum(
            com_vel_xy_w * support_x_axis_xy,
            dim=-1,
        )
        com_vy = torch.sum(
            com_vel_xy_w * support_y_axis_xy,
            dim=-1,
        )
        com_vz = self.whole_body_com_vel_w[:, 2]

        com_vel_support = torch.stack(
            [com_vx, com_vy, com_vz],
            dim=-1,
        )

        return com_pos_support, com_vel_support

    def obs_noisy_vec_and_buffer(self):
        if self.add_noise:
            current_actor_obs, _ = self.compute_current_observations()

            noise_obs_vec = torch.zeros_like(current_actor_obs[0])

            num_joints = self.robot.data.default_joint_pos.shape[1]  # 29
            num_actions = self.num_actions                           # 15

            idx = 0

            # ang_vel: 3
            noise_obs_vec[idx:idx + 3] = (
                self.obs_scales.ang_vel
                * self.noisy.ang_vel
            )
            idx += 3

            # projected_gravity: 3
            noise_obs_vec[idx:idx + 3] = (
                self.obs_scales.projected_gravity
                * self.noisy.projected_gravity
            )
            idx += 3

            # height command: 1, no noise
            noise_obs_vec[idx:idx + 1] = 0.0
            idx += 1

            # joint_pos: 29
            noise_obs_vec[idx:idx + num_joints] = (
                self.obs_scales.joint_pos
                * self.noisy.joint_pos
            )
            idx += num_joints

            # joint_vel: 29
            noise_obs_vec[idx:idx + num_joints] = (
                self.obs_scales.joint_vel
                * self.noisy.joint_vel
            )
            idx += num_joints

            # last_action: 15, no noise
            noise_obs_vec[idx:idx + num_actions] = 0.0
            idx += num_actions

            if idx != current_actor_obs.shape[-1]:
                raise RuntimeError(
                    "Actor observation/noise dimension mismatch: "
                    f"noise layout={idx}, "
                    f"actor dim={current_actor_obs.shape[-1]}."
                )

            self.noise_obs_vec = noise_obs_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = (
                    self.noisy.height_scan
                    * self.obs_scales.height_scan
                )
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
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
    
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()


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

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sample_new_right_arm_motion(env_ids)
        current_corners_w = self.get_foot_support_corners_w()
        self.foot_support_corners_w[env_ids] = current_corners_w[env_ids]
        
        # 如果 reset event 会随机质量，这一步非常重要：
        # 重新读取当前 PhysX 中的实际刚体质量
        self.update_body_mass_cache(env_ids)

        current_com_pos_w, current_com_vel_w = (
            self.get_whole_body_com_state_w(
                env_ids=env_ids,
            )
        )

        self.whole_body_com_pos_w[env_ids] = current_com_pos_w
        self.whole_body_com_vel_w[env_ids] = current_com_vel_w
    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(
            delayed_actions,
            -self.clip_actions,
            self.clip_actions,
        ).to(self.device)

        # RL controls only 15 joints: two legs + waist.
        processed_actions = (
            self.action * self.action_scale
            + self.robot.data.default_joint_pos[
                :,
                self.control_joint_ids,
            ]
        )

        # Right-arm planner runs once per RL/control step.
        self.update_right_arm_motion()

        # Keep the left arm explicitly fixed at the nominal pose so the
        # right arm remains the only arm-side disturbance in this stage.
        left_arm_q_des = self.robot.data.default_joint_pos[
            :,
            self.left_arm_ids,
        ]
        left_arm_dq_des = torch.zeros_like(left_arm_q_des)

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

            # -----------------------------------------------------
            # Legs + waist: lower-body policy
            # -----------------------------------------------------
            self.robot.set_joint_position_target(
                processed_actions,
                joint_ids=self.control_joint_ids,
            )

            # -----------------------------------------------------
            # Left arm: fixed nominal pose
            # -----------------------------------------------------
            self.robot.set_joint_position_target(
                left_arm_q_des,
                joint_ids=self.left_arm_ids,
            )
            self.robot.set_joint_velocity_target(
                left_arm_dq_des,
                joint_ids=self.left_arm_ids,
            )

            # -----------------------------------------------------
            # Right arm: procedural dynamic trajectory
            # -----------------------------------------------------
            if self.cfg.right_arm_motion.enable:
                self.robot.set_joint_position_target(
                    self.arm_q_des,
                    joint_ids=self.right_arm_ids,
                )
                self.robot.set_joint_velocity_target(
                    self.arm_dq_des,
                    joint_ids=self.right_arm_ids,
                )

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[
                    :,
                    self.feet_cfg.body_ids,
                    :3,
                ],
                dim=-1,
            )
            self.avg_feet_speed_per_step += torch.norm(
                self.robot.data.body_lin_vel_w[
                    :,
                    self.ankle_link_ids,
                    :,
                ],
                dim=-1,
            )

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1

        self.command_generator.compute(self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="interval",
                dt=self.step_dt,
            )

        self.reset_buf, self.time_out_buf = self.check_reset()

        # =========================================================
        # Refresh support geometry and COM state BEFORE rewards and
        # observations consume them.
        # =========================================================
        current_corners_w = self.get_foot_support_corners_w()
        self.foot_support_corners_w.copy_(current_corners_w)

        current_com_pos_w, current_com_vel_w = (
            self.get_whole_body_com_state_w()
        )
        self.whole_body_com_pos_w.copy_(current_com_pos_w)
        self.whole_body_com_vel_w.copy_(current_com_vel_w)

        right_arm_statistics = self._get_right_arm_statistics()

        reward_buf = self.reward_manager.compute(self.step_dt)

        self.reset_env_ids = (
            self.reset_buf
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.reset(self.reset_env_ids)

        self.extras.setdefault("log", {}).update(
            right_arm_statistics
        )

        actor_obs, critic_obs = self.compute_observations()

        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = critic_obs

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
    
    def get_foot_support_corners_w(self) -> torch.Tensor:
        """将左右脚底支撑点从脚踝局部坐标系转换到世界坐标系。

        Returns:
            corners_w:
                shape = (num_envs, 2, 4, 3)

                dim=1:
                    0：左脚
                    1：右脚

                dim=2:
                    每只脚的 4 个支撑点

                dim=3:
                    世界坐标 [x, y, z]
        """

        # 局部脚底支撑点
        # shape: (4, 3)
        corners_b = self.foot_support_corners_b

        # 左右脚踝 link 在世界坐标系中的位置
        # shape: (num_envs, 2, 3)
        foot_pos_w = self.robot.data.body_link_pos_w[
            :, self.ankle_link_ids, :
        ]

        # 左右脚踝 link 在世界坐标系中的姿态
        # shape: (num_envs, 2, 4)
        foot_quat_w = self.robot.data.body_link_quat_w[
            :, self.ankle_link_ids, :
        ]

        num_envs = foot_pos_w.shape[0]
        num_feet = foot_pos_w.shape[1]       # 2
        num_corners = corners_b.shape[0]     # 4

        # 将 4 个局部点复制到所有环境、两只脚
        # shape: (num_envs, 2, 4, 3)
        corners_b_expanded = corners_b.view(
            1, 1, num_corners, 3
        ).expand(
            num_envs, num_feet, num_corners, 3
        )

        # 每只脚的四元数复制给它的 4 个角点
        # shape: (num_envs, 2, 4, 4)
        foot_quat_expanded = foot_quat_w.unsqueeze(2).expand(
            num_envs, num_feet, num_corners, 4
        )

        # 局部点经过脚踝姿态旋转到世界坐标方向
        rotated_corners_w = quat_apply(
            foot_quat_expanded.reshape(-1, 4),
            corners_b_expanded.reshape(-1, 3),
        ).reshape(
            num_envs, num_feet, num_corners, 3
        )

        # 加上左右脚踝 link 的世界坐标位置
        corners_w = (
            rotated_corners_w
            + foot_pos_w.unsqueeze(2)
        )

        return corners_w

    def update_body_mass_cache(
        self,
        env_ids: torch.Tensor | None = None,
    ):
        """从 PhysX 中读取机器人所有刚体的实际质量。

        Args:
            env_ids:
                None 时更新全部环境；
                否则只更新指定环境。
        """

        # shape: (num_envs, num_bodies)
        current_masses = (
            self.robot.root_physx_view
            .get_masses()
            .to(device=self.device, dtype=torch.float)
        )

        if env_ids is None:
            self.body_masses.copy_(current_masses)
        else:
            self.body_masses[env_ids] = current_masses[env_ids]

        # 更新总质量
        self.total_body_mass[:] = self.body_masses.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
    def get_whole_body_com_state_w(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算整机质心的位置和线速度（世界坐标系）。

        Returns:
            whole_body_com_pos_w:
                shape = (N, 3)

            whole_body_com_vel_w:
                shape = (N, 3)

            N = num_envs，当 env_ids is None
            N = len(env_ids)，当指定 env_ids
        """

        # ---------------------------------------------------------
        # 1. 每个刚体自身质心的位置
        # shape: (num_envs, num_bodies, 3)
        # ---------------------------------------------------------
        body_com_pos_w = self.robot.data.body_com_pos_w

        # ---------------------------------------------------------
        # 2. 每个刚体自身质心的线速度
        #
        # body_com_vel_w:
        # shape = (num_envs, num_bodies, 6)
        #
        # [:, :, :3] 为线速度 xyz
        # ---------------------------------------------------------
        body_com_lin_vel_w = self.robot.data.body_com_vel_w[..., :3]

        # ---------------------------------------------------------
        # 3. 质量
        # shape: (num_envs, num_bodies)
        # ---------------------------------------------------------
        body_masses = self.body_masses

        # ---------------------------------------------------------
        # 4. 如果只计算部分环境，则全部一起索引
        # ---------------------------------------------------------
        if env_ids is not None:
            body_com_pos_w = body_com_pos_w[env_ids]
            body_com_lin_vel_w = body_com_lin_vel_w[env_ids]
            body_masses = body_masses[env_ids]

        # ---------------------------------------------------------
        # 5. 总质量
        # shape: (N, 1)
        # ---------------------------------------------------------
        total_mass = body_masses.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)

        # ---------------------------------------------------------
        # 6. 整机 COM position
        #
        # p_com = sum(m_i * p_i) / sum(m_i)
        # ---------------------------------------------------------
        whole_body_com_pos_w = (
            body_com_pos_w
            * body_masses.unsqueeze(-1)
        ).sum(dim=1) / total_mass

        # ---------------------------------------------------------
        # 7. 整机 COM velocity
        #
        # v_com = sum(m_i * v_i) / sum(m_i)
        # ---------------------------------------------------------
        whole_body_com_vel_w = (
            body_com_lin_vel_w
            * body_masses.unsqueeze(-1)
        ).sum(dim=1) / total_mass

        return whole_body_com_pos_w, whole_body_com_vel_w

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()

        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = critic_obs

        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore
 
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
