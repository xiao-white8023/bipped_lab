# ============================================
# G1 23dof high-level LP / VP selector
# ============================================
"""5 Hz 离散 PPO 独立配置；高层环境、网络和 runner 在后续步骤接入。"""

import math
import os
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.g1.g1_29 import G1_23CFG, G1_23DOF_LINKS
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CameraCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    Privileged_Info,
    RobotCfg,
    SimCfg,
    left_feet_ray_caster_cfg,
    right_feet_ray_caster_cfg,
)
from legged_lab.perception.lp_height import LPHeightCfg, PgttNoiseCfg, resolve_pgtt_height_cfg
from legged_lab.high_level.high_level_fault_config import (
    FaultScheduleCfg, HighLevelFaultCfg, LpFaultCfg, VpFaultCfg,
)
from legged_lab.sensors.camera_noise import DistanceDependentGaussianNoiseCfg
from legged_lab.sensors.foot_sole_pattern import FootSolePatternCfg
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg, get_link_prim_targets
from legged_lab.terrains import ROUGH_TERRAINS_CFG

LEGGED_LAB_ROOT = os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab")


@configclass
class FrozenRENetCfg:
    # 复用 selector.frozen_policy 的精确重建/parity 检验，避免用当前
    # multi-head agent 重建旧 checkpoint，也避免将 CPU trace 直接搬到 CUDA。
    loader: str = "validated_checkpoint"
    checkpoint_path: str = os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "logs/g1_renet/2026-09-04_14-11-25/model_60000.pt",
    )
    reference_policy_path: str = os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "logs/g1_renet/2026-09-04_14-11-25/exported/policy.pt",
    )
    freeze: bool = True
    deterministic: bool = True
    # 原策略的观测缩放、归一化及 mask 约定保持不变。
    validate_observation_contract: bool = True


@configclass
class SelectorRENetCfg:
    # 底层仅使用 vp/lp 初始化；后续高层环境接管 renet_estimator_mask。
    mask_mode: str = "vp"
    lp_height_source: str = "pgtt_full_atten"
    domain_rand_start_iteration: int = 0


@configclass
class SelectorControlCfg:
    frequency: float = 5.0
    # VP=0、LP=1；这是一个具有两个取值的离散动作，不是两个关节动作。
    mode_names: tuple[str, str] = ("vp", "lp")
    initial_mode: int = 0
    num_modes: int = 2
    action_dim: int = 1
    # 后续高层环境拥有模式控制权；底层环境只用 vp/lp 完成初始化。
    mode_control: str = "external"

    # 每个 5 Hz 决策立即接受模式变化，直接执行新模式动作，不做动作混合。
    switch_confirmations: int = 1
    min_dwell_s: float = 0.0
    # 每个环境保持初始地形等级；episode reset 不调用 update_terrain_levels。
    terrain_curriculum: bool = False

    # 由环境 sim.dt * sim.decimation 解析，不修改底层物理 decimation。
    low_level_dt: float = 0.02
    low_level_steps_per_decision: int = 10

    # 六项独立奖励已在 50 Hz 下乘 dt，高层只求和，不再缩放时间。
    # 默认每条 transition 最多累计十步，首次终止步计入后立即停止；不混入 reset 后数据。
    aggregation: str = "sum"
    stop_on_done: bool = True
    bootstrap_timeouts: bool = True
    # 真实失败的后继价值为 0；超时使用 reset 前的末端观测估值。
    timeout_value_source: str = "terminal_observation"


@configclass
class SelectorObservationCfg:
    # 本体缓存仍按底层 50 Hz 更新，高层每 0.2 s 读取最近十帧。
    # 每帧已含速度命令及前一步实际关节动作，不重复拼接。
    proprio_history_frames: int = 10
    proprio_frame_dim: int = 78
    depth_history_frames: int = 2
    height_history_frames: int = 1
    use_depth_valid_support: bool = True
    use_height_valid_mask: bool = True

    # 有效性必须在填补/模糊/裁剪之前保留；可为 [0, 1] 的有效支撑比例。
    # 模式、有效性及固定缩放图像均不使用整向量经验归一化。
    sensor_age_clip_s: float = 2.0
    switch_time_clip_s: float = 2.0
    state_names: tuple[str, ...] = (
        "mode_vp", "mode_lp",
        "vp_old_valid_ratio", "vp_new_valid_ratio", "lp_valid_ratio",
        "vp_source_age", "lp_source_age", "time_since_switch",
    )

    # 独立高层 critic 可见当前仿真真值；actor 不接收这些特权信息。
    critic_use_privileged: bool = True
    critic_base_lin_vel_dim: int = 3
    critic_feet_contact_dim: int = 2
    critic_feet_height_dim: int = 2
    critic_use_clean_height: bool = True

    # 以下尺寸由环境配置解析；仍须与所加载冻结策略的输入契约一致。
    depth_shape: tuple[int, int, int] = (2, 18, 32)
    height_shape: tuple[int, int, int] = (1, 11, 17)
    proprio_dim: int = 780
    depth_input_channels: int = 4
    height_input_channels: int = 2
    state_dim: int = 8
    actor_obs_dim: int = 3466
    critic_privileged_dim: int = 194
    critic_obs_dim: int = 3660


@configclass
class SelectorFaultCfg:
    enable: bool = True
    # 正常/仅 LP/仅 VP/双路 = 30/25/25/20%，按新帧 10 Hz 显式安排事件。
    schedule: FaultScheduleCfg = FaultScheduleCfg()
    # LP 全部 11 类；参数范围均匀采样，与持续时间独立，不使用轻中重等级。
    lp: LpFaultCfg = LpFaultCfg()
    # VP 全部 8 类 Film 结构故障；不使用 Film 触发概率和回报课程。
    vp: VpFaultCfg = VpFaultCfg()


@configclass
class Reward:
    # 独立高层奖励：两种模式共用同一评价，不混入底层其他奖励或 AMP 奖励。
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

    # RewardManager 按底层 dt=0.02 计算，真实失败一次实际扣 4.0；超时不计失败。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    alive = RewTerm(func=mdp.alive, weight=0.15)
    # 只在实际切换的底层步计一次，权重 -1.0 对应每次实际扣 0.02。
    mode_switch_penalty = RewTerm(func=mdp.selector_mode_switch, weight=-1.0)
    # 实际执行并写入底层缓存的相邻 23 维归一化动作之差，不是高层模式编号之差。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class G1SelectorEnvCfg:
    seed: int = 42
    # 底层观测和动作显示兼容字段；高层训练不使用 AMP 判别器。
    amp_observation_dim: int = 58
    lp_height = LPHeightCfg(noise=PgttNoiseCfg(enable=False, start_iteration=0))
    amp_motion_files_display = [
        f"{LEGGED_LAB_ROOT}/envs/g1/datasets/make_g1_23dof/training/visualization/walk_turns_40s.txt"
    ]
    device: str = "cuda:0"
    # 原底层感知噪声关闭；仅由 high_level_faults 管线统一注入选定的故障。
    # 保留字段供底层环境兼容读取，不把它们作为故障启动开关。
    depth_noise_start_iteration: int = 0
    depth_noise_force_active: bool = False  # 显式噪声回放评估可覆盖训练阶段。
    # 每个环境每次采集独立触发；0 关闭模糊，1 每帧模糊。跟随 add_camera_noise 开关。
    depth_blur_probability: float = 0.0
    depth_noise: DistanceDependentGaussianNoiseCfg = DistanceDependentGaussianNoiseCfg(
        near_std=0.0,
        far_std=0.0,
        distance_exponent=2.0,
    )
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        # 与当前 RENet 场景保持相同的初始并行规模。
        num_envs=128,
        env_spacing=2.5,
        robot=G1_23CFG,
        terrain_type="generator",
        # 保留按行生成的难度分层；训练中的升降级由 selector.terrain_curriculum 关闭。
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
        lidar=None,
        camera=CameraCfg(
            add_camera_noise=False,
            add_camera=True,
            camera=GroupedRayCasterCameraCfg(
                prim_path="/World/envs/env_.*/Robot/torso_link",
                pattern_cfg=PinholeCameraPatternCfg(
                    focal_length=1.0,
                    horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
                    vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
                    width=64,
                    height=36,
                ),
                offset=GroupedRayCasterCameraCfg.OffsetCfg(
                    pos=(0.0576235, 0.01753, 0.42987),
                    rot=(0.9149595, 0.0, 0.4035447, 0.0),
                    convention="world",
                ),
                mesh_prim_paths=["/World/ground"] + get_link_prim_targets(G1_23DOF_LINKS),
                ray_alignment="yaw",
                debug_vis=False,
                data_types=["distance_to_image_plane"],
                update_period=0.02,
                # 保留未命中的 NaN/Inf，先计算支持度，再由统一管线填补、裁剪和归一化。
                depth_clipping_behavior="none",
                min_distance=0.1,
            ),
            depth_max=2.5,
            depth_update_interval=5,
            depth_crop=(18, 0, 16, 16),
            depth_history_frames=2,
        ),
        left_feet_ray_caster=left_feet_ray_caster_cfg(
            add_left_feet_ray_caster=True,
            left_feet_ray_caster=RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.025)),
                pattern_cfg=FootSolePatternCfg(
                    mesh_path=f"{LEGGED_LAB_ROOT}/assets/g1/g1_23dof/meshes/left_ankle_roll_link.STL",
                    resolution=0.01,
                    edge_margin=0.002,
                ),
                ray_alignment="base",
                update_period=0.02,
                debug_vis=False,
            ),
        ),
        right_feet_ray_caster=right_feet_ray_caster_cfg(
            add_right_feet_ray_caster=True,
            right_feet_ray_caster=RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.025)),
                pattern_cfg=FootSolePatternCfg(
                    mesh_path=f"{LEGGED_LAB_ROOT}/assets/g1/g1_23dof/meshes/right_ankle_roll_link.STL",
                    resolution=0.01,
                    edge_margin=0.002,
                ),
                ray_alignment="base",
                update_period=0.02,
                debug_vis=False,
            ),
        ),
        privileged_info=Privileged_Info(
            enable_feet_info=False,
            enable_root_height=False,
            enable_feet_contact_force=False,
        ),
    )

    reward = Reward()
    renet = SelectorRENetCfg()

    robot: RobotCfg = RobotCfg(
        # 保持冻结策略的观测布局，底层动作仍为 23 维。
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        depth_history_frames=2,
        depth_max=2.5,
        depth_update_interval=5,
        depth_crop=(18, 0, 16, 16),  # 上、下、左、右裁剪像素数。
        action_scale=0.25,
        terminate_contacts_body_names=[".*torso.*"],
        feet_body_names=["left_ankle_roll.*", "right_ankle_roll.*"],
        limit_angle=0.8,
    )

    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )

    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )

    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )

    domain_rand: DomainRandCfg = DomainRandCfg(
        # 保留底层事件/动作缓存接口；物理参数固定，只保留原有 episode 重置。
        events=EventCfg(
            physics_material=None,
            add_base_mass=None,
            push_robot=None,
            # EventCfg 默认带入六组手臂 PD 增益随机化，需逐项显式关闭。
            right_arm_shoulder_pitch_roll_gains=None,
            right_arm_shoulder_yaw_elbow_gains=None,
            right_arm_wrist_gains=None,
            left_arm_shoulder_pitch_roll_gains=None,
            left_arm_shoulder_yaw_elbow_gains=None,
            left_arm_wrist_gains=None,
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
        ),
        # 延迟关闭，但保留至少两帧动作历史，供 action_rate_l2 和冻结策略使用。
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(
            use_gpu=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_heap_capacity=2**26,
        ),
    )

    low_level: FrozenRENetCfg = FrozenRENetCfg()
    selector: SelectorControlCfg = SelectorControlCfg()
    observation: SelectorObservationCfg = SelectorObservationCfg()
    faults: SelectorFaultCfg = SelectorFaultCfg()

    def __post_init__(self):
        self.resolve_observations()

    def resolve_observations(self):
        """校验多速率配置，并从底层传感器配置解析高层观测尺寸。"""
        control, obs = self.selector, self.observation
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Selector seed must be a nonnegative integer.")
        if not math.isfinite(self.robot.limit_angle) or not 0.0 < self.robot.limit_angle < math.pi:
            raise ValueError("Selector tilt limit must lie between zero and pi radians.")
        if self.domain_rand.action_delay.enable:
            raise ValueError("Selector action delay must remain disabled.")
        action_history = self.domain_rand.action_delay.params
        if (type(action_history.get("max_delay")) is not int or action_history["max_delay"] < 1
                or action_history.get("min_delay") != 0):
            raise ValueError("Selector requires at least two action-history slots and zero action delay.")
        if not math.isfinite(control.frequency) or control.frequency <= 0.0:
            raise ValueError("Selector frequency must be finite and positive.")
        if (control.mode_names != ("vp", "lp") or control.num_modes != 2
                or control.action_dim != 1 or type(control.initial_mode) is not int
                or control.initial_mode not in (0, 1)):
            raise ValueError("Selector requires one discrete action: VP=0, LP=1.")
        # 初始化与命令行覆盖均同步；wrapper 后续必须接管实际 mask 的更新。
        self.renet.mask_mode = control.mode_names[control.initial_mode]
        if control.mode_control != "external":
            raise ValueError("The high-level selector must own mode selection.")
        if not self.low_level.freeze or not self.low_level.deterministic:
            raise ValueError("The low-level RENet must be frozen and deterministic.")
        if (self.low_level.loader != "validated_checkpoint" or not self.low_level.checkpoint_path
                or not self.low_level.reference_policy_path or not self.low_level.validate_observation_contract):
            raise ValueError("Frozen RENet requires a checkpoint, reference policy and contract validation.")

        control.low_level_dt = self.sim.dt * self.sim.decimation
        if not math.isclose(control.low_level_dt, 0.02, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("The frozen low-level controller must run at 50 Hz.")
        ratio = 1.0 / (control.frequency * control.low_level_dt)
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Selector period must be an integer number of low-level steps.")
        control.low_level_steps_per_decision = round(ratio)
        if (type(control.switch_confirmations) is not int or control.switch_confirmations < 1
                or not math.isfinite(control.min_dwell_s) or control.min_dwell_s < 0.0):
            raise ValueError("Selector confirmation or dwell settings are invalid.")
        if control.switch_confirmations != 1 or control.min_dwell_s != 0.0:
            raise ValueError("High-level mode requests must switch immediately without confirmation or dwell.")
        if control.terrain_curriculum is not False:
            raise ValueError("Selector environments must keep their initial terrain levels without a curriculum.")

        if (obs.proprio_frame_dim != 78 or obs.proprio_history_frames != 10
                or self.robot.actor_obs_history_length != obs.proprio_history_frames):
            raise ValueError("Selector proprioception requires ten 78-D low-level frames.")
        camera = self.scene.camera
        if not camera.add_camera or not self.scene.height_scanner.enable_height_scan:
            raise ValueError("Selector training requires both depth and height observations.")
        if camera.camera.depth_clipping_behavior != "none":
            raise ValueError("Preserve raw depth misses before computing selector valid support.")
        if (obs.depth_history_frames != 2 or self.robot.depth_history_frames != 2
                or camera.depth_history_frames != 2 or obs.height_history_frames != 1):
            raise ValueError("Selector requires two depth frames and one height frame.")
        if (tuple(self.robot.depth_crop) != tuple(camera.depth_crop)
                or self.robot.depth_update_interval != camera.depth_update_interval):
            raise ValueError("Robot and camera depth sampling settings must agree.")
        top, bottom, left, right = camera.depth_crop
        if any(type(value) is not int or value < 0 for value in camera.depth_crop):
            raise ValueError("Depth crop must contain four nonnegative integers.")
        pattern = camera.camera.pattern_cfg
        rows, cols = pattern.height - top - bottom, pattern.width - left - right
        if rows <= 0 or cols <= 0:
            raise ValueError("Depth crop must leave a nonempty image.")
        obs.depth_shape = (obs.depth_history_frames, rows, cols)
        height = resolve_pgtt_height_cfg(self.lp_height, self.renet.lp_height_source)
        obs.height_shape = (obs.height_history_frames, *height.shape)
        periods = [height.update_period, camera.depth_update_interval * control.low_level_dt]
        if self.faults.enable:
            periods.append(self.faults.schedule.update_period_s)
        for period in periods:
            sensor_steps = period / control.low_level_dt
            if (not math.isfinite(period) or period <= 0.0
                    or not math.isclose(sensor_steps, round(sensor_steps), rel_tol=0.0, abs_tol=1e-8)):
                raise ValueError("Sensor periods must align with low-level control steps.")

        if (not math.isfinite(obs.sensor_age_clip_s) or obs.sensor_age_clip_s <= 0.0
                or not math.isfinite(obs.switch_time_clip_s) or obs.switch_time_clip_s <= 0.0):
            raise ValueError("Selector age and switch-time scales must be finite and positive.")
        obs.proprio_dim = obs.proprio_history_frames * obs.proprio_frame_dim
        obs.depth_input_channels = obs.depth_history_frames * (1 + int(obs.use_depth_valid_support))
        obs.height_input_channels = obs.height_history_frames * (1 + int(obs.use_height_valid_mask))
        obs.state_dim = len(obs.state_names)
        obs.actor_obs_dim = (
            obs.proprio_dim + obs.depth_input_channels * rows * cols
            + obs.height_input_channels * math.prod(height.shape) + obs.state_dim
        )
        obs.critic_privileged_dim = 0
        if obs.critic_use_privileged:
            obs.critic_privileged_dim = (
                obs.critic_base_lin_vel_dim + obs.critic_feet_contact_dim + obs.critic_feet_height_dim
                + int(obs.critic_use_clean_height) * math.prod(height.shape)
            )
        obs.critic_obs_dim = obs.actor_obs_dim + obs.critic_privileged_dim
        if (control.aggregation != "sum" or not control.stop_on_done or not control.bootstrap_timeouts
                or control.timeout_value_source != "terminal_observation"):
            raise ValueError("Selector rewards require terminal-safe sums and pre-reset timeout values.")
        fault_cfg = HighLevelFaultCfg.from_config(self.faults)
        if self.faults.enable and (
            tuple(height.shape) != (11, 17) or (rows, cols) != (18, 32)
            or not math.isclose(height.update_period, fault_cfg.schedule.update_period_s, abs_tol=1e-9)
            or not math.isclose(camera.depth_update_interval * control.low_level_dt,
                                fault_cfg.schedule.update_period_s, abs_tol=1e-9)
            or camera.camera.min_distance != fault_cfg.vp.depth_min_m
            or self.robot.depth_max != fault_cfg.vp.depth_max_m
            or self.lp_height.invalid_height != fault_cfg.lp.lp_invalid_height
        ):
            raise ValueError("High-level faults require paired 10 Hz LP 11x17 and metric VP 18x32 packets.")
        if (self.lp_height.noise.enable or camera.add_camera_noise or self.depth_noise_force_active
                or self.depth_blur_probability != 0.0 or self.depth_noise.near_std != 0.0
                or self.depth_noise.far_std != 0.0):
            raise ValueError("Use only the high-level LP/Film fault pipeline; disable legacy perception noise.")


@configclass
class SelectorDepthEncoderCfg:
    input_dim: tuple[int, int] = MISSING
    input_channels: int = MISSING
    output_channels: list[int] = MISSING
    kernel_size: list[int] = MISSING
    stride: list[int] = MISSING
    padding: list[int] = MISSING
    mlp_output_dim: int = MISSING


@configclass
class SelectorActorCriticCfg:
    # 独立 categorical actor，不继承 Gaussian 策略的 std / 连续动作接口。
    class_name: str = "SelectorActorCritic"
    action_distribution: str = "categorical"
    num_modes: int = 2
    critic_output_dim: int = 1
    activation: str = "elu"
    share_encoders: bool = False
    proprio_hidden_dims: list[int] = MISSING
    depth_encoder: SelectorDepthEncoderCfg = MISSING
    height_hidden_dims: list[int] = MISSING
    state_hidden_dims: list[int] = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    # 输入尺寸还会在命令行覆盖后由 configure_selector_agent 同步。
    proprio_input_dim: int = MISSING
    height_input_dim: int = MISSING
    state_input_dim: int = MISSING
    actor_obs_dim: int = MISSING
    critic_obs_dim: int = MISSING
    critic_privileged_dim: int = MISSING


@configclass
class SelectorPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # KL 按分类分布计算，不复用 Gaussian mu/sigma 公式。
    kl_method: str = "categorical"


@configclass
class G1SelectorAgentCfg:
    seed = 42
    device = "cuda:0"
    # 每个环境 24 个高层 transition，正常覆盖 4.8 s / 240 个底层步。
    num_steps_per_env = 24
    max_iterations = 5000
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = SelectorActorCriticCfg(
        class_name="SelectorActorCritic",
        action_distribution="categorical",
        num_modes=2,
        critic_output_dim=1,
        activation="elu",
        share_encoders=False,
        proprio_hidden_dims=[256, 128],
        depth_encoder=SelectorDepthEncoderCfg(
            input_dim=(18, 32),
            input_channels=4,
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            mlp_output_dim=128,
        ),
        height_hidden_dims=[128, 64],
        state_hidden_dims=[64, 32],
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        # 四路特征拼接：128 + 128 + 64 + 32 = 352。
        proprio_input_dim=780,
        height_input_dim=374,
        state_input_dim=8,
        actor_obs_dim=3466,
        critic_obs_dim=3660,
        critic_privileged_dim=194,
    )
    algorithm = SelectorPpoAlgorithmCfg(
        class_name="DiscretePPO",
        kl_method="categorical",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # gamma/lam 的一步是高层 0.2 s，并非底层 0.02 s。
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "SelectorOnPolicyRunner"
    experiment_name = "g1_selector_ppo"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "leggedlab"
    wandb_project = "leggedlab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"


def configure_selector_agent(env_cfg: G1SelectorEnvCfg, agent_cfg: G1SelectorAgentCfg):
    """在命令行覆盖环境参数后，同步网络维度并校验高层 PPO 配置。"""
    env_cfg.seed = agent_cfg.seed
    env_cfg.resolve_observations()
    obs, policy, algorithm = env_cfg.observation, agent_cfg.policy, agent_cfg.algorithm
    policy.proprio_input_dim = obs.proprio_dim
    policy.depth_encoder.input_dim = obs.depth_shape[1:]
    policy.depth_encoder.input_channels = obs.depth_input_channels
    policy.height_input_dim = obs.height_input_channels * math.prod(obs.height_shape[1:])
    policy.state_input_dim = obs.state_dim
    policy.actor_obs_dim = obs.actor_obs_dim
    policy.critic_obs_dim = obs.critic_obs_dim
    policy.critic_privileged_dim = obs.critic_privileged_dim
    if policy.action_distribution != "categorical" or policy.num_modes != 2 or policy.critic_output_dim != 1:
        raise ValueError("Selector policy requires two categorical logits and one critic value.")
    for dims in (policy.proprio_hidden_dims, policy.height_hidden_dims, policy.state_hidden_dims,
                 policy.actor_hidden_dims, policy.critic_hidden_dims):
        if not dims or any(type(value) is not int or value < 1 for value in dims):
            raise ValueError("Selector network dimensions must be positive integers.")
    cnn = policy.depth_encoder
    if not (len(cnn.output_channels) == len(cnn.kernel_size) == len(cnn.stride) == len(cnn.padding)):
        raise ValueError("Depth CNN layer parameters must have equal lengths.")
    rows, cols = cnn.input_dim
    for channels, kernel, stride, padding in zip(cnn.output_channels, cnn.kernel_size, cnn.stride, cnn.padding):
        if (any(type(value) is not int for value in (channels, kernel, stride, padding))
                or min(channels, kernel, stride) < 1 or padding < 0):
            raise ValueError("Depth CNN layer parameters are invalid.")
        rows, cols = (rows + 2 * padding - kernel) // stride + 1, (cols + 2 * padding - kernel) // stride + 1
        if min(rows, cols) < 1:
            raise ValueError("Depth CNN collapses the input image.")
    if algorithm.class_name != "DiscretePPO" or algorithm.kl_method != "categorical":
        raise ValueError("Selector training requires the categorical PPO implementation.")
    for value in (algorithm.learning_rate, algorithm.clip_param, algorithm.desired_kl, algorithm.max_grad_norm):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Selector learning rate, clip, KL and gradient limits must be positive.")
    for value in (algorithm.entropy_coef, algorithm.value_loss_coef):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Selector entropy and value loss weights must be nonnegative.")
    if not 0.0 < algorithm.gamma <= 1.0 or not 0.0 <= algorithm.lam <= 1.0:
        raise ValueError("Selector gamma and lam must lie in their probability ranges.")
    for value in (agent_cfg.num_steps_per_env, agent_cfg.max_iterations, algorithm.num_learning_epochs,
                  algorithm.num_mini_batches, env_cfg.scene.num_envs):
        if type(value) is not int or value < 1:
            raise ValueError("Selector rollout, iteration, epoch and environment counts must be positive.")
    batch_size = agent_cfg.num_steps_per_env * env_cfg.scene.num_envs
    if batch_size % algorithm.num_mini_batches:
        raise ValueError("The high-level rollout batch must divide evenly into minibatches.")
    if agent_cfg.empirical_normalization or agent_cfg.clip_actions is not None:
        raise ValueError("Do not normalize the whole multimodal observation or clip discrete mode IDs.")

__all__ = [
    "FrozenRENetCfg", "G1SelectorAgentCfg", "G1SelectorEnvCfg", "Reward",
    "SelectorActorCriticCfg", "SelectorControlCfg", "SelectorDepthEncoderCfg",
    "SelectorFaultCfg", "SelectorObservationCfg", "SelectorPpoAlgorithmCfg",
    "SelectorRENetCfg", "configure_selector_agent",
]