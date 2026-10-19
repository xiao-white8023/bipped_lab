# ============================================
# G1 23dof
# ============================================
import math
from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


from isaaclab.sensors.ray_caster import RayCasterCfg

from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from legged_lab.sensors.grouped_ray_caster import get_link_prim_targets

import legged_lab.mdp as mdp
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    Privileged_Info,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
    CameraCfg,
    left_feet_ray_caster_cfg,
    right_feet_ray_caster_cfg
)
from legged_lab.terrains import ROUGH_TERRAINS_CFG
from legged_lab.assets.g1.g1_29 import G1_23CFG, G1_23DOF_LINKS
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from legged_lab.sensors.camera_noise import DistanceDependentGaussianNoiseCfg
from legged_lab.sensors.foot_sole_pattern import FootSolePatternCfg
from legged_lab.perception.foot_edge import FootEdgeCfg
from legged_lab.perception.foot_edge_geometry import EdgeGeometryCfg
from legged_lab.mdp.foot_edge import foot_edge_penalty
import os
import legged_lab
LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)
# AMP 格式开关：50 = 19个非踝关节；58 = 全部23个关节。
# 只修改这一行，环境特征、数据路径、加载器和判别器维度会同步切换。
G1_DEPTH_AMP_DIM = 58
if type(G1_DEPTH_AMP_DIM) is not int or G1_DEPTH_AMP_DIM not in (50, 58):
    raise ValueError("G1_DEPTH_AMP_DIM must be 50 or 58.")
G1_DEPTH_VP_PROPRIO_HISTORY_FRAMES = 10
G1_DEPTH_TRAINING_DIR = f"{LEGGED_LAB_ROOT}/envs/g1/datasets/make_g1_23dof/training"
G1_DEPTH_AMP_MOTION_DIR = (
    f"{G1_DEPTH_TRAINING_DIR}/amp_no_ankle" if G1_DEPTH_AMP_DIM == 50
    else f"{G1_DEPTH_TRAINING_DIR}/amp_full"
)



@configclass
class DepthTrainCfg:
    # 材质/质量在 startup 随机化，推力从训练开始计时。
    domain_rand_start_iteration: int = 0


@configclass
class Reward:
    # 论文脚体积点边缘惩罚：函数返回非负代价，RewardManager统一乘负权重和dt。
    # -2.0来自InstinctLab官方任务；不新增足底支撑奖励。
    foot_edge_penalty = RewTerm(func=foot_edge_penalty, weight=-4.0)

    # 跟踪速度奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

    # 有平移速度指令但实际水平速度过低时，惩罚原地站立。
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-0.5,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    energy = RewTerm(func=mdp.energy, weight=-1e-3)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )

    fly = RewTerm(
        func=mdp.fly,
        weight=-1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
        weight=-2.0
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )

    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    # 脚接触后，按足底采样点的支撑不足比例处罚。
    # stair_edge_penalty_L = RewTerm(
    #     func=mdp.single_foot_contact_area_penalty,
    #     weight=-0.02,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="left_ankle_roll_link"),
    #         "ray_sensor_cfg": SceneEntityCfg("left_feet_ray_caster"),
    #         "threshold": 0.04,
    #     },
    # )
    # stair_edge_penalty_R = RewTerm(
    #     func=mdp.single_foot_contact_area_penalty,
    #     weight=-0.02,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="right_ankle_roll_link"),
    #         "ray_sensor_cfg": SceneEntityCfg("right_feet_ray_caster"),
    #         "threshold": 0.04,
    #     },
    # )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[ ".*_shoulder_roll.*", ".*_shoulder_yaw.*", 
                           ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"]
            )
        },
    )
    joint_deviation_waist=RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["waist_yaw_joint"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    alive = RewTerm(func=mdp.alive, weight=0.15)

@configclass
class G1DEPTHENVCFG:
    # 只影响g1_depth的训练奖励/诊断，不进入actor、critic或部署观测。
    foot_edge: FootEdgeCfg = FootEdgeCfg(
        enabled=True,  # 总开关；False且不显示几何时，不构建边缘索引。
        geometry=EdgeGeometryCfg(
            angle_threshold_deg=70.0,  # 官方锐边二面角阈值，单位：度。
            cylinder_radius=0.03, # 0.05  # 官方parkour任务的边缘圆柱半径，单位：米。
            adjacent_angle_threshold_deg=30.0,  # 官方贪心拼接方向阈值。
            fit_distance_threshold=0.05,  # 官方实现实际使用的拟合最大偏离，单位：米。
            min_points=2,  # 官方任务最少拼接点数。
            num_grid_cells=64**3,  # 官方空间索引网格参数。
            seed=42,  # 仅用于几何拼接，不消耗策略/环境的全局随机数。
        ),
        x_range=(-0.06, 0.14),  # G1 23dof脚局部坐标：覆盖后跟至脚尖，单位：米。
        y_range=(-0.03, 0.03),
        z_range=(-0.03, 0.0),  # 虚拟足部体积近似，不改URDF四球物理碰撞。
        grid_shape=(10, 5, 2),  # 每脚100个三维规则网格点。
        velocity_epsilon=1e-3,  # 严格采用论文公式(3)；官方源码是1e-6，两者不同。
        tolerance=0.0,  # 不对穿入深度增加额外容忍距离。
        debug_vis=False,  # 显示原脚模型上的体积点和边缘圆柱，默认关闭。
        support_diagnostics=True,  # 仅记录足底支撑率，不参与奖励。
        support_distance_threshold=0.04,  # 逐射线支撑距离阈值，单位：米。
        support_contact_threshold=1.0,  # 负载接触的竖直力阈值，单位：牛顿。
    )
    amp_observation_dim: int = G1_DEPTH_AMP_DIM
    # 可视化始终读取完整机器人姿态，不使用50/58维 AMP 特征文件。
    amp_motion_files_display = [
    f"{G1_DEPTH_TRAINING_DIR}/visualization/walk_turns_40s.txt"
    ]
    device: str = "cuda:0"
    # 深度高斯噪声与概率模糊从第一张图像开始启用。
    depth_noise_start_iteration: int = 0
    depth_noise_force_active: bool = False  # 显式噪声回放评估可覆盖训练阶段。
    # 每个环境每次采集独立触发；0 关闭模糊，1 每帧模糊。跟随 add_camera_noise 开关。
    depth_blur_probability: float = 0.5
    depth_noise: DistanceDependentGaussianNoiseCfg = DistanceDependentGaussianNoiseCfg(
        near_std=0.0,
        far_std=0.10,
        distance_exponent=2.0,
    )
    scene: BaseSceneCfg = BaseSceneCfg(  
        max_episode_length_s=20.0,
        # 默认深度训练环境数量。
        num_envs=128,
        env_spacing=2.5,
        robot=G1_23CFG,
        terrain_type="generator",
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
            add_camera_noise=True,
            add_camera=True,
            camera = GroupedRayCasterCameraCfg(
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
                mesh_prim_paths=["/World/ground"]+get_link_prim_targets(G1_23DOF_LINKS),
                ray_alignment="yaw",
                debug_vis=False,
                data_types=["distance_to_image_plane"],
                update_period=0.02,
                depth_clipping_behavior="max",
                min_distance=0.1
            ), 
            depth_max = 2.5,
            depth_update_interval=5,
            depth_crop=(18,0,16,16),
            depth_history_frames=2
        ),
        left_feet_ray_caster=left_feet_ray_caster_cfg(
            add_left_feet_ray_caster=True,
            left_feet_ray_caster = RayCasterCfg(
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
                    debug_vis=False
            ),
        ),
        right_feet_ray_caster=right_feet_ray_caster_cfg(
            add_right_feet_ray_caster=True,
            right_feet_ray_caster = RayCasterCfg(
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
                    debug_vis=False
            ),
        ),
        privileged_info=Privileged_Info(
            enable_feet_info=False,
            enable_root_height=False,
            enable_feet_contact_force=False,
        ),
    )

    reward=Reward()
    depth_training=DepthTrainCfg()

    robot: RobotCfg = RobotCfg(
        # 深度策略保留10帧本体历史和2帧深度历史。
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        depth_history_frames=2,
        depth_max=2.5,
        depth_update_interval=5,
        depth_crop=(18,0,16,16), # up=18, down=0, left=16, right=16
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
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5,0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
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
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=[".*torso.*"]),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
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

            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            )  
    ),
        # 动作延迟关闭；保留范围供后续显式开启时使用。
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(  
                                                                use_gpu=True,            # 开启 GPU 物理
                                                                gpu_max_rigid_contact_count=2**23, # 必须改大，否则 4096 环境会崩
                                                                gpu_max_rigid_patch_count=2**23,   # 同上
                                                                gpu_heap_capacity=2**26  # 必须改大
                                                                ) 
                        )         


@configclass
class CnnMlpCfg:
    input_dim:tuple[int,int] = MISSING                   # 图像的高和宽
    input_channels:int = MISSING                         # 图像的通道数
    output_channels:list[int] = MISSING                  # 每一层卷积输出的通道数列表
    kernel_size:list[int] | int= MISSING                 # 卷积核大小，可以是一个固定的整数（比如 3，代表所有层都是 3x3），也可以是列表指定每一层。
    stride: int | tuple[int, ...] | list[int] = 1        # 步长，决定了每次卷积图像缩小的比例
    dilation: int | tuple[int, ...] | list[int] = 1      # 膨胀系数（用于扩大感受野，一般填 1 即可） 
    padding: str = "none"                                # 边缘填充方式（防止图像越卷越小）
    norm: str | tuple[str] | list[str] = "none"          # 归一化方式（可以选不加 'none'，或者 'batch'、'layer'）
    activation: str = "elu"                              # 激活函数的名字（比如 'elu'、'relu'）
    max_pool: bool | tuple[bool] | list[bool] = False    # 是否在卷积后加最大池化层（进一步降维）
    global_pool: str = "none"                            # 设定在所有卷积层结束后，要不要加全局池化，把整个特征图直接压缩成 1×1 的大小。
    flatten: bool = False                                # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
    mlp_hidden_dim: list[int] | None =None
    mlp_output_dim:int=128
    mlp_activation:str="relu"
    num_heads:int = 16
    embed_dim:int = 64


@configclass
class DepthRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    vel_estimation_warmup_iters:int=0
    vel_estimation_coef: float = 1.0
    terrain_recon_coef: float = 0.5
    terrain_recon_target_normalization: str = "fixed"

    vel_dim: int = 3
    terrain_recon_target_clip:int=1
    terrain_recon_warmup_iters:int=500
    terrain_scan_dim: int = 187
    terrain_recon_front_only: bool = True
    terrain_recon_grid_cols: int = 17
    terrain_recon_grid_rows: int = 11
    terrain_recon_x_min: float = 0.0
    single_critic_dim: int = 85
    critic_history_len: int = 10
    vel_in_critic_offset: int = 78
    feet_height_coef: float = 0.2
    feet_height_warmup_iters: int = 500
    use_feet_height_valid_mask: bool = True
    use_terrain_height_valid_mask: bool = True
    feet_height_dim: int = 2
    feet_height_in_critic_offset: int = 83
    vp_vae_coef: float = 1.0
    vp_vae_beta: float = 0.01
    vp_vae_warmup_iters: int = 500


@configclass
class DepthRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    CnnMlp: CnnMlpCfg = None  # 告诉配置系统，我们多加了一个 CnnMlp 的参数
    single_proprio_dim: int = 78
    estimator_latent_dim: int = 64
    vp_output_type: str = "multi_head"
    vp_context_dim: int = 32
    vp_terrain_latent_dim: int = 32
    vp_head_hidden_dims: list[int] = [64]
    vp_vae_decoder_dims: list[int] = [128, 256]
    proprio_embed_dim: int = 64
    proprio_embed_dims: list[int] = [256, 128]
    fusion_type: str = "attention"
    attention_num_heads: int = 1
    vp_proprio_history_frames: int = G1_DEPTH_VP_PROPRIO_HISTORY_FRAMES
    # Critic 保留10帧特权历史，不将地形真值送入 Actor。
    single_critic_dim: int = 85
    critic_history_frames: int = 10
    vp_encoder_dims: list[int] = [128]
    use_vel_estimation: bool = True
    vel_activation: str = "elu"
    vel_hidden_dim: list[int] = [32]
    vel_dim: int = 3
    use_terrain_recon: bool = True
    terrain_activation: str = "elu"
    terrain_hidden_dim: list[int] = [256, 128]
    terrain_scan_dim: int = 187
    terrain_recon_front_only: bool = True
    terrain_recon_grid_cols: int = 17
    terrain_recon_grid_rows: int = 11
    terrain_recon_x_min: float = 0.0
    use_feet_height_prediction: bool = True
    feet_height_activation: str = "elu"
    feet_height_hidden_dim: list[int] = [64]
    feet_height_dim: int = 2

@configclass
class G1DEPTHAGENTCFG:
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = DepthRslRlPpoActorCriticCfg(
        class_name="G1DepthActorCritic",
        # Four GRU heads: v_t(3), h_f(2), posterior mean z_o(32), z_m(32).
        vp_output_type="multi_head",
        vp_context_dim=32,
        vp_terrain_latent_dim=32,
        vp_head_hidden_dims=[64],
        vp_vae_decoder_dims=[128, 256],
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        single_proprio_dim=78,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[256, 128],
        fusion_type="attention",
        attention_num_heads=1,
        vp_proprio_history_frames=G1_DEPTH_VP_PROPRIO_HISTORY_FRAMES,
        single_critic_dim=85,
        critic_history_frames=10,
        vp_encoder_dims=[128],
        use_vel_estimation=True,
        vel_activation="elu",
        vel_hidden_dim=[32],
        vel_dim=3,
        use_terrain_recon=True,
        terrain_activation="elu",
        terrain_hidden_dim=[256, 128],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        use_feet_height_prediction=True,
        feet_height_activation="elu",
        feet_height_hidden_dim=[64],
        feet_height_dim=2,
        CnnMlp=CnnMlpCfg(
            input_dim=(18,32),
            input_channels=2,
            output_channels=[8,16,32],
            kernel_size=[3,3,3],
            stride=[1,1,1],
            dilation=[1,1,1],
            padding="zeros",
            norm="none",
            activation="relu",
            max_pool=[True,True,False],
            global_pool="none",
            flatten=True,
            mlp_hidden_dim=[256,128],
            mlp_output_dim=64,
            mlp_activation="elu",
        ),
    )
    algorithm = DepthRslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        # MSE is averaged over 78 coordinates; KL is summed over 32 latent coordinates.
        vp_vae_coef=1.0,
        vp_vae_beta=0.01,
        vp_vae_warmup_iters=500,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # 0.005
        num_learning_epochs=5,
        num_mini_batches=4,  # 4
        learning_rate=1.0e-3, # 1.0e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
        vel_estimation_coef=1.0,
        terrain_recon_coef=0.5,
        terrain_recon_target_normalization="fixed",
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        single_critic_dim=85,
        critic_history_len=10,
        vel_in_critic_offset=78,
        feet_height_coef=0.2,
        feet_height_warmup_iters=500,
        use_feet_height_valid_mask=True,
        use_terrain_height_valid_mask=True,
        feet_height_dim=2,
        feet_height_in_critic_offset=83,
        terrain_recon_target_clip=1,
        terrain_recon_warmup_iters=500,
        vel_estimation_warmup_iters=0,
    )

    clip_actions = None
    save_interval = 1000
    runner_class_name = "DepthAmpOnPolicyRunner"
    experiment_name = "g1_depth"
    run_name = "vp_multihead69_beta_vae"
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3  # 风格奖励系数 动作像专家数据
    amp_observation_dim = G1_DEPTH_AMP_DIM
    amp_joint_count = (G1_DEPTH_AMP_DIM - 12) // 2
    amp_data_class_name = "G1AnkleAMPLoader" if G1_DEPTH_AMP_DIM == 58 else "AMPLoader"
    amp_motion_files = [
    f"{G1_DEPTH_AMP_MOTION_DIR}/walk_backward_20s.txt",
    f"{G1_DEPTH_AMP_MOTION_DIR}/walk_left_20s.txt",
    f"{G1_DEPTH_AMP_MOTION_DIR}/walk_right_20s.txt",
    f"{G1_DEPTH_AMP_MOTION_DIR}/walk_turns_40s.txt"
    ]
    amp_num_preload_transitions = 200000 
    # Discriminator 实际执行加权相加：
    # r_total = 0.7 * r_task + 0.3 * r_amp
    # r_amp = amp_reward_coef * max(1 - 0.25 * (D - 1)^2, 0)
    # r_task 已由 RewardManager 乘控制步长；D 是实数输出。
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23
