# ============================================
# G1 23dof
# ============================================
import math
from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

from isaaclab.sensors.ray_caster.patterns import GridPatternCfg

from isaaclab.sensors.ray_caster import RayCasterCfg

from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from legged_lab.sensors.grouped_ray_caster import get_link_prim_targets

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, Flat_terrain, flex_terrain_CFG  # noqa:F401
from legged_lab.assets.g1.g1_29 import G1_29CFG,G1_29DOF_LINKS,G1_23CFG,G1_23DOF_LINKS
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from legged_lab.sensors.camera_noise import DistanceDependentGaussianNoiseCfg
from legged_lab.sensors.foot_sole_pattern import FootSolePatternCfg
from legged_lab.perception.lp_height import (
    LPHeightCfg,
    PgttNoiseCfg,
    resolve_pgtt_height_cfg,
)
import os
import legged_lab
LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)
# AMP 格式开关：50 = 原始 PKL 四段的 no_ankle 格式；58 = 原 v3（全部 23 个关节）。
# 只修改这一行，环境特征、数据路径、加载器和判别器维度会同步切换。
G1_RENET_AMP_DIM = 58
if type(G1_RENET_AMP_DIM) is not int or G1_RENET_AMP_DIM not in (50, 58):
    raise ValueError("G1_RENET_AMP_DIM must be 50 or 58.")
G1_RENET_AMP_DATASET_NAME = (
    "motion_amp_expert_no_ankle" if G1_RENET_AMP_DIM == 50 else "g1_23dof_locomotion_v3"
)
G1_RENET_AMP_DATASET_DIR = (
    f"{LEGGED_LAB_ROOT}/envs/g1/datasets/{G1_RENET_AMP_DATASET_NAME}"
)
G1_RENET_AMP_CLIP_NAMES = (
    ("stand_to_walk", "walk_turn_around", "walk_turn_left", "walk_turn_right")
    if G1_RENET_AMP_DIM == 50 else ("walk_forward_turns", "walk_backward", "walk_left", "walk_right")
)
G1_RENET_AMP_MOTION_DIR = (
    G1_RENET_AMP_DATASET_DIR if G1_RENET_AMP_DIM == 50 else f"{G1_RENET_AMP_DATASET_DIR}/amp"
)
# 可视化使用完整机器人姿态，不能直接读取 50-D AMP 文件。
G1_RENET_AMP_DISPLAY_DIR = (
    f"{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_visualization/original_pkl"
    if G1_RENET_AMP_DIM == 50 else f"{G1_RENET_AMP_DATASET_DIR}/visualization"
)

# VP 本体历史开关：5 = 最近五帧；改为 10 即恢复十帧。只修改这一行。
# 观测缓存仍保留 10 帧（actor=2120），LP 查询仍为 5 帧，critic 仍为 10 帧。
G1_RENET_VP_PROPRIO_HISTORY_FRAMES = 10
if (type(G1_RENET_VP_PROPRIO_HISTORY_FRAMES) is not int
        or not 1 <= G1_RENET_VP_PROPRIO_HISTORY_FRAMES <= 10):
    raise ValueError("G1_RENET_VP_PROPRIO_HISTORY_FRAMES must be an integer from 1 to 10.")
from legged_lab.assets.g1.unitree import G1_CFG


@configclass
class RENetTrainCfg:
    """Training-time estimator mask.

    mask=1 selects the LP estimator, mask=0 selects the VP estimator.
    The deployment selector can later replace this random training mask.
    """
    mask_mode: str = "alternate"
    lp_probability: float = 0.5
    alternate_interval_iters: int = 20
    # Only the selected LP source is instantiated; VP and critic are unchanged.
    lp_height_source: str = "pgtt_full_atten"
    # 材质/质量在 startup 随机化，推力从训练开始计时。
    domain_rand_start_iteration: int = 0



@configclass
class Reward:
    # 跟踪速度奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    # vp_idle_when_commanded = RewTerm(
    #     func=mdp.vp_idle_when_commanded,
    #     weight=-0.1,
    #     params={"cmd_threshold": 0.1, "vel_threshold": 0.1,
    #             "ang_cmd_threshold": 0.1, "ang_vel_threshold": 0.1},
    # )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    energy = RewTerm(func=mdp.energy, weight=-1e-3)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    # G1 的小腿刚体：原来的非 ankle 正则也已包含这两项。
                    "left_knee_link",
                    "right_knee_link",
                    # 其余非脚踝刚体；排除 knee，避免重复匹配/重复计罚。
                    "(?!.*(?:ankle|knee).*).*",
                ],
            ),
            "threshold": 1.0,
        },
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
    # 两种模式共用：脚接触后，按足底采样点的支撑不足比例处罚。
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
class G1RENETENVCFG:
    amp_observation_dim: int = G1_RENET_AMP_DIM
    lp_height = LPHeightCfg(noise=PgttNoiseCfg(start_iteration=30000))
    amp_motion_files_display = [
    "/home/cjh/TienKung-Lab/legged_lab/envs/g1/datasets/make_g1_23dof/training/visualization/walk_turns_40s.txt"
    ]
    device: str = "cuda:0"
    # VP 深度高斯噪声与概率模糊从第一张图像开始启用；LP 仍保留 30000 次门槛。
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
        # Default AME training batch size.
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

    renet=RENetTrainCfg()


    robot: RobotCfg = RobotCfg(
        # Keep the archived observation layout; VP selects its configured window.
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
class CustomRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    vel_estimation_warmup_iters:int=0
    vel_estimation_coef: float = 1.0
    terrain_recon_coef: float = 0.5
    terrain_recon_target_normalization: str = "fixed"
    lp_terrain_recon_coef: float = 0.5
    lp_terrain_recon_warmup_iters: int = 500
    use_moe_balance_loss: bool = False
    moe_balance_coef: float = 0.0
    use_moe_gate_entropy_loss: bool = False
    moe_gate_entropy_coef: float = 0.0

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
    lp_feet_height_coef: float = 0.0
    lp_feet_height_warmup_iters: int = 500
    use_feet_height_valid_mask: bool = True
    use_terrain_height_valid_mask: bool = True
    feet_height_dim: int = 2
    feet_height_in_critic_offset: int = 83
    vp_vae_coef: float = 1.0
    vp_vae_beta: float = 2
    vp_vae_warmup_iters: int = 500


@configclass
class CustomRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    CnnMlp: CnnMlpCfg = None  # 告诉配置系统，我们多加了一个 CnnMlp 的参数
    LidarCnnMlp: CnnMlpCfg = None
    single_proprio_dim: int = 78
    estimator_mask_dim: int = 1
    estimator_latent_dim: int = 64
    vp_output_type: str = "legacy_latent"
    vp_context_dim: int = 32
    vp_terrain_latent_dim: int = 32
    vp_head_hidden_dims: list[int] = [64]
    vp_vae_decoder_dims: list[int] = [128, 256]
    proprio_embed_dim: int = 64
    proprio_embed_dims: list[int] = [256, 128]
    fusion_type: str = "attention"
    attention_num_heads: int = 1
    lp_encoder_type: str = "ame_point_cross_attention"
    ame_lp_attention_num_heads: int = 16
    ame_lp_query_history_frames: int = 5
    vp_proprio_history_frames: int = G1_RENET_VP_PROPRIO_HISTORY_FRAMES
    # Both modes retain the full privileged history, independently of VP actor history.
    single_critic_dim: int = 85
    critic_history_frames: int = 10
    vp_critic_history_frames: int = 10
    lp_encoder_dims: list[int] = [128]
    lp_height_history_frames: int = 1
    lp_terrain_recon_front_only: bool = False
    vp_encoder_dims: list[int] = [128]
    use_vel_estimation: bool = True
    vel_activation: str = "elu"
    vel_hidden_dim: list[int] = [32]
    vel_dim: int = 3
    use_terrain_recon: bool = True
    terrain_activation: str = "elu"
    terrain_hidden_dim: list[int] = [256, 128]
    lp_terrain_hidden_dim: list[int] = [256, 128]
    terrain_scan_dim: int = 187
    terrain_recon_front_only: bool = True
    terrain_recon_grid_cols: int = 17
    terrain_recon_grid_rows: int = 11
    terrain_recon_x_min: float = 0.0
    use_feet_height_prediction: bool = True
    use_lp_feet_height_prediction: bool = False
    feet_height_activation: str = "elu"
    feet_height_hidden_dim: list[int] = [64]
    feet_height_dim: int = 2

@configclass
class G1RENETAGENTCFG:
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = CustomRslRlPpoActorCriticCfg(
        class_name="RENetActorCritic",
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
        estimator_mask_dim=1,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[256, 128],
        fusion_type="attention",
        attention_num_heads=1,
        lp_encoder_type="ame_point_cross_attention",
        ame_lp_attention_num_heads=16,
        ame_lp_query_history_frames=5,
        vp_proprio_history_frames=G1_RENET_VP_PROPRIO_HISTORY_FRAMES,
        single_critic_dim=85,
        critic_history_frames=10,
        vp_critic_history_frames=10,
        lp_encoder_dims=[128],
        vp_encoder_dims=[128],
        use_vel_estimation=True,
        vel_activation="elu",
        vel_hidden_dim=[32],
        vel_dim=3,
        use_terrain_recon=True,
        terrain_activation="elu",
        terrain_hidden_dim=[256, 128],
        lp_terrain_hidden_dim=[256, 128],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        use_feet_height_prediction=True,
        use_lp_feet_height_prediction=False,
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
        LidarCnnMlp=CnnMlpCfg(
            input_dim=(11,17),
            input_channels=1,
            output_channels=[8,16],
            kernel_size=[3,3],
            stride=[1,1],
            dilation=[1,1],
            padding="zeros",
            norm="none",
            activation="elu",
            max_pool=[True,False],
            global_pool="none",
            flatten=True,
            mlp_hidden_dim=[128],
            mlp_output_dim=64,
            mlp_activation="elu",
        ),
    )
    algorithm = CustomRslRlPpoAlgorithmCfg(
        class_name="RENetAMPPPO",
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
        lp_terrain_recon_coef=0.5,
        lp_terrain_recon_warmup_iters=500,
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
        lp_feet_height_coef=0.0,
        lp_feet_height_warmup_iters=500,
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
    runner_class_name = "RENetAmpOnPolicyRunner"
    experiment_name = "g1_renet"
    run_name = "vp_multihead69_beta_vae"
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3  # 风格奖励系数 动作像专家数据
    amp_observation_dim = G1_RENET_AMP_DIM
    amp_joint_count = (G1_RENET_AMP_DIM - 12) // 2
    amp_data_class_name = "G1AnkleAMPLoader" if G1_RENET_AMP_DIM == 58 else "AMPLoader"
    # 50-D: the four original-PKL no_ankle clips only. 58-D: original v3 clips.
    # AMP files use the selected 50/58-D training layout, not the 58-D preview layout.
    # 50-D: keep each original clip's MotionWeight=0.5; normalized sampling is 25% each.
    amp_motion_files = [
    "/home/cjh/TienKung-Lab/legged_lab/envs/g1/datasets/make_g1_23dof/training/amp_full/walk_backward_20s.txt",
    "/home/cjh/TienKung-Lab/legged_lab/envs/g1/datasets/make_g1_23dof/training/amp_full/walk_left_20s.txt",
    "/home/cjh/TienKung-Lab/legged_lab/envs/g1/datasets/make_g1_23dof/training/amp_full/walk_right_20s.txt",
    "/home/cjh/TienKung-Lab/legged_lab/envs/g1/datasets/make_g1_23dof/training/amp_full/walk_turns_40s.txt"
    ]
    amp_num_preload_transitions = 200000 
    # Discriminator 实际执行加权相加：
    # r_total = 0.7 * r_task + 0.3 * r_amp
    # r_amp = amp_reward_coef * max(1 - 0.25 * (D - 1)^2, 0)
    # r_task 已由 RewardManager 乘控制步长；D 是实数输出。
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23


def configure_agent_for_lp_source(env_cfg, agent_cfg):
    """Configure the AME point-cross-attention LP used by g1_renet."""
    runtime_cfg = resolve_pgtt_height_cfg(env_cfg.lp_height, env_cfg.renet.lp_height_source)
    agent_cfg.policy.lp_height_history_frames = 1
    agent_cfg.policy.LidarCnnMlp.input_dim = tuple(runtime_cfg.shape)
    agent_cfg.policy.LidarCnnMlp.input_channels = 1
    agent_cfg.policy.lp_terrain_recon_front_only = False
    agent_cfg.policy.lp_encoder_type = "ame_point_cross_attention"
    agent_cfg.policy.attention_num_heads = 1
    agent_cfg.policy.ame_lp_attention_num_heads = 16
    agent_cfg.policy.ame_lp_query_history_frames = 5
    return agent_cfg
