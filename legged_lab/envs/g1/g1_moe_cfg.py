import math

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
from rsl_rl.modules import CnnMlpCfg

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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401
from legged_lab.assets.g1.g1_29 import G1_29CFG,G1_29DOF_LINKS

from isaaclab.sim import DomeLightCfg
from isaaclab.assets import AssetBaseCfg

from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
import os
import legged_lab
LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)

@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85

@configclass
class Reward:
    # 跟踪速度奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

    # 爬楼梯不需要对z轴的速度进行惩罚
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    
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
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
        weight=-2.0
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
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

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )

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
                joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", 
                           ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"]
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    alive = RewTerm(func=mdp.alive, weight=0.15)
    
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5)

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    # stand_still_pose = RewTerm(
    #     func=mdp.stand_still_pose,
    #     weight=-0.7,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             #joint_names=[".*_hip_.*", ".*_knee.*", ".*_ankle.*"] # 仅约束双腿
    #         ), 
    #         "threshold": 0.1,
    #         "offset": 4, 
    #     },
    # )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    stair_edge_penalty_L = RewTerm(
        func=mdp.single_foot_contact_area_penalty,
        weight=-0.1,  # 权重可以根据训练效果调整
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="left_ankle_roll_link"),
            "ray_sensor_cfg": SceneEntityCfg("left_feet_ray_caster"), # 在 SceneCfg 中定义的左脚射线
            "threshold": 0.04
        }
    )
    stair_edge_penalty_R = RewTerm(
        func=mdp.single_foot_contact_area_penalty,
        weight=-0.1, 
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="right_ankle_roll_link"),
            "ray_sensor_cfg": SceneEntityCfg("right_feet_ray_caster"), # 在 SceneCfg 中定义的右脚射线
            "threshold": 0.04
        }
    )

    # 落地平整度惩罚 (仅在接触时生效)
    # feet_flat_contact = RewTerm(
    #     func=mdp.feet_orientation_contact,
    #     weight=-0.1, 
    #     params={
    #         # 严格按照 [左, 右] 的顺序传入连杆名称
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_sensor", 
    #             body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    #         )
    #     }
    # )

@configclass
class G129WALK_MOEROUGHENVCFG:
    amp_motion_files_display=[f'{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_visualization/g1_walk.txt']
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_29CFG,
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
        camera=CameraCfg(
            add_camera_noise=False,
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
                mesh_prim_paths=["/World/ground"]+get_link_prim_targets(G1_29DOF_LINKS),
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
            depth_history_frames=8
        ),
        left_feet_ray_caster=left_feet_ray_caster_cfg(
            add_left_feet_ray_caster=True,
            left_feet_ray_caster = RayCasterCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
                    mesh_prim_paths=["/World/ground"],
                    offset=RayCasterCfg.OffsetCfg(pos=(0.035, 0.0, -0.025)),
                    pattern_cfg=GridPatternCfg(
                        resolution=0.02,  # 每0.02米一个射线
                        size=[0.12, 0.035] # 使用内接矩形避免边缘空气噪点
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
                    offset=RayCasterCfg.OffsetCfg(pos=(0.035, 0.0, -0.025)),
                    pattern_cfg=GridPatternCfg(
                        resolution=0.02,  # 每0.02米一个射线
                        size=[0.12, 0.035] # 使用内接矩形避免边缘空气噪点
                    ),
                    ray_alignment="base",
                    update_period=0.02,
                    debug_vis=False
            ),
        )
    )

    reward=Reward()

    gait=GaitCfg()

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        depth_history_frames=8,
        depth_max=2.5,
        depth_update_interval=5,
        depth_crop=(18,0,16,16), # up=18, down=0, left=16, right=16
        action_scale=0.25,
        terminate_contacts_body_names=[".*torso.*"],
        feet_body_names=["left_ankle_roll.*", "right_ankle_roll.*"],
        limit_angle=0.8
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
        debug_vis=False,
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
class CustomRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    vel_estimation_warmup_iters:int=0
    vel_estimation_coef: float = 1.0
    terrain_recon_coef: float = 0.5
             
    vel_dim: int = 3
    terrain_recon_target_clip:int=1
    terrain_recon_warmup_iters:int=500
    terrain_scan_dim: int = 187
    terrain_recon_front_only: bool = True
    terrain_recon_grid_cols: int = 17
    terrain_recon_grid_rows: int = 11
    terrain_recon_x_min: float = 0.0
    single_critic_dim: int = 107
    critic_history_len: int = 10
    vel_in_critic_offset: int = 104

@configclass
class CustomRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    CnnMlp: CnnMlpCfg = None  # 告诉配置系统，多加了一个 CnnMlp 的参数
    single_proprio_dim:int=1
    num_experts:int=4 # 门控网络的专家
    gate_hidden_dim:list=[] # 门控网络的专家隐藏层维度

    his_encoder_dims: list = [256, 128]
    his_latent_dim: int = 64

    use_terrain_recon:bool=True # 是否使用地形重建
    terrain_activation:str = "elu"
    terrain_hidden_dim:list=[256,128]
    terrain_scan_dim:int=1
    terrain_recon_front_only:bool=False
    terrain_recon_grid_cols:int=1
    terrain_recon_grid_rows:int=1
    terrain_recon_x_min:float=0.0
    use_vel_estimation:bool=False # 是否使用速度重建
    vel_activation:str = "elu"
    vel_hidden_dim:list=[32]
    
    vel_dim: int = 3



@configclass
class G129WALK_MOEROUGHAGENTENV:
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = CustomRslRlPpoActorCriticCfg(
        class_name="MoeActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_experts=4,
        gate_hidden_dim=[256,128,64],
        single_proprio_dim=102,
        his_encoder_dims = [256, 128],
        his_latent_dim = 64,
        
        use_terrain_recon=False,
        terrain_activation = "elu",
        terrain_hidden_dim=[256,128],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        use_vel_estimation=True,
        vel_activation = "elu",
        vel_hidden_dim=[32],
        vel_dim = 3,
        
        CnnMlp=CnnMlpCfg(
                input_dim = (18,32),
                input_channels = 8,
                output_channels = [4],
                kernel_size = [3],
                stride = [1],
                dilation = [1],
                padding = "zeros",
                norm = "none", # 归一化方式（可以选不加 'none'，或者 'batch'、'layer'）
                activation = "relu", # 激活函数的名字（比如 'elu'、'relu'）
                max_pool = True, # 是否在卷积后加最大池化层（进一步降维）
                global_pool = "none", # 设定在所有卷积层结束后，要不要加全局池化，把整个特征图直接压缩成 1×1 的大小。
                flatten = True, # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
                mlp_hidden_dim = [256,256],
                mlp_output_dim = 128,
                mlp_activation = "elu"
            )
    )
    algorithm = CustomRslRlPpoAlgorithmCfg(
        class_name="MoeAmpPpO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, #0.005
        num_learning_epochs=5,
        num_mini_batches=4,  # 4
        learning_rate=1.0e-3, #1.0e-4
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
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        single_critic_dim = 107,
        critic_history_len = 10,
        vel_in_critic_offset= 104,
        terrain_recon_target_clip=1,
        terrain_recon_warmup_iters=500,
        vel_estimation_warmup_iters=0
    )

    clip_actions = None
    save_interval = 1000
    runner_class_name = "MoeAmpOnPolicyRunner"
    experiment_name = "g1_rough"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3  # 风格奖励系数 动作像专家数据。值越大 机器人越重视模仿专家数据 从而忘记走路
    amp_motion_files = [f'{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_amp_expert/g1_walk.txt']
    amp_num_preload_transitions = 200000 # 在训练开始前，从amp_motion_files指定的文件中，提前加载 200,000 条运动转换数据到内存中，形成一个大型专家数据缓冲区，供后续训练时快速采样使用。
    '''
    Taskfinal​=Taskraw​×[0.7+(1−0.7)×D] 判别器输入的就是0或1 如果任务奖励100分 完成的很好 但是不像专家数据 判别器打0分 则作中的任务奖励只有70分 
    '''
    amp_task_reward_lerp = 0.7  # 这是一个惩罚机制。它把“任务奖励”和“动作质量”挂钩了 
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 29