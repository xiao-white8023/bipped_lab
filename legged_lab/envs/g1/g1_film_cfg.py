# ================================================
# G1 29dof
# ================================================
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
    right_feet_ray_caster_cfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG,flex_terrain_CFG,ROUGH_PERLIN_TERRAINS_CFG  # noqa:F401
from legged_lab.assets.g1.g1_29 import G1_29CFG,G1_29DOF_LINKS
from legged_lab.assets.g1.unitree import G1_CFG

from isaaclab.sim import DomeLightCfg
from isaaclab.assets import AssetBaseCfg

from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
import os
import legged_lab
LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)

@configclass
class GaitCfg:
    """Configuration for gait phase used by periodic gait rewards."""
    enable: bool = True            # Whether to update the gait phase signal
    period: float = 0.8            # Gait period in seconds
    offset: float = 0.5            # Phase offset between left and right leg (0.5 = alternating)

@configclass
class DepthNoiseCurriculumCfg:
    enable: bool = True
    return_window_size: int = 4096
    cv_epsilon: float = 1.0e-6
    ema_beta: float = 0.9
    gaussian_std_range: tuple[float, float] = (0.0, 0.1) # 0.15
    failure_probability_range: tuple[float, float] = (0.0, 0.06) # 0.12

@configclass
class Reward:
    # 跟踪速度奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

    # 爬楼梯不需要对z轴的速度进行惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.075) # -0.05

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
        weight=-0.5,
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

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
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
                joint_names=[".*waist.*",".*_shoulder_roll.*", ".*_shoulder_yaw.*", 
                           ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"]
            )
        },
    )

    joint_deviation_waist_roll=RewTerm(func=mdp.joint_deviation_l1_always,
                                       weight=-0.1,
                                       params={
                                                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist_roll.*"])
                                              },
                                      )
    joint_deviation_waist_yaw=RewTerm(func=mdp.joint_deviation_l1_always,
                                       weight=-0.01,
                                       params={
                                                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist_yaw.*"])
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

    gait_phase_contact = RewTerm(
    func=mdp.new_gait_phase_contact,
    weight=0.2,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_sensor",
            body_names=[
                "left_ankle_roll.*",
                "right_ankle_roll.*",
            ],
        ),
        "stance_threshold": 0.55,
        "command_threshold": 0.1,
    },
    )

    stand_still_pose = RewTerm(
    func=mdp.stand_still_pose,
    weight=-0.25,          # 负权重：误差大时惩罚，误差小时转为正向奖励
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "threshold": 0.1,   # 指令阈值，低于此值视为"零指令"
        "offset": 4.0,      # 偏移量：误差超过此值开始惩罚，低于此值给予奖励
   },
    )

   # stair_edge_penalty_L = RewTerm(
       # func=mdp.single_foot_contact_area_penalty,
       # weight=-0.1,  # 权重可以根据训练效果调整
      #  params={
     #       "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="left_ankle_roll_link"),
    #        "ray_sensor_cfg": SceneEntityCfg("left_feet_ray_caster"), # 在 SceneCfg 中定义的左脚射线
   #         "threshold": 0.04
    #    }
   # )
   # stair_edge_penalty_R = RewTerm(
       # func=mdp.single_foot_contact_area_penalty,
      #  weight=-0.1, 
     #   params={
    #        "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names="right_ankle_roll_link"),
   #         "ray_sensor_cfg": SceneEntityCfg("right_feet_ray_caster"), # 在 SceneCfg 中定义的右脚射线
  #          "threshold": 0.04
  #      }
   # )

@configclass
class G129MOE_FILMENVCFG:
    
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_CFG,
        terrain_type="generator",
        terrain_generator=flex_terrain_CFG,#
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
            depth_history_frames=4 #8
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
        ),
        privileged_info=Privileged_Info(    
                                        enable_feet_info = True,
                                        enable_root_height= True,
                                        enable_feet_contact_force = True
                                       )
    )

    reward=Reward()

    gait=GaitCfg()

    depth_noise_curriculum: DepthNoiseCurriculumCfg = DepthNoiseCurriculumCfg()

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        depth_history_frames=4, # 8
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
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0, 0), ang_vel_z=(-0, 0), heading=(-0, 0)
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
    flatten: bool = False                                 # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
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
    single_critic_dim: int = 107
    critic_history_len: int = 10
    vel_in_critic_offset: int = 104

@configclass
class CustomRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    CnnMlp: CnnMlpCfg = None  # 告诉配置系统，多加了一个 CnnMlp 的参数
    use_gru: bool = False # 是否使用GRU模块
    use_film_cnn: bool = False # 是否使用FiLM调制CNN特征图
    use_film_moe_gate: bool = False # 是否使用FiLM调制MoE门控网络
    use_separate_moe_gate_input: bool = False # Gate是否只使用 depth_latent
    moe_gate_command_start_idx: int = 6
    moe_gate_command_dim: int = 3
    use_moe_topk: bool = False
    moe_topk: int = 2
    moe_topk_start_iter: int = 2000
    single_proprio_dim:int=1
    num_experts:int=4 # 门控网络的专家
    gate_hidden_dim:list=[] # 门控网络的专家隐藏层维度

    his_encoder_dims: list = [256, 128]
    his_latent_dim: int = 64

    use_terrain_recon:bool=False # 是否使用地形重建
    terrain_activation:str = "elu"
    terrain_hidden_dim:list=[256,128]
    terrain_scan_dim:int=1
    terrain_recon_front_only:bool=False
    terrain_recon_grid_cols:int=1
    terrain_recon_grid_rows:int=1
    terrain_recon_x_min:float=0.0
    use_vel_estimation:bool=True # 是否使用速度重建
    vel_activation:str = "elu"
    vel_hidden_dim:list=[32]
    vel_dim: int = 3

@configclass
class  G129MOE_FILMAGENTENV:
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
        single_proprio_dim=96, # 100
        his_encoder_dims = [256, 128],
        his_latent_dim = 64,
        use_gru=False,
        use_film_cnn= True, # 是否使用FiLM调制CNN特征图
        use_film_moe_gate= False, # 是否使用FiLM调制MoE门控网络
        use_separate_moe_gate_input=True,
        use_moe_topk=False,
        moe_topk=2,
        moe_topk_start_iter=2000,
        use_vel_estimation=True,
        use_terrain_recon=True,
        terrain_activation = "elu",
        terrain_hidden_dim=[256,128],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        vel_activation = "elu",
        vel_hidden_dim=[32],
        vel_dim = 3,
        
        CnnMlp=CnnMlpCfg(
                input_dim = (18,32),
                input_channels = 4, # 8
                output_channels = [8,16],# [4]
                kernel_size = [3,3],# [3]
                stride = [1,1],#[1]
                dilation = [1,1],# [1]
                padding = "zeros",
                norm = "none", # 归一化方式（可以选不加 'none'，或者 'batch'、'layer'）
                activation = "relu", # 激活函数的名字（比如 'elu'、'relu'）
                max_pool = [True,True],# True # 是否在卷积后加最大池化层（进一步降维）
                global_pool = "none", # 设定在所有卷积层结束后，要不要加全局池化，把整个特征图直接压缩成 1×1 的大小。
                flatten = True, # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
                mlp_hidden_dim = [256,128],# [256,128]
                mlp_output_dim = 64, #128
                mlp_activation = "elu"
            )
    )
    algorithm = CustomRslRlPpoAlgorithmCfg(
        class_name="MoePPO",
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
        single_critic_dim =120, # 124
        critic_history_len = 10,
        vel_in_critic_offset= 98, # 102
        terrain_recon_target_clip=1,
        terrain_recon_warmup_iters=500,
        vel_estimation_warmup_iters=0,
        use_moe_balance_loss=True,
        moe_balance_coef=0.01,
        use_moe_gate_entropy_loss=False,
        moe_gate_entropy_coef=0.001
    )

    clip_actions = None
    save_interval = 1000
    runner_class_name = "FilmOnPolicyRunner"
    experiment_name = "g1_noamp"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
