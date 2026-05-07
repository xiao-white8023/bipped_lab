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

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    Privileged_Info,
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
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG,Flat_terrain  # noqa:F401
from legged_lab.assets.g1 import G1_29CFG
from legged_lab.assets.g1.unitree import G1_CFG
import legged_lab
import os
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
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
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
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
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
    alive = RewTerm(func=mdp.alive, weight=0.2)
    stand_still_pose = RewTerm(
        func=mdp.stand_still_pose,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
            ), 
            "threshold": 0.1,
            "offset": 4, 
        },
    )

@configclass
class G129WALK_FLATENVCFG:
    amp_motion_files_display=[f'{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_visualization/g1_walk.txt']
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_29CFG,
        terrain_type="generator",
        terrain_generator=Flat_terrain,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
        privileged_info=Privileged_Info(    
                                        enable_feet_Privileged = False,
                                        enable_root_height_Privileged= False,
                                        enable_feet_contact_force = False
                                       )
    )

    reward=Reward()
    
    gait=GaitCfg()

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=[".*torso.*"],
        feet_body_names=["left_ankle_roll.*", "right_ankle_roll.*"],
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
            feet_pos=1.0,
            feet_vel=1.0,
            contact_force= 0.01
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )

    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
    # heading_command
    #False（默认）：「直接角速度控制」—— 直接从 ranges.ang_vel_z 均匀采样角速度（比如 [-1.0, 1.0] rad/s），机器人直接按这个角速度转；
    #True：「航向误差控制」—— 先从 ranges.heading 均匀采样「目标航向角」（比如 [-π, π] rad），然后根据「当前航向与目标航向的误差」计算角速度（类似 
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
class G129WALK_FLATAGENTENV:
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 1000
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "g1_walk"
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
    amp_num_preload_transitions = 200000 
    '''
    Taskfinal​=Taskraw​×[0.7+(1−0.7)×D] 判别器输入的就是0或1 如果任务奖励100分 完成的很好 但是不像专家数据 判别器打0分 则作中的任务奖励只有70分 
    '''
    amp_task_reward_lerp = 0.7  # 这是一个惩罚机制。它把“任务奖励”和“动作质量”挂钩了 
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 29