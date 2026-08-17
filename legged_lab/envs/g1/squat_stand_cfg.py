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
    HeightCommandCfg,
    HeightCommandRangeCfg
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG,ROUGH_PERLIN_TERRAINS_CFG,Flat_terrain  # noqa:F401
from legged_lab.assets.g1.g1_29 import G1_29CFG,G1_29DOF_LINKS
from legged_lab.assets.g1.unitree import G1_CFG

from isaaclab.sim import DomeLightCfg
from isaaclab.assets import AssetBaseCfg

from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
import os
import legged_lab
LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)

@configclass
class Reward:
    track_height_cmd=RewTerm(func=mdp.track_height_cmd,
                            weight=5.0,
                            params={
                                "std": 0.05,
                                "asset_cfg": SceneEntityCfg("robot"),
                            },)

    energy = RewTerm(func=mdp.energy, weight=-1e-3)

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
        weight=-2.0
    )
    base_ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    feet_xy_velocity=RewTerm(func=mdp.feet_xy_velocity,
                             weight=-1,
                             params={                           
                                "threshold": 0.02,
                                "asset_cfg": SceneEntityCfg("robot",body_names="(.*ankle_roll.*).*"),
                            },)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_no_contact = RewTerm(
        func=mdp.feet_no_contact,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=".*ankle_roll.*",
            ),
            "threshold": 5.0,
        },
    )
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist_yaw.*"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    alive = RewTerm(func=mdp.alive, weight=0.15)
    com_support_margin = RewTerm(
        func=mdp.com_support_margin_penalty,
        weight=-2.0,
        params={
            "std": 0.03,
            "outside_scale": 10.0,
            "rear_scale": 1.0,
        },
    )
    com_xy_velocity = RewTerm(
    func=mdp.com_xy_velocity_l2,
    weight=-1.0,
    )






@configclass
class SaquatStandENVCFG:
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_CFG,
        terrain_type="generator",
        terrain_generator=Flat_terrain,#
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )

    reward=Reward()
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
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

    commands: HeightCommandCfg = HeightCommandCfg(
        resampling_time_range = (10.0, 10.0),
        rel_standing_envs = 0.2,
        stand_height = 0.77,
        debug_vis = False,
        ranges=HeightCommandRangeCfg(
            height=(0.46,0.77)  # 目标高度范围(0.46,0.76)
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
            # reset_robot_joints=EventTerm(
            #     func=mdp.reset_joints_by_scale,
            #     mode="reset",
            #     params={
            #         "position_range": (0.5, 1.5),
            #         "velocity_range": (0.0, 0.0),
            #     },
            # ),

            # push_robot=EventTerm(
            #     func=mdp.push_by_setting_velocity,
            #     mode="interval",
            #     interval_range_s=(10.0, 15.0),
            #     params={"velocity_range": {"x": (-0.5, 1.0), "y": (-0.5, 1.0)}},
            # ),  

            reset_arm_pose_and_hold=EventTerm(
            func=mdp.reset_arm_pose_and_hold,
            mode="reset",
            params={        
                "position_ranges": {
                            "right_shoulder_pitch_joint": (-0.75, 0.8),
                            "right_shoulder_roll_joint": (-0.78, 0.22),
                            "right_shoulder_yaw_joint":(-1,1),
                            "right_elbow_joint": (-0.23, 1.57),
                },
                "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=[
                                "right_shoulder_pitch_joint",
                                "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint",
                                "right_elbow_joint",
                            ],
                            preserve_order=True,
            )
            }
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
class SaquatStandAGENTENV:
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
        class_name="PPO", # AMPPPO
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
    )

    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner" # "AmpOnPolicyRunner"
    experiment_name = "g1_squat"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"