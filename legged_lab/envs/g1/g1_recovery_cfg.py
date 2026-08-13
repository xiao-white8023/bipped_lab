# ============================================
# G1 23dof recovery locomotion
# ============================================
import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
import legged_lab
from legged_lab.assets.g1.g1_29 import G1_23CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import Flat_terrain, ROUGH_TERRAINS_CFG

LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)


@configclass
class RecoveryResetCfg:
    """G1-native dangerous reset ranges for the first recovery version."""

    pose_range: dict = {
        "x": (-3.0, 3.0),
        "y": (-2.0, 2.0),
        "z": (0.15, 0.25),
        "roll": (-0.75, 0.75), # -43～43
        "pitch": (-0.75, 0.75),
        "yaw": (-0.25, 0.25), # -14～14
    }
    randomize_terrain_xy: bool = True
    terrain_x_range: tuple = (-3.0, 3.0)
    terrain_y_range: tuple = (-2.0, 2.0)
    extreme_reset_enable: bool = True
    extreme_reset_prob: float = 0.02
    extreme_data_path: str = f"{LEGGED_LAB_ROOT}/envs/g1/datasets/recovery/extrem_data_g1.npy"
    extreme_use_data_commands: bool = True
    extreme_yaw_range: tuple = (-0.25, 0.25)
    # 这是reset瞬间就有的速度
    velocity_range: dict = {
        "x": (-1.2, 1.2), 
        "y": (-1.0, 1.0),
        "z": (-0.4, 0.4),
        "roll": (-2.0, 2.0),
        "pitch": (-2.0, 2.0),
        "yaw": (-1.5, 1.5),
    }
    joint_pos_offset_range: tuple = (-0.35, 0.35) # 在默认的关节位置上随机加入一个offset
    joint_vel_range: tuple = (-2.0, 2.0)


@configclass
class RecoveryCommandCurriculumCfg:
    enable: bool = True
    steps: int = 24 * 5000
    start_ranges: CommandRangesCfg = CommandRangesCfg(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(-math.pi, math.pi),
    )
    target_ranges: CommandRangesCfg = CommandRangesCfg(
        lin_vel_x=(-0.6, 2.5),
        lin_vel_y=(-0.6, 0.6),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
    )

# 机器人的终止状态
'''
身体前后左右倾斜 < 77°
    → 还允许 Recovery 尝试救

身体前后左右倾斜 > 77°
    → 认为已经基本侧倒
    → episode terminate
'''
@configclass
class RecoveryTerminationCfg:
    contact_force_threshold: float = 1.0
    pelvis_contact_force_threshold: float = 1.0
    roll_threshold: float = 1.35
    pitch_threshold: float = 1.35


@configclass
class G1SupportPolygonCfg:
    foot_body_names: list = ["left_ankle_roll_link", "right_ankle_roll_link"]
    support_points_local: list = [
        [0.12, 0.03, -0.03],
        [0.12, -0.03, -0.03],
        [-0.05, -0.025, -0.03],
        [-0.05, 0.025, -0.03],
    ]
    collision_sphere_radius: float = 0.005
    support_point_contact_tolerance: float = 0.012
    contact_force_threshold: float = 1.0


@configclass
class ZmpCostCfg:
    use_zmp_cost: bool = True
    zmp_cost_type: str = "margin"
    zmp_margin_slack: float = 0.01 # ZMP 即使稍微超出理论支撑边界一点，也先不给 cost。允许 ZMP 有一个 1 cm 的容忍带。
    zmp_cost_clip: float = 0.10 # ZMP 再怎么严重跑出去，每一步 cost 最多按 0.10 计算
    zmp_single_support_weight: float = 0.5 # 这个是在机器人单脚支撑时降低 ZMP cost。
    zmp_double_support_weight: float = 1.0
    zmp_no_contact_cost: float = 0.0 # 双脚都没有接触的时候机器人的cost
    zmp_com_vel_filter_alpha: float = 0.25 # 它是在 计算 CoM 加速度之前，对 CoM velocity 做低通滤波


@configclass
class HwcRecoveryObservationCfg:
    """HWC-Loco recovery observation layout migrated to G1 23DoF."""

    num_proprio: int = 80
    prop_history_len: int = 5
    history_buffer_len: int = 6
    num_demo: int = 0
    latent_dim: int = 16
    zmp_frequency_count: int = 4
    policy_label_dim: int = 12
    text_feat_output_dim: int = 16
    height_context_dim: int = 132

    @property
    def feature_dim(self) -> int:
        return self.prop_history_len * self.num_proprio

    @property
    def decoder_out_dim(self) -> int:
        return self.latent_dim + self.policy_label_dim

    @property
    def actor_obs_dim(self) -> int:
        return self.feature_dim + self.num_proprio + self.num_demo + self.policy_label_dim

    @property
    def actor_estimated_obs_dim(self) -> int:
        return self.feature_dim + self.num_proprio + self.num_demo + self.decoder_out_dim

    @property
    def dr_label_dim(self) -> int:
        # mass/com(4), friction(1), motor strength P/D(2*23), kp/kd(2*23), push(5), terrain heights.
        return 4 + 1 + 23 * 2 + 23 * 2 + 5 + self.height_context_dim

    @property
    def privileged_proprio_dim(self) -> int:
        return self.num_proprio + self.dr_label_dim + self.policy_label_dim

    @property
    def critic_obs_dim(self) -> int:
        return self.prop_history_len * self.privileged_proprio_dim + self.privileged_proprio_dim


@configclass
class ConstrainedRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Constrained PPO parameters consumed by the G1 recovery runner."""

    use_zmp_cost: bool = True
    zmp_cost_limit: float = 0.03
    zmp_lambda_init: float = 0.0
    zmp_lambda_lr: float = 1.0e-3
    zmp_lambda_max: float = 100.0
    zmp_cost_value_loss_coef: float = 1.0
    normalize_cost_advantages: bool = False
    use_amp: bool = True
    amp_replay_buffer_size: int = 100000
    amp_loss_coef: float = 1.0
    amp_grad_penalty_coef: float = 10.0
    amp_walk_only: bool = True


@configclass
class HwcEstimatorCfg:
    train_with_estimated_states: bool = True
    learning_rate: float = 1.0e-4
    future_horizon: int = 1
    encoder_hidden_dims: list = [256, 128, 64]
    decoder_hidden_dims: list = [256, 128, 64]
    n_demo: int = 0
    priv_latent_dim: int = 16
    history_len: int = 5
    priv_states_dim: int = 12
    state_label_dim: int = 12
    dr_label_dim: int = 234
    priv_start: int = 480
    prop_start: int = 400
    prop_dim: int = 80
    priv_prop_start: int = 0


@configclass
class RecoveryActorCriticCfg(RslRlPpoActorCriticCfg):
    num_prop: int = 80
    num_demo: int = 0
    text_feat_input_dim: int = 400
    text_feat_output_dim: int = 16
    feat_hist_len: int = 5
    n_decoder_out: int = 28
    num_priv_explicit: int = 12
    num_hist: int = 6
    tanh_encoder_output: bool = False


@configclass
class RecoveryReward:
    """Initial G1 recovery reward weights; these are training hyperparameters to tune."""

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.8})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.2, params={"std": 0.8})
    alive = RewTerm(func=mdp.alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-150.0)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-3)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["pelvis", ".*torso.*", ".*hip.*"]), "threshold": 1.0},
    )
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
        weight=-2.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 600,
            "max_reward": 500,
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_shoulder_pitch.*", ".*_elbow.*", ".*_wrist.*"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )


@configclass
class RecoveryEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*torso.*"]),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.2, 1.2), "yaw": (-1.0, 1.0)}},
    )


@configclass
class G123RECOVERYENVCFG:
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=10.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_23CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )

    reward = RecoveryReward()
    recovery_reset: RecoveryResetCfg = RecoveryResetCfg()
    command_curriculum: RecoveryCommandCurriculumCfg = RecoveryCommandCurriculumCfg()
    termination: RecoveryTerminationCfg = RecoveryTerminationCfg()
    support_polygon: G1SupportPolygonCfg = G1SupportPolygonCfg()
    zmp: ZmpCostCfg = ZmpCostCfg()
    hwc_observation: HwcRecoveryObservationCfg = HwcRecoveryObservationCfg()
    amp_walk_command_threshold: float = 0.1

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=6,
        critic_obs_history_length=6,
        depth_history_frames=8,
        depth_max=2.5,
        depth_update_interval=5,
        depth_crop=(18, 0, 16, 16),
        action_scale=0.25,
        terminate_contacts_body_names=["pelvis", ".*torso.*", ".*hip.*"],
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
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )

    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            #heading=(-math.pi, math.pi),
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
        events=RecoveryEventCfg(),
        action_delay=ActionDelayCfg(enable=True, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=2,
        physx=PhysxCfg(
            use_gpu=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_heap_capacity=2**26,
        ),
    )


@configclass
class G123RECOVERYAGENTCFG(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RecoveryActorCriticCfg(
        class_name="ActorCriticCost",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_prop=80,
        num_demo=0,
        text_feat_input_dim=400,
        text_feat_output_dim=16,
        feat_hist_len=5,
        n_decoder_out=28,
        num_priv_explicit=12,
        num_hist=6,
        tanh_encoder_output=False,
    )
    algorithm = ConstrainedRslRlPpoAlgorithmCfg(
        class_name="ConstrainedPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        use_zmp_cost=True,
        zmp_cost_limit=0.03,
        zmp_lambda_init=0.0,
        zmp_lambda_lr=1.0e-3,
        zmp_lambda_max=100.0,
        zmp_cost_value_loss_coef=1.0,
        normalize_cost_advantages=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    estimator = HwcEstimatorCfg()

    clip_actions = None
    save_interval = 1000
    runner_class_name = "ConstrainedOnPolicyRunner"
    experiment_name = "g1_recovery"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "recovery"
    wandb_project = "recovery"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    min_normalized_std = [0.05] * 23
    use_amp = True
    amp_walk_command_threshold = 0.1
    amp_reward_coef = 0.3
    amp_motion_files = [
        f"{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_amp_expert_no_ankle/stand_to_walk.txt",
        f"{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_amp_expert_no_ankle/walk_turn_around.txt",
        f"{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_amp_expert_no_ankle/walk_turn_left.txt",
        f"{LEGGED_LAB_ROOT}/envs/g1/datasets/motion_amp_expert_no_ankle/walk_turn_right.txt",
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
