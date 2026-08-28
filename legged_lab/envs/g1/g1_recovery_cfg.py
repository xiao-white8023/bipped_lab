# ================================================
# G1 23dof recovery
# ================================================
import os

import legged_lab
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from legged_lab.assets.g1.g1_29 import G1_23CFG
from legged_lab.envs.base.base_config import (
    BaseSceneCfg,
    HeightScannerCfg,
    PhysxCfg,
    SimCfg,
)
from legged_lab.envs.base.base_env_config import BaseAgentCfg
from legged_lab.terrains import Flat_terrain


LEGGED_LAB_ROOT = os.path.dirname(legged_lab.__file__)


# IsaacLab filtered contacts are one-to-many only.  Recovery therefore creates
# one self-collision sensor per rigid body and filters each one against this
# complete robot-body set.  The order is the articulation's verified body order.
G1_RECOVERY_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


# Keep this override local to standalone Recovery even though the source G1
# asset currently enables self-collision as well.
G1_RECOVERY_ROBOT_CFG = G1_23CFG.replace(
    spawn=G1_23CFG.spawn.replace(
        articulation_props=G1_23CFG.spawn.articulation_props.replace(
            enabled_self_collisions=True
        )
    )
)


@configclass
class RecoveryControlCfg:
    # 23 个关节统一采用 default joint position 上的动作偏移。
    action_scale: float = 0.25
    clip_actions: float = 1.0


@configclass
class RecoveryObservationCfg:
    # Actor/Critic 均保留最近 4 个 policy step 的本体观测。
    history_length: int = 4
    clip_observations: float = 100.0


@configclass
class RecoveryResetCfg:
    # Stage 4.2b 只使用已筛选的 single-state whole-body NPZ。
    state_dir: str = os.path.join(
        LEGGED_LAB_ROOT,
        "envs/g1/datasets/npz/recovery_reset_states_v1",
    )
    # 保留后续阶段的配置入口；当前必须关闭以验证 exact dataset reset。
    random_yaw: bool = False


@configclass
class RecoveryRewardCfg:
    height_std: float = 0.3
    height_scale: float = 3.5

    # The corresponding terms are non-negative costs; negative weights make
    # them reward penalties exactly once.
    joint_acc_weight: float = -2.5e-7
    joint_pos_limits_weight: float = -10.0
    action_rate_weight: float = -0.01
    self_collision_weight: float = -0.1
    self_collision_force_threshold: float = 10.0


@configclass
class RecoverySuccessCfg:
    height_ratio: float = 0.85
    upright_threshold: float = 0.93
    hold_s: float = 0.5


@configclass
class G1RecoveryEnvCfg:

    device: str = "cuda:0"

    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=6.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_RECOVERY_ROBOT_CFG,
        terrain_type="generator",
        terrain_generator=Flat_terrain,
        max_init_terrain_level=0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
        ),
    )

    control: RecoveryControlCfg = RecoveryControlCfg()

    observation: RecoveryObservationCfg = RecoveryObservationCfg()

    reset: RecoveryResetCfg = RecoveryResetCfg()

    reward: RecoveryRewardCfg = RecoveryRewardCfg()

    success: RecoverySuccessCfg = RecoverySuccessCfg()

    # physics frequency = 200 Hz, policy frequency = 50 Hz
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    def __post_init__(self):
        if self.control.action_scale <= 0.0:
            raise ValueError("Recovery action_scale must be positive.")
        if self.control.clip_actions != 1.0:
            raise ValueError("Recovery clip_actions must remain 1.0 in Stage 4.1.")
        if self.observation.history_length != 4:
            raise ValueError(
                "Recovery observation history_length must remain 4 in Stage 4.1."
            )
        if self.reset.random_yaw:
            raise ValueError("Recovery random_yaw must remain disabled in Stage 4.2b.")
        if self.reward.height_std <= 0.0 or self.reward.height_scale <= 0.0:
            raise ValueError("Recovery height reward parameters must be positive.")
        if self.reward.self_collision_force_threshold != 10.0:
            raise ValueError(
                "Recovery self-collision force threshold must remain 10.0 N."
            )
        expected_weights = (-2.5e-7, -10.0, -0.01, -0.1)
        actual_weights = (
            self.reward.joint_acc_weight,
            self.reward.joint_pos_limits_weight,
            self.reward.action_rate_weight,
            self.reward.self_collision_weight,
        )
        if actual_weights != expected_weights:
            raise ValueError(
                "Recovery V1 regularization weights must remain "
                f"{expected_weights}, got {actual_weights}."
            )
        if not 0.0 < self.success.height_ratio <= 1.0:
            raise ValueError("Recovery success height_ratio must be in (0, 1].")
        if not -1.0 <= self.success.upright_threshold <= 1.0:
            raise ValueError(
                "Recovery success upright_threshold must be in [-1, 1]."
            )
        if self.success.hold_s <= 0.0:
            raise ValueError("Recovery success hold_s must be positive.")


@configclass
class G1RecoveryAgentCfg(BaseAgentCfg):
    """Existing AMPPPO stack configured for standalone NPZ Recovery data."""

    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "g1_recovery"
    logger = "tensorboard"
    save_interval = 1000

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
        symmetry_cfg=None,
        rnd_cfg=None,
    )

    # Select the Stage-3 Recovery NPZ adapter without changing legacy AMP tasks.
    amp_data_class_name = "RecoveryAMPData"
    amp_motion_dir = os.path.join(
        LEGGED_LAB_ROOT,
        "envs/g1/datasets/npz/recovery_candidates_v3",
    )
    amp_transition_dt = 0.02
    amp_expected_fps = 50.0
    amp_sampling = "uniform_frame"
    amp_preload_transitions = True
    amp_num_preload_transitions = 200000

    amp_reward_coef = 0.3
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23

    def __post_init__(self):
        super().__post_init__()
        if self.amp_transition_dt != 0.02:
            raise ValueError("Recovery formal AMP transition_dt must remain 0.02 s.")
        if self.amp_reward_coef != 0.3:
            raise ValueError("Recovery amp_reward_coef must remain 0.3.")
        if self.amp_task_reward_lerp != 0.7:
            raise ValueError("Recovery amp_task_reward_lerp must remain 0.7.")


__all__ = [
    "G1RecoveryAgentCfg",
    "G1RecoveryEnvCfg",
    "G1_RECOVERY_BODY_NAMES",
    "RecoveryControlCfg",
    "RecoveryObservationCfg",
    "RecoveryRewardCfg",
    "RecoveryResetCfg",
    "RecoverySuccessCfg",
]
