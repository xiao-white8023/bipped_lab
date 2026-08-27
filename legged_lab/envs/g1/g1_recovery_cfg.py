# ================================================
# G1 23dof recovery
# ================================================
from isaaclab.utils import configclass

from legged_lab.assets.g1.g1_29 import G1_23CFG
from legged_lab.envs.base.base_config import (
    BaseSceneCfg,
    HeightScannerCfg,
    PhysxCfg,
    SimCfg,
)
from legged_lab.terrains import Flat_terrain


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
class G1RecoveryEnvCfg:

    device: str = "cuda:0"

    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=6.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_23CFG,
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


__all__ = [
    "G1RecoveryEnvCfg",
    "RecoveryControlCfg",
    "RecoveryObservationCfg",
]
