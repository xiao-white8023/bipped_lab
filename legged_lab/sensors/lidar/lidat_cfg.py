from typing import Literal

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

from legged_lab.sensors.grouped_ray_caster import Mid360PatternCfg


@configclass
class LidarNoiseCfg:
    """Point-cloud noise applied after ray casting and before mapping."""

    enable: bool = False
    range_std_range_m: tuple[float, float] = (0.01, 0.03)
    point_dropout_prob_range: tuple[float, float] = (0.0, 0.05)
    translation_bias_range_m: tuple[float, float] = (-0.005, 0.005)
    rotation_bias_range_deg: tuple[float, float] = (-0.2, 0.2)


@configclass
class LidarCfg:
    enable_lidar: bool = False
    body_name: str = "torso_link"
    # Official G1 23-DoF MID-360 extrinsic relative to torso_link. The pi roll
    # describes the sensor coordinate convention, not an upside-down housing.
    offset_pos: tuple[float, float, float] = (0.0002835, 0.00003, 0.428434)
    offset_rot: tuple[float, float, float, float] = (
        0.0,
        0.9996733521,
        0.0,
        -0.0255575638,
    )
    ray_alignment: Literal["base", "yaw", "world"] = "base"
    update_period: float = 0.1
    pattern_cfg: PatternBaseCfg = Mid360PatternCfg()
    debug_vis: bool = False
    min_distance: float = 0.1
    max_distance: float = 5.0
    noise: LidarNoiseCfg = LidarNoiseCfg()
    mesh_prim_paths: list[str | MultiMeshRayCasterCfg.RaycastTargetCfg] = ["/World/ground"]
