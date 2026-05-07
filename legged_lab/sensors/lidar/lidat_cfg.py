from isaaclab.sensors import patterns
from isaaclab.utils import configclass


@configclass
class LidarCfg:
    enable_lidar: bool = False
    prim_body_name: str = "pelvis"
    # Adjusted offset for head-like placement relative to pelvis
    offset: tuple = (0.15, 0.0, 0.6)
    rotation: tuple = (1.0, 0.0, 0.0, 0.0)
    pattern_cfg = patterns.LidarPatternCfg(
        channels=64,  # Number of vertical beams
        horizontal_fov_range=(0, 360),  # Full 360 degree horizontal scan
        vertical_fov_range=(-90, 90),  # Full spherical vertical scan
        horizontal_res=0.5,  # Horizontal resolution in degrees
    )
    debug_vis: bool = False
    max_distance: float = 20.0
    mesh_prim_paths = ["/World"]
