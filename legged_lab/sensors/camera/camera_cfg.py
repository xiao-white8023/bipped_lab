from dataclasses import dataclass
from typing import Literal

import torch
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg as BaseCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.spawners import PreviewSurfaceCfg, SphereCfg
from isaaclab.utils import configclass

from .camera import Camera


@dataclass
class SensorNoiseCfg:
    """Configuration for sensor noise."""

    enable: bool = False
    mode: Literal["gaussian", "dropout", "combined"] = "gaussian"

    # Gaussian noise parameters
    depth_std: float = 0.01
    depth_std_multiplier: float = 0.01

    # Dropout noise parameters
    dropout_prob: float = 0.01
    dropout_value: float = 0.0


@configclass
class CameraCfg(BaseCameraCfg):
    class_type: type = Camera

    enable_depth_camera: bool = False
    prim_body_name: str = "pelvis/depth_camera" # 这里表示将相机安装在机器人的 pelvis（骨盆/躯干）部件上，命名为 depth_camera。

    # Camera parameters
    width: int = 480
    height: int = 270  # 480x270 是一个非常经典的低分辨率配置，能在保留足够的环境几何特征的同时，大幅度节省 GPU 显存和渲染时间
    max_range: float = 15.0
    min_range: float = 0.2 # 相机的有效探测距离（裁剪面）。最近只能看清 0.2 米外的东西，最远只能探测到 15 米。这模拟了真实深度相机的物理限制。

    '''
    告诉渲染器我们要提取什么数据。这里不要 RGB 彩色图像，只要深度数据。distance_to_image_plane 指的是物体到相机成像平面的垂直距离。
    '''

    data_types: list[str] = ["distance_to_image_plane"]
    '''
    相机的位姿偏移量（包含 xyz 平移和四元数旋转）。比如你想把相机在骨盆的基础上往前挪 10 厘米，往下倾斜一点，就在这里设置
    '''
    offset: BaseCameraCfg.OffsetCfg = BaseCameraCfg.OffsetCfg()
    spawn: PinholeCameraCfg = PinholeCameraCfg()
    sensor_noise: SensorNoiseCfg = SensorNoiseCfg()

    # Camera Visualization Configuration
    debug_vis: bool = False
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/CameraPointCloud",
        markers={
            "point": SphereCfg(
                radius=0.02,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            )
        },
    )
    visualizer_cfg.decimation = 10

    far_out_of_range_value = torch.inf
    near_out_of_range_value = torch.inf
