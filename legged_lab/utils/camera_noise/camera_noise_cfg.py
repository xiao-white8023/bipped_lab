import torch
from dataclasses import MISSING
from typing import Callable, Optional

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .camera_noise import (
    ImageNoiseModel,
    StructuredDepthFailureModel,
    depth_normalization,
    crop_and_resize,
    blind_spot_noise,
    range_based_gaussian_noise,
    distance_dependent_gaussian_noise,
    depth_artifact_noise,
    LatencyNoiseModel,
    SensorDeadNoiseModel
    )


@configclass
class ImageNoiseCfg(NoiseCfg):
    func: Callable[[torch.Tensor, NoiseCfg, torch.Tensor], torch.Tensor] | type[ImageNoiseModel] = ImageNoiseModel
    """ The callable function to apply noise to the image.
    The function should take two arguments:
        - the image in shape (N_, H, W, C) where N_ = len(env_ids)
        - the cfg object (as this configclass's object).
        - the env_ids tensor for specifying the environment ids
    """
    device: str | torch.device = "cpu"

@configclass
class DepthNormalizationCfg(ImageNoiseCfg):
    """Configuration for normalizing depth values to a specific range."""

    depth_range: tuple[float, float] = (0.0, 10.0)
    """将传进来的图片的深度值的范围，裁剪到（0,10） 为了方便归一化"""

    normalize: bool = True
    """是否把值归一化到（0，1）."""

    output_range: tuple[float, float] = (0.0, 1.0)
    """Range to normalize depth values to."""

    func = depth_normalization
    """执行函数"""

@configclass
class CropAndResizeCfg(ImageNoiseCfg):
    """Configuration for cropping and resizing images."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """The size to be cropped, corresponding to up, down, left, right, respectively."""
    """要裁剪的像素数 是上下左右的顺序"""

    resize_shape: tuple[int, int] = None
    """目标尺寸 (高度, 宽度)，None表示不缩放"""

    func = crop_and_resize

@configclass
class BlindSpotNoiseCfg(ImageNoiseCfg):
    """Configuration for adding blind spot noise (zeroing out regions of the image)."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """# (上, 下, 左, 右) 需置零的像素数"""

    func = blind_spot_noise

@configclass
class RangeBasedGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding range-based Gaussian noise to images."""
    # 如仅设 min_value=5.0，表示对所有≥5.0 的像素加噪声
    min_value: float | None = None
    # 噪声作用范围的最小值（None表示无下界）

    max_value: float | None = None
    # 噪声作用范围的最大值（None表示无上界）

    noise_std: float = 1.0
    # 高斯噪声的标准差（控制噪声强度）

    func = range_based_gaussian_noise


@configclass
class DistanceDependentGaussianNoiseCfg(ImageNoiseCfg):
    """Gaussian depth noise with standard deviation determined by metric distance."""

    near_std: float = 0.0
    far_std: float = 0.10
    distance_exponent: float = 2.0

    func = distance_dependent_gaussian_noise


@configclass
class DepthArtifactNoiseCfg(ImageNoiseCfg):
    artifacts_prob: float = 0.0001  # 每个像素点成为伪影中心的概率
    artifacts_height_mean_std: list[float] = [2, 0.5] # 伪影块高度的均值和标准差
    artifacts_width_mean_std: list[float] = [2, 0.5] # 伪影块宽度的均值和标准差
    noise_value: float = 0.0 # 噪声
    func = depth_artifact_noise

@configclass
class StructuredDepthFailureCfg(ImageNoiseCfg):
    """Configuration for large, structured depth failures that persist for multiple frames."""

    failure_probability: float = 0.02
    """Probability of starting a failure for each currently healthy environment."""

    failure_duration_range: tuple[int, int] = (3, 12)
    """Default duration range in camera frames. Minimum is intentionally multi-frame."""

    failure_modes: tuple[str, ...] = (
        "large_hole", # 大面积空洞
        "lower_occlusion",  # 深度图底部一大片区域被遮挡或不可见。
        "central_landing_occlusion", # 深度图中间偏下的落脚区域出现一块遮挡/缺失。
        "stripe_missing",  # 深度图中出现横向或纵向条纹缺失。
        "full_dropout",
        "consecutive_dropout",
        "freeze_frame",
        "distant_dust",
    )
    """Enabled structured failure modes."""

    failure_mode_probabilities: Optional[list[float]] = None
    """Sampling probability for each mode. None means uniform over failure_modes."""

    fill_value: float = 0.0
    """Depth value used for holes, occlusions and dropout."""

    large_hole_height_range: tuple[float, float] = (0.35, 0.75)
    large_hole_width_range: tuple[float, float] = (0.35, 0.80)
    lower_occlusion_height_range: tuple[float, float] = (0.35, 0.70)
    central_landing_height_range: tuple[float, float] = (0.35, 0.70)
    central_landing_width_range: tuple[float, float] = (0.35, 0.75)
    # 这两行确保了是中心区域被遮挡。
    central_landing_center_y_range: tuple[float, float] = (0.55, 0.85)
    central_landing_center_x_range: tuple[float, float] = (0.40, 0.60)

    stripe_width_range: tuple[float, float] = (0.08, 0.25)
    stripe_count_range: tuple[int, int] = (1, 3) # 条纹的数量
    stripe_orientations: tuple[str, ...] = ("vertical", "horizontal") # 条纹的方向

    consecutive_dropout_duration_range: tuple[int, int] = (6, 20)
    freeze_duration_range: tuple[int, int] = (4, 15)

    # 深度图中远距离区域受到灰尘、雾、反射、远距离测量不稳定等因素影响，出现随机扰动，甚至部分远处像素直接失效变成 0
    distant_dust_duration_range: tuple[int, int] = (8, 25)
    distant_dust_start: float = 1.5
    distant_dust_noise_std: float = 0.08 # 方差
    distant_dust_dropout_probability: float = 0.35 # 远距离droupt的概率
    distant_dust_fill_value: float = 0.0

    func: type[StructuredDepthFailureModel] = StructuredDepthFailureModel

@configclass
class LatencyNoiseCfg(ImageNoiseCfg):
    history_length: int = 5 # 定义延迟缓冲区的最大长度。意味着最多可以模拟 “过去 5 帧” 的延迟

    # sample frequency related settings
    '''
    重采样的触发模式：
    None: 从不重采样，延迟固定不变。
    "every_n_steps": 每隔固定步数重采样一次。
    "random_with_probability": 每一步都有一定概率随机重采样。
    '''
    sample_frequency: Optional[str] = None 
    # 当使用 "every_n_steps" 时，基础的重采样间隔步数
    sample_frequency_steps: int = 50  # used when sample_frequency is "every_n_steps"
    # 为了增加随机性，在基础步数上加减的偏移量范围
    sample_frequency_steps_offset: int = 5  # the offset for the sample frequency steps
    # 当使用 "random_with_probability" 时，每一步触发重采样的概率
    sample_probability: float = 0.1  # used when sample_frequency is "random_with_probability"

    # sample distribution related settings
    '''
    延迟的概率分布类型：
    "constant": 固定延迟。
    "uniform": 均匀分布（在范围内随机）。
    "normal": 正态分布（围绕均值波动）。
    "choice": 从一个预定义列表中选择。
    '''
    latency_distribution: Optional[str] = "constant"

    '''均匀或正态分布的取值范围 [min, max]'''
    latency_range: tuple[int, int] = (1, history_length)
    
    '''
    正态分布的 (均值, 标准差)
    '''
    latency_mean_std: tuple[float, float] = (3, 1)  # used when latency_distribution is "normal"
    # "choice" 模式下的可选延迟步数列表。
    latency_choices: list[int] = [1, 2, 3, 4, 5]  # used when latency_distribution is "choice"
    ''' "choice" 模式下对应列表中每个选项的概率（不设置则均匀随机） '''
    latency_choices_probabilities: Optional[list[float]] = (
        None  # probabilities for each choice, default to None (uniform distribution)
    )
        # "constant" 模式下的固定延迟步数
    latency_steps: int = 5  # used when latency_distribution is "constant"

    func: type[LatencyNoiseModel] = LatencyNoiseModel

@configclass
class SensorDeadNoiseCfg(ImageNoiseCfg):
    """Configuration for adding sensor dead behavior, which might be autonomous restarted.
    Thus causing some frames of non-refreshed data.
    """
    # 传感器失效的概率
    dead_probability: float = 0.01
    """The probability of the sensor dead."""

    '''
    # 2. 传感器失效后，数据不刷新的帧数（默认90帧，60Hz下对应1.5秒）
    #    支持两种格式：
    #    - 单个int：固定失效帧数
    #    - list[int]：从列表中均匀随机选择失效帧数
    '''
    dead_frames: int | list[int] = 90  # 1.5 second at 60Hz
   
    func = SensorDeadNoiseModel
