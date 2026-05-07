import torch
from dataclasses import MISSING
from typing import Callable, Optional

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .camera_noise import (
    ImageNoiseModel,
    
    SensorDeadNoiseModel,
    blind_spot_noise,
    crop_and_resize,
    depth_artifact_noise,
    depth_contour_noise,
    depth_normalization,
    depth_sky_artifact_noise,
    depth_stero_noise,
    gaussian_blur_noise,
    random_gaussian_noise,
    range_based_gaussian_noise,
    stereo_too_close_noise,
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
class SensorDeadNoiseCfg(ImageNoiseCfg):
    """Configuration for adding sensor dead behavior, which might be autonomous restarted.
    Thus causing some frames of non-refreshed data.
    """
    dead_probability: float = 0.01
    """The probability of the sensor dead."""
    dead_frames: int | list[int] = 90  # 1.5 second at 60Hz
    """The number of frames to be non-refreshed (before the sensor is restarted).
    Can be a single number or a list of numbers to be uniformly selected from.
    """
    func = SensorDeadNoiseModel

@configclass
class DepthSteroNoiseCfg(ImageNoiseCfg):
    stero_far_distance: float = 2.0
    stero_min_distance: float = 0.12  # when using (240,424) resolution
    stero_far_noise_std: float = 0.08  # the noise std of pixels that are greater than stero_far_noise_distance
    stero_near_noise_std: float = 0.02  # the noise std of pixels that are less than stero_far_noise_distance

    stero_full_block_artifacts_prob: float = (
        0.001  # the probability of adding artifacts to pixels that are less than stero_min_distance
    )
    stero_full_block_values: list = [0.0, 0.25, 0.5, 1.0, 3.0]
    stero_full_block_height_mean_std: list = [62, 1.5]
    stero_full_block_width_mean_std: list = [3, 0.01]

    stero_half_block_spark_prob: float = 0.02
    stero_half_block_value: int = 3000  # to the maximum value directly

    func = depth_stero_noise

@configclass
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    """Configuration for adding Gaussian blur noise to images."""

    kernel_size: int = 3
    """The size of the Gaussian kernel. It should be an odd number."""

    sigma: float = 1.0
    """The standard deviation of the Gaussian kernel."""

    func = gaussian_blur_noise

@configclass
class RangeBasedGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding range-based Gaussian noise to images."""

    min_value: float | None = None
    """The minimum value of the range."""

    max_value: float | None = None
    """The maximum value of the range."""

    noise_std: float = 1.0
    """The standard deviation of the Gaussian noise."""

    func = range_based_gaussian_noise