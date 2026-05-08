from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Sequence

from isaacsim.core.utils.torch.maths import torch_rand_float
from torchvision.transforms import GaussianBlur
from legged_lab.utils.buffer import AsyncDelayBuffer

if TYPE_CHECKING:
    from .camera_noise_cfg import (
        ImageNoiseCfg,
        DepthNormalizationCfg,
        CropAndResizeCfg,
        BlindSpotNoiseCfg,
        RangeBasedGaussianNoiseCfg,
        DepthArtifactNoiseCfg,
        LatencyNoiseCfg,
        SensorDeadNoiseCfg
    )


class ImageNoiseModel:
    """This serves as an example of a noise model for images.
    It should be replaced with a specific noise model implementation.
    """

    def __init__(self, cfg: ImageNoiseCfg, num_envs: int = 1, device: str | torch.device = "cpu"):
        """Initialize the noise model with the configuration.

        Args:
            cfg: The configuration for the noise model.
            num_envs: The number of environments (default is 1).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    def __call__(self, data: torch.Tensor, cfg: ImageNoiseCfg, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Apply noise to the image data.

        Args:
            data: The image data in shape (N, H, W, C).
            cfg: The configuration for the noise model.
            env_ids: The environment IDs for the current image sensor.

        Returns:
            The noisy image data.
        """
        return data

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model state if needed.

        Args:
            env_ids: The environment IDs to reset. If None, reset all environments.
        """
        pass

'''
实现了立体深度相机伪影（Artifacts）的模拟功能，通过在深度图上随机生成矩形失效区域（填充为 noise_value，默认为 0），来模拟真实深度传感器在特定区域的深度丢失现象

artifacts_prob: 每个像素点成为伪影 “种子” 的概率
artifacts_height_mean_std: 伪影块高度的均值和标准差（例如 [2, 0.5]）。
artifacts_width_mean_std: 伪影块宽度的均值和标准差。
N, H, W, _ = data.shape: 解包输入数据的维度，忽略通道数 C（因为深度图通常是单通道）。
'''
def _add_depth_artifacts(
    data, artifacts_prob, artifacts_height_mean_std, artifacts_width_mean_std, device, noise_value=0.0
):
    """Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
    should be an artifacts, the mask is 1.
    """

    N, H, W, _ = data.shape

    def _clip(data, dim):
        return torch.clip(data, 0.0, (H, W)[dim])  #dim=0 对应高度维度（上限为 H），dim=1 对应宽度维度（上限为 W） 将数据裁剪到[0, H]或[0, W]

    # random patched artifacts
    # 生成一个形状为 (N, H * W)、值在 [0.0, 1.0] 之间均匀分布的随机浮点数张量
    
    '''功能：将重塑后的随机张量与 artifacts_prob（伪影中心概率）进行逐元素比较，生成一个布尔张量（True/False）。
    逻辑：
    对于张量中的每个像素值 x：
        如果 x < artifacts_prob，结果为 True（标记为伪影中心）；
        如果 x ≥ artifacts_prob，结果为 False（不标记）。
    概率意义：
    artifacts_prob 是一个 0~1 之间的超参数（例如 0.01），控制伪影中心出现的概率：
        值越大，越多像素会被标记为伪影中心（伪影越密集）；
        值越小，伪影中心越稀疏
    
    '''
    artifacts_mask = torch_rand_float(0.0, 1.0, (N, H * W), device=device).view(N, H, W) < artifacts_prob
    artifacts_mask = artifacts_mask & (data[:, :, :, 0] > 0.0) # 确保深度值大于0的像素点才能做为中心伪影点
    # 提取伪影中心的坐标，并将坐标转换为浮点数
    # torch.nonzero返回输入张量中非零元素（或布尔张量中 True 元素）的索引
    '''
    输入 artifacts_mask：形状为 (N, H, W)（批量大小 × 图像高度 × 图像宽度）。
    输出：形状为 (n, 3) 的张量，其中：

    n 是 artifacts_mask 中 True 的个数（即伪影中心的数量）；
    每一行是一个伪影中心的坐标，格式为 [batch_idx, height_idx, width_idx]
    '''
   
    artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32)  # (n_, 3) n_ <= N * H * W

    if len(artifacts_coord) == 0:
        return data

    '''
    这段代码的核心作用是为每个伪影独立采样 “高度” 和 “宽度”，尺寸从 ** 高斯分布（正态分布）** 中随机生成，并裁剪到合理范围（不能为负，也不能超过图像大小）
    '''
    # artifacts_size是两个张量 分别是每一个伪影的高度 和每一个伪影的宽度
    artifacts_size = (
        torch.clip(
            artifacts_height_mean_std[0]
            + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_height_mean_std[1],
            0.0, # 伪影的高度不能是负数
            H, # 伪影的高度不能超过图片的高度
        ),
        torch.clip(
            artifacts_width_mean_std[0]
            + torch.randn((artifacts_coord.shape[0],), device=device) * artifacts_width_mean_std[1],
            0.0,# 伪影的宽不能是负数
            W,# 伪影的宽度不能超过图片的宽度
        ),
    )  # (n_,), (n_,)

    '''
    artifacts_coord[:, 1] 获取伪影中心点的H坐标 由于是中心点 且artifacts_top是求的顶端的坐标 则需要减artifacts_size[0] / 2
    artifacts_top是一维的列表
    '''
    artifacts_top = _clip(artifacts_coord[:, 1] - artifacts_size[0] / 2, 0)
    artifacts_left = _clip(artifacts_coord[:, 2] - artifacts_size[1] / 2, 1)
    artifacts_bottom = _clip(artifacts_coord[:, 1] + artifacts_size[0] / 2, 0)
    artifacts_right = _clip(artifacts_coord[:, 2] + artifacts_size[1] / 2, 1)

    # create one-hot encoding for environment IDs
    # 取 artifacts_coord 的第 0 列（所有行的第 0 个元素），也就是每个伪影的 batch_idx（属于第几个机器人）。
    # 输出 env_ids：形状为 (n_,) 的一维张量，每个元素是对应伪影的 batch 样本索引（比如 [0, 0, 1] 表示第 1、2 个伪影属于第 0 张图，第 3 个伪影属于第 1 张图）
    # 取 artifacts_coord 的第 0 列（所有行的第 0 个元素），也就是每个伪影的 batch_idx（属于第几个机器人）。
    env_ids = artifacts_coord[:, 0].long() 
    # 输出 env_onehot：形状为 (n_, N) 的全零张量，行对应 “伪影”，列对应 “batch 样本”。 N：batch size
    env_onehot = torch.zeros((len(artifacts_coord), N), device=device)
    '''
    torch.arange(len(artifacts_coord))：生成 0 到 n_-1 的一维张量
    '''
    env_onehot[torch.arange(len(artifacts_coord)), env_ids] = 1.0 # 将每一个伪影所对应的那张图置为1

    # batch generate all artifacts
    num_artifacts = len(artifacts_coord)  #统计总共有多少个伪影
    tops_expanded = artifacts_top[:, None, None] # artifacts_top从一维列表变成3维的 shape(3,1,1) 为了后续的广播计算
    lefts_expanded = artifacts_left[:, None, None] # 同上
    bottoms_expanded = artifacts_bottom[:, None, None]# 同上
    rights_expanded = artifacts_right[:, None, None]# 同上

    # build the source patch
    # 给每个伪影做一个「中间实心、边缘空心」的 25x25 小模板
    # source_patch 四维形状 (num_artifacts, 1, 25, 25)
    source_patch = torch.zeros((num_artifacts, 1, 25, 25), device=device)
    # 选中所有伪影、所有通道，高度方向：第 1 行～第 23 行宽度方向：第 1 列～第 23 列 赋值为1
    source_patch[:, :, 1:24, 1:24] = 1.0
    '''
    中间 1、外圈 0 的模板，后续用 grid_sample 拉伸时，会自动生成平滑的边缘，不是生硬的直角，模拟的伪影更真实。
    不用给每个伪影单独画矩形，用这个小模板拉伸到任意大小，计算更快
    数值 1 的区域 = 后续要把深度图置零的伪影区域
    数值 0 的区域 = 保留原深度图
    '''
    # build the grid
    grid = torch.zeros((num_artifacts, H, W, 2), device=device)
    grid[..., 0] = torch.linspace(-1, 1, W, device=device).view(1, 1, W)
    grid[..., 1] = torch.linspace(-1, 1, H, device=device).view(1, H, 1)
    grid[..., 0] = (grid[..., 0] * W + W - rights_expanded - lefts_expanded) / (rights_expanded - lefts_expanded)
    grid[..., 1] = (grid[..., 1] * H + H - bottoms_expanded - tops_expanded) / (bottoms_expanded - tops_expanded)

    # sample using the grid and form the artifacts for the entire depth image
    # F.grid_sample 是 PyTorch 的 网格采样函数
    all_artifacts = F.grid_sample(
        source_patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(
        1
    )  # (num_artifacts, H, W)

    # combine the artifacts with the environment one-hot encoding
    # env_onehot: (num_artifacts, N)
    # all_artifacts: (num_artifacts, H, W)
    # final_masks: (N, H, W)
    final_masks = torch.einsum("an,ahw->nhw", env_onehot, all_artifacts)
    final_masks = torch.clamp(final_masks, 0, 1)

    data = data.squeeze(-1)  # (N, H, W)
    data = data * (1 - final_masks) + final_masks * noise_value
    data = data.unsqueeze(-1)

    return data

def depth_artifact_noise(
    data: torch.Tensor,
    cfg: DepthArtifactNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Add depth artifacts to the image data."""
    return _add_depth_artifacts(
        data,
        artifacts_prob=cfg.artifacts_prob,
        artifacts_height_mean_std=cfg.artifacts_height_mean_std,
        artifacts_width_mean_std=cfg.artifacts_width_mean_std,
        device=cfg.device,
        noise_value=cfg.noise_value,
    )

def depth_normalization(
    data: torch.Tensor, cfg: DepthNormalizationCfg, env_ids: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    """Clip the depth values to  givenrange and choose whether to normalize them to [0, 1] range."""

    if data.dim() == 4 and data.shape[-1] == 1:
        # Convert from (N, H, W, C) to (N, C, H, W)
        # 维度转换的意义：兼容不同框架的输入格式（如 OpenCV 读入是 (H,W,C)，PyTorch 卷积用 (C,H,W)）。
        data = data.permute(0, 3, 1, 2)

    # Clip depth values to [min_depth, max_depth]
    min_depth = cfg.depth_range[0]
    max_depth = cfg.depth_range[1]
    data = data.clip(min_depth, max_depth)

    if cfg.normalize:
        # Normalize depth values to [0, 1]
        data = (data - min_depth) / (max_depth - min_depth)
        # Normalize to output range
        data = data * (cfg.output_range[1] - cfg.output_range[0]) + cfg.output_range[0]

    if len(data.shape) == 4 and data.shape[1] == 1:
        # Convert back to (N, H, W, C)
        data = data.permute(0, 2, 3, 1)

    return data

def crop_and_resize(
    data: torch.Tensor,
    cfg: CropAndResizeCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Crop and resize the input image tensor."""
    # Crop the image
    crop_region = cfg.crop_region
    start_up = crop_region[0]
    end_down = data.shape[1] - crop_region[1]
    start_left = crop_region[2]
    end_right = data.shape[2] - crop_region[3]
    cropped = data[:, start_up:end_down, start_left:end_right, :]
    # Resize the image
    if cfg.resize_shape is None:
        return cropped
    else:
        cropped = cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(cropped, size=cfg.resize_shape, mode="bilinear", align_corners=False)
        resized = resized.permute(0, 2, 3, 1)
        return resized
    
def blind_spot_noise(
    data: torch.Tensor,
    cfg: BlindSpotNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Remove data in the leftmost columns to mimic blind spot of stereo-matching."""
    # Crop the image
    crop_region = cfg.crop_region
    start_up = crop_region[0] # 上边置0的结束位置
    end_down = data.shape[1] - crop_region[1] # 下边置0的开始位置
    start_left = crop_region[2] # 左边置0的结束位置
    end_right = data.shape[2] - crop_region[3]# 右边置0的开始位置
    # Set the cropped region to 0
    data_modified = data.clone()
    data_modified[:, :start_up, :, :] = 0.0
    data_modified[:, end_down:, :, :] = 0.0
    data_modified[:, :, :start_left, :] = 0.0
    data_modified[:, :, end_right:, :] = 0.0
    return data_modified

'''
这段代码实现了基于范围的选择性高斯噪声注入功能，仅对图像中落在指定值范围内的像素添加高斯噪声，常用于模拟特定物理条件下的传感器噪声（如深度图中仅对特定距离范围添加噪声）
'''
def range_based_gaussian_noise(
    data: torch.Tensor,
    cfg: RangeBasedGaussianNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Apply gaussian noise to the data where the original value is in the range [min_value, max_value]
    if min_value or max_value is None, the boundary is not considered.
    """
    # 生成与输入同形状、同设备的高斯噪声
    N, H, W, C = data.shape
    noise = torch.randn((N, H, W, C), device=data.device) * cfg.noise_std # # 噪声均值0，标准差noise_std
    # 创建与输入同形状的噪声作用掩码（标记哪些像素需要加噪声）
    apply_mask = torch.ones((N, H, W, C), device=data.device, dtype=bool)
    # 获取data中可以加噪声的地方
    if cfg.min_value is not None:
        apply_mask = apply_mask & (data >= cfg.min_value)
    if cfg.max_value is not None:
        apply_mask = apply_mask & (data <= cfg.max_value)

    noisy_data = data + noise * apply_mask # 加入噪声

    return noisy_data

class LatencyNoiseModel(ImageNoiseModel):
    def __init__(self, cfg: LatencyNoiseCfg, num_envs, device):
        super().__init__(cfg, num_envs, device)

        # check if the cfg is valid
        if cfg.latency_distribution == "choice" and max(cfg.latency_choices) > cfg.history_length:
            raise RuntimeError(f"Latency choices {cfg.latency_choices} exceed the history length {cfg.history_length}.")
        if cfg.latency_distribution == "constant" and cfg.latency_steps > cfg.history_length:
            raise RuntimeError(f"Latency steps {cfg.latency_steps} exceed the history length {cfg.history_length}.")
        if (cfg.latency_choices == "uniform" or cfg.latency_choices == "normal") and (
            (cfg.latency_range[1] > cfg.history_length)
            or (cfg.latency_range[0] < 0)
            or cfg.latency_range[0] > cfg.latency_range[1]
        ):
            raise RuntimeError(f"Latency range {cfg.latency_range} is invalid.")

        self.delay_buffer = AsyncDelayBuffer(cfg.history_length, num_envs, device)
        self.cfg = cfg
        self.num_envs = num_envs

        # independent counters for each environment
        self.env_step_counters = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.last_resample_steps = torch.zeros(num_envs, dtype=torch.int, device=device)

        # set different resamplpe intervals for each environment
        if cfg.sample_frequency == "every_n_steps":
            self.resample_intervals = self._generate_resample_intervals()

        # initialize the delay settings
        self._resample_delays(torch.arange(num_envs, device=device))

    def __call__(self, data, cfg, env_ids: torch.Tensor | Sequence[int]):
        # convert env_ids to tensor
        if isinstance(env_ids, Sequence):
            env_ids = torch.tensor(env_ids, device=self.device)

        if data.shape[0] != len(env_ids):
            raise RuntimeError(
                f"Data batch shape {data.shape[0]} does not match the number of environments {len(env_ids)}."
            )

        # update step counters
        self.env_step_counters[env_ids] += 1

        # check environments that should resample delays
        should_resample = self._should_resample(env_ids)

        if torch.any(should_resample):
            resample_env_ids = env_ids[should_resample]
            self._resample_delays(resample_env_ids)
            self.last_resample_steps[resample_env_ids] = self.env_step_counters[resample_env_ids]

        # get the delayed data
        delayed = self.delay_buffer.compute(data, batch_ids=env_ids.tolist())
        return delayed

    def _generate_resample_intervals(self, env_ids: Sequence[int] | None = None):
        """Generate resample intervals for each environment"""
        base_interval = self.cfg.sample_frequency_steps
        offset_range = self.cfg.sample_frequency_steps_offset
        if env_ids is None:
            offsets = torch.randint(-offset_range, offset_range + 1, (self.num_envs,), device=self.device)
        else:
            offsets = torch.randint(-offset_range, offset_range + 1, (len(env_ids),), device=self.device)
        intervals = base_interval + offsets
        intervals = intervals.clamp(min=1)  # ensure intervals are at least 1
        return intervals

    def _should_resample(self, env_ids: torch.Tensor):
        """Check which environments should resample delays."""
        if self.cfg.sample_frequency is not None:
            # Sample new delays based on the configured frequency
            if self.cfg.sample_frequency == "every_n_steps":
                # Resample every n steps
                current_steps = self.env_step_counters[env_ids]
                last_resample_steps = self.last_resample_steps[env_ids]
                intervals = self.resample_intervals[env_ids]
                return current_steps - last_resample_steps >= intervals

            elif self.cfg.sample_frequency == "random_with_probability":
                prob = self.cfg.sample_probability
                return torch.rand(len(env_ids), device=self.device) < prob

        # do not resample by default
        return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    def _resample_delays(self, env_ids: torch.Tensor):
        """Resample the delays based on the configured distribution."""

        num_envs_to_resample = len(env_ids)

        if self.cfg.latency_distribution == "uniform":
            # Uniform distribution of delays
            new_delays = torch.randint(
                self.cfg.latency_range[0],
                self.cfg.latency_range[1] + 1,
                (num_envs_to_resample,),
                dtype=torch.int,
                device=self.device,
            )
        elif self.cfg.latency_distribution == "normal":
            new_delays = (
                torch.normal(
                    mean=self.cfg.latency_mean_std[0],
                    std=self.cfg.latency_mean_std[1],
                    size=(num_envs_to_resample,),
                    device=self.device,
                )
                .round()
                .int()
                .clamp(
                    min=self.cfg.latency_range[0],
                    max=self.cfg.latency_range[1],
                )
            )
        elif self.cfg.latency_distribution == "choice":
            # Choose delays from a predefined set
            choices = torch.tensor(self.cfg.latency_choices, dtype=torch.int, device=self.device)
            if self.cfg.latency_choices_probabilities is not None:
                prob = torch.tensor(self.cfg.latency_choices_probabilities, device=self.device)
                indices = torch.multinomial(prob, num_envs_to_resample, replacement=True)
            else:
                indices = torch.randint(0, len(choices), (num_envs_to_resample,), device=self.device)
            new_delays = choices[indices]
        elif self.cfg.latency_distribution == "constant":
            new_delays = torch.full(
                (num_envs_to_resample,), self.cfg.latency_steps, dtype=torch.int, device=self.device
            )

        self.delay_buffer.set_time_lag(new_delays, env_ids.tolist())

    def reset(self, env_ids: Sequence[int] | None = None):
        """reset the noise model state for given environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        env_ids_tensor = torch.tensor(env_ids, device=self.device)

        # reset the environment step counters
        self.env_step_counters[env_ids_tensor] = 0
        self.last_resample_steps[env_ids_tensor] = 0

        # reset resample intervals if "every_n_steps" is used
        if self.cfg.sample_frequency == "every_n_steps":
            new_intervals = self._generate_resample_intervals(env_ids)
            self.resample_intervals[env_ids_tensor] = new_intervals

        # reset the delay buffer
        self.delay_buffer.reset(env_ids)

        # resample delays for given environments
        self._resample_delays(env_ids_tensor)


class SensorDeadNoiseModel(ImageNoiseModel):
    def __init__(self, cfg: SensorDeadNoiseCfg, num_envs, device):
        """Simulating when the sensor is dead and restarting, this may lead to several frames of non-refreshed data."""
        super().__init__(cfg, num_envs, device)
        # 存储每个环境 “上一次有效” 的图像数据。当传感器失效时，就从这个缓冲区里取旧数据返回，模拟 “数据不刷新”
        self._data_buffer = None
        
        # 它是一个一维张量，长度等于 num_envs，每个元素对应一个环境：
        # 如果元素值为 0：表示该环境的传感器正常；
        # 如果元素值为 5：表示该环境的传感器还会失效 5 帧，5 帧后恢复正常。
        self._remain_dead_frames = torch.zeros(num_envs, device=device)
        #  “失效帧数的可选值”
        self._dead_frames_options = (
            self.cfg.dead_frames
            if isinstance(self.cfg.dead_frames, int)
            else torch.tensor(self.cfg.dead_frames, device=device)
        )

    '''
    直接调用对象本身：g() （注意！直接在对象后面加括号，不需要 .方法名 了）
    '''
    def __call__(self, data, cfg: SensorDeadNoiseCfg, env_ids: torch.Tensor | Sequence[int]):
        
        # 创建_data_buffer，(self.num_envs, 图像高度, 图像宽度, 通道数)  内容：全是 0
        if self._data_buffer is None:
            self._data_buffer = torch.zeros_like(data[0]).unsqueeze(0).repeat(self.num_envs, *([1] * (data.ndim - 1)))

        # determine if the sensor is dead this time.
        could_be_dead_mask = self._remain_dead_frames[env_ids] <= 0
        dead_this_time_mask = torch.logical_and(
            torch.rand(env_ids.shape[0], device=self.device) < self.cfg.dead_probability,
            could_be_dead_mask,
        )
        dead_frames = (
            self.cfg.dead_frames
            if isinstance(self.cfg.dead_frames, int)
            else torch.randint(
                len(self._dead_frames_options),
                size=(len(env_ids),),
                device=self.device,
            )
        )
        self._remain_dead_frames[env_ids] = torch.where(
            dead_this_time_mask, dead_frames, self._remain_dead_frames[env_ids] - 1
        )
        self._remain_dead_frames[env_ids].clamp_(min=0)

        # refresh the data buffer if it is not dead.
        data_to_refresh_mask = self._remain_dead_frames[env_ids] <= 0  # (len(env_ids),)
        buffer_to_refresh_mask = self._remain_dead_frames <= 0  # (self.num_envs,)
        self._data_buffer[buffer_to_refresh_mask] = data[data_to_refresh_mask]
        return self._data_buffer[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None):
        self._remain_dead_frames[env_ids] = 0
        if self._data_buffer is not None:
            self._data_buffer[env_ids] = 0