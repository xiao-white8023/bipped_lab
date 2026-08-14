from __future__ import annotations

import math
import random
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import TYPE_CHECKING

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
        DistanceDependentGaussianNoiseCfg,
        DepthArtifactNoiseCfg,
        StructuredDepthFailureCfg,
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


def _as_env_ids(env_ids: torch.Tensor | Sequence[int], device: str | torch.device) -> torch.Tensor:
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.tensor(env_ids, device=device, dtype=torch.long)


def _validate_duration_range(name: str, duration_range: tuple[int, int]):
    if duration_range[0] < 2 or duration_range[1] < duration_range[0]:
        raise RuntimeError(f"{name} must be a valid multi-frame range, got {duration_range}.")


def _sample_int_range(duration_range: tuple[int, int], device: str | torch.device) -> torch.Tensor:
    return torch.randint(duration_range[0], duration_range[1] + 1, (1,), device=device, dtype=torch.long)[0] # (1,)表示生成一个形状为 [1] 的张量，也就是只采样一个数。 [0]的作用是把采样出来的数字取出来变成标量


def _sample_ratio(ratio_range: tuple[float, float]) -> float:
    return random.uniform(ratio_range[0], ratio_range[1]) # 从一个范围里随机采样一个浮点数


class StructuredDepthFailureModel(ImageNoiseModel):
    """Stateful depth failure model for large structured corruptions.

    Each environment either stays healthy or enters one sampled failure mode. Once a
    failure starts it persists for multiple camera frames, which matches bursty
    sensor failures better than independent per-frame pixel dropout.
    """

    MODE_TO_INDEX = {
        "large_hole": 0,
        "lower_occlusion": 1,
        "central_landing_occlusion": 2,
        "stripe_missing": 3,
        "full_dropout": 4,
        "consecutive_dropout": 5,
        "freeze_frame": 6,
        "distant_dust": 7,
    }

    def __init__(self, cfg: StructuredDepthFailureCfg, num_envs: int = 1, device: str | torch.device = "cpu"):
        super().__init__(cfg, num_envs, device)
        self.mode_names = tuple(cfg.failure_modes)
        unknown_modes = set(self.mode_names) - set(self.MODE_TO_INDEX)
        if unknown_modes:
           raise RuntimeError(f"Unknown structured depth failure modes: {sorted(unknown_modes)}.")
        if len(self.mode_names) == 0:
            raise RuntimeError("StructuredDepthFailureCfg.failure_modes must not be empty.")

        _validate_duration_range("failure_duration_range", cfg.failure_duration_range)
        _validate_duration_range("consecutive_dropout_duration_range", cfg.consecutive_dropout_duration_range)
        _validate_duration_range("freeze_duration_range", cfg.freeze_duration_range)
        _validate_duration_range("distant_dust_duration_range", cfg.distant_dust_duration_range)

        # 噪声状态模式索引 [0,1,2,3,4,5,6,7] 每一个数字都代表一种索引
        self.mode_indices = torch.tensor(
            [self.MODE_TO_INDEX[name] for name in self.mode_names], dtype=torch.long, device=device
        )
        if cfg.failure_mode_probabilities is None:
            self.mode_probabilities = torch.ones(len(self.mode_names), device=device) / len(self.mode_names) # 这样就保证当不设置每一个模式的概率时，确保每一个模式默认有相同的触发概率
        else:
            if len(cfg.failure_mode_probabilities) != len(self.mode_names):
                raise RuntimeError("failure_mode_probabilities must match failure_modes length.")
            probabilities = torch.tensor(cfg.failure_mode_probabilities, dtype=torch.float, device=device)
            if torch.any(probabilities < 0) or torch.sum(probabilities) <= 0:
                raise RuntimeError("failure_mode_probabilities must be non-negative and sum to a positive value.")
            self.mode_probabilities = probabilities / torch.sum(probabilities) # 这样就保证了即使所有的模式的出发概率大于1,也不会报错

        self.active_modes = torch.full((num_envs,), -1, dtype=torch.long, device=device) # 这一行创建了一个长度为 num_envs 的 tensor，用来记录每个环境当前激活的失效模式，其中当值为-1的时候 说明没有失效模式
        self.remaining_frames = torch.zeros(num_envs, dtype=torch.long, device=device) # 记录当前失效模式还要持续多少帧
        self.mask_buffer = None # 缓存 遮挡/丢失区域的 mask
        self.freeze_buffer = None # 缓存冻结帧图像

    def __call__(self, data: torch.Tensor, cfg: StructuredDepthFailureCfg, env_ids: torch.Tensor | Sequence[int]):
        env_ids = _as_env_ids(env_ids, self.device)
        # depth的shape是[N,H,W,C]
        if data.shape[0] != env_ids.numel(): # .numel()返回张量中所有元素的总个数
            raise RuntimeError(f"Data batch shape {data.shape[0]} does not match env_ids length {env_ids.numel()}.")
        self._ensure_buffers(data) # 执行完毕之后 self.mask_buffer变成了形状是(self.num_envs,H,W,C) 全0二值数组，self.freeze_buffer变成了形状是(self.num_envs,H,W,C)全0数组
        self._maybe_start_failures(data, env_ids)

        output = data.clone()
        active_mask = self.remaining_frames[env_ids] > 0
        if active_mask.any():
            active_env_ids = env_ids[active_mask]
            active_rows = active_mask.nonzero(as_tuple=False).flatten()
            active_modes = self.active_modes[active_env_ids]

            freeze_rows = active_rows[active_modes == self.MODE_TO_INDEX["freeze_frame"]]
            if freeze_rows.numel() > 0:
                freeze_env_ids = env_ids[freeze_rows]
                output[freeze_rows] = self.freeze_buffer[freeze_env_ids]

            dust_rows = active_rows[active_modes == self.MODE_TO_INDEX["distant_dust"]]
            if dust_rows.numel() > 0:
                output[dust_rows] = self._apply_distant_dust(output[dust_rows], cfg)

            mask_rows = active_rows[
                (active_modes != self.MODE_TO_INDEX["freeze_frame"])
                & (active_modes != self.MODE_TO_INDEX["distant_dust"])
            ]
            if mask_rows.numel() > 0:
                mask_env_ids = env_ids[mask_rows]
                masks = self.mask_buffer[mask_env_ids].to(dtype=torch.bool)
                fill = torch.full_like(output[mask_rows], cfg.fill_value)
                output[mask_rows] = torch.where(masks, fill, output[mask_rows])

            self.remaining_frames[active_env_ids] -= 1
            finished_env_ids = active_env_ids[self.remaining_frames[active_env_ids] <= 0]
            if finished_env_ids.numel() > 0:
                self.active_modes[finished_env_ids] = -1
                self.remaining_frames[finished_env_ids] = 0
                self.mask_buffer[finished_env_ids] = 0

        return output

    def _ensure_buffers(self, data: torch.Tensor):
        full_shape = (self.num_envs, *data.shape[1:]) # data.shape[1:]的意思是 变成[H,W,C], full_shape=[self.num_envs,H,W,C]
        if self.mask_buffer is None or tuple(self.mask_buffer.shape) != full_shape:
            self.mask_buffer = torch.zeros(full_shape, dtype=torch.bool, device=self.device) # 布尔
            self.freeze_buffer = torch.zeros(full_shape, dtype=data.dtype, device=self.device)

    # 检查当前这一批环境里，哪些环境可以开始新的深度图失效；然后按概率决定是否触发失效；如果触发，就随机采样一种失效模式，并为该环境初始化对应的持续时间、mask 或冻结帧。
    def _maybe_start_failures(self, data: torch.Tensor, env_ids: torch.Tensor):
        inactive = self.remaining_frames[env_ids] <= 0  # 这些环境当前是空闲状态，也就是没有正在持续的失效。
        # 当前环境中没有正在进行的失效 并且给这些环境生成的随机数小于失效的概率，则对这些环境实施失效模式
        # 当前环境是空闲的，没有正在失效；
        # 当前环境随机数小于故障触发概率。starts = tensor([False, True, False, True])
        starts = inactive & (torch.rand(env_ids.numel(), device=self.device) < self.cfg.failure_probability)
        if not starts.any():
            return

        # nonzero用于获取张量内所有非零元素的索引位置
        start_rows = starts.nonzero(as_tuple=False).flatten() # 找到对应的非零元素的索引
        start_env_ids = env_ids[start_rows] # 转换成真实环境编号
        
        sampled_positions = torch.multinomial(self.mode_probabilities, start_env_ids.numel(), replacement=True) # 这一行是为每一个新触发故障的环境，随机选择一种失效模式。 # self.mode_probabilities 是每种失效模式的概率，start_env_ids.numel()表示本次有多少个环境开始故障，replacement=True表示又放回的采样。
        sampled_modes = self.mode_indices[sampled_positions]  # 找出对应的失效模式
        self.active_modes[start_env_ids] = sampled_modes  # 记录每一个环境对应的失效模式对应的数字
        # 这一行把这些新触发故障环境的 mask 清零：为这些环境可能之前触发过别的故障，mask_buffer 里可能残留旧的遮挡区域。
        # 新故障开始前必须先清空旧 mask，否则新旧故障区域可能混在一起。
        # 例如，之前 env 100 是 large_hole，mask 中某个大块区域是 True。
        # 现在 env 100 新触发 stripe_missing，如果不先清零，就可能同时保留旧的大洞和新的条纹缺失。
        self.mask_buffer[start_env_ids] = 0  

        for row, env_id, mode in zip(start_rows.tolist(), start_env_ids.tolist(), sampled_modes.tolist()):
            self.remaining_frames[env_id] = self._sample_duration(mode)
            if mode == self.MODE_TO_INDEX["freeze_frame"]:
                self.freeze_buffer[env_id] = data[row]
            elif mode == self.MODE_TO_INDEX["distant_dust"]:
                continue
            else:
                self._write_failure_mask(env_id, mode, data.shape)

    def _sample_duration(self, mode: int) -> torch.Tensor:
        if mode == self.MODE_TO_INDEX["consecutive_dropout"]:
            return _sample_int_range(self.cfg.consecutive_dropout_duration_range, self.device)
        if mode == self.MODE_TO_INDEX["freeze_frame"]:
            return _sample_int_range(self.cfg.freeze_duration_range, self.device)
        if mode == self.MODE_TO_INDEX["distant_dust"]:
            return _sample_int_range(self.cfg.distant_dust_duration_range, self.device)
        return _sample_int_range(self.cfg.failure_duration_range, self.device)

    def _write_failure_mask(self, env_id: int, mode: int, data_shape: torch.Size):
        _, height, width, channels = data_shape
        mask = self.mask_buffer[env_id]
        if mode in (self.MODE_TO_INDEX["full_dropout"], self.MODE_TO_INDEX["consecutive_dropout"]):
            mask[:] = True # 本文根据采样到的失效类型生成二值
            return

        if mode == self.MODE_TO_INDEX["large_hole"]:
            hole_h = max(1, round(height * _sample_ratio(self.cfg.large_hole_height_range))) # round是四舍五入
            hole_w = max(1, round(width * _sample_ratio(self.cfg.large_hole_width_range)))
            top = random.randint(0, max(height - hole_h, 0))
            left = random.randint(0, max(width - hole_w, 0))
            mask[top : top + hole_h, left : left + hole_w, :] = True
            return

        if mode == self.MODE_TO_INDEX["lower_occlusion"]:
            occ_h = max(1, round(height * _sample_ratio(self.cfg.lower_occlusion_height_range)))
            mask[height - occ_h :, :, :] = True
            return

        if mode == self.MODE_TO_INDEX["central_landing_occlusion"]:
            occ_h = max(1, round(height * _sample_ratio(self.cfg.central_landing_height_range)))
            occ_w = max(1, round(width * _sample_ratio(self.cfg.central_landing_width_range)))
            center_y = round(height * _sample_ratio(self.cfg.central_landing_center_y_range))
            center_x = round(width * _sample_ratio(self.cfg.central_landing_center_x_range))
            top = min(max(center_y - occ_h // 2, 0), max(height - occ_h, 0))
            left = min(max(center_x - occ_w // 2, 0), max(width - occ_w, 0))
            mask[top : top + occ_h, left : left + occ_w, :] = True
            return

        if mode == self.MODE_TO_INDEX["stripe_missing"]:
            stripe_count = random.randint(self.cfg.stripe_count_range[0], self.cfg.stripe_count_range[1]) # 采样产生的条纹的数量
            for _ in range(stripe_count):
                orientation = random.choice(self.cfg.stripe_orientations) # 采样选择条纹的方向
                if orientation == "horizontal": # 水平条纹
                    stripe_h = max(1, round(height * _sample_ratio(self.cfg.stripe_width_range)))
                    top = random.randint(0, max(height - stripe_h, 0))
                    mask[top : top + stripe_h, :, :] = True
                else: # 垂直条纹
                    stripe_w = max(1, round(width * _sample_ratio(self.cfg.stripe_width_range)))
                    left = random.randint(0, max(width - stripe_w, 0))
                    mask[:, left : left + stripe_w, :] = True
            return

        raise RuntimeError(f"Unhandled structured depth failure mode index: {mode}.")

    # 深度图中远距离区域受到灰尘、雾、反射、远距离测量不稳定等因素影响，出现随机扰动，甚至部分远处像素直接失效变成 0
    def _apply_distant_dust(self, data: torch.Tensor, cfg: StructuredDepthFailureCfg):
        far_mask = data >= cfg.distant_dust_start
        if not far_mask.any():
            return data
        noise = torch.randn_like(data) * cfg.distant_dust_noise_std
        noisy_data = data + noise * far_mask
        dropout = (torch.rand_like(data) < cfg.distant_dust_dropout_probability) & far_mask
        dust_fill = torch.full_like(data, cfg.distant_dust_fill_value)
        return torch.where(dropout, dust_fill, noisy_data)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = _as_env_ids(env_ids, self.device)
        self.active_modes[env_ids_tensor] = -1
        self.remaining_frames[env_ids_tensor] = 0
        if self.mask_buffer is not None:
            self.mask_buffer[env_ids_tensor] = 0
        if self.freeze_buffer is not None:
            self.freeze_buffer[env_ids_tensor] = 0

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
            H, # 伪影的高度不能超过图片的高度randint
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


def compute_distance_dependent_gaussian_std(
    depth: torch.Tensor,
    min_distance: float,
    max_distance: float,
    near_std: float,
    far_std: float,
    exponent: float,
) -> torch.Tensor:
    """Compute a metric-depth Gaussian standard-deviation map."""
    scalar_parameters = {
        "min_distance": min_distance,
        "max_distance": max_distance,
        "near_std": near_std,
        "far_std": far_std,
        "exponent": exponent,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}.")
    if float(max_distance) <= float(min_distance):
        raise ValueError("max_distance must be greater than min_distance.")
    if float(near_std) < 0.0:
        raise ValueError("near_std cannot be negative.")
    if float(far_std) < float(near_std):
        raise ValueError("far_std cannot be smaller than near_std.")
    if float(exponent) <= 0.0:
        raise ValueError("exponent must be positive.")

    normalized_distance = ((depth - min_distance) / (max_distance - min_distance)).clamp(0.0, 1.0)
    return near_std + (far_std - near_std) * normalized_distance.pow(exponent)


def distance_dependent_gaussian_noise(
    data: torch.Tensor,
    cfg: DistanceDependentGaussianNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
    *,
    min_distance: float,
    max_distance: float,
) -> torch.Tensor:
    """Add Gaussian noise whose standard deviation grows with clean metric depth."""
    num_env_ids = env_ids.numel() if isinstance(env_ids, torch.Tensor) else len(env_ids)
    if data.shape[0] != num_env_ids:
        raise ValueError(
            f"Data batch shape {data.shape[0]} does not match env_ids length {num_env_ids}."
        )
    sigma = compute_distance_dependent_gaussian_std(
        data,
        min_distance=min_distance,
        max_distance=max_distance,
        near_std=cfg.near_std,
        far_std=cfg.far_std,
        exponent=cfg.distance_exponent,
    )
    if float(cfg.far_std) == 0.0:
        return data.clone()
    return data + torch.randn_like(data) * sigma


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
        env_ids = _as_env_ids(env_ids, self.device)
        
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
            torch.full((len(env_ids),), self.cfg.dead_frames, dtype=torch.long, device=self.device)
            if isinstance(self.cfg.dead_frames, int)
            else self._dead_frames_options[
                torch.randint(len(self._dead_frames_options), size=(len(env_ids),), device=self.device)
            ]
        )
        self._remain_dead_frames[env_ids] = torch.where(
            dead_this_time_mask, dead_frames, self._remain_dead_frames[env_ids] - 1
        )
        self._remain_dead_frames[env_ids].clamp_(min=0)

        # refresh the data buffer if it is not dead.
        data_to_refresh_mask = self._remain_dead_frames[env_ids] <= 0  # (len(env_ids),)
        refresh_env_ids = env_ids[data_to_refresh_mask]
        if refresh_env_ids.numel() > 0:
            self._data_buffer[refresh_env_ids] = data[data_to_refresh_mask]
        return self._data_buffer[env_ids]

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = _as_env_ids(env_ids, self.device)
        self._remain_dead_frames[env_ids] = 0
        if self._data_buffer is not None:
            self._data_buffer[env_ids] = 0
