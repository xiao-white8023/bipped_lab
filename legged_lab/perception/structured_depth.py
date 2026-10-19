"""Pure Torch Film depth-fault core, shared by legacy Film and high-level selection."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


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
    return torch.randint(duration_range[0], duration_range[1] + 1,
                         (1,), device=device, dtype=torch.long)[0]


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
            self.mode_probabilities = torch.ones(len(self.mode_names), device=device) / len(self.mode_names)
        else:
            if len(cfg.failure_mode_probabilities) != len(self.mode_names):
                raise RuntimeError("failure_mode_probabilities must match failure_modes length.")
            probabilities = torch.tensor(cfg.failure_mode_probabilities, dtype=torch.float, device=device)
            if torch.any(probabilities < 0) or torch.sum(probabilities) <= 0:
                raise RuntimeError("failure_mode_probabilities must be non-negative and sum to a positive value.")
            self.mode_probabilities = probabilities / torch.sum(probabilities) # 这样就保证了即使所有的模式的出发概率大于1,也不会报错

        self.active_modes = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.remaining_frames = torch.zeros(num_envs, dtype=torch.long, device=device) # 记录当前失效模式还要持续多少帧
        self.mask_buffer = None # 缓存 遮挡/丢失区域的 mask
        self.freeze_buffer = None # 缓存冻结帧图像

    def __call__(self, data: torch.Tensor, cfg: StructuredDepthFailureCfg, env_ids: torch.Tensor | Sequence[int]):
        env_ids = _as_env_ids(env_ids, self.device)
        # depth的shape是[N,H,W,C]
        if data.shape[0] != env_ids.numel(): # .numel()返回张量中所有元素的总个数
            raise RuntimeError(f"Data batch shape {data.shape[0]} does not match env_ids length {env_ids.numel()}.")
        self._ensure_buffers(data)
        self._maybe_start_failures(data, env_ids)

        return self._apply_frame(data, cfg, env_ids)

    def start_external(self, data, valid_support, env_ids, mode_ids, duration_frames, *, rng=None):
        """显式启动故障；允许一帧，不执行 Film 的内部概率或时长抽样。"""
        env_ids = _as_env_ids(env_ids, self.device)
        modes = torch.as_tensor(mode_ids, device=self.device, dtype=torch.long).expand(len(env_ids))
        durations = torch.as_tensor(duration_frames, device=self.device, dtype=torch.long).expand(len(env_ids))
        if (data.ndim != 4 or data.shape[0] != len(env_ids) or valid_support.shape != data.shape
                or ((modes < 0) | (modes >= len(self.MODE_TO_INDEX))).any() or (durations < 1).any()):
            raise ValueError("External Film faults require NHWC data/support, valid modes and positive frame counts")
        self._ensure_buffers(data)
        if (self.remaining_frames[env_ids] > 0).any():
            raise ValueError("Cannot overwrite an active external depth fault")
        self.active_modes[env_ids], self.remaining_frames[env_ids] = modes, durations
        self.mask_buffer[env_ids] = False
        for row, env_id, mode in zip(range(len(env_ids)), env_ids.tolist(), modes.tolist()):
            if mode == self.MODE_TO_INDEX["freeze_frame"]:
                self.freeze_buffer[env_id] = data[row]
                self.freeze_support[env_id] = valid_support[row]
            elif mode != self.MODE_TO_INDEX["distant_dust"]:
                self._write_failure_mask(env_id, mode, data.shape, rng=rng)

    def apply_external(self, data, valid_support, env_ids, *, rng=None):
        """新帧应用一次并返回 (米制深度, 有效支持)；不会自动启动故障。"""
        env_ids = _as_env_ids(env_ids, self.device)
        if data.ndim != 4 or data.shape[0] != len(env_ids) or valid_support.shape != data.shape:
            raise ValueError("External Film input must contain matching NHWC depth and support")
        self._ensure_buffers(data)
        return self._apply_frame(data, self.cfg, env_ids, support=valid_support, rng=rng)

    def _apply_frame(self, data, cfg, env_ids, *, support=None, rng=None):
        output = data.clone()
        output_support = None if support is None else support.clone()
        active_mask = self.remaining_frames[env_ids] > 0
        if active_mask.any():
            active_env_ids = env_ids[active_mask]
            active_rows = active_mask.nonzero(as_tuple=False).flatten()
            active_modes = self.active_modes[active_env_ids]

            freeze_rows = active_rows[active_modes == self.MODE_TO_INDEX["freeze_frame"]]
            if freeze_rows.numel() > 0:
                freeze_env_ids = env_ids[freeze_rows]
                output[freeze_rows] = self.freeze_buffer[freeze_env_ids]
                if output_support is not None:
                    output_support[freeze_rows] = self.freeze_support[freeze_env_ids]

            dust_rows = active_rows[active_modes == self.MODE_TO_INDEX["distant_dust"]]
            if dust_rows.numel() > 0:
                if output_support is None:
                    output[dust_rows] = self._apply_distant_dust(output[dust_rows], cfg)
                else:
                    output[dust_rows], dropout = self._apply_distant_dust(
                        output[dust_rows], cfg, env_ids=env_ids[dust_rows], rng=rng, return_dropout=True)
                    output_support[dust_rows] = torch.where(dropout, 0.0, output_support[dust_rows])

            mask_rows = active_rows[
                (active_modes != self.MODE_TO_INDEX["freeze_frame"])
                & (active_modes != self.MODE_TO_INDEX["distant_dust"])
            ]
            if mask_rows.numel() > 0:
                mask_env_ids = env_ids[mask_rows]
                masks = self.mask_buffer[mask_env_ids].to(dtype=torch.bool)
                fill = torch.full_like(output[mask_rows], cfg.fill_value)
                output[mask_rows] = torch.where(masks, fill, output[mask_rows])
                if output_support is not None:
                    output_support[mask_rows] = torch.where(masks, 0.0, output_support[mask_rows])

            self.remaining_frames[active_env_ids] -= 1
            finished_env_ids = active_env_ids[self.remaining_frames[active_env_ids] <= 0]
            if finished_env_ids.numel() > 0:
                self.active_modes[finished_env_ids] = -1
                self.remaining_frames[finished_env_ids] = 0
                self.mask_buffer[finished_env_ids] = 0

        return output if output_support is None else (output, output_support)

    def _ensure_buffers(self, data: torch.Tensor):
        full_shape = (self.num_envs, *data.shape[1:]) # data.shape[1:]的意思是 变成[H,W,C], full_shape=[self.num_envs,H,W,C]
        if self.mask_buffer is None or tuple(self.mask_buffer.shape) != full_shape:
            self.mask_buffer = torch.zeros(full_shape, dtype=torch.bool, device=self.device) # 布尔
            self.freeze_buffer = torch.zeros(full_shape, dtype=data.dtype, device=self.device)
            self.freeze_support = torch.zeros(full_shape, dtype=data.dtype, device=self.device)

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

        sampled_positions = torch.multinomial(self.mode_probabilities, start_env_ids.numel(), replacement=True)
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

    def _write_failure_mask(self, env_id: int, mode: int, data_shape: torch.Size, *, rng=None):
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)

        def ratio(bounds):
            return (_sample_ratio(bounds) if rng is None else
                    bounds[0] + float(rng.uniform(ids)[0]) * (bounds[1] - bounds[0]))

        def integer(low, high):
            return random.randint(low, high) if rng is None else int(rng.randint(low, high + 1, ids)[0])

        def choice(values):
            return random.choice(values) if rng is None else values[integer(0, len(values) - 1)]

        _, height, width, channels = data_shape
        mask = self.mask_buffer[env_id]
        if mode in (self.MODE_TO_INDEX["full_dropout"], self.MODE_TO_INDEX["consecutive_dropout"]):
            mask[:] = True # 本文根据采样到的失效类型生成二值
            return

        if mode == self.MODE_TO_INDEX["large_hole"]:
            hole_h = max(1, round(height * ratio(self.cfg.large_hole_height_range))) # round是四舍五入
            hole_w = max(1, round(width * ratio(self.cfg.large_hole_width_range)))
            top = integer(0, max(height - hole_h, 0))
            left = integer(0, max(width - hole_w, 0))
            mask[top : top + hole_h, left : left + hole_w, :] = True
            return

        if mode == self.MODE_TO_INDEX["lower_occlusion"]:
            occ_h = max(1, round(height * ratio(self.cfg.lower_occlusion_height_range)))
            mask[height - occ_h :, :, :] = True
            return

        if mode == self.MODE_TO_INDEX["central_landing_occlusion"]:
            occ_h = max(1, round(height * ratio(self.cfg.central_landing_height_range)))
            occ_w = max(1, round(width * ratio(self.cfg.central_landing_width_range)))
            center_y = round(height * ratio(self.cfg.central_landing_center_y_range))
            center_x = round(width * ratio(self.cfg.central_landing_center_x_range))
            top = min(max(center_y - occ_h // 2, 0), max(height - occ_h, 0))
            left = min(max(center_x - occ_w // 2, 0), max(width - occ_w, 0))
            mask[top : top + occ_h, left : left + occ_w, :] = True
            return

        if mode == self.MODE_TO_INDEX["stripe_missing"]:
            stripe_count = integer(self.cfg.stripe_count_range[0], self.cfg.stripe_count_range[1]) # 采样产生的条纹的数量
            for _ in range(stripe_count):
                orientation = choice(self.cfg.stripe_orientations) # 采样选择条纹的方向
                if orientation == "horizontal": # 水平条纹
                    stripe_h = max(1, round(height * ratio(self.cfg.stripe_width_range)))
                    top = integer(0, max(height - stripe_h, 0))
                    mask[top : top + stripe_h, :, :] = True
                else: # 垂直条纹
                    stripe_w = max(1, round(width * ratio(self.cfg.stripe_width_range)))
                    left = integer(0, max(width - stripe_w, 0))
                    mask[:, left : left + stripe_w, :] = True
            return

        raise RuntimeError(f"Unhandled structured depth failure mode index: {mode}.")

    # 深度图中远距离区域受到灰尘、雾、反射、远距离测量不稳定等因素影响，出现随机扰动，甚至部分远处像素直接失效变成 0
    def _apply_distant_dust(self, data, cfg, *, env_ids=None, rng=None, return_dropout=False):
        far_mask = data >= cfg.distant_dust_start
        if not far_mask.any():
            return (data, torch.zeros_like(data, dtype=torch.bool)) if return_dropout else data
        normal = torch.randn_like(data) if rng is None else rng.normal(env_ids, tuple(data.shape[1:]))
        noise = normal * cfg.distant_dust_noise_std
        noisy_data = data + noise * far_mask
        uniform = torch.rand_like(data) if rng is None else rng.uniform(env_ids, tuple(data.shape[1:]))
        dropout = (uniform < cfg.distant_dust_dropout_probability) & far_mask
        dust_fill = torch.full_like(data, cfg.distant_dust_fill_value)
        output = torch.where(dropout, dust_fill, noisy_data)
        return (output, dropout) if return_dropout else output

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
            self.freeze_support[env_ids_tensor] = 0

