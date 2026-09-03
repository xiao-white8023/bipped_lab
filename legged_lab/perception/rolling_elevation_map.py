"""GPU rolling elevation maps built from time-stamped point-cloud packets.

The map intentionally does not classify point sources.  Terrain returns and
robot self-returns are rasterized identically so that the simulation matches a
deployment pipeline that consumes an unfiltered LiDAR cloud.

功能是把激光雷达三维点云，转换成以机器人躯干为中心、对齐机身偏航角的**二维栅格高程图**（每个格子存该位置的地面最大高度）。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class RollingElevationMap:
    """Fuse a fixed number of body-yaw-aligned point-cloud packets.

    Each packet is first rasterized in the torso-yaw frame in which it was
    captured.  The packet maps are kept in a ring buffer.  Fusion reprojects
    every live packet into the newest scan frame and retains the highest world
    height in each cell.  The timestamp of the winning height is retained for
    the confidence channel.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        history_frames: int = 8,
        history_seconds: float = 0.8,
        resolution: float = 0.1,
        internal_size: tuple[float, float] = (3.2, 3.2),
        output_size: tuple[float, float] = (1.6, 1.0),
        invalid_height: float = -1.0,
        height_clip: tuple[float, float] = (-1.0, 1.0),
        reprojection_chunk_size: int = 512,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.history_frames = int(history_frames) # 8
        self.history_seconds = float(history_seconds) # 0.8
        self.resolution = float(resolution) # 0.1
        self.internal_size = tuple(float(value) for value in internal_size)
        self.output_size = tuple(float(value) for value in output_size)
        self.invalid_height = float(invalid_height)
        self.height_clip = tuple(float(value) for value in height_clip)
        self.reprojection_chunk_size = int(reprojection_chunk_size)
        self._validate_configuration()

        self.internal_cols = self._cell_count(self.internal_size[0]) # 3.2
        self.internal_rows = self._cell_count(self.internal_size[1]) # 3.2
        self.output_cols = self._cell_count(self.output_size[0]) # 1.6
        self.output_rows = self._cell_count(self.output_size[1]) # 1.0

        '''
        这两行代码是**滚动高程图（Rolling Elevation Map）模块的坐标基准初始化**，核心作用是**预计算两套栅格地图对应的真实物理坐标网格**，为后续点云栅格化、多帧点云融合重投影、最终观测输出提供坐标换算基准。
        '''
        self._internal_x, self._internal_y = self._make_metric_grid(
            self.internal_size, self.internal_rows, self.internal_cols
        )
        self._output_x, self._output_y = self._make_metric_grid(
            self.output_size, self.output_rows, self.output_cols
        )
        # 每个环境，存 8 张历史高程栅格，每张栅格是 33×33 的方格
        history_shape = (
            self.num_envs,
            self.history_frames,
            self.internal_rows,
            self.internal_cols,
        )
        # 对应历史帧、对应格子的**世界坐标系绝对高度（z 值，单位米）** 保存每帧激光扫描栅格化后的高度结果，后续多帧融合时从这里读取历史高度数据。
        self.history_height_w = torch.zeros(history_shape, device=self.device)
        # 标记每个格子有没有被有效激光点命中。`True` 表示有有效高度数据，`False` 表示该格子没扫描到点（超出雷达范围、被遮挡等）。 融合时只计算标记为有效的格子，避免无效值干扰结果。初始全为 `False`，随着扫描逐步标记有效。
        self.history_valid = torch.zeros(history_shape, dtype=torch.bool, device=self.device)
        # 每帧历史扫描发生时，机器人躯干在世界坐标系下的 xy 平面位置。 多帧重投影的关键参数 —— 要把历史帧的栅格对齐到当前机器人坐标系，必须先知道这帧扫描时机器人在世界的哪个位置。
        self.history_torso_xy_w = torch.zeros(
            self.num_envs, self.history_frames, 2, device=self.device
        )
        # 每帧历史扫描发生时，机器人躯干在世界坐标系下的水平朝向（偏航角 yaw） 和躯干位置配套，重投影时用这个角度做坐标旋转，把历史帧的栅格转到当前机器人的朝向下
        self.history_torso_yaw_w = torch.zeros(
            self.num_envs, self.history_frames, device=self.device
        )
        # 每帧历史扫描的时间戳（单位：秒）。初始化为负无穷 `-inf`，表示该帧还没有数据、天然 “过期”。
        self.history_timestamp = torch.full(
            (self.num_envs, self.history_frames),
            -torch.inf,
            device=self.device,
        )
        # 下一帧新数据要写入历史缓存的第几帧
        self.write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 每个环境已经缓存的有效历史帧数量。
        self.frame_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 上一次处理过的激光雷达扫描序号。激光雷达每次扫描都会生成一个递增的 `scan_count` 计数器，这里存上一次处理的序号，下一次更新时对比，如果序号没变就说明没有新扫描数据，不需要重新处理。
        self.last_sensor_scan_count = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        fused_shape = (self.num_envs, self.internal_rows, self.internal_cols)
        self.fused_height_w = torch.zeros(fused_shape, device=self.device)
        self.fused_valid = torch.zeros(fused_shape, dtype=torch.bool, device=self.device)
        self.fused_timestamp = torch.full(fused_shape, -torch.inf, device=self.device)
        self.fused_torso_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.fused_torso_yaw_w = torch.zeros(self.num_envs, device=self.device)

    def _validate_configuration(self) -> None:
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        if self.history_frames <= 0:
            raise ValueError(f"history_frames must be positive, got {self.history_frames}.")
        if self.history_seconds <= 0.0:
            raise ValueError(f"history_seconds must be positive, got {self.history_seconds}.")
        if self.resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {self.resolution}.")
        if self.reprojection_chunk_size <= 0:
            raise ValueError(
                "reprojection_chunk_size must be positive, "
                f"got {self.reprojection_chunk_size}."
            )
        for name, size in (("internal_size", self.internal_size), ("output_size", self.output_size)):
            if len(size) != 2 or min(size) <= 0.0:
                raise ValueError(f"{name} must contain two positive lengths, got {size}.")
            for extent in size:
                cells = extent / self.resolution
                if abs(cells - round(cells)) > 1.0e-6:
                    raise ValueError(
                        f"{name}={size} must be divisible by resolution={self.resolution}."
                    )
        if self.output_size[0] > self.internal_size[0] or self.output_size[1] > self.internal_size[1]:
            raise ValueError(
                f"output_size={self.output_size} must fit inside internal_size={self.internal_size}."
            )
        if self.height_clip[0] >= self.height_clip[1]:
            raise ValueError(f"height_clip must be increasing, got {self.height_clip}.")

    def _cell_count(self, extent: float) -> int:
        return int(round(extent / self.resolution)) + 1
    '''
    它的输入是「物理尺寸」和「栅格行列数」，输出是两个二维张量，分别对应栅格中每个格子中心点的 **x 物理坐标** 和 **y 物理坐标**（机器人局部坐标系，原点在躯干正下方地面，x 向前、y 向左，与机身偏航对齐）。
    '''
    def _make_metric_grid(
        self,
        size: tuple[float, float],
        rows: int,
        cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(-size[0] / 2.0, size[0] / 2.0, cols, device=self.device)
        y = torch.linspace(-size[1] / 2.0, size[1] / 2.0, rows, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return grid_x, grid_y

    @staticmethod
    def yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Extract Z yaw from WXYZ quaternions."""

        w, x, y, z = quat_wxyz.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear map history for selected environments."""

        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return

        self.history_height_w[ids] = 0.0
        self.history_valid[ids] = False
        self.history_torso_xy_w[ids] = 0.0
        self.history_torso_yaw_w[ids] = 0.0
        self.history_timestamp[ids] = -torch.inf
        self.write_index[ids] = 0
        self.frame_count[ids] = 0
        self.last_sensor_scan_count[ids] = -1
        self.fused_height_w[ids] = 0.0
        self.fused_valid[ids] = False
        self.fused_timestamp[ids] = -torch.inf
        self.fused_torso_xy_w[ids] = 0.0
        self.fused_torso_yaw_w[ids] = 0.0

    def update_from_point_cloud(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        torso_pos_w: torch.Tensor,
        torso_yaw_w: torch.Tensor,
        sensor_scan_count: torch.Tensor,
        timestamp: torch.Tensor,
        *,
        min_distance: float,
        max_distance: float,
    ) -> torch.Tensor:
        """Rasterize and fuse environments whose sensor scan count advanced.

        Returns:
            Environment indices for which a new packet was ingested.
        """

        if ray_hits_w.ndim != 3 or ray_hits_w.shape[0] != self.num_envs or ray_hits_w.shape[-1] != 3:
            raise ValueError(
                "ray_hits_w must have shape (num_envs, num_rays, 3), "
                f"got {tuple(ray_hits_w.shape)}."
            )
        if sensor_pos_w.shape != (self.num_envs, 3):
            raise ValueError(f"sensor_pos_w must have shape ({self.num_envs}, 3).")
        if torso_pos_w.shape != (self.num_envs, 3):
            raise ValueError(f"torso_pos_w must have shape ({self.num_envs}, 3).")

        scan_count = sensor_scan_count.to(device=self.device, dtype=torch.long).reshape(self.num_envs)
        timestamp = timestamp.to(device=self.device, dtype=torch.float32).reshape(self.num_envs)
        torso_yaw_w = torso_yaw_w.to(device=self.device, dtype=torch.float32).reshape(self.num_envs)
        new_env_ids = torch.nonzero(scan_count != self.last_sensor_scan_count, as_tuple=False).flatten()
        if new_env_ids.numel() == 0:
            return new_env_ids

        packet_height, packet_valid = self._rasterize_points(
            ray_hits_w[new_env_ids],
            sensor_pos_w[new_env_ids],
            torso_pos_w[new_env_ids],
            torso_yaw_w[new_env_ids],
            min_distance=float(min_distance),
            max_distance=float(max_distance),
        )
        slots = self.write_index[new_env_ids]
        self.history_height_w[new_env_ids, slots] = packet_height
        self.history_valid[new_env_ids, slots] = packet_valid
        self.history_torso_xy_w[new_env_ids, slots] = torso_pos_w[new_env_ids, :2]
        self.history_torso_yaw_w[new_env_ids, slots] = torso_yaw_w[new_env_ids]
        self.history_timestamp[new_env_ids, slots] = timestamp[new_env_ids]

        self.write_index[new_env_ids] = (slots + 1) % self.history_frames
        self.frame_count[new_env_ids] = torch.clamp(
            self.frame_count[new_env_ids] + 1, max=self.history_frames
        )
        self.last_sensor_scan_count[new_env_ids] = scan_count[new_env_ids]
        self._rebuild_fused_map(
            new_env_ids,
            torso_pos_w[new_env_ids, :2],
            torso_yaw_w[new_env_ids],
            timestamp[new_env_ids],
        )
        return new_env_ids

    def _rasterize_points(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        torso_pos_w: torch.Tensor,
        torso_yaw_w: torch.Tensor,
        *,
        min_distance: float,
        max_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_rays, _ = ray_hits_w.shape
        hits = ray_hits_w.to(self.device)
        sensor_pos = sensor_pos_w.to(self.device)
        torso_pos = torso_pos_w.to(self.device)
        torso_yaw = torso_yaw_w.to(self.device)

        finite = torch.isfinite(hits).all(dim=-1)
        # The custom grouped ray-caster initializes misses to
        # (max_distance, max_distance, max_distance), unlike IsaacLab's inf.
        miss_sentinel = torch.isclose(
            hits,
            torch.as_tensor(max_distance, dtype=hits.dtype, device=self.device),
            rtol=0.0,
            atol=1.0e-6,
        ).all(dim=-1)
        ray_delta = hits - sensor_pos[:, None, :]
        distance_sq = ray_delta.square().sum(dim=-1)
        valid = finite & ~miss_sentinel
        valid &= distance_sq >= min_distance * min_distance
        valid &= distance_sq <= max_distance * max_distance

        point_delta = hits[..., :2] - torso_pos[:, None, :2]
        cos_yaw = torch.cos(torso_yaw)[:, None]
        sin_yaw = torch.sin(torso_yaw)[:, None]
        local_x = cos_yaw * point_delta[..., 0] + sin_yaw * point_delta[..., 1]
        local_y = -sin_yaw * point_delta[..., 0] + cos_yaw * point_delta[..., 1]

        x_half = self.internal_size[0] / 2.0
        y_half = self.internal_size[1] / 2.0
        cols = torch.round((local_x + x_half) / self.resolution).long()
        rows = torch.round((local_y + y_half) / self.resolution).long()
        valid &= (cols >= 0) & (cols < self.internal_cols)
        valid &= (rows >= 0) & (rows < self.internal_rows)

        cell_count = self.internal_rows * self.internal_cols
        flat_height = torch.full(
            (batch_size * cell_count,), -torch.inf, dtype=hits.dtype, device=self.device
        )
        if torch.any(valid):
            env_offsets = torch.arange(batch_size, device=self.device)[:, None] * cell_count
            flat_indices = env_offsets + rows.clamp(0, self.internal_rows - 1) * self.internal_cols
            flat_indices += cols.clamp(0, self.internal_cols - 1)
            flat_height.scatter_reduce_(
                0,
                flat_indices[valid],
                hits[..., 2][valid],
                reduce="amax",
                include_self=True,
            )

        packet_height = flat_height.view(batch_size, self.internal_rows, self.internal_cols)
        packet_valid = torch.isfinite(packet_height)
        packet_height = torch.where(packet_valid, packet_height, torch.zeros_like(packet_height))
        return packet_height, packet_valid

    def _rebuild_fused_map(
        self,
        env_ids: torch.Tensor,
        target_xy_w: torch.Tensor,
        target_yaw_w: torch.Tensor,
        current_timestamp: torch.Tensor,
    ) -> None:
        for start in range(0, env_ids.numel(), self.reprojection_chunk_size):
            stop = min(start + self.reprojection_chunk_size, env_ids.numel())
            ids = env_ids[start:stop]
            target_xy = target_xy_w[start:stop]
            target_yaw = target_yaw_w[start:stop]
            timestamp = current_timestamp[start:stop]

            sampling_grid = self._make_sampling_grid(
                target_xy[:, None, :],
                target_yaw[:, None],
                self.history_torso_xy_w[ids],
                self.history_torso_yaw_w[ids],
                self._internal_x,
                self._internal_y,
            )
            batch_size = ids.numel()
            source_height = self.history_height_w[ids].reshape(
                batch_size * self.history_frames, 1, self.internal_rows, self.internal_cols
            )
            source_valid = self.history_valid[ids].to(torch.float32).reshape(
                batch_size * self.history_frames, 1, self.internal_rows, self.internal_cols
            )
            flat_grid = sampling_grid.reshape(
                batch_size * self.history_frames,
                self.internal_rows,
                self.internal_cols,
                2,
            )
            sampled_height = F.grid_sample(
                source_height,
                flat_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(batch_size, self.history_frames, self.internal_rows, self.internal_cols)
            sampled_valid = F.grid_sample(
                source_valid,
                flat_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(batch_size, self.history_frames, self.internal_rows, self.internal_cols) > 0.5

            frame_age = timestamp[:, None] - self.history_timestamp[ids]
            frame_alive = (frame_age >= 0.0) & (frame_age < self.history_seconds)
            sampled_valid &= frame_alive[:, :, None, None]
            masked_height = torch.where(
                sampled_valid, sampled_height, torch.full_like(sampled_height, -torch.inf)
            )
            fused_height = masked_height.amax(dim=1)
            fused_valid = torch.isfinite(fused_height)

            winner = sampled_valid & (sampled_height == fused_height[:, None])
            candidate_time = torch.where(
                winner,
                self.history_timestamp[ids, :, None, None],
                torch.full_like(sampled_height, -torch.inf),
            )
            fused_time = candidate_time.amax(dim=1)

            self.fused_height_w[ids] = torch.where(
                fused_valid, fused_height, torch.zeros_like(fused_height)
            )
            self.fused_valid[ids] = fused_valid
            self.fused_timestamp[ids] = fused_time
            self.fused_torso_xy_w[ids] = target_xy
            self.fused_torso_yaw_w[ids] = target_yaw

    def _make_sampling_grid(
        self,
        target_xy_w: torch.Tensor,
        target_yaw_w: torch.Tensor,
        source_xy_w: torch.Tensor,
        source_yaw_w: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ) -> torch.Tensor:
        """Map target-frame cell centers into normalized source-grid coordinates."""

        cos_target = torch.cos(target_yaw_w)[..., None, None]
        sin_target = torch.sin(target_yaw_w)[..., None, None]
        x_world = (
            cos_target * target_x
            - sin_target * target_y
            + target_xy_w[..., 0, None, None]
        )
        y_world = (
            sin_target * target_x
            + cos_target * target_y
            + target_xy_w[..., 1, None, None]
        )

        delta_x = x_world - source_xy_w[..., 0, None, None]
        delta_y = y_world - source_xy_w[..., 1, None, None]
        cos_source = torch.cos(source_yaw_w)[..., None, None]
        sin_source = torch.sin(source_yaw_w)[..., None, None]
        source_x = cos_source * delta_x + sin_source * delta_y
        source_y = -sin_source * delta_x + cos_source * delta_y

        normalized_x = source_x / (self.internal_size[0] / 2.0)
        normalized_y = source_y / (self.internal_size[1] / 2.0)
        return torch.stack((normalized_x, normalized_y), dim=-1)

    def get_observation(
        self,
        torso_pos_w: torch.Tensor,
        torso_yaw_w: torch.Tensor,
        timestamp: torch.Tensor,
        *,
        height_offset: float,
        height_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current body-yaw-aligned height and confidence maps."""

        torso_pos = torso_pos_w.to(self.device)
        torso_yaw = torso_yaw_w.to(self.device).reshape(self.num_envs)
        timestamp = timestamp.to(self.device).reshape(self.num_envs)
        sampling_grid = self._make_sampling_grid(
            torso_pos[:, None, :2],
            torso_yaw[:, None],
            self.fused_torso_xy_w[:, None, :],
            self.fused_torso_yaw_w[:, None],
            self._output_x,
            self._output_y,
        ).squeeze(1)

        sampled_height = F.grid_sample(
            self.fused_height_w[:, None],
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        sampled_valid = F.grid_sample(
            self.fused_valid[:, None].to(torch.float32),
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1) > 0.5
        sampled_time = F.grid_sample(
            torch.where(
                self.fused_valid,
                self.fused_timestamp,
                torch.zeros_like(self.fused_timestamp),
            )[:, None],
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

        age = timestamp[:, None, None] - sampled_time
        sampled_valid &= (age >= 0.0) & (age < self.history_seconds)
        relative_height = (torso_pos[:, 2, None, None] - sampled_height - height_offset) * height_scale
        relative_height = relative_height.clamp(self.height_clip[0], self.height_clip[1])
        relative_height = torch.where(
            sampled_valid,
            relative_height,
            torch.full_like(relative_height, self.invalid_height),
        )
        confidence = torch.clamp(1.0 - age / self.history_seconds, min=0.0, max=1.0)
        confidence = torch.where(sampled_valid, confidence, torch.zeros_like(confidence))
        return relative_height, confidence
