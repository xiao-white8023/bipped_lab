"""GPU-vectorized point-cloud noise for simulated LiDAR packets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .lidat_cfg import LidarNoiseCfg


def _validate_range(name: str, value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    lower, upper = float(value[0]), float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} must contain finite values, got {value}.")
    if lower > upper:
        raise ValueError(f"{name} must be increasing, got {value}.")
    return lower, upper


def validate_lidar_noise_cfg(cfg: LidarNoiseCfg) -> None:
    """Validate all LiDAR noise ranges before allocating simulator state."""

    range_std = _validate_range("range_std_range_m", cfg.range_std_range_m)
    dropout = _validate_range("point_dropout_prob_range", cfg.point_dropout_prob_range)
    _validate_range("translation_bias_range_m", cfg.translation_bias_range_m)
    _validate_range("rotation_bias_range_deg", cfg.rotation_bias_range_deg)
    if range_std[0] < 0.0:
        raise ValueError(f"range_std_range_m cannot be negative, got {cfg.range_std_range_m}.")
    if dropout[0] < 0.0 or dropout[1] > 1.0:
        raise ValueError(
            "point_dropout_prob_range must stay inside [0, 1], "
            f"got {cfg.point_dropout_prob_range}."
        )


def _quat_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a batch of WXYZ quaternions to rotation matrices."""

    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        torch.finfo(quaternion.dtype).eps
    )
    w, x, y, z = quaternion.unbind(dim=-1)
    matrix = torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return matrix.reshape(*quaternion.shape[:-1], 3, 3)


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert batched roll-pitch-yaw angles to Rz(yaw) Ry(pitch) Rx(roll)."""

    roll, pitch, yaw = rpy.unbind(dim=-1)
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    matrix = torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    )
    return matrix.reshape(*rpy.shape[:-1], 3, 3)


class LidarNoiseModel:
    """Apply per-packet noise with per-environment, per-episode parameters."""

    def __init__(
        self,
        cfg: LidarNoiseCfg,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        validate_lidar_noise_cfg(cfg)
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.range_std_m = torch.zeros(self.num_envs, device=self.device)
        self.point_dropout_prob = torch.zeros(self.num_envs, device=self.device)
        self.translation_bias_body_m = torch.zeros(self.num_envs, 3, device=self.device)
        self.rotation_bias_rpy_rad = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_scan_count = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.reset()

    def _uniform(
        self,
        value_range: Sequence[float],
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        lower, upper = float(value_range[0]), float(value_range[1])
        if lower == upper:
            return torch.full(shape, lower, device=self.device)
        return lower + (upper - lower) * torch.rand(shape, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Resample episode-constant parameters for selected environments."""

        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return

        if self.cfg.enable:
            count = int(ids.numel())
            self.range_std_m[ids] = self._uniform(self.cfg.range_std_range_m, (count,))
            self.point_dropout_prob[ids] = self._uniform(
                self.cfg.point_dropout_prob_range, (count,)
            )
            self.translation_bias_body_m[ids] = self._uniform(
                self.cfg.translation_bias_range_m, (count, 3)
            )
            rotation_deg = self._uniform(self.cfg.rotation_bias_range_deg, (count, 3))
            self.rotation_bias_rpy_rad[ids] = torch.deg2rad(rotation_deg)
        else:
            self.range_std_m[ids] = 0.0
            self.point_dropout_prob[ids] = 0.0
            self.translation_bias_body_m[ids] = 0.0
            self.rotation_bias_rpy_rad[ids] = 0.0
        self.last_scan_count[ids] = -1

    def apply_new_scans(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        torso_pos_w: torch.Tensor,
        torso_quat_w: torch.Tensor,
        sensor_scan_count: torch.Tensor,
        *,
        min_distance: float,
        max_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply noise once to each newly observed sensor packet.

        Returns the possibly copied hit tensor, the possibly biased sensor
        positions, and the environment indices whose scan counter advanced.
        Repeated calls for the same packet return the original tensors because
        the rolling map independently ignores unchanged scan counters.
        """

        self._validate_inputs(
            ray_hits_w,
            sensor_pos_w,
            torso_pos_w,
            torso_quat_w,
            sensor_scan_count,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        scan_count = sensor_scan_count.to(device=self.device, dtype=torch.long).reshape(self.num_envs)
        new_env_ids = torch.nonzero(scan_count != self.last_scan_count, as_tuple=False).flatten()
        if new_env_ids.numel() == 0:
            return ray_hits_w, sensor_pos_w, new_env_ids

        if not self.cfg.enable:
            self.last_scan_count[new_env_ids] = scan_count[new_env_ids]
            return ray_hits_w, sensor_pos_w, new_env_ids

        noisy_hits = ray_hits_w.clone()
        noisy_sensor_pos = sensor_pos_w.clone()
        selected_hits, selected_sensor_pos = self._apply(
            ray_hits_w[new_env_ids],
            sensor_pos_w[new_env_ids],
            torso_pos_w[new_env_ids],
            torso_quat_w[new_env_ids],
            new_env_ids,
            min_distance=float(min_distance),
            max_distance=float(max_distance),
        )
        noisy_hits[new_env_ids] = selected_hits
        noisy_sensor_pos[new_env_ids] = selected_sensor_pos
        self.last_scan_count[new_env_ids] = scan_count[new_env_ids]
        return noisy_hits, noisy_sensor_pos, new_env_ids

    def _apply(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        torso_pos_w: torch.Tensor,
        torso_quat_w: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        min_distance: float,
        max_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hits = ray_hits_w
        sensor_pos = sensor_pos_w
        ray_delta_w = hits - sensor_pos[:, None, :]
        ray_distance = torch.linalg.vector_norm(ray_delta_w, dim=-1)
        finite = torch.isfinite(hits).all(dim=-1) & torch.isfinite(ray_distance)
        miss_sentinel = torch.isclose(
            hits,
            torch.as_tensor(max_distance, dtype=hits.dtype, device=hits.device),
            rtol=0.0,
            atol=1.0e-6,
        ).all(dim=-1)
        valid = finite & ~miss_sentinel
        valid &= ray_distance >= min_distance
        valid &= ray_distance <= max_distance
        finite_out_of_range = finite & ~miss_sentinel & ~valid

        safe_distance = ray_distance.clamp_min(torch.finfo(hits.dtype).eps)
        ray_unit_w = ray_delta_w / safe_distance[..., None]
        range_std = self.range_std_m[env_ids].to(hits.dtype)[:, None]
        noisy_distance = (ray_distance + torch.randn_like(ray_distance) * range_std).clamp(
            min=min_distance,
            max=max_distance,
        )

        body_rotation_w = _quat_wxyz_to_matrix(torso_quat_w.to(hits.dtype))
        world_rotation_body = body_rotation_w.transpose(-1, -2)
        ray_unit_body = torch.einsum("bij,brj->bri", world_rotation_body, ray_unit_w)
        bias_rotation_body = _rpy_to_matrix(self.rotation_bias_rpy_rad[env_ids].to(hits.dtype))
        biased_ray_unit_body = torch.einsum("bij,brj->bri", bias_rotation_body, ray_unit_body)
        biased_ray_unit_w = torch.einsum("bij,brj->bri", body_rotation_w, biased_ray_unit_body)

        nominal_sensor_body = torch.einsum(
            "bij,bj->bi", world_rotation_body, sensor_pos - torso_pos_w
        )
        biased_sensor_body = nominal_sensor_body + self.translation_bias_body_m[env_ids].to(hits.dtype)
        biased_sensor_pos_w = torso_pos_w + torch.einsum(
            "bij,bj->bi", body_rotation_w, biased_sensor_body
        )
        candidate_hits = biased_sensor_pos_w[:, None, :] + biased_ray_unit_w * noisy_distance[..., None]
        noisy_hits = torch.where(valid[..., None], candidate_hits, hits)
        noisy_hits = noisy_hits.masked_fill(finite_out_of_range[..., None], torch.nan)

        dropout_prob = self.point_dropout_prob[env_ids].to(hits.dtype)[:, None]
        dropped = valid & (torch.rand_like(ray_distance) < dropout_prob)
        noisy_hits = noisy_hits.masked_fill(dropped[..., None], torch.nan)
        return noisy_hits, biased_sensor_pos_w

    def _validate_inputs(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        torso_pos_w: torch.Tensor,
        torso_quat_w: torch.Tensor,
        sensor_scan_count: torch.Tensor,
        *,
        min_distance: float,
        max_distance: float,
    ) -> None:
        if not ray_hits_w.is_floating_point():
            raise TypeError(f"ray_hits_w must be floating point, got {ray_hits_w.dtype}.")
        if ray_hits_w.ndim != 3 or ray_hits_w.shape[0] != self.num_envs or ray_hits_w.shape[-1] != 3:
            raise ValueError(
                f"ray_hits_w must have shape ({self.num_envs}, num_rays, 3), "
                f"got {tuple(ray_hits_w.shape)}."
            )
        expected_vec_shape = (self.num_envs, 3)
        if sensor_pos_w.shape != expected_vec_shape:
            raise ValueError(f"sensor_pos_w must have shape {expected_vec_shape}.")
        if torso_pos_w.shape != expected_vec_shape:
            raise ValueError(f"torso_pos_w must have shape {expected_vec_shape}.")
        if torso_quat_w.shape != (self.num_envs, 4):
            raise ValueError(f"torso_quat_w must have shape ({self.num_envs}, 4).")
        if sensor_scan_count.numel() != self.num_envs:
            raise ValueError(f"sensor_scan_count must contain {self.num_envs} values.")
        tensors = (ray_hits_w, sensor_pos_w, torso_pos_w, torso_quat_w, sensor_scan_count)
        if any(value.device != self.device for value in tensors):
            raise ValueError(f"All inputs must be on {self.device}.")
        if not math.isfinite(float(min_distance)) or not math.isfinite(float(max_distance)):
            raise ValueError("LiDAR distance limits must be finite.")
        if float(min_distance) < 0.0 or float(max_distance) <= float(min_distance):
            raise ValueError(
                f"Expected 0 <= min_distance < max_distance, got {min_distance}, {max_distance}."
            )
