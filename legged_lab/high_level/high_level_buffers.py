"""高层选择器使用的逐环境历史缓存，不依赖仿真 SDK。"""

from __future__ import annotations

import torch


class SelectorHistoryBuffer:
    def __init__(self, max_len: int, batch_size: int, feature_dim: int, device="cpu"):
        for name, value in (("max_len", max_len), ("batch_size", batch_size), ("feature_dim", feature_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_len = max_len
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.device = torch.empty(0, device=device).device
        # 第二维按旧到新排列；未取得首帧的环境可以直接读取全零缓存。
        self._buffer = torch.zeros(batch_size, max_len, feature_dim, device=self.device)
        self._num_pushes = torch.zeros(batch_size, device=self.device, dtype=torch.long)

    @property
    def max_length(self) -> int:
        return self.max_len

    @property
    def current_length(self) -> torch.Tensor:
        return self._num_pushes.clamp_max(self.max_len)

    @property
    def num_pushes(self) -> torch.Tensor:
        return self._num_pushes

    @property
    def buffer(self) -> torch.Tensor:
        """返回缓存视图，形状为 (环境数, 历史帧数, 单帧维度)。"""
        return self._buffer

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.batch_size, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Environment IDs must be a one-dimensional integer sequence")
        ids = ids.long()
        if ids.unique().numel() != ids.numel() or ((ids < 0) | (ids >= self.batch_size)).any():
            raise ValueError("Environment IDs must be unique and in range")
        return ids

    @torch.no_grad()
    def append(self, data: torch.Tensor, env_ids=None):
        ids = self._ids(env_ids)
        if not isinstance(data, torch.Tensor) or data.shape != (len(ids), self.feature_dim):
            raise ValueError(f"History data must have shape {(len(ids), self.feature_dim)}")
        if data.device != self.device:
            raise ValueError("History data must already be on the buffer device")
        if not torch.is_floating_point(data):
            raise ValueError("History data must be floating-point tensors")
        if not len(ids):
            return
        first = self._num_pushes[ids] == 0
        rows = self._buffer[ids]
        rows[:, :-1] = rows[:, 1:].clone()
        rows[:, -1] = data
        # 首帧复制到所有槽；不会把 episode 重置前的动作用于动作变化奖励。
        rows[first] = data[first, None, :]
        self._buffer[ids] = rows
        self._num_pushes[ids] += 1

    @torch.no_grad()
    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self._buffer[ids] = 0
        self._num_pushes[ids] = 0


class SelectorActionBuffer:
    def __init__(self, history_length: int, batch_size: int, action_dim: int = 23, device="cpu"):
        if type(history_length) is not int or history_length < 0:
            raise ValueError("history_length must be a nonnegative integer")
        self._history_length = history_length
        # history_length 表示过去的步数；另加一槽保存当前动作，单位为底层控制步。
        self._circular_buffer = SelectorHistoryBuffer(history_length + 1, batch_size, action_dim, device)
        self._time_lags = torch.zeros(batch_size, device=self.device, dtype=torch.long)

    @property
    def batch_size(self) -> int:
        return self._circular_buffer.batch_size

    @property
    def device(self) -> torch.device:
        return self._circular_buffer.device

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        return 0

    @property
    def max_time_lag(self) -> int:
        return 0

    @property
    def time_lags(self) -> torch.Tensor:
        return self._time_lags.clone()

    @torch.no_grad()
    def compute(self, actions: torch.Tensor, env_ids=None) -> torch.Tensor:
        ids = self._circular_buffer._ids(env_ids)
        self._circular_buffer.append(actions, ids)
        # 高层训练关闭动作延迟；返回刚写入的实际执行动作。
        return self._circular_buffer.buffer[ids, -1].clone()

    def set_time_lag(self, time_lag, batch_ids=None):
        ids = self._circular_buffer._ids(batch_ids)
        lag = torch.as_tensor(time_lag, device=self.device)
        if lag.dtype not in (torch.int32, torch.int64) or lag.ndim > 1:
            raise ValueError("Action time lag must be an integer or a one-dimensional integer tensor")
        if lag.ndim == 1 and lag.shape != (len(ids),):
            raise ValueError("Action time lag requires one value per selected environment")
        if (lag != 0).any():
            raise ValueError("Selector action delay is disabled; time lag must be zero")

    @torch.no_grad()
    def reset(self, env_ids=None):
        self._circular_buffer.reset(env_ids)


__all__ = ["SelectorHistoryBuffer", "SelectorActionBuffer"]
