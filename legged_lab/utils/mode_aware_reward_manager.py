"""Reward manager with per-environment source-level mode routing."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import RewardManager


class ModeAwareRewardManager(RewardManager):
    """Apply an environment mask before reward and statistic accumulation."""

    def compute(
        self,
        dt: float,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise TypeError(f"dt must be a real scalar, got {type(dt).__name__}.")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}.")

        mask_weight = None
        if active_mask is not None:
            if not isinstance(active_mask, torch.Tensor):
                raise TypeError("active_mask must be a torch.Tensor or None.")
            if active_mask.dtype != torch.bool:
                raise TypeError(
                    f"active_mask must have dtype bool, got {active_mask.dtype}."
                )
            expected_shape = (self.num_envs,)
            if active_mask.shape != expected_shape:
                raise ValueError(
                    f"active_mask must have shape {expected_shape}, "
                    f"got {tuple(active_mask.shape)}."
                )
            if active_mask.device != self._reward_buf.device:
                raise ValueError(
                    "active_mask must be on the reward manager device "
                    f"{self._reward_buf.device}, got {active_mask.device}."
                )
            mask_weight = active_mask.to(dtype=self._reward_buf.dtype)

        self._reward_buf.zero_()
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs)
        ):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            weighted_value = (
                term_cfg.func(self._env, **term_cfg.params)
                * term_cfg.weight
                * dt
            )
            # Source-level routing: inactive rows are removed before any
            # reward buffer, episodic statistic, or per-term step write.
            if mask_weight is not None:
                weighted_value = weighted_value * mask_weight

            self._reward_buf += weighted_value
            self._episode_sums[name] += weighted_value
            self._step_reward[:, term_idx] = weighted_value / dt

        return self._reward_buf
