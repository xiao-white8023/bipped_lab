# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from collections.abc import Sequence

import torch

from isaaclab.sensors.ray_caster import RayCaster as BaseRayCaster


class RayCaster(BaseRayCaster):
    @property
    def scan_counts(self) -> torch.Tensor:
        """Completed packet counts.  Consumers must treat this tensor as read-only."""

        return self._scan_counts

    @property
    def scan_timestamps(self) -> torch.Tensor:
        """Capture time of the most recently completed packet per environment."""

        return self._scan_timestamps

    @property
    def current_timestamps(self) -> torch.Tensor:
        """Current per-environment sensor clock."""

        return self._timestamp

    @property
    def ray_starts_w(self) -> torch.Tensor:
        """Actual world-space ray origins after pose and yaw alignment."""

        return self._ray_starts_w

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._scan_counts = torch.zeros(self._view.count, dtype=torch.long, device=self._device)
        self._scan_timestamps = torch.zeros(self._view.count, device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids, 0] = self.drift[env_ids, 0].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 1] = self.drift[env_ids, 1].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 2] = self.drift[env_ids, 2].uniform_(
            *(self.cfg.drift_range[0] * 0.1, self.cfg.drift_range[1] * 0.1)
        )
        scan_counts = getattr(self, "_scan_counts", None)
        if scan_counts is not None:
            scan_counts[env_ids] = 0
            self._scan_timestamps[env_ids] = 0

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        self._scan_counts[env_ids] += 1
        self._scan_timestamps[env_ids] = self._timestamp[env_ids]
