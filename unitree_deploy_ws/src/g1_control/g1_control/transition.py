"""Manual VP/LP selection and output-space transition logic."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .contracts import JOINT_COUNT


class EstimatorMode(IntEnum):
    VP = 0
    LP = 1


class ActionTransition:
    """Blend final actions after a mode change, never inputs or latent vectors."""

    def __init__(self, blend_steps: int = 5) -> None:
        if blend_steps < 1:
            raise ValueError("blend_steps must be at least one")
        self.blend_steps = int(blend_steps)
        self._last = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._start = self._last.copy()
        self._remaining = 0
        self._initialized = False

    def reset(self) -> None:
        self._last.fill(0.0)
        self._start.fill(0.0)
        self._remaining = 0
        self._initialized = False

    def set_executed(self, action: np.ndarray) -> None:
        """Keep later mode blends anchored to the action actually published."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (JOINT_COUNT,):
            raise ValueError(
                f"action must have shape {(JOINT_COUNT,)}, got {action.shape}"
            )
        self._last = action.copy()
        self._initialized = True

    def step(self, target: np.ndarray, switched: bool) -> tuple[np.ndarray, float]:
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (JOINT_COUNT,):
            raise ValueError(f"target must have shape {(JOINT_COUNT,)}, got {target.shape}")
        if not self._initialized:
            self._last = target.copy()
            self._initialized = True
            return self._last.copy(), 1.0
        if switched:
            self._start = self._last.copy()
            self._remaining = self.blend_steps
        if self._remaining > 0:
            completed = self.blend_steps - self._remaining + 1
            alpha = float(completed) / float(self.blend_steps)
            output = (1.0 - alpha) * self._start + alpha * target
            self._remaining -= 1
        else:
            alpha = 1.0
            output = target
        self._last = output.astype(np.float32, copy=True)
        return self._last.copy(), alpha
