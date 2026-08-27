"""Expert-data adapter for the standalone whole-body G1 Recovery AMP task."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .recovery_amp_features import RecoveryAMPFeatureBuilder
from .recovery_npz_motion_loader import RecoveryNPZMotionLoader


class RecoveryAMPData:
    """Adapt Recovery NPZ motions to the expert-transition API consumed by AMPPPO."""

    def __init__(
        self,
        motion_dir=None,
        motion_files=None,
        device="cpu",
        transition_dt=0.01,
        expected_fps=50.0,
        sampling="uniform_frame",
        preload_transitions=False,
        num_preload_transitions=0,
    ):
        self.transition_dt = self._validate_transition_dt(transition_dt)
        if not isinstance(preload_transitions, bool):
            raise ValueError(
                f"preload_transitions must be a bool, got {preload_transitions!r}."
            )
        self.preload_transitions = preload_transitions
        self.num_preload_transitions = self._validate_preload_count(
            num_preload_transitions, preload_transitions
        )

        self.motion_loader = RecoveryNPZMotionLoader(
            motion_dir=motion_dir,
            motion_files=motion_files,
            device=device,
            expected_fps=expected_fps,
            sampling=sampling,
        )
        self.feature_builder = RecoveryAMPFeatureBuilder(self.motion_loader.body_names)
        if self.feature_builder.feature_dim != 195:
            raise RuntimeError(
                "RecoveryAMPFeatureBuilder must produce exactly 195 features, got "
                f"{self.feature_builder.feature_dim}."
            )
        self.device = self.motion_loader.device

        if self.preload_transitions:
            self.preloaded_s, self.preloaded_s_next = self.sample_transition_batch(
                self.num_preload_transitions
            )

    @staticmethod
    def _validate_transition_dt(transition_dt) -> float:
        if isinstance(transition_dt, bool):
            raise ValueError(
                f"transition_dt must be positive and finite, got {transition_dt!r}."
            )
        try:
            value = float(transition_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"transition_dt must be positive and finite, got {transition_dt!r}."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"transition_dt must be positive and finite, got {transition_dt!r}."
            )
        return value

    @staticmethod
    def _validate_preload_count(num_preload_transitions, preload_transitions: bool) -> int:
        if isinstance(num_preload_transitions, bool) or not isinstance(
            num_preload_transitions, Integral
        ):
            raise ValueError(
                "num_preload_transitions must be a non-negative integer, got "
                f"{num_preload_transitions!r}."
            )
        value = int(num_preload_transitions)
        if value < 0 or (preload_transitions and value == 0):
            requirement = "positive" if preload_transitions else "non-negative"
            raise ValueError(
                f"num_preload_transitions must be {requirement}, got {value}."
            )
        return value

    @staticmethod
    def _positive_int(value, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{field} must be a positive integer, got {value!r}.")
        return int(value)

    @property
    def observation_dim(self) -> int:
        return self.feature_builder.feature_dim

    @property
    def num_motions(self) -> int:
        return self.motion_loader.num_motions

    @property
    def body_names(self):
        return self.motion_loader.body_names

    @property
    def target_body_names(self):
        return self.feature_builder.target_body_names

    @property
    def anchor_body_name(self):
        return self.feature_builder.anchor_body_name

    def sample_state_batch(self, batch_size) -> torch.Tensor:
        """Sample a batch of final 195D AMP states."""

        raw_state = self.motion_loader.sample_state_batch(batch_size)
        return self.feature_builder.build_from_state(raw_state)

    def sample_transition_batch(self, batch_size):
        """Sample separate 195D state/next-state features from one motion each."""

        raw_state, raw_next_state = self.motion_loader.sample_transition_batch(
            batch_size,
            transition_dt=self.transition_dt,
        )
        return self.feature_builder.build_transition(raw_state, raw_next_state)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Yield expert ``(state, next_state)`` pairs for the existing AMPPPO."""

        num_mini_batch = self._positive_int(num_mini_batch, "num_mini_batch")
        mini_batch_size = self._positive_int(mini_batch_size, "mini_batch_size")
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                indices = torch.randint(
                    self.num_preload_transitions,
                    (mini_batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
                amp_state = self.preloaded_s[indices]
                amp_next_state = self.preloaded_s_next[indices]
            else:
                amp_state, amp_next_state = self.sample_transition_batch(mini_batch_size)
            yield amp_state, amp_next_state


__all__ = ["RecoveryAMPData"]
