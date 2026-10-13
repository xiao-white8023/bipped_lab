"""Typed tensor contracts shared by collection, training, and deployment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SensorHealth:
    """Eight observable health values for one sensor branch."""

    frame_age: torch.Tensor
    update_interval: torch.Tensor
    valid_ratio: torch.Tensor
    abnormal_ratio: torch.Tensor
    temporal_l1: torch.Tensor
    frozen_ratio: torch.Tensor
    spatial_mean: torch.Tensor
    spatial_std: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        values = (
            self.frame_age,
            self.update_interval,
            self.valid_ratio,
            self.abnormal_ratio,
            self.temporal_l1,
            self.frozen_ratio,
            self.spatial_mean,
            self.spatial_std,
        )
        normalized = [value.reshape(value.shape[0], -1)[:, :1] for value in values]
        return torch.cat(normalized, dim=-1)


@dataclass
class DualBranchOutput:
    """Frozen RENet outputs evaluated for both binary masks."""

    lp_latent: torch.Tensor
    vp_latent: torch.Tensor
    lp_velocity: torch.Tensor
    vp_velocity: torch.Tensor
    lp_action: torch.Tensor
    vp_action: torch.Tensor


@dataclass
class SelectorInput:
    timestamp: torch.Tensor
    proprio: torch.Tensor
    lp_height_map: torch.Tensor
    lp_timestamp: torch.Tensor
    vp_depth_history: torch.Tensor
    vp_timestamp: torch.Tensor
    current_mask: torch.Tensor
    time_since_switch: torch.Tensor
    lp_valid_mask: torch.Tensor | None = None
    vp_valid_support: torch.Tensor | None = None
    lp_packet_timestamp: torch.Tensor | None = None
    vp_packet_timestamp: torch.Tensor | None = None


@dataclass
class SelectorDecision:
    selected_mask: torch.Tensor
    risks: torch.Tensor
    lp_available: torch.Tensor
    vp_available: torch.Tensor
    both_unreliable: torch.Tensor
    switched: torch.Tensor
    reason_code: torch.Tensor

    @property
    def lp_risk_1s(self) -> torch.Tensor:
        return self.risks[..., 0]

    @property
    def vp_risk_1s(self) -> torch.Tensor:
        return self.risks[..., 1]


@dataclass
class SelectorOutput:
    """Public runtime output matching the deployment contract."""

    selected_mask: torch.Tensor
    lp_risk_1s: torch.Tensor
    vp_risk_1s: torch.Tensor
    lp_available: torch.Tensor
    vp_available: torch.Tensor
    both_unreliable: torch.Tensor
    switch_reason: torch.Tensor
