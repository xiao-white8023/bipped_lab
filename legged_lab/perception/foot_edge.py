"""Simulation-independent mathematics for the G1 foot-volume edge penalty.

The regular point grid is a virtual foot volume, not a replacement for the
robot's physical collision geometry. Support rays are diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .foot_edge_geometry import EdgeGeometryCfg


@dataclass
class FootEdgeCfg:
    """Parameters shared by the reward, its diagnostics, and saved metadata."""

    enabled: bool = True
    geometry: EdgeGeometryCfg = field(default_factory=EdgeGeometryCfg)
    x_range: tuple[float, float] = (-0.06, 0.14)
    y_range: tuple[float, float] = (-0.03, 0.03)
    z_range: tuple[float, float] = (-0.03, 0.0)
    grid_shape: tuple[int, int, int] = (10, 5, 2)
    velocity_epsilon: float = 1e-3
    tolerance: float = 0.0
    debug_vis: bool = False
    support_diagnostics: bool = True
    support_distance_threshold: float = 0.04
    support_contact_threshold: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name in ("enabled", "debug_vis", "support_diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        for name in ("x_range", "y_range", "z_range"):
            bounds = getattr(self, name)
            if (len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds)
                    or not bounds[0] < bounds[1]):
                raise ValueError(f"{name} must contain two increasing finite bounds.")
        if (len(self.grid_shape) != 3
                or any(type(n) is not int or n < 2 for n in self.grid_shape)):
            raise ValueError("grid_shape must contain three integers of at least two.")
        for name in ("velocity_epsilon", "tolerance", "support_contact_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (isinstance(self.support_distance_threshold, bool)
                or not math.isfinite(self.support_distance_threshold)
                or self.support_distance_threshold <= 0):
            raise ValueError("support_distance_threshold must be finite and positive.")
        if not isinstance(self.geometry, EdgeGeometryCfg):
            raise ValueError("geometry must be an EdgeGeometryCfg.")
        if hasattr(self.geometry, "validate"):
            self.geometry.validate()


def make_volume_points(cfg: FootEdgeCfg, device: str | torch.device) -> torch.Tensor:
    """Return the fixed P-by-3 regular grid in an ankle-roll link frame."""
    cfg.validate()
    axes = [torch.linspace(*bounds, steps=n, device=device, dtype=torch.float32)
            for bounds, n in zip((cfg.x_range, cfg.y_range, cfg.z_range), cfg.grid_shape)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def point_kinematics(
    local_points: torch.Tensor,
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    com_pos: torch.Tensor,
    com_vel: torch.Tensor,
    ang_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform P points for N environments and B bodies using unit wxyz quaternions.

    Positions use the link frame; the point velocity uses the body's COM
    position and COM linear velocity together. All velocities are world-frame
    quantities. The result has shape (N, B, P, 3).
    """
    if local_points.ndim != 2 or local_points.shape[-1] != 3:
        raise ValueError("local_points must have shape (P, 3).")
    if link_pos.ndim != 3 or link_pos.shape[-1] != 3:
        raise ValueError("link_pos must have shape (N, B, 3).")
    if link_quat.shape != (*link_pos.shape[:-1], 4):
        raise ValueError("link_quat must have shape (N, B, 4).")
    if any(x.shape != link_pos.shape for x in (com_pos, com_vel, ang_vel)):
        raise ValueError("COM position, COM velocity, and angular velocity must match link_pos.")
    local = local_points[None, None].expand(*link_pos.shape[:2], -1, -1)
    q_vec = link_quat[..., 1:].unsqueeze(-2)
    q_w = link_quat[..., :1].unsqueeze(-2)
    uv = torch.linalg.cross(q_vec, local, dim=-1)
    rotated = local + 2.0 * (q_w * uv + torch.linalg.cross(q_vec, uv, dim=-1))
    points = link_pos.unsqueeze(-2) + rotated
    velocities = com_vel.unsqueeze(-2) + torch.linalg.cross(
        ang_vel.unsqueeze(-2), points - com_pos.unsqueeze(-2), dim=-1)
    return points, velocities


def edge_cost(
    depths: torch.Tensor,
    velocities: torch.Tensor,
    epsilon: float = 1e-3,
    tolerance: float = 0.0,
) -> torch.Tensor:
    """Unweighted Instinct-style sum over both feet and all volume points.

    No stance gate, averaging over points, or velocity-direction gate is used.
    Tolerance gates the original depth; it does not subtract from that depth.
    RewardManager supplies the negative coefficient and the control timestep.
    """
    if depths.ndim != 3 or velocities.shape != (*depths.shape, 3):
        raise ValueError("Expected depths (N, B, P) and velocities (N, B, P, 3).")
    for name, value in (("epsilon", epsilon), ("tolerance", tolerance)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    return ((depths > tolerance).to(depths.dtype) * depths
            * (torch.linalg.vector_norm(velocities, dim=-1) + epsilon)).sum(dim=(1, 2))


def foot_support_diagnostics(
    ray_starts: torch.Tensor,
    ray_hits: torch.Tensor,
    ray_directions: torch.Tensor,
    contact_forces: torch.Tensor,
    distance_threshold: float = 0.04,
    contact_threshold: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return per-body support diagnostics, never a reward.

    Ray tensors have shape (N, B, R, 3); forces have shape (N, B, 3).
    Distances are measured from each actual ray origin along its actual
    direction. Missing/nonfinite hits count as unsupported, including in the
    denominator. This is a ray support proxy, not a measured contact patch.
    """
    if (ray_starts.ndim != 4 or ray_starts.shape[-1] != 3
            or ray_hits.shape != ray_starts.shape or ray_directions.shape != ray_starts.shape
            or contact_forces.shape != (*ray_starts.shape[:2], 3)):
        raise ValueError("Expected ray tensors (N, B, R, 3) and forces (N, B, 3).")
    if ray_starts.shape[-2] == 0:
        raise ValueError("Support diagnostics require at least one ray per foot.")
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive.")
    if not math.isfinite(contact_threshold) or contact_threshold < 0:
        raise ValueError("contact_threshold must be finite and nonnegative.")
    direction_length = torch.linalg.vector_norm(ray_directions, dim=-1)
    valid = (torch.isfinite(ray_starts).all(dim=-1) & torch.isfinite(ray_hits).all(dim=-1)
             & torch.isfinite(ray_directions).all(dim=-1) & (direction_length > 1e-8))
    displacement = torch.where(valid[..., None], ray_hits - ray_starts, 0.0)
    directions = torch.where(valid[..., None], ray_directions, 0.0)
    distance = (displacement * directions).sum(dim=-1) / direction_length.clamp_min(1e-8)
    supported = valid & (distance >= 0.0) & (distance < distance_threshold)
    support_fraction = supported.float().mean(dim=-1)
    finite_force = torch.isfinite(contact_forces).all(dim=-1)
    any_contact = finite_force & (torch.linalg.vector_norm(contact_forces, dim=-1) > contact_threshold)
    # Load-bearing support excludes a pure horizontal wall impact and forces
    # pointing downwards. This gate is diagnostic only, never part of edge_cost.
    contact = finite_force & (contact_forces[..., 2] > contact_threshold)
    return {
        "support_fraction": support_fraction,
        "valid_fraction": valid.float().mean(dim=-1),
        "contact": contact,
        "loaded_contact": contact,
        "any_contact": any_contact,
        "unsupported_fraction_in_contact": (1.0 - support_fraction) * contact.float(),
    }
