"""Height/support-preserving bilinear transforms in physical map coordinates."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def transform_lp(
    data: torch.Tensor,
    translation_m: torch.Tensor,
    yaw_deg: torch.Tensor,
    *,
    resolution_m: float = 0.1,
    invalid_height: float = -1.1,
    valid_support: torch.Tensor | None = None,
    return_support: bool = False,
):
    """Sample at R(yaw) * xy - translation, in physical metres.

    Height and raw support use the same bilinear grid. Unknown heights
    contribute zero weight; -1.1 is never part of a height average. A
    rectangular map needs physical aspect factors in normalized coordinates.
    Supports a subset of rows so unrelated environments need no warp.
    """
    n = data.shape[0]
    shape = tuple(data.shape[1:])
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("LP transform resolution must be finite and positive")
    if data.ndim != 3 or translation_m.shape != (n, 2) or yaw_deg.shape != (n,):
        raise ValueError("LP transform data, translation and yaw shapes must match")
    if valid_support is not None and valid_support.shape != data.shape:
        raise ValueError("LP transform support must match height")
    if n == 0:
        support = torch.zeros_like(data) if valid_support is None else valid_support.clone()
        return (data.clone(), support) if return_support else data.clone()

    angle = torch.deg2rad(yaw_deg)
    cosine, sine = angle.cos(), angle.sin()
    theta = torch.zeros(n, 2, 3, device=data.device, dtype=data.dtype)
    width_m = max((shape[1] - 1) * resolution_m, resolution_m)
    height_m = max((shape[0] - 1) * resolution_m, resolution_m)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine * height_m / width_m
    theta[:, 1, 0] = sine * width_m / height_m
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = -2.0 * translation_m[:, 0] / width_m
    theta[:, 1, 2] = -2.0 * translation_m[:, 1] / height_m
    grid = F.affine_grid(
        theta, (n, 2, *shape), align_corners=True
    )
    finite = torch.isfinite(data) & (data != invalid_height)
    weights = finite.to(data.dtype)
    if valid_support is not None:
        weights = weights * valid_support.to(data.dtype)
    weighted_height = torch.where(finite, data, 0.0) * weights
    sampled = F.grid_sample(
        torch.stack((weighted_height, weights), 1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    support = sampled[:, 1].clamp(0, 1)
    has_support = support > 1.0e-6
    height = torch.where(
        has_support,
        sampled[:, 0] / support.clamp_min(1.0e-6),
        torch.full_like(support, invalid_height),
    )
    support = torch.where(has_support, support, 0.0)
    return (height, support) if return_support else height

