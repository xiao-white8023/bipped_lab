"""CUDA spatial queries adapted from InstinctLab, CC BY-NC 4.0.

Upstream attribution (reviewed 2026-10-15):
commit de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/utils/warp/cylinder.py
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/utils/warp/kernels.py
Changes: standalone finite-cylinder index, empty/invalid input handling and guards
against zero-length axes and points exactly on an axis. This module is imported
only for CUDA geometry. No Isaac Sim APIs or physical colliders are created.
See FOOT_EDGE_UPSTREAM_LICENSE.txt; this derived-module notice does not relicense
unrelated files in this repository.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _maximum_cylinder_offset(
    points: wp.array(dtype=wp.vec3),
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    radius: float,
    cell_offsets: wp.array(dtype=wp.int32),
    cell_edges: wp.array(dtype=wp.int32),
    grid_res: wp.vec3i,
    bbox_min: wp.vec3,
    cell_size: wp.vec3,
    output: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point = points[tid]
    delta = point - bbox_min
    ix = int(wp.floor(delta[0] / cell_size[0]))
    iy = int(wp.floor(delta[1] / cell_size[1]))
    iz = int(wp.floor(delta[2] / cell_size[2]))
    best_depth = float(0.0)
    best_offset = wp.vec3(0.0, 0.0, 0.0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                x = ix + dx
                y = iy + dy
                z = iz + dz
                if x < 0 or x >= grid_res[0] or y < 0 or y >= grid_res[1] or z < 0 or z >= grid_res[2]:
                    continue
                cell = x * grid_res[1] * grid_res[2] + y * grid_res[2] + z
                for entry in range(cell_offsets[cell], cell_offsets[cell + 1]):
                    edge = cell_edges[entry]
                    start = starts[edge]
                    difference = ends[edge] - start
                    length = wp.length(difference)
                    if length <= 1.0e-12:
                        continue
                    direction = difference / length
                    axial = wp.dot(point - start, direction)
                    if axial < 0.0 or axial > length:
                        continue
                    toward = start + axial * direction - point
                    distance = wp.length(toward)
                    if distance < radius:
                        depth = radius - distance
                        if depth > best_depth:
                            best_depth = depth
                            if distance > 1.0e-12:
                                best_offset = toward * (depth / distance)
                            else:
                                reference = wp.vec3(1.0, 0.0, 0.0)
                                if wp.abs(direction[0]) > 0.9:
                                    reference = wp.vec3(0.0, 1.0, 0.0)
                                best_offset = wp.normalize(wp.cross(direction, reference)) * depth
    output[tid] = best_offset


class CylinderSpatialGrid:
    """AABB-to-CSR index built once for a static set of finite cylinders."""

    def __init__(self, edges, radius: float, num_grid_cells: int, device):
        self.device = torch.device(device)
        self.radius = float(radius)
        self.construction_count = 1
        edges = np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3)
        self.empty = not len(edges)
        if self.empty:
            return
        self.bbox_min = edges.min(axis=(0, 1)) - radius
        extent = edges.max(axis=(0, 1)) + radius - self.bbox_min
        relative = extent / extent.max()
        self.grid_res = np.maximum(np.round(relative * num_grid_cells ** (1 / 3)), 1).astype(np.int32)
        self.cell_size = extent / self.grid_res
        total = int(np.prod(self.grid_res, dtype=np.int64))
        cells = defaultdict(list)
        for edge_id, edge in enumerate(edges):
            low = np.floor((edge.min(axis=0) - radius - self.bbox_min) / self.cell_size).astype(np.int64)
            high = np.floor((edge.max(axis=0) + radius - self.bbox_min) / self.cell_size).astype(np.int64)
            low = np.maximum(low, 0)
            high = np.minimum(high, self.grid_res - 1)
            for x in range(int(low[0]), int(high[0]) + 1):
                for y in range(int(low[1]), int(high[1]) + 1):
                    for z in range(int(low[2]), int(high[2]) + 1):
                        index = int(x * self.grid_res[1] * self.grid_res[2] + y * self.grid_res[2] + z)
                        cells[index].append(edge_id)
        offsets = np.zeros(total + 1, dtype=np.int32)
        indices = []
        for cell in range(total):
            offsets[cell] = len(indices)
            indices.extend(cells.get(cell, ()))
        offsets[-1] = len(indices)
        device_name = str(self.device)
        self._offsets = wp.array(offsets, dtype=wp.int32, device=device_name)
        self._indices = wp.array(np.asarray(indices, dtype=np.int32), dtype=wp.int32, device=device_name)
        self._starts = wp.array(np.ascontiguousarray(edges[:, 0], dtype=np.float32), dtype=wp.vec3, device=device_name)
        self._ends = wp.array(np.ascontiguousarray(edges[:, 1], dtype=np.float32), dtype=wp.vec3, device=device_name)

    @torch.no_grad()
    def penetration_offsets(self, points: torch.Tensor) -> torch.Tensor:
        if self.empty or points.numel() == 0:
            return torch.zeros_like(points)
        flat = points.reshape(-1, 3)
        finite = torch.isfinite(flat).all(dim=-1)
        work = torch.where(finite[:, None], flat, 0.0).to(torch.float32).contiguous()
        output = torch.zeros_like(work)
        wp.launch(
            _maximum_cylinder_offset,
            dim=len(work),
            inputs=[
                wp.from_torch(work, dtype=wp.vec3), self._starts, self._ends, self.radius,
                self._offsets, self._indices, wp.vec3i(*[int(value) for value in self.grid_res]),
                wp.vec3(*[float(value) for value in self.bbox_min]),
                wp.vec3(*[float(value) for value in self.cell_size]), wp.from_torch(output, dtype=wp.vec3),
            ],
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None,
        )
        output.masked_fill_(~finite[:, None], 0.0)
        return output.to(dtype=points.dtype).reshape_as(points)
