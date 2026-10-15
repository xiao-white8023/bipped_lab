"""Terrain-edge geometry for the foot-volume penalty in Hiking in the Wild.

The sharp-edge extraction and greedy concatenation adapt Project Instinct's
InstinctLab (CC BY-NC 4.0), rather than the surrounding repository's BSD license.
Sources, reviewed 2026-10-15, commit de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057:
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder.py
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/utils/warp/cylinder.py
See FOOT_EDGE_UPSTREAM_LICENSE.txt; this notice concerns these derived modules,
not unrelated files in this repository.
Changes: local RNG, configurable fit tolerance, finite/degenerate guards, and a
standalone Torch reference. Retain this attribution when redistributing.

No Isaac Sim or Warp imports are needed for the CPU reference. A point is inside
a finite cylinder only when its axial projection is on the segment: there are
no spherical end caps. Overlapping cylinders use the maximum radial penetration.
Offsets point towards the cylinder axis, as upstream; they are reward geometry,
not a physical collision-resolution displacement.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import trimesh


UPSTREAM_COMMIT = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
UPSTREAM_FILE_SHA256 = {
    "source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder.py":
        "a35a607939fbcb96d54bff51b7c7253fe800ba02151bf09f2ce2dea69ce1cc91",
    "source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder_cfg.py":
        "c5759cc1bd568ead45476265da10e6a63e2f31bfbd6d1001691e20ccbc2b425a",
    "source/instinctlab/instinctlab/utils/warp/cylinder.py":
        "429670aef24a039bc1afe5157b1572c75d35aad4a8ddc1c54ffc82601de646cf",
    "source/instinctlab/instinctlab/utils/warp/kernels.py":
        "1e8534984dd41d5fbcba38706b516d9d167172357ce7ce28768fb443a8609f4c",
    "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py":
        "4b6d2b43ad0971a4b69091863d7c7cb931ed3e142d54bbd3956785bd2b681ed7",
    "source/instinctlab/instinctlab/envs/mdp/rewards/volume_points.py":
        "e3341a7546d696e3b17b7d7cdeff2c08d916b92a496cec0ca912381deb3d1322",
}


@dataclass
class EdgeGeometryCfg:
    angle_threshold_deg: float = 70.0
    cylinder_radius: float = 0.05
    adjacent_angle_threshold_deg: float = 30.0
    fit_distance_threshold: float = 0.05
    min_points: int = 2
    num_grid_cells: int = 64**3
    seed: int = 42

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate also after Isaac Lab updates nested configuration attributes."""
        for name in ("angle_threshold_deg", "adjacent_angle_threshold_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 180:
                raise ValueError(f"{name} must be finite and in [0, 180).")
        for name in ("cylinder_radius", "fit_distance_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, minimum in (("min_points", 2), ("num_grid_cells", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")


def _array_hash(array: np.ndarray, dtype) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _mesh_hash(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(_array_hash(mesh.vertices, "<f8").encode("ascii"))
    digest.update(_array_hash(mesh.faces, "<i8").encode("ascii"))
    return digest.hexdigest()


def _valid_edges(edges) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    if edges.ndim == 2 and edges.shape[1] == 6:
        edges = edges.reshape(-1, 2, 3)
    if edges.ndim != 3 or edges.shape[1:] != (2, 3):
        raise ValueError("Edges must have shape (E, 2, 3) or (E, 6).")
    finite = np.isfinite(edges).all(axis=(1, 2))
    edges = edges[finite]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=-1)
    return np.ascontiguousarray(edges[lengths > 1e-12])


def greedy_concatenate_edges(edges, cfg: EdgeGeometryCfg) -> np.ndarray:
    """Upstream greedy graph traversal, with an isolated RNG and fit tolerance.

    The source implementation hardcodes a 0.05 m line-fit distance; its separate
    0.06 m configuration field is unused. Here the actual 0.05 m value is exposed.
    All directions and both convex and concave sharp edges are retained.
    """
    edges = _valid_edges(edges)
    if not len(edges):
        return edges
    vertices, inverse = np.unique(edges.reshape(-1, 3), axis=0, return_inverse=True)
    pairs = inverse.reshape(-1, 2)
    adjacency = {i: set() for i in range(len(vertices))}
    for start, end in pairs:
        if start != end:
            adjacency[start].add(end)
            adjacency[end].add(start)
    degrees = np.array([len(adjacency[i]) for i in range(len(vertices))])
    available = set(np.where(degrees > 0)[0])
    rng = random.Random(cfg.seed)
    cos_threshold = np.cos(np.deg2rad(cfg.adjacent_angle_threshold_deg))
    result = []

    def consume(start, end):
        adjacency[start].remove(end)
        adjacency[end].remove(start)
        for vertex in (start, end):
            degrees[vertex] -= 1
            if degrees[vertex] == 0:
                available.discard(vertex)

    def max_distance_to_line(path):
        first, last = vertices[path[0]], vertices[path[-1]]
        length = np.linalg.norm(last - first)
        if length <= 1e-12:
            return 0.0
        return np.linalg.norm(np.cross(vertices[path] - first, vertices[path] - last), axis=1).max() / length

    while available:
        vertex = rng.choice(list(available))
        path = [vertex]
        if adjacency[vertex]:
            neighbor = next(iter(adjacency[vertex]))
            path.append(neighbor)
            consume(vertex, neighbor)
        while True:
            extended = False
            start, end = path[0], path[-1]
            neighbors = list(adjacency[start])
            if neighbors:
                directions = vertices[start] - vertices[neighbors]
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                reference = vertices[path[1]] - vertices[start]
                reference /= np.linalg.norm(reference)
                candidates = np.where(directions @ reference > cos_threshold)[0]
                if candidates.size:
                    neighbor = neighbors[candidates[0]]
                    path.insert(0, neighbor)
                    consume(start, neighbor)
                    extended = True
            neighbors = list(adjacency[end])
            if neighbors:
                directions = vertices[neighbors] - vertices[end]
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                reference = vertices[end] - vertices[path[-2]]
                reference /= np.linalg.norm(reference)
                candidates = np.where(directions @ reference > cos_threshold)[0]
                if candidates.size:
                    neighbor = neighbors[candidates[0]]
                    path.append(neighbor)
                    consume(end, neighbor)
                    extended = True
            if not extended:
                break
        while len(path) >= cfg.min_points:
            for index in range(len(path) - 1):
                if max_distance_to_line(path[index:]) < cfg.fit_distance_threshold:
                    break
            if len(path) - index >= cfg.min_points:
                result.append([vertices[path[index]], vertices[path[-1]]])
            path = path[: index + 1]
    # Upstream emits float32 endpoints. Retain that rounding before hashing or
    # building either backend, so CPU and CUDA query the same constructed mesh.
    return _valid_edges(np.asarray(result, dtype=np.float32))


class EdgeGeometry:
    """Build static edge geometry once, then query many foot-point batches."""

    def __init__(self, mesh: trimesh.Trimesh, cfg: EdgeGeometryCfg, device: str | torch.device):
        cfg.validate()
        self.cfg = copy.deepcopy(cfg)
        cfg = self.cfg
        self.device = torch.device(device)
        self.construction_count = 1
        self._input_mesh_sha256 = _mesh_hash(mesh)
        working = mesh.copy()
        if len(working.faces):
            finite_faces = np.isfinite(working.vertices[working.faces]).all(axis=(1, 2))
            working.update_faces(finite_faces)
        working.remove_unreferenced_vertices()
        working.merge_vertices()
        if len(working.faces):
            if hasattr(working, "unique_faces"):
                working.update_faces(working.unique_faces())
            else:
                _, unique = np.unique(np.sort(working.faces, axis=1), axis=0, return_index=True)
                working.update_faces(np.sort(unique))
            working.remove_unreferenced_vertices()
        self._mesh_sha256 = _mesh_hash(working)
        self._mesh_counts = {"vertices": len(working.vertices), "faces": len(working.faces)}
        sharp = working.face_adjacency_angles > np.deg2rad(cfg.angle_threshold_deg)
        raw = working.vertices[working.face_adjacency_edges[sharp]]
        self._raw_edge_count = len(raw)
        self._initialize_edges(greedy_concatenate_edges(raw, cfg))

    @classmethod
    def from_edges(cls, edges, cfg: EdgeGeometryCfg, device: str | torch.device):
        """Construct explicit finite cylinders without extracting/merging a mesh."""
        cfg.validate()
        instance = cls.__new__(cls)
        instance.cfg = copy.deepcopy(cfg)
        instance.device = torch.device(device)
        instance.construction_count = 1
        instance._input_mesh_sha256 = instance._mesh_sha256 = None
        instance._mesh_counts = None
        instance._raw_edge_count = int(np.asarray(edges).shape[0]) if np.asarray(edges).ndim else 0
        instance._initialize_edges(edges)
        return instance

    def _initialize_edges(self, edges):
        self.edges = _valid_edges(edges).copy()
        self.edges.setflags(write=False)
        self._edges_sha256 = _array_hash(self.edges, "<f8")
        self._backend = None
        self.query_count = 0
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("EdgeGeometry supports CPU reference and CUDA/Warp devices only.")
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            from .foot_edge_warp import CylinderSpatialGrid

            self._backend = CylinderSpatialGrid(self.edges, self.cfg.cylinder_radius, self.cfg.num_grid_cells, self.device)

    @torch.no_grad()
    def penetration_offsets(self, points: torch.Tensor) -> torch.Tensor:
        """Maximum radial offset per point; invalid points return zero offsets.

        CPU execution is an independent, chunked Torch reference for tests and
        small audits. CUDA execution uses a prebuilt spatial index, never an
        all-points-by-all-edges tensor. No contact/stance mask is applied.
        """
        if not isinstance(points, torch.Tensor) or points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("Points must be a floating tensor of shape (..., 3).")
        if not points.is_floating_point():
            raise ValueError("Points must have a floating dtype.")
        if points.device.type != self.device.type or (
            self.device.type == "cuda" and points.device.index != self.device.index
        ):
            raise ValueError(f"Point device {points.device} differs from geometry device {self.device}.")
        self.query_count += 1
        if self._backend is not None:
            return self._backend.penetration_offsets(points)
        return self._torch_reference(points)

    def _torch_reference(self, points):
        flat = points.reshape(-1, 3)
        output = torch.zeros_like(flat)
        if not len(self.edges) or not len(flat):
            return output.reshape_as(points)
        work_dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
        edges = torch.tensor(self.edges, dtype=work_dtype, device=points.device)
        for point_start in range(0, len(flat), 512):
            values = flat[point_start : point_start + 512].to(work_dtype)
            finite = torch.isfinite(values).all(dim=-1)
            values = torch.where(finite[:, None], values, 0.0)
            best_depth = torch.zeros(len(values), dtype=work_dtype, device=points.device)
            best_offset = torch.zeros_like(values)
            for edge_start in range(0, len(edges), 256):
                selected = edges[edge_start : edge_start + 256]
                start = selected[:, 0]
                difference = selected[:, 1] - start
                lengths = difference.norm(dim=-1)
                direction = difference / lengths[:, None]
                axial = ((values[:, None] - start[None]) * direction[None]).sum(dim=-1)
                toward_axis = start[None] + axial[..., None] * direction[None] - values[:, None]
                distance = toward_axis.norm(dim=-1)
                inside = (axial >= 0) & (axial <= lengths[None]) & (distance < self.cfg.cylinder_radius)
                depths = torch.where(inside, self.cfg.cylinder_radius - distance, 0.0)
                depths = torch.where(finite[:, None], depths, 0.0)
                chosen_depth, chosen = depths.max(dim=-1)
                rows = torch.arange(len(values), device=points.device)
                toward = toward_axis[rows, chosen]
                norm = distance[rows, chosen]
                axis = direction[chosen]
                reference = torch.zeros_like(axis)
                reference[:, 0] = 1.0
                reference[axis[:, 0].abs() > 0.9] = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=work_dtype, device=points.device
                )
                perpendicular = torch.linalg.cross(axis, reference, dim=-1)
                perpendicular /= perpendicular.norm(dim=-1, keepdim=True)
                unit = torch.where((norm > 1e-12)[:, None], toward / norm.clamp_min(1e-12)[:, None], perpendicular)
                offsets = unit * chosen_depth[:, None]
                improved = chosen_depth > best_depth
                best_depth[improved] = chosen_depth[improved]
                best_offset[improved] = offsets[improved]
            output[point_start : point_start + len(values)] = best_offset.to(output.dtype)
        return output.reshape_as(points)

    def metadata(self) -> dict:
        """Return actual configuration, mesh/edge hashes and pinned upstream sources."""
        directory = Path(__file__).resolve().parent
        implementation_hashes = {}
        for name in ("foot_edge_geometry.py", "foot_edge_warp.py"):
            path = directory / name
            if path.is_file():
                implementation_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return copy.deepcopy({
            "method": "instinct_finite_edge_cylinders_max_radial_penetration",
            "config": asdict(self.cfg),
            "input_mesh_sha256": self._input_mesh_sha256,
            "mesh_sha256": self._mesh_sha256,
            "edges_sha256": self._edges_sha256,
            "mesh_counts": self._mesh_counts,
            "raw_edge_count": self._raw_edge_count,
            "edge_count": len(self.edges),
            "construction_count": self.construction_count,
            "backend": "warp_spatial_grid" if self._backend is not None else "torch_reference",
            "upstream_source": {
                "repository": "https://github.com/project-instinct/InstinctLab",
                "reviewed_ref": UPSTREAM_COMMIT,
                "commit_sha": UPSTREAM_COMMIT,
                "license": "CC-BY-NC-4.0",
                "license_file": "legged_lab/perception/FOOT_EDGE_UPSTREAM_LICENSE.txt",
                "license_sha256": "0afbf192c4dfb4f8ffd7783eb61a8829d1ebee17d3f2b1dc1b0bdb62838f2a93",
                "file_sha256": UPSTREAM_FILE_SHA256,
            },
            "implementation_sha256": implementation_hashes,
        })
