"""Geometry checks for the paper's foot-volume edge penalty; no simulator/training.

The frozen greedy reference below adapts InstinctLab edge_cylinder.py under
CC BY-NC 4.0. It intentionally retains the upstream hardcoded 0.05 m fit bound:
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder.py
"""
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.perception.foot_edge_geometry import EdgeGeometry, EdgeGeometryCfg, greedy_concatenate_edges


def _official_greedy_reference(edge_coords, seed=42, angle=30.0, min_points=2):
    """Frozen upstream traversal, using only a local RNG instead of global random."""
    line_pts = np.asarray(edge_coords).reshape(-1, 3)
    vertices, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
    pairs = inv_idx.reshape(-1, 2)
    adjacency = {i: set() for i in range(vertices.shape[0])}
    for start, end in pairs:
        if start != end:
            adjacency[start].add(end)
            adjacency[end].add(start)
    degrees = np.array([len(adjacency[i]) for i in range(vertices.shape[0])], dtype=int)
    available = set(np.where(degrees > 0)[0])
    cos_threshold = np.cos(np.deg2rad(angle))
    rng = random.Random(seed)
    results = []
    while available:
        selected = rng.choice(list(available))
        path = [selected]
        if adjacency[selected]:
            neighbor = next(iter(adjacency[selected]))
            path.append(neighbor)
            adjacency[selected].remove(neighbor)
            adjacency[neighbor].remove(selected)
            for vertex in (selected, neighbor):
                degrees[vertex] -= 1
                if degrees[vertex] == 0:
                    available.discard(vertex)
        while True:
            found = False
            start, end = path[0], path[-1]
            neighbors = list(adjacency[start])
            if neighbors:
                directions = vertices[start] - vertices[neighbors]
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                start_dir = vertices[path[1]] - vertices[start]
                start_dir /= np.linalg.norm(start_dir)
                choices = np.where(directions @ start_dir > cos_threshold)[0]
                if choices.size:
                    neighbor = neighbors[choices[0]]
                    path.insert(0, neighbor)
                    adjacency[start].remove(neighbor)
                    adjacency[neighbor].remove(start)
                    for vertex in (start, neighbor):
                        degrees[vertex] -= 1
                        if degrees[vertex] == 0:
                            available.discard(vertex)
                    found = True
            neighbors = list(adjacency[end])
            if neighbors:
                directions = vertices[neighbors] - vertices[end]
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                end_dir = vertices[end] - vertices[path[-2]]
                end_dir /= np.linalg.norm(end_dir)
                choices = np.where(directions @ end_dir > cos_threshold)[0]
                if choices.size:
                    neighbor = neighbors[choices[0]]
                    path.append(neighbor)
                    adjacency[end].remove(neighbor)
                    adjacency[neighbor].remove(end)
                    for vertex in (end, neighbor):
                        degrees[vertex] -= 1
                        if degrees[vertex] == 0:
                            available.discard(vertex)
                    found = True
            if not found:
                break
        while len(path) >= min_points:
            for index in range(len(path) - 1):
                first, last = vertices[path[index]], vertices[path[-1]]
                length = np.linalg.norm(last - first)
                maximum = 0.0 if length == 0 else np.linalg.norm(
                    np.cross(vertices[path[index:]] - first, vertices[path[index:]] - last), axis=1
                ).max() / length
                if maximum < 0.05:
                    break
            if len(path) - index >= min_points:
                results.append([vertices[path[index]], vertices[path[-1]]])
            path = path[:index + 1]
    return np.asarray(results, dtype=np.float32)


@pytest.mark.parametrize("seed", [0, 42, 109])
def test_greedy_matches_frozen_official_reference_and_does_not_change_global_rng(seed):
    mesh = trimesh.creation.box(extents=(1.2, 0.7, 0.2))
    raw = mesh.vertices[mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.deg2rad(70)]]
    # Add fragmented, near-collinear terrain edges and a disconnected vertical edge.
    extra = np.array([
        [[2, 0, 0], [2.2, 0, 0]], [[2.2, 0, 0], [2.4, 0.02, 0]],
        [[2.4, 0.02, 0], [2.6, 0.01, 0]], [[2.6, 0.01, 0], [2.8, 0.03, 0]],
        [[4, 1, 0], [4, 1, 0.3]],
    ])
    raw = np.concatenate((raw, extra))
    before = random.getstate()
    actual = greedy_concatenate_edges(raw, EdgeGeometryCfg(seed=seed))
    expected = _official_greedy_reference(raw, seed=seed)
    np.testing.assert_array_equal(actual, expected)
    assert random.getstate() == before


def test_fit_distance_configuration_actually_changes_fitted_segments():
    raw = np.array([[[0, 0, 0], [0.1, 0.03, 0]], [[0.1, 0.03, 0], [0.2, 0, 0]]])
    cfg = EdgeGeometryCfg(adjacent_angle_threshold_deg=60)
    assert len(greedy_concatenate_edges(raw, cfg)) == 1
    assert len(greedy_concatenate_edges(raw, replace(cfg, fit_distance_threshold=0.01))) == 2


def test_box_has_all_twelve_spatial_edges_and_input_mesh_is_unchanged():
    box = trimesh.creation.box(extents=(1.0, 0.7, 0.2))
    # Deliberately split triangle vertices: preprocessing must reconnect faces.
    mesh = trimesh.Trimesh(vertices=box.triangles.reshape(-1, 3), faces=np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
    before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
    geometry = EdgeGeometry(mesh, EdgeGeometryCfg(), "cpu")
    np.testing.assert_array_equal(mesh.vertices, before_vertices)
    np.testing.assert_array_equal(mesh.faces, before_faces)
    assert geometry.edges.shape == (12, 2, 3)
    assert sum(abs(edge[1, 2] - edge[0, 2]) > 0.1 for edge in geometry.edges) == 4


@pytest.mark.parametrize("sign", [-1, 1])
def test_sharp_extraction_keeps_both_fold_directions(sign):
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, sign]], faces=[[0, 1, 2], [1, 0, 3]], process=False)
    geometry = EdgeGeometry(mesh, EdgeGeometryCfg(), "cpu")
    assert len(geometry.edges) == 1
    np.testing.assert_allclose(np.sort(geometry.edges[0], axis=0), [[0, 0, 0], [1, 0, 0]])


def test_empty_plane_and_open_boundaries_do_not_invent_sharp_edges():
    plane = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], faces=[[0, 1, 2], [1, 3, 2]], process=False)
    geometry = EdgeGeometry(plane, EdgeGeometryCfg(), "cpu")
    assert geometry.edges.shape == (0, 2, 3)
    assert torch.count_nonzero(geometry.penetration_offsets(torch.randn(2, 4, 3))) == 0
    empty = EdgeGeometry.from_edges([], EdgeGeometryCfg(), "cpu")
    assert empty.penetration_offsets(torch.empty(0, 3)).shape == (0, 3)


def test_finite_cylinder_depth_axis_and_no_capsule_endcaps():
    geometry = EdgeGeometry.from_edges([[[0, 0, 0], [1, 0, 0]]], EdgeGeometryCfg(), "cpu")
    points = torch.tensor([[0.5, 0.02, 0], [0.5, 0.06, 0], [-0.001, 0, 0], [1.001, 0, 0], [0.5, 0, 0], [0, 0.02, 0], [1, 0.02, 0], [0.5, 0.05, 0]], dtype=torch.float64)
    offset = geometry.penetration_offsets(points)
    torch.testing.assert_close(offset.norm(dim=-1), torch.tensor([0.03, 0, 0, 0, 0.05, 0.03, 0.03, 0], dtype=torch.float64))
    torch.testing.assert_close(offset[0], torch.tensor([0, -0.03, 0], dtype=torch.float64))
    assert torch.isfinite(offset).all()


def test_overlapping_and_duplicate_cylinders_take_maximum_not_sum():
    edges = [[[0, 0, 0], [1, 0, 0]], [[0, 0.02, 0], [1, 0.02, 0]], [[0, 0.02, 0], [1, 0.02, 0]]]
    geometry = EdgeGeometry.from_edges(edges, EdgeGeometryCfg(), "cpu")
    point = torch.tensor([[0.5, 0.015, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(geometry.penetration_offsets(point), torch.tensor([[0, 0.045, 0]], dtype=torch.float64))


def test_all_world_orientations_and_translations_have_same_penetration():
    edges = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=float)
    points = torch.tensor([[0.2, 0.02, 0], [0.7, 0, -0.01]], dtype=torch.float64)
    reference = EdgeGeometry.from_edges(edges, EdgeGeometryCfg(), "cpu").penetration_offsets(points)
    rotation = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    translation = torch.tensor([5.0, -3.0, 1.7], dtype=torch.float64)
    geometry = EdgeGeometry.from_edges((torch.tensor(edges) @ rotation.T + translation).numpy(), EdgeGeometryCfg(), "cpu")
    actual = geometry.penetration_offsets(points @ rotation.T + translation)
    torch.testing.assert_close(actual, reference @ rotation.T)


def test_zero_length_nonfinite_edges_and_nonfinite_points_are_safe():
    edges = np.array([[[0, 0, 0], [0, 0, 0]], [[np.nan, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]]])
    geometry = EdgeGeometry.from_edges(edges, EdgeGeometryCfg(), "cpu")
    assert len(geometry.edges) == 1
    points = torch.tensor([[float("nan"), 0, 0], [float("inf"), 0, 0], [0.5, 0, 0]])
    output = geometry.penetration_offsets(points)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:2]) == 0
    assert output[2].norm() == pytest.approx(0.05)


def test_noncontiguous_shapes_repeated_queries_hashes_and_no_autograd():
    geometry = EdgeGeometry.from_edges([[[0, 0, 0], [1, 0, 0]]], EdgeGeometryCfg(), "cpu")
    storage = torch.tensor([[[0.5, 0.02, 0.0]] * 8] * 2, requires_grad=True)
    points = storage[:, ::2]
    assert not points.is_contiguous()
    initial = geometry.metadata()
    for _ in range(3):
        output = geometry.penetration_offsets(points)
        assert output.shape == points.shape and not output.requires_grad
    assert geometry.construction_count == 1
    assert geometry.query_count == 3
    assert geometry.metadata() == initial
    assert initial["config"]["cylinder_radius"] == 0.05
    assert len(initial["edges_sha256"]) == 64
    assert initial["upstream_source"]["commit_sha"] == "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
    assert len(initial["upstream_source"]["file_sha256"]) == 6
    assert set(initial["implementation_sha256"]) == {"foot_edge_geometry.py", "foot_edge_warp.py"}
    assert not geometry.edges.flags.writeable


@pytest.mark.parametrize("field,value", [("cylinder_radius", 0), ("cylinder_radius", float("nan")), ("angle_threshold_deg", 180), ("min_points", 1), ("num_grid_cells", True), ("seed", -1), ("fit_distance_threshold", 0)])
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        EdgeGeometryCfg(**{field: value})


def test_mutable_configuration_roundtrip_and_revalidation():
    cfg = EdgeGeometryCfg()
    payload = asdict(cfg)
    payload.update(cylinder_radius=0.06, seed=19)
    # Isaac Lab's nested update uses setattr on an existing config object.
    restored = EdgeGeometryCfg()
    for name, value in payload.items():
        setattr(restored, name, value)
    restored.validate()
    assert asdict(restored) == payload
    restored.cylinder_radius = 0.0
    with pytest.raises(ValueError, match="cylinder_radius"):
        restored.validate()
    with pytest.raises(ValueError, match="cylinder_radius"):
        EdgeGeometry.from_edges([[[0, 0, 0], [1, 0, 0]]], restored, "cpu")
    with pytest.raises(ValueError, match="cylinder_radius"):
        EdgeGeometry(trimesh.creation.box(), restored, "cpu")


@pytest.mark.parametrize("use_mesh", [False, True])
def test_geometry_snapshots_mutable_configuration(use_mesh):
    cfg = EdgeGeometryCfg()
    geometry = EdgeGeometry(trimesh.creation.box(), cfg, "cpu") if use_mesh else EdgeGeometry.from_edges(
        [[[0, 0, 0], [1, 0, 0]]], cfg, "cpu"
    )
    before = geometry.metadata()
    points = torch.tensor([[0.5, 0.02, 0.0]])
    offsets_before = geometry.penetration_offsets(points)
    cfg.cylinder_radius = 0.2
    cfg.seed = 109
    assert geometry.cfg is not cfg
    assert geometry.metadata() == before
    torch.testing.assert_close(geometry.penetration_offsets(points), offsets_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA; CPU suite remains simulator-free.")
def test_warp_spatial_index_matches_independent_torch_reference():
    pytest.importorskip("warp")
    rng = np.random.default_rng(17)
    starts = rng.uniform(-1, 1, size=(23, 3))
    ends = starts + rng.normal(size=(23, 3)) * 0.2
    edges = np.stack((starts, ends), axis=1)
    cfg = EdgeGeometryCfg()
    cpu = EdgeGeometry.from_edges(edges, cfg, "cpu")
    gpu = EdgeGeometry.from_edges(edges, cfg, "cuda")
    probes = torch.tensor(rng.uniform(-1.5, 1.5, size=(101, 3)), dtype=torch.float64)
    probes = torch.cat((probes, torch.tensor((starts + ends) / 2), torch.tensor(starts)), dim=0)
    expected = cpu.penetration_offsets(probes)
    actual = gpu.penetration_offsets(probes.cuda()).cpu()
    # At an axis, offset direction is arbitrary; its radial depth is defined.
    torch.testing.assert_close(actual.norm(dim=-1), expected.norm(dim=-1), atol=3e-6, rtol=3e-5)
    ordinary = torch.arange(len(probes)) < 101
    torch.testing.assert_close(actual[ordinary], expected[ordinary], atol=3e-6, rtol=3e-5)
    assert gpu._backend.construction_count == 1
