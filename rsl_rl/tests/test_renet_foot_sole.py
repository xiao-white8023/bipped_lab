from __future__ import annotations

import ast
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import torch

from test_renet_auxiliary_updates import ROOT, _extract_function


def _pattern_functions():
    # Exercise the actual STL loader and point-generation functions without
    # starting Kit merely to import IsaacLab's configuration decorators.
    path = ROOT / 'legged_lab/sensors/foot_sole_pattern.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    scope = dict(np=np, torch=torch, ConvexHull=ConvexHull, math=math, Path=Path, struct=struct)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope['load_sole_outline'], scope['foot_sole_pattern']


def _pattern(side='left', **kwargs):
    load, generate = _pattern_functions()
    cfg = SimpleNamespace(mesh_path=str(ROOT / f'legged_lab/assets/g1/g1_23dof/meshes/{side}_ankle_roll_link.STL'),
                          resolution=.01, edge_margin=.002, sole_slice_tolerance=.001)
    cfg.__dict__.update(kwargs)
    starts, directions = generate(cfg, 'cpu')
    return starts, directions, load(cfg.mesh_path)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_ray_grid_covers_heel_toe_and_sides_without_sampling_outside_sole(side):
    starts, directions, outline = _pattern(side)
    assert starts.shape[0] > 100 and starts.shape == directions.shape
    assert torch.isfinite(starts).all() and not starts[:, 2].any()
    torch.testing.assert_close(directions, torch.tensor([0., 0., -1.]).expand_as(directions))
    xy = starts[:, :2].numpy().astype(np.float64)
    hull = ConvexHull(outline)
    distances = xy @ hull.equations[:, :2].T + hull.equations[:, 2]
    assert (distances <= -.002 + 1e-8).all()
    np.testing.assert_allclose(xy.min(0), [-.06, -.03], atol=1e-8)
    np.testing.assert_allclose(xy.max(0), [.14, .03], atol=1e-8)
    np.testing.assert_allclose(xy / .01, np.round(xy / .01), atol=1e-6)
    points = {(round(x, 3), round(y, 3)) for x, y in xy}
    assert points == {(x, -y) for x, y in points}
    assert (-.06, .03) not in points  # Rounded heel corner is outside the inset.


def test_left_and_right_sole_patterns_are_mirrored():
    left, _, _ = _pattern('left')
    right, _, _ = _pattern('right')
    lp = {(round(float(x), 3), round(float(y), 3)) for x, y in left[:, :2]}
    rp = {(round(float(x), 3), -round(float(y), 3)) for x, y in right[:, :2]}
    assert lp == rp


@pytest.mark.parametrize('axis,edge,sign', [(0, -.04, 1), (0, .11, -1), (1, .02, -1), (1, -.02, 1)])
def test_expanded_grid_detects_heel_toe_and_lateral_overhang_missed_by_old_grid(axis, edge, sign):
    reward = _extract_function(ROOT / 'legged_lab/mdp/rewards.py', 'single_foot_contact_area_penalty')
    starts, _, _ = _pattern()
    x, y = torch.meshgrid(torch.arange(7) * .02 - .025, torch.tensor([-.0175, .0025]), indexing='xy')
    old = torch.stack((x.flatten(), y.flatten(), torch.zeros(14)), -1)
    def evaluate(points, edge_enabled=True):
        supported = sign * (points[:, axis] - edge) >= 0 if edge_enabled else torch.ones(len(points), dtype=torch.bool)
        hits = torch.zeros(1, len(points), 3)
        hits[..., 2] = torch.where(supported, .0, -.10)
        forces = torch.zeros(1, 1, 1, 3)
        forces[..., 2] = 20.
        env = SimpleNamespace(scene=SimpleNamespace(sensors={
            'contact_sensor': SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces)),
            'foot': SimpleNamespace(data=SimpleNamespace(pos_w=torch.zeros(1, 3), ray_hits_w=hits))}))
        return reward(env, threshold=.04,
            contact_sensor_cfg=SimpleNamespace(name='contact_sensor', body_ids=[0]),
            ray_sensor_cfg=SimpleNamespace(name='foot')).item()
    assert evaluate(old) == 0.
    assert 0. < evaluate(starts) < 1.
    assert evaluate(starts, edge_enabled=False) == 0.


@pytest.mark.parametrize('kwargs', [{'resolution': 0}, {'resolution': float('nan')},
    {'edge_margin': -.001}, {'edge_margin': float('inf')}, {'edge_margin': 1.}])
def test_invalid_or_empty_sole_patterns_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _pattern(**kwargs)
