"""Analytic foot-volume, COM velocity, and support diagnostics regressions.

These tests never construct a policy, optimizer, or learning runner.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_lab.perception.foot_edge import (
    FootEdgeCfg,
    edge_cost,
    foot_support_diagnostics,
    make_volume_points,
    point_kinematics,
)
from legged_lab.perception.foot_edge_geometry import EdgeGeometry, EdgeGeometryCfg


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("standalone_foot_edge_adapter", ROOT / "legged_lab/mdp/foot_edge.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_default_100_point_grid_covers_g1_heel_to_toe_in_link_frame():
    points = make_volume_points(FootEdgeCfg(), "cpu")
    assert points.shape == (100, 3)
    torch.testing.assert_close(points.amin(dim=0), torch.tensor([-.06, -.03, -.03]))
    torch.testing.assert_close(points.amax(dim=0), torch.tensor([.14, .03, .0]))
    assert len(points.unique(dim=0)) == 100
    assert len(points[points[:, 0] < -.025]) > 0  # Heel omitted by the upstream default.
    assert len(points[points[:, 0] > .12]) > 0
    torch.testing.assert_close(points[:, 1].sort().values, (-points[:, 1]).sort().values)


@pytest.mark.parametrize("change", [
    {"x_range": (1., 0.)}, {"z_range": (0., float("inf"))},
    {"grid_shape": (10, 5, True)}, {"grid_shape": (10, 5, 1)},
    {"velocity_epsilon": -.1}, {"velocity_epsilon": float("nan")},
    {"tolerance": -.1}, {"enabled": 1}, {"support_distance_threshold": 0.},
])
def test_invalid_cfg_is_rejected(change):
    with pytest.raises(ValueError):
        FootEdgeCfg(**change)


def test_world_point_velocity_uses_com_and_includes_rotation():
    # The COM is 1m above the link origin; rotating about +Z should use the
    # rotated point's displacement from that COM, not the local point vector.
    local = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    pos = torch.tensor([[[2., 3., 4.], [2., 3., 4.]]])
    q = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)]]]).expand(1, 2, 4)
    com = pos + torch.tensor([0., 0., 1.])
    vel = torch.tensor([[[.5, .25, 0.]]]).expand(1, 2, 3)
    omega = torch.tensor([[[0., 0., 2.]]]).expand(1, 2, 3)
    world, point_vel = point_kinematics(local, pos, q, com, vel, omega)
    expected_world = torch.tensor([[[[2., 4., 4.], [2., 3., 4.]],
                                    [[2., 4., 4.], [2., 3., 4.]]]])
    expected_velocity = torch.tensor([[[[-1.5, .25, 0.], [.5, .25, 0.]],
                                       [[-1.5, .25, 0.], [.5, .25, 0.]]]])
    torch.testing.assert_close(world, expected_world)
    torch.testing.assert_close(point_vel, expected_velocity)


def test_com_offset_changes_angular_point_speed_even_for_zero_local_point():
    pos = torch.zeros(1, 2, 3)
    q = torch.tensor([1., 0., 0., 0.]).expand(1, 2, 4)
    com = torch.tensor([.026505, 0., -.016425]).expand(1, 2, 3)
    omega = torch.tensor([0., 2., 0.]).expand(1, 2, 3)
    _, speed = point_kinematics(torch.zeros(1, 3), pos, q, com, pos, omega)
    torch.testing.assert_close(speed[0, 0, 0], torch.tensor([.03285, 0., .05301]))


def test_cost_is_original_sum_with_speed_norm_and_tolerance_gate_only():
    depth = torch.tensor([[[.01, .02], [.03, .04]], [[0., 0.], [0., 0.]]])
    velocity = torch.zeros(2, 2, 2, 3)
    velocity[0, 0, 0] = torch.tensor([3., 4., 0.])
    velocity[0, 1, 0] = torch.tensor([-3., 0., 4.])
    expected = torch.tensor([.01 * 5.001 + .02 * .001 + .03 * 5.001 + .04 * .001, 0.])
    torch.testing.assert_close(edge_cost(depth, velocity), expected)
    # Tolerance is not subtracted, and exactly-equal depths are excluded.
    expected_gated = torch.tensor([.03 * 5.001 + .04 * .001, 0.])
    torch.testing.assert_close(edge_cost(depth, velocity, tolerance=.02), expected_gated)
    torch.testing.assert_close(edge_cost(depth, torch.zeros_like(velocity)), depth.sum((1, 2)) * .001)
    # Neither a direction projection nor a contact gate appears in the API.
    torch.testing.assert_close(edge_cost(depth, -velocity), expected)


def _support_example():
    starts = torch.zeros(1, 2, 4, 3)
    starts[..., 2] = .02
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.
    hits = starts + directions * .02
    forces = torch.tensor([[[0., 0., 20.], [0., 0., 20.]]])
    return starts, hits, directions, forces


def test_support_counts_overhang_and_missed_rays_in_fixed_denominator():
    starts, hits, directions, forces = _support_example()
    hits[:, 0, 2:, 2] = -.2
    hits[:, 1, 3] = torch.inf
    result = foot_support_diagnostics(starts, hits, directions, forces)
    torch.testing.assert_close(result["support_fraction"], torch.tensor([[.5, .75]]))
    torch.testing.assert_close(result["valid_fraction"], torch.tensor([[1., .75]]))
    torch.testing.assert_close(result["unsupported_fraction_in_contact"], torch.tensor([[.5, .25]]))
    assert all(torch.isfinite(value).all() for value in result.values())


def test_support_loaded_contact_excludes_horizontal_or_downward_force():
    starts, hits, directions, forces = _support_example()
    hits[..., 2] = -.2
    forces[0] = torch.tensor([[20., 0., 0.], [0., 0., -20.]])
    result = foot_support_diagnostics(starts, hits, directions, forces)
    assert result["any_contact"].all()
    assert not result["loaded_contact"].any()
    assert not result["unsupported_fraction_in_contact"].any()


def test_support_uses_each_origin_and_direction_for_a_tilted_foot():
    starts, _, directions, forces = _support_example()
    # Distinct world heights, with each hit exactly 2cm along the foot's ray.
    starts[..., 2] = torch.tensor([.02, .05, .08, .11])
    directions[..., 0] = .6
    directions[..., 2] = -.8
    hits = starts + directions * .02
    result = foot_support_diagnostics(starts, hits, directions, forces)
    torch.testing.assert_close(result["support_fraction"], torch.ones(1, 2))


def test_nonfinite_direction_zero_direction_and_backward_hit_are_unsupported():
    starts, hits, directions, forces = _support_example()
    directions[:, 0, 0] = torch.nan
    directions[:, 0, 1] = 0.
    hits[:, 0, 2, 2] = .1  # Hit behind the start, opposite to ray direction.
    result = foot_support_diagnostics(starts, hits, directions, forces)
    torch.testing.assert_close(result["support_fraction"], torch.tensor([[.25, 1.]]))


class _Scene(dict):
    def __init__(self, robot, geometry):
        super().__init__(robot=robot)
        self.terrain = SimpleNamespace(foot_edge_geometry=geometry)
        self.sensors = {}


def _monitor_env(*, enabled=True, geometry=True):
    cfg = FootEdgeCfg(enabled=enabled, support_diagnostics=False)
    query = EdgeGeometry.from_edges(np.array([[[-1., 0., 0.], [1., 0., 0.]]]),
                                    EdgeGeometryCfg(), "cpu") if geometry else None
    position = torch.zeros(2, 2, 3)
    data = SimpleNamespace(
        body_link_pos_w=position,
        body_link_quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 2, 4),
        body_com_pos_w=position,
        body_com_lin_vel_w=torch.tensor([1., 0., 0.]).expand(2, 2, 3),
        body_com_ang_vel_w=torch.zeros(2, 2, 3),
    )
    robot = SimpleNamespace(data=data, find_bodies=lambda names, preserve_order: ([0, 1], names))
    env = SimpleNamespace(cfg=SimpleNamespace(foot_edge=cfg), device="cpu", num_envs=2,
                          scene=_Scene(robot, query), sim_step_counter=0,
                          _foot_edge_reward_weight=-4.0, step_dt=.02)
    env.foot_edge_monitor = adapter.FootEdgeMonitor(env)
    return env


def test_monitor_queries_once_per_step_and_reset_invalidates_without_history():
    env = _monitor_env()
    monitor = env.foot_edge_monitor
    cost = adapter.foot_edge_penalty(env).clone()
    assert (cost > 0.).all()
    assert monitor.points_world.shape == monitor.point_velocities_world.shape == (2, 2, 100, 3)
    assert monitor.penetration_depths.shape == (2, 2, 100)
    calls = monitor.geometry.query_count
    log = monitor.diagnostics()
    torch.testing.assert_close(monitor.cost(), cost)
    assert monitor.geometry.query_count == calls
    assert "FootEdge/raw_cost" in log and not any(name.startswith("FootSupport/") for name in log)
    torch.testing.assert_close(log["FootEdge/weighted_reward"], cost.mean() * -.08)
    torch.testing.assert_close(log["FootEdge/left/raw_cost"] + log["FootEdge/right/raw_cost"], cost.mean())
    monitor.reset(torch.tensor([], dtype=torch.long))
    monitor.cost()
    assert monitor.geometry.query_count == calls
    env.sim_step_counter += 1
    monitor.cost()
    assert monitor.geometry.query_count == calls + 1
    monitor.reset(torch.tensor([0]))
    monitor.cost()
    assert monitor.geometry.query_count == calls + 2
    assert monitor.geometry.construction_count == 1


@pytest.mark.parametrize("enabled,geometry", [(False, False), (False, True), (True, False)])
def test_disabled_or_zero_weight_without_query_is_zero_cost(enabled, geometry):
    env = _monitor_env(enabled=enabled, geometry=geometry)
    assert not env.foot_edge_monitor.cost().any()
    assert not env.foot_edge_monitor.penetration_depths.any()
    if geometry:
        assert env.foot_edge_monitor.geometry.query_count == 0


def test_support_logs_are_independent_of_edge_cost_and_work_when_disabled():
    env = _monitor_env(enabled=False, geometry=False)
    env.cfg.foot_edge.support_diagnostics = True
    force_data = SimpleNamespace(net_forces_w=torch.tensor([
        [[0., 0., 20.], [20., 0., 0.]],
        [[0., 0., 20.], [0., 0., 20.]],
    ]))
    contact_sensor = SimpleNamespace(
        data=force_data, find_bodies=lambda names, preserve_order: ([0, 1], names))
    env.scene.sensors["contact_sensor"] = contact_sensor
    for side in ("left", "right"):
        starts = torch.tensor([[[-.01, 0., .02], [.01, 0., .02]]]).expand(2, 2, 3)
        hits = starts.clone()
        hits[:, 0, 2] = .0
        hits[:, 1, 2] = -.2
        direction = torch.tensor([0., 0., -1.]).expand_as(starts)
        env.scene.sensors[f"{side}_feet_ray_caster"] = SimpleNamespace(
            ray_starts=starts,
            ray_directions=direction,
            data=SimpleNamespace(ray_hits_w=hits, pos_w=torch.zeros(2, 3),
                                 quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 4)))
    monitor = adapter.FootEdgeMonitor(env)
    assert not monitor.cost().any()
    diagnostics = monitor.diagnostics()
    assert diagnostics["FootSupport/available"] == 1.
    assert diagnostics["FootSupport/left/support_fraction_when_loaded"] == .5
    assert diagnostics["FootSupport/right/support_fraction_when_loaded"] == .5
    assert diagnostics["FootSupport/right/loaded_contact_fraction"] == .5
    assert diagnostics["FootSupport/right/any_contact_fraction"] == 1.
    assert diagnostics["FootEdge/raw_cost"] == 0.


def test_metadata_records_complete_geometry_and_keeps_configuration_immutable(tmp_path):
    env = _monitor_env()
    monitor = env.foot_edge_monitor
    monitor.save_metadata(tmp_path)
    meta = json.loads((tmp_path / "foot_edge_contract.json").read_text())
    saved = np.load(tmp_path / "foot_edge_geometry.npz")
    assert meta["points_per_foot"] == 100 and meta["support_diagnostics_only"]
    assert not meta["physical_collision_geometry_changed"]
    assert meta["config"]["geometry"]["cylinder_radius"] == .05
    assert meta["effective_reward_weight"] == -4.
    assert meta["epsilon_provenance"]["configured"] == 1e-3
    np.testing.assert_array_equal(saved["local_points"], monitor.local_points.numpy())
    np.testing.assert_array_equal(saved["edges"], monitor.geometry.edges)
