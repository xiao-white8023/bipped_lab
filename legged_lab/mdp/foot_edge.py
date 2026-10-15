"""RewardManager adapter and diagnostics for the G1 virtual foot-volume penalty.

The environment owns one monitor, constructed after scene initialization. No
policy, optimizer, physical collision shape, or training update is created here.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from legged_lab.perception.foot_edge import (
    edge_cost,
    foot_support_diagnostics,
    make_volume_points,
    point_kinematics,
)


def foot_edge_penalty(env) -> torch.Tensor:
    """Return the positive raw cost; RewardManager applies weight and timestep."""
    return env.foot_edge_monitor.cost()


class FootEdgeMonitor:
    """Cache shared foot-volume queries once per physics-step counter.

    ``geometry=None`` is valid when the feature is disabled or its reward weight
    is zero. The environment validates that a nonzero active reward has a query.
    Support diagnostics remain independent of that switch.
    """

    BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.foot_edge
        self.cfg.validate()
        self.device = env.device
        self.geometry = getattr(env.scene.terrain, "foot_edge_geometry", None)
        self.robot = env.scene["robot"]
        body_ids, names = self.robot.find_bodies(list(self.BODY_NAMES), preserve_order=True)
        if tuple(names) != self.BODY_NAMES:
            raise ValueError(f"Expected left/right ankle-roll links, received {names}.")
        self.body_ids = list(body_ids)
        self.local_points = make_volume_points(self.cfg, self.device)
        shape = (env.num_envs, 2, len(self.local_points))
        self.points_world = torch.zeros(*shape, 3, device=self.device)
        self.point_velocities_world = torch.zeros_like(self.points_world)
        self.penetration_depths = torch.zeros(shape, device=self.device)
        self.penetration_offsets = torch.zeros_like(self.points_world)
        self._cost = torch.zeros(env.num_envs, device=self.device)
        self._cached_step = None
        self._support_cache = None
        self._debug_vis = False
        self._point_markers = None
        self._edge_markers = None
        self._edges_world = torch.tensor(
            self.geometry.edges if self.geometry is not None else np.empty((0, 2, 3)),
            device=self.device, dtype=torch.float32)
        self._support_body_ids = None
        if self.cfg.support_diagnostics:
            sensor = env.scene.sensors.get("contact_sensor")
            if sensor is not None:
                ids, contact_names = sensor.find_bodies(list(self.BODY_NAMES), preserve_order=True)
                if tuple(contact_names) != self.BODY_NAMES:
                    raise ValueError("Foot support contact-sensor order does not match the two foot links.")
                self._support_body_ids = list(ids)
        self.set_debug_vis(self.cfg.debug_vis)

    @torch.no_grad()
    def update(self):
        step = getattr(self.env, "sim_step_counter", 0)
        if step == self._cached_step:
            return
        data = self.robot.data
        self.points_world, self.point_velocities_world = point_kinematics(
            self.local_points,
            data.body_link_pos_w[:, self.body_ids],
            data.body_link_quat_w[:, self.body_ids],
            data.body_com_pos_w[:, self.body_ids],
            data.body_com_lin_vel_w[:, self.body_ids],
            data.body_com_ang_vel_w[:, self.body_ids],
        )
        if (self.cfg.enabled or self._debug_vis) and self.geometry is not None:
            self.penetration_offsets = self.geometry.penetration_offsets(self.points_world)
            self.penetration_depths = torch.linalg.vector_norm(self.penetration_offsets, dim=-1)
            if self.cfg.enabled:
                self._cost = edge_cost(self.penetration_depths, self.point_velocities_world,
                                       self.cfg.velocity_epsilon, self.cfg.tolerance)
            else:
                self._cost.zero_()
        else:
            self.penetration_offsets.zero_()
            self.penetration_depths.zero_()
            self._cost.zero_()
        self._cached_step = step
        self._support_cache = None
        if self._debug_vis:
            self._update_debug_vis()

    def cost(self) -> torch.Tensor:
        self.update()
        return self._cost

    @torch.no_grad()
    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Flat scalar logs; all support entries are diagnostic quantities."""
        self.update()
        depth = self.penetration_depths
        speed = torch.linalg.vector_norm(self.point_velocities_world, dim=-1)
        penetrating = depth > self.cfg.tolerance
        penetrating_count = penetrating.sum().clamp_min(1)
        effective_weight = float(getattr(self.env, "_foot_edge_reward_weight", 0.0))
        step_dt = float(getattr(self.env, "step_dt", 0.0))
        result = {
            "FootEdge/raw_cost": self._cost.mean(),
            "FootEdge/weighted_reward": self._cost.mean() * effective_weight * step_dt,
            "FootEdge/penetrating_point_fraction": penetrating.float().mean(),
            "FootEdge/penetration_depth_mean_m": depth.mean(),
            "FootEdge/penetration_depth_max_m": depth.max(),
            "FootEdge/penetration_depth_when_penetrating_m": (depth * penetrating).sum() / penetrating_count,
            "FootEdge/point_speed_mean_m_s": speed.mean(),
            "FootEdge/point_speed_when_penetrating_m_s": (speed * penetrating).sum() / penetrating_count,
            "FootEdge/finite_point_fraction": (
                torch.isfinite(self.points_world).all(dim=-1)
                & torch.isfinite(self.point_velocities_world).all(dim=-1)).float().mean(),
        }
        for index, side in enumerate(("left", "right")):
            selected = penetrating[:, index]
            result[f"FootEdge/{side}/penetration_depth_max_m"] = depth[:, index].max()
            result[f"FootEdge/{side}/penetrating_point_fraction"] = selected.float().mean()
            result[f"FootEdge/{side}/point_speed_when_penetrating_m_s"] = (
                speed[:, index] * selected).sum() / selected.sum().clamp_min(1)
            result[f"FootEdge/{side}/raw_cost"] = edge_cost(
                depth[:, index:index + 1], self.point_velocities_world[:, index:index + 1],
                self.cfg.velocity_epsilon, self.cfg.tolerance).mean()
        if self.cfg.support_diagnostics:
            support = self._support_diagnostics()
            result["FootSupport/available"] = depth.new_tensor(float(support is not None))
            if support is not None:
                loaded = support["loaded_contact"].float()
                for index, side in enumerate(("left", "right")):
                    count = loaded[:, index].sum().clamp_min(1.0)
                    result[f"FootSupport/{side}/valid_ray_fraction"] = support["valid_fraction"][:, index].mean()
                    result[f"FootSupport/{side}/loaded_contact_fraction"] = loaded[:, index].mean()
                    result[f"FootSupport/{side}/any_contact_fraction"] = support["any_contact"][:, index].float().mean()
                    result[f"FootSupport/{side}/support_fraction_when_loaded"] = (
                        support["support_fraction"][:, index] * loaded[:, index]).sum() / count
                    result[f"FootSupport/{side}/unsupported_fraction_when_loaded"] = (
                        support["unsupported_fraction_in_contact"][:, index]).sum() / count
        return result

    def _support_diagnostics(self):
        if self._support_cache is not None:
            return self._support_cache
        if self._support_body_ids is None:
            return None
        sensors = self.env.scene.sensors
        rays = [sensors.get(f"{side}_feet_ray_caster") for side in ("left", "right")]
        if any(sensor is None for sensor in rays):
            return None
        starts, hits, directions = [], [], []
        for sensor in rays:
            data = sensor.data  # Flush the lazy sensor before reading actual ray origins.
            hits.append(data.ray_hits_w)
            if hasattr(sensor, "ray_starts_w"):
                starts.append(sensor.ray_starts_w)
            else:
                starts.append(self._rotate(data.quat_w, sensor.ray_starts) + data.pos_w[:, None])
            directions.append(self._rotate(data.quat_w, sensor.ray_directions))
        forces = sensors["contact_sensor"].data.net_forces_w[:, self._support_body_ids]
        self._support_cache = foot_support_diagnostics(
            torch.stack(starts, dim=1), torch.stack(hits, dim=1),
            torch.stack(directions, dim=1), forces,
            self.cfg.support_distance_threshold, self.cfg.support_contact_threshold)
        return self._support_cache

    @staticmethod
    def _rotate(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        q_vec = quaternion[:, None, 1:]
        uv = torch.linalg.cross(q_vec, vectors, dim=-1)
        return vectors + 2.0 * (quaternion[:, None, :1] * uv + torch.linalg.cross(q_vec, uv, dim=-1))

    def reset(self, env_ids=None):
        """Invalidate cached current state; this monitor has no episode history."""
        if env_ids is not None and len(env_ids) == 0:
            return
        self._cached_step = None
        self._support_cache = None

    def metadata(self) -> dict:
        return {
            "version": 1,
            "mechanism": "virtual foot points penetrating finite cylinders around sharp terrain edges",
            "raw_cost": "sum(1[depth>tolerance] * depth * (point_speed_norm + velocity_epsilon))",
            "coefficient_and_timestep": "applied only by RewardManager",
            "contact_gate": False,
            "point_reduction": "sum",
            "support_diagnostics_only": True,
            "physical_collision_geometry_changed": False,
            "point_frame": "ankle_roll_link",
            "volume_approximation": "fixed regular lower-foot grid, not exact STL interior or physical collision",
            "quaternion_order": "wxyz",
            "point_velocity": "v_com + omega_world cross (point_world - com_world)",
            "body_names": list(self.BODY_NAMES),
            "body_ids": self.body_ids,
            "points_per_foot": len(self.local_points),
            "effective_reward_weight": float(getattr(self.env, "_foot_edge_reward_weight", 0.0)),
            "control_timestep_s": float(getattr(self.env, "step_dt", 0.0)),
            "epsilon_provenance": {
                "paper": 1e-3,
                "reference_implementation": 1e-6,
                "configured": self.cfg.velocity_epsilon,
                "choice": "paper value by default; the discrepancy is recorded rather than silently mixed",
            },
            "support_proxy": {
                "loaded_contact": "world Fz > support_contact_threshold; not abs(Fz) or force norm",
                "distance": "from each ray's actual origin along its actual direction",
                "missing_hits": "unsupported; retained in the ray-count denominator",
                "contact_patch_measurement": False,
            },
            "config": asdict(self.cfg),
            "geometry": self.geometry.metadata() if self.geometry is not None else None,
            "implementation_sha256": {
                "reward_adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "foot_math": hashlib.sha256(
                    (Path(__file__).parents[1] / "perception/foot_edge.py").read_bytes()).hexdigest(),
            },
            "debug_overlay": "first environment's two feet; edge cylinders within 2m, at most 2048",
        }

    def save_metadata(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "foot_edge_contract.json").write_text(
            json.dumps(self.metadata(), indent=2, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(directory / "foot_edge_geometry.npz",
                            edges=self.geometry.edges if self.geometry is not None else np.empty((0, 2, 3)),
                            local_points=self.local_points.cpu().numpy())

    def set_debug_vis(self, debug_vis: bool):
        self._debug_vis = bool(debug_vis)
        if self._debug_vis and self._point_markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            root = "/Visuals/G1DepthFootEdge"
            self._point_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=f"{root}/Points", markers={
                    "clear": sim_utils.SphereCfg(radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.9, 0.15))),
                    "penetrating": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.05, 0.02))),
                }))
            self._edge_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=f"{root}/Cylinders", markers={
                    "edge": sim_utils.CylinderCfg(radius=1.0, height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.05), opacity=0.25)),
                }))
        for marker in (self._point_markers, self._edge_markers):
            if marker is not None:
                marker.set_visibility(self._debug_vis)

    def _update_debug_vis(self):
        self._point_markers.visualize(
            translations=self.points_world[0].reshape(-1, 3),
            marker_indices=(self.penetration_depths[0].reshape(-1) > self.cfg.tolerance).long())
        if not len(self._edges_world):
            self._edge_markers.set_visibility(False)
            return
        edges = self._edges_world
        center = self.robot.data.root_pos_w[0]
        # 长台阶边可能经过脚旁，而其中点在远处；按点到线段距离筛选显示。
        segment = edges[:, 1] - edges[:, 0]
        fraction = ((center - edges[:, 0]) * segment).sum(dim=-1)
        fraction = (fraction / segment.square().sum(dim=-1).clamp_min(1e-12)).clamp(0, 1)
        closest = edges[:, 0] + fraction[:, None] * segment
        selected = ((closest - center).norm(dim=-1) < 2.0).nonzero().flatten()[:2048]
        if not len(selected):
            self._edge_markers.set_visibility(False)
            return
        self._edge_markers.set_visibility(True)
        edge = edges[selected]
        vector = edge[:, 1] - edge[:, 0]
        length = vector.norm(dim=-1)
        direction = vector / length[:, None].clamp_min(1e-8)
        # Unit quaternion taking local +Z to the cylinder direction.
        orientation = torch.stack((1.0 + direction[:, 2], -direction[:, 1],
                                   direction[:, 0], torch.zeros_like(length)), dim=-1)
        opposite = direction[:, 2] < -1.0 + 1e-6
        orientation[opposite] = orientation.new_tensor([0.0, 1.0, 0.0, 0.0])
        orientation /= orientation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.ones_like(vector)
        scale[:, :2] = self.cfg.geometry.cylinder_radius
        scale[:, 2] = length
        self._edge_markers.visualize(translations=edge.mean(dim=1), orientations=orientation, scales=scale)
