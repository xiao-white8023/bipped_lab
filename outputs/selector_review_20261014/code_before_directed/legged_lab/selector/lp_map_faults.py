"""Collection-only elevation-map corruption; world pose never enters features.

This is an input-level map surrogate, not a human collision simulator. Accepted
high returns stay valid, use the policy height sign, and persist until observed
ground confirms replacement or the cell leaves the retention radius. Source
packet timestamps stay fresh: individual old cells do not age the whole map.
"""

from __future__ import annotations

import math
import torch


MAP_AUDIT_ORDER = (
    "drift_x_m", "drift_y_m", "drift_yaw_deg", "drift_height_m",
    "person_x_world_m", "person_y_world_m", "person_radius_m",
    "person_phase", "ghost_cells", "oldest_ghost_age_s", "burst_on",
    "lp_support_ratio", "lp_saturation_ratio", "lp_observed_effect",
)
NOISE_AUDIT_ORDER = MAP_AUDIT_ORDER + (
    "vp_burst_on_frames", "vp_burst_off_frames", "vp_burst_on", "vp_observed_effect",
)


class ElevationMapFaults:
    def __init__(self, num_envs, shape, device, cfg, rng):
        self.cfg, self.rng, self.device = cfg, rng, torch.device(device)
        self.shape, self.num_envs = tuple(shape), num_envs
        self.side = 33  # 3.2 m persistent window, as in the deployed map.
        self.state = {}
        for name, tail, dtype in (
            ("origin", (3,), torch.float32), ("direction", (2,), torch.float32),
            ("person_pos", (2,), torch.float32), ("drift", (4,), torch.float32),
            ("radius", (), torch.float32), ("human_height", (), torch.float32),
            ("approach_distance", (), torch.float32), ("near_distance", (), torch.float32),
            ("ghost_height", (self.side**2,), torch.float32),
            ("ghost_valid", (self.side**2,), torch.bool),
            ("confirm", (self.side**2,), torch.int64),
            ("clean_confirm", (self.side**2,), torch.int64),
            ("last_seen", (self.side**2,), torch.int64),
            ("initialized", (), torch.bool), ("observed_effect", (), torch.bool),
            ("phase", (), torch.int64), ("burst_on", (), torch.bool),
            ("on_frames", (), torch.int64), ("off_frames", (), torch.int64),
            ("correction_frames", (), torch.int64),
            ("support_ratio", (), torch.float32), ("saturation_ratio", (), torch.float32),
        ):
            self.state[name] = torch.zeros((num_envs, *tail), device=device, dtype=dtype)
        axis = (torch.arange(self.side, device=device) - self.side // 2) * cfg.lp_resolution_m
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.cells = torch.stack((x.flatten(), y.flatten()), -1)
        h, w = shape
        y, x = torch.meshgrid(
            (torch.arange(h, device=device) - (h-1)/2) * cfg.lp_resolution_m,
            (torch.arange(w, device=device) - (w-1)/2) * cfg.lp_resolution_m,
            indexing="ij",
        )
        self.query = torch.stack((x.flatten(), y.flatten()), -1)

    def snapshot(self):
        return {name: value.clone() for name, value in self.state.items()}

    def restore(self, values):
        for name, value in values.items():
            self.state[name].copy_(value)

    def reset(self, ids):
        for value in self.state.values():
            value[ids] = 0

    def schedule(self, ids, strength):
        self.reset(ids)
        s = self.state
        for name, bounds in (("radius", self.cfg.lp_person_radius_m),
                             ("human_height", self.cfg.lp_person_height_above_ground_m)):
            s[name][ids] = bounds[0] + strength * (bounds[1] - bounds[0])
        s["approach_distance"][ids] = 1.2 + self.rng.uniform(ids) * 0.5
        s["near_distance"][ids] = 0.2 + self.rng.uniform(ids) * 0.5
        for name, bounds in (("on_frames", self.cfg.burst_on_frames),
                             ("off_frames", self.cfg.burst_off_frames),
                             ("correction_frames", self.cfg.lp_map_correction_frames)):
            s[name][ids] = self.rng.randint(bounds[0], bounds[1]+1, ids)

    def residual(self):
        return self.state["ghost_valid"].any(-1)

    def apply(self, lp, support, ids, *, pose, elapsed, duration, frame,
              drift_ids_mask, person_ids_mask, compound_ids_mask,
              drift_velocity, yaw_velocity, height_velocity, warp):
        if not len(ids):
            return lp, support
        if pose is None or pose.shape != (self.num_envs, 4) or not torch.isfinite(pose).all():
            raise ValueError("map faults require finite torso world [x,y,z,yaw] for every row")
        s, cfg = self.state, self.cfg
        new = ids[~s["initialized"][ids]]
        if len(new):
            s["origin"][new] = pose[new, :3]
            # A side/front approach is fixed in world coordinates at event onset.
            angle = pose[new, 3] + (self.rng.uniform(new) - .5) * math.pi
            s["direction"][new] = torch.stack((angle.cos(), angle.sin()), -1)
            s["initialized"][new] = True
        e = elapsed.long()
        cycle = (s["on_frames"] + s["off_frames"]).clamp_min(1)
        s["burst_on"][ids] = (e[ids] % cycle[ids]) < s["on_frames"][ids]
        drifting = ids[drift_ids_mask[ids] | compound_ids_mask[ids]]
        if len(drifting):
            dt = cfg.update_period_s
            s["drift"][drifting, :2] += (
                drift_velocity[drifting] * dt
                + self.rng.normal(drifting, (2,)) * cfg.lp_map_random_walk_m
            )
            s["drift"][drifting, 2] += yaw_velocity[drifting] * dt
            s["drift"][drifting, 3] += height_velocity[drifting] * dt
            correction = (e[drifting] > 0) & (e[drifting] % s["correction_frames"][drifting] == 0)
            s["drift"][drifting[correction]] *= .35
            delta = s["drift"][drifting]
            delta[:, :2] = delta[:, :2].clamp(-cfg.lp_map_translation_limit_m, cfg.lp_map_translation_limit_m)
            delta[:, 2] = delta[:, 2].clamp(-cfg.lp_map_yaw_limit_deg, cfg.lp_map_yaw_limit_deg)
            delta[:, 3] = delta[:, 3].clamp(-cfg.lp_map_height_limit_m, cfg.lp_map_height_limit_m)
            s["drift"][drifting] = delta
            lp[drifting], support[drifting] = warp(
                lp[drifting], delta[:, :2], delta[:, 2],
                valid_support=support[drifting], return_support=True,
            )
            # A positive world-height bias reduces the policy height.
            lp[drifting] = torch.where(support[drifting] > 0,
                (lp[drifting] - delta[:, 3, None, None]).clamp(*cfg.lp_height_clip),
                cfg.lp_invalid_height)

        people = ids[person_ids_mask[ids] | compound_ids_mask[ids]]
        if len(people):
            self._people(lp, support, people, pose, e, duration, frame)
        # Compound on/off bursts belong to this parent event, including gaps.
        blocked = ids[compound_ids_mask[ids] & s["burst_on"][ids]]
        if len(blocked):
            drop = self.rng.uniform(blocked, self.shape) < .20
            lp[blocked] = torch.where(drop, cfg.lp_invalid_height, lp[blocked])
            support[blocked] = torch.where(drop, 0., support[blocked])
        s["support_ratio"][ids] = support[ids].flatten(1).mean(-1)
        clipped = ((lp[ids] <= cfg.lp_height_clip[0]) | (lp[ids] >= cfg.lp_height_clip[1])) & (support[ids] > 0)
        s["saturation_ratio"][ids] = clipped.float().flatten(1).mean(-1)
        return lp, support

    def _people(self, lp, support, ids, pose, elapsed, duration, frame):
        s, cfg = self.state, self.cfg
        cells = s["origin"][ids, None, :2] + self.cells[None]
        xy = self.query[None].expand(len(ids), -1, -1)
        yaw = pose[ids, 3, None]
        world = torch.stack((xy[..., 0]*yaw.cos() - xy[..., 1]*yaw.sin(),
                             xy[..., 0]*yaw.sin() + xy[..., 1]*yaw.cos()), -1)
        world += pose[ids, None, :2]
        index_xy = ((world - s["origin"][ids, None, :2]) / cfg.lp_resolution_m
                    + self.side//2).round().long()
        inside = ((index_xy >= 0) & (index_xy < self.side)).all(-1)
        index = (index_xy[..., 1].clamp(0, self.side-1)*self.side
                 + index_xy[..., 0].clamp(0, self.side-1))
        # Observation footprint, once per cell per scan; no multiple counts from
        # rotated query pixels hitting the same map cell.
        visible = torch.zeros(len(ids), self.side**2, device=self.device)
        visible.scatter_add_(1, index, inside.float())
        visible = visible > 0
        main = (duration[ids].float()*cfg.lp_person_main_fraction).clamp_min(6)
        progress = elapsed[ids].float()/main
        inbound = progress < .35
        near = (progress >= .35) & (progress < .65)
        departing = (progress >= .65) & (progress < 1)
        distance = torch.where(inbound,
            torch.lerp(s["approach_distance"][ids], s["near_distance"][ids], (progress/.35).clamp(0,1)),
            torch.where(near, s["near_distance"][ids],
                torch.lerp(s["near_distance"][ids], s["approach_distance"][ids]+1., ((progress-.65)/.35).clamp(0,1))))
        person = s["origin"][ids, :2] + s["direction"][ids]*distance[:, None]
        s["person_pos"][ids] = person
        s["phase"][ids] = torch.where(inbound, 0, torch.where(near, 1, torch.where(departing, 2, 3)))
        # Highest accepted return in a cell: represent body/leg surfaces, not
        # unbounded head returns outside the deployed capture volume.
        human_z = s["origin"][ids, 2] - .75 + s["human_height"][ids]
        in_capture = (human_z - pose[ids, 2] >= -1.5) & (human_z - pose[ids, 2] <= .5)
        footprint = ((cells - person[:, None]).square().sum(-1) <= s["radius"][ids, None].square())
        incoming = footprint & visible & (progress < 1)[:, None] & in_capture[:, None]
        confirm = torch.where(incoming, s["confirm"][ids]+1, 0)
        s["confirm"][ids] = confirm
        accepted = incoming & (confirm >= cfg.lp_map_confirm_frames)
        heights = s["ghost_height"][ids]
        present = s["ghost_valid"][ids]
        heights = torch.where(accepted, torch.where(present, torch.maximum(heights, human_z[:, None]), human_z[:, None]), heights)
        present |= accepted
        s["last_seen"][ids] = torch.where(accepted, frame[ids, None], s["last_seen"][ids])
        # Only valid new ground observations can clear ghosts; age alone cannot.
        ground_observed = torch.zeros_like(visible, dtype=torch.float32)
        ground_observed.scatter_add_(1, index, (inside & (support[ids].flatten(1) > 0)).float())
        clean = (ground_observed > 0) & ~incoming
        replacement = torch.where(clean, s["clean_confirm"][ids]+1, 0)
        s["clean_confirm"][ids] = replacement
        retained = (cells - pose[ids, None, :2]).square().sum(-1) <= cfg.lp_map_retention_radius_m**2
        present &= retained & (replacement < cfg.lp_map_confirm_frames)
        s["ghost_height"][ids], s["ghost_valid"][ids] = heights, present
        ghost = present.gather(1, index) & inside
        world_z = heights.gather(1, index)
        ghost &= ((world_z - pose[ids, 2, None]) >= -1.5) & ((world_z - pose[ids, 2, None]) <= .5)
        policy = (pose[ids, 2, None] - world_z - .5).clamp(*cfg.lp_height_clip)
        current = lp[ids].flatten(1)
        # Max world elevation == min policy scalar. Unknown is not a height.
        ghost_height = torch.where(support[ids].flatten(1) > 0, torch.minimum(current, policy), policy)
        lp[ids] = torch.where(ghost, ghost_height, current).reshape_as(lp[ids])
        support[ids] = torch.where(ghost, 1., support[ids].flatten(1)).reshape_as(support[ids])

    def audit(self, frame):
        s = self.state
        age = ((frame[:, None]-s["last_seen"]).clamp_min(0)*self.cfg.update_period_s)
        oldest = torch.where(s["ghost_valid"], age, 0.).max(-1).values
        return torch.cat((s["drift"], s["person_pos"], s["radius"][:, None],
            s["phase"].float()[:, None], s["ghost_valid"].sum(-1, keepdim=True).float(),
            oldest[:, None], s["burst_on"].float()[:, None],
            s["support_ratio"][:, None], s["saturation_ratio"][:, None],
            s["observed_effect"].float()[:, None]), -1)
