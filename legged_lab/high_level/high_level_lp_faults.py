"""Explicit LP events with source-aligned history and persistent map recovery."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from legged_lab.selector.fault_kinds import FaultKind as K
from .lp_geometry import transform_lp
from .lp_map_faults import ElevationMapFaults
from legged_lab.selector.randomness import PerEnvironmentRNG


class HighLevelLpFaults:
    def __init__(self, num_envs, cfg, *, period=0.1, seed=43, device="cpu"):
        self.cfg, self.period, self.device = cfg, period, torch.device(device)
        self.num_envs, self.shape = num_envs, (11, 17)
        self.rng = PerEnvironmentRNG(num_envs, seed, self.device)
        for name in ("kind", "remaining", "duration", "frame", "delay", "person_start", "person_duration"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long, device=self.device))
        for name, tail in (("linear", (4,)), ("map_drift", (4,)), ("velocity", (4,)),
                           ("registration", (3,)), ("bias", ()), ("dropout", ()),
                           ("person_origin", (3,)), ("frozen", self.shape), ("frozen_support", self.shape),
                           ("frozen_stamp", ())):
            setattr(self, name, torch.zeros(num_envs, *tail, device=self.device))
        self.frozen_stamp = self.frozen_stamp.double()
        self.frozen_ready = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.block_mask = torch.zeros(num_envs, *self.shape, dtype=torch.bool, device=self.device)
        self.history = torch.zeros(num_envs, cfg.max_delay_frames + 1, *self.shape, device=self.device)
        self.support_history = torch.zeros_like(self.history)
        self.stamp_history = torch.zeros(num_envs, cfg.max_delay_frames + 1, device=self.device, dtype=torch.float64)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        map_cfg = SimpleNamespace(**vars(cfg), update_period_s=period)
        self.map = ElevationMapFaults(num_envs, self.shape, self.device, map_cfg, self.rng)

    def _uniform(self, ids, bounds, shape=()):
        return bounds[0] + self.rng.uniform(ids, shape) * (bounds[1] - bounds[0])

    def begin(self, ids, kinds, durations, pose):
        """事件开始只更新事件参数；不重置历史、漂移或地图残影。"""
        c, s = self.cfg, self.map.state
        self.kind[ids], self.remaining[ids], self.duration[ids] = kinds, durations, durations
        self.frozen_ready[ids] = False
        self.delay[ids] = self.rng.randint(1, c.max_delay_frames + 1, ids)
        self.bias[ids] = self._uniform(ids, c.lp_height_bias_m_range)
        self.dropout[ids] = self._uniform(ids, c.lp_dropout_fraction)
        self.registration[ids, :2] = self._uniform(ids, c.lp_translation_range_m, (2,))
        self.registration[ids, 2] = self._uniform(ids, c.lp_yaw_range_deg)
        angle = self.rng.uniform(ids) * (2 * math.pi)
        speed = self._uniform(ids, c.lp_drift_m_per_s)
        self.velocity[ids, :2] = torch.stack((angle.cos(), angle.sin()), -1) * speed[:, None]
        self.velocity[ids, 2] = self._uniform(ids, c.lp_drift_deg_per_s) * torch.where(
            self.rng.uniform(ids) < 0.5, -1., 1.)
        self.velocity[ids, 3] = self._uniform(ids, c.lp_map_height_velocity_m_per_s)
        fraction = self._uniform(ids, c.lp_block_fraction)
        h, w = self.shape
        rh = (fraction * h * h).sqrt().round().long().clamp(1, h)
        rw = (fraction * h * w / rh).ceil().long().clamp(1, w)
        top = (self.rng.uniform(ids) * (h - rh + 1)).long()
        left = (self.rng.uniform(ids) * (w - rw + 1)).long()
        yy = torch.arange(h, device=self.device)[None, :, None]
        xx = torch.arange(w, device=self.device)[None, None, :]
        self.block_mask[ids] = ((yy >= top[:, None, None]) & (yy < (top + rh)[:, None, None])
                                & (xx >= left[:, None, None]) & (xx < (left + rw)[:, None, None])
                                & (fraction > 0)[:, None, None])
        # Recenter only an empty map; accepted world cells retain their origin.
        empty = ids[~self.map.residual()[ids]]
        s["origin"][empty] = pose[empty, :3]
        s["initialized"][ids] = True
        for name, bounds in (("on_frames", c.burst_on_frames), ("off_frames", c.burst_off_frames),
                             ("correction_frames", c.lp_map_correction_frames)):
            s[name][ids] = self.rng.randint(bounds[0], bounds[1] + 1, ids)
        people = ids[(kinds == int(K.LP_PERSON)) | (kinds == int(K.LP_COMPOUND))]
        self.person_start[people], self.person_duration[people] = self.frame[people], self.duration[people]
        self.person_origin[people] = pose[people, :3]
        angle = pose[people, 3] + (self.rng.uniform(people) - .5) * math.pi
        s["direction"][people] = torch.stack((angle.cos(), angle.sin()), -1)
        s["radius"][people] = self._uniform(people, c.lp_person_radius_m)
        s["human_height"][people] = self._uniform(people, c.lp_person_height_above_ground_m)
        s["approach_distance"][people] = self._uniform(people, (1.2, 1.7))
        s["near_distance"][people] = self._uniform(people, (0.2, 0.7))
        # Confirmation counters describe the new trajectory; accepted ghosts survive.
        s["confirm"][people] = 0

    def _drift(self, ids, active, elapsed):
        c, dt = self.cfg, self.period
        linear_on = active & (self.kind == int(K.LP_DRIFT))
        map_on = active & ((self.kind == int(K.LP_MAP_DRIFT)) | (self.kind == int(K.LP_COMPOUND)))
        decay = math.exp(-dt / c.recovery_tau_s)
        self.linear[ids[~linear_on[ids]]] *= decay
        self.map_drift[ids[~map_on[ids]]] *= decay
        linear = ids[linear_on[ids] & (elapsed[ids] > 0)]
        self.linear[linear, :3] += self.velocity[linear, :3] * dt
        maps = ids[map_on[ids]]
        self.map_drift[maps] += self.velocity[maps] * dt
        self.map_drift[maps, :2] += self.rng.normal(maps, (2,)) * c.lp_map_random_walk_m
        correction = maps[(elapsed[maps] > 0) & (elapsed[maps] % self.map.state["correction_frames"][maps] == 0)]
        self.map_drift[correction] *= .35
        for name in ("linear", "map_drift"):
            values = getattr(self, name)
            values[ids, :2] = values[ids, :2].clamp(-c.lp_map_translation_limit_m, c.lp_map_translation_limit_m)
            values[ids, 2] = values[ids, 2].clamp(-c.lp_map_yaw_limit_deg, c.lp_map_yaw_limit_deg)
            values[ids, 3] = values[ids, 3].clamp(-c.lp_map_height_limit_m, c.lp_map_height_limit_m)
            small = ((values[ids][:, [0, 1, 3]].abs() < c.recovery_translation_epsilon_m).all(-1)
                     & (values[ids, 2].abs() < c.recovery_yaw_epsilon_deg))
            # Never suppress a small active increment; thresholds apply to recovery only.
            ongoing = linear_on if name == "linear" else map_on
            values[ids[small & ~ongoing[ids]]] = 0

    def apply(self, height, support, source_stamp, ids, pose):
        """Full-sized input/output arrays; advance only rows containing a new LP packet."""
        c = self.cfg
        active = self.remaining > 0
        elapsed = self.duration - self.remaining
        self._drift(ids, active, elapsed)
        out, valid, stamp = height.clone(), support.clone(), source_stamp.clone()
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        selected[ids] = True

        def rows(kind):
            return (selected & active & (self.kind == int(kind))).nonzero().flatten()

        dropped = rows(K.LP_RANDOM_DROPOUT)
        mask = self.rng.uniform(dropped, self.shape) < self.dropout[dropped, None, None]
        out[dropped] = torch.where(mask, c.lp_invalid_height, out[dropped])
        valid[dropped] = torch.where(mask, 0., valid[dropped])
        blocked = rows(K.LP_BLOCK_DROPOUT)
        out[blocked] = torch.where(self.block_mask[blocked], c.lp_invalid_height, out[blocked])
        valid[blocked] = torch.where(self.block_mask[blocked], 0., valid[blocked])
        biased = rows(K.LP_HEIGHT_BIAS)
        out[biased] = torch.where(valid[biased] > 0,
            (out[biased] + self.bias[biased, None, None]).clamp(*c.lp_height_clip), c.lp_invalid_height)
        registration = rows(K.LP_REGISTRATION)
        out[registration], valid[registration] = transform_lp(
            out[registration], self.registration[registration, :2], self.registration[registration, 2],
            valid_support=valid[registration], return_support=True)
        residual = self.linear + self.map_drift
        shifted = ids[residual[ids].abs().sum(-1) > 0]
        out[shifted], valid[shifted] = transform_lp(out[shifted], residual[shifted, :2], residual[shifted, 2],
            valid_support=valid[shifted], return_support=True)
        out[shifted] = torch.where(valid[shifted] > 0,
            (out[shifted] - residual[shifted, 3, None, None]).clamp(*c.lp_height_clip), c.lp_invalid_height)
        compound = rows(K.LP_COMPOUND)
        state = self.map.state
        burst = compound[(elapsed[compound] % (state["on_frames"][compound] + state["off_frames"][compound]))
                         < state["on_frames"][compound]]
        drop = self.rng.uniform(burst, self.shape) < c.compound_dropout_probability
        out[burst] = torch.where(drop, c.lp_invalid_height, out[burst])
        valid[burst] = torch.where(drop, 0., valid[burst])
        # Expired person events continue only as stored world cells. Frozen or
        # missing measurements must not count as fresh ground confirmations.
        people_on = active & ((self.kind == int(K.LP_PERSON)) | (self.kind == int(K.LP_COMPOUND)))
        people = ids[people_on[ids] | self.map.residual()[ids]]
        map_support = valid.clone()
        no_measurement = torch.cat((rows(K.FREEZE), rows(K.DELAY), rows(K.LP_COLLAPSE)))
        map_support[no_measurement] = 0
        self.map.recenter_external(people, pose)
        self.map.apply_people_external(out, valid, people, pose, self.frame - self.person_start,
            self.person_duration, self.frame, origin=self.person_origin, active=people_on,
            ground_support=map_support)
        freeze = rows(K.FREEZE)
        new = freeze[~self.frozen_ready[freeze]]
        self.frozen[new], self.frozen_support[new], self.frozen_stamp[new] = (
            out[new], valid[new], stamp[new])
        self.frozen_ready[new] = True
        out[freeze], valid[freeze], stamp[freeze] = (
            self.frozen[freeze], self.frozen_support[freeze], self.frozen_stamp[freeze])
        collapsed = rows(K.LP_COLLAPSE)
        out[collapsed], valid[collapsed] = c.lp_invalid_height, 0
        # Store each generated map once. A later delivery delay must retain
        # that frame's corruption, support and measurement time together.
        self.history[ids] = self.history[ids].roll(-1, dims=1)
        self.support_history[ids] = self.support_history[ids].roll(-1, dims=1)
        self.stamp_history[ids] = self.stamp_history[ids].roll(-1, dims=1)
        self.history[ids, -1], self.support_history[ids, -1] = out[ids], valid[ids]
        self.stamp_history[ids, -1] = stamp[ids]
        first = ids[~self.initialized[ids]]
        self.history[first] = out[first, None]
        self.support_history[first] = valid[first, None]
        self.stamp_history[first] = stamp[first, None]
        self.initialized[ids] = True
        delayed = rows(K.DELAY)
        index = c.max_delay_frames - self.delay[delayed]
        out[delayed], valid[delayed], stamp[delayed] = (
            self.history[delayed, index], self.support_history[delayed, index], self.stamp_history[delayed, index])
        self.remaining[ids] = (self.remaining[ids] - 1).clamp_min(0)
        self.frame[ids] += 1
        return out, valid, stamp

    def residual(self):
        return self.map.residual() | ((self.linear.abs() + self.map_drift.abs()).sum(-1) > 0)

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        self.map.reset(ids)
