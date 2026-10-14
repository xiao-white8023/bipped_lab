"""Capture-once depth faults for directed supplements.

Each captured source frame is corrupted in metres exactly once. Delivery faults
select stored frames; the two delivered history frames are never re-corrupted.
This is a policy-input surrogate, not a physical camera renderer.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .fault_kinds import FaultKind as K
from .directed_audit import VP_AUDIT_ORDER


class DepthFrameFaults:
    def __init__(self, num_envs, shape, device, cfg, rng):
        if len(shape) != 3 or shape[0] != 2:
            raise ValueError("capture-once VP faults require two depth history frames")
        self.cfg, self.rng, self.device = cfg, rng, torch.device(device)
        self.n, self.h, self.w = num_envs, shape[-2], shape[-1]
        self.capacity = cfg.max_delay_frames + 1
        self.state = {}
        for name, tail, dtype in (
            ("frames", (self.capacity, self.h, self.w), torch.float32),
            ("supports", (self.capacity, self.h, self.w), torch.float32),
            ("times", (self.capacity,), torch.float64),
            ("history", (2, self.h, self.w), torch.float32),
            ("history_support", (2, self.h, self.w), torch.float32),
            ("history_times", (2,), torch.float64),
            ("cursor", (), torch.int64), ("captures", (), torch.int64),
            ("initialized", (), torch.bool), ("delivered", (), torch.bool),
            ("last_capture", (), torch.float64), ("last_source", (), torch.float64),
            ("last_packet", (), torch.float64), ("hold_left", (), torch.int64),
            ("freeze_initialized", (), torch.bool),
            ("frozen", (self.h, self.w), torch.float32),
            ("frozen_support", (self.h, self.w), torch.float32),
            ("frozen_time", (), torch.float64), ("frozen_packet", (), torch.float64),
            ("field", (3, 6), torch.float32), ("centers", (3, 2), torch.float32),
            ("mask_direction", (2,), torch.float32), ("bias_m", (), torch.float32),
            ("scale_error", (), torch.float32), ("edge_mix", (), torch.float32),
            ("shift", (2,), torch.float32), ("rotation_deg", (), torch.float32),
            ("mask_fraction", (), torch.float32), ("mask_speed", (), torch.float32),
            ("return_mode", (), torch.int64), ("alpha", (), torch.float32),
            ("filter_output", (self.h, self.w), torch.float32),
            ("filter_support", (self.h, self.w), torch.float32),
            ("filter_error", (self.h, self.w), torch.float32),
            ("filter_initialized", (), torch.bool),
            ("previous_pose", (4,), torch.float32), ("pose_initialized", (), torch.bool),
            ("actual_delay", (), torch.float32), ("holding", (), torch.bool),
            ("effect", (), torch.bool), ("support_ratio", (), torch.float32),
            ("saturation_ratio", (), torch.float32), ("new_capture", (), torch.bool),
            ("history_append", (), torch.bool), ("source_age", (), torch.float32),
            ("changed_fraction", (), torch.float32),
        ):
            self.state[name] = torch.zeros((num_envs, *tail), device=device, dtype=dtype)
        y, x = torch.meshgrid(torch.linspace(0, 1, self.h, device=device),
                              torch.linspace(0, 1, self.w, device=device), indexing="ij")
        self.xy = torch.stack((x, y), -1)

    def snapshot(self):
        return {name: value.clone() for name, value in self.state.items()}

    def restore(self, values):
        if set(values) != set(self.state):
            raise ValueError("VP fault snapshot schema changed")
        for name, value in values.items():
            self.state[name].copy_(value)

    def reset(self, ids):
        for value in self.state.values():
            value[ids] = 0
        self.state["last_source"][ids] = -torch.inf
        self.state["last_packet"][ids] = -torch.inf
        self.state["source_age"][ids] = -1

    def schedule(self, ids, strength):
        # A new parent event changes parameters, not previously captured images.
        s = self.state
        for name in ("hold_left", "freeze_initialized", "filter_initialized", "effect"):
            s[name][ids] = 0
        s["filter_error"][ids] = 0
        sign = torch.where(self.rng.uniform(ids) < .5, -1., 1.)
        for name, bounds in (
            ("bias_m", self.cfg.vp_local_bias_m), ("scale_error", self.cfg.vp_local_scale_error),
            ("edge_mix", self.cfg.vp_edge_mix), ("rotation_deg", self.cfg.vp_rotation_deg),
            ("mask_fraction", self.cfg.vp_occlusion_fraction),
            ("mask_speed", self.cfg.vp_mask_speed_px_per_s), ("alpha", self.cfg.vp_filter_alpha),
        ):
            s[name][ids] = bounds[0] + strength * (bounds[1]-bounds[0])
        s["bias_m"][ids] *= sign
        s["rotation_deg"][ids] *= sign
        low, high = self.cfg.vp_subpixel_shift_px
        s["shift"][ids] = (low + strength[:, None]*(high-low)) * torch.where(
            self.rng.uniform(ids, (2,)) < .5, -1., 1.)
        s["field"][ids] = self.rng.normal(ids, (3, 6))
        s["centers"][ids] = .15 + .7*self.rng.uniform(ids, (3, 2))
        angle = 2*math.pi*self.rng.uniform(ids)
        s["mask_direction"][ids] = torch.stack((angle.cos(), angle.sin()), -1)
        s["return_mode"][ids] = self.rng.randint(0, 2, ids)

    def residual(self):
        return self.state["filter_error"].abs().flatten(1).amax(-1) > self.cfg.vp_residual_epsilon_m

    def _field(self, ids):
        field = F.interpolate(self.state["field"][ids, None], size=(self.h, self.w),
                              mode="bilinear", align_corners=True)[:, 0]
        return field / field.abs().flatten(1).amax(-1).clamp_min(1.)[:, None, None]

    def _mask(self, ids, elapsed, duration, *, moving=False, near=None):
        s = self.state
        seconds = elapsed.float()*self.cfg.update_period_s
        centers = s["centers"][ids].clone()
        drift = s["mask_direction"][ids] * s["mask_speed"][ids, None]
        drift = drift * seconds[:, None] / torch.tensor([self.w, self.h], device=self.device)
        centers += drift[:, None]
        if moving:
            progress = (elapsed.float()/duration.clamp_min(1)).clamp(0, 1)
            centers[..., 0] = -.35 + 1.7*progress[:, None] + .2*(centers[..., 0]-.5)
        delta = self.xy[None, None] - centers[:, :, None, None]
        radius = (s["mask_fraction"][ids]/(3*math.pi)).sqrt().clamp_min(.01)
        distance = ((delta[..., 0]/(1.4*radius[:, None, None, None])).square()
                    + (delta[..., 1]/(.75*radius[:, None, None, None])).square()).amin(1)
        distance += .20*self._field(ids)
        if near is not None:
            distance -= .35*near
        # Smooth borders represent fractional raw support after resampling.
        mask = ((1.15-distance)/.3).clamp(0, 1)
        return mask * ((elapsed.float()+1)/3).clamp(0, 1)[:, None, None]

    def _warp(self, data, valid, shift, angle, field=None):
        n = len(data)
        radians = torch.deg2rad(angle)
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 1, 1] = radians.cos(), radians.cos()
        theta[:, 0, 1] = -radians.sin()*(self.h-1)/max(1, self.w-1)
        theta[:, 1, 0] = radians.sin()*(self.w-1)/max(1, self.h-1)
        theta[:, 0, 2] = -2*shift[:, 0]/max(1, self.w-1)
        theta[:, 1, 2] = -2*shift[:, 1]/max(1, self.h-1)
        grid = F.affine_grid(theta, (n, 2, self.h, self.w), align_corners=True)
        if field is not None:
            grid[..., 0] += .5*field/max(1, self.w-1)
            grid[..., 1] += .5*field/max(1, self.h-1)
        sampled = F.grid_sample(torch.stack((data*valid, valid), 1), grid,
                                align_corners=True, mode="bilinear", padding_mode="zeros")
        support = sampled[:, 1].clamp(0, 1)
        depth = torch.where(support > 1e-6, sampled[:, 0]/support.clamp_min(1e-6),
                            self.cfg.vp_depth_max_m)
        return depth, support

    def _corrupt(self, ids, raw, support, active, plan, pose):
        s, cfg = self.state, self.cfg
        depth, valid = raw.clone(), support.clone()
        kinds = plan.vp_kind[ids]
        elapsed = (plan.step_count[ids]-plan.vp_start_frame[ids]).clamp_min(0)
        duration = plan.vp_duration_frames[ids]
        on = elapsed % (plan.vp_burst_on_frames[ids]+plan.vp_burst_off_frames[ids]).clamp_min(1) < plan.vp_burst_on_frames[ids]

        def rows(kind):
            return (active & (kinds == int(kind))).nonzero().flatten()

        evolving = active & ((kinds == int(K.VP_SPATIAL_BIAS)) | (kinds == int(K.VP_COMPOUND)))
        evolving_ids = ids[evolving]
        s["field"][evolving_ids] = .98*s["field"][evolving_ids] + .04*self.rng.normal(evolving_ids, (3, 6))
        field = self._field(ids)
        chosen = rows(K.VP_SPATIAL_BIAS)
        ramp = (.3 + .7*(elapsed.float()/duration.clamp_min(1)).clamp(0, 1))
        depth[chosen] = depth[chosen]*(1+s["scale_error"][ids[chosen], None, None]*field[chosen]) + s["bias_m"][ids[chosen], None, None]*field[chosen]*ramp[chosen, None, None]

        chosen = rows(K.VP_EDGE_MISMATCH)
        if len(chosen):
            d, v = depth[chosen], valid[chosen]
            low = -F.max_pool2d(-torch.where(v > .5, d, float("inf"))[:, None], 3, 1, 1)[:, 0]
            high = F.max_pool2d(torch.where(v > .5, d, -float("inf"))[:, None], 3, 1, 1)[:, 0]
            edges = (high-low > .06) & torch.isfinite(low) & torch.isfinite(high) & (v > 0)
            wrong = torch.where(s["bias_m"][ids[chosen], None, None] > 0, high, low)
            weight = s["edge_mix"][ids[chosen], None, None]*edges
            depth[chosen] = torch.where(edges, d*(1-weight)+wrong*weight, d)

        chosen = rows(K.VP_MOTION_WARP)
        if len(chosen):
            motion = torch.zeros(len(chosen), device=self.device)
            if pose is not None:
                previous = s["previous_pose"][ids[chosen]]
                delta = pose[ids[chosen]]-previous
                yaw = torch.atan2(delta[:, 3].sin(), delta[:, 3].cos()).abs()
                motion = ((delta[:, :2].norm(dim=-1)+.2*yaw)/cfg.update_period_s).clamp(0, 2)
                motion *= s["pose_initialized"][ids[chosen]]
            gain = (.5+motion) * (.8+.2*torch.sin(elapsed[chosen].float()*.2))
            depth[chosen], valid[chosen] = self._warp(depth[chosen], valid[chosen],
                s["shift"][ids[chosen]]*gain[:, None], s["rotation_deg"][ids[chosen]]*gain, field[chosen])

        for kind in (K.VP_SCALE, K.VP_BIAS, K.VP_SHIFT, K.VP_COLLAPSE, K.VP_OCCLUSION, K.VP_COMPOUND, K.VP_BURST):
            chosen = rows(kind)
            if not len(chosen):
                continue
            env = ids[chosen]
            if kind in (K.VP_SCALE, K.VP_COMPOUND):
                depth[chosen] *= plan.vp_scale[env, None, None]
            if kind in (K.VP_BIAS, K.VP_COMPOUND):
                depth[chosen] += plan.vp_bias[env, None, None]*cfg.vp_depth_max_m
            if kind in (K.VP_SHIFT, K.VP_COMPOUND):
                depth[chosen], valid[chosen] = self._warp(depth[chosen], valid[chosen],
                    torch.stack((plan.vp_shift_x[env], plan.vp_shift_y[env]), -1).float(), torch.zeros(len(env), device=self.device))
            if kind == K.VP_COLLAPSE:
                depth[chosen] = plan.vp_collapse_value[env, None, None]*cfg.vp_depth_max_m
                valid[chosen] = 0
            if kind == K.VP_OCCLUSION:
                mask = plan.vp_occlusion_mask[env]
                depth[chosen] = torch.where(mask, cfg.vp_depth_max_m, depth[chosen])
                valid[chosen] *= ~mask
            if kind in (K.VP_COMPOUND, K.VP_BURST):
                mask = self._mask(env, elapsed[chosen], duration[chosen], moving=True) * on[chosen, None, None]
                depth[chosen] = depth[chosen]*(1-mask)+cfg.vp_depth_max_m*mask
                valid[chosen] *= 1-mask
                if kind == K.VP_COMPOUND:
                    depth[chosen] += s["bias_m"][env, None, None]*field[chosen]
                else:
                    depth[chosen] += .25*plan.vp_bias[env, None, None]*cfg.vp_depth_max_m*(~on[chosen, None, None])

        for kind in (K.VP_LOCAL_DROPOUT, K.VP_DYNAMIC_OCCLUSION):
            chosen = rows(kind)
            if not len(chosen):
                continue
            env = ids[chosen]
            near = (1-raw[chosen]/cfg.vp_depth_max_m).clamp(0, 1)
            mask = self._mask(env, elapsed[chosen], duration[chosen], moving=kind == K.VP_DYNAMIC_OCCLUSION,
                              near=near if kind == K.VP_LOCAL_DROPOUT else None)
            if kind == K.VP_LOCAL_DROPOUT:
                # Bias holes toward currently visible near surfaces and edges.
                fill = torch.where(s["return_mode"][env] == 0, 0., cfg.vp_depth_max_m)
                depth[chosen] = depth[chosen]*(1-mask)+fill[:, None, None]*mask
                valid[chosen] *= 1-mask
            else:
                # A foreground obstruction is a valid depth return, not a hole.
                foreground = (raw[chosen]-s["bias_m"][env, None, None].abs()-.1).clamp_min(cfg.vp_depth_min_m)
                depth[chosen] = depth[chosen]*(1-mask)+foreground*mask
                valid[chosen] = torch.maximum(valid[chosen], mask)

        filtered = rows(K.VP_FILTER_TRAIL)
        env = ids[filtered]
        first = env[~s["filter_initialized"][env]]
        # Initialize from the previously delivered frame, when available.
        s["filter_output"][first] = torch.where(s["delivered"][first, None, None], s["history"][first, -1], raw[filtered][~s["filter_initialized"][env]])
        s["filter_support"][first] = torch.where(s["delivered"][first, None, None], s["history_support"][first, -1], support[filtered][~s["filter_initialized"][env]])
        s["filter_initialized"][env] = True
        alpha = s["alpha"][env, None, None]
        depth[filtered] = (1-alpha)*depth[filtered]+alpha*s["filter_output"][env]
        valid[filtered] = (1-alpha)*valid[filtered]+alpha*s["filter_support"][env]
        s["filter_output"][env], s["filter_support"][env] = depth[filtered], valid[filtered]
        s["filter_error"][env] = depth[filtered]-raw[filtered]
        residual = (~active & self.residual()[ids]).nonzero().flatten()
        env = ids[residual]
        s["filter_error"][env] *= s["alpha"][env, None, None]
        small = s["filter_error"][env].abs().flatten(1).amax(-1) <= cfg.vp_residual_epsilon_m
        s["filter_error"][env[small]] = 0
        depth[residual] += s["filter_error"][env]
        if pose is not None:
            s["previous_pose"][ids] = pose[ids]
            s["pose_initialized"][ids] = True
        return depth.clamp(cfg.vp_depth_min_m, cfg.vp_depth_max_m), valid.clamp(0, 1)

    def apply(self, vp, support, source_time, packet_time, ids, *, active, plan, pose=None, now=None):
        s, cfg = self.state, self.cfg
        clock = (packet_time if now is None else torch.as_tensor(now, device=self.device, dtype=torch.float64).expand(self.n))
        ready = torch.isfinite(source_time[ids])
        captured = ids[ready & (~s["initialized"][ids] | (source_time[ids] != s["last_capture"][ids]))]
        s["new_capture"][ids] = False
        s["history_append"][ids] = False
        if len(captured):
            raw = torch.nan_to_num(vp[captured, -1], nan=1., posinf=1., neginf=1.)*cfg.vp_depth_max_m
            raw = raw.clamp(cfg.vp_depth_min_m, cfg.vp_depth_max_m)
            clean_support = support[captured, -1]
            # Pass the scheduled active window, excluding filter residuals.
            scheduled = active[captured] & (plan.step_count[captured] < plan.vp_start_frame[captured]+plan.vp_duration_frames[captured])
            frame, valid = self._corrupt(captured, raw, clean_support, scheduled, plan, pose)
            s["cursor"][captured] = (s["cursor"][captured]+1) % self.capacity
            s["captures"][captured] += 1
            s["frames"][captured, s["cursor"][captured]] = frame
            s["supports"][captured, s["cursor"][captured]] = valid
            s["times"][captured, s["cursor"][captured]] = source_time[captured]
            s["last_capture"][captured] = source_time[captured]
            s["initialized"][captured] = True
            s["new_capture"][captured] = True
            kinds = plan.vp_kind[captured]
            delay = torch.zeros(len(captured), dtype=torch.long, device=self.device)
            delayed = scheduled & ((kinds == int(K.DELAY)) | (kinds == int(K.VP_COMPOUND)))
            delay[delayed] = plan.vp_delay[captured[delayed]]
            jitter = scheduled & (kinds == int(K.VP_TIME_JITTER))
            env = captured[jitter]
            delay[jitter] = (self.rng.uniform(env)*(plan.vp_delay[env]+1)).long()
            start_hold = (s["hold_left"][env] == 0) & (self.rng.uniform(env) < .25) & s["delivered"][env]
            holding_ids = env[start_hold]
            s["hold_left"][holding_ids] = self.rng.randint(1, cfg.vp_max_hold_frames+1, holding_ids)
            delay = torch.minimum(delay, (s["captures"][captured]-1).clamp(0, cfg.max_delay_frames))
            slot = (s["cursor"][captured]-delay) % self.capacity
            image = s["frames"][captured, slot].clone()
            image_support = s["supports"][captured, slot].clone()
            timestamp = s["times"][captured, slot].clone()
            packet = packet_time[captured].clone()
            freeze = scheduled & (kinds == int(K.FREEZE))
            new = captured[freeze & ~s["freeze_initialized"][captured]]
            local_new = freeze & ~s["freeze_initialized"][captured]
            s["frozen"][new], s["frozen_support"][new] = image[local_new], image_support[local_new]
            s["frozen_time"][new], s["frozen_packet"][new] = timestamp[local_new], packet[local_new]
            s["freeze_initialized"][new] = True
            image[freeze] = s["frozen"][captured[freeze]]
            image_support[freeze] = s["frozen_support"][captured[freeze]]
            timestamp[freeze] = s["frozen_time"][captured[freeze]]
            packet[freeze] = s["frozen_packet"][captured[freeze]]
            hold = jitter & (s["hold_left"][captured] > 0) & s["delivered"][captured]
            image[hold], image_support[hold] = s["history"][captured[hold], -1], s["history_support"][captured[hold], -1]
            timestamp[hold], packet[hold] = s["last_source"][captured[hold]], s["last_packet"][captured[hold]]
            s["hold_left"][captured[hold]] -= 1
            append = ~s["delivered"][captured] | (timestamp != s["last_source"][captured])
            first = captured[~s["delivered"][captured]]
            s["history"][first] = image[~s["delivered"][captured], None]
            s["history_support"][first] = image_support[~s["delivered"][captured], None]
            s["history_times"][first] = timestamp[~s["delivered"][captured], None]
            env = captured[append]
            s["history"][env] = s["history"][env].roll(-1, 1)
            s["history_support"][env] = s["history_support"][env].roll(-1, 1)
            s["history_times"][env] = s["history_times"][env].roll(-1, 1)
            s["history"][env, -1], s["history_support"][env, -1] = image[append], image_support[append]
            s["history_times"][env, -1] = timestamp[append]
            s["last_source"][captured], s["last_packet"][captured] = timestamp, packet
            s["delivered"][captured] = True
            s["history_append"][env] = True
            s["actual_delay"][captured] = (source_time[captured]-timestamp).float().clamp_min(0)
            s["holding"][captured] = hold | freeze
            s["effect"][captured] = ((image-raw).abs() > 1e-5).flatten(1).any(-1) | ((image_support-clean_support).abs() > 1e-5).flatten(1).any(-1) | (timestamp != source_time[captured])
            changed = ((image-raw).abs() > 1e-5) | ((image_support-clean_support).abs() > 1e-5)
            s["changed_fraction"][captured] = changed.float().flatten(1).mean(-1)
            s["support_ratio"][captured] = image_support.flatten(1).mean(-1)
            s["saturation_ratio"][captured] = ((image <= cfg.vp_depth_min_m) | (image >= cfg.vp_depth_max_m)).float().flatten(1).mean(-1)
        s["source_age"][ids] = torch.where(s["delivered"][ids],
            (clock[ids]-s["last_source"][ids]).float().clamp_min(0), -1.)
        return s["history"]/cfg.vp_depth_max_m, s["history_support"], s["last_source"], s["last_packet"]

    def audit(self, plan):
        s = self.state
        values = (
            plan.lp_strength, plan.vp_strength, s["bias_m"], s["scale_error"], s["edge_mix"],
            s["shift"][:, 0], s["shift"][:, 1], s["rotation_deg"], s["mask_fraction"],
            s["mask_speed"], s["return_mode"], s["alpha"], s["filter_error"].abs().flatten(1).amax(-1),
            s["actual_delay"], s["holding"], s["effect"], s["support_ratio"],
            s["saturation_ratio"], s["new_capture"], s["history_append"], s["source_age"],
            s["centers"][:, 0, 0], s["centers"][:, 0, 1],
            s["centers"][:, 1, 0], s["centers"][:, 1, 1],
            s["centers"][:, 2, 0], s["centers"][:, 2, 1],
            s["mask_direction"][:, 0], s["mask_direction"][:, 1],
            s["field"].mean((-2, -1)), s["field"].std((-2, -1)), s["changed_fraction"],
            (s["history_times"][:, 1]-s["history_times"][:, 0]).float(),
        )
        assert len(values) == len(VP_AUDIT_ORDER)
        return torch.stack(tuple(value.float() for value in values), -1)
