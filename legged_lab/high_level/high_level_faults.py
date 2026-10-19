"""10 Hz sensor-fault pipeline for the future high-level environment.

Inputs are clean LP policy-height samples and cropped, unnormalized VP depth
in metres. Feed the SAME returned data to the actor and frozen low-level
policy. Do not apply the old PGTT, RENet depth or selector fault managers too.
No environment, network, PPO runner or simulator is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from legged_lab.perception.structured_depth import StructuredDepthFailureModel
from .high_level_fault_config import HighLevelFaultCfg, LP_KINDS, VP_MODES, SCENARIOS
from .high_level_lp_faults import HighLevelLpFaults
from .high_level_schedule import HighLevelFaultSchedule
from legged_lab.selector.randomness import PerEnvironmentRNG


@dataclass
class SensorFrame:
    """One newly captured batch; all tensors have len(env_ids) rows.

    sequence identifies a capture, not a read. LP is (B,11,17) in existing
    policy units; VP is (B,18,32) in metres, before filling/clipping. source_stamp
    is the observable measurement/header time. delivery_stamp is receive time.
    """
    data: torch.Tensor
    valid_support: torch.Tensor
    source_stamp: torch.Tensor
    delivery_stamp: torch.Tensor
    sequence: torch.Tensor


@dataclass
class FaultObservation:
    """Copied data only: hidden plan/content-age labels are deliberately absent."""
    lp: torch.Tensor
    lp_valid_support: torch.Tensor
    lp_source_stamp: torch.Tensor
    lp_delivery_stamp: torch.Tensor
    vp: torch.Tensor
    vp_valid_support: torch.Tensor
    vp_source_stamp: torch.Tensor
    vp_delivery_stamp: torch.Tensor
    initialized: torch.Tensor

    def ages(self, now):
        """Observable source ages, retaining the two distinct VP frame times."""
        time = torch.as_tensor(now, device=self.lp.device, dtype=torch.float64)
        if time.ndim == 0:
            time = time.expand(len(self.lp))
        return ((time - self.lp_source_stamp).clamp_min(0),
                (time[:, None] - self.vp_source_stamp).clamp_min(0))


class HighLevelSensorFaultPipeline:
    def __init__(self, num_envs, cfg=None, *, seed=42, device="cpu"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.cfg = HighLevelFaultCfg.from_config(cfg)
        self.num_envs, self.seed = num_envs, int(seed)
        self.device = torch.empty(0, device=device).device
        self.schedule = HighLevelFaultSchedule(num_envs, self.cfg.schedule, seed=seed, device=device)
        self.lp_faults = HighLevelLpFaults(num_envs, self.cfg.lp,
            period=self.cfg.schedule.update_period_s, seed=seed ^ 0x4C50, device=device)
        self.vp_rng = PerEnvironmentRNG(num_envs, seed ^ 0x5650, device)
        # The legacy constructor still validates automatic multi-frame ranges.
        # External starts bypass those ranges, accepting one frame explicitly.
        vp = vars(self.cfg.vp).copy()
        vp.update(failure_modes=vp.pop("modes"), failure_mode_probabilities=None,
                  failure_probability=0.0, failure_duration_range=(2, 2),
                  consecutive_dropout_duration_range=(2, 2), freeze_duration_range=(2, 2),
                  distant_dust_duration_range=(2, 2))
        self.vp_faults = StructuredDepthFailureModel(SimpleNamespace(**vp), num_envs, device)
        self.lp = torch.full((num_envs, 11, 17), self.cfg.lp.lp_invalid_height, device=device)
        self.lp_support = torch.zeros_like(self.lp)
        self.vp = torch.zeros(num_envs, 2, 18, 32, device=device)
        self.vp_support = torch.zeros_like(self.vp)
        for name, shape in (("lp_source", (num_envs,)), ("lp_delivery", (num_envs,)),
                            ("vp_source", (num_envs, 2)), ("vp_delivery", (num_envs, 2)),
                            ("vp_content", (num_envs, 2)), ("frozen_content", (num_envs,)),
                            ("last_capture", (num_envs,))):
            setattr(self, name, torch.full(shape, -torch.inf, device=device, dtype=torch.float64))
        self.lp_sequence = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.vp_sequence = self.lp_sequence.clone()
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.actual_counts = torch.zeros(num_envs, 4, device=device, dtype=torch.long)
        self.visible_counts = torch.zeros_like(self.actual_counts)
        self.planned_counts = torch.zeros_like(self.actual_counts)
        self.vp_effect_history = torch.zeros(num_envs, 2, device=device, dtype=torch.bool)
        self.residual_frames = torch.zeros(num_envs, device=device, dtype=torch.long)

    def _ids(self, env_ids):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None else
               torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1))
        if ids.unique().numel() != len(ids) or ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError("Environment IDs must be unique and in range")
        return ids

    def _check_frame(self, packet, ids, shape):
        if packet.data.shape != (len(ids), *shape) or packet.valid_support.shape != packet.data.shape:
            raise ValueError(f"Sensor data/support require shape {(len(ids), *shape)}")
        for value in (packet.sequence, packet.source_stamp, packet.delivery_stamp):
            if value.shape != (len(ids),):
                raise ValueError("Each packet needs a sequence, source time and delivery time per row")
        if packet.sequence.dtype not in (torch.int32, torch.int64) or (packet.sequence < 0).any():
            raise ValueError("Capture sequence numbers must be nonnegative integers")
        if (not torch.isfinite(packet.source_stamp).all() or not torch.isfinite(packet.delivery_stamp).all()
                or (packet.source_stamp > packet.delivery_stamp + 1e-7).any()):
            raise ValueError("Measurement times must be finite and no later than delivery")
        if (not torch.isfinite(packet.valid_support).all() or
                ((packet.valid_support < 0) | (packet.valid_support > 1)).any()):
            raise ValueError("Raw valid support must lie in [0, 1]")
        for value in vars(packet).values():
            if value.device != self.device:
                raise ValueError("Sensor tensors must already be on the pipeline device")

    @torch.no_grad()
    def update(self, lp: SensorFrame, vp: SensorFrame, world_pose, *, env_ids=None):
        """Accept paired new 10 Hz captures; duplicates are ignored atomically.

        world_pose has (B,4) torso world [x,y,z,yaw_rad]. Call read() at 50/5 Hz.
        Capture delivery must advance by 0.1 s; skipped sensor updates fail
        explicitly instead of silently stretching event durations.
        """
        ids = self._ids(env_ids)
        self._check_frame(lp, ids, (11, 17))
        self._check_frame(vp, ids, (18, 32))
        if world_pose.shape != (len(ids), 4) or not torch.isfinite(world_pose).all():
            raise ValueError("Provide finite torso world [x,y,z,yaw] for each capture row")
        if world_pose.device != self.device:
            raise ValueError("World pose must be on the pipeline device")
        lp_new, vp_new = lp.sequence > self.lp_sequence[ids], vp.sequence > self.vp_sequence[ids]
        if (lp_new != vp_new).any():
            raise ValueError("LP/VP must be paired new captures on the same 10 Hz tick")
        if (lp.sequence < self.lp_sequence[ids]).any() or (vp.sequence < self.vp_sequence[ids]).any():
            raise ValueError("Out-of-order capture sequence; reset rows before starting a new episode")
        if not lp_new.any():
            return self.read()
        rows = lp_new.nonzero().flatten()
        fresh = ids[rows]
        now = vp.delivery_stamp[rows].double()
        if not torch.allclose(now, lp.delivery_stamp[rows].double(), rtol=0, atol=1e-6):
            raise ValueError("Paired LP/VP deliveries must have a common sensor-clock time")
        previous = self.initialized[fresh]
        if previous.any() and not torch.allclose(
            (now - self.last_capture[fresh])[previous],
            torch.full_like(now[previous], self.cfg.schedule.update_period_s), rtol=0, atol=1e-5,
        ):
            raise ValueError("New captures must advance by exactly one 10 Hz sensor tick")
        full_lp, full_support = self.lp.clone(), self.lp_support.clone()
        clean = lp.data[rows].float()
        support = lp.valid_support[rows].float()
        support = torch.where(torch.isfinite(clean) & (clean != self.cfg.lp.lp_invalid_height), support, 0.)
        full_lp[fresh] = torch.where(support > 0, clean.clamp(*self.cfg.lp.lp_height_clip),
                                     self.cfg.lp.lp_invalid_height)
        full_support[fresh] = support
        lp_stamp = self.lp_source.clone()
        lp_stamp[fresh] = lp.source_stamp[rows].double()
        pose = torch.zeros(self.num_envs, 4, device=self.device)
        pose[fresh] = world_pose[rows]
        raw_vp = vp.data[rows].float()
        vp_support = torch.where(torch.isfinite(raw_vp) & (raw_vp > 0),
                                 vp.valid_support[rows].float(), 0.)
        raw_vp = torch.where(torch.isfinite(raw_vp), raw_vp, self.cfg.vp.depth_max_m).unsqueeze(-1)
        vp_support = vp_support.unsqueeze(-1)
        if self.cfg.enable:
            lp_begin, vp_begin = self.schedule.advance(fresh)
            self.lp_faults.begin(lp_begin, self.schedule.lp_kind[lp_begin],
                                 self.schedule.lp_duration[lp_begin], pose)
            if len(vp_begin):
                lookup = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
                lookup[fresh] = torch.arange(len(fresh), device=self.device)
                local = lookup[vp_begin]
                self.vp_faults.start_external(raw_vp[local], vp_support[local], vp_begin,
                    self.schedule.vp_mode[vp_begin], self.schedule.vp_duration[vp_begin], rng=self.vp_rng)
                frozen = vp_begin[self.schedule.vp_mode[vp_begin] == 6]
                self.frozen_content[frozen] = vp.source_stamp[rows[lookup[frozen]]].double()
            frozen_now = self.vp_faults.active_modes[fresh] == 6
            full_lp, full_support, lp_stamp = self.lp_faults.apply(
                full_lp, full_support, lp_stamp, fresh, pose)
            depth, vp_support = self.vp_faults.apply_external(raw_vp, vp_support, fresh, rng=self.vp_rng)
        else:
            depth = raw_vp
            frozen_now = torch.zeros(len(fresh), dtype=torch.bool, device=self.device)
        # Keep the frozen RENet/deployment convention. Zero failures become
        # 0.04 numerically while their raw measurement support stays zero.
        depth = depth.squeeze(-1).clamp(self.cfg.vp.depth_min_m, self.cfg.vp.depth_max_m)
        depth = depth / self.cfg.vp.depth_max_m
        vp_support = vp_support.squeeze(-1)
        self.lp[fresh], self.lp_support[fresh], self.lp_source[fresh] = (
            full_lp[fresh], full_support[fresh], lp_stamp[fresh])
        self.lp_delivery[fresh] = lp.delivery_stamp[rows].double()
        content = torch.where(frozen_now, self.frozen_content[fresh], vp.source_stamp[rows].double())
        for buffer, value in ((self.vp, depth), (self.vp_support, vp_support),
                              (self.vp_source, vp.source_stamp[rows].double()),
                              (self.vp_delivery, now), (self.vp_content, content)):
            buffer[fresh] = buffer[fresh].roll(-1, dims=1)
            buffer[fresh, -1] = value
            first = ~self.initialized[fresh]
            buffer[fresh[first]] = value[first, None]
        if self.cfg.enable:
            lp_effect = ((full_lp[fresh] - lp.data[rows]).abs() > 1e-5).flatten(1).any(-1)
            lp_effect |= ((full_support[fresh] - lp.valid_support[rows]).abs() > 1e-5).flatten(1).any(-1)
            lp_effect |= lp_stamp[fresh] != lp.source_stamp[rows].double()
            vp_effect = ((depth - raw_vp.squeeze(-1).clamp(
                self.cfg.vp.depth_min_m, self.cfg.vp.depth_max_m) / self.cfg.vp.depth_max_m).abs()
                > 1e-5).flatten(1).any(-1)
            vp_effect |= (vp_support != vp.valid_support[rows]).flatten(1).any(-1)
            vp_effect |= content != vp.source_stamp[rows].double()
            category = lp_effect.long() + 2 * vp_effect.long()
            self.actual_counts[fresh, category] += 1
            self.vp_effect_history[fresh] = self.vp_effect_history[fresh].roll(-1, dims=1)
            self.vp_effect_history[fresh, -1] = vp_effect
            first = ~self.initialized[fresh]
            self.vp_effect_history[fresh[first]] = vp_effect[first, None]
            visible_category = lp_effect.long() + 2 * self.vp_effect_history[fresh].any(-1).long()
            self.visible_counts[fresh, visible_category] += 1
            planned_category = self.schedule.active("lp")[fresh].long() + 2 * self.schedule.active("vp")[fresh].long()
            self.planned_counts[fresh, planned_category] += 1
            self.residual_frames[fresh] += (
                self.lp_faults.residual()[fresh] & ~self.schedule.active("lp")[fresh]).long()
        else:
            self.actual_counts[fresh, 0] += 1
            self.visible_counts[fresh, 0] += 1
            self.planned_counts[fresh, 0] += 1
        self.lp_sequence[fresh], self.vp_sequence[fresh] = lp.sequence[rows], vp.sequence[rows]
        self.last_capture[fresh] = now
        self.initialized[fresh] = True
        return self.read()

    def read(self):
        return FaultObservation(*(value.clone() for value in (
            self.lp, self.lp_support, self.lp_source, self.lp_delivery,
            self.vp, self.vp_support, self.vp_source, self.vp_delivery, self.initialized)))

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.schedule.reset(ids)
        self.lp_faults.reset(ids)
        self.vp_faults.reset(ids)
        self.lp[ids], self.lp_support[ids] = self.cfg.lp.lp_invalid_height, 0
        self.vp[ids], self.vp_support[ids] = 0, 0
        for name in ("lp_source", "lp_delivery", "vp_source", "vp_delivery", "vp_content",
                     "frozen_content", "last_capture"):
            getattr(self, name)[ids] = -torch.inf
        self.lp_sequence[ids], self.vp_sequence[ids] = -1, -1
        self.initialized[ids] = False
        self.vp_effect_history[ids] = False

    def audit(self):
        """Hidden training diagnostics, NEVER concatenate into actor observations."""
        return dict(seed=self.seed, config=self.cfg.to_dict(), scenarios=SCENARIOS,
                    lp_kinds=LP_KINDS, vp_modes=VP_MODES,
                    plans=self.schedule.scenario_counts.clone(),
                    lp_events=self.schedule.lp_counts.clone(), vp_events=self.schedule.vp_counts.clone(),
                    planned_frames=self.planned_counts.clone(), observed_frames=self.actual_counts.clone(),
                    visible_frames=self.visible_counts.clone(), residual_frames=self.residual_frames.clone(),
                    vp_content_stamp=self.vp_content.clone(),
                    frame=self.schedule.frame.clone(), scenario=self.schedule.scenario.clone(),
                    lp_kind=self.schedule.lp_kind.clone(), vp_mode=self.schedule.vp_mode.clone(),
                    lp_start=self.schedule.lp_start.clone(), vp_start=self.schedule.vp_start.clone(),
                    lp_duration=self.schedule.lp_duration.clone(), vp_duration=self.schedule.vp_duration.clone(),
                    next_plan=self.schedule.next_plan.clone(),
                    lp_remaining=self.lp_faults.remaining.clone(),
                    vp_remaining=self.vp_faults.remaining_frames.clone(),
                    lp_parameters={name: getattr(self.lp_faults, name).clone() for name in
                        ("delay", "dropout", "bias", "registration", "velocity", "linear", "map_drift")})
