"""Explicit recurring plans; sensor-frame time is independent of mode selection."""

from __future__ import annotations

import torch

from legged_lab.selector.fault_kinds import FaultKind as K
from .high_level_fault_config import LP_KINDS, VP_MODES
from legged_lab.selector.randomness import PerEnvironmentRNG


class HighLevelFaultSchedule:
    def __init__(self, num_envs, cfg, *, seed=42, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.num_envs = num_envs
        self.rng = PerEnvironmentRNG(num_envs, seed, self.device)
        for name in ("frame", "scenario", "lp_kind", "vp_mode", "lp_start", "vp_start",
                     "lp_duration", "vp_duration", "end", "next_plan", "plan_count"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self.frame.fill_(-1)
        self.vp_mode.fill_(-1)
        self.next_plan.fill_(-1)
        self.scenario_counts = torch.zeros(num_envs, 4, dtype=torch.long, device=self.device)
        self.lp_counts = torch.zeros(num_envs, len(LP_KINDS), dtype=torch.long, device=self.device)
        self.vp_counts = torch.zeros(num_envs, len(VP_MODES), dtype=torch.long, device=self.device)

    def reset(self, ids):
        for name in ("scenario", "lp_kind", "lp_start", "vp_start", "lp_duration", "vp_duration", "end"):
            getattr(self, name)[ids] = 0
        self.vp_mode[ids] = -1
        self.frame[ids] = self.next_plan[ids] = -1
        # Lifetime counters and RNG continue; other environments remain untouched.

    def _duration(self, name, ids):
        low, high = self.cfg.frames(getattr(self.cfg, name))
        return self.rng.randint(low, high + 1, ids)

    def sample(self, ids):
        """Sample plans once. LP/VP names are explicit; old scenario IDs are not reused."""
        if not len(ids):
            return
        scenario = self.rng.categorical(tuple(self.cfg.scenario_probabilities), ids)
        self.scenario[ids] = scenario
        self.lp_kind[ids], self.vp_mode[ids] = 0, -1
        self.lp_duration[ids], self.vp_duration[ids] = 0, 0
        self.lp_start[ids] = self.vp_start[ids] = self.frame[ids]
        for branch, options, selected in (
            ("lp", LP_KINDS, (scenario == 1) | (scenario == 3)),
            ("vp", tuple(range(len(VP_MODES))), (scenario == 2) | (scenario == 3)),
        ):
            rows = ids[selected]
            choices = self.rng.randint(0, len(options), rows)
            kinds = torch.tensor(options, device=self.device)[choices]
            getattr(self, branch + ("_kind" if branch == "lp" else "_mode"))[rows] = kinds
            getattr(self, branch + "_counts")[rows, choices] += 1
            # Draw each branch duration independently of its parameter sampler.
            for kind in options:
                chosen = rows[kinds == kind]
                if not len(chosen):
                    continue
                if branch == "lp":
                    name = ("temporal_duration_s" if kind in (int(K.FREEZE), int(K.DELAY)) else
                            "missing_duration_s" if kind in (int(K.LP_RANDOM_DROPOUT),
                                int(K.LP_BLOCK_DROPOUT), int(K.LP_COLLAPSE)) else "persistent_duration_s")
                else:
                    name = ("temporal_duration_s" if kind == 6 else
                            "persistent_duration_s" if kind == 7 else
                            "full_dropout_duration_s" if kind == 4 else
                            "consecutive_dropout_duration_s" if kind == 5 else "missing_duration_s")
                duration = self._duration(name, chosen)
                if branch == "vp" and kind == 4:
                    dual = self.scenario[chosen] == 3
                    duration[dual] = self._duration("dual_full_dropout_duration_s", chosen[dual])
                getattr(self, branch + "_duration")[chosen] = duration
        dual = ids[scenario == 3]
        staggered = dual[self.rng.uniform(dual) < self.cfg.stagger_probability]
        if len(staggered):
            low, high = self.cfg.frames(self.cfg.stagger_s)
            overlap = round(self.cfg.min_overlap_s / self.cfg.update_period_s)
            lp_max = torch.minimum(self.lp_duration[staggered] - overlap,
                                   torch.full_like(staggered, high))
            vp_max = torch.minimum(self.vp_duration[staggered] - overlap,
                                   torch.full_like(staggered, high))
            lp_ok, vp_ok = lp_max >= low, vp_max >= low
            if not (lp_ok | vp_ok).all():
                raise ValueError("Sampled dual durations have no feasible stagger")
            lp_first = lp_ok & (~vp_ok | (self.rng.uniform(staggered) < 0.5))
            maximum = torch.where(lp_first, lp_max, vp_max)
            gap = low + (self.rng.uniform(staggered) * (maximum - low + 1)).long()
            self.vp_start[staggered] += torch.where(lp_first, gap, 0)
            self.lp_start[staggered] += torch.where(lp_first, 0, gap)
        self.end[ids] = torch.maximum(self.lp_start[ids] + self.lp_duration[ids],
                                      self.vp_start[ids] + self.vp_duration[ids])
        normal = ids[scenario == 0]
        self.end[normal] = self.frame[normal] + self._duration("normal_duration_s", normal)
        self.next_plan[ids] = self.end[ids] + self._duration("recovery_duration_s", ids)
        self.scenario_counts[ids, scenario] += 1
        self.plan_count[ids] += 1

    def advance(self, ids):
        self.frame[ids] += 1
        first = ids[self.next_plan[ids] < 0]
        self.next_plan[first] = self.frame[first] + self._duration("recovery_duration_s", first)
        ready = ids[self.frame[ids] >= self.next_plan[ids]]
        self.sample(ready)
        return (ids[(self.lp_duration[ids] > 0) & (self.frame[ids] == self.lp_start[ids])],
                ids[(self.vp_duration[ids] > 0) & (self.frame[ids] == self.vp_start[ids])])

    def active(self, branch):
        return ((getattr(self, branch + "_duration") > 0)
                & (self.frame >= getattr(self, branch + "_start"))
                & (self.frame < getattr(self, branch + "_start") + getattr(self, branch + "_duration")))
