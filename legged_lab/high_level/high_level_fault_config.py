"""Simulator-independent parameters for high-level sensor-fault training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from legged_lab.selector.fault_kinds import EXTENDED_LP_FAULTS


LP_KINDS = tuple(int(kind) for kind in EXTENDED_LP_FAULTS)
VP_MODES = (
    "large_hole", "lower_occlusion", "central_landing_occlusion", "stripe_missing",
    "full_dropout", "consecutive_dropout", "freeze_frame", "distant_dust",
)
SCENARIOS = ("normal", "lp_only", "vp_only", "dual")


@dataclass
class FaultScheduleCfg:
    update_period_s: float = 0.1
    scenario_probabilities: tuple = (0.30, 0.25, 0.25, 0.20)
    normal_duration_s: tuple = (1.0, 5.0)
    missing_duration_s: tuple = (0.4, 2.0)
    temporal_duration_s: tuple = (0.5, 3.0)
    persistent_duration_s: tuple = (1.0, 5.0)
    full_dropout_duration_s: tuple = (0.1, 0.3)
    dual_full_dropout_duration_s: tuple = (0.2, 0.3)
    consecutive_dropout_duration_s: tuple = (0.2, 0.5)
    recovery_duration_s: tuple = (0.5, 3.0)
    stagger_probability: float = 0.5
    stagger_s: tuple = (0.1, 0.5)
    min_overlap_s: float = 0.2

    def frames(self, bounds):
        result = []
        for value in bounds:
            steps = value / self.update_period_s
            if not math.isfinite(steps) or not math.isclose(steps, round(steps), abs_tol=1e-7):
                raise ValueError("Fault times must be multiples of the sensor period")
            result.append(round(steps))
        if len(result) != 2 or result[0] < 1 or result[1] < result[0]:
            raise ValueError("Fault time ranges must contain positive ordered endpoints")
        return tuple(result)


@dataclass
class LpFaultCfg:
    kinds: tuple = LP_KINDS
    lp_resolution_m: float = 0.1
    lp_invalid_height: float = -1.1
    lp_height_clip: tuple = (-1.0, 1.0)
    max_delay_frames: int = 5
    lp_dropout_fraction: tuple = (0.10, 1.0)
    lp_block_fraction: tuple = (0.10, 1.0)
    lp_height_bias_m_range: tuple = (-0.15, 0.15)
    lp_translation_range_m: tuple = (-0.20, 0.20)
    lp_yaw_range_deg: tuple = (-5.0, 5.0)
    lp_drift_m_per_s: tuple = (0.0, 0.05)
    lp_drift_deg_per_s: tuple = (0.0, 1.0)
    lp_map_height_velocity_m_per_s: tuple = (-0.05, 0.05)
    lp_map_translation_limit_m: float = 0.45
    lp_map_yaw_limit_deg: float = 12.0
    lp_map_height_limit_m: float = 0.25
    lp_map_random_walk_m: float = 0.012
    lp_map_correction_frames: tuple = (15, 45)
    lp_person_radius_m: tuple = (0.14, 0.32)
    lp_person_height_above_ground_m: tuple = (0.25, 1.0)
    lp_person_main_fraction: float = 0.65
    lp_map_confirm_frames: int = 3
    lp_map_retention_radius_m: float = 2.5
    burst_on_frames: tuple = (1, 5)
    burst_off_frames: tuple = (1, 4)
    compound_dropout_probability: float = 0.20
    recovery_tau_s: float = 0.5
    recovery_translation_epsilon_m: float = 0.001
    recovery_yaw_epsilon_deg: float = 0.01


@dataclass
class VpFaultCfg:
    modes: tuple = VP_MODES
    depth_min_m: float = 0.1
    depth_max_m: float = 2.5
    fill_value: float = 0.0
    large_hole_height_range: tuple = (0.35, 0.75)
    large_hole_width_range: tuple = (0.35, 0.80)
    lower_occlusion_height_range: tuple = (0.35, 0.70)
    central_landing_height_range: tuple = (0.35, 0.70)
    central_landing_width_range: tuple = (0.35, 0.75)
    central_landing_center_y_range: tuple = (0.55, 0.85)
    central_landing_center_x_range: tuple = (0.40, 0.60)
    stripe_width_range: tuple = (0.08, 0.25)
    stripe_count_range: tuple = (1, 3)
    stripe_orientations: tuple = ("vertical", "horizontal")
    distant_dust_start: float = 1.5
    distant_dust_noise_std: float = 0.08
    distant_dust_dropout_probability: float = 0.35
    distant_dust_fill_value: float = 0.0


@dataclass
class HighLevelFaultCfg:
    enable: bool = True
    schedule: FaultScheduleCfg = field(default_factory=FaultScheduleCfg)
    lp: LpFaultCfg = field(default_factory=LpFaultCfg)
    vp: VpFaultCfg = field(default_factory=VpFaultCfg)

    @classmethod
    def from_config(cls, value=None):
        if value is None:
            result = cls()
        else:
            values = value if isinstance(value, dict) else (
                value.to_dict() if hasattr(value, "to_dict") else asdict(value))
            result = cls(enable=values["enable"], schedule=FaultScheduleCfg(**values["schedule"]),
                         lp=LpFaultCfg(**values["lp"]), vp=VpFaultCfg(**values["vp"]))
        result.validate()
        return result

    def to_dict(self):
        return asdict(self)

    def validate(self):
        s, lp, vp = self.schedule, self.lp, self.vp
        if type(self.enable) is not bool or not math.isclose(s.update_period_s, 0.1, rel_tol=0, abs_tol=1e-9):
            raise ValueError("High-level sensor faults run at 10 Hz")
        probabilities = s.scenario_probabilities
        if (len(probabilities) != 4 or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7)):
            raise ValueError("Scenario probabilities must sum to one: normal, LP, VP, dual")
        for name, value in vars(s).items():
            if name.endswith("_s") and name != "update_period_s" and name != "min_overlap_s":
                s.frames(value)
        overlap = s.frames((s.min_overlap_s, s.min_overlap_s))[0]
        if not 0 <= s.stagger_probability <= 1:
            raise ValueError("Stagger probability must lie in [0, 1]")
        # The shortest LP event must permit a positive stagger and the overlap.
        shortest_lp = min(s.frames(getattr(s, name))[0] for name in
                          ("missing_duration_s", "temporal_duration_s", "persistent_duration_s"))
        shortest_vp = min(s.frames(getattr(s, name))[0] for name in
                          ("dual_full_dropout_duration_s", "consecutive_dropout_duration_s"))
        if shortest_vp < overlap or shortest_lp < overlap + s.frames(s.stagger_s)[0]:
            raise ValueError("Fault durations cannot satisfy the dual overlap and stagger")
        if tuple(lp.kinds) != LP_KINDS or tuple(vp.modes) != VP_MODES:
            raise ValueError("Use the complete 11 LP and 8 Film VP fault pools")
        for config in (lp, vp):
            for name, value in vars(config).items():
                if name in ("kinds", "modes", "stripe_orientations"):
                    continue
                values = value if isinstance(value, (tuple, list)) else (value,)
                if any(not math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite fault parameter: {name}")
                if len(values) == 2 and values[1] < values[0]:
                    raise ValueError(f"Reversed fault parameter range: {name}")
        if (lp.lp_resolution_m != 0.1 or tuple(lp.lp_height_clip) != (-1.0, 1.0)
                or lp.lp_invalid_height != -1.1 or type(lp.max_delay_frames) is not int
                or lp.max_delay_frames != 5 or lp.recovery_tau_s <= 0):
            raise ValueError("LP geometry, invalid value, delay and recovery contract changed")
        for bounds in (lp.lp_dropout_fraction, lp.lp_block_fraction):
            if not 0 <= bounds[0] <= bounds[1] <= 1:
                raise ValueError("LP missing fractions must lie in [0, 1]")
        if not 0 < vp.depth_min_m < vp.depth_max_m or vp.distant_dust_noise_std < 0:
            raise ValueError("Invalid VP metric-depth range or dust standard deviation")
        if not 0 <= vp.distant_dust_dropout_probability <= 1:
            raise ValueError("Dust dropout probability must lie in [0, 1]")
        for name in ("recovery_translation_epsilon_m", "recovery_yaw_epsilon_deg",
                     "lp_map_translation_limit_m", "lp_map_yaw_limit_deg", "lp_map_height_limit_m",
                     "lp_map_retention_radius_m"):
            if getattr(lp, name) <= 0:
                raise ValueError(f"LP parameter must be positive: {name}")
        if lp.lp_map_random_walk_m < 0 or not 0 <= lp.compound_dropout_probability <= 1:
            raise ValueError("Invalid map random walk or compound dropout")
        if type(lp.lp_map_confirm_frames) is not int or lp.lp_map_confirm_frames < 1:
            raise ValueError("Map confirmation count must be a positive integer")
        if not 0 < lp.lp_person_main_fraction <= 1:
            raise ValueError("Person main fraction must lie in (0,1]")
        for bounds in (lp.burst_on_frames, lp.burst_off_frames, lp.lp_map_correction_frames, vp.stripe_count_range):
            if any(type(v) is not int for v in bounds) or bounds[0] < 1:
                raise ValueError("Frame/count ranges require positive integers")
        for name, bounds in vars(vp).items():
            if name.endswith("_range") and name != "stripe_count_range":
                if not 0 < bounds[0] <= bounds[1] <= 1:
                    raise ValueError(f"VP image fractions must lie in (0,1]: {name}")
        if not vp.stripe_orientations or set(vp.stripe_orientations) - {"vertical", "horizontal"}:
            raise ValueError("Unknown stripe orientation")
        if vp.fill_value != 0 or vp.distant_dust_fill_value != 0:
            raise ValueError("Missing VP depth must use zero before RENet preprocessing")
