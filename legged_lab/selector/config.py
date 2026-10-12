"""Configuration owned exclusively by the G1 RENet selector pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[2] / "logs/g1_renet/2026-09-04_14-11-25"
)
# selector 采集数据与导出双分支策略默认使用服务器上的冻结 checkpoint。
DEFAULT_CHECKPOINT_PATH = Path("/data/Tienkung_lab/logs/g1_renet/model_60000.pt")
# 原始部署 policy.pt 用于严格校验；服务器存放位置不同时通过 --reference-policy 指定。
DEFAULT_REFERENCE_POLICY_PATH = DEFAULT_RUN_DIR / "exported/policy.pt"


@dataclass(frozen=True)
class FaultConfig:
    """Policy-input fault ranges at the 10 Hz exteroceptive update rate."""

    update_period_s: float = 0.1
    max_delay_frames: int = 5
    min_duration_frames: int = 3
    max_duration_frames: int = 50
    max_start_delay_frames: int = 30
    min_recovery_frames: int = 5
    max_recovery_frames: int = 30
    max_dual_stagger_frames: int = 10
    vp_occlusion_fraction: tuple[float, float] = (0.1, 0.70)
    vp_scale_range: tuple[float, float] = (0.85, 1.15)
    vp_bias_m_range: tuple[float, float] = (-0.20, 0.20)
    vp_max_shift_pixels: int = 3
    vp_depth_max_m: float = 2.5
    lp_dropout_fraction: tuple[float, float] = (0.10, 0.70)
    lp_block_fraction: tuple[float, float] = (0.10, 0.50)
    lp_height_bias_m_range: tuple[float, float] = (-0.15, 0.15)
    lp_translation_range_m: tuple[float, float] = (-0.20, 0.20)
    lp_resolution_m: float = 0.1
    lp_yaw_range_deg: tuple[float, float] = (-5.0, 5.0)
    lp_drift_m_per_s: tuple[float, float] = (0.0, 0.05)
    lp_drift_deg_per_s: tuple[float, float] = (0.0, 1.0)
    lp_invalid_height: float = -1.1
    lp_height_clip: tuple[float, float] = (-1.0, 1.0)
    scenario_probabilities: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)
    severity_probabilities: tuple[float, float, float] = (0.10, 0.30, 0.60)
    # New families are opt-in. Existing V3 collection retains its original pool.
    extended_faults: bool = False
    profile: str = "baseline"
    lp_kind_weights: tuple[float, ...] = ()
    vp_kind_weights: tuple[float, ...] = ()
    burst_on_frames: tuple[int, int] = (1, 5)
    burst_off_frames: tuple[int, int] = (1, 4)
    lp_map_translation_limit_m: float = 0.45
    lp_map_yaw_limit_deg: float = 12.0
    lp_map_height_limit_m: float = 0.25
    lp_map_random_walk_m: float = 0.012
    lp_map_correction_frames: tuple[int, int] = (15, 45)
    lp_person_radius_m: tuple[float, float] = (0.14, 0.32)
    lp_person_height_above_ground_m: tuple[float, float] = (0.25, 1.0)
    lp_person_main_fraction: float = 0.65
    lp_map_confirm_frames: int = 3
    lp_map_retention_radius_m: float = 2.5

    def validate(self) -> None:
        for name in ("update_period_s", "lp_resolution_m", "vp_depth_max_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.vp_max_shift_pixels) is not int or self.vp_max_shift_pixels < 0:
            raise ValueError("vp_max_shift_pixels must be a nonnegative integer")
        if not 1 <= self.max_delay_frames <= 5:
            raise ValueError("max_delay_frames must be in [1, 5]")
        if not 1 <= self.min_duration_frames <= self.max_duration_frames:
            raise ValueError("fault duration range is invalid")
        if self.extended_faults and self.max_duration_frames < 15:
            raise ValueError("extended faults need a parent event of at least 15 frames; "
                             "use burst_on/off_frames for subsecond interruptions")
        if not 1 <= self.min_recovery_frames <= self.max_recovery_frames:
            raise ValueError("fault recovery range is invalid")
        if self.max_start_delay_frames < 0:
            raise ValueError("max_start_delay_frames must be nonnegative")
        if self.max_dual_stagger_frames < 0:
            raise ValueError("max_dual_stagger_frames must be nonnegative")
        for name, count in (("scenario_probabilities", 4), ("severity_probabilities", 3)):
            probabilities = getattr(self, name)
            if (len(probabilities) != count or
                    any(not math.isfinite(p) or p < 0 for p in probabilities) or
                    abs(sum(probabilities) - 1.0) > 1.0e-6):
                raise ValueError(f"{name} must contain {count} nonnegative probabilities summing to one")
        for name in (
            "vp_occlusion_fraction",
            "vp_scale_range",
            "vp_bias_m_range",
            "lp_dropout_fraction",
            "lp_block_fraction",
            "lp_height_bias_m_range",
            "lp_translation_range_m",
            "lp_yaw_range_deg",
            "lp_drift_m_per_s",
            "lp_drift_deg_per_s",
            "lp_height_clip",
        ):
            lo, hi = getattr(self, name)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError(f"{name} must contain finite increasing bounds")
        for name in ("vp_occlusion_fraction", "lp_dropout_fraction", "lp_block_fraction"):
            lo, hi = getattr(self, name)
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.vp_scale_range[0] <= 0:
            raise ValueError("vp_scale_range must be positive")
        if self.lp_drift_m_per_s[0] < 0 or self.lp_drift_deg_per_s[0] < 0:
            raise ValueError("drift speed ranges must be nonnegative")
        if (not math.isfinite(self.lp_invalid_height) or
                self.lp_height_clip[0] <= self.lp_invalid_height <= self.lp_height_clip[1]):
            raise ValueError("lp_invalid_height must be finite and outside lp_height_clip")
        for name, count in (("lp_kind_weights", 11), ("vp_kind_weights", 9)):
            weights = getattr(self, name)
            if weights and (not self.extended_faults or len(weights) != count
                            or any(not math.isfinite(x) or x < 0 for x in weights)
                            or abs(sum(weights) - 1.0) > 1e-6):
                raise ValueError(f"{name} requires {count} normalized extended-family weights")
        for name in ("burst_on_frames", "burst_off_frames", "lp_map_correction_frames"):
            lo, hi = getattr(self, name)
            if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
                raise ValueError(f"{name} must contain positive integer frame bounds")
        for name in ("lp_person_radius_m", "lp_person_height_above_ground_m"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi or not all(math.isfinite(x) for x in (lo, hi)):
                raise ValueError(f"{name} must contain positive finite bounds")
        for name in ("lp_map_translation_limit_m", "lp_map_yaw_limit_deg",
                     "lp_map_height_limit_m", "lp_map_retention_radius_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.lp_map_random_walk_m) or self.lp_map_random_walk_m < 0:
            raise ValueError("lp_map_random_walk_m must be nonnegative and finite")
        if not 0 < self.lp_person_main_fraction < 1 or self.lp_map_confirm_frames < 1:
            raise ValueError("person main fraction and map confirmation count are invalid")


@dataclass(frozen=True)
class SelectorConfig:
    """Network and feature contract from the approved implementation plan."""

    proprio_dim: int = 78
    latent_dim: int = 64
    velocity_dim: int = 3
    action_dim: int = 23
    health_dim_per_sensor: int = 8
    cross_dim: int = 7
    state_dim: int = 2
    history_frames: int = 5
    frame_dim: int = 283
    frame_hidden_dims: tuple[int, int] = (256, 128)
    gru_hidden_dim: int = 128
    head_hidden_dim: int = 64
    output_dim: int = 4

    def expected_frame_dim(self) -> int:
        return (
            self.proprio_dim
            + 2 * self.latent_dim
            + 2 * self.velocity_dim
            + 2 * self.action_dim
            + 2 * self.health_dim_per_sensor
            + self.cross_dim
            + self.state_dim
        )

    def validate(self) -> None:
        expected = self.expected_frame_dim()
        if self.frame_dim != expected:
            raise ValueError(
                f"selector frame_dim={self.frame_dim}, expected {expected}"
            )
        if self.output_dim != 4:
            raise ValueError("selector output must be LP/VP x short/long = 4")
        if self.history_frames != 5:
            raise ValueError("selector history must contain five 10 Hz frames")


@dataclass(frozen=True)
class RuntimeConfig:
    """Deterministic hysteresis and hard-health policy."""

    short_risk_weight: float = 0.65
    long_risk_weight: float = 0.35
    sensor_timeout_s: float = 0.30
    min_valid_ratio: float = 0.05
    invalid_confirmations: int = 2
    switch_margin: float = 0.10
    switch_confirmations: int = 3
    min_dwell_s: float = 1.0
    vp_tie_margin: float = 0.05
    both_unreliable_risk: float = 0.70
    normal_blend_steps: int = 5
    emergency_blend_steps: int = 2

    def validate(self) -> None:
        if abs(self.short_risk_weight + self.long_risk_weight - 1.0) > 1.0e-6:
            raise ValueError("risk weights must sum to one")
        if self.sensor_timeout_s <= 0 or self.min_dwell_s < 0:
            raise ValueError("runtime times must be nonnegative")
        for value in (
            self.min_valid_ratio,
            self.switch_margin,
            self.vp_tie_margin,
            self.both_unreliable_risk,
        ):
            if not 0 <= value <= 1:
                raise ValueError("risk and ratio thresholds must be in [0, 1]")
        if (
            min(
                self.invalid_confirmations,
                self.switch_confirmations,
                self.normal_blend_steps,
                self.emergency_blend_steps,
            )
            < 1
        ):
            raise ValueError("confirmation and blend step counts must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8192
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    patience: int = 10
    gradient_clip_norm: float = 1.0
    seeds: tuple[int, int, int] = (42, 43, 44)


@dataclass(frozen=True)
class CollectionConfig:
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)
    num_envs: int = 4096
    target_samples: int = 1_048_576
    shard_size: int = 8192
    short_horizon_steps: int = 20
    long_horizon_steps: int = 50
    max_train_terrain_level: int = 6
    ood_terrain_levels: tuple[int, int, int] = (7, 8, 9)
    split_probabilities: tuple[float, float, float] = (0.70, 0.15, 0.15)
    reference_terrain_mean: float = 5.27
    allow_nonproduction_num_envs: bool = False
    candidate_stride_steps: int = 5
    behavior_min_dwell_frames: int = 10
    behavior_max_dwell_frames: int = 50
    balanced_scenarios: bool = True
    max_candidate_batches: int = 100_000

    def validate(self) -> None:
        if self.num_envs != 4096 and not self.allow_nonproduction_num_envs:
            raise ValueError(
                "the approved selector collector uses exactly 4096 environments"
            )
        if self.short_horizon_steps != 20 or self.long_horizon_steps != 50:
            raise ValueError("horizons must be 0.4 s and 1.0 s at 50 Hz")
        if self.target_samples <= 0 or self.shard_size <= 0:
            raise ValueError("collection sizes must be positive")
        if self.candidate_stride_steps != 5:
            raise ValueError(
                "selector candidates must advance at 10 Hz (5 control steps)"
            )
        if not 1 <= self.behavior_min_dwell_frames <= self.behavior_max_dwell_frames:
            raise ValueError("behavior dwell range is invalid")
        if self.max_candidate_batches < 1:
            raise ValueError("max_candidate_batches must be positive")
        if abs(sum(self.split_probabilities) - 1.0) > 1.0e-6:
            raise ValueError("split_probabilities must sum to one")


@dataclass(frozen=True)
class SelectorExperimentConfig:
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)

    def validate(self) -> None:
        self.selector.validate()
        self.faults.validate()
        self.runtime.validate()
        self.collection.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_checkpoint(self, repo_root: str | Path) -> Path:
        checkpoint = Path(self.collection.checkpoint_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(repo_root) / checkpoint
        return checkpoint.resolve()
