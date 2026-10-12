"""Aligned 10 Hz candidates and isolated 1 s counterfactual label forks."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import hashlib
import time
from typing import Any
import torch
from .config import CollectionConfig, SelectorConfig, RuntimeConfig
from .dataset import SelectorBatch, SelectorDatasetWriter
from .collection_audit import CollectionAudit, collection_audit_contract
from .faults import FaultSnapshot, SelectorFaultManager
from .features import FeatureHistory, SensorHealthTracker, build_selector_frame
from .lp_map_faults import NOISE_AUDIT_ORDER
from .supplement import OutcomeQuotaSampler
from .feature_contract import selector_feature_contract
from .labels import (
    RolloutOutcome,
    SafetyOutcomeAccumulator,
    capture_safety_step,
    safety_label_contract,
    step_with_safety_capture,
    TILT_LIMIT_RAD, CLEARANCE_LIMIT_M,
)
from .policy_adapter import FrozenRENetAdapter
from .snapshot import RENetSnapshot, RENetSnapshotManager, snapshot_restore_contract
from .collection_physics import collection_physics_settings
from .runtime import ActionBlender, SelectorStateMachine
from .randomness import (
    PerEnvironmentRNG,
    EnvironmentRandomness,
    bind_environment_randomness,
)


REPLAY_STABILITY_FIELDS = ("fail_short", "fail_long", "label_valid")
LEGACY_REPLAY_STABILITY_FIELDS = (
    "fail_short", "fail_long", "first_failure_step", "first_reset_step",
    "terminated", "torso_contact", "label_valid", "first_timeout_step",
    "first_invalid_step", "observed_steps",
)
REPLAY_AUDIT_FIELDS = (*LEGACY_REPLAY_STABILITY_FIELDS, "first_failure_flags")
REPLAY_TIMING_FIELDS = ("first_failure_step", "first_reset_step", "observed_steps")


def replay_filter_contract() -> dict:
    return {
        "version": 2,
        "repetitions_per_branch": 2,
        "hard_conditions": ["fail_short_equal", "fail_long_equal", "both_replays_label_valid"],
        "pair_acceptance": "both LP and VP satisfy all hard conditions",
        "audit_fields": list(REPLAY_AUDIT_FIELDS),
        "audit_value_source": "second replay; time and cause differences do not reject valid agreeing labels",
        "legacy_comparison": "old persistence gate: ten legacy discrete fields equal and both outcomes valid",
        "both_replays_valid_scope": "all four LP/VP rollouts, before scenario quotas",
        "startup_hard_checks": ["initial_restore", "label_agreement", "label_validity_agreement"],
        "startup_label_agreement_min": 0.999,
        "startup_timing_scope": "audit only",
    }


def replay_mismatch_masks(first: RolloutOutcome, second: RolloutOutcome) -> dict:
    """Return audit mismatches, including fields that no longer reject a row."""
    return {name: getattr(first, name) != getattr(second, name)
            for name in REPLAY_AUDIT_FIELDS}


def replay_rejection_masks(first, second) -> dict:
    return {
        "label_flip": (first.fail_short != second.fail_short) | (first.fail_long != second.fail_long),
        "first_replay_invalid": ~first.label_valid,
        "second_replay_invalid": ~second.label_valid,
    }


def legacy_outcome_replay_stability(first, second) -> torch.Tensor:
    """The old persistence gate, including its separate outcome-validity gate."""
    return (torch.stack([getattr(first, name) == getattr(second, name)
                         for name in LEGACY_REPLAY_STABILITY_FIELDS]).all(dim=0)
            & first.label_valid & second.label_valid)


def outcome_replay_stability(
    first: RolloutOutcome, second: RolloutOutcome
) -> torch.Tensor:
    """Keep only valid agreeing 20/50-step labels; timing and causes are audit."""
    return ~torch.stack(list(replay_rejection_masks(first, second).values())).any(dim=0)


def replay_disagreement_summary(first, second, eligible=None) -> dict:
    """Count label flips separately from agreeing labels with timing jitter."""
    mismatch = replay_mismatch_masks(first, second)
    if eligible is None:
        eligible = torch.ones_like(first.fail_long, dtype=torch.bool)
    reasons = replay_rejection_masks(first, second)
    rejected = ~outcome_replay_stability(first, second)
    legacy_rejected = ~legacy_outcome_replay_stability(first, second)
    audit_difference = torch.stack(list(mismatch.values())).any(dim=0)
    label_flip = reasons["label_flip"]
    non_timing = torch.stack([value for name, value in mismatch.items()
                              if name not in REPLAY_TIMING_FIELDS]).any(dim=0)
    both_failed = (first.first_failure_step > 0) & (second.first_failure_step > 0)
    delta = (first.first_failure_step - second.first_failure_step).abs()

    def count(mask):
        return int((eligible & mask).sum().item())

    return {
        "checked": int(eligible.sum().item()),
        "rejected": count(rejected),
        "label_flip": count(label_flip),
        "label_invalid": count(reasons["first_replay_invalid"] | reasons["second_replay_invalid"]),
        "first_replay_invalid": count(reasons["first_replay_invalid"]),
        "second_replay_invalid": count(reasons["second_replay_invalid"]),
        "labels_agree_but_rejected": count(rejected & ~label_flip),
        "legacy_rejected": count(legacy_rejected),
        "retained_despite_legacy_difference": count(~rejected & legacy_rejected),
        "retained_with_audit_difference": count(~rejected & audit_difference),
        "retained_with_timing_only_difference": count(~rejected & audit_difference & ~non_timing),
        # Field counts overlap; the categories above must not be added to them.
        "field_mismatch": {name: count(value) for name, value in mismatch.items()},
        "failure_step_changes": {
            "appeared": count((first.first_failure_step < 0) & (second.first_failure_step > 0)),
            "disappeared": count((first.first_failure_step > 0) & (second.first_failure_step < 0)),
            "one_step": count(both_failed & (delta == 1)),
            "two_steps": count(both_failed & (delta == 2)),
            "three_to_five_steps": count(both_failed & (delta >= 3) & (delta <= 5)),
            "more_than_five_steps": count(both_failed & (delta > 5)),
        },
    }


def _add_replay_counts(total, update):
    for name, value in update.items():
        if isinstance(value, dict):
            _add_replay_counts(total.setdefault(name, {}), value)
        else:
            total[name] = total.get(name, 0) + value


def _robot_diagnostic_tensors(env) -> dict:
    """Read controller buffers plus independent PhysX state when available."""
    robot = env.robot
    data = robot.data
    result = {}
    for name in ("root_state_w", "joint_pos", "joint_vel", "joint_pos_target",
                 "joint_vel_target", "joint_effort_target", "joint_stiffness",
                 "joint_damping", "computed_torque", "applied_torque"):
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            result[name] = value.clone()
    for name in ("_joint_pos_target_sim", "_joint_vel_target_sim", "_joint_effort_target_sim",
                 "_external_force_b", "_external_torque_b"):
        value = getattr(robot, name, None)
        if isinstance(value, torch.Tensor):
            result[name] = value.clone()
    view = getattr(robot, "root_physx_view", None)
    for name in ("get_root_transforms", "get_root_velocities", "get_dof_positions", "get_dof_velocities"):
        getter = getattr(view, name, None)
        if callable(getter):
            # Unsupported backend getters are reported, without preventing the
            # remaining diagnostic fields from being captured.
            try:
                value = getter()
                if isinstance(value, torch.Tensor):
                    result[f"physx.{name}"] = value.clone()
            except (AttributeError, NotImplementedError, RuntimeError) as error:
                result.setdefault("getter_errors", {})[name] = str(error)
    return result


@contextmanager
def capture_physics_substeps(env, diagnostics):
    """Observe the first control step, restoring both hooks even on errors."""
    if diagnostics is None:
        yield
        return
    sim_step = getattr(env.sim, "step", None)
    scene_update = getattr(env.scene, "update", None)
    if not callable(sim_step) or not callable(scene_update):
        diagnostics["physics_trace_unavailable"] = "sim.step or scene.update is unavailable"
        yield
        return
    trace = diagnostics["physics_substeps"] = []
    missing = object()
    previous = [(obj, name, vars(obj).get(name, missing))
                for obj, name in ((env.sim, "step"), (env.scene, "update"))]

    def traced_step(*args, **kwargs):
        trace.append({"before_sim_step": _robot_diagnostic_tensors(env)})
        return sim_step(*args, **kwargs)

    def traced_update(*args, **kwargs):
        result = scene_update(*args, **kwargs)
        if trace:
            trace[-1]["after_scene_update"] = _robot_diagnostic_tensors(env)
        return result

    try:
        env.sim.step = traced_step
        env.scene.update = traced_update
        yield
    finally:
        for obj, name, value in previous:
            if value is missing:
                vars(obj).pop(name, None)
            else:
                setattr(obj, name, value)


def _diagnostic_tensor_errors(first, second, live, atol):
    errors = {}
    for name in sorted(set(first) | set(second)):
        a, b = first.get(name), second.get(name)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            if name == "getter_errors":
                errors[name] = {"first": a, "second": b}
            else:
                errors[name] = {"first_present": isinstance(a, torch.Tensor),
                                "second_present": isinstance(b, torch.Tensor)}
            continue
        if a.shape != b.shape:
            errors[name] = {"shape_changed": [list(a.shape), list(b.shape)]}
            continue
        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
        delta = torch.where(a == b, torch.zeros_like(delta), delta)
        delta = torch.nan_to_num(delta, nan=float("inf"), posinf=float("inf"))
        masked = delta[live] if delta.ndim and delta.shape[0] == len(live) else delta

        def maximum(value):
            if not value.numel():
                return None
            result = float(value.max().item())
            return result if math.isfinite(result) else "nonfinite"

        errors[name] = {
            "max_abs_error": maximum(delta),
            "no_reset_max_abs_error": maximum(masked),
            "elements_above_atol": int((delta > atol).sum().item()),
        }
    return errors


def physics_substep_comparison(first, second, atol):
    trace_a, trace_b = first.get("physics_substeps", []), second.get("physics_substeps", [])
    if not trace_a or not trace_b:
        return {"available": False, "reason": [first.get("physics_trace_unavailable"),
                                                second.get("physics_trace_unavailable")]}
    live = ~(first["reset"] | second["reset"])
    rows = []
    for index, (a, b) in enumerate(zip(trace_a, trace_b), 1):
        rows.append({"physics_substep": index, **{
            phase: _diagnostic_tensor_errors(a.get(phase, {}), b.get(phase, {}), live, atol)
            for phase in ("before_sim_step", "after_scene_update")
        }})
    return {
        "available": True,
        "first_substep_count": len(trace_a), "second_substep_count": len(trace_b),
        "substeps": rows,
        "torque_note": "applied/computed torque is telemetry; implicit actuator telemetry does not establish the solver's actual joint effort",
        "scope": "first control step only; unobserved simulator state is not proven equal by these fields",
    }


@dataclass
class ObservationStreamSnapshot:
    fault: FaultSnapshot
    lp_health: Any
    vp_health: Any
    tensors: dict[str, torch.Tensor]
    control_step: int


class FaultedObservationStream:
    def __init__(self, adapter, fault_manager, *, control_steps_per_sensor_update=5):
        self.adapter, self.policy, self.faults = adapter, adapter.policy, fault_manager
        self.num_envs, self.device = fault_manager.num_envs, fault_manager.device
        self.control_steps_per_sensor_update = control_steps_per_sensor_update
        self.proprio_end = int(self.policy.proprio_actor_dim)
        self.lidar_end = self.proprio_end + int(self.policy.lidar_flat_dim)
        self.depth_end = self.lidar_end + int(self.policy.depth_flat_dim)
        self.lp_shape, self.vp_shape = fault_manager.lp_shape, fault_manager.vp_shape
        self.lp_health_tracker = SensorHealthTracker(
            self.num_envs, self.lp_shape, self.device
        )
        self.vp_health_tracker = SensorHealthTracker(
            self.num_envs, self.vp_shape, self.device
        )
        self.last_lp = torch.full(
            (self.num_envs, *self.lp_shape),
            self.faults.cfg.lp_invalid_height,
            device=self.device,
        )
        self.last_vp = torch.ones(self.num_envs, *self.vp_shape, device=self.device)
        self.last_lp_valid = torch.zeros_like(self.last_lp)
        self.last_vp_valid = torch.zeros_like(self.last_vp)
        self.last_lp_timestamp = torch.full(
            (self.num_envs,), -torch.inf, device=self.device, dtype=torch.float64
        )
        self.last_vp_timestamp = self.last_lp_timestamp.clone()
        self.last_lp_packet_timestamp = self.last_lp_timestamp.clone()
        self.last_vp_packet_timestamp = self.last_lp_timestamp.clone()
        self.initialized = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.control_step = 0
        self.lp_health = None
        self.vp_health = None

    def snapshot(self):
        names = (
            "last_lp",
            "last_vp",
            "last_lp_valid",
            "last_vp_valid",
            "last_lp_timestamp",
            "last_vp_timestamp",
            "last_lp_packet_timestamp",
            "last_vp_packet_timestamp",
            "initialized",
        )
        return ObservationStreamSnapshot(
            self.faults.snapshot(),
            self.lp_health_tracker.snapshot(),
            self.vp_health_tracker.snapshot(),
            {n: getattr(self, n).clone() for n in names},
            self.control_step,
        )

    def restore(self, snapshot):
        self.faults.restore(snapshot.fault)
        self.lp_health_tracker.restore(snapshot.lp_health)
        self.vp_health_tracker.restore(snapshot.vp_health)
        for name, value in snapshot.tensors.items():
            getattr(self, name).copy_(value)
        self.control_step = snapshot.control_step
        self.lp_health = None
        self.vp_health = None

    def reset_envs(self, ids):
        self.faults.reset(ids)
        self.lp_health_tracker.reset(ids)
        self.vp_health_tracker.reset(ids)
        self.initialized[ids] = False

    def query_health(self, now):
        self.lp_health = self.lp_health_tracker.update(
            self.last_lp,
            self.last_lp_timestamp,
            now,
            valid_mask=self.last_lp_valid,
            packet_timestamp=self.last_lp_packet_timestamp,
        )
        self.vp_health = self.vp_health_tracker.update(
            self.last_vp,
            self.last_vp_timestamp,
            now,
            valid_mask=self.last_vp_valid,
            packet_timestamp=self.last_vp_packet_timestamp,
        )
        return self.lp_health, self.vp_health

    def process(
        self,
        observations,
        *,
        lp_timestamp,
        vp_timestamp,
        now,
        force_update=False,
        advance_control_step=True,
        lp_valid=None,
        vp_valid=None,
        world_pose=None,
    ):
        if advance_control_step:
            self.control_step += 1
        tick = (
            force_update
            or self.control_step % self.control_steps_per_sensor_update == 0
        )
        rows = torch.ones_like(self.initialized) if tick else ~self.initialized
        if rows.any():
            lp = observations[:, self.proprio_end : self.lidar_end].reshape(
                self.num_envs, *self.lp_shape
            )
            vp = observations[:, self.lidar_end : self.depth_end].reshape(
                self.num_envs, *self.vp_shape
            )
            if lp_valid is None or vp_valid is None:
                raise ValueError(
                    "collection requires raw LP validity and VP valid support from environment"
                )
            faulted = self.faults.apply(
                lp,
                vp,
                lp_timestamp,
                vp_timestamp,
                lp_valid=lp_valid,
                vp_valid=vp_valid,
                env_ids=rows.nonzero().flatten(),
                lp_packet_timestamp=torch.full_like(self.last_lp_timestamp, float(now)),
                vp_packet_timestamp=vp_timestamp,
                world_pose=world_pose,
            )
            for name, value in (
                ("last_lp", faulted.lp),
                ("last_vp", faulted.vp),
                ("last_lp_valid", faulted.lp_valid),
                ("last_vp_valid", faulted.vp_valid),
                ("last_lp_timestamp", faulted.lp_timestamp),
                ("last_vp_timestamp", faulted.vp_timestamp),
                ("last_lp_packet_timestamp", faulted.lp_packet_timestamp),
                ("last_vp_packet_timestamp", faulted.vp_packet_timestamp),
            ):
                getattr(self, name)[rows] = value[rows]
            self.initialized[rows] = True
        self.query_health(now)
        output = observations.clone()
        output[:, self.proprio_end : self.lidar_end] = self.last_lp.flatten(1)
        output[:, self.lidar_end : self.depth_end] = self.last_vp.flatten(1)
        return output, tick


@dataclass
class CollectionSnapshot:
    environment: RENetSnapshot
    observation_stream: ObservationStreamSnapshot
    feature_history: tuple[torch.Tensor, torch.Tensor]
    observations: torch.Tensor
    tensors: dict[str, torch.Tensor]
    blender: dict[str, torch.Tensor]
    environment_rng: dict[str, torch.Tensor]
    behavior_rng: torch.Tensor


class CounterfactualCollector:
    def __init__(
        self,
        env,
        adapter,
        writer,
        *,
        faults,
        selector_cfg=None,
        collection_cfg=None,
        seed=42,
        terrain_seed=None,
        behavior_seed=None,
        supplement_cfg=None,
        allowed_groups=None,
        progress=None,
    ):
        self.env, self.adapter, self.writer = env, adapter, writer
        self.selector_cfg = selector_cfg or SelectorConfig()
        self.collection_cfg = collection_cfg or CollectionConfig()
        self.selector_cfg.validate()
        self.collection_cfg.validate()
        self.runtime_cfg = RuntimeConfig()
        if env.num_envs != self.collection_cfg.num_envs:
            raise ValueError("environment count differs from collection config")
        if getattr(env, "height_scanner", None) is None:
            raise ValueError("safety labels require a ground-truth height scanner")
        self.device = torch.device(env.device)
        self.seed = int(seed)
        self.terrain_seed = self.seed if terrain_seed is None else int(terrain_seed)
        self.supplement_cfg = supplement_cfg
        self.progress = progress
        self.sampling_progress = {"real_candidate_ticks": 0, "labelled_batches": 0,
                                  "paced_or_interval_skips": 0, "no_observable_rows": 0}
        self.allowed_groups = set(allowed_groups or ())
        self.quota_sampler = None
        self.candidate_prefix = None
        if supplement_cfg is not None:
            supplement_cfg.validate(self.collection_cfg.target_samples)
            if not self.allowed_groups:
                raise ValueError("supplement collection requires the original split's terrain allowlist")
            self.quota_sampler = OutcomeQuotaSampler(supplement_cfg, self.collection_cfg.target_samples, seed+5000)
            if writer is not None:
                run_id = writer.metadata["run_info"]["collection_run_id"]
                self.candidate_prefix = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "big") % (2**31)
        self.snapshot_manager = RENetSnapshotManager(env)
        self.stream = FaultedObservationStream(adapter, faults)
        self.feature_history = FeatureHistory(
            env.num_envs, self.selector_cfg, self.device
        )
        self.environment_rng = EnvironmentRandomness(
            env.num_envs, (seed if behavior_seed is None else behavior_seed) + 3000, self.device
        )
        bind_environment_randomness(env, self.environment_rng)
        self.behavior_rng = PerEnvironmentRNG(env.num_envs, (seed if behavior_seed is None else behavior_seed) + 4000, self.device)
        ids = torch.arange(env.num_envs, device=self.device)
        self.behavior_mask = self.behavior_rng.randint(0, 2, ids).float()
        self.last_behavior_switch = torch.full(
            (env.num_envs,), self._time(), device=self.device, dtype=torch.float64
        )
        self.behavior_deadline = self.last_behavior_switch + self._sample_dwell(ids)
        self.last_feature_time = self.last_behavior_switch.clone()
        self.episode_serial = torch.zeros(
            env.num_envs, device=self.device, dtype=torch.long
        )
        self.lp_invalid_count = torch.zeros_like(self.episode_serial)
        self.vp_invalid_count = self.lp_invalid_count.clone()
        self.blender = ActionBlender(
            env.num_envs, env.num_actions, device=self.device, cfg=self.runtime_cfg
        )
        self.blender.seed(env.action.clone())
        self.batch_index = 0
        self.observations = None

    def _sample_dwell(self, ids):
        cfg = self.collection_cfg
        return (
            self.behavior_rng.randint(
                cfg.behavior_min_dwell_frames, cfg.behavior_max_dwell_frames + 1, ids
            ).double()
            * 0.1
        )

    def _time(self):
        return float(
            getattr(
                self.env.sim,
                "current_time",
                self.env.sim_step_counter * self.env.physics_dt,
            )
        )

    def _progress(self, stage, **fields):
        if getattr(self, "progress", None) is not None:
            self.progress.set_stage(stage, candidate_tick=self.batch_index,
                                    sim_time_s=self._time(), **fields)

    def _warmup(self):
        cfg = getattr(self, "supplement_cfg", None)
        if cfg is None or not cfg.warmup_s:
            return
        steps = math.ceil(cfg.warmup_s / (self.collection_cfg.candidate_stride_steps*.02))
        start = self._time()
        print(f"[SelectorCollector] real warmup started: {cfg.warmup_s:.1f}s; no labels are collected", flush=True)
        for step in range(steps):
            if step % 10 == 0:
                self._progress("real_warmup", completed_s=self._time()-start,
                               requested_s=cfg.warmup_s)
            self._advance_behavior()
        print(f"[SelectorCollector] real warmup complete: {self._time()-start:.2f}s", flush=True)

    def _sensor_timestamps(self):
        now = self._time()
        lp_sensor = getattr(self.env, "lp_sensor", None)
        camera = getattr(self.env, "camera", None)
        lp_age = (
            getattr(self.env, "extras", {}).get("observations", {}).get("lp_frame_age")
        )
        if lp_age is not None:
            lp = now - lp_age.reshape(self.env.num_envs, -1)[:, -1].double()
        elif lp_sensor is not None and hasattr(lp_sensor, "scan_timestamps"):
            lp = (
                now
                - (lp_sensor.current_timestamps - lp_sensor.scan_timestamps).double()
            )
        else:
            lp = torch.full(
                (self.env.num_envs,), now, device=self.device, dtype=torch.float64
            )
        if camera is not None and hasattr(camera, "current_timestamps"):
            vp = (
                now
                - (
                    camera.current_timestamps - self.env.depth_last_capture_time
                ).double()
            )
        else:
            vp = torch.full_like(lp, now)
        return lp, vp

    def _quality(self):
        values = self.env.extras.get("observations", {})
        return dict(
            lp_valid=values.get("lp_valid_mask"),
            vp_valid=values.get("vp_valid_support"),
        )

    def _world_pose(self):
        if getattr(self.stream.faults, "map_faults", None) is None:
            return None
        data = self.env.robot.data
        ids = getattr(self.env, "torso_link_ids", None)
        if ids is None or len(ids) != 1:
            raise ValueError("world map faults require exactly one torso_link body")
        position = data.body_pos_w[:, ids[0]]
        quat = data.body_quat_w[:, ids[0]]  # Isaac uses w,x,y,z.
        yaw = torch.atan2(2*(quat[:, 0]*quat[:, 3]+quat[:, 1]*quat[:, 2]),
                          1-2*(quat[:, 2].square()+quat[:, 3].square()))
        return torch.cat((position, yaw[:, None]), -1)

    def initialize(self):
        raw, _ = self.env.get_observations()
        lp, vp = self._sensor_timestamps()
        self.observations, _ = self.stream.process(
            raw,
            lp_timestamp=lp,
            vp_timestamp=vp,
            now=self._time(),
            force_update=True,
            advance_control_step=False,
            world_pose=self._world_pose(),
            **self._quality(),
        )
        self.observations = self._set_mask(self.observations, self.behavior_mask)
        self._append_feature(self.observations, self.behavior_mask)

    def _set_mask(self, observations, mask):
        out = observations.clone()
        out[:, self.stream.depth_end : self.stream.depth_end + 1] = mask[:, None]
        self.env.renet_estimator_mask.copy_(mask[:, None])
        return out

    def _availability(self, update=False):
        if not update:

            def available(health, count):
                return (
                    torch.isfinite(health.as_tensor()).all(-1)
                    & (health.frame_age <= self.runtime_cfg.sensor_timeout_s)
                    & (count < self.runtime_cfg.invalid_confirmations)
                )

            return available(self.stream.lp_health, self.lp_invalid_count), available(
                self.stream.vp_health, self.vp_invalid_count
            )
        helper = SelectorStateMachine._availability
        holder = type("Availability", (), {"cfg": self.runtime_cfg})()
        lp, lc = helper(holder, self.stream.lp_health, self.lp_invalid_count)
        vp, vc = helper(holder, self.stream.vp_health, self.vp_invalid_count)
        if update:
            self.lp_invalid_count.copy_(lc)
            self.vp_invalid_count.copy_(vc)
        return lp, vp

    def _append_feature(self, observations, mask):
        self.stream.query_health(self._time())
        branches = self.adapter.evaluate(observations)
        normalized = self.adapter.normalize(observations)
        end = self.adapter.policy.proprio_actor_dim
        frame = build_selector_frame(
            normalized[:, end - self.selector_cfg.proprio_dim : end],
            branches,
            self.stream.lp_health,
            self.stream.vp_health,
            mask,
            (self._time() - self.last_behavior_switch).clamp_min(0).float(),
            self.selector_cfg,
        )
        self.last_feature_time.fill_(self._time())
        self._availability(update=True)
        return self.feature_history.append(frame)

    def _capture(self):
        names = (
            "behavior_mask",
            "last_behavior_switch",
            "behavior_deadline",
            "last_feature_time",
            "episode_serial",
            "lp_invalid_count",
            "vp_invalid_count",
        )
        return CollectionSnapshot(
            self.snapshot_manager.capture(),
            self.stream.snapshot(),
            self.feature_history.snapshot(),
            self.observations.clone(),
            {n: getattr(self, n).clone() for n in names},
            self.blender.snapshot(),
            self.environment_rng.snapshot(),
            self.behavior_rng.get_state(),
        )

    def _restore(self, snapshot):
        self.snapshot_manager.restore(snapshot.environment)
        self.stream.restore(snapshot.observation_stream)
        self.feature_history.restore(snapshot.feature_history)
        self.observations = snapshot.observations.clone()
        for name, value in snapshot.tensors.items():
            getattr(self, name).copy_(value)
        self.blender.restore(snapshot.blender)
        self.environment_rng.restore(snapshot.environment_rng)
        self.behavior_rng.set_state(snapshot.behavior_rng)
        self.stream.query_health(self._time())

    def _accept_target(self, target):
        changed = target != self.behavior_mask
        lp, vp = self._availability()
        current_available = torch.where(self.behavior_mask > 0.5, lp, vp)
        target_available = torch.where(target > 0.5, lp, vp)
        emergency = changed & ~current_available & target_available
        self.last_behavior_switch[changed] = self._time()
        self.behavior_mask.copy_(target)
        return changed, emergency

    def _control_step(self, target, switched, emergency, diagnostics=None):
        branches = self.adapter.evaluate(self.observations)
        actions = self.blender.step(
            branches.lp_action, branches.vp_action, target, switched, emergency
        )
        self.observations = self._set_mask(self.observations, target)
        command = self.env.command_generator.command.clone()
        if diagnostics is not None:
            diagnostics["observations"] = self.observations.clone()
            diagnostics["actions"] = actions.clone()
            diagnostics["commands"] = command.clone()
        with capture_physics_substeps(self.env, diagnostics):
            result, metrics = step_with_safety_capture(self.env, actions, command)
        raw, _, dones, extras = result
        if diagnostics is not None:
            diagnostics["reset"] = dones.clone()
            diagnostics["applied_actions"] = self.env.action.clone()
            torque = getattr(self.env.robot.data, "applied_torque", None)
            if isinstance(torque, torch.Tensor):
                diagnostics["applied_torque"] = torque.clone()
        reset_ids = dones.nonzero().flatten()
        if len(reset_ids):
            self.stream.reset_envs(reset_ids)
            self.feature_history.reset(reset_ids)
            self.blender.reset(reset_ids)
            self.last_behavior_switch[reset_ids] = self._time()
            self.behavior_deadline[reset_ids] = self._time() + self._sample_dwell(
                reset_ids
            )
            self.lp_invalid_count[reset_ids] = 0
            self.vp_invalid_count[reset_ids] = 0
            self.episode_serial[reset_ids] += 1
        lp, vp = self._sensor_timestamps()
        self.observations, tick = self.stream.process(
            raw, lp_timestamp=lp, vp_timestamp=vp, now=self._time(),
            world_pose=self._world_pose(), **self._quality()
        )
        self.observations = self._set_mask(self.observations, target)
        if tick:
            self._append_feature(self.observations, target)
        if diagnostics is not None:
            diagnostics["next_observations"] = self.observations.clone()
        return metrics, dones, extras

    def _rollout(
        self, mask_value, *, preserve_as_behavior=False, record_trajectory=True,
        start_diagnostics=None,
    ):
        target = (
            torch.as_tensor(mask_value, device=self.device, dtype=torch.float32)
            .expand(self.env.num_envs)
            .clone()
        )
        self.stream.query_health(self._time())
        switched, emergency = self._accept_target(target)
        outcome = SafetyOutcomeAccumulator(
            self.env.num_envs,
            device=self.device,
            short_horizon_steps=self.collection_cfg.short_horizon_steps,
            long_horizon_steps=self.collection_cfg.long_horizon_steps,
        )
        trajectory = []
        for step in range(self.collection_cfg.long_horizon_steps):
            metrics, dones, extras = self._control_step(
                target,
                switched if step == 0 else torch.zeros_like(switched),
                emergency if step == 0 else torch.zeros_like(emergency),
                diagnostics=start_diagnostics if step == 0 else None,
            )
            outcome.update(metrics, dones, extras["time_outs"])
            if record_trajectory:
                trajectory.append(
                    torch.cat(
                        (
                            self.env.robot.data.root_state_w,
                            self.env.robot.data.joint_pos,
                            self.env.robot.data.joint_vel,
                        ),
                        -1,
                    ).clone()
                )
        return outcome.outcome(), torch.stack(
            trajectory
        ) if trajectory else torch.empty(0, device=self.device)

    def _repeated_branch_rollout(
        self, snapshot, mask_value, *, preserve_second_as_behavior=False
    ):
        name = "LP" if mask_value else "VP"
        self._restore(snapshot)
        self._progress(f"{name}_replay_1_of_2")
        first, _ = self._rollout(mask_value, record_trajectory=False)
        self._restore(snapshot)
        self._progress(f"{name}_replay_2_of_2")
        second, _ = self._rollout(mask_value, record_trajectory=False)
        second.first_failure_cause_stable = (
            first.first_failure_flags == second.first_failure_flags
        )
        second.replay_label_valid = first.label_valid & second.label_valid
        if not hasattr(self, "_last_replay_comparisons"):
            self._last_replay_comparisons = {}
        self._last_replay_comparisons["LP" if mask_value else "VP"] = (first, second)
        return second, outcome_replay_stability(first, second)

    def _record_replay_rejections(self, eligible, written=None):
        """Account for the joint LP/VP filter without blaming the wrong branch."""
        audit = self.writer.metadata.setdefault("per_candidate_replay_diagnostics", {
            "version": 2,
            "filter_contract": replay_filter_contract(),
            "eligible_scope": "same rows as per_candidate_replay_filter.checked, before outcome-validity filtering",
            "retained_scope": "eligible rows passing the new replay gate, before scenario quotas; written counters apply quotas",
            "field_counts_overlap": True,
            "diagnostic_label_note": "second-replay labels classify rejected rows; they are not reliable ground truth",
            "by_branch": {"LP": {}, "VP": {}},
            "joint": {},
            "by_second_replay_1s": {"both_safe": {}, "any_failure": {}},
            "examples": [], "examples_limit": 64,
            "retained_examples": [], "retained_examples_limit": 64,
        })
        comparisons = getattr(self, "_last_replay_comparisons", {})
        if set(comparisons) != {"LP", "VP"}:
            # Alternate/test rollout implementations may provide stability
            # masks without retaining both outcomes. Do not fabricate reasons.
            audit["unavailable_batches"] = audit.get("unavailable_batches", 0) + 1
            return
        masks = {branch: replay_mismatch_masks(*pair)
                 for branch, pair in comparisons.items()}
        unstable = {branch: ~outcome_replay_stability(*pair)
                    for branch, pair in comparisons.items()}
        legacy_unstable = {branch: ~legacy_outcome_replay_stability(*pair)
                           for branch, pair in comparisons.items()}
        reasons = {branch: replay_rejection_masks(*pair) for branch, pair in comparisons.items()}
        for branch, pair in comparisons.items():
            _add_replay_counts(audit["by_branch"][branch], replay_disagreement_summary(*pair, eligible))
        lp, vp = unstable["LP"], unstable["VP"]
        rejected = lp | vp
        legacy_rejected = legacy_unstable["LP"] | legacy_unstable["VP"]
        audit_difference = torch.stack([value for fields in masks.values()
                                        for value in fields.values()]).any(dim=0)
        label_flip = reasons["LP"]["label_flip"] | reasons["VP"]["label_flip"]
        label_invalid = torch.stack([~outcome.label_valid for pair in comparisons.values()
                                     for outcome in pair]).any(dim=0)
        written = torch.zeros_like(eligible) if written is None else written

        def count(mask):
            return int((eligible & mask).sum().item())

        _add_replay_counts(audit["joint"], {
            "checked": int(eligible.sum().item()), "rejected": count(rejected),
            "lp_only_mismatch": count(lp & ~vp),
            "vp_only_mismatch": count(vp & ~lp),
            "both_branch_mismatch": count(lp & vp),
            "label_flip": count(label_flip),
            "label_invalid": count(label_invalid),
            "labels_agree_but_rejected": count(rejected & ~label_flip),
            "legacy_rejected": count(legacy_rejected),
            "retained_despite_legacy_difference": count(~rejected & legacy_rejected),
            "retained_with_audit_difference": count(~rejected & audit_difference),
            "written": count(written),
            "written_retained_despite_legacy_difference": count(written & ~rejected & legacy_rejected),
            "written_retained_with_audit_difference": count(written & ~rejected & audit_difference),
        })
        any_failure = comparisons["LP"][1].fail_long | comparisons["VP"][1].fail_long
        for name, group in (("both_safe", ~any_failure), ("any_failure", any_failure)):
            _add_replay_counts(audit["by_second_replay_1s"][name], {
                "checked": count(group), "rejected": count(group & rejected),
                "label_flip": count(group & label_flip),
                "label_invalid": count(group & label_invalid),
                "labels_agree_but_rejected": count(group & rejected & ~label_flip),
                "legacy_rejected": count(group & legacy_rejected),
                "retained_despite_legacy_difference": count(group & ~rejected & legacy_rejected),
                "written": count(group & written),
                "written_retained_despite_legacy_difference": count(group & written & ~rejected & legacy_rejected),
            })
        for key, selection in (("examples", rejected), ("retained_examples", ~rejected & audit_difference)):
            room = audit[f"{key}_limit"] - len(audit[key])
            for env_id in (eligible & selection).nonzero().flatten()[:max(0, room)].tolist():
                row = {"candidate_batch": self.batch_index, "env_id": env_id, "branches": {},
                       "accepted_by_replay_filter": not bool(rejected[env_id]),
                       "legacy_filter_accepted": not bool(legacy_rejected[env_id]),
                       "written": bool(written[env_id])}
                for branch, pair in comparisons.items():
                    row["branches"][branch] = {
                        "mismatched_fields": [name for name, mask in masks[branch].items() if bool(mask[env_id])],
                        "rejection_reasons": [name for name, mask in reasons[branch].items() if bool(mask[env_id])],
                        "values": {name: [getattr(outcome, name)[env_id].item() for outcome in pair]
                                   for name in REPLAY_AUDIT_FIELDS},
                    }
                audit[key].append(row)

    def _advance_behavior(self):
        now = self._time()
        target = self.behavior_mask.clone()
        expired = now >= self.behavior_deadline
        target[expired] = 1.0 - target[expired]
        ids = expired.nonzero().flatten()
        self.behavior_deadline[ids] = now + self._sample_dwell(ids)
        lp, vp = self._availability()
        target = torch.where(
            (self.behavior_mask > 0.5) & ~lp & vp, torch.zeros_like(target), target
        )
        target = torch.where(
            (self.behavior_mask <= 0.5) & ~vp & lp, torch.ones_like(target), target
        )
        changed, emergency = self._accept_target(target)
        early = (changed & ~expired).nonzero().flatten()
        self.behavior_deadline[early] = now + self._sample_dwell(early)
        for step in range(self.collection_cfg.candidate_stride_steps):
            self._control_step(
                target,
                changed if step == 0 else torch.zeros_like(changed),
                emergency if step == 0 else torch.zeros_like(emergency),
            )
            audit = getattr(self, "_collection_audit", None)
            if audit is not None:
                audit.observe_behavior_plans(self.episode_serial, self.stream.faults)

    def finalize_audit(self):
        audit = getattr(self, "_collection_audit", None)
        if audit is not None:
            self.writer.metadata["collection_audit"] = audit.summary()
        if getattr(self, "quota_sampler", None) is not None and self.writer is not None:
            self.writer.metadata["supplement_coverage"] = self.quota_sampler.summary()
            self.writer.metadata["sampling_progress"] = dict(self.sampling_progress)

    @torch.no_grad()
    def validate_replay(
        self, steps: int = 10, atol: float = 1.0e-4, *, mask_value: float | None = None
    ) -> dict[str, Any]:
        if steps <= 0:
            raise ValueError("replay trajectory prefix must contain at least one step")
        if self.observations is None:
            self.initialize()
        snapshot = self._capture()
        try:
            return self._validate_replay_snapshot(snapshot, steps, atol, mask_value)
        finally:
            self._restore(snapshot)

    def _validate_replay_snapshot(self, snapshot, steps, atol, mask_value):
        branch = float(self.behavior_mask[0].item()) if mask_value is None else float(mask_value)
        if branch not in (0.0, 1.0):
            raise ValueError("replay branch must be LP=1.0 or VP=0.0")
        # Replay validation uses the normal rollout path and compares only the
        # requested prefix; the full 1 s rollout remains the label contract.
        # Both repetitions must start after restoration, just like persisted
        # candidates. Comparing a live run with a restored run tests a different
        # path and can confound the startup result with restore-side effects.
        self._restore(snapshot)
        first_initial_error = self._initial_restore_error(snapshot)
        first_start = {}
        first_outcome, first = self._rollout(
            branch, preserve_as_behavior=False, start_diagnostics=first_start
        )
        self._restore(snapshot)
        second_initial_error = self._initial_restore_error(snapshot)
        initial_restore_error = max(first_initial_error, second_initial_error)
        second_start = {}
        second_outcome, second = self._rollout(
            branch, preserve_as_behavior=False, start_diagnostics=second_start
        )
        first, second = first[:steps], second[:steps]
        error = (first - second).abs()
        compared_steps = first.shape[0]
        no_reset = torch.full_like(first_outcome.first_reset_step, compared_steps + 1)
        first_reset = torch.where(
            first_outcome.first_reset_step > 0,
            first_outcome.first_reset_step,
            no_reset,
        )
        second_reset = torch.where(
            second_outcome.first_reset_step > 0,
            second_outcome.first_reset_step,
            no_reset,
        )
        # The environment auto-resets terminated robots inside env.step().
        # States from the newly sampled episode are irrelevant to the original
        # candidate label and must not contaminate replay equivalence.
        cutoff = torch.minimum(first_reset, second_reset).clamp_max(compared_steps + 1)
        time = torch.arange(compared_steps, device=self.device)[:, None]
        pre_failure = time < (cutoff[None] - 1)
        compared_error = error[pre_failure]
        max_error = (
            float(compared_error.max().item()) if compared_error.numel() else 0.0
        )
        rmse = (
            float(torch.sqrt(compared_error.square().mean()).item())
            if compared_error.numel()
            else 0.0
        )
        labels_equal_per_env = (
            first_outcome.fail_short == second_outcome.fail_short
        ) & (first_outcome.fail_long == second_outcome.fail_long)
        failure_step_equal_per_env = (
            first_outcome.first_failure_step == second_outcome.first_failure_step
        )
        reset_step_equal_per_env = (
            first_outcome.first_reset_step == second_outcome.first_reset_step
        )
        validity_equal_per_env = first_outcome.label_valid == second_outcome.label_valid
        masked_error = torch.where(
            pre_failure[..., None], error, torch.zeros_like(error)
        )
        per_env_trajectory_max = masked_error.amax(dim=(0, 2))
        per_step_max = masked_error.amax(dim=(1, 2))
        divergent = torch.nonzero(per_step_max > atol, as_tuple=False).flatten()
        first_divergent_step = int(divergent[0].item() + 1) if len(divergent) else -1
        checks = {
            "initial_restore": initial_restore_error <= 1.0e-6,
            "label_agreement": float(labels_equal_per_env.float().mean().item()) >= 0.999,
            "label_validity_agreement": float(validity_equal_per_env.float().mean().item()) >= 0.999,
        }
        timing_checks = {
            "failure_step_agreement": float(failure_step_equal_per_env.float().mean().item()) >= 0.995,
            "reset_step_agreement": float(reset_step_equal_per_env.float().mean().item()) >= 0.995,
        }
        stable = outcome_replay_stability(first_outcome, second_outcome)
        audit_mismatch = replay_mismatch_masks(first_outcome, second_outcome)
        rejection_masks = replay_rejection_masks(first_outcome, second_outcome)
        audit_difference = torch.stack(list(audit_mismatch.values())).any(dim=0)

        def details(selected):
            rows = []
            for env_id in selected.nonzero().flatten()[:64].tolist():
                row = {"env_id": env_id,
                       "mismatched_fields": [name for name, mask in audit_mismatch.items() if bool(mask[env_id])],
                       "rejection_reasons": [name for name, mask in rejection_masks.items() if bool(mask[env_id])]}
                for name in REPLAY_AUDIT_FIELDS:
                    row[name] = [getattr(outcome, name)[env_id].item()
                                 for outcome in (first_outcome, second_outcome)]
                rows.append(row)
            return rows
        first_step_errors = {}
        for name in ("observations", "actions", "commands", "applied_actions",
                     "applied_torque", "next_observations"):
            if name not in first_start or name not in second_start:
                first_step_errors[name] = None
                continue
            delta = (first_start[name] - second_start[name]).abs()
            # Returned observations and torques may belong to a new episode.
            # Keep their raw maximum, but compare original-episode rows separately.
            live = ~(first_start["reset"] | second_start["reset"])
            masked = delta[live]
            first_step_errors[name] = {
                "max_abs_error": float(delta.max().item()),
                "no_reset_max_abs_error": float(masked.max().item()) if masked.numel() else None,
            }
        result = {
            # Initial-state and label thresholds remain hard checks; event
            # timing and continuous trajectories are explicit audit results.
            "passed": all(checks.values()),
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "audit_checks": timing_checks,
            "replay_filter_contract": replay_filter_contract(),
            "branch": "LP" if branch else "VP",
            "comparison": "restored_vs_restored",
            "snapshot_restore_contract": snapshot_restore_contract(),
            "physics_settings": collection_physics_settings(self.env),
            "first_initial_restore_max_abs_error": first_initial_error,
            "second_initial_restore_max_abs_error": second_initial_error,
            "first_control_step_errors": first_step_errors,
            "physics_substep_comparison": physics_substep_comparison(first_start, second_start, atol),
            "rejection_reasons": replay_disagreement_summary(first_outcome, second_outcome),
            "actuator_models": {
                name: {"class": f"{type(actuator).__module__}.{type(actuator).__name__}",
                       "implicit": bool(getattr(actuator, "is_implicit_model", False))}
                for name, actuator in getattr(self.env.robot, "actuators", {}).items()
            },
            "candidate_stable_count": int(stable.sum().item()),
            "candidate_mismatch_count": int((~stable).sum().item()),
            "mismatch_details": details(~stable),
            "retained_audit_difference_details": details(stable & audit_difference),
            "audit_mismatch_count": int(audit_difference.sum().item()),
            "mismatch_details_limit": 64,
            "trajectory_within_atol": bool(max_error <= atol),
            "trajectory_env_agreement": float(
                (per_env_trajectory_max <= atol).float().mean().item()
            ),
            "trajectory_env_mismatch_count": int(
                (per_env_trajectory_max > atol).sum().item()
            ),
            "max_abs_error": max_error,
            "rmse": rmse,
            "post_reset_max_abs_error": float(error.max().item()),
            "initial_restore_max_abs_error": initial_restore_error,
            "first_divergent_step": first_divergent_step,
            "root_state_max_abs_error": float(masked_error[..., :13].max().item()),
            "joint_position_max_abs_error": float(
                masked_error[..., 13:36].max().item()
            ),
            "joint_velocity_max_abs_error": float(masked_error[..., 36:].max().item()),
            "label_agreement": float(labels_equal_per_env.float().mean().item()),
            "label_mismatch_count": int((~labels_equal_per_env).sum().item()),
            "failure_step_agreement": float(
                failure_step_equal_per_env.float().mean().item()
            ),
            "failure_step_mismatch_count": int(
                (~failure_step_equal_per_env).sum().item()
            ),
            "reset_step_agreement": float(
                reset_step_equal_per_env.float().mean().item()
            ),
            "reset_step_mismatch_count": int((~reset_step_equal_per_env).sum().item()),
            "label_validity_agreement": float(
                validity_equal_per_env.float().mean().item()
            ),
            "first_replay_valid_label_count": int(
                first_outcome.label_valid.sum().item()
            ),
            "second_replay_valid_label_count": int(
                second_outcome.label_valid.sum().item()
            ),
            "first_replay_fail_short_count": int(first_outcome.fail_short.sum().item()),
            "second_replay_fail_short_count": int(
                second_outcome.fail_short.sum().item()
            ),
            "first_replay_fail_long_count": int(first_outcome.fail_long.sum().item()),
            "second_replay_fail_long_count": int(second_outcome.fail_long.sum().item()),
            "atol": float(atol),
        }
        return result

    def _initial_restore_error(self, snapshot):
        restored = torch.cat((self.env.robot.data.root_state_w,
                              self.env.robot.data.joint_pos,
                              self.env.robot.data.joint_vel), dim=-1)
        expected = torch.cat((snapshot.environment.root_state,
                              snapshot.environment.joint_position,
                              snapshot.environment.joint_velocity), dim=-1)
        error = (restored - expected).abs()
        return float(error.max().item()) if torch.isfinite(error).all() else float("inf")

    def _terrain_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        terrain = getattr(self.env.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None)
        types = getattr(terrain, "terrain_types", None)
        if levels is None:
            levels = torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.device
            )
        if types is None:
            types = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        return levels.clone(), types.clone()

    @torch.no_grad()
    def collect(self):
        if not hasattr(self, "sampling_progress"):
            self.sampling_progress = {"real_candidate_ticks": 0, "labelled_batches": 0,
                                      "paced_or_interval_skips": 0, "no_observable_rows": 0}
        self._progress("initialize_observations")
        if self.observations is None:
            self.initialize()
        # Keep the global startup probe at the original initial state. Faulted
        # warmup visits failure boundaries and auto-resets, which can change a
        # few discrete labels and must not redefine this startup check. Each
        # actual candidate below still repeats BOTH branches and rejects any
        # inconsistent or invalid label after warmup.
        self._progress("validate_replay")
        replay = self.validate_replay()
        metadata = self.writer.metadata
        metadata["collection_complete"] = False
        metadata["replay_filter_contract"] = replay_filter_contract()
        metadata["snapshot_replay_validation"] = replay
        metadata["snapshot_replay_validation_scope"] = {
            "phase": "before_real_warmup", "simulation_time_s": self._time(),
        }
        if not replay["passed"]:
            raise RuntimeError("paired rollout replay check failed: "
                f"checks={replay.get('failed_checks')}, "
                f"label_mismatches={replay.get('label_mismatch_count')}, "
                f"label_agreement={replay.get('label_agreement')}; "
                "full diagnostics are saved in manifest metadata.snapshot_replay_validation")
        metadata["safety_label_contract"] = safety_label_contract()
        metadata["selector_feature_contract"] = selector_feature_contract()
        self._warmup()
        metadata["collection_config"] = asdict(self.collection_cfg)
        metadata["collection_audit_contract"] = collection_audit_contract()
        if getattr(self.stream.faults.cfg, "extended_faults", False):
            metadata["noise_audit_contract"] = {"version": 1, "order": list(NOISE_AUDIT_ORDER),
                "scope": "audit only, excluded from selector features",
                "person_phase_order": ["approach", "near", "depart", "residual"],
                "fault_event": "parent event includes bursts, gaps, corrections and residual cells"}
        self._collection_audit = CollectionAudit(self.env.num_envs, self.device)
        self._collection_audit.observe_behavior_plans(self.episode_serial, self.stream.faults)
        filters = {
            "insufficient_episode_horizon": 0,
            "invalid_lp_outcome": 0,
            "invalid_vp_outcome": 0,
            "unobservable_future_fault": 0,
            "outside_source_partition": 0,
            "initially_unsafe_or_invalid": 0,
        }
        metadata["safety_label_filter"] = filters
        replay_filter = {"checked": 0, "rejected": 0, "agreement": 1.0}
        metadata["per_candidate_replay_filter"] = replay_filter
        weights = self.stream.faults.cfg.scenario_probabilities
        quotas = [int(self.collection_cfg.target_samples * w) for w in weights]
        quotas[0] += self.collection_cfg.target_samples - sum(quotas)
        accepted = [0] * 4
        seen = [0] * 4
        rejected = [0] * 4
        metadata["coverage"] = {
            "scenario_order": ["normal", "vp_only", "lp_only", "dual"],
            "quotas": quotas if self.collection_cfg.balanced_scenarios and getattr(self, "quota_sampler", None) is None else None,
            "seen": seen,
            "accepted": accepted,
            "replay_rejected": rejected,
            "label_positive": [0] * 4,
            "behavior_mode": [0, 0],
            "dwell_bins_s": ["[0,.4)", "[.4,1)", "[1,2)", "[2,inf)"],
            "dwell_counts": [0] * 4,
            "phase_counts": {"before": 0, "active": 0, "recovered": 0},
        }
        collected = 0
        labelled_batches = 0
        last_progress_batch = 0
        last_log_time = time.monotonic()
        sampler = getattr(self, "quota_sampler", None)
        if sampler is not None:
            sampler.start(self._time())
        self._progress("collect_candidates", accepted=0, target=self.collection_cfg.target_samples)
        while collected < self.collection_cfg.target_samples:
            if self.batch_index >= self.collection_cfg.max_candidate_batches:
                metadata["collection_stop_reason"] = "candidate_budget_exhausted"
                self.finalize_audit()
                self.writer.close()
                raise RuntimeError(
                    f"candidate limit reached ({collected} accepted); inspect manifest coverage and use a new output directory"
                )
            supplement = getattr(self, "supplement_cfg", None)
            if supplement is not None and labelled_batches-last_progress_batch >= supplement.max_no_progress_batches:
                metadata["collection_stop_reason"] = "supplement_quota_no_progress"
                self.finalize_audit()
                self.writer.close()
                raise RuntimeError(f"supplement made no progress for {supplement.max_no_progress_batches} labelled batches; "
                    "inspect supplement_coverage missing quotas and eligible outcomes before recollecting")
            now = self._time()
            if sampler is not None:
                sampler.observe_time(now)
                # Features and fault processes still advance at 10 Hz. Only
                # label forks are spaced out, so burn-in/pacing adds real time
                # without paying for four 1 s forks at every skipped tick.
                if not sampler.should_label_tick(self.batch_index) or sampler.sample_budget(now) == 0:
                    self.sampling_progress["paced_or_interval_skips"] += 1
                    self.batch_index += 1
                    self.sampling_progress["real_candidate_ticks"] = self.batch_index
                    self._progress("advance_real_behavior", accepted=collected,
                                   real_span_s=sampler.last_time-sampler.collection_start)
                    self._advance_behavior()
                    continue
            if not torch.allclose(
                self.last_feature_time,
                torch.full_like(self.last_feature_time, now),
                atol=1e-6,
                rtol=0,
            ):
                raise RuntimeError(
                    "candidate snapshot does not align with last feature frame"
                )
            history = self.feature_history.buffer.clone()
            level, terrain = self._terrain_metadata()
            faults = self.stream.faults.audit_metadata()
            parameters = self.stream.faults.audit_parameters()
            noise_metadata = (self.stream.faults.audit_map_noise()
                              if getattr(self.stream.faults.cfg, "extended_faults", False) else None)
            event_identity = torch.stack((
                torch.arange(self.env.num_envs, device=self.device),
                self.episode_serial, self.stream.faults.schedule_count,
            ), -1).clone()
            initial = capture_safety_step(self.env, self.env.command_generator.command)
            # Invalid raw measurements remain explicitly marked; JSON/reporting
            # uses the validity column instead of interpreting replacement zeros.
            candidate_safety = torch.stack((
                initial.tilt, initial.clearance, initial.torso_contact.float(),
                initial.valid.float(),
            ), -1)
            candidate_safety = torch.nan_to_num(candidate_safety, nan=0.0, posinf=0.0, neginf=0.0)
            scenario = torch.where(
                faults[:, 6].bool(),
                torch.where(faults[:, 7].bool(), 3, 2),
                torch.where(faults[:, 7].bool(), 1, 0),
            )
            future_fault = self.stream.faults.unobservable_onset_within(10)
            filters["unobservable_future_fault"] += int(
                (self.feature_history.initialized & future_fault).sum()
            )
            history_ready = self.feature_history.initialized.clone()
            observable = history_ready & ~future_fault
            valid = observable.clone()
            enough = (
                self.env.max_episode_length - self.env.episode_length_buf
                >= self.collection_cfg.long_horizon_steps
            )
            filters["insufficient_episode_horizon"] += int((valid & ~enough).sum())
            valid &= enough
            group_seed = getattr(self, "terrain_seed", self.seed)
            all_groups = [f"terrain_seed{group_seed}:type{int(t)}:level{int(l)}"
                          for t,l in zip(terrain.tolist(), level.tolist())]
            supplement = getattr(self, "supplement_cfg", None)
            if supplement is not None:
                allowed = torch.tensor([g in self.allowed_groups for g in all_groups], device=self.device)
                filters["outside_source_partition"] += int((valid & ~allowed).sum())
                valid &= allowed
                if supplement.require_initial_safe:
                    safe = initial.valid & (initial.tilt <= TILT_LIMIT_RAD) & (initial.clearance >= CLEARANCE_LIMIT_M) & ~initial.torso_contact
                    filters["initially_unsafe_or_invalid"] += int((valid & ~safe).sum())
                    valid &= safe
            labelled_batches += 1
            self.sampling_progress["labelled_batches"] = labelled_batches
            if sampler is not None and not valid.any():
                # No observable, safe start state: avoid running four forks
                # whose outcomes cannot be written, but retain funnel counts.
                sampler.record_unlabelled(faults, parameters, noise_metadata)
                self.sampling_progress["no_observable_rows"] += 1
                zeros = torch.zeros_like(valid)
                self._collection_audit.record_candidates(faults,
                    torch.zeros((self.env.num_envs, 4), device=self.device, dtype=torch.bool),
                    (history_ready, observable, valid, zeros, zeros), zeros)
                self.batch_index += 1
                self.sampling_progress["real_candidate_ticks"] = self.batch_index
                self._progress("no_observable_candidates", accepted=collected,
                               labelled_batches=labelled_batches)
                self._advance_behavior()
                continue
            snapshot = self._capture()
            lp, ls = self._repeated_branch_rollout(snapshot, 1.0)
            vp, vs = self._repeated_branch_rollout(snapshot, 0.0)
            self._restore(snapshot)
            stable = ls & vs
            replay_filter["checked"] += int(valid.sum())
            replay_filter["rejected"] += int((valid & ~stable).sum())
            replay_filter["agreement"] = 1.0 - replay_filter["rejected"] / max(
                1, replay_filter["checked"]
            )
            lp_valid = getattr(lp, "replay_label_valid", lp.label_valid)
            vp_valid = getattr(vp, "replay_label_valid", vp.label_valid)
            filters["invalid_lp_outcome"] += int((valid & ~lp_valid).sum())
            filters["invalid_vp_outcome"] += int((valid & ~vp_valid).sum())
            good = valid & stable & lp_valid & vp_valid
            labels = torch.stack(
                (lp.fail_short, lp.fail_long, vp.fail_short, vp.fail_long), -1
            )
            selected = []
            for category in range(4):
                mask = scenario == category
                seen[category] += int((valid & mask).sum())
                rejected[category] += int((valid & mask & ~stable).sum())
                if getattr(self, "quota_sampler", None) is not None:
                    continue
                ids = (good & mask).nonzero().flatten()
                room = (
                    quotas[category] - accepted[category]
                    if self.collection_cfg.balanced_scenarios
                    else self.collection_cfg.target_samples
                    - collected
                    - sum(len(v) for v in selected)
                )
                ids = ids[: max(0, room)]
                selected.append(ids)
                accepted[category] += len(ids)
            if getattr(self, "quota_sampler", None) is not None:
                ids = self.quota_sampler.select(good, labels, event_identity, now,
                    faults=faults, parameters=parameters, noise=noise_metadata)
                for category in range(4):
                    accepted[category] += int((scenario[ids] == category).sum())
            else:
                ids = torch.cat(selected)
            if len(ids):
                groups = [all_groups[i] for i in ids.tolist()]
                dwell = (now - self.last_behavior_switch).float()
                execution = torch.stack(
                    (self.behavior_mask, dwell, self.blender.remaining.float()), -1
                )
                batch = SelectorBatch(
                    history[ids],
                    labels[ids],
                    groups,
                    level[ids],
                    terrain[ids],
                    (self.candidate_prefix * 1_000_000_000 if getattr(self, "candidate_prefix", None) is not None
                     else self.seed * 100_000_000)
                    + self.batch_index * self.env.num_envs
                    + ids,
                    torch.stack((lp.metrics(), vp.metrics()), 1)[ids],
                    faults[ids],
                    torch.full(
                        (len(ids),), now, device=self.device, dtype=torch.float64
                    ),
                    self.last_feature_time[ids].clone(),
                    execution[ids],
                    event_identity=event_identity[ids],
                    fault_parameters=parameters[ids],
                    first_failure_flags=torch.stack(
                        (lp.first_failure_flags, vp.first_failure_flags), -1
                    )[ids],
                    first_failure_cause_stable=torch.stack(
                        (lp.first_failure_cause_stable, vp.first_failure_cause_stable), -1
                    )[ids],
                    candidate_safety=candidate_safety[ids],
                    noise_metadata=noise_metadata[ids] if noise_metadata is not None else None,
                )
                self._progress("write_samples", rows=len(ids), accepted=collected)
                self.writer.append(batch)
                collected += len(ids)
                last_progress_batch = labelled_batches
                coverage = metadata["coverage"]
                for j, count in enumerate(labels[ids].sum(0).tolist()):
                    coverage["label_positive"][j] += int(count)
                for j in (0, 1):
                    coverage["behavior_mode"][j] += int(
                        (self.behavior_mask[ids] == j).sum()
                    )
                bins = torch.bucketize(
                    dwell[ids],
                    torch.tensor([0.4, 1.0, 2.0], device=self.device),
                    right=True,
                )
                for j in range(4):
                    coverage["dwell_counts"][j] += int((bins == j).sum())
                for value, name in ((-1, "before"), (0, "active"), (1, "recovered")):
                    coverage["phase_counts"][name] += int(
                        (faults[ids, 3] == value).sum()
                    )
            written = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            written[ids] = True
            self._record_replay_rejections(valid, written)
            self._collection_audit.record_candidates(
                faults, labels,
                (history_ready, observable, valid, valid & lp_valid & vp_valid, good),
                written,
            )
            self.batch_index += 1
            self.sampling_progress["real_candidate_ticks"] = self.batch_index
            self._progress("collect_candidates", accepted=collected,
                           labelled_batches=labelled_batches, target=self.collection_cfg.target_samples)
            if labelled_batches == 1 or labelled_batches % 10 == 0 or time.monotonic()-last_log_time >= 30 or collected == self.collection_cfg.target_samples:
                last_log_time = time.monotonic()
                self.finalize_audit()
                print(
                    f"[SelectorCollector] ticks={self.batch_index} labelled_batches={labelled_batches} accepted={collected}/{self.collection_cfg.target_samples} scenarios={accepted}",
                    flush=True,
                )
                if getattr(self, "quota_sampler", None) is not None:
                    summary = self.quota_sampler.summary()
                    print(f"[SelectorSupplement] long={summary['accepted_long']} missing={summary['missing_long']} "
                          f"short_vp_only={summary['reserved_short_vp_only']} late={summary['late_failure']} "
                          f"events={summary['independent_events']}", flush=True)
                    temporal = summary["temporal_coverage"]
                    stages = dict(zip(temporal["lifecycle_order"], temporal["written"]))
                    print(f"[SelectorTemporal] real_span_s={temporal['collection_elapsed_s']:.2f} "
                          f"drift_mature={stages['drift_mature']} person_near={stages['person_near']} "
                          f"person_depart={stages['person_depart']} residual={stages['person_residual']} "
                          f"recovered={stages['recovered']}", flush=True)
            if collected < self.collection_cfg.target_samples:
                self._advance_behavior()
        self.finalize_audit()
        if getattr(self, "quota_sampler", None) is not None and not self.quota_sampler.summary()["complete"]:
            raise RuntimeError("supplement target reached without secondary coverage; inspect supplement_coverage")
        metadata["collection_complete"] = True
        self._progress("collection_finished", accepted=collected,
                       labelled_batches=labelled_batches)
