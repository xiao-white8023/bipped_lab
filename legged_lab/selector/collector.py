"""Aligned 10 Hz candidates and isolated 1 s counterfactual label forks."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Any
import torch
from .config import CollectionConfig, SelectorConfig, RuntimeConfig
from .dataset import SelectorBatch, SelectorDatasetWriter
from .faults import FaultSnapshot, SelectorFaultManager
from .features import FeatureHistory, SensorHealthTracker, build_selector_frame
from .feature_contract import selector_feature_contract
from .labels import (
    RolloutOutcome,
    SafetyOutcomeAccumulator,
    safety_label_contract,
    step_with_safety_capture,
)
from .policy_adapter import FrozenRENetAdapter
from .snapshot import RENetSnapshot, RENetSnapshotManager
from .runtime import ActionBlender, SelectorStateMachine
from .randomness import (
    PerEnvironmentRNG,
    EnvironmentRandomness,
    bind_environment_randomness,
)


def outcome_replay_stability(
    first: RolloutOutcome, second: RolloutOutcome
) -> torch.Tensor:
    """Return environments whose discrete safety outcome is replay-stable.

    Continuous states are intentionally excluded: PhysX GPU contact caches are
    not part of the public snapshot API.  Persisted candidates must still agree
    exactly on both labels and on the first failure/reset timing.
    """

    return (
        (first.fail_short == second.fail_short)
        & (first.fail_long == second.fail_long)
        & (first.first_failure_step == second.first_failure_step)
        & (first.first_reset_step == second.first_reset_step)
        & (first.terminated == second.terminated)
        & (first.torso_contact == second.torso_contact)
        & (first.label_valid == second.label_valid)
        & (first.first_timeout_step == second.first_timeout_step)
        & (first.first_invalid_step == second.first_invalid_step)
        & (first.observed_steps == second.observed_steps)
    )


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
        self.snapshot_manager = RENetSnapshotManager(env)
        self.stream = FaultedObservationStream(adapter, faults)
        self.feature_history = FeatureHistory(
            env.num_envs, self.selector_cfg, self.device
        )
        self.environment_rng = EnvironmentRandomness(
            env.num_envs, seed + 3000, self.device
        )
        bind_environment_randomness(env, self.environment_rng)
        self.behavior_rng = PerEnvironmentRNG(env.num_envs, seed + 4000, self.device)
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

    def _control_step(self, target, switched, emergency):
        branches = self.adapter.evaluate(self.observations)
        actions = self.blender.step(
            branches.lp_action, branches.vp_action, target, switched, emergency
        )
        self.observations = self._set_mask(self.observations, target)
        command = self.env.command_generator.command.clone()
        result, metrics = step_with_safety_capture(self.env, actions, command)
        raw, _, dones, extras = result
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
            raw, lp_timestamp=lp, vp_timestamp=vp, now=self._time(), **self._quality()
        )
        self.observations = self._set_mask(self.observations, target)
        if tick:
            self._append_feature(self.observations, target)
        return metrics, dones, extras

    def _rollout(
        self, mask_value, *, preserve_as_behavior=False, record_trajectory=True
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
        self._restore(snapshot)
        first, _ = self._rollout(mask_value, record_trajectory=False)
        self._restore(snapshot)
        second, _ = self._rollout(mask_value, record_trajectory=False)
        return second, outcome_replay_stability(first, second)

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

    @torch.no_grad()
    def validate_replay(
        self, steps: int = 10, atol: float = 1.0e-4
    ) -> dict[str, float | bool | int]:
        if self.observations is None:
            self.initialize()
        snapshot = self._capture()
        # Replay validation uses the normal rollout path and compares only the
        # requested prefix; the full 1 s rollout remains the label contract.
        first_outcome, first = self._rollout(
            float(self.behavior_mask[0].item()), preserve_as_behavior=False
        )
        self._restore(snapshot)
        restored_initial = torch.cat(
            (
                self.env.robot.data.root_state_w,
                self.env.robot.data.joint_pos,
                self.env.robot.data.joint_vel,
            ),
            dim=-1,
        )
        expected_initial = torch.cat(
            (
                snapshot.environment.root_state,
                snapshot.environment.joint_position,
                snapshot.environment.joint_velocity,
            ),
            dim=-1,
        )
        initial_restore_error = float(
            (restored_initial - expected_initial).abs().max().item()
        )
        second_outcome, second = self._rollout(
            float(self.behavior_mask[0].item()), preserve_as_behavior=False
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
        result = {
            # PhysX does not expose its contact warm-start cache through the
            # public tensor API, so future floating-point trajectories need
            # not be bit-identical after a rewind.  The approved acceptance
            # contract is exact initial-state restoration plus a high raw
            # agreement rate; exactness is enforced per persisted candidate.
            # The startup probe diagnoses the raw PhysX rewind quality.  A
            # tiny fraction of marginal 4096-way GPU simulations can cross a
            # discrete boundary because contact warm-start caches are not
            # restorable.  Collection therefore duplicates every LP and VP
            # branch and rejects every such candidate before persistence.
            "passed": bool(
                initial_restore_error <= 1.0e-6
                and float(labels_equal_per_env.float().mean().item()) >= 0.999
                and float(failure_step_equal_per_env.float().mean().item()) >= 0.995
                and float(reset_step_equal_per_env.float().mean().item()) >= 0.995
                and float(validity_equal_per_env.float().mean().item()) >= 0.999
            ),
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
        self._restore(snapshot)
        return result

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
        if self.observations is None:
            self.initialize()
        replay = self.validate_replay()
        if not replay["passed"]:
            raise RuntimeError(f"paired rollout replay check failed: {replay}")
        metadata = self.writer.metadata
        metadata["collection_complete"] = False
        metadata["snapshot_replay_validation"] = replay
        metadata["safety_label_contract"] = safety_label_contract()
        metadata["selector_feature_contract"] = selector_feature_contract()
        metadata["collection_config"] = asdict(self.collection_cfg)
        filters = {
            "insufficient_episode_horizon": 0,
            "invalid_lp_outcome": 0,
            "invalid_vp_outcome": 0,
            "unobservable_future_fault": 0,
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
            "quotas": quotas if self.collection_cfg.balanced_scenarios else None,
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
        while collected < self.collection_cfg.target_samples:
            if self.batch_index >= self.collection_cfg.max_candidate_batches:
                self.writer.close()
                raise RuntimeError(
                    f"candidate limit reached ({collected} accepted); inspect manifest coverage and use a new output directory"
                )
            now = self._time()
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
            scenario = torch.where(
                faults[:, 6].bool(),
                torch.where(faults[:, 7].bool(), 3, 2),
                torch.where(faults[:, 7].bool(), 1, 0),
            )
            future_fault = self.stream.faults.unobservable_onset_within(10)
            filters["unobservable_future_fault"] += int(
                (self.feature_history.initialized & future_fault).sum()
            )
            valid = self.feature_history.initialized & ~future_fault
            enough = (
                self.env.max_episode_length - self.env.episode_length_buf
                >= self.collection_cfg.long_horizon_steps
            )
            filters["insufficient_episode_horizon"] += int((valid & ~enough).sum())
            valid &= enough
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
            filters["invalid_lp_outcome"] += int((valid & ~lp.label_valid).sum())
            filters["invalid_vp_outcome"] += int((valid & ~vp.label_valid).sum())
            good = valid & stable & lp.label_valid & vp.label_valid
            selected = []
            for category in range(4):
                mask = scenario == category
                seen[category] += int((valid & mask).sum())
                rejected[category] += int((valid & mask & ~stable).sum())
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
            ids = torch.cat(selected)
            if len(ids):
                labels = torch.stack(
                    (lp.fail_short, lp.fail_long, vp.fail_short, vp.fail_long), -1
                )
                groups = [
                    f"terrain_seed{self.seed}:type{int(terrain[i])}:level{int(level[i])}"
                    for i in ids.tolist()
                ]
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
                    self.seed * 100_000_000
                    + self.batch_index * self.env.num_envs
                    + ids,
                    torch.stack((lp.metrics(), vp.metrics()), 1)[ids],
                    faults[ids],
                    torch.full(
                        (len(ids),), now, device=self.device, dtype=torch.float64
                    ),
                    self.last_feature_time[ids].clone(),
                    execution[ids],
                )
                self.writer.append(batch)
                collected += len(ids)
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
            self.batch_index += 1
            if self.batch_index % 10 == 0:
                print(
                    f"[SelectorCollector] candidates={self.batch_index} accepted={collected}/{self.collection_cfg.target_samples} scenarios={accepted}",
                    flush=True,
                )
            if collected < self.collection_cfg.target_samples:
                self._advance_behavior()
        metadata["collection_complete"] = True
