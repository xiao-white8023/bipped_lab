"""Paired LP/VP counterfactual collection without modifying ``G1RENetEnv``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import CollectionConfig, SelectorConfig
from .dataset import SelectorBatch, SelectorDatasetWriter
from .faults import FaultSnapshot, SelectorFaultManager
from .features import FeatureHistory, SensorHealthTracker, build_selector_frame
from .labels import (
    RolloutOutcome, SafetyOutcomeAccumulator, safety_label_contract,
    step_with_safety_capture,
)
from .policy_adapter import FrozenRENetAdapter
from .snapshot import RENetSnapshot, RENetSnapshotManager


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
    last_lp: torch.Tensor
    last_vp: torch.Tensor
    last_lp_timestamp: torch.Tensor
    last_vp_timestamp: torch.Tensor
    control_step: int
    initialized: bool


class FaultedObservationStream:
    """Patch only the LP/depth slices consumed by the frozen actor."""

    def __init__(
        self,
        adapter: FrozenRENetAdapter,
        fault_manager: SelectorFaultManager,
        *,
        control_steps_per_sensor_update: int = 5,
    ):
        self.adapter = adapter
        self.policy = adapter.policy
        self.faults = fault_manager
        self.num_envs = fault_manager.num_envs
        self.device = fault_manager.device
        self.control_steps_per_sensor_update = int(control_steps_per_sensor_update)
        self.proprio_end = int(self.policy.proprio_actor_dim)
        self.lidar_end = self.proprio_end + int(self.policy.lidar_flat_dim)
        self.depth_end = self.lidar_end + int(self.policy.depth_flat_dim)
        self.lp_shape = fault_manager.lp_shape
        self.vp_shape = fault_manager.vp_shape
        if self.policy.lidar_flat_dim != self.lp_shape[0] * self.lp_shape[1]:
            raise ValueError("fault-manager LP shape does not match the frozen policy")
        if self.policy.depth_flat_dim != self.vp_shape[0] * self.vp_shape[1] * self.vp_shape[2]:
            raise ValueError("fault-manager VP shape does not match the frozen policy")
        self.lp_health_tracker = SensorHealthTracker(self.num_envs, self.lp_shape, self.device)
        self.vp_health_tracker = SensorHealthTracker(self.num_envs, self.vp_shape, self.device)
        self.last_lp = torch.full(
            (self.num_envs, *self.lp_shape), self.faults.cfg.lp_invalid_height, device=self.device
        )
        self.last_vp = torch.ones((self.num_envs, *self.vp_shape), device=self.device)
        self.last_lp_timestamp = torch.zeros(self.num_envs, device=self.device)
        self.last_vp_timestamp = torch.zeros(self.num_envs, device=self.device)
        self.lp_health = None
        self.vp_health = None
        self.control_step = 0
        self.initialized = False

    def snapshot(self) -> ObservationStreamSnapshot:
        return ObservationStreamSnapshot(
            fault=self.faults.snapshot(),
            lp_health=self.lp_health_tracker.snapshot(),
            vp_health=self.vp_health_tracker.snapshot(),
            last_lp=self.last_lp.clone(),
            last_vp=self.last_vp.clone(),
            last_lp_timestamp=self.last_lp_timestamp.clone(),
            last_vp_timestamp=self.last_vp_timestamp.clone(),
            control_step=self.control_step,
            initialized=self.initialized,
        )

    def restore(self, snapshot: ObservationStreamSnapshot) -> None:
        self.faults.restore(snapshot.fault)
        self.lp_health_tracker.restore(snapshot.lp_health)
        self.vp_health_tracker.restore(snapshot.vp_health)
        self.last_lp.copy_(snapshot.last_lp)
        self.last_vp.copy_(snapshot.last_vp)
        self.last_lp_timestamp.copy_(snapshot.last_lp_timestamp)
        self.last_vp_timestamp.copy_(snapshot.last_vp_timestamp)
        self.control_step = snapshot.control_step
        self.initialized = snapshot.initialized
        self.lp_health = None
        self.vp_health = None

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.faults.reset(env_ids)
        self.lp_health_tracker.reset(env_ids)
        self.vp_health_tracker.reset(env_ids)
        self.last_lp[env_ids] = self.faults.cfg.lp_invalid_height
        self.last_vp[env_ids] = 1.0

    def process(
        self,
        observations: torch.Tensor,
        *,
        lp_timestamp: torch.Tensor,
        vp_timestamp: torch.Tensor,
        now: float | torch.Tensor,
        force_update: bool = False,
        advance_control_step: bool = True,
    ) -> tuple[torch.Tensor, bool]:
        update = force_update or not self.initialized or self.control_step % self.control_steps_per_sensor_update == 0
        if update:
            lp = observations[:, self.proprio_end : self.lidar_end].reshape(self.num_envs, *self.lp_shape)
            vp = observations[:, self.lidar_end : self.depth_end].reshape(self.num_envs, *self.vp_shape)
            faulted = self.faults.apply(lp, vp, lp_timestamp, vp_timestamp)
            self.last_lp.copy_(faulted.lp)
            self.last_vp.copy_(faulted.vp)
            self.last_lp_timestamp.copy_(faulted.lp_timestamp)
            self.last_vp_timestamp.copy_(faulted.vp_timestamp)
            lp_valid = torch.isfinite(self.last_lp) & (
                self.last_lp != self.faults.cfg.lp_invalid_height
            )
            vp_valid = torch.isfinite(self.last_vp) & (self.last_vp >= 0.0) & (self.last_vp <= 1.0)
            vp_abnormal = (self.last_vp < 0.0) | (self.last_vp > 1.0)
            self.lp_health = self.lp_health_tracker.update(
                self.last_lp,
                self.last_lp_timestamp,
                now,
                valid_mask=lp_valid,
            )
            self.vp_health = self.vp_health_tracker.update(
                self.last_vp,
                self.last_vp_timestamp,
                now,
                valid_mask=vp_valid,
                abnormal_mask=vp_abnormal,
            )
            self.initialized = True
        output = observations.clone()
        output[:, self.proprio_end : self.lidar_end] = self.last_lp.reshape(self.num_envs, -1)
        output[:, self.lidar_end : self.depth_end] = self.last_vp.reshape(self.num_envs, -1)
        if advance_control_step:
            self.control_step += 1
        return output, update


@dataclass
class CollectionSnapshot:
    environment: RENetSnapshot
    observation_stream: ObservationStreamSnapshot
    feature_history: tuple[torch.Tensor, torch.Tensor]
    observations: torch.Tensor
    behavior_mask: torch.Tensor
    last_behavior_switch: torch.Tensor


class CounterfactualCollector:
    """Collect fair LP/VP labels from one restored candidate state."""

    def __init__(
        self,
        env: Any,
        adapter: FrozenRENetAdapter,
        writer: SelectorDatasetWriter,
        *,
        faults: SelectorFaultManager,
        selector_cfg: SelectorConfig | None = None,
        collection_cfg: CollectionConfig | None = None,
        seed: int = 42,
    ):
        self.env = env
        self.adapter = adapter
        self.writer = writer
        self.selector_cfg = selector_cfg or SelectorConfig()
        self.collection_cfg = collection_cfg or CollectionConfig()
        self.selector_cfg.validate()
        self.collection_cfg.validate()
        if int(env.num_envs) != self.collection_cfg.num_envs:
            raise ValueError(
                f"collector requires {self.collection_cfg.num_envs} environments, got {env.num_envs}"
            )
        self.device = torch.device(env.device)
        if getattr(env, "height_scanner", None) is None:
            raise ValueError("selector safety labels require a ground-truth height scanner")
        self.snapshot_manager = RENetSnapshotManager(env)
        self.stream = FaultedObservationStream(adapter, faults)
        self.feature_history = FeatureHistory(env.num_envs, self.selector_cfg, self.device)
        self.generator = torch.Generator(device=self.device.type)
        self.generator.manual_seed(seed)
        self.seed = int(seed)
        self.behavior_mask = torch.zeros(env.num_envs, device=self.device)
        self.last_behavior_switch = torch.zeros(env.num_envs, device=self.device)
        self.episode_serial = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.batch_index = 0
        self.observations: torch.Tensor | None = None

    def _time(self) -> float:
        return float(getattr(self.env.sim, "current_time", self.env.sim_step_counter * self.env.physics_dt))

    def _sensor_timestamps(self) -> tuple[torch.Tensor, torch.Tensor]:
        lp_sensor = getattr(self.env, "lp_sensor", None)
        camera = getattr(self.env, "camera", None)
        now = self._time()
        lp_age = getattr(self.env, "extras", {}).get("observations", {}).get("lp_frame_age")
        if lp_age is not None:
            # Includes the actual packet selected by the LP delay history.
            lp_timestamp = now - lp_age.reshape(self.env.num_envs, -1)[:, -1]
        elif lp_sensor is not None and hasattr(lp_sensor, "scan_timestamps"):
            lp_timestamp = now - (lp_sensor.current_timestamps - lp_sensor.scan_timestamps)
        else:
            lp_timestamp = torch.full((self.env.num_envs,), now, device=self.device)
        if camera is not None and hasattr(camera, "current_timestamps"):
            # Sensor clocks restart at episode reset; global sim time does not.
            captured = self.env.depth_last_capture_time
            vp_timestamp = now - (camera.current_timestamps - captured)
        else:
            vp_timestamp = torch.full((self.env.num_envs,), now, device=self.device)
        return lp_timestamp, vp_timestamp

    def initialize(self) -> None:
        observations, _ = self.env.get_observations()
        lp_time, vp_time = self._sensor_timestamps()
        self.observations, _ = self.stream.process(
            observations,
            lp_timestamp=lp_time,
            vp_timestamp=vp_time,
            now=self._time(),
            force_update=True,
            advance_control_step=False,
        )
        self._append_feature(self.observations, self.behavior_mask)

    def _set_mask(self, observations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = observations.clone()
        mask_index = self.stream.depth_end
        output[:, mask_index : mask_index + 1] = mask[:, None]
        self.env.renet_estimator_mask.copy_(mask[:, None])
        return output

    def _append_feature(self, observations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.stream.lp_health is None or self.stream.vp_health is None:
            raise RuntimeError("observation stream has not produced health metrics")
        branches = self.adapter.evaluate(observations)
        normalized = self.adapter.normalize(observations)
        proprio_start = self.adapter.policy.proprio_actor_dim - self.selector_cfg.proprio_dim
        proprio = normalized[:, proprio_start : self.adapter.policy.proprio_actor_dim]
        now = torch.full((self.env.num_envs,), self._time(), device=self.device)
        dwell = (now - self.last_behavior_switch).clamp_min(0.0)
        frame = build_selector_frame(
            proprio,
            branches,
            self.stream.lp_health,
            self.stream.vp_health,
            mask,
            dwell,
            self.selector_cfg,
        )
        return self.feature_history.append(frame)

    def _capture(self) -> CollectionSnapshot:
        assert self.observations is not None
        return CollectionSnapshot(
            environment=self.snapshot_manager.capture(),
            observation_stream=self.stream.snapshot(),
            feature_history=self.feature_history.snapshot(),
            observations=self.observations.clone(),
            behavior_mask=self.behavior_mask.clone(),
            last_behavior_switch=self.last_behavior_switch.clone(),
        )

    def _restore(self, snapshot: CollectionSnapshot) -> None:
        self.snapshot_manager.restore(snapshot.environment)
        self.stream.restore(snapshot.observation_stream)
        self.feature_history.restore(snapshot.feature_history)
        self.observations = snapshot.observations.clone()
        self.behavior_mask.copy_(snapshot.behavior_mask)
        self.last_behavior_switch.copy_(snapshot.last_behavior_switch)

    def _rollout(self, mask_value: float, *, preserve_as_behavior: bool) -> tuple[RolloutOutcome, torch.Tensor]:
        assert self.observations is not None
        forced_mask = torch.full((self.env.num_envs,), mask_value, device=self.device)
        self.env.cfg.renet.mask_mode = "lp" if mask_value > 0.5 else "vp"
        observations = self._set_mask(self.observations, forced_mask)
        outcome = SafetyOutcomeAccumulator(
            self.env.num_envs,
            device=self.device,
            short_horizon_steps=self.collection_cfg.short_horizon_steps,
            long_horizon_steps=self.collection_cfg.long_horizon_steps,
        )
        trajectory = []
        for _ in range(self.collection_cfg.long_horizon_steps):
            branches = self.adapter.evaluate(observations)
            actions = branches.lp_action if mask_value > 0.5 else branches.vp_action
            command = self.env.command_generator.command.clone()
            step_result, safety_metrics = step_with_safety_capture(self.env, actions, command)
            observations, _, dones, extras = step_result
            # Labels consume the completed control step, before env.step's
            # automatic reset and before the next command/sensor histories.
            outcome.update(safety_metrics, dones, extras["time_outs"])
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids):
                self.stream.reset_envs(reset_ids)
                self.feature_history.reset(reset_ids)
            lp_time, vp_time = self._sensor_timestamps()
            observations, sensor_updated = self.stream.process(
                observations,
                lp_timestamp=lp_time,
                vp_timestamp=vp_time,
                now=self._time(),
                # Keep the simulated LP/VP cadence at 10 Hz.  With thousands
                # of environments there is usually at least one reset on most
                # control steps; forcing a global update here would therefore
                # silently turn both sensors into 50 Hz streams.
                force_update=False,
            )
            observations = self._set_mask(observations, forced_mask)
            trajectory.append(
                torch.cat(
                    (
                        self.env.robot.data.root_state_w,
                        self.env.robot.data.joint_pos,
                        self.env.robot.data.joint_vel,
                    ),
                    dim=-1,
                ).clone()
            )
            if sensor_updated:
                self._append_feature(observations, forced_mask)
            if preserve_as_behavior and len(reset_ids):
                self.episode_serial[reset_ids] += 1
        self.observations = observations
        return outcome.outcome(), torch.stack(trajectory, dim=0)

    def _repeated_branch_rollout(
        self,
        snapshot: CollectionSnapshot,
        mask_value: float,
        *,
        preserve_second_as_behavior: bool,
    ) -> tuple[RolloutOutcome, torch.Tensor]:
        """Replay one branch twice and return only exactly stable labels.

        The second rollout supplies the stored audit metrics.  For the branch
        chosen to advance data collection it also supplies the next behavior
        state.  This makes the final state transition unambiguous while the
        first rollout remains an independent label-consistency check.
        """

        first, _ = self._rollout(mask_value, preserve_as_behavior=False)
        self._restore(snapshot)
        second, _ = self._rollout(
            mask_value,
            preserve_as_behavior=preserve_second_as_behavior,
        )
        return second, outcome_replay_stability(first, second)

    @torch.no_grad()
    def validate_replay(
        self, steps: int = 10, atol: float = 1.0e-4
    ) -> dict[str, float | bool | int]:
        if self.observations is None:
            self.initialize()
        snapshot = self._capture()
        # Replay validation uses the normal rollout path and compares only the
        # requested prefix; the full 1 s rollout remains the label contract.
        first_outcome, first = self._rollout(float(self.behavior_mask[0].item()), preserve_as_behavior=False)
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
        initial_restore_error = float((restored_initial - expected_initial).abs().max().item())
        second_outcome, second = self._rollout(float(self.behavior_mask[0].item()), preserve_as_behavior=False)
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
        max_error = float(compared_error.max().item()) if compared_error.numel() else 0.0
        rmse = (
            float(torch.sqrt(compared_error.square().mean()).item())
            if compared_error.numel()
            else 0.0
        )
        labels_equal_per_env = (first_outcome.fail_short == second_outcome.fail_short) & (
            first_outcome.fail_long == second_outcome.fail_long
        )
        failure_step_equal_per_env = (
            first_outcome.first_failure_step == second_outcome.first_failure_step
        )
        reset_step_equal_per_env = (
            first_outcome.first_reset_step == second_outcome.first_reset_step
        )
        validity_equal_per_env = first_outcome.label_valid == second_outcome.label_valid
        masked_error = torch.where(pre_failure[..., None], error, torch.zeros_like(error))
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
            "trajectory_env_mismatch_count": int((per_env_trajectory_max > atol).sum().item()),
            "max_abs_error": max_error,
            "rmse": rmse,
            "post_reset_max_abs_error": float(error.max().item()),
            "initial_restore_max_abs_error": initial_restore_error,
            "first_divergent_step": first_divergent_step,
            "root_state_max_abs_error": float(masked_error[..., :13].max().item()),
            "joint_position_max_abs_error": float(masked_error[..., 13:36].max().item()),
            "joint_velocity_max_abs_error": float(masked_error[..., 36:].max().item()),
            "label_agreement": float(labels_equal_per_env.float().mean().item()),
            "label_mismatch_count": int((~labels_equal_per_env).sum().item()),
            "failure_step_agreement": float(
                failure_step_equal_per_env.float().mean().item()
            ),
            "failure_step_mismatch_count": int((~failure_step_equal_per_env).sum().item()),
            "reset_step_agreement": float(reset_step_equal_per_env.float().mean().item()),
            "reset_step_mismatch_count": int((~reset_step_equal_per_env).sum().item()),
            "label_validity_agreement": float(validity_equal_per_env.float().mean().item()),
            "first_replay_valid_label_count": int(first_outcome.label_valid.sum().item()),
            "second_replay_valid_label_count": int(second_outcome.label_valid.sum().item()),
            "first_replay_fail_short_count": int(first_outcome.fail_short.sum().item()),
            "second_replay_fail_short_count": int(second_outcome.fail_short.sum().item()),
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
            levels = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        if types is None:
            types = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        return levels.clone(), types.clone()

    @torch.no_grad()
    def collect(self) -> None:
        if self.observations is None:
            self.initialize()
        replay = self.validate_replay()
        if not replay["passed"]:
            raise RuntimeError(f"paired rollout replay check failed: {replay}")
        self.writer.metadata["snapshot_replay_validation"] = replay
        self.writer.metadata["safety_label_contract"] = safety_label_contract()
        self.writer.metadata["per_candidate_replay_filter"] = {
            "contract": (
                "LP and VP are each replayed twice; persisted samples have exact "
                "label, validity, first-failure, first-reset, censoring, observed-step, "
                "termination and torso-contact agreement"
            ),
            "checked": 0,
            "rejected": 0,
            "agreement": 1.0,
        }
        label_filter = {
            "insufficient_episode_horizon": 0,
            "invalid_lp_outcome": 0,
            "invalid_vp_outcome": 0,
        }
        self.writer.metadata["safety_label_filter"] = label_filter
        print(f"[SelectorCollector] snapshot replay gate passed: {replay}")
        collected = 0
        replay_checked = 0
        replay_rejected = 0
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        while collected < self.collection_cfg.target_samples:
            assert self.observations is not None
            candidate_history = self.feature_history.buffer.clone()
            terrain_level, terrain_type = self._terrain_metadata()
            fault_metadata = self.stream.faults.audit_metadata()
            base_valid_candidate = self.feature_history.initialized & ~self.stream.faults.unobservable_onset_within(
                self.collection_cfg.long_horizon_steps // self.stream.control_steps_per_sensor_update
            )
            # Do not reinterpret an early time limit as a full safe horizon.
            enough_horizon = (
                self.env.max_episode_length - self.env.episode_length_buf
                >= self.collection_cfg.long_horizon_steps
            )
            label_filter["insufficient_episode_horizon"] += int((base_valid_candidate & ~enough_horizon).sum().item())
            base_valid_candidate &= enough_horizon
            group_ids = [
                # A generated sub-terrain is uniquely identified by generator
                # seed, type and level.  Grouping at this granularity is
                # stricter than grouping by episode: neither one terrain
                # instance nor any episode on it can leak across splits.
                f"terrain_seed{self.seed}:type{int(terrain_type[env_id])}:"
                f"level{int(terrain_level[env_id])}"
                for env_id in range(self.env.num_envs)
            ]
            snapshot = self._capture()
            behavior_lp = bool(self.batch_index % 2)
            if behavior_lp:
                vp_outcome, vp_stable = self._repeated_branch_rollout(
                    snapshot,
                    0.0,
                    preserve_second_as_behavior=False,
                )
                self._restore(snapshot)
                lp_outcome, lp_stable = self._repeated_branch_rollout(
                    snapshot,
                    1.0,
                    preserve_second_as_behavior=True,
                )
                self.behavior_mask.fill_(1.0)
            else:
                lp_outcome, lp_stable = self._repeated_branch_rollout(
                    snapshot,
                    1.0,
                    preserve_second_as_behavior=False,
                )
                self._restore(snapshot)
                vp_outcome, vp_stable = self._repeated_branch_rollout(
                    snapshot,
                    0.0,
                    preserve_second_as_behavior=True,
                )
                self.behavior_mask.fill_(0.0)
            replay_stable = lp_stable & vp_stable
            unstable_candidate = base_valid_candidate & ~replay_stable
            replay_checked += int(base_valid_candidate.sum().item())
            replay_rejected += int(unstable_candidate.sum().item())
            replay_filter_metadata = self.writer.metadata["per_candidate_replay_filter"]
            replay_filter_metadata["checked"] = replay_checked
            replay_filter_metadata["rejected"] = replay_rejected
            replay_filter_metadata["agreement"] = 1.0 - replay_rejected / max(replay_checked, 1)
            label_filter["invalid_lp_outcome"] += int((base_valid_candidate & ~lp_outcome.label_valid).sum().item())
            label_filter["invalid_vp_outcome"] += int((base_valid_candidate & ~vp_outcome.label_valid).sum().item())
            valid_candidate = base_valid_candidate & replay_stable & lp_outcome.label_valid & vp_outcome.label_valid
            switched = self.behavior_mask != snapshot.behavior_mask
            self.last_behavior_switch[switched] = self._time()
            labels = torch.stack(
                (
                    lp_outcome.fail_short,
                    lp_outcome.fail_long,
                    vp_outcome.fail_short,
                    vp_outcome.fail_long,
                ),
                dim=-1,
            )
            future_metrics = torch.stack((lp_outcome.metrics(), vp_outcome.metrics()), dim=1)
            scenario_seed = self.seed * 10_000_000 + self.batch_index * self.env.num_envs + env_ids
            remaining = self.collection_cfg.target_samples - collected
            candidate_ids = torch.nonzero(valid_candidate, as_tuple=False).flatten()
            take = min(remaining, len(candidate_ids))
            selected = candidate_ids[:take]
            if take == 0:
                self.batch_index += 1
                continue
            batch = SelectorBatch(
                features=candidate_history[selected],
                labels=labels[selected],
                group_ids=[group_ids[int(index)] for index in selected.cpu().tolist()],
                terrain_level=terrain_level[selected],
                terrain_type=terrain_type[selected],
                scenario_seed=scenario_seed[selected],
                future_metrics=future_metrics[selected],
                fault_metadata=fault_metadata[selected],
            )
            self.writer.append(batch)
            collected += take
            self.batch_index += 1
            if self.batch_index % 8 == 0:
                print(f"[SelectorCollector] collected {collected}/{self.collection_cfg.target_samples}")
