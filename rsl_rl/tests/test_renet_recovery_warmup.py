from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO
from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"


def _load_env_methods(*method_names):
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    class_body = next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    selected = []
    for method_name in method_names:
        method = next(
            node for node in class_body if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method.decorator_list = []
        selected.append(method)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"math": math}
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _warmup_algorithm(rollout_samples=0, replay_samples=0):
    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.storage = SimpleNamespace(recovery_masks=torch.ones(rollout_samples, dtype=torch.bool))
    algorithm.amp_storage_recovery = SimpleNamespace(num_samples=replay_samples)
    algorithm.recovery_ppo_min_rollout_samples = 2048
    algorithm.recovery_drec_min_replay_samples = 2048
    algorithm.drec_reward_ready = False
    algorithm._drec_replay_ready = False
    algorithm.recovery_discriminator_updated_this_update = False
    algorithm._recovery_discriminator_loss_computed_this_update = False
    return algorithm


def _mode_aware_loss_algorithm():
    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.recovery_state_machine_enabled = True
    algorithm.enable_recovery_learning = True
    algorithm.recovery_advantage_weights = (2.5, 1.0, 0.1)
    algorithm.clip_param = 0.2
    algorithm.use_clipped_value_loss = False
    return algorithm


def _actor_rollout_data():
    return {
        "recovery_mask_t": torch.tensor([[False], [True]]),
        "recovery_task_advantages": torch.ones(2, 1),
        "recovery_amp_advantages": torch.ones(2, 1),
        "recovery_reg_advantages": torch.ones(2, 1),
    }


class _CountingRecoveryCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, observations):
        self.calls += 1
        prediction = self.weight.expand(observations.shape[0], 1)
        return {"task": prediction, "amp": prediction, "reg": prediction}


def _critic_rollout_data():
    data = {"recovery_mask_t": torch.tensor([[False], [True]])}
    for name in ("task", "amp", "reg"):
        data[f"recovery_{name}_values"] = torch.zeros(2, 1)
        data[f"recovery_{name}_returns"] = torch.ones(2, 1)
    return data


def test_recovery_ppo_gate_uses_current_rollout_2048_boundary():
    algorithm = _mode_aware_loss_algorithm()
    algorithm.recovery_ppo_min_rollout_samples = 2048
    algorithm.recovery_drec_min_replay_samples = 2048
    algorithm.amp_storage_recovery = SimpleNamespace(num_samples=0)
    surrogate = torch.tensor([2.0, 99.0], requires_grad=True)
    ratio = torch.ones(2)

    algorithm.storage = SimpleNamespace(recovery_masks=torch.ones(2047, dtype=torch.bool))
    algorithm._prepare_recovery_warmup_for_update()
    assert algorithm._current_recovery_rollout_samples == 2047
    assert algorithm._recovery_ppo_ready is False
    skipped_actor_loss = algorithm._compute_surrogate_loss(
        surrogate, surrogate, _actor_rollout_data(), ratio
    )
    torch.testing.assert_close(skipped_actor_loss, torch.tensor(2.0))

    critic = _CountingRecoveryCritic()
    algorithm.recovery_critic = critic
    value = torch.zeros(2, 1, requires_grad=True)
    locomotion_value_loss, recovery_value_loss, metrics = algorithm._compute_value_losses(
        value,
        torch.zeros_like(value),
        torch.tensor([[2.0], [1.0]]),
        torch.zeros(2, 3),
        _critic_rollout_data(),
    )
    assert locomotion_value_loss.item() == 4.0
    assert recovery_value_loss.item() == 0.0
    assert all(metrics[f"RecoveryValue/{name}_loss"].item() == 0.0 for name in ("task", "amp", "reg"))
    assert critic.calls == 0

    algorithm.storage.recovery_masks = torch.ones(2048, dtype=torch.bool)
    algorithm._prepare_recovery_warmup_for_update()
    assert algorithm._recovery_ppo_ready is True
    active_actor_loss = algorithm._compute_surrogate_loss(surrogate, surrogate, _actor_rollout_data(), ratio)
    assert active_actor_loss.item() != skipped_actor_loss.item()
    _, recovery_value_loss, _ = algorithm._compute_value_losses(
        value,
        torch.zeros_like(value),
        torch.tensor([[2.0], [1.0]]),
        torch.zeros(2, 3),
        _critic_rollout_data(),
    )
    assert recovery_value_loss.item() == 3.0
    assert critic.calls == 1

    algorithm.storage.recovery_masks = torch.ones(2050, dtype=torch.bool)
    algorithm._prepare_recovery_warmup_for_update()
    assert algorithm._recovery_ppo_ready is True


class _BatchSource:
    def __init__(self, name, num_samples=0):
        self.name = name
        self.num_samples = num_samples
        self.yield_count = 0

    def feed_forward_generator(self, num_updates, _mini_batch_size):
        for index in range(num_updates):
            self.yield_count += 1
            yield (self.name, index)


@pytest.mark.parametrize(
    (
        "rollout_samples",
        "replay_samples",
        "expected_ppo_ready",
        "expected_replay_ready",
        "expected_drec_ready",
    ),
    [
        (300, 100_000, False, True, False),
        (4_000, 1_000, True, False, False),
        (4_000, 100_000, True, True, True),
    ],
)
def test_drec_generator_follows_ppo_and_replay_snapshot_without_stopping_loco(
    rollout_samples,
    replay_samples,
    expected_ppo_ready,
    expected_replay_ready,
    expected_drec_ready,
):
    algorithm = _warmup_algorithm(rollout_samples, replay_samples)
    algorithm.recovery_drec_min_replay_samples = 2048
    algorithm.amp_storage_loco = _BatchSource("loco_policy")
    algorithm.amp_data_loco = _BatchSource("loco_expert")
    algorithm.amp_storage_recovery = _BatchSource("recovery_policy", replay_samples)
    algorithm.amp_data_recovery = _BatchSource("recovery_expert")

    algorithm._prepare_recovery_warmup_for_update()
    assert algorithm._recovery_ppo_ready is expected_ppo_ready
    assert algorithm._drec_replay_ready is expected_replay_ready
    assert algorithm._drec_update_ready is expected_drec_ready

    batches = list(algorithm._amp_mini_batch_generator(2, 8))
    assert len(batches) == 2
    # The generator consumes the snapshot and cannot write back into either gate.
    assert algorithm._recovery_ppo_ready is expected_ppo_ready
    assert algorithm._drec_update_ready is expected_drec_ready
    assert algorithm.amp_storage_loco.yield_count == 2
    assert algorithm.amp_data_loco.yield_count == 2
    assert algorithm.amp_storage_recovery.yield_count == (2 if expected_drec_ready else 0)
    assert algorithm.amp_data_recovery.yield_count == (2 if expected_drec_ready else 0)
    assert all((batch["recovery_policy"] is not None) is expected_drec_ready for batch in batches)
    assert all(batch["recovery_num_samples"] == replay_samples for batch in batches)


@pytest.mark.parametrize(
    (
        "rollout_samples",
        "replay_samples",
        "expected_blocked_by_ppo",
        "expected_blocked_by_replay",
    ),
    [
        (300, 100_000, 1.0, 0.0),
        (4_000, 1_000, 0.0, 1.0),
        (4_000, 100_000, 0.0, 0.0),
    ],
)
def test_drec_diagnostics_report_the_actual_blocking_gate(
    rollout_samples,
    replay_samples,
    expected_blocked_by_ppo,
    expected_blocked_by_replay,
):
    algorithm = _warmup_algorithm(rollout_samples, replay_samples)
    algorithm.enable_recovery_learning = True
    algorithm.storage.enter_recovery = torch.zeros(rollout_samples, dtype=torch.bool)
    algorithm.storage.exit_recovery = torch.zeros(rollout_samples, dtype=torch.bool)
    algorithm.storage.recovery_failed = torch.zeros(rollout_samples, dtype=torch.bool)
    for name in ("task", "amp", "reg"):
        setattr(
            algorithm.storage,
            f"recovery_{name}_advantages",
            torch.zeros(rollout_samples),
        )

    algorithm._prepare_recovery_warmup_for_update()
    algorithm._capture_recovery_rollout_diagnostics()
    diagnostics = algorithm.get_recovery_learning_diagnostics()

    assert diagnostics["RecoveryWarmup/drec_blocked_by_ppo"] == expected_blocked_by_ppo
    assert diagnostics["RecoveryWarmup/drec_blocked_by_replay"] == expected_blocked_by_replay


class _RewardDiscriminator:
    def __init__(self, reward_value, input_dim=4):
        self.reward_value = reward_value
        self.input_dim = input_dim
        self.calls = 0

    def predict_amp_reward(self, state, _next_state, _task_reward, normalizer=None):
        del normalizer
        self.calls += 1
        reward = torch.full((state.shape[0],), self.reward_value, device=state.device)
        logits = torch.full((state.shape[0], 1), self.reward_value, device=state.device)
        return reward, logits


def _reward_algorithm():
    algorithm = _warmup_algorithm(rollout_samples=0, replay_samples=0)
    algorithm.device = "cpu"
    algorithm.discriminator_loco = _RewardDiscriminator(3.0)
    algorithm.discriminator_recovery = _RewardDiscriminator(7.0)
    algorithm.amp_normalizer_loco = None
    algorithm.amp_normalizer_recovery = SimpleNamespace(mean=np.zeros(53))
    return algorithm


def test_drec_reward_is_zero_when_fresh_and_replay_threshold_alone_does_not_open_it():
    algorithm = _reward_algorithm()
    amp_obs = {
        "loco": torch.zeros(2, 2),
        "recovery": torch.zeros(2, 2),
    }
    recovery_mask = torch.tensor([False, True])
    reward, logits = algorithm.predict_routed_amp_reward(
        amp_obs, amp_obs, torch.ones(2), recovery_mask
    )
    torch.testing.assert_close(reward, torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(logits[:, 0], torch.tensor([3.0, 0.0]))
    assert algorithm.discriminator_recovery.calls == 0

    algorithm.amp_storage_recovery.num_samples = 2048
    algorithm._prepare_recovery_warmup_for_update()
    assert algorithm._recovery_ppo_ready is False
    assert algorithm._drec_replay_ready is True
    assert algorithm._drec_update_ready is False
    assert algorithm.drec_reward_ready is False
    reward, _ = algorithm.predict_routed_amp_reward(amp_obs, amp_obs, torch.ones(2), recovery_mask)
    torch.testing.assert_close(reward, torch.tensor([3.0, 0.0]))
    assert algorithm.discriminator_recovery.calls == 0


def test_real_drec_loss_and_completed_optimizer_step_make_reward_persistently_ready(monkeypatch):
    algorithm = _warmup_algorithm(rollout_samples=2048, replay_samples=2048)
    algorithm.enable_recovery_learning = True
    algorithm.storage.enter_recovery = torch.zeros(2048, dtype=torch.bool)
    algorithm.storage.exit_recovery = torch.zeros(2048, dtype=torch.bool)
    algorithm.storage.recovery_failed = torch.zeros(2048, dtype=torch.bool)
    for name in ("task", "amp", "reg"):
        setattr(
            algorithm.storage,
            f"recovery_{name}_advantages",
            torch.zeros(2048),
        )
    algorithm._last_recovery_learning_diagnostics = {}
    algorithm.discriminator_loco = object()
    algorithm.discriminator_recovery = object()
    algorithm.amp_normalizer_loco = None
    algorithm.amp_normalizer_recovery = None
    loco_parameter = torch.nn.Parameter(torch.tensor(1.0))
    recovery_parameter = torch.nn.Parameter(torch.tensor(1.0))
    algorithm.optimizer = torch.optim.SGD([loco_parameter, recovery_parameter], lr=0.1)

    def compute_single(discriminator, _normalizer, _policy, _expert):
        parameter = recovery_parameter if discriminator is algorithm.discriminator_recovery else loco_parameter
        loss = parameter.square()
        metrics = {"loss": float(loss.detach()), "grad_pen": 0.0, "policy_pred": 0.0, "expert_pred": 0.0}
        return loss, metrics, None

    algorithm._compute_single_amp_discriminator_loss = compute_single

    def parent_update(self):
        sample = {
            "loco_policy": object(),
            "loco_expert": object(),
            "recovery_policy": object(),
            "recovery_expert": object(),
            "recovery_num_samples": 2048,
        }
        loss, metrics, _ = self._compute_amp_loss(sample)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return metrics

    monkeypatch.setattr(AMPPPO, "update", parent_update)
    algorithm.update()

    assert recovery_parameter.item() < 1.0
    assert algorithm.recovery_discriminator_updated_this_update is True
    assert algorithm.drec_reward_ready is True
    diagnostics = algorithm.get_recovery_learning_diagnostics()
    assert diagnostics["RecoveryWarmup/drec_updated_this_update"] == 1.0
    assert diagnostics["RecoveryWarmup/drec_reward_ready"] == 1.0

    algorithm.device = "cpu"
    algorithm.discriminator_loco = _RewardDiscriminator(3.0)
    algorithm.discriminator_recovery = _RewardDiscriminator(7.0)
    amp_obs = {
        "loco": torch.zeros(2, 2),
        "recovery": torch.zeros(2, 2),
    }
    reward, _ = algorithm.predict_routed_amp_reward(
        amp_obs,
        amp_obs,
        torch.ones(2),
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(reward, torch.tensor([3.0, 7.0]))
    assert algorithm.discriminator_recovery.calls == 1

    algorithm.recovery_discriminator_updated_this_update = False
    assert algorithm.get_recovery_warmup_state() == {"drec_reward_ready": True}


def _curriculum_env():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(
                curriculum_min_attempts=1024,
                min_assist_force=0.0,
                initial_assist_force=200.0,
                min_beta=0.25,
                initial_beta=1.0,
            )
        ),
        recovery_curriculum_level=7,
        recovery_curriculum_window_attempts=333,
        recovery_curriculum_window_successes=201,
        current_recovery_assist_force=60.0,
        current_recovery_beta=0.74,
        recovery_curriculum_last_window_success_ratio=0.63,
        recovery_curriculum_last_window_attempts=1024,
        recovery_curriculum_last_window_successes=768,
        recovery_curriculum_last_window_advanced=True,
        recovery_curriculum_total_completed_attempts=12345,
        recovery_curriculum_total_windows=12,
        recovery_curriculum_total_level_advances=7,
    )


def test_curriculum_state_round_trip_restores_exact_force_beta_and_counters():
    get_state, load_state = _load_env_methods(
        "get_recovery_curriculum_state", "load_recovery_curriculum_state"
    )
    expected = get_state(_curriculum_env())
    restored = _curriculum_env()
    restored.recovery_curriculum_level = 0
    restored.current_recovery_assist_force = 200.0
    restored.current_recovery_beta = 1.0
    load_state(restored, expected)
    assert get_state(restored) == expected
    assert restored.current_recovery_assist_force == 60.0
    assert restored.current_recovery_beta == 0.74

    invalid = dict(expected, window_attempts=1024)
    with pytest.raises(ValueError, match="window_attempts"):
        load_state(restored, invalid)
    invalid = dict(expected, current_beta=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        load_state(restored, invalid)


class _CheckpointModule:
    def __init__(self, load_result=None):
        self.load_result = load_result

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return self.load_result


class _CheckpointOptimizer:
    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


def _checkpoint_runner(curriculum_state):
    runner = RENetAmpOnPolicyRunner.__new__(RENetAmpOnPolicyRunner)
    runner.env = SimpleNamespace(
        recovery_state_machine_enabled=True,
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(enable=True, curriculum_min_attempts=1024)
        ),
        get_recovery_curriculum_state=lambda: dict(curriculum_state),
    )
    algorithm = _warmup_algorithm(rollout_samples=0, replay_samples=0)
    algorithm.policy = _CheckpointModule(load_result=False)
    algorithm.optimizer = _CheckpointOptimizer()
    algorithm.discriminator_loco = _CheckpointModule()
    algorithm.amp_normalizer_loco = None
    algorithm.discriminator_recovery = _CheckpointModule()
    algorithm.amp_normalizer_recovery = None
    algorithm.recovery_critic = _CheckpointModule()
    algorithm.rnd = False
    runner.alg = algorithm
    runner.current_learning_iteration = 19
    runner.empirical_normalization = False
    runner.logger_type = "tensorboard"
    runner.disable_logs = True
    return runner


def test_runner_checkpoint_saves_scalar_states_without_replay_buffer(tmp_path):
    get_state, _ = _load_env_methods("get_recovery_curriculum_state", "load_recovery_curriculum_state")
    curriculum_state = get_state(_curriculum_env())
    runner = _checkpoint_runner(curriculum_state)
    runner.alg.drec_reward_ready = True
    checkpoint_path = tmp_path / "recovery.pt"
    runner.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["recovery_curriculum_state"] == curriculum_state
    assert checkpoint["recovery_warmup_state"] == {"drec_reward_ready": True}
    assert not any("replay" in key for key in checkpoint)


def test_runner_resume_restores_ready_true_with_empty_replay_but_keeps_update_gate_closed(tmp_path):
    get_state, _ = _load_env_methods("get_recovery_curriculum_state", "load_recovery_curriculum_state")
    curriculum_state = get_state(_curriculum_env())
    runner = _checkpoint_runner(curriculum_state)
    loaded_curriculum = []
    runner.env.load_recovery_curriculum_state = lambda state: loaded_curriculum.append(dict(state))
    checkpoint_path = tmp_path / "resume.pt"
    checkpoint = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "discriminator_state_dict": {},
        "amp_normalizer": None,
        "recovery_discriminator_state_dict": {},
        "recovery_amp_normalizer": SimpleNamespace(mean=np.zeros(53)),
        "recovery_critic_state_dict": {},
        "recovery_curriculum_state": curriculum_state,
        "recovery_warmup_state": {"drec_reward_ready": True},
        "iter": 19,
        "infos": {"source": "test"},
    }
    torch.save(checkpoint, checkpoint_path)
    with pytest.warns(UserWarning, match="Restored Recovery curriculum state"):
        infos = runner.load(str(checkpoint_path))
    assert infos == {"source": "test"}
    assert loaded_curriculum == [curriculum_state]
    assert runner.alg.drec_reward_ready is True
    runner.alg._prepare_recovery_warmup_for_update()
    assert runner.alg._drec_update_ready is False
    assert runner.alg.drec_reward_ready is True


def test_legacy_checkpoint_without_new_states_loads_without_key_error_and_gates_drec(tmp_path):
    get_state, _ = _load_env_methods("get_recovery_curriculum_state", "load_recovery_curriculum_state")
    runner = _checkpoint_runner(get_state(_curriculum_env()))
    loaded_curriculum = []
    runner.env.load_recovery_curriculum_state = lambda state: loaded_curriculum.append(state)
    runner.alg.drec_reward_ready = True
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "discriminator_state_dict": {},
            "amp_normalizer": None,
            "iter": 3,
            "infos": None,
        },
        checkpoint_path,
    )
    with pytest.warns(UserWarning) as warning_records:
        runner.load(str(checkpoint_path))
    warning_text = "\n".join(str(record.message) for record in warning_records)
    assert "fresh curriculum state" in warning_text
    assert "legacy checkpoint has no D_REC reward-ready state" in warning_text
    assert loaded_curriculum == []
    assert runner.alg.drec_reward_ready is False


def test_warmup_state_validation_is_strict_and_contains_no_transient_fields():
    algorithm = _warmup_algorithm()
    algorithm.load_recovery_warmup_state({"drec_reward_ready": True})
    assert algorithm.get_recovery_warmup_state() == {"drec_reward_ready": True}
    with pytest.raises(ValueError, match="exactly"):
        algorithm.load_recovery_warmup_state(
            {"drec_reward_ready": True, "drec_update_ready": True}
        )
    with pytest.raises(TypeError, match="bool"):
        algorithm.load_recovery_warmup_state({"drec_reward_ready": 1})
