from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsl_rl.algorithms import PPO, SquatPPO
from rsl_rl.modules import ActorCritic, SquatActorCritic
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage, SquatRolloutStorage


REPO_ROOT = Path(__file__).parents[2]


def _cenet_cfg(enabled: bool = True):
    return {
        "enabled": enabled,
        "single_obs_dim": 80,
        "history_length": 10,
        "encoder_hidden_dims": [32, 16],
        "com_position_dim": 3,
        "com_velocity_dim": 3,
        "com_position_scale": [10.0, 10.0, 2.0],
        "com_velocity_scale": [2.0, 2.0, 2.0],
        "latent_dim": 16,
        "decoder_hidden_dims": [16, 32],
        "logvar_min": -10.0,
        "logvar_max": 2.0,
    }


def _policy(enabled: bool, critic_dim: int = 1060):
    return SquatActorCritic(
        800,
        critic_dim,
        15,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        activation="elu",
        cenet=_cenet_cfg(enabled),
    )


class _DummySquatEnv:
    def __init__(self):
        self.num_envs = 2
        self.num_actions = 15
        self.device = "cpu"
        self.cenet_training_enabled = None

    def set_cenet_training_enabled(self, enabled):
        self.cenet_training_enabled = bool(enabled)

    def get_observations(self):
        return torch.zeros(2, 800), {
            "observations": {"critic": torch.zeros(2, 1060)}
        }


def _runner_cfg(enabled: bool):
    return {
        "policy": {
            "class_name": "SquatActorCritic",
            "actor_hidden_dims": [32, 16],
            "critic_hidden_dims": [32, 16],
            "activation": "elu",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "cenet": _cenet_cfg(enabled),
        },
        "algorithm": {
            "class_name": "SquatPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "critic_single_obs_dim": 106,
            "critic_history_length": 10,
            "com_position_target_index": 86,
            "com_velocity_target_index": 89,
        },
        "num_steps_per_env": 2,
        "save_interval": 100,
        "empirical_normalization": False,
    }


def test_disabled_policy_preserves_legacy_state_dict_and_outputs():
    torch.manual_seed(7)
    legacy = ActorCritic(
        800,
        1060,
        15,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        activation="elu",
    )
    torch.manual_seed(7)
    switchable = _policy(enabled=False)

    assert list(legacy.state_dict()) == list(switchable.state_dict())
    assert not any(key.startswith("cenet_module.") for key in switchable.state_dict())

    observations = torch.randn(4, 800)
    critic_observations = torch.randn(4, 1060)
    torch.testing.assert_close(
        legacy.act_inference(observations),
        switchable.act_inference(observations),
    )
    torch.testing.assert_close(
        legacy.evaluate(critic_observations),
        switchable.evaluate(critic_observations),
    )


def test_disabled_algorithm_uses_legacy_storage():
    algorithm = SquatPPO(_policy(enabled=False), desired_kl=None)
    algorithm.init_storage("rl", 2, 2, [800], [1060], [15])

    assert type(algorithm.storage) is RolloutStorage
    assert not hasattr(algorithm.storage, "com_velocity_targets")
    assert not hasattr(algorithm.policy, "cenet_module")


def test_disabled_algorithm_matches_one_legacy_ppo_update():
    torch.manual_seed(19)
    legacy_policy = ActorCritic(
        800,
        1060,
        15,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        activation="elu",
    )
    torch.manual_seed(19)
    switchable_policy = _policy(enabled=False)
    legacy = PPO(
        legacy_policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        desired_kl=None,
        learning_rate=1.0e-3,
    )
    switchable = SquatPPO(
        switchable_policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        desired_kl=None,
        learning_rate=1.0e-3,
    )
    legacy.init_storage("rl", 2, 2, [800], [1060], [15])
    switchable.init_storage("rl", 2, 2, [800], [1060], [15])

    torch.manual_seed(23)
    rollout = [
        (
            torch.randn(2, 800),
            torch.randn(2, 1060),
            torch.randn(2),
            torch.tensor([False, step == 1]),
        )
        for step in range(2)
    ]
    for step, (observations, critic_observations, rewards, dones) in enumerate(rollout):
        torch.manual_seed(100 + step)
        legacy.act(observations, critic_observations)
        torch.manual_seed(100 + step)
        switchable.act(observations, critic_observations)
        legacy.process_env_step(rewards, dones, {})
        switchable.process_env_step(rewards, dones, {})

    last_critic_observations = torch.randn(2, 1060)
    legacy.compute_returns(last_critic_observations)
    switchable.compute_returns(last_critic_observations)
    torch.manual_seed(31)
    legacy_losses = legacy.update()
    torch.manual_seed(31)
    switchable_losses = switchable.update()

    assert legacy_losses.keys() == switchable_losses.keys()
    for key in legacy_losses:
        assert switchable_losses[key] == pytest.approx(legacy_losses[key])
    for key, legacy_parameter in legacy.policy.state_dict().items():
        torch.testing.assert_close(
            switchable.policy.state_dict()[key],
            legacy_parameter,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_propagates_the_single_switch_to_policy_algorithm_and_environment(enabled):
    env = _DummySquatEnv()
    runner = OnPolicyRunner(env, _runner_cfg(enabled), device="cpu")

    assert env.cenet_training_enabled is enabled
    assert runner.alg.cenet_enabled is enabled
    assert runner.alg.policy.cenet_enabled is enabled
    expected_storage = SquatRolloutStorage if enabled else RolloutStorage
    assert type(runner.alg.storage) is expected_storage


def test_enabled_policy_shapes_and_current_frame_contract():
    policy = _policy(enabled=True)
    observations = torch.randn(4, 800)
    observations[:, -80:] = 3.25

    actor_input = policy._process_obs(observations)
    torch.testing.assert_close(actor_input[:, :80], observations[:, -80:])
    assert actor_input.shape == (4, 102)
    assert policy.actor[0].in_features == 102

    actions = policy.act_inference(observations)
    outputs = policy.get_cenet_auxiliary_outputs()
    assert actions.shape == (4, 15)
    assert outputs["com_position"].shape == (4, 3)
    assert outputs["com_velocity"].shape == (4, 3)
    assert outputs["latent_mu"].shape == (4, 16)
    assert outputs["latent_logvar"].shape == (4, 16)
    assert outputs["reconstruction"].shape == (4, 80)
    assert policy.cenet_module.decoder[0].in_features == 22
    torch.testing.assert_close(
        policy.cenet_module.com_position_scale,
        torch.tensor([10.0, 10.0, 2.0]),
    )
    torch.testing.assert_close(
        policy.cenet_module.com_velocity_scale,
        torch.tensor([2.0, 2.0, 2.0]),
    )


def test_invalid_com_head_dimensions_and_scales_are_rejected():
    invalid_dim_cfg = _cenet_cfg(True)
    invalid_dim_cfg["com_position_dim"] = 2
    with pytest.raises(ValueError, match="three-dimensional COM"):
        SquatActorCritic(800, 1060, 15, cenet=invalid_dim_cfg)

    invalid_scale_cfg = _cenet_cfg(True)
    invalid_scale_cfg["com_velocity_scale"] = [2.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="three positive values"):
        SquatActorCritic(800, 1060, 15, cenet=invalid_scale_cfg)

    old_cfg = _cenet_cfg(True)
    old_cfg.pop("com_position_dim")
    old_cfg.pop("com_velocity_dim")
    old_cfg["velocity_dim"] = 1
    with pytest.raises(ValueError, match="obsolete scalar-v_z CENet"):
        SquatActorCritic(800, 1060, 15, cenet=old_cfg)


def test_actor_uses_deterministic_mu_while_decoder_samples_latent():
    torch.manual_seed(41)
    policy = _policy(enabled=True).eval()
    observations = torch.randn(2, 800)

    action_1 = policy.act_inference(observations)
    reconstruction_1 = policy.get_cenet_auxiliary_outputs()["reconstruction"]
    action_2 = policy.act_inference(observations)
    reconstruction_2 = policy.get_cenet_auxiliary_outputs()["reconstruction"]

    torch.testing.assert_close(action_1, action_2)
    assert not torch.equal(reconstruction_1, reconstruction_2)


def test_checkpoint_mode_mismatch_is_rejected():
    legacy_state = _policy(enabled=False).state_dict()
    cenet_state = _policy(enabled=True).state_dict()

    with pytest.raises(RuntimeError, match="matching policy.cenet.enabled"):
        _policy(enabled=True).load_state_dict(legacy_state)
    with pytest.raises(RuntimeError, match="matching policy.cenet.enabled"):
        _policy(enabled=False).load_state_dict(cenet_state)


def test_old_scalar_vz_checkpoint_has_explicit_incompatibility_error():
    legacy_cenet_state = dict(_policy(enabled=True).state_dict())
    legacy_cenet_state.pop("cenet_module.com_position_head.weight")
    legacy_cenet_state.pop("cenet_module.com_position_head.bias")
    legacy_cenet_state.pop("cenet_module.com_velocity_head.weight")
    legacy_cenet_state.pop("cenet_module.com_velocity_head.bias")
    legacy_cenet_state["cenet_module.velocity_head.weight"] = torch.randn(1, 16)
    legacy_cenet_state["cenet_module.velocity_head.bias"] = torch.randn(1)

    with pytest.raises(RuntimeError, match="obsolete scalar-v_z CENet"):
        _policy(enabled=True).load_state_dict(legacy_cenet_state)


def test_done_mask_makes_reconstruction_loss_autograd_safe_zero():
    prediction = torch.randn(3, 80, requires_grad=True)
    target = torch.randn(3, 80)
    dones = torch.ones(3, dtype=torch.bool)

    loss = SquatPPO._masked_reconstruction_loss(prediction, target, dones)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_storage_rejects_wrong_com_target_shape_explicitly():
    storage = SquatRolloutStorage(
        "rl",
        2,
        1,
        [800],
        [1060],
        [15],
        reconstruction_target_dim=80,
    )
    transition = SquatRolloutStorage.Transition()
    transition.com_position_target = torch.zeros(2, 2)
    transition.com_velocity_target = torch.zeros(2, 3)
    transition.reconstruction_target = torch.zeros(2, 80)

    with pytest.raises(ValueError, match="COM position target shape mismatch"):
        storage.add_transitions(transition)


def test_enabled_rollout_keeps_com_state_at_t_and_reconstruction_at_t_plus_one():
    policy = _policy(enabled=True)
    algorithm = SquatPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        desired_kl=None,
        critic_single_obs_dim=106,
        critic_history_length=10,
        com_position_target_index=86,
        com_velocity_target_index=89,
    )
    algorithm.init_storage("rl", 2, 1, [800], [1060], [15])
    assert type(algorithm.storage) is SquatRolloutStorage

    observations_t = torch.randn(2, 800)
    critic_observations_t = torch.randn(2, 1060)
    com_position_t = torch.tensor(
        [[0.01, -0.02, 0.63], [0.03, 0.01, 0.52]]
    )
    com_velocity_t = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.4, 0.3, -0.08]]
    )
    critic_observations_t[:, 9 * 106 + 86 : 9 * 106 + 89] = com_position_t
    critic_observations_t[:, 9 * 106 + 89 : 9 * 106 + 92] = com_velocity_t
    reconstruction_t1 = torch.randn(2, 80)

    with torch.inference_mode():
        algorithm.act(observations_t, critic_observations_t)
    algorithm.process_env_step(
        rewards=torch.zeros(2),
        dones=torch.tensor([False, True]),
        infos={"observations": {"cenet_reconstruction_target": reconstruction_t1}},
    )

    torch.testing.assert_close(
        algorithm.storage.com_position_targets[0], com_position_t
    )
    torch.testing.assert_close(
        algorithm.storage.com_velocity_targets[0], com_velocity_t
    )
    torch.testing.assert_close(algorithm.storage.reconstruction_targets[0], reconstruction_t1)

    batch = next(algorithm.storage.mini_batch_generator(1, 1))
    position_batch = batch[-3]
    velocity_batch = batch[-2]
    reconstruction_batch = batch[-1]
    position_order = torch.argsort(position_batch[:, 0])
    expected_order = torch.argsort(com_position_t[:, 0])
    torch.testing.assert_close(
        position_batch[position_order], com_position_t[expected_order]
    )
    velocity_order = torch.argsort(velocity_batch[:, 0])
    expected_velocity_order = torch.argsort(com_velocity_t[:, 0])
    torch.testing.assert_close(
        velocity_batch[velocity_order], com_velocity_t[expected_velocity_order]
    )
    torch.testing.assert_close(
        reconstruction_batch[position_order], reconstruction_t1[expected_order]
    )


def test_missing_clean_target_falls_back_to_last_frame_of_next_actor_obs():
    policy = _policy(enabled=True)
    algorithm = SquatPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        desired_kl=None,
        critic_single_obs_dim=106,
        critic_history_length=10,
        com_position_target_index=86,
        com_velocity_target_index=89,
    )
    algorithm.init_storage("rl", 2, 1, [800], [1060], [15])
    next_actor_obs = torch.randn(2, 800)

    with torch.inference_mode():
        algorithm.act(torch.randn(2, 800), torch.randn(2, 1060))
    algorithm.process_env_step(
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        infos={"observations": {}},
        next_actor_obs=next_actor_obs,
    )

    torch.testing.assert_close(
        algorithm.storage.reconstruction_targets[0],
        next_actor_obs[:, -80:],
    )
    assert algorithm._warned_missing_reconstruction_target


def test_enabled_ppo_update_is_finite_and_updates_cenet():
    torch.manual_seed(3)
    policy = _policy(enabled=True)
    algorithm = SquatPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        desired_kl=None,
        critic_single_obs_dim=106,
        critic_history_length=10,
        com_position_target_index=86,
        com_velocity_target_index=89,
    )
    algorithm.set_training_iteration(600)
    algorithm.init_storage("rl", 2, 2, [800], [1060], [15])
    encoder_before = policy.cenet_module.encoder[0].weight.detach().clone()

    for step in range(2):
        observations = torch.randn(2, 800)
        critic_observations = torch.randn(2, 1060)
        with torch.inference_mode():
            algorithm.act(observations, critic_observations)
        algorithm.process_env_step(
            rewards=torch.randn(2),
            dones=torch.tensor([False, step == 1]),
            infos={
                "observations": {
                    "cenet_reconstruction_target": torch.randn(2, 80),
                }
            },
        )

    algorithm.compute_returns(torch.randn(2, 1060))
    losses = algorithm.update()

    assert algorithm.storage.step == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert losses["cenet/com_position_weight"] == 1.0
    assert losses["cenet/com_velocity_weight"] == 1.0
    assert losses["cenet/reconstruction_weight"] == pytest.approx(0.5)
    assert losses["cenet/kl_weight"] > 0.0
    assert losses["cenet/grad_com_position"] > 0.0
    assert losses["cenet/grad_com_velocity"] > 0.0
    assert not torch.equal(encoder_before, policy.cenet_module.encoder[0].weight)


def test_com_targets_are_scaled_for_losses_and_rmse_is_reported_in_si_units():
    policy = _policy(enabled=True)
    position_m = torch.tensor([[0.1, -0.2, 0.5]])
    velocity_mps = torch.tensor([[0.3, -0.4, 0.2]])
    torch.testing.assert_close(
        position_m * policy.cenet_module.com_position_scale,
        torch.tensor([[1.0, -2.0, 1.0]]),
    )
    torch.testing.assert_close(
        velocity_mps * policy.cenet_module.com_velocity_scale,
        torch.tensor([[0.6, -0.8, 0.4]]),
    )


def test_torchscript_contract_remains_800_history_to_15_actions():
    policy = _policy(enabled=True).eval()
    traced = torch.jit.trace(policy, torch.zeros(1, 800))
    assert traced(torch.randn(3, 800)).shape == (3, 15)


def test_task_config_exposes_one_cenet_switch_and_specialized_classes():
    config_text = (
        REPO_ROOT / "legged_lab" / "envs" / "g1" / "squat_stand_cfg.py"
    ).read_text(encoding="utf-8")
    assert "class SquatCENetCfg" in config_text
    assert "enabled: bool = True" in config_text
    assert "com_position_scale: list[float] = [10.0, 10.0, 2.0]" in config_text
    assert "com_velocity_scale: list[float] = [2.0, 2.0, 2.0]" in config_text
    assert "com_position_target_index: int = 86" in config_text
    assert "com_velocity_target_index: int = 89" in config_text
    assert 'class_name="SquatActorCritic"' in config_text
    assert 'class_name="SquatPPO"' in config_text

    runner_text = (
        REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"
    ).read_text(encoding="utf-8")
    assert '{"PPO", "SquatPPO"}' in runner_text
