"""Four-head VP, stochastic AUX loss, transition alignment and export regressions."""
from __future__ import annotations

import copy

import pytest
import torch

from legged_lab.scripts.export import RENetPolicyExporter
from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import Discriminator, RENetActorCritic
from rsl_rl.runners import RENetAmpOnPolicyRunner
from rsl_rl.utils.renet_checkpoint import validate_renet_vp_checkpoint_contract
from test_g1_amp_format_switch import _assert_optimizer_equal, _env, _train_cfg
from test_renet_auxiliary_updates import _Expert
from test_renet_deployed_checkpoint import production_policy_kwargs
from test_renet_lp_vp_training import _depth_cnn_cfg, _has_nonzero_grad, _lidar_cnn_cfg, _obs


def _kwargs(frames=5):
    kwargs = production_policy_kwargs(frames)
    kwargs.update(actor_hidden_dims=[32], critic_hidden_dims=[32], proprio_embed_dims=[32],
                  vp_encoder_dims=[32], CnnMlp=_depth_cnn_cfg(), LidarCnnMlp=_lidar_cnn_cfg())
    return kwargs


def _policy(frames=5):
    torch.set_num_threads(2)
    return RENetActorCritic(2120, 1037, 23, **_kwargs(frames))


def _algorithm(policy):
    return AMPPPO(policy, Discriminator(100, 1., [16, 8], "cpu", .5), _Expert(), None,
                  amp_replay_buffer_size=64, num_mini_batches=2, num_learning_epochs=1,
                  single_critic_dim=85, critic_history_len=10, vel_in_critic_offset=78,
                  feet_height_in_critic_offset=83, vel_estimation_coef=1., feet_height_coef=.2,
                  terrain_recon_coef=.5, terrain_recon_front_only=True,
                  terrain_recon_target_normalization="fixed", lp_terrain_recon_coef=.5,
                  vp_vae_coef=1., vp_vae_beta=.01)


@pytest.mark.parametrize("frames", [5, 10])
def test_four_heads_are_the_actor_features_and_replace_old_vp_modules(frames):
    policy = _policy(frames)
    observations = _obs(policy)
    observations[:, -1] = torch.tensor([0., 1.])
    features = policy._process_actor_obs(observations)
    expected = torch.cat((policy._last_vp_velocity, policy._last_vp_feet_height,
                          policy._last_vp_context_mu, policy._last_vp_terrain_latent), dim=1)
    assert expected.shape == (2, 69) and features.shape == (2, 211)
    torch.testing.assert_close(features[0, 142:], expected[0])
    assert not features[0, 78:142].any() and not features[1, 142:].any()
    assert policy.vp_terrain_decoder[0].in_features == 32
    assert policy.vp_next_proprio_decoder[-1].out_features == 78
    assert not any(hasattr(policy, name) for name in ("vp_vel_estimator", "feet_height_decoder", "terrain_decoder"))
    torch.testing.assert_close(policy.predict_velocity()["vp"], expected[:, :3])
    torch.testing.assert_close(policy.predict_feet_height(), expected[:, 3:5])
    terrain = policy.predict_terrain()
    policy._last_current_proprio = torch.randn(2, 78) * 100
    torch.testing.assert_close(policy.predict_terrain(), terrain, rtol=0, atol=0)
    assert terrain.shape == (2, 99)


def test_actor_uses_mean_and_vae_sampling_does_not_change_ppo_probabilities():
    policy = _policy().train()
    observations = _obs(policy, mask=0.)
    actions = policy.act(observations)
    logprob = policy.get_actions_log_prob(actions).detach().clone()
    mean = policy.action_mean.detach().clone()
    reconstruction, mu, logvar = policy.predict_next_proprio()
    second, _, _ = policy.predict_next_proprio()
    assert reconstruction.shape == (2, 78) and mu.shape == logvar.shape == (2, 32)
    assert not torch.equal(reconstruction, second)
    policy.act(observations)
    torch.testing.assert_close(policy.action_mean, mean, rtol=0, atol=0)
    torch.testing.assert_close(policy.get_actions_log_prob(actions), logprob, rtol=0, atol=0)
    policy.eval()
    torch.testing.assert_close(policy.act_inference(observations), mean, rtol=0, atol=0)


def test_four_head_auxiliary_losses_train_the_shared_gru_and_all_decoders():
    policy = _policy()
    policy._process_actor_obs(_obs(policy, mask=1.))
    prediction, mu, logvar = policy.predict_next_proprio()
    reconstruction, kl = AMPPPO._vp_vae_losses(prediction, torch.randn_like(prediction), mu, logvar, torch.ones(2, 1).bool())
    loss = reconstruction + .01 * kl + policy.predict_velocity()["vp"].square().mean()
    loss += policy.predict_feet_height().square().mean() + policy.predict_terrain().square().mean()
    loss.backward()
    for name in ("vp_gru", "vp_velocity_head", "vp_feet_height_head", "vp_context_head",
                 "vp_terrain_latent_head", "vp_terrain_decoder", "vp_next_proprio_decoder"):
        assert _has_nonzero_grad(getattr(policy, name)), name


def test_beta_vae_loss_reductions_and_invalid_labels():
    prediction = torch.ones(2, 78, requires_grad=True)
    target = torch.zeros_like(prediction)
    mu = torch.ones(2, 32, requires_grad=True)
    logvar = torch.zeros(2, 32, requires_grad=True)
    reconstruction, kl = AMPPPO._vp_vae_losses(prediction, target, mu, logvar, torch.tensor([[True], [False]]))
    assert reconstruction.item() == 1. and kl.item() == 16.
    torch.testing.assert_close(reconstruction + .01 * kl, torch.tensor(1.16))
    (reconstruction + .01 * kl).backward()
    assert not prediction.grad[1].any() and not mu.grad[1].any()
    prediction.grad = mu.grad = logvar.grad = None
    reconstruction, kl = AMPPPO._vp_vae_losses(prediction, torch.full_like(target, float("nan")), mu, logvar, torch.zeros(2, 1).bool())
    assert reconstruction.item() == kl.item() == 0.
    (reconstruction + .01 * kl).backward()
    assert not prediction.grad.any() and not mu.grad.any() and not logvar.grad.any()


def _rollout(algorithm, terminal=False):
    policy = algorithm.policy
    algorithm.init_storage("rl", 2, 2, [2120], [1037], [23])
    with torch.no_grad():
        for step in range(2):
            observations = _obs(policy, mask=float(step))
            observations[:, 0] = torch.tensor([2 * step + 1., 2 * step + 2.])
            critic = torch.randn(2, 1037) * .1
            algorithm.act(observations, critic, torch.randn(2, 50))
            next_target = observations[:, :78].clone() + 10
            done = torch.tensor([False, terminal])
            algorithm.process_env_step(torch.ones(2), done, {}, torch.randn(2, 50), next_proprio=next_target)
            next_target.fill_(1000)  # labels must own their snapshot
        algorithm.compute_returns(torch.zeros(2, 1037))


def test_shuffled_next_frame_targets_remain_paired_and_dones_are_excluded():
    algorithm = _algorithm(_policy())
    _rollout(algorithm, terminal=True)
    assert algorithm.storage.next_proprio_valid.flatten().tolist() == [True, False, True, False]
    for sample in algorithm.storage.mini_batch_generator(2, 1, include_next_proprio=True):
        observations, target, valid = sample[0], sample[-2], sample[-1][:, 0]
        torch.testing.assert_close(target[valid, 0], observations[valid, 0] + 10)
        assert not target[~valid].any()


def test_vae_targets_reject_missing_labels_and_mask_nonfinite_next_observations():
    algorithm = _algorithm(_policy())
    algorithm.init_storage("rl", 2, 1, [2120], [1037], [23])
    with torch.no_grad():
        algorithm.act(_obs(algorithm.policy), torch.zeros(2, 1037), torch.zeros(2, 50))
        with pytest.raises(ValueError, match="clean next"):
            algorithm.process_env_step(torch.ones(2), torch.zeros(2).bool(), {}, torch.zeros(2, 50))
        target = torch.ones(2, 78)
        target[1, 0] = float("nan")
        algorithm.process_env_step(torch.ones(2), torch.zeros(2).bool(), {}, torch.zeros(2, 50), next_proprio=target)
    assert algorithm.storage.next_proprio_valid[0, :, 0].tolist() == [True, False]
    assert not algorithm.storage.next_proprio[0, 1].any()


def test_actual_ppo_amp_step_updates_posterior_decoder_and_logs_beta_loss():
    algorithm = _algorithm(_policy())
    _rollout(algorithm)
    before = {name: value.detach().clone() for name, value in algorithm.policy.named_parameters()
              if name.startswith(("vp_context_head.", "vp_next_proprio_decoder."))}
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert losses["vp_vae/reconstruction"] > 0 and losses["vp_vae/kl"] > 0
    assert losses["vp_vae/valid_fraction"] == 1. and losses["vp_vae/beta"] == .01
    assert losses["vp_vae/weighted_loss"] == pytest.approx(losses["vp_vae/reconstruction"] + .01 * losses["vp_vae/kl"])
    for prefix in ("vp_context_head.", "vp_next_proprio_decoder."):
        assert any(not torch.equal(value, algorithm.policy.state_dict()[name]) for name, value in before.items() if name.startswith(prefix))


def test_new_checkpoint_resumes_with_adam_and_rejects_archived_vp(tmp_path):
    env = _env("g1_renet", 50, critic_frames=10)
    config = _train_cfg("g1_renet", 50, critic_frames=10)
    config["policy"] = {"class_name": "RENetActorCritic", **_kwargs()}
    config["algorithm"].update(vp_vae_coef=1., vp_vae_beta=.01)
    runner = RENetAmpOnPolicyRunner(env, copy.deepcopy(config), log_dir=None, device="cpu")
    _rollout(runner.alg)
    runner.alg.update()
    runner.logger_type = "tensorboard"
    checkpoint = tmp_path / "new.pt"
    runner.save(str(checkpoint))
    restored = RENetAmpOnPolicyRunner(env, copy.deepcopy(config), log_dir=None, device="cpu")
    restored.load(str(checkpoint))
    for key, value in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(restored.alg.policy.state_dict()[key], value, rtol=0, atol=0)
    _assert_optimizer_equal(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict())
    current = runner.vp_encoder_contract
    with pytest.raises(ValueError, match="Legacy"):
        validate_renet_vp_checkpoint_contract(None, current, 10)
    legacy = _policy().get_vp_encoder_contract().copy()
    legacy["version"] = 1
    with pytest.raises(ValueError, match="incompatible"):
        validate_renet_vp_checkpoint_contract(legacy, current, 10)


@pytest.mark.parametrize("frames", [5, 10])
def test_torchscript_four_head_export_keeps_mask_routing_and_dynamic_batch(frames, tmp_path):
    policy = _policy(frames).eval()
    exporter = RENetPolicyExporter(policy)
    observations = _obs(policy, 2, mask=0.)
    traced = torch.jit.trace(exporter, observations, check_trace=False)
    path = tmp_path / "policy.pt"
    traced.save(str(path))
    restored = torch.jit.load(str(path))
    for count in (1, 3):
        for mask in (0., 1.):
            probe = _obs(policy, count, mask=mask)
            with torch.inference_mode():
                torch.testing.assert_close(restored(probe), policy.act_inference(probe), rtol=1e-4, atol=1e-5)
