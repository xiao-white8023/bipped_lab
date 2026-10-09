from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.utils import Normalizer
from rsl_rl.utils.renet_checkpoint import upgrade_renet_auxiliary_checkpoint
from test_renet_auxiliary_updates import ROOT, _extract_function
from test_renet_lp_vp_training import _make_policy, _obs


class RecordingDiscriminator(Discriminator):
    def forward(self, inputs):
        self.classification_inputs = inputs.detach().clone()
        return super().forward(inputs)

    def compute_grad_pen(self, state, next_state, lambda_=10):
        self.penalty_inputs = torch.cat((state, next_state), -1).detach().clone()
        torch.testing.assert_close(self.classification_inputs, self.penalty_inputs)
        return super().compute_grad_pen(state, next_state, lambda_)


class RecordingNormalizer(Normalizer):
    def __init__(self, dim):
        super().__init__(dim)
        self.updates = []

    def update(self, values):
        self.updates.append(values.copy())
        super().update(values)


def test_amp_helper_uses_raw_statistics_and_normalized_penalty():
    raw = torch.tensor([[8., 18.], [12., 22.]])
    normalizer = RecordingNormalizer(2)
    normalizer.update(raw.numpy())
    algorithm = AMPPPO.__new__(AMPPPO)
    algorithm.device = 'cpu'
    algorithm.amploss_coef = 1.
    discriminator = RecordingDiscriminator(4, .3, [8], 'cpu')
    loss, _, update = algorithm._compute_single_amp_discriminator_loss(
        discriminator, normalizer, (raw, raw), (raw, raw))
    assert discriminator.penalty_inputs.abs().max() < 1.01
    assert update[1] is raw and update[2] is raw
    for values in update[1:]:
        update[0].update(values.numpy())
    np.testing.assert_allclose(normalizer.mean, [10., 20.], atol=4e-4)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in discriminator.parameters() if p.grad is not None)


def test_terrain_hits_make_finite_labels_and_an_independent_validity_mask():
    target_fn = _extract_function(ROOT / 'legged_lab/envs/g1/RENet_env.py',
                                  'compute_terrain_height_target', 'G1RENetEnv')
    hits = torch.zeros(3, 5, 3)
    hits[0, 1, 2] = float('inf')
    hits[0, 2, 0] = float('nan')  # Finite z alone is insufficient.
    hits[0, 3, 2] = -100.       # Beyond max_distance from the real ray origin.
    hits[1] = float('nan')
    position = torch.tensor([[0., 0., .8], [0., 0., .8], [0., 0., float('nan')]])
    origins = torch.zeros_like(hits)
    origins[..., 2] = 20.8
    env = SimpleNamespace(height_scanner=SimpleNamespace(
        data=SimpleNamespace(pos_w=position, ray_hits_w=hits),
        ray_starts_w=origins, cfg=SimpleNamespace(max_distance=40.)),
        cfg=SimpleNamespace(normalization=SimpleNamespace(height_scan_offset=.5)),
        obs_scales=SimpleNamespace(height_scan=1.))
    target = target_fn(env)
    assert env.terrain_height_valid.tolist() == [[True, False, False, False, True], [False]*5, [False]*5]
    torch.testing.assert_close(target[0], torch.tensor([.3, 0., 0., 0., .3]))
    assert torch.isfinite(target).all()
    assert not target[1:].any()


def test_masked_terrain_loss_ignores_invalid_nan_and_has_zero_invalid_gradient():
    prediction = torch.tensor([[.2, float('nan'), 10.]], requires_grad=True)
    target = torch.tensor([[.1, float('nan'), float('inf')]])
    valid = torch.tensor([[True, False, False]])
    loss = AMPPPO._terrain_height_mse(prediction, target, valid)
    torch.testing.assert_close(loss, torch.tensor(.01))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[.2, 0., 0.]]))
    prediction.grad = None
    loss = AMPPPO._terrain_height_mse(prediction, target, torch.zeros_like(valid))
    assert loss.item() == 0
    loss.backward()
    assert not prediction.grad.any()


def test_batch_terrain_normalization_excludes_invalid_labels_and_handles_one_valid_point():
    algorithm = AMPPPO.__new__(AMPPPO)
    algorithm.terrain_recon_target_clip = 1.
    algorithm.terrain_recon_target_normalization = 'batch'
    valid = torch.tensor([[True, True, False]])
    target = algorithm._prepare_vp_terrain_target(torch.tensor([[.1, .3, float('nan')]]), valid)
    torch.testing.assert_close(target, torch.tensor([[-1., 1., 0.]]))
    for bits in [torch.tensor([[True, False, False]]), torch.zeros_like(valid)]:
        assert torch.isfinite(algorithm._prepare_vp_terrain_target(torch.tensor([[.1, float('nan'), float('inf')]]), bits)).all()


@pytest.mark.parametrize('mode', [0., 1.])
@pytest.mark.parametrize('lp_front', [False, True])
@pytest.mark.parametrize('label_region', ['none', 'front', 'rear'])
def test_real_amp_update_aligns_masks_and_supervises_only_valid_terrain(mode, lp_front, label_region):
    torch.manual_seed(315)
    policy = _make_policy(lp_front=lp_front)
    raw = torch.linspace(10., 20., 58).repeat(2, 1)
    class Expert:
        def feed_forward_generator(self, batches, batch_size):
            for _ in range(batches):
                yield raw[:1].repeat(batch_size, 1), raw[:1].repeat(batch_size, 1)
    normalizer = RecordingNormalizer(58)
    normalizer.update(raw.numpy())
    normalizer.updates.clear()
    discriminator = RecordingDiscriminator(116, .3, [16, 8], 'cpu')
    algorithm = AMPPPO(policy, discriminator, Expert(), normalizer, amp_replay_buffer_size=64,
                       num_mini_batches=2, num_learning_epochs=1, terrain_recon_coef=.5,
                       lp_terrain_recon_coef=.5, terrain_recon_target_normalization='fixed',
                       use_terrain_height_valid_mask=True, use_feet_height_valid_mask=True,
                       single_critic_dim=85, critic_history_len=10)
    algorithm.init_storage('rl', 2, 2, [policy.num_actor_obs], [1037], [23])
    valid = torch.zeros(2, 187, dtype=torch.bool)
    indices = algorithm.terrain_front_indices
    if label_region == 'front':
        valid[:, indices] = True
    elif label_region == 'rear':
        valid.fill_(True)
        valid[:, indices] = False
    valid[1].fill_(False)  # One environment has no ground truth at all.
    before_lp = [p.detach().clone() for p in policy.lp_terrain_decoder.parameters()]
    before_vp = [p.detach().clone() for p in policy.terrain_decoder.parameters()]
    expected_valid = valid.clone()
    with torch.no_grad():
        for _ in range(2):
            obs = _obs(policy, mask=mode)
            obs[:, :187] = expected_valid.float()  # Encode IDs to verify shuffled alignment.
            critic = torch.zeros(2, 1037)
            critic[:, -187:] = torch.where(expected_valid, .2, 0.)
            current_mask = expected_valid.clone()
            algorithm.act(obs, critic, raw, feet_height_valid=torch.ones(2, 2, dtype=torch.bool),
                          terrain_height_valid=current_mask)
            current_mask.fill_(False)
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, raw)
        algorithm.compute_returns(torch.zeros(2, 1037))
    # Storage includes both label masks in exactly the same minibatch order.
    for sample in algorithm.storage.mini_batch_generator(2, 1, include_feet_height_valid=True,
                                                         include_terrain_height_valid=True):
        torch.testing.assert_close(sample[0][:, :187].bool(), sample[-1])
        assert sample[-2].all()
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert losses['terrain_height/valid_fraction'] == pytest.approx(expected_valid.float().mean().item())
    lp_should_learn = label_region == 'front' or (label_region == 'rear' and not lp_front)
    vp_should_learn = label_region == 'front'
    assert (losses[f"terrain_recon/{'lp_front' if lp_front else 'lp_full'}"] > 0) == lp_should_learn
    assert (losses['terrain_recon'] > 0) == vp_should_learn
    assert any(not torch.equal(a, b) for a, b in zip(before_lp, policy.lp_terrain_decoder.parameters())) == lp_should_learn
    assert any(not torch.equal(a, b) for a, b in zip(before_vp, policy.terrain_decoder.parameters())) == vp_should_learn
    assert len(normalizer.updates) == 4
    for values in normalizer.updates:
        np.testing.assert_allclose(values, raw.numpy())
    np.testing.assert_allclose(normalizer.mean, raw[0].numpy(), atol=2e-4)


@pytest.mark.parametrize('has_legacy_contract', [False, True])
def test_old_58d_checkpoint_resets_amp_only_and_new_checkpoint_roundtrips(has_legacy_contract):
    def optimizer(policy, disc):
        return torch.optim.Adam([{'params': policy.parameters(), 'name': 'policy'},
                                 {'params': disc.trunk.parameters(), 'name': 'amp_trunk'},
                                 {'params': disc.amp_linear.parameters(), 'name': 'amp_head'}])
    old = _make_policy().eval()
    old_disc = Discriminator(116, .3, [16, 8], 'cpu')
    old_opt = optimizer(old, old_disc)
    sum(p.square().sum() for group in old_opt.param_groups for p in group['params']).backward()
    old_opt.step()
    old_normalizer = Normalizer(58)
    old_normalizer.update(np.full((2, 58), 15.))
    aux = {'vp_terrain_target_normalization': 'fixed', 'terrain_target_clip': 1.}
    old_contract = {'version': 1, 'dimension': 58, 'joint_names': ['joint'], 'frame_layout': 'test'}
    new_contract = {**old_contract, 'version': 2,
                    'normalizer_statistics': 'raw_current_policy_and_expert_states',
                    'gradient_penalty_space': 'normalized_discriminator_inputs'}
    checkpoint = {'model_state_dict': old.state_dict(), 'optimizer_state_dict': old_opt.state_dict(),
                  'discriminator_state_dict': old_disc.state_dict(), 'amp_normalizer': old_normalizer,
                  'auxiliary_training_contract': aux}
    if has_legacy_contract:
        checkpoint['amp_observation_contract'] = old_contract
    policy = _make_policy().eval()
    disc = Discriminator(116, .3, [16, 8], 'cpu')
    opt = optimizer(policy, disc)
    fresh_norm = Normalizer(58)
    fresh_disc = {k: v.clone() for k, v in disc.state_dict().items()}
    upgraded, messages = upgrade_renet_auxiliary_checkpoint(checkpoint, policy, opt, aux,
        discriminator=disc, amp_normalizer=fresh_norm, amp_contract=new_contract)
    assert any(message.startswith('AMP reset:') for message in messages)
    assert upgraded['amp_normalizer'] is fresh_norm
    policy.load_state_dict(upgraded['model_state_dict'])
    disc.load_state_dict(upgraded['discriminator_state_dict'])
    opt.load_state_dict(upgraded['optimizer_state_dict'])
    for mode in (0., 1.):
        obs = _obs(policy, mask=mode)
        with torch.no_grad():
            torch.testing.assert_close(policy.act_inference(obs), old.act_inference(obs), rtol=0, atol=0)
    for name, p in policy.named_parameters():
        torch.testing.assert_close(opt.state[p]['exp_avg'], old_opt.state[dict(old.named_parameters())[name]]['exp_avg'])
    assert all(p not in opt.state for p in disc.parameters())
    for k, v in disc.state_dict().items():
        torch.testing.assert_close(v, fresh_disc[k])
    unchanged, messages = upgrade_renet_auxiliary_checkpoint(upgraded, policy, opt, aux,
        discriminator=disc, amp_normalizer=fresh_norm, amp_contract=new_contract)
    assert not messages
    assert unchanged['amp_normalizer'] is fresh_norm
    assert unchanged['optimizer_state_dict'] is upgraded['optimizer_state_dict']
    broken = {**upgraded, 'amp_observation_contract': {**new_contract, 'joint_names': ['wrong']}}
    with pytest.raises(ValueError, match='joint order/layout'):
        upgrade_renet_auxiliary_checkpoint(broken, policy, opt, aux,
            discriminator=disc, amp_normalizer=fresh_norm, amp_contract=new_contract)
    broken = {**upgraded, 'amp_observation_contract': {**new_contract, 'gradient_penalty_space': 'unknown'}}
    with pytest.raises(ValueError, match='preprocessing contract'):
        upgrade_renet_auxiliary_checkpoint(broken, policy, opt, aux,
            discriminator=disc, amp_normalizer=fresh_norm, amp_contract=new_contract)
    with pytest.raises(ValueError, match='requires a current AMP contract'):
        upgrade_renet_auxiliary_checkpoint(upgraded, policy, opt, aux,
            discriminator=disc, amp_normalizer=fresh_norm)
