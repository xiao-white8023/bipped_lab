from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from rsl_rl.utils.renet_checkpoint import validate_renet_vp_checkpoint_contract
from test_renet_lp_vp_training import _make_policy, _obs


def _vp_outputs(policy, observations):
    action = policy.act_inference(observations)
    return (action, policy.predict_velocity()['vp'], policy.predict_terrain(),
            policy.predict_feet_height(), policy._last_vp_latent)


@pytest.mark.parametrize('encoder,fusion', [
    ('ame_point_cross_attention', 'attention'),
])
def test_vp_actions_and_every_auxiliary_head_ignore_older_five_frames(encoder, fusion):
    torch.manual_seed(67)
    policy = _make_policy(lp_encoder_type=encoder, fusion_type=fusion).eval()
    obs = _obs(policy, batch_size=3, mask=0.)
    old_changed = obs.clone()
    old_changed[:, :5 * 78] += 10.
    recent_changed = obs.clone()
    recent_changed[:, 5 * 78:10 * 78] += .5
    lengths = []
    hook = policy.vp_gru.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    with torch.inference_mode():
        baseline = _vp_outputs(policy, obs)
        old = _vp_outputs(policy, old_changed)
        recent = _vp_outputs(policy, recent_changed)
    hook.remove()
    assert lengths == [5, 5, 5]
    for expected, actual, changed in zip(baseline, old, recent, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not torch.allclose(changed, expected)

    obs.requires_grad_()
    sum(value.square().mean() for value in _vp_outputs(policy, obs)).backward()
    assert torch.count_nonzero(obs.grad[:, :5 * 78]) == 0
    assert torch.count_nonzero(obs.grad[:, 5 * 78:10 * 78]) > 0


def test_vp_attention_sees_only_five_body_tokens_plus_two_depth_tokens():
    policy = _make_policy(lp_encoder_type='ame_point_cross_attention').eval()
    tokens = []
    hook = policy.vp_attention.register_forward_pre_hook(
        lambda module, args: tokens.append(tuple(value.shape for value in args)))
    with torch.inference_mode():
        policy.act_inference(_obs(policy, batch_size=2, mask=0.))
    hook.remove()
    assert tokens == [((2, 7, 64), (2, 7, 64), (2, 7, 64))]


@pytest.mark.parametrize('encoder', ['ame_point_cross_attention'])
def test_vp_window_change_preserves_parameter_layout_and_lp_outputs(encoder):
    legacy = _make_policy(lp_encoder_type=encoder, vp_frames=10).eval()
    current = _make_policy(lp_encoder_type=encoder, vp_frames=5).eval()
    assert list(legacy.named_parameters()) != []
    assert list(dict(legacy.named_parameters())) == list(dict(current.named_parameters()))
    current.load_state_dict(legacy.state_dict(), strict=True)
    obs = _obs(current, mask=1.)
    with torch.inference_mode():
        expected = legacy.act_inference(obs)
        actual = current.act_inference(obs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(current.predict_velocity()['lp'], legacy.predict_velocity()['lp'], atol=0, rtol=0)
        torch.testing.assert_close(current.predict_lp_terrain(), legacy.predict_lp_terrain(), atol=0, rtol=0)
        torch.testing.assert_close(current.predict_lp_feet_height(), legacy.predict_lp_feet_height(), atol=0, rtol=0)
        obs[:, -1] = 0.
        assert not torch.allclose(current.act_inference(obs), legacy.act_inference(obs))


@pytest.mark.parametrize('frames', [0, 11, -1, 5.0, True])
def test_invalid_vp_history_is_rejected(frames):
    with pytest.raises(ValueError, match='vp_proprio_history_frames'):
        _make_policy(vp_frames=frames)


def test_checkpoint_history_contract_roundtrip_and_legacy_migration():
    current = _make_policy().get_vp_encoder_contract()
    assert current['proprio_history_frames'] == 5
    assert validate_renet_vp_checkpoint_contract(current, current, 10) == []
    legacy = {**current, 'proprio_history_frames': 10}
    for saved in (None, legacy):
        messages = validate_renet_vp_checkpoint_contract(saved, current, 10)
        assert len(messages) == 1 and 'from 10 to 5' in messages[0]
        assert 'VP actions may change' in messages[0]
    for incompatible in ({**current, 'depth_history_frames': 3},
                         {**current, 'unexpected': True},
                         {**current, 'proprio_history_frames': 0}):
        with pytest.raises(ValueError):
            validate_renet_vp_checkpoint_contract(incompatible, current, 10)


def test_exported_policy_keeps_vp_five_frame_semantics_and_dynamic_batch(tmp_path):
    path = Path(__file__).parents[2] / 'legged_lab/scripts/export.py'
    spec = importlib.util.spec_from_file_location('renet_history_export', path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    policy = _make_policy(lp_encoder_type='ame_point_cross_attention').eval()
    output = exporter.export_policy_as_jit(
        policy, None, str(tmp_path), metadata={'vp_encoder_contract': policy.get_vp_encoder_contract()})
    metadata = {'renet_contract.json': ''}
    traced = torch.jit.load(output, _extra_files=metadata).eval()
    assert b'"proprio_history_frames": 5' in metadata['renet_contract.json']
    for batch in (1, 3):
        for mode in (0., 1.):
            obs = _obs(policy, batch_size=batch, mask=mode)
            changed = obs.clone()
            changed[:, :5 * 78] += 10.
            with torch.inference_mode():
                torch.testing.assert_close(traced(obs), policy.act_inference(obs), rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(traced(obs), traced(changed), atol=0, rtol=0)
