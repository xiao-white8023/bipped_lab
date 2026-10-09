from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsl_rl.modules import Discriminator
from rsl_rl.utils import AMPLoader, G1AnkleAMPLoader, Normalizer
from rsl_rl.utils.renet_checkpoint import upgrade_renet_auxiliary_checkpoint
from test_renet_lp_vp_training import _make_policy, _obs

ROOT = Path(__file__).parents[2]
DATASETS = ROOT / 'legged_lab/envs/g1/datasets'
FULL = DATASETS / 'g1_23dof_locomotion_v3'
LEGACY = DATASETS / 'g1_23dof_locomotion_v2'


def test_ankle_dataset_preserves_reviewed_motion_and_existing_amp_features():
    manifest = json.loads((FULL / 'manifest.json').read_text())
    assert manifest['amp_joint_count'] == 23 and manifest['amp_dimension'] == 58
    assert manifest['includes_ankles']
    ankle = [4, 5, 10, 11]
    non_ankle = [i for i in range(23) if i not in ankle]
    for name in manifest['clips']:
        full = np.asarray(json.loads((FULL / 'amp' / f'{name}.txt').read_text())['Frames'])
        old = np.asarray(json.loads((LEGACY / 'amp' / f'{name}.txt').read_text())['Frames'])
        visual = np.asarray(json.loads((FULL / 'visualization' / f'{name}.txt').read_text())['Frames'])
        np.testing.assert_array_equal(full[:, :23], visual[:, 6:29])
        np.testing.assert_array_equal(full[:, 23:46], visual[:, 35:58])
        np.testing.assert_array_equal(full[:, non_ankle], old[:, :19])
        np.testing.assert_array_equal(full[:, np.asarray(non_ankle) + 23], old[:, 19:38])
        np.testing.assert_array_equal(full[:, 46:], old[:, 38:])
        assert np.any(np.ptp(full[:, ankle], axis=0) > .01)
        assert np.any(np.abs(full[:, np.asarray(ankle) + 23]) > .01)
        assert (FULL / 'visualization' / f'{name}.txt').read_bytes() == (LEGACY / 'visualization' / f'{name}.txt').read_bytes()
        assert (FULL / 'sources' / f'{name}.pkl').read_bytes() == (LEGACY / 'sources' / f'{name}.pkl').read_bytes()


def test_full_loader_slices_joint_velocity_and_interpolates_ankles():
    loader = G1AnkleAMPLoader('cpu', .02, motion_files=[str(FULL / 'amp/walk_backward.txt')],
                            preload_transitions=True, num_preload_transitions=64)
    assert loader.observation_dim == 58 and loader.preloaded_s.shape == (64, 58)
    frame = loader.get_full_frame_at_time(0, 3.007)
    assert loader.get_joint_pose(frame).shape == (23,)
    assert loader.get_joint_vel(frame).shape == (23,)
    torch.testing.assert_close(loader.get_joint_vel(frame), frame[23:46])
    torch.testing.assert_close(loader.get_end_pos(frame), frame[46:58])
    torch.testing.assert_close(loader.get_joint_pose_batch(frame[None]), frame[None, :23])
    data = np.asarray(json.loads((FULL / 'amp/walk_backward.txt').read_text())['Frames'])
    expected = np.array([np.interp(3.007, np.arange(len(data))/30, data[:, j]) for j in range(58)])
    np.testing.assert_allclose(frame.numpy(), expected, atol=2e-6)


def test_equal_width_visualization_is_rejected_and_legacy_loader_is_unchanged():
    with pytest.raises(ValueError, match='not a visualization frame'):
        G1AnkleAMPLoader('cpu', .02, motion_files=[str(FULL / 'visualization/walk_backward.txt')])
    old = AMPLoader('cpu', .02, motion_files=[str(LEGACY / 'amp/walk_backward.txt')])
    assert old.observation_dim == 50
    assert old.get_joint_pose(old.get_full_frame_at_time(0, 0)).shape == (19,)
    with pytest.raises(ValueError, match='must contain 50'):
        AMPLoader('cpu', .02, motion_files=[str(FULL / 'amp/walk_backward.txt')])


def test_loader_rejects_mixed_joint_order(tmp_path):
    first = FULL / 'amp/walk_backward.txt'
    changed = json.loads(first.read_text())
    changed['AMPJointNames'] = changed['AMPJointNames'][::-1]
    second = tmp_path / 'reordered.txt'
    second.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='joint order differs'):
        G1AnkleAMPLoader('cpu', .02, motion_files=[str(first), str(second)])


@pytest.mark.parametrize('old_dim,new_dim', [(50, 58), (58, 50)])
def test_50_58_checkpoint_switch_resumes_policy_but_resets_amp_and_its_adam_state(old_dim, new_dim):
    old_policy = _make_policy().eval()
    old_disc = Discriminator(2 * old_dim, .3, [16, 8], 'cpu')
    def optimizer(policy, discriminator):
        return torch.optim.Adam([{'params': policy.parameters(), 'name': 'policy'},
                                 {'params': discriminator.trunk.parameters(), 'name': 'amp_trunk'},
                                 {'params': discriminator.amp_linear.parameters(), 'name': 'amp_head'}])
    old_optimizer = optimizer(old_policy, old_disc)
    sum(p.square().sum() for group in old_optimizer.param_groups for p in group['params']).backward()
    old_optimizer.step()
    aux_contract = {'vp_terrain_target_normalization': 'fixed', 'terrain_target_clip': 1.}
    saved = {'model_state_dict': old_policy.state_dict(), 'optimizer_state_dict': old_optimizer.state_dict(),
             'discriminator_state_dict': old_disc.state_dict(), 'amp_normalizer': Normalizer(old_dim),
             'auxiliary_training_contract': aux_contract}
    policy = _make_policy().eval()
    discriminator = Discriminator(2 * new_dim, .3, [16, 8], 'cpu')
    opt = optimizer(policy, discriminator)
    normalizer = Normalizer(new_dim)
    fresh_weights = {name: value.clone() for name, value in discriminator.state_dict().items()}
    amp_contract = {'dimension': new_dim}
    upgraded, messages = upgrade_renet_auxiliary_checkpoint(saved, policy, opt, aux_contract,
                                                            discriminator=discriminator,
                                                            amp_normalizer=normalizer, amp_contract=amp_contract)
    change = 'AMP upgraded from 50 to 58' if new_dim == 58 else 'AMP switched from 58 to 50'
    assert any(change in message for message in messages)
    policy.load_state_dict(upgraded['model_state_dict'])
    discriminator.load_state_dict(upgraded['discriminator_state_dict'])
    opt.load_state_dict(upgraded['optimizer_state_dict'])
    assert upgraded['amp_normalizer'] is normalizer
    for mode in [0., 1.]:
        obs = _obs(policy, mask=mode)
        with torch.no_grad():
            torch.testing.assert_close(policy.act_inference(obs), old_policy.act_inference(obs), rtol=0, atol=0)
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(opt.state[parameter]['exp_avg'], old_optimizer.state[dict(old_policy.named_parameters())[name]]['exp_avg'])
    for parameter in discriminator.parameters():
        assert parameter not in opt.state
    for name, value in discriminator.state_dict().items():
        torch.testing.assert_close(value, fresh_weights[name])
    opt.zero_grad()
    sum(p.square().sum() for group in opt.param_groups for p in group['params']).backward()
    opt.step()
    upgraded['optimizer_state_dict'] = opt.state_dict()
    upgraded['model_state_dict'] = policy.state_dict()
    upgraded['discriminator_state_dict'] = discriminator.state_dict()
    _, messages = upgrade_renet_auxiliary_checkpoint(upgraded, policy, opt, aux_contract,
                                                    discriminator=discriminator, amp_normalizer=normalizer,
                                                    amp_contract=amp_contract)
    assert not messages
