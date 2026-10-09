from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import torch

from rsl_rl.algorithms import RENetAMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.runners import RENetAmpOnPolicyRunner
from rsl_rl.utils.renet_checkpoint import validate_renet_critic_checkpoint_contract
from test_g1_amp_format_switch import _assert_optimizer_equal, _env, _train_cfg, _update_runner
from test_renet_auxiliary_updates import _Expert
from test_renet_lp_vp_training import _make_policy, _obs


def test_five_frame_critic_and_every_auxiliary_task_train_with_latest_frame_labels():
    torch.manual_seed(115)
    policy = _make_policy(critic_frames=5, vp_critic_frames=5,
                          lp_encoder_type='ame_point_cross_attention')
    assert policy.critic[0].in_features == 612
    algorithm = RENetAMPPPO(
        policy, Discriminator(100, 1., [16, 8], 'cpu', .5), _Expert(), None,
        amp_replay_buffer_size=64, num_mini_batches=2, num_learning_epochs=1,
        single_critic_dim=85, critic_history_len=5, vel_in_critic_offset=78,
        feet_height_in_critic_offset=83, vel_estimation_coef=1.,
        terrain_recon_coef=.5, lp_terrain_recon_coef=.5,
        terrain_recon_target_normalization='fixed',
        feet_height_coef=.2, lp_feet_height_coef=.2,
        use_feet_height_valid_mask=True, use_terrain_height_valid_mask=True,
    )
    assert algorithm.vel_obs_start_idx == 418
    assert algorithm.feet_height_obs_start_idx == 423
    algorithm.init_storage('rl', 2, 2, [policy.num_actor_obs], [612], [23])
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    with torch.no_grad():
        for step in range(2):
            actor = _obs(policy)
            actor[:, -1] = torch.tensor([float(step), float(1 - step)])
            critic = torch.randn(2, 612) * .1
            critic[:, 418:421] = torch.tensor([.3, .1, -.05])
            critic[:, 423:425] = torch.tensor([.12, .08])
            algorithm.act(actor, critic, torch.randn(2, 50),
                          feet_height_valid=torch.ones(2, 2, dtype=torch.bool),
                          terrain_height_valid=torch.ones(2, 187, dtype=torch.bool))
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, torch.randn(2, 50))
        algorithm.compute_returns(critic, last_actor_obs=actor)
    for batch in algorithm.storage.mini_batch_generator(2, 1):
        assert batch[1].shape[1] == 612
        torch.testing.assert_close(batch[1][:, 418:421], torch.tensor([.3, .1, -.05]).expand(2, -1))
        torch.testing.assert_close(batch[1][:, 423:425], torch.tensor([.12, .08]).expand(2, -1))
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    for key in ('vel_estimation/lp', 'vel_estimation/vp', 'terrain_recon', 'terrain_recon/lp_full',
                'feet_height/vp', 'feet_height/lp'):
        assert losses[key] > 0
    for prefix in ('critic.', 'lp_vel_estimator.', 'vp_vel_estimator.', 'terrain_decoder.',
                   'lp_terrain_decoder.', 'feet_height_decoder.', 'lp_feet_height_decoder.'):
        assert any(not torch.equal(before[name], value) for name, value in policy.named_parameters()
                   if name.startswith(prefix))


@pytest.mark.parametrize('saved_vp_frames', [5, 10])
@pytest.mark.parametrize('has_metadata', [False, True])
def test_runner_crops_old_critic_weights_and_adam_then_can_continue_training(
    saved_vp_frames, has_metadata, tmp_path, capsys,
):
    old_env = _env('g1_renet', 50, critic_frames=10)
    old_cfg = _train_cfg('g1_renet', 50, critic_frames=10, vp_critic_frames=saved_vp_frames)
    old = RENetAmpOnPolicyRunner(old_env, old_cfg, device='cpu')
    _update_runner(old, old_env)
    old.logger_type = 'tensorboard'
    path = tmp_path / 'ten_frame_critic.pt'
    old.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    if not has_metadata:
        checkpoint.pop('critic_observation_contract')
        torch.save(checkpoint, path)
    current_env = _env('g1_renet', 50)
    current = RENetAmpOnPolicyRunner(current_env, _train_cfg('g1_renet', 50), device='cpu')
    current.load(str(path))
    assert 'Critic history changed from 10 to 5' in capsys.readouterr().out
    assert current.alg.policy.critic[0].in_features == 612
    for name, value in old.alg.policy.state_dict().items():
        expected = value[:, 425:] if name == 'critic.0.weight' else value
        torch.testing.assert_close(current.alg.policy.state_dict()[name], expected, atol=0, rtol=0)
    expected_optimizer = copy.deepcopy(old.alg.optimizer.state_dict())
    names = list(dict(old.alg.policy.named_parameters()))
    weight_id = expected_optimizer['param_groups'][0]['params'][names.index('critic.0.weight')]
    for key in ('exp_avg', 'exp_avg_sq'):
        expected_optimizer['state'][weight_id][key] = expected_optimizer['state'][weight_id][key][:, 425:].clone()
    _assert_optimizer_equal(current.alg.optimizer.state_dict(), expected_optimizer)
    actor = _obs(current.alg.policy)
    actor[:, -1] = torch.tensor([0., 1.])
    critic = torch.randn(2, 612)
    padded = torch.cat((torch.zeros(2, 425), critic), dim=-1)
    with torch.no_grad():
        torch.testing.assert_close(current.alg.policy.evaluate(critic),
                                   old.alg.policy.evaluate(padded, actor_observations=actor), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(current.alg.policy.act_inference(actor),
                                   old.alg.policy.act_inference(actor), atol=0, rtol=0)
    # The checkpoint is untouched, and another optimizer step must work after cropping.
    assert torch.load(path, weights_only=False)['model_state_dict']['critic.0.weight'].shape[1] == 1037
    _update_runner(current, current_env)
    current.logger_type = 'tensorboard'
    current.save(str(tmp_path / 'five_frame_critic.pt'))
    restored = RENetAmpOnPolicyRunner(current_env, _train_cfg('g1_renet', 50), device='cpu')
    restored.load(str(tmp_path / 'five_frame_critic.pt'))
    _assert_optimizer_equal(restored.alg.optimizer.state_dict(), current.alg.optimizer.state_dict())


def test_critic_normalizer_is_cropped_together_with_the_observation_layout(tmp_path):
    old_cfg = _train_cfg('g1_renet', 50, critic_frames=10)
    old_cfg['empirical_normalization'] = True
    old = RENetAmpOnPolicyRunner(_env('g1_renet', 50, critic_frames=10), old_cfg, device='cpu')
    old.privileged_obs_normalizer(torch.randn(3, 1037))
    old.obs_normalizer(torch.randn(3, 2120))
    old.logger_type = 'tensorboard'
    path = tmp_path / 'normalized_ten_frames.pt'
    old.save(str(path))
    current_cfg = _train_cfg('g1_renet', 50)
    current_cfg['empirical_normalization'] = True
    current = RENetAmpOnPolicyRunner(_env('g1_renet', 50), current_cfg, device='cpu')
    current.load(str(path))
    for key, value in old.privileged_obs_normalizer.state_dict().items():
        expected = value[:, 425:] if key != 'count' else value
        torch.testing.assert_close(current.privileged_obs_normalizer.state_dict()[key], expected, atol=0, rtol=0)
    for key, value in old.obs_normalizer.state_dict().items():
        torch.testing.assert_close(current.obs_normalizer.state_dict()[key], value, atol=0, rtol=0)


def test_critic_contract_rejects_unrelated_layout_changes_and_false_metadata():
    current = _make_policy(critic_frames=5, vp_critic_frames=5).get_critic_observation_contract()
    assert validate_renet_critic_checkpoint_contract(current, current, 5) == []
    previous = {**current, 'history_frames': 10, 'vp_history_frames': 5}
    assert '10 to 5' in validate_renet_critic_checkpoint_contract(previous, current, 10)[0]
    for saved, observed_frames in ((current, 10), ({**previous, 'single_frame_dim': 84}, 10),
                                   ({**previous, 'history_frames': 8}, 8)):
        with pytest.raises(ValueError):
            validate_renet_critic_checkpoint_contract(saved, current, observed_frames)


def test_production_cfg_keeps_buffer_policy_and_auxiliary_label_history_in_sync():
    path = Path(__file__).parents[2] / 'legged_lab/envs/g1/RENet_cfg.py'
    tree = ast.parse(path.read_text())
    env = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1RENETENVCFG')
    robot = next(n.value for n in env.body if isinstance(n, ast.AnnAssign) and n.target.id == 'robot')
    assert next(ast.literal_eval(k.value) for k in robot.keywords if k.arg == 'critic_obs_history_length') == 10
    agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1RENETAGENTCFG')
    for name, argument in (('policy', 'critic_history_frames'), ('algorithm', 'critic_history_len')):
        call = next(n.value for n in agent.body if isinstance(n, ast.Assign) and n.targets[0].id == name)
        assert next(ast.literal_eval(k.value) for k in call.keywords if k.arg == argument) == 10
