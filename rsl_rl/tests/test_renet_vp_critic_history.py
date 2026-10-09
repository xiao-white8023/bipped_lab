from __future__ import annotations

import ast
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


def test_mixed_critic_values_and_gradients_use_only_the_selected_history():
    torch.manual_seed(112)
    policy = _make_policy(vp_critic_frames=5).eval()
    actor = _obs(policy, batch_size=2)
    actor[:, -1] = torch.tensor([0., 1.])
    critic = torch.randn(2, 1037, requires_grad=True)
    older = critic.detach().clone()
    older[:, :5 * 85] += 10.
    recent = critic.detach().clone()
    recent[:, 5 * 85:10 * 85] += 1.
    terrain = critic.detach().clone()
    terrain[:, 10 * 85:] += 1.
    baseline = policy.evaluate(critic, actor_observations=actor)
    changed_old = policy.evaluate(older, actor_observations=actor)
    torch.testing.assert_close(baseline[0], changed_old[0], atol=0, rtol=0)
    assert not torch.allclose(baseline[1], changed_old[1])
    for changed in (recent, terrain):
        assert not torch.allclose(baseline[0], policy.evaluate(changed, actor_observations=actor)[0])
    baseline.sum().backward()
    assert torch.count_nonzero(critic.grad[0, :5 * 85]) == 0
    assert torch.count_nonzero(critic.grad[0, 5 * 85:10 * 85]) > 0
    assert torch.count_nonzero(critic.grad[1, :5 * 85]) > 0
    # Ignored history must not contaminate VP values even if its entries are NaN.
    older[0, :5 * 85] = float('nan')
    assert torch.isfinite(policy.evaluate(older, actor_observations=actor)[0]).all()


def test_vp_window_preserves_lp_values_and_parameter_layout():
    previous = _make_policy(vp_critic_frames=10).eval()
    current = _make_policy(vp_critic_frames=5).eval()
    current.load_state_dict(previous.state_dict(), strict=True)
    critic = torch.randn(3, 1037)
    actor = _obs(current, batch_size=3, mask=1.)
    with torch.no_grad():
        torch.testing.assert_close(current.evaluate(critic, actor_observations=actor),
                                   previous.evaluate(critic), atol=0, rtol=0)
        torch.testing.assert_close(current.act_inference(actor), previous.act_inference(actor), atol=0, rtol=0)
    with pytest.raises(ValueError, match='matching actor observations'):
        current.evaluate(critic)


@pytest.mark.parametrize('frames', [0, 11, -1, 5.0, True])
def test_invalid_vp_critic_windows_are_rejected(frames):
    with pytest.raises(ValueError, match='vp_critic_history_frames'):
        _make_policy(vp_critic_frames=frames)


def test_real_ppo_collection_bootstrap_and_shuffled_updates_use_matching_modes():
    torch.manual_seed(113)
    policy = _make_policy(vp_critic_frames=5)
    algorithm = RENetAMPPPO(
        policy, Discriminator(100, 1., [16, 8], 'cpu', .5), _Expert(), None,
        amp_replay_buffer_size=64, num_mini_batches=2, num_learning_epochs=1,
        single_critic_dim=85, critic_history_len=10,
    )
    algorithm.init_storage('rl', 2, 2, [policy.num_actor_obs], [1037], [23])
    evaluate = policy.evaluate
    seen = []

    def capture(critic, actor_observations=None, **kwargs):
        assert actor_observations is not None
        torch.testing.assert_close(critic[:, 0], actor_observations[:, 0], atol=0, rtol=0)
        seen.append(actor_observations[:, -1].detach().clone())
        return evaluate(critic, actor_observations=actor_observations, **kwargs)

    policy.evaluate = capture
    with torch.no_grad():
        for step in range(2):
            actor = _obs(policy)
            actor[:, -1] = torch.tensor([float(step), float(1 - step)])
            actor[:, 0] = torch.tensor([step * 2., step * 2. + 1.])
            critic = torch.randn(2, 1037)
            critic[:, 0] = actor[:, 0]
            algorithm.act(actor, critic, torch.randn(2, 50))
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, torch.randn(2, 50))
        # The bootstrap mode differs from the preceding action's mode.
        bootstrap_actor = actor.clone()
        bootstrap_actor[:, -1] = torch.tensor([0., 1.])
        algorithm.compute_returns(critic, last_actor_obs=bootstrap_actor)
    torch.testing.assert_close(seen[2], bootstrap_actor[:, -1])
    before = policy.critic[0].weight.detach().clone()
    losses = algorithm.update()
    assert len(seen) == 5  # Two collection calls, bootstrap, two shuffled batches.
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(before, policy.critic[0].weight)


def test_checkpoint_contract_accepts_explicit_legacy_window_migration_only():
    current = _make_policy(vp_critic_frames=5).get_critic_observation_contract()
    assert validate_renet_critic_checkpoint_contract(current, current) == []
    previous = {**current, 'vp_history_frames': 10}
    for saved in (None, previous):
        messages = validate_renet_critic_checkpoint_contract(saved, current)
        assert len(messages) == 1 and 'from 10 to 5' in messages[0]
    for incompatible in ({**current, 'terrain_dim': 99}, {**current, 'unexpected': True},
                         {**current, 'vp_history_frames': 0}):
        with pytest.raises(ValueError):
            validate_renet_critic_checkpoint_contract(incompatible, current)


def test_runner_restores_legacy_critic_and_adam_without_changing_lp_values(tmp_path, capsys):
    env = _env('g1_renet', 50, critic_frames=10)
    old_cfg = _train_cfg('g1_renet', 50, critic_frames=10)
    old_cfg['policy']['vp_critic_history_frames'] = 10
    old = RENetAmpOnPolicyRunner(env, old_cfg, device='cpu')
    _update_runner(old, env)
    old.logger_type = 'tensorboard'
    path = tmp_path / 'legacy_critic.pt'
    old.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.pop('critic_observation_contract')
    torch.save(checkpoint, path)
    current = RENetAmpOnPolicyRunner(env, _train_cfg('g1_renet', 50, critic_frames=10), device='cpu')
    current.load(str(path))
    assert 'VP critic history changed from 10 to 5' in capsys.readouterr().out
    _assert_optimizer_equal(current.alg.optimizer.state_dict(), old.alg.optimizer.state_dict())
    for name, value in old.alg.policy.state_dict().items():
        torch.testing.assert_close(current.alg.policy.state_dict()[name], value, atol=0, rtol=0)
    actor = _obs(current.alg.policy, mask=1.)
    critic = torch.randn(2, 1037)
    with torch.no_grad():
        torch.testing.assert_close(current.alg.policy.evaluate(critic, actor_observations=actor),
                                   old.alg.policy.evaluate(critic), atol=0, rtol=0)
        actor[:, -1] = 0.
        assert not torch.allclose(current.alg.policy.evaluate(critic, actor_observations=actor),
                                  old.alg.policy.evaluate(critic))


def test_production_cfg_has_one_vp_actor_window_setting_and_ten_frame_critic():
    path = Path(__file__).parents[2] / 'legged_lab/envs/g1/RENet_cfg.py'
    tree = ast.parse(path.read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'G1RENETAGENTCFG')
    policy = next(node.value for node in agent.body if isinstance(node, ast.Assign) and node.targets[0].id == 'policy')
    arguments = {keyword.arg: keyword.value for keyword in policy.keywords}
    setting = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'G1_RENET_VP_PROPRIO_HISTORY_FRAMES')
    assert 1 <= ast.literal_eval(setting) <= 10
    assert arguments['vp_proprio_history_frames'].id == 'G1_RENET_VP_PROPRIO_HISTORY_FRAMES'
    policy_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'CustomRslRlPpoActorCriticCfg')
    default = next(node.value for node in policy_class.body if isinstance(node, ast.AnnAssign)
                   and node.target.id == 'vp_proprio_history_frames')
    assert default.id == 'G1_RENET_VP_PROPRIO_HISTORY_FRAMES'
    assert ast.literal_eval(arguments['vp_critic_history_frames']) == 10
    assert ast.literal_eval(arguments['critic_history_frames']) == 10
