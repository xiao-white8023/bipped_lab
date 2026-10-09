from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.utils.renet_checkpoint import upgrade_renet_auxiliary_checkpoint
from test_renet_lp_vp_training import _has_nonzero_grad, _has_zero_grad, _make_policy, _obs


ROOT = Path(__file__).parents[2]


def _extract_function(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"torch": torch, "Articulation": object,
             "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def test_vp_idle_cost_covers_backward_lateral_turn_and_exempts_lp_standing():
    reward = _extract_function(ROOT / "legged_lab/mdp/rewards.py", "vp_idle_when_commanded")
    command = torch.tensor([[.5, 0, 0], [-.5, 0, 0], [0, .5, 0], [0, -.5, 0],
                            [0, 0, .8], [0, 0, 0], [.05, 0, .05], [.5, 0, 0],
                            [.5, 0, 0], [0, 0, -.8]])
    velocity = torch.zeros(10, 3)
    velocity[8, 0] = .3
    angular_velocity = torch.zeros(10, 3)
    angular_velocity[9, 2] = -.5
    mask = torch.zeros(10, 1)
    mask[7] = 1
    env = SimpleNamespace(command_generator=SimpleNamespace(command=command),
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(
                              root_lin_vel_w=velocity, root_ang_vel_w=angular_velocity))},
                          renet_estimator_mask=mask)
    torch.testing.assert_close(reward(env), torch.tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]))
    # Simulate a resample at the end of the step: judge the command/mode that acted.
    env._renet_reward_command = command.clone()
    env._renet_reward_mask = mask[:, 0].clone()
    env.command_generator.command = torch.zeros_like(command)
    env.renet_estimator_mask.fill_(1)
    assert reward(env)[0].item() == 1


def test_fixed_terrain_target_is_batch_independent_and_keeps_metric_scale():
    algorithm = AMPPPO.__new__(AMPPPO)
    algorithm.terrain_recon_target_clip = 1.0
    algorithm.terrain_recon_target_normalization = "fixed"
    sample = torch.tensor([.1, .2, .3], requires_grad=True)
    batch_a = torch.stack((sample, sample))
    batch_b = torch.stack((sample, torch.tensor([.6, .7, .8])))
    target_a = algorithm._prepare_vp_terrain_target(batch_a)
    target_b = algorithm._prepare_vp_terrain_target(batch_b)
    torch.testing.assert_close(target_a[0], sample.detach())
    torch.testing.assert_close(target_b[0], target_a[0])
    assert not target_a.requires_grad
    torch.testing.assert_close(algorithm._prepare_vp_terrain_target(torch.tensor([-2., 2.])), torch.tensor([-1., 1.]))
    # Other AMP tasks retain their legacy opt-in/default batch target behavior.
    algorithm.terrain_recon_target_normalization = "batch"
    assert not torch.allclose(algorithm._prepare_vp_terrain_target(batch_a)[0], algorithm._prepare_vp_terrain_target(batch_b)[0])


def test_feet_height_uses_valid_hits_and_masks_all_invalid_labels():
    target_fn = _extract_function(ROOT / "legged_lab/envs/g1/RENet_env.py", "compute_feet_height_target", "G1RENetEnv")
    def sensor(hits):
        return SimpleNamespace(data=SimpleNamespace(pos_w=torch.tensor([[0., 0., .1], [0., 0., .1]]),
                                                    ray_hits_w=torch.stack((torch.zeros(2, 2), torch.zeros(2, 2), hits), -1)))
    env = SimpleNamespace(num_envs=2, device="cpu",
                          left_feet_ray_caster=sensor(torch.tensor([[0., float("inf")], [float("inf"), float("nan")]])),
                          right_feet_ray_caster=sensor(torch.zeros(2, 2)))
    targets = target_fn(env)
    torch.testing.assert_close(targets, torch.tensor([[.1, .1], [0., .1]]))
    assert env.feet_height_valid.tolist() == [[True, True], [False, True]]
    prediction = torch.tensor([[.2, .2], [10., .2]], requires_grad=True)
    loss = AMPPPO._feet_height_mse(prediction, targets, env.feet_height_valid)
    torch.testing.assert_close(loss, torch.tensor(.01))
    loss.backward()
    assert prediction.grad[1, 0].item() == 0
    prediction.grad = None
    all_invalid = torch.zeros_like(env.feet_height_valid)
    loss = AMPPPO._feet_height_mse(prediction, torch.full_like(targets, float("nan")), all_invalid)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 0
    env.left_feet_ray_caster = None
    assert torch.count_nonzero(target_fn(env)) == 0
    assert not env.feet_height_valid.any()


@pytest.mark.parametrize("encoder", ["ame_point_cross_attention"])
@pytest.mark.parametrize("mode", [0., 1.])
def test_lp_feet_learning_updates_lp_not_vp_encoder(encoder, mode):
    torch.manual_seed(81)
    policy = _make_policy(lp_encoder_type=encoder)
    policy._process_actor_obs(_obs(policy, mask=mode))
    prediction = policy.predict_lp_feet_height()
    assert prediction.shape == (2, 2)
    (prediction - .12).square().mean().backward()
    assert _has_nonzero_grad(policy.lp_feet_height_decoder)
    assert _has_nonzero_grad(policy.lidar_cnn)
    assert _has_nonzero_grad(policy.lp_attention)
    assert _has_zero_grad(policy.vp_gru)
    assert _has_zero_grad(policy.feet_height_decoder)


class _Expert:
    def feed_forward_generator(self, batches, batch_size):
        for _ in range(batches):
            yield torch.randn(batch_size, 50), torch.randn(batch_size, 50)


@pytest.mark.parametrize("valid_labels", [True, False])
def test_real_amp_ppo_update_aligns_validity_and_trains_both_feet_heads(valid_labels):
    torch.manual_seed(89)
    policy = _make_policy(lp_encoder_type="ame_point_cross_attention")
    discriminator = Discriminator(100, 1., [16, 8], "cpu", .5)
    algorithm = AMPPPO(policy, discriminator, _Expert(), None, amp_replay_buffer_size=64,
                       num_mini_batches=2, num_learning_epochs=1, learning_rate=1e-3,
                       terrain_recon_coef=.5, terrain_recon_target_normalization="fixed",
                       lp_terrain_recon_coef=.5, vel_estimation_coef=1., single_critic_dim=85,
                       critic_history_len=10, vel_in_critic_offset=78, feet_height_in_critic_offset=83,
                       feet_height_coef=.2, lp_feet_height_coef=.2, use_feet_height_valid_mask=True)
    algorithm.init_storage("rl", 2, 2, [policy.num_actor_obs], [1037], [23])
    before_lp = [p.detach().clone() for p in policy.lp_feet_height_decoder.parameters()]
    before_vp = [p.detach().clone() for p in policy.feet_height_decoder.parameters()]
    with torch.no_grad():
        for step in range(2):
            obs = _obs(policy, mask=float(step))
            critic = torch.zeros(2, 1037)
            critic[:, 9 * 85 + 83:9 * 85 + 85] = torch.tensor([.12, .08])
            # Match an observation identifier to its shuffled per-foot valid bits.
            valid = torch.full((2, 2), valid_labels, dtype=torch.bool)
            if valid_labels:
                valid[step, 0] = False
            obs[:, 0:2] = valid.float()
            algorithm.act(obs, critic, torch.randn(2, 50), feet_height_valid=valid)
            valid.fill_(False)  # The environment changes its next-state labels.
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, torch.randn(2, 50))
        algorithm.compute_returns(torch.zeros(2, 1037))
    for sample in algorithm.storage.mini_batch_generator(2, 1, include_feet_height_valid=True):
        torch.testing.assert_close(sample[0][:, :2].bool(), sample[-1])
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    changed_lp = any(not torch.equal(a, b) for a, b in zip(before_lp, policy.lp_feet_height_decoder.parameters(), strict=True))
    changed_vp = any(not torch.equal(a, b) for a, b in zip(before_vp, policy.feet_height_decoder.parameters(), strict=True))
    assert changed_lp == valid_labels and changed_vp == valid_labels
    if valid_labels:
        assert losses["feet_height/lp"] > 0 and losses["feet_height/vp"] > 0
        assert losses["feet_height/valid_fraction"] == .75
    else:
        assert losses["feet_height/lp"] == losses["feet_height/vp"] == 0


def _optimizer(policy, discriminator):
    return torch.optim.Adam([{"name": "policy", "params": policy.parameters()},
                             {"name": "amp_trunk", "params": discriminator[0].parameters()},
                             {"name": "amp_head", "params": discriminator[1].parameters()}])


def test_legacy_checkpoint_preserves_actions_and_adam_slots_when_adding_lp_head():
    torch.manual_seed(91)
    old = _make_policy(lp_encoder_type="ame_point_cross_attention", lp_feet=False).eval()
    discriminator = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    old_optimizer = _optimizer(old, discriminator)
    sum(p.square().sum() for group in old_optimizer.param_groups for p in group["params"]).backward()
    old_optimizer.step()
    checkpoint = {"model_state_dict": old.state_dict(), "optimizer_state_dict": old_optimizer.state_dict()}
    new = _make_policy(lp_encoder_type="ame_point_cross_attention").eval()
    new_discriminator = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    new_discriminator.load_state_dict(discriminator.state_dict())
    new_optimizer = _optimizer(new, new_discriminator)
    fresh_decoder = {name: value.clone() for name, value in new.terrain_decoder.state_dict().items()}
    contract = {"vp_terrain_target_normalization": "fixed", "terrain_target_clip": 1.}
    upgraded, messages = upgrade_renet_auxiliary_checkpoint(checkpoint, new, new_optimizer, contract)
    assert len(messages) == 3
    new.load_state_dict(upgraded["model_state_dict"], strict=True)
    new_optimizer.load_state_dict(upgraded["optimizer_state_dict"])
    for mode in [0., 1.]:
        observation = _obs(old, mask=mode)
        with torch.no_grad():
            torch.testing.assert_close(old.act_inference(observation), new.act_inference(observation), rtol=0, atol=0)
    for name, parameter in new.named_parameters():
        if name.startswith(("terrain_decoder.", "lp_feet_height_decoder.")):
            assert parameter not in new_optimizer.state
        else:
            torch.testing.assert_close(new_optimizer.state[parameter]["exp_avg"], old_optimizer.state[dict(old.named_parameters())[name]]["exp_avg"])
    for old_parameter, new_parameter in zip(discriminator.parameters(), new_discriminator.parameters(), strict=True):
        torch.testing.assert_close(old_optimizer.state[old_parameter]["exp_avg"], new_optimizer.state[new_parameter]["exp_avg"])
    for name, value in new.terrain_decoder.state_dict().items():
        torch.testing.assert_close(value, fresh_decoder[name])
    new_optimizer.zero_grad()
    sum(p.square().sum() for group in new_optimizer.param_groups for p in group["params"]).backward()
    new_optimizer.step()  # All restored and new Adam slots must be usable.
    current_checkpoint = {"model_state_dict": new.state_dict(), "optimizer_state_dict": new_optimizer.state_dict(),
                          "policy_parameter_names": list(dict(new.named_parameters())), "auxiliary_training_contract": contract}
    unchanged, messages = upgrade_renet_auxiliary_checkpoint(current_checkpoint, new, new_optimizer, contract)
    assert not messages
    assert unchanged["optimizer_state_dict"] is current_checkpoint["optimizer_state_dict"]
    for name in new.state_dict():
        torch.testing.assert_close(unchanged["model_state_dict"][name], current_checkpoint["model_state_dict"][name])
    broken = {**checkpoint, "model_state_dict": {key: value for key, value in checkpoint["model_state_dict"].items() if key != "actor.0.weight"}}
    with pytest.raises(ValueError, match="Incompatible RENet checkpoint"):
        upgrade_renet_auxiliary_checkpoint(broken, new, new_optimizer, contract)
