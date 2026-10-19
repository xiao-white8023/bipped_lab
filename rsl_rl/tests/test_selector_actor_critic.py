"""独立高层网络的分类分布、观测布局及梯度隔离检查；不启动仿真。"""

from __future__ import annotations

import ast
import importlib.util
import io
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "rsl_rl/rsl_rl/modules/selector_actor_critic.py"
CFG_PATH = ROOT / "legged_lab/envs/g1/g1_selector_cfg.py"

# 直接加载唯一的新实现，不借助现有网络模块或现有测试工具。
spec = importlib.util.spec_from_file_location("_independent_selector_actor_critic", MODEL_PATH)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)
SelectorActorCritic = network.SelectorActorCritic


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _policy_config():
    """只读解析实际配置的字面量，不导入配置、项目网络或仿真 SDK。"""
    tree = ast.parse(CFG_PATH.read_text())
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1SelectorAgentCfg")
    policy = next(node.value for node in agent.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "policy" for target in node.targets))

    def literals(node):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            assert node.func.id in ("SelectorActorCriticCfg", "SelectorDepthEncoderCfg")
            assert not node.args and all(item.arg is not None for item in node.keywords)
            return {item.arg: literals(item.value) for item in node.keywords}
        return ast.literal_eval(node)

    return literals(policy)


def _observations(model, batch=6):
    actor = torch.randn(batch, model.num_actor_obs)
    critic = torch.cat((actor.clone(), torch.randn(batch, model.critic_privileged_dim)), dim=-1)
    return actor, critic


def _parameters(model, branch):
    return [*getattr(model, branch + "_encoder").parameters(), *getattr(model, branch).parameters()]


def test_actual_configuration_constructs_without_existing_network_dependencies():
    cfg = _policy_config()
    model = SelectorActorCritic(num_actor_obs=3466, num_critic_obs=3660, num_actions=1, **cfg)
    actor, critic = _observations(model)
    assert model(actor).shape == (6, 2)
    assert model.evaluate(critic).shape == (6, 1)
    assert model.actor_encoder.output_dim == model.critic_encoder.output_dim == 352
    assert model.actor[0].in_features == 352
    assert model.critic[0].in_features == 546
    assert model.num_actions == 1 and model.num_modes == 2 and not model.is_recurrent
    assert not any("std" in name or "log_std" in name for name, _ in model.named_parameters())


def test_multimodal_layout_preserves_image_order_masks_and_state():
    model = SelectorActorCritic()
    batch = 2
    proprio = torch.full((batch, 780), 0.15)
    depth = torch.stack([torch.full((batch, 18, 32), value) for value in (.2, .6, .7, .9)], dim=1)
    height = torch.cat((torch.full((batch, 187), -.3), torch.ones(batch, 187)), dim=-1)
    state = torch.arange(8).float().repeat(batch, 1) / 8
    obs = torch.cat((proprio, depth.flatten(1), height, state), dim=-1)
    captured = {}
    handles = []
    for name in ("proprio", "depth", "height", "state"):
        def capture(_module, inputs, key=name):
            captured[key] = inputs[0].detach().clone()
        handles.append(getattr(model.actor_encoder, name + "_encoder").register_forward_pre_hook(capture))
    try:
        model(obs)
    finally:
        for handle in handles:
            handle.remove()
    for name, expected in (("proprio", proprio), ("depth", depth), ("height", height), ("state", state)):
        torch.testing.assert_close(captured[name], expected, rtol=0, atol=0)


def test_categorical_probability_log_prob_and_entropy_match_analytic_values():
    model = SelectorActorCritic()
    with torch.no_grad():
        model.actor[-1].weight.zero_()
        model.actor[-1].bias.copy_(torch.tensor([math.log(3), 0.]))
    obs, _ = _observations(model, 4)
    model.update_distribution(obs)
    expected = torch.tensor([[.75, .25]]).repeat(4, 1)
    torch.testing.assert_close(model.action_probs, expected)
    torch.testing.assert_close(model.action_logits, torch.tensor([[math.log(3), 0.]]).repeat(4, 1))
    actions = torch.tensor([[0], [1], [1], [0]])
    log_prob = torch.tensor([.75, .25, .25, .75]).log()
    torch.testing.assert_close(model.get_actions_log_prob(actions), log_prob)
    torch.testing.assert_close(model.get_actions_log_prob(actions[:, 0].int()), log_prob)
    entropy = -(.75 * math.log(.75) + .25 * math.log(.25))
    torch.testing.assert_close(model.entropy, torch.full((4,), entropy))


def test_sampled_modes_are_integer_and_deterministic_inference_uses_argmax():
    model = SelectorActorCritic()
    obs, _ = _observations(model, 64)
    with torch.no_grad():
        model.actor[-1].weight.zero_()
        model.actor[-1].bias.zero_()
    torch.manual_seed(42)
    first = model.act(obs)
    torch.manual_seed(42)
    second = model.act(obs)
    assert first.shape == (64, 1) and first.dtype == torch.int64
    assert set(first.flatten().tolist()) == {0, 1}
    assert torch.equal(first, second)
    distribution = model.distribution
    with torch.no_grad():
        model.actor[-1].bias.copy_(torch.tensor([-1., 2.]))
    chosen = model.act_inference(obs)
    assert chosen.shape == (64, 1) and chosen.dtype == torch.int64
    assert (chosen == 1).all()
    assert model.distribution is distribution


def test_actor_loss_reaches_every_actor_encoder_without_touching_critic():
    model = SelectorActorCritic()
    actor, critic = _observations(model, 4)
    actions = torch.tensor([[0], [1], [0], [1]])
    values, log_prob, entropy = model.evaluate_actions(actor, critic, actions)
    assert values.shape == (4, 1) and log_prob.shape == entropy.shape == (4,)
    loss = -(log_prob * torch.tensor([1., -1., .5, -.5])).mean() - .01 * entropy.mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in _parameters(model, "critic"))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in _parameters(model, "actor"))
    for name in ("proprio", "depth", "height", "state"):
        gradients = [p.grad for p in getattr(model.actor_encoder, name + "_encoder").parameters()]
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_critic_uses_privileged_information_and_has_independent_parameters():
    model = SelectorActorCritic()
    _, critic = _observations(model)
    critic.requires_grad_()
    before_actor = {name: tensor.detach().clone() for name, tensor in model.actor_encoder.state_dict().items()}
    value = model.evaluate(critic)
    value.mean().backward()
    assert critic.grad[:, model.num_actor_obs:].abs().sum() > 0
    assert critic.grad[:, :model.num_actor_obs].abs().sum() > 0
    actor_parameters, critic_parameters = _parameters(model, "actor"), _parameters(model, "critic")
    assert not ({id(p) for p in actor_parameters} & {id(p) for p in critic_parameters})
    assert all(parameter.grad is None for parameter in actor_parameters)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in critic_parameters)
    torch.optim.Adam(critic_parameters, lr=1e-3).step()
    for name, tensor in model.actor_encoder.state_dict().items():
        torch.testing.assert_close(tensor, before_actor[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="actor observations"):
        model(critic.detach())


def test_state_dict_round_trip_and_reset_preserve_deterministic_outputs():
    cfg = _policy_config()
    model = SelectorActorCritic(**cfg)
    actor, critic = _observations(model)
    expected_logits, expected_values = model(actor).detach(), model.evaluate(critic).detach()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = SelectorActorCritic(**cfg)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.reset(torch.ones(len(actor), dtype=torch.bool))
    torch.testing.assert_close(restored(actor), expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(restored.evaluate(critic), expected_values, rtol=0, atol=0)


def test_support_channels_and_privileged_information_can_be_disabled_by_configuration():
    cfg = _policy_config()
    cfg["depth_encoder"]["input_channels"] = 2
    cfg.update(height_input_dim=187, actor_obs_dim=2127, critic_obs_dim=2127, critic_privileged_dim=0)
    model = SelectorActorCritic(**cfg)
    actor, critic = _observations(model)
    assert model(actor).shape == (6, 2)
    assert model.evaluate(critic).shape == (6, 1)


@pytest.mark.parametrize("property_name", ["action_probs", "action_logits", "entropy"])
def test_distribution_properties_require_an_observation(property_name):
    with pytest.raises(RuntimeError, match="update_distribution"):
        getattr(SelectorActorCritic(), property_name)


@pytest.mark.parametrize("actions", [torch.zeros(6, 1), torch.zeros(6, 2, dtype=torch.long),
                                    torch.zeros(5, 1, dtype=torch.long), torch.full((6, 1), 2, dtype=torch.long)])
def test_invalid_actions_are_rejected_instead_of_clipped(actions):
    model = SelectorActorCritic()
    actor, _ = _observations(model)
    model.update_distribution(actor)
    with pytest.raises(ValueError):
        model.get_actions_log_prob(actions)


@pytest.mark.parametrize("overrides", [
    {"num_actions": 2}, {"num_modes": 3}, {"action_distribution": "gaussian"},
    {"share_encoders": True}, {"critic_output_dim": 2}, {"critic_privileged_dim": -1},
    {"actor_obs_dim": 3467}, {"num_actor_obs": 3467}, {"critic_obs_dim": 3661},
    {"state_input_dim": 0}, {"proprio_hidden_dims": []}, {"activation": "unknown"},
])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        SelectorActorCritic(**overrides)


def test_depth_cnn_rejects_misaligned_layers_and_collapsed_spatial_dimensions():
    cfg = _policy_config()["depth_encoder"]
    cfg["stride"] = [2]
    with pytest.raises(ValueError, match="equal lengths"):
        SelectorActorCritic(depth_encoder=cfg)
    cfg = _policy_config()["depth_encoder"]
    cfg.update(input_dim=(1, 1), kernel_size=[5, 5], padding=[0, 0])
    with pytest.raises(ValueError, match="collapses"):
        SelectorActorCritic(depth_encoder=cfg)


def test_unknown_gaussian_arguments_are_not_silently_ignored():
    with pytest.raises(TypeError, match="init_noise_std"):
        SelectorActorCritic(init_noise_std=1.0)


def test_evaluate_actions_rejects_unpaired_actor_and_critic_batches():
    model = SelectorActorCritic()
    actor, critic = _observations(model)
    with pytest.raises(ValueError, match="same batch size"):
        model.evaluate_actions(actor, critic[:2], torch.zeros(6, 1, dtype=torch.long))
