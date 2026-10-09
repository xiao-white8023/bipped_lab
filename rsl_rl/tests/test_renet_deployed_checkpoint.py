# 中文导读
# 文件用途：验证存档 RENet 与 selector 的策略来源、历史长度、双分支导出和数值精度兼容性。
# 使用位置：通过 pytest 执行冻结策略和 selector 兼容回归；CUDA 用例在无 GPU 时跳过。
# 说明：测试使用本地存档文件，生产默认 checkpoint 仍指向服务器路径。
# 说明：runner 加载/更新仅属于测试覆盖，不表示 selector 监督训练会调用 RENet runner。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - archived：pytest 模块级 fixture，加载本地存档 checkpoint、原部署策略和冻结身份信息。
# - production_policy_kwargs：解析实际训练配置源码提取网络参数，可覆盖 VP 历史长度和输出类型。
# - test_real_checkpoint_loads_configurable_vp_and_preserves_deployed_ten_frame_actions：验证不同 VP 窗口可加载存档权重，十帧模式保持部署动作一致。
# - test_real_checkpoint_runner_load_preserves_actor_and_critic_and_can_update：验证实际 runner 加载存档时保持核心权重和动作，并能按需恢复优化器更新。
# - test_configured_vp_window_reaches_production_attention_and_gru：验证配置的 VP 历史长度确实作用于注意力和 GRU 的输入形状。
# - test_dual_export_preserves_all_six_outputs_for_different_batch_sizes：验证导出并重载双分支模型后，不同批量的六个输出保持一致。
# - test_parity_validation_rejects_same_weights_with_five_frame_vp：验证仅权重相同仍不够，错误的 VP 历史窗口会被动作一致性检查拒绝。
# - test_policy_provenance_rejects_missing_or_different_underlying_model：验证旧数据、结构差异和参考模型哈希差异被拒绝，而路径搬迁允许。
# - test_sensor_capture_time_uses_global_epoch_after_episode_reset：验证局部传感器时钟 reset 后仍正确计算全局采样时间。
# - test_half_precision_probability_loss_and_gradient_are_finite：验证极端半精度概率下损失和梯度保持有限。
# - test_cuda_adapter_validates_against_immutable_cpu_reference：验证 GPU 适配器与不迁移的 CPU 原部署策略保持动作一致。
# - test_cuda_autocast_selector_retains_finite_probability_loss：验证 CUDA 自动混合精度下 selector 极端输出的概率、损失和梯度有限。
# 中文导读结束

"""Regression coverage for the archived policy and selector data contract."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from legged_lab.selector.collector import CounterfactualCollector
from legged_lab.selector.config import CollectionConfig, DEFAULT_CHECKPOINT_PATH
from legged_lab.selector.frozen_policy import (
    assert_same_policy, dataset_policy_identity, load_frozen_adapter,
    validate_actions, validate_reference_weights, validation_inputs,
)
from legged_lab.selector.model import SelectorNetwork, weighted_binary_cross_entropy
from rsl_rl.modules import RENetActorCritic
from rsl_rl.runners import RENetAmpOnPolicyRunner
from test_g1_amp_format_switch import _env, _train_cfg, _update_runner


ROOT = Path(__file__).parents[2]
# Regression fixtures stay local even when production defaults target a server.
ARCHIVED_RUN_DIR = ROOT / "logs/g1_renet/2026-09-04_14-11-25"
ARCHIVED_CHECKPOINT_PATH = ARCHIVED_RUN_DIR / "model_60000.pt"
ARCHIVED_REFERENCE_POLICY_PATH = ARCHIVED_RUN_DIR / "exported/policy.pt"


@pytest.fixture(scope="module")
def archived():
    torch.set_num_threads(2)
    return load_frozen_adapter(ARCHIVED_CHECKPOINT_PATH, ARCHIVED_REFERENCE_POLICY_PATH)


def production_policy_kwargs(vp_frames=None, output_type=None):
    tree = ast.parse((ROOT / "legged_lab/envs/g1/RENet_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1RENETAGENTCFG")
    call = next(n.value for n in cls.body if isinstance(n, ast.Assign) and n.targets[0].id == "policy")
    setting = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == "G1_RENET_VP_PROPRIO_HISTORY_FRAMES")
    kwargs = eval(compile(ast.Expression(call), "RENet_cfg.py", "eval"), {
        "CustomRslRlPpoActorCriticCfg": lambda **kw: kw,
        "CnnMlpCfg": lambda **kw: kw,
        "G1_RENET_VP_PROPRIO_HISTORY_FRAMES": ast.literal_eval(setting),
    })
    kwargs.pop("class_name")
    if output_type is not None:
        kwargs["vp_output_type"] = output_type
    if vp_frames is not None:
        kwargs["vp_proprio_history_frames"] = vp_frames
    return kwargs


@pytest.mark.parametrize("vp_frames", [5, 10])
def test_real_checkpoint_loads_configurable_vp_and_preserves_deployed_ten_frame_actions(archived, vp_frames):
    adapter, reference, metadata = archived
    current = RENetActorCritic(2120, 1037, 23, **production_policy_kwargs(vp_frames, "legacy_latent")).eval()
    current.load_state_dict(adapter.policy.state_dict(), strict=True)
    validate_reference_weights(current, reference)
    assert current.vp_proprio_history_frames == vp_frames
    assert current.critic_history_frames == current.vp_critic_history_frames == 10
    assert not current.use_lp_feet_height_prediction
    assert CollectionConfig().checkpoint_path == str(DEFAULT_CHECKPOINT_PATH)
    assert metadata["source_iteration"] == 60000
    assert metadata["normalization"] == "identity"
    for observations in validation_inputs():
        for mask in (0., 1.):
            observations[:, -1] = mask
            with torch.inference_mode():
                if mask == 1. or vp_frames == 10:
                    torch.testing.assert_close(current.act_inference(observations), reference(observations), rtol=1e-4, atol=1e-5)
    if vp_frames == 5:
        observations = validation_inputs()[2]
        observations[:, -1] = 0.
        changed = observations.clone()
        changed[:, :5*78] += 10.
        with torch.inference_mode():
            expected = current.act_inference(observations)
            torch.testing.assert_close(current.act_inference(changed), expected, rtol=0, atol=0)
            assert not torch.allclose(expected, reference(observations))


@pytest.mark.parametrize("load_optimizer", [False, True])
@pytest.mark.parametrize("vp_frames", [5, 10])
def test_real_checkpoint_runner_load_preserves_actor_and_critic_and_can_update(archived, load_optimizer, vp_frames):
    adapter, reference, metadata = archived
    env = _env("g1_renet", 50, critic_frames=10)
    # Exercise the real pgtt_full -> pgtt_full_atten AME contract route,
    # including its strict model registration and Adam-state checks.
    env.lp_observation_contract = copy.deepcopy(metadata["lp_observation_contract"])
    env.lp_observation_contract["source"] = "pgtt_full_atten"
    env.lp_height_source = "pgtt_full_atten"
    env.lp_height_history_frames = 1
    env.lp_observation_shape = (11, 17)
    env.lp_terrain_recon_front_only = False
    cfg = _train_cfg("g1_renet", 50, critic_frames=10)
    cfg["policy"] = {"class_name": "RENetActorCritic", **production_policy_kwargs(vp_frames, "legacy_latent")}
    cfg["amp_discr_hidden_dims"] = [1024, 512, 256]
    cfg["algorithm"]["lp_feet_height_coef"] = 0.
    runner = RENetAmpOnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    runner.load(str(ARCHIVED_CHECKPOINT_PATH), load_optimizer=load_optimizer)
    assert runner.current_learning_iteration == runner.alg.current_iteration == 60000
    assert runner.alg.policy.vp_proprio_history_frames == vp_frames
    for name, tensor in adapter.policy.state_dict().items():
        # Current fixed-scale AUX targets deliberately reset only this decoder.
        if not name.startswith("terrain_decoder."):
            torch.testing.assert_close(runner.alg.policy.state_dict()[name], tensor, rtol=0, atol=0)
    observations = validation_inputs()[1]
    expected_policy = copy.deepcopy(adapter.policy).eval()
    expected_policy.vp_proprio_history_frames = vp_frames
    with torch.inference_mode():
        for mask in (0., 1.):
            observations[:, -1] = mask
            torch.testing.assert_close(runner.alg.policy.act_inference(observations), expected_policy.act_inference(observations), rtol=1e-4, atol=1e-5)
    if load_optimizer:
        _update_runner(runner, env)


def test_configured_vp_window_reaches_production_attention_and_gru():
    kwargs = production_policy_kwargs()
    frames = kwargs["vp_proprio_history_frames"]
    policy = RENetActorCritic(2120, 1037, 23, **kwargs).eval()
    shapes = {}
    attention_hook = policy.vp_attention.register_forward_pre_hook(
        lambda module, args: shapes.update(attention=args[0].shape))
    gru_hook = policy.vp_gru.register_forward_pre_hook(
        lambda module, args: shapes.update(gru=args[0].shape))
    with torch.inference_mode():
        output = policy.act_inference(validation_inputs()[1])
    attention_hook.remove()
    gru_hook.remove()
    assert output.shape == (7, 23)
    assert shapes == {"attention": (7, frames+2, 64), "gru": (7, frames, 64)}


def test_dual_export_preserves_all_six_outputs_for_different_batch_sizes(archived, tmp_path):
    adapter, reference, _ = archived
    probes = validation_inputs()
    traced = torch.jit.trace(adapter, probes[1][:2], check_trace=False)
    path = tmp_path / "dual.pt"
    torch.jit.save(traced, str(path))
    reloaded = torch.jit.load(str(path))
    validate_actions(reloaded, reference, probes)
    for observations in probes:
        with torch.inference_mode():
            for expected, actual in zip(adapter(observations), reloaded(observations), strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_parity_validation_rejects_same_weights_with_five_frame_vp(archived):
    adapter, reference, _ = archived
    altered = copy.deepcopy(adapter)
    altered.policy.vp_proprio_history_frames = 5
    with pytest.raises(RuntimeError, match="VP parity"):
        validate_actions(altered, reference, validation_inputs())


def test_policy_provenance_rejects_missing_or_different_underlying_model(archived):
    _, _, metadata = archived
    assert_same_policy(metadata, {**metadata, "source_checkpoint": "/relocated/model_60000.pt"})
    with pytest.raises(ValueError, match="recollect"):
        dataset_policy_identity({"metadata": {"checkpoint": "model_63999.pt"}})
    with pytest.raises(ValueError, match="vp_proprio_history_frames"):
        assert_same_policy(metadata, {**metadata, "vp_proprio_history_frames": 5})
    with pytest.raises(ValueError, match="reference_policy_sha256"):
        assert_same_policy(metadata, {**metadata, "reference_policy_sha256": "0" * 64})


def test_sensor_capture_time_uses_global_epoch_after_episode_reset():
    collector = CounterfactualCollector.__new__(CounterfactualCollector)
    collector.env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(current_time=100.1),
        sim_step_counter=20020, physics_dt=.005,
        camera=SimpleNamespace(current_timestamps=torch.tensor([.02, .2])),
        depth_last_capture_time=torch.tensor([0., .1]),
        extras={"observations": {"lp_frame_age": torch.tensor([[.03], [.05]])}})
    collector.device = torch.device("cpu")
    lp, vp = collector._sensor_timestamps()
    torch.testing.assert_close(lp, torch.tensor([100.07, 100.05]))
    torch.testing.assert_close(vp, torch.tensor([100.08, 100.]))


def test_half_precision_probability_loss_and_gradient_are_finite():
    probabilities = torch.tensor([[1., 1., 0., 0.]], dtype=torch.float16, requires_grad=True)
    loss = weighted_binary_cross_entropy(probabilities, torch.ones(1, 4), torch.ones(4))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_adapter_validates_against_immutable_cpu_reference(archived):
    adapter, reference, _ = archived
    gpu_adapter = copy.deepcopy(adapter).cuda()
    assert next(reference.parameters()).device.type == "cpu"
    validate_actions(gpu_adapter, reference, [x.cuda() for x in validation_inputs()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_autocast_selector_retains_finite_probability_loss():
    model = SelectorNetwork().cuda()
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(20.)
    with torch.amp.autocast("cuda"):
        probability = model(torch.zeros(2, 5, 283, device="cuda"))
        loss = weighted_binary_cross_entropy(probability, torch.ones(2, 4, device="cuda"), torch.ones(4, device="cuda"))
    assert probability.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
