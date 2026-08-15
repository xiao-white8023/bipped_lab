from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO
from rsl_rl.modules import Discriminator, RENetActorCritic
from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner
from rsl_rl.storage import RolloutStorage


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
REWARDS_PATH = REPO_ROOT / "legged_lab" / "mdp" / "rewards.py"


def _load_methods(path: Path, class_name: str | None, *method_names):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = []
    for method_name in method_names:
        method = next(
            node
            for node in body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
        )
        method.decorator_list = []
        selected.append(method)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _make_policy(fusion_type="attention"):
    return RENetActorCritic(
        num_actor_obs=2 * 78 + 2 * 4 * 4 + 2,
        num_critic_obs=5,
        num_actions=23,
        actor_hidden_dims=[512, 16],
        critic_hidden_dims=[8],
        activation="elu",
        single_proprio_dim=78,
        estimator_mask_dim=1,
        actor_control_dim=2,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[32],
        op_encoder_dims=[32],
        vp_encoder_dims=[32],
        fusion_type=fusion_type,
        attention_num_heads=1,
        use_vel_estimation=False,
        use_terrain_recon=False,
        use_feet_height_prediction=False,
        CnnMlp={
            "input_dim": (4, 4),
            "input_channels": 2,
            "output_channels": [1],
            "kernel_size": [1],
            "stride": [1],
            "dilation": [1],
            "padding": "none",
            "norm": "none",
            "activation": "relu",
            "max_pool": [False],
            "global_pool": "none",
            "flatten": True,
            "mlp_hidden_dim": [16],
            "mlp_output_dim": 64,
            "mlp_activation": "elu",
        },
    ).eval()


def _actor_obs(history, depth, mode, beta):
    return torch.cat(
        [history.reshape(history.shape[0], -1), depth.reshape(depth.shape[0], -1), mode, beta],
        dim=1,
    )


def test_actor_routes_vp_op_recovery_to_exact_209d_layout():
    torch.manual_seed(7)
    policy = _make_policy()
    history = torch.randn(3, 2, 78)
    depth = torch.randn(3, 2, 4, 4)
    mode = torch.tensor([[0.0], [1.0], [2.0]])
    beta = torch.tensor([[0.25], [0.25], [0.73]])
    actor_input = policy._process_actor_obs(_actor_obs(history, depth, mode, beta))

    assert actor_input.shape == (3, 209)
    assert policy.actor[0].weight.shape == (512, 209)
    torch.testing.assert_close(actor_input[0, 78:142], torch.zeros(64))
    torch.testing.assert_close(actor_input[1, 142:206], torch.zeros(64))
    torch.testing.assert_close(actor_input[2, 142:206], torch.zeros(64))
    torch.testing.assert_close(actor_input[:, 206:208], torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    torch.testing.assert_close(actor_input[:, 208], beta[:, 0])
    torch.testing.assert_close(actor_input[1, 78:142], policy._last_op_latent[1])
    torch.testing.assert_close(actor_input[2, 78:142], policy._last_recovery_latent[2])
    torch.testing.assert_close(actor_input[0, 142:206], policy._last_vp_latent[0])
    torch.testing.assert_close(policy._last_op_latent[2], torch.zeros(64))
    torch.testing.assert_close(policy._last_vp_latent[2], torch.zeros(64))
    torch.testing.assert_close(policy._last_recovery_latent[:2], torch.zeros(2, 64))


def test_recovery_uses_proprio_history_but_is_depth_independent():
    torch.manual_seed(11)
    policy = _make_policy()
    current = torch.randn(1, 78)
    history_a = torch.cat([torch.zeros(1, 1, 78), current.unsqueeze(1)], dim=1)
    history_b = torch.cat([torch.ones(1, 1, 78), current.unsqueeze(1)], dim=1)
    depth_a = torch.zeros(1, 2, 4, 4)
    depth_b = torch.randn(1, 2, 4, 4) * 10.0
    mode = torch.tensor([[2.0]])
    beta = torch.tensor([[0.6]])

    input_a = policy._process_actor_obs(_actor_obs(history_a, depth_a, mode, beta)).detach()
    input_history_changed = policy._process_actor_obs(_actor_obs(history_b, depth_a, mode, beta)).detach()
    input_depth_changed = policy._process_actor_obs(_actor_obs(history_a, depth_b, mode, beta)).detach()
    mean_a = policy.actor(input_a)
    mean_depth_changed = policy.actor(input_depth_changed)

    torch.testing.assert_close(input_a[:, :78], input_history_changed[:, :78])
    assert not torch.allclose(input_a[:, 78:142], input_history_changed[:, 78:142])
    torch.testing.assert_close(input_a, input_depth_changed)
    torch.testing.assert_close(mean_a, mean_depth_changed)
    input_nonfinite_depth = policy._process_actor_obs(
        _actor_obs(history_a, torch.full_like(depth_a, torch.nan), mode, beta)
    ).detach()
    assert torch.isfinite(input_nonfinite_depth).all()
    torch.testing.assert_close(input_a, input_nonfinite_depth)


def test_legacy_actor_weight_migration_preserves_old_outputs():
    old_weight = torch.randn(512, 206)
    bias = torch.randn(512)
    target = {"actor.0.weight": torch.randn(512, 209), "actor.0.bias": bias.clone()}
    checkpoint = {"actor.0.weight": old_weight.clone(), "actor.0.bias": bias.clone()}

    migrated, changed = RENetAmpOnPolicyRunner._migrate_legacy_actor_input(checkpoint, target)
    assert changed
    torch.testing.assert_close(migrated["actor.0.weight"][:, :206], old_weight)
    torch.testing.assert_close(migrated["actor.0.weight"][:, 206:], torch.zeros(512, 3))
    torch.testing.assert_close(migrated["actor.0.bias"], bias)

    old_input = torch.randn(4, 206)
    new_control = torch.tensor([[0.0, 0.0, 0.25], [1.0, 0.0, 0.25], [0.0, 0.0, 0.25], [1.0, 0.0, 0.25]])
    new_input = torch.cat([old_input, new_control], dim=1)
    old_actor = torch.nn.Sequential(torch.nn.Linear(206, 512), torch.nn.ELU(), torch.nn.Linear(512, 23))
    new_actor = torch.nn.Sequential(torch.nn.Linear(209, 512), torch.nn.ELU(), torch.nn.Linear(512, 23))
    with torch.no_grad():
        old_actor[0].weight.copy_(old_weight)
        old_actor[0].bias.copy_(bias)
        new_actor[0].weight.copy_(migrated["actor.0.weight"])
        new_actor[0].bias.copy_(bias)
        new_actor[2].load_state_dict(old_actor[2].state_dict())
    torch.testing.assert_close(old_actor(old_input), new_actor(new_input))


def _assert_module_has_nonzero_grad(module):
    parameters = list(module.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad).item() > 0 for parameter in parameters)


def _assert_module_has_no_grad(module):
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.parameters()
    )


def _make_random_actor_obs(modes):
    batch_size = len(modes)
    return _actor_obs(
        torch.randn(batch_size, 2, 78),
        torch.randn(batch_size, 2, 4, 4),
        torch.tensor(modes, dtype=torch.float).view(-1, 1),
        torch.full((batch_size, 1), 0.25),
    )


def test_recovery_history_modules_are_independent_and_op_initialized():
    torch.manual_seed(17)
    policy = _make_policy()
    module_pairs = (
        (policy.proprio_embedding, policy.recovery_proprio_embedding),
        (policy.op_attention, policy.recovery_attention),
        (policy.op_encoder, policy.recovery_encoder),
        (policy.op_gru, policy.recovery_gru),
    )
    for op_module, recovery_module in module_pairs:
        op_parameters = list(op_module.parameters())
        recovery_parameters = list(recovery_module.parameters())
        assert len(op_parameters) == len(recovery_parameters)
        assert {id(parameter) for parameter in op_parameters}.isdisjoint(
            {id(parameter) for parameter in recovery_parameters}
        )
        for op_parameter, recovery_parameter in zip(op_parameters, recovery_parameters):
            torch.testing.assert_close(op_parameter, recovery_parameter)
            assert op_parameter.data_ptr() != recovery_parameter.data_ptr()


@pytest.mark.parametrize("fusion_type", ["attention", "mlp"])
def test_recovery_history_encoder_supports_both_fusion_types(fusion_type):
    policy = _make_policy(fusion_type=fusion_type)
    actor_input = policy._process_actor_obs(_make_random_actor_obs([2.0, 2.0]))
    assert actor_input.shape == (2, 209)
    assert policy._last_recovery_latent.shape == (2, 64)
    assert torch.isfinite(actor_input).all()
    assert policy.is_recurrent is False


def test_mode_sparse_execution_uses_only_active_row_subsets():
    policy = _make_policy()
    calls = {"op": [], "vp": [], "recovery": [], "cnn": []}

    def record_rows(name):
        def hook(_module, args, _output):
            calls[name].append(args[0].shape[0])

        return hook

    handles = [
        policy.op_gru.register_forward_hook(record_rows("op")),
        policy.vp_gru.register_forward_hook(record_rows("vp")),
        policy.recovery_gru.register_forward_hook(record_rows("recovery")),
        policy.cnn.register_forward_hook(record_rows("cnn")),
    ]
    try:
        policy._process_actor_obs(_make_random_actor_obs([0.0, 1.0, 0.0]))
        assert calls == {"op": [3], "vp": [3], "recovery": [], "cnn": [6]}

        for values in calls.values():
            values.clear()
        policy._process_actor_obs(_make_random_actor_obs([2.0, 2.0, 2.0]))
        assert calls == {"op": [], "vp": [], "recovery": [3], "cnn": []}

        for values in calls.values():
            values.clear()
        policy._process_actor_obs(_make_random_actor_obs([2.0, 0.0, 1.0, 2.0]))
        assert calls == {"op": [2], "vp": [2], "recovery": [2], "cnn": [4]}
    finally:
        for handle in handles:
            handle.remove()


def test_recovery_only_actor_backward_is_isolated_from_op_and_vp():
    torch.manual_seed(23)
    policy = _make_policy()
    loss = policy.act_inference(_make_random_actor_obs([2.0, 2.0])).square().sum()
    loss.backward()

    for module in (
        policy.recovery_proprio_embedding,
        policy.recovery_attention,
        policy.recovery_encoder,
        policy.recovery_gru,
        policy.actor,
    ):
        _assert_module_has_nonzero_grad(module)
    for module in (
        policy.proprio_embedding,
        policy.op_attention,
        policy.op_encoder,
        policy.op_gru,
        policy.cnn,
        policy.vp_attention,
        policy.vp_encoder,
        policy.vp_gru,
    ):
        _assert_module_has_no_grad(module)


def test_op_and_vp_actor_backward_preserve_normal_paths_without_recovery_gradients():
    op_policy = _make_policy()
    op_policy.act_inference(_make_random_actor_obs([1.0, 1.0])).square().sum().backward()
    for module in (
        op_policy.proprio_embedding,
        op_policy.op_attention,
        op_policy.op_encoder,
        op_policy.op_gru,
        op_policy.actor,
    ):
        _assert_module_has_nonzero_grad(module)
    for module in (
        op_policy.recovery_proprio_embedding,
        op_policy.recovery_attention,
        op_policy.recovery_encoder,
        op_policy.recovery_gru,
    ):
        _assert_module_has_no_grad(module)

    vp_policy = _make_policy()
    vp_policy.act_inference(_make_random_actor_obs([0.0, 0.0])).square().sum().backward()
    for module in (
        vp_policy.proprio_embedding,
        vp_policy.cnn,
        vp_policy.vp_attention,
        vp_policy.vp_encoder,
        vp_policy.vp_gru,
        vp_policy.actor,
    ):
        _assert_module_has_nonzero_grad(module)
    for module in (
        vp_policy.recovery_proprio_embedding,
        vp_policy.recovery_attention,
        vp_policy.recovery_encoder,
        vp_policy.recovery_gru,
    ):
        _assert_module_has_no_grad(module)


def test_normal_only_sparse_path_matches_original_full_batch_computation():
    policy = _make_policy()
    observations = _make_random_actor_obs([0.0, 1.0, 1.0, 0.0])
    proprio_history, depth_flat, actor_mode, beta_obs, current_proprio = policy._split_actor_obs(observations)
    proprio_embed = policy._embed_proprio_history(proprio_history)
    depth_embed = policy._embed_depth_history(depth_flat)
    depth_context = policy._align_depth_to_proprio_history(depth_embed)
    op_features = policy._fuse_op_features(proprio_embed)
    vp_features = policy._fuse_vp_features(proprio_embed, depth_embed, depth_context)
    _, op_hidden = policy.op_gru(op_features)
    _, vp_hidden = policy.vp_gru(vp_features)
    is_op = (actor_mode == 1.0).to(current_proprio.dtype)
    is_recovery = torch.zeros_like(is_op)
    is_vp = actor_mode == 0.0
    expected_input = torch.cat(
        [
            current_proprio,
            torch.where(is_op.bool(), op_hidden[-1], torch.zeros_like(op_hidden[-1])),
            torch.where(is_vp, vp_hidden[-1], torch.zeros_like(vp_hidden[-1])),
            is_op,
            is_recovery,
            beta_obs,
        ],
        dim=-1,
    )
    actual_input = policy._process_actor_obs(observations)
    torch.testing.assert_close(actual_input, expected_input)
    torch.testing.assert_close(policy.actor(actual_input), policy.actor(expected_input))


def _recovery_prefix_map():
    return {
        "recovery_proprio_embedding.": "proprio_embedding.",
        "recovery_attention.": "op_attention.",
        "recovery_encoder.": "op_encoder.",
        "recovery_gru.": "op_gru.",
    }


def _without_recovery_history(state_dict):
    prefixes = tuple(_recovery_prefix_map())
    return {
        key: value.clone()
        for key, value in state_dict.items()
        if not key.startswith(prefixes)
    }


def test_missing_recovery_checkpoint_branch_is_copied_from_checkpoint_op():
    policy = _make_policy()
    target = policy.state_dict()
    checkpoint = _without_recovery_history(target)
    migrated, changed = RENetAmpOnPolicyRunner._migrate_missing_recovery_history_branch(checkpoint, target)
    assert changed

    for recovery_prefix, op_prefix in _recovery_prefix_map().items():
        for recovery_key in (key for key in target if key.startswith(recovery_prefix)):
            op_key = op_prefix + recovery_key.removeprefix(recovery_prefix)
            torch.testing.assert_close(migrated[recovery_key], checkpoint[op_key])
            assert migrated[recovery_key].data_ptr() != checkpoint[op_key].data_ptr()
    policy.load_state_dict(migrated)


def test_complete_recovery_checkpoint_is_preserved_and_partial_is_rejected():
    policy = _make_policy()
    complete = {key: value.clone() for key, value in policy.state_dict().items()}
    unchanged, changed = RENetAmpOnPolicyRunner._migrate_missing_recovery_history_branch(
        complete,
        policy.state_dict(),
    )
    assert not changed
    assert unchanged is complete

    partial = complete.copy()
    recovery_key = next(key for key in partial if key.startswith("recovery_gru."))
    partial.pop(recovery_key)
    with pytest.raises(RuntimeError, match="partial Recovery history encoder"):
        RENetAmpOnPolicyRunner._migrate_missing_recovery_history_branch(partial, policy.state_dict())


def test_legacy_206_actor_and_missing_recovery_branch_migrate_together():
    policy = _make_policy()
    target = policy.state_dict()
    checkpoint = _without_recovery_history(target)
    old_actor_weight = checkpoint["actor.0.weight"][:, :206].clone()
    checkpoint["actor.0.weight"] = old_actor_weight

    migrated, actor_changed = RENetAmpOnPolicyRunner._migrate_legacy_actor_input(checkpoint, target)
    migrated, recovery_changed = RENetAmpOnPolicyRunner._migrate_missing_recovery_history_branch(
        migrated,
        target,
    )
    assert actor_changed and recovery_changed
    torch.testing.assert_close(migrated["actor.0.weight"][:, :206], old_actor_weight)
    torch.testing.assert_close(migrated["actor.0.weight"][:, 206:], torch.zeros_like(target["actor.0.weight"][:, 206:]))
    policy.load_state_dict(migrated)


def test_mode_dependent_action_mapping_uses_current_pose_and_does_not_clamp_actions():
    (mapping,) = _load_methods(ENV_PATH, "G1RENetEnv", "_mode_dependent_joint_targets")
    default = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    current = torch.tensor([[5.0, 6.0], [5.0, 6.0]])
    action = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
    mask = torch.tensor([False, True])
    targets = mapping(default, current, action, mask, 0.8, 0.25)
    torch.testing.assert_close(targets[0], default[0] + 0.25 * action[0])
    torch.testing.assert_close(targets[1], current[1] + 0.8 * action[1])
    assert targets[1, 0] == 13.0  # proves no newly introduced [-1, 1] clamp


def test_action_time_reward_routing_is_mutually_exclusive_and_preserves_tensor_contract():
    (route_rewards,) = _load_methods(ENV_PATH, "G1RENetEnv", "_route_action_time_rewards")
    raw_loco = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    raw_task = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    raw_reg = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32)
    recovery_mask_t = torch.tensor([False, True, False, True])

    loco, task, reg = route_rewards(raw_loco, raw_task, raw_reg, recovery_mask_t)

    torch.testing.assert_close(loco, torch.tensor([1.0, 0.0, 3.0, 0.0]))
    torch.testing.assert_close(task, torch.tensor([0.0, 20.0, 0.0, 40.0]))
    torch.testing.assert_close(reg, torch.tensor([0.0, -2.0, 0.0, -4.0]))
    for routed in (loco, task, reg):
        assert routed.shape == raw_loco.shape
        assert routed.device == raw_loco.device
        assert routed.dtype == raw_loco.dtype

    with pytest.raises(TypeError, match="dtype"):
        route_rewards(raw_loco, raw_task.double(), raw_reg, recovery_mask_t)
    with pytest.raises(ValueError, match="shape"):
        route_rewards(raw_loco, raw_task, raw_reg, recovery_mask_t[:-1])


def test_reward_routing_debug_metrics_report_owner_ratios_and_zero_leakage():
    safe_mean, update_diagnostics = _load_methods(
        ENV_PATH,
        "G1RENetEnv",
        "_safe_masked_mean",
        "_update_reward_routing_diagnostics",
    )
    env = SimpleNamespace(_recovery_diagnostics={}, _safe_masked_mean=safe_mean)
    recovery_mask_t = torch.tensor([False, True, False, True])
    update_diagnostics(
        env,
        recovery_mask_t,
        torch.tensor([1.0, 0.0, 3.0, 0.0]),
        torch.tensor([0.0, 2.0, 0.0, 4.0]),
        torch.tensor([0.0, -2.0, 0.0, -4.0]),
    )
    assert env._recovery_diagnostics["RewardRouting/locomotion_ratio"].item() == 0.5
    assert env._recovery_diagnostics["RewardRouting/recovery_ratio"].item() == 0.5
    for key in (
        "RewardRouting/loco_reward_on_recovery_mean",
        "RewardRouting/recovery_task_on_loco_mean",
        "RewardRouting/recovery_reg_on_loco_mean",
    ):
        assert env._recovery_diagnostics[key].item() == 0.0


def test_recovery_action_rate_has_no_cross_mode_boundary():
    (update_rate,) = _load_methods(ENV_PATH, "G1RENetEnv", "_update_recovery_action_rate")
    env = SimpleNamespace(
        action=torch.tensor([[100.0, -100.0]]),
        recovery_prev_action=torch.zeros(1, 2),
        recovery_prev_action_valid=torch.zeros(1, dtype=torch.bool),
        recovery_action_rate_value=torch.zeros(1),
        recovery_action_rate_valid_sample=torch.zeros(1, dtype=torch.bool),
    )
    update_rate(env, torch.tensor([False]))
    env.action = torch.tensor([[2.0, 3.0]])
    update_rate(env, torch.tensor([True]))
    torch.testing.assert_close(env.recovery_action_rate_value, torch.zeros(1))
    env.action = torch.tensor([[5.0, 7.0]])
    update_rate(env, torch.tensor([True]))
    torch.testing.assert_close(env.recovery_action_rate_value, torch.tensor([25.0]))
    update_rate(env, torch.tensor([False]))
    env.action = torch.tensor([[-8.0, 9.0]])
    update_rate(env, torch.tensor([True]))
    torch.testing.assert_close(env.recovery_action_rate_value, torch.zeros(1))


def test_original_action_rate_formula_is_unchanged():
    (action_rate,) = _load_methods(REWARDS_PATH, None, "action_rate_l2")
    buffer = torch.tensor([[[1.0, 2.0], [4.0, 6.0]]])
    env = SimpleNamespace(action_buffer=SimpleNamespace(_circular_buffer=SimpleNamespace(buffer=buffer)))
    torch.testing.assert_close(action_rate(env), torch.tensor([25.0]))


def test_task_is_raw_upright_height_product_and_invalid_height_is_zero():
    tolerance, compute_task = _load_methods(
        ENV_PATH,
        "G1RENetEnv",
        "_gaussian_lower_bound_tolerance",
        "compute_recovery_task_reward",
    )
    env = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 3))),
        recovery_task_height_threshold=1.0,
        recovery_upright_reward_buf=torch.zeros(3),
        recovery_height_reward_buf=torch.zeros(3),
        recovery_task_reward_buf=torch.zeros(3),
        recovery_torso_height_buf=torch.zeros(3),
        recovery_torso_height_valid_buf=torch.zeros(3, dtype=torch.bool),
        compute_local_torso_height=lambda: (
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([True, True, False]),
            torch.zeros(3),
        ),
    )
    env._gaussian_lower_bound_tolerance = tolerance
    reward = compute_task(env, torch.tensor([True, True, False]))
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.1, 0.0]))
    assert torch.all((reward >= 0.0) & (reward <= 1.0))
    # No dt exists in the computation: the exact target product stays one.
    assert reward[0] == 1.0


def test_central_height_crop_ignores_periphery_and_nonfinite_hits():
    central_indices, finite_median, torso_height = _load_methods(
        ENV_PATH,
        "G1RENetEnv",
        "_central_height_crop_indices",
        "_finite_ray_median",
        "compute_local_torso_height",
    )
    x = torch.arange(-0.8, 0.800001, 0.1)
    y = torch.arange(-0.5, 0.500001, 0.1)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    starts = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros(grid_x.numel())], dim=1)
    indices = central_indices(starts, 0.2)
    assert indices.numel() == 25

    hits = torch.full((2, starts.shape[0], 3), 1000.0)
    hits[:, :, 2] = torch.where(torch.arange(starts.shape[0]) % 2 == 0, 1000.0, -1000.0)
    hits[0, indices, 2] = 2.0
    hits[0, indices[0], 2] = torch.inf
    hits[0, indices[1], 2] = torch.nan
    hits[1, indices, 2] = torch.inf
    env = SimpleNamespace(
        height_scanner=SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
        recovery_height_crop_indices=indices,
        robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.tensor([[[0.0, 0.0, 3.0]], [[0.0, 0.0, 3.0]]]))),
        torso_body_id=0,
        _finite_ray_median=finite_median,
    )
    height, valid, ground = torso_height(env)
    torch.testing.assert_close(height, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(ground, torch.tensor([2.0, 0.0]))
    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.isfinite(height).all()


def test_curriculum_uses_exact_1024_attempt_windows_and_persists_window_stats():
    (record_attempts,) = _load_methods(ENV_PATH, "G1RENetEnv", "_record_recovery_curriculum_attempts")

    def make_env():
        return SimpleNamespace(
            recovery_state_machine_enabled=True,
            cfg=SimpleNamespace(
                recovery=SimpleNamespace(
                    enable_curriculum=True,
                    curriculum_min_attempts=1024,
                    curriculum_success_ratio=0.60,
                    assist_force_step=20.0,
                    min_assist_force=0.0,
                    beta_step=0.02,
                    min_beta=0.25,
                )
            ),
            recovery_curriculum_window_attempts=0,
            recovery_curriculum_window_successes=0,
            recovery_curriculum_last_window_success_ratio=0.0,
            recovery_curriculum_last_window_attempts=0,
            recovery_curriculum_last_window_successes=0,
            recovery_curriculum_last_window_advanced=False,
            recovery_curriculum_total_completed_attempts=0,
            recovery_curriculum_total_level_advances=0,
            recovery_curriculum_total_windows=0,
            recovery_curriculum_level=0,
            current_recovery_assist_force=200.0,
            current_recovery_beta=1.0,
        )

    env = make_env()
    record_attempts(env, torch.ones(1023, dtype=torch.bool), torch.tensor([True] * 614 + [False] * 409))
    assert env.recovery_curriculum_level == 0
    assert env.recovery_curriculum_window_attempts == 1023
    assert env.recovery_curriculum_total_completed_attempts == 1023
    assert env.recovery_curriculum_total_windows == 0
    record_attempts(env, torch.ones(1, dtype=torch.bool), torch.tensor([False]))
    assert env.recovery_curriculum_window_attempts == 0
    assert env.recovery_curriculum_last_window_attempts == 1024
    assert env.recovery_curriculum_last_window_successes == 614
    assert env.recovery_curriculum_last_window_success_ratio == 614 / 1024
    assert env.recovery_curriculum_last_window_advanced is False
    assert env.recovery_curriculum_total_completed_attempts == 1024
    assert env.recovery_curriculum_total_windows == 1
    assert env.recovery_curriculum_total_level_advances == 0

    env = make_env()
    record_attempts(env, torch.ones(1024, dtype=torch.bool), torch.tensor([True] * 615 + [False] * 409))
    assert env.recovery_curriculum_level == 1
    assert env.current_recovery_assist_force == 180.0
    assert env.current_recovery_beta == 0.98
    assert env.recovery_curriculum_window_attempts == 0
    assert env.recovery_curriculum_last_window_successes == 615
    assert env.recovery_curriculum_last_window_advanced is True
    assert env.recovery_curriculum_total_level_advances == 1

    env.current_recovery_assist_force = 0.0
    env.current_recovery_beta = 0.25
    record_attempts(env, torch.ones(1024, dtype=torch.bool), torch.ones(1024, dtype=torch.bool))
    assert env.current_recovery_assist_force == 0.0
    assert env.current_recovery_beta == 0.25


def test_force_gate_is_recovery_only_upright_gated_and_clears_to_zero():
    (force_gate,) = _load_methods(ENV_PATH, "G1RENetEnv", "_assist_force_gate")
    active, force = force_gate(
        torch.tensor([False, True, True, True]),
        torch.tensor([0.9, 0.7, 0.9, 0.9]),
        200.0,
        0.8,
    )
    assert torch.equal(active, torch.tensor([False, False, True, True]))
    torch.testing.assert_close(force, torch.tensor([0.0, 0.0, 200.0, 200.0]))
    active_zero, force_zero = force_gate(torch.ones(2, dtype=torch.bool), torch.ones(2), 0.0, 0.8)
    assert not torch.any(active_zero)
    assert not torch.any(force_zero)


def test_force_writer_overwrites_inactive_rows_to_prevent_stale_force():
    force_gate, set_force = _load_methods(
        ENV_PATH,
        "G1RENetEnv",
        "_assist_force_gate",
        "_set_recovery_assist_force",
    )

    class Composer:
        def __init__(self):
            self.last_force = None

        def set_forces_and_torques(self, *, forces, **_kwargs):
            self.last_force = forces.clone()

    composer = Composer()
    env = SimpleNamespace(
        robot=SimpleNamespace(
            data=SimpleNamespace(projected_gravity_b=torch.tensor([[0.0, 0.0, -0.9], [0.0, 0.0, -0.7]])),
            permanent_wrench_composer=composer,
        ),
        cfg=SimpleNamespace(recovery=SimpleNamespace(force_upright_gate=0.8)),
        current_recovery_assist_force=200.0,
        recovery_assist_force_active_buf=torch.zeros(2, dtype=torch.bool),
        recovery_assist_force_values=torch.zeros(2, 1, 3),
        recovery_assist_torque_values=torch.zeros(2, 1, 3),
        torso_body_ids=torch.tensor([0]),
        _assist_force_gate=force_gate,
    )
    set_force(env, torch.tensor([True, True]))
    torch.testing.assert_close(composer.last_force[:, 0, 2], torch.tensor([200.0, 0.0]))
    set_force(env, torch.tensor([False, False]))
    assert not torch.any(composer.last_force)


def test_ready_requires_50_consecutive_steps_at_002_seconds():
    (advance_ready,) = _load_methods(ENV_PATH, "G1RENetEnv", "_advance_recovery_ready_counter")
    counter = torch.zeros(1, dtype=torch.long)
    was_recovery = torch.ones(1, dtype=torch.bool)
    ready = torch.ones(1, dtype=torch.bool)
    for _ in range(49):
        counter, exit_recovery = advance_ready(was_recovery, ready, counter, 50)
    assert counter.item() == 49
    assert not exit_recovery.item()
    counter, exit_recovery = advance_ready(was_recovery, ready, counter, 50)
    assert counter.item() == 50
    assert exit_recovery.item()
    counter, exit_recovery = advance_ready(was_recovery, torch.zeros(1, dtype=torch.bool), counter, 50)
    assert counter.item() == 0
    assert not exit_recovery.item()


def test_recovery_advantage_weights_are_applied_once_without_renormalization():
    task = torch.tensor([1.0, -1.0])
    amp = torch.tensor([2.0, -2.0])
    reg = torch.tensor([3.0, -3.0])
    combined = RENetAMPPPO.combine_recovery_advantages(task, amp, reg, (2.5, 1.0, 0.1))
    torch.testing.assert_close(combined, 2.5 * task + amp + 0.1 * reg)


def test_runner_collects_reset_only_reward_log_keys_from_entire_rollout():
    ep_infos = [
        {"Recovery/active_ratio": torch.tensor(0.0)},
        {"Recovery/active_ratio": torch.tensor(0.2), "Episode_Reward/track_lin_vel": torch.tensor(1.5)},
        {"DepthNoise/enabled": torch.tensor(1.0)},
    ]
    assert RENetAmpOnPolicyRunner._collect_episode_info_keys(ep_infos) == [
        "Recovery/active_ratio",
        "Episode_Reward/track_lin_vel",
        "DepthNoise/enabled",
    ]


def test_runner_splits_routed_amp_reward_into_disjoint_action_time_streams():
    routed = torch.tensor([3.0, 7.0, -2.0, 5.0], dtype=torch.float32)
    recovery_mask_t = torch.tensor([False, True, False, True])
    loco, recovery = RENetAmpOnPolicyRunner._split_action_time_amp_rewards(
        routed,
        recovery_mask_t,
    )
    torch.testing.assert_close(loco, torch.tensor([3.0, 0.0, -2.0, 0.0]))
    torch.testing.assert_close(recovery, torch.tensor([0.0, 7.0, 0.0, 5.0]))
    assert loco.shape == recovery.shape == routed.shape
    assert loco.device == recovery.device == routed.device
    assert loco.dtype == recovery.dtype == routed.dtype


def test_amp_replay_storage_receives_disjoint_action_time_transitions():
    class CaptureStorage:
        def __init__(self):
            self.states = []

        def insert(self, states, _next_states):
            self.states.append(states.clone())

    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.device = "cpu"
    algorithm.discriminator_loco = SimpleNamespace(input_dim=100)
    algorithm.discriminator_recovery = SimpleNamespace(input_dim=106)
    algorithm.amp_storage_loco = CaptureStorage()
    algorithm.amp_storage_recovery = CaptureStorage()
    amp_obs = {
        "loco": torch.arange(4 * 50, dtype=torch.float32).reshape(4, 50),
        "recovery": torch.arange(4 * 53, dtype=torch.float32).reshape(4, 53),
    }
    next_amp_obs = {key: value + 10.0 for key, value in amp_obs.items()}
    recovery_mask_t = torch.tensor([False, True, False, True])

    algorithm._store_amp_transition(amp_obs, next_amp_obs, recovery_mask_t)

    torch.testing.assert_close(
        algorithm.amp_storage_loco.states[0],
        amp_obs["loco"][[0, 2]],
    )
    torch.testing.assert_close(
        algorithm.amp_storage_recovery.states[0],
        amp_obs["recovery"][[1, 3]],
    )


def test_ppo_rollout_boundary_masks_every_reward_stream_with_action_time_owner(monkeypatch):
    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.device = "cpu"
    algorithm.transition = SimpleNamespace(values=torch.zeros(4, 1))
    captured = {}

    def capture_parent(_self, rewards, _dones, _infos, _amp_obs, recovery_mask_t=None):
        captured["rewards"] = rewards.clone()
        captured["mask"] = recovery_mask_t.clone()

    monkeypatch.setattr(AMPPPO, "process_env_step", capture_parent)
    recovery_mask_t = torch.tensor([False, True, False, True])
    infos = {
        "recovery_task_reward": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        "recovery_amp_reward": torch.tensor([11.0, 21.0, 31.0, 41.0]),
        "recovery_reg_reward": torch.tensor([-1.0, -2.0, -3.0, -4.0]),
    }
    algorithm.process_env_step(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.zeros(4, dtype=torch.bool),
        infos,
        torch.zeros(4, 2),
        recovery_mask_t,
    )

    torch.testing.assert_close(captured["rewards"], torch.tensor([1.0, 0.0, 3.0, 0.0]))
    assert torch.equal(captured["mask"], recovery_mask_t)
    torch.testing.assert_close(
        algorithm.transition.recovery_task_reward,
        torch.tensor([0.0, 20.0, 0.0, 40.0]),
    )
    torch.testing.assert_close(
        algorithm.transition.recovery_amp_reward,
        torch.tensor([0.0, 21.0, 0.0, 41.0]),
    )
    torch.testing.assert_close(
        algorithm.transition.recovery_reg_reward,
        torch.tensor([0.0, -2.0, 0.0, -4.0]),
    )
    assert torch.equal(algorithm.transition.recovery_rewards_valid, recovery_mask_t)


def test_recovery_amp_is_pure_while_locomotion_keeps_03_07_mix():
    recovery = Discriminator(4, 1.0, [2], "cpu", task_reward_lerp=0.0)
    locomotion = Discriminator(4, 0.3, [2], "cpu", task_reward_lerp=0.7)
    for discriminator in (recovery, locomotion):
        for parameter in discriminator.parameters():
            parameter.data.zero_()
        discriminator.amp_linear.bias.data.fill_(1.0)
    state = torch.zeros(2, 2)
    next_state = torch.zeros(2, 2)
    task_a = torch.tensor([0.0, 0.0])
    task_b = torch.tensor([1.0, 1.0])
    rec_a = recovery.predict_amp_reward(state, next_state, task_a)[0]
    rec_b = recovery.predict_amp_reward(state, next_state, task_b)[0]
    loco_a = locomotion.predict_amp_reward(state, next_state, task_a)[0]
    loco_b = locomotion.predict_amp_reward(state, next_state, task_b)[0]
    torch.testing.assert_close(rec_a, rec_b)
    assert not torch.allclose(loco_a, loco_b)
    torch.testing.assert_close(rec_a, torch.ones(2))
    torch.testing.assert_close(loco_a, torch.full((2,), 0.09))
    torch.testing.assert_close(loco_b, torch.full((2,), 0.79))


def test_fixed_cfg_contracts_and_exact_five_reg_terms():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    recovery_values = {}
    for node in classes["RecoveryStateMachineCfg"].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            recovery_values[node.target.id] = ast.literal_eval(node.value)
    assert recovery_values["enable"] is True
    assert recovery_values["ready_hold_s"] == 1.0
    assert recovery_values["absolute_episode_timeout_s"] == 27.0
    assert recovery_values["max_duration_s"] == 6.0
    assert recovery_values["curriculum_min_attempts"] == 1024
    assert recovery_values["initial_assist_force"] == 200.0
    assert recovery_values["initial_beta"] == 1.0
    assert recovery_values["min_beta"] == 0.25

    algorithm_values = {}
    for node in classes["CustomRslRlPpoAlgorithmCfg"].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            algorithm_values[node.target.id] = ast.literal_eval(node.value)
    assert algorithm_values["recovery_ppo_min_rollout_samples"] == 2048
    assert algorithm_values["recovery_drec_min_replay_samples"] == 2048

    reg_assignments = [
        node for node in classes["RecoveryRegReward"].body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    reg_names = [node.targets[0].id for node in reg_assignments if isinstance(node, ast.Assign)]
    assert reg_names == ["joint_acc", "action_rate", "torque", "joint_pos_limit", "joint_vel_limit"]
    reg_weights = {}
    for node in reg_assignments:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        reg_weights[node.targets[0].id] = ast.literal_eval(
            next(keyword.value for keyword in call.keywords if keyword.arg == "weight")
        )
    assert reg_weights == {
        "joint_acc": -2.5e-7,
        "action_rate": -0.01,
        "torque": -2.5e-6,
        "joint_pos_limit": -2.0,
        "joint_vel_limit": -1.0,
    }
    source = CFG_PATH.read_text(encoding="utf-8")
    assert "recovery_amp_reward_coef = 1.0" in source
    assert "recovery_amp_task_reward_lerp = 0.0" in source
    assert "amp_reward_coef = 0.3" in source
    assert "amp_task_reward_lerp = 0.7" in source
    assert "self.baseline_max_episode_length_s" in ENV_PATH.read_text(encoding="utf-8")
    assert math.ceil(recovery_values["ready_hold_s"] / 0.02) == 50
    assert math.ceil(recovery_values["absolute_episode_timeout_s"] / 0.02) == 1350
    assert math.ceil(recovery_values["max_duration_s"] / 0.02) == 300


def test_enter_recovery_event_and_reward_manager_weight_contract():
    (event_fn,) = _load_methods(REWARDS_PATH, None, "enter_recovery_event")
    env = SimpleNamespace(enter_recovery_buf=torch.tensor([False, True, False]))
    torch.testing.assert_close(event_fn(env), torch.tensor([0.0, 1.0, 0.0]))
    source = CFG_PATH.read_text(encoding="utf-8")
    assert "enter_recovery_penalty = RewTerm(func=mdp.enter_recovery_event, weight=-200.0)" in source
    # RewardManager applies weight * dt, so at step_dt=.02 the event contribution is -4.
    torch.testing.assert_close(event_fn(env) * -200.0 * 0.02, torch.tensor([0.0, -4.0, 0.0]))


def test_enter_recovery_ends_locomotion_gae_without_recovery_bootstrap():
    _, advantages = RolloutStorage.compute_segmented_gae(
        rewards=torch.tensor([[[1.0]], [[1000.0]]]),
        values=torch.zeros(2, 1, 1),
        last_values=torch.tensor([[5000.0]]),
        sample_mask=torch.tensor([[[True]], [[False]]]),
        trace_end=torch.tensor([[[True]], [[False]]]),
        env_terminal=torch.zeros(2, 1, 1, dtype=torch.bool),
        time_outs=torch.zeros(2, 1, 1, dtype=torch.bool),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=False,
    )
    torch.testing.assert_close(advantages[0], torch.tensor([[1.0]]))


def test_exit_recovery_transition_remains_recovery_owned_and_ends_recovery_gae():
    _, advantages = RolloutStorage.compute_segmented_gae(
        rewards=torch.tensor([[[2.0]], [[1000.0]]]),
        values=torch.zeros(2, 1, 1),
        last_values=torch.tensor([[5000.0]]),
        sample_mask=torch.tensor([[[True]], [[False]]]),
        trace_end=torch.tensor([[[True]], [[False]]]),
        env_terminal=torch.zeros(2, 1, 1, dtype=torch.bool),
        time_outs=torch.zeros(2, 1, 1, dtype=torch.bool),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=False,
    )
    torch.testing.assert_close(advantages[0], torch.tensor([[2.0]]))


def test_no_deferred_or_forbidden_recovery_shaping_was_added():
    changed_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ENV_PATH, CFG_PATH, REWARDS_PATH)
    )
    for forbidden in (
        "action_smoothness",
        "action_second_difference",
        "success_bonus",
        "refall_penalty",
        "recovery_soft_bound",
        "recovery_joint_power",
    ):
        assert forbidden not in changed_sources
