from __future__ import annotations

import ast
from pathlib import Path

import torch

from rsl_rl.modules.renet_actor_critic import RENetActorCritic


REPO_ROOT = Path(__file__).parents[2]
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
ACTOR_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "modules" / "renet_actor_critic.py"
ALGORITHM_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "algorithms" / "amp_ppo.py"
RUNNER_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "renet_amp_on_policy_runner.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _class_assignments(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(_source(path))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [
        statement.targets[0].id
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ]


def _depth_cnn_cfg() -> dict:
    return {
        "input_dim": (18, 32),
        "input_channels": 2,
        "output_channels": [4, 8],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "dilation": [1, 1],
        "padding": "zeros",
        "norm": "none",
        "activation": "relu",
        "max_pool": [True, True],
        "global_pool": "none",
        "flatten": True,
        "mlp_hidden_dim": [32],
        "mlp_output_dim": 64,
        "mlp_activation": "elu",
    }


def _lidar_cnn_cfg() -> dict:
    return {
        "input_dim": (11, 17),
        "input_channels": 1,
        "output_channels": [8, 16],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "dilation": [1, 1],
        "padding": "zeros",
        "norm": "none",
        "activation": "elu",
        "max_pool": [True, False],
        "global_pool": "none",
        "flatten": True,
        "mlp_hidden_dim": [128],
        "mlp_output_dim": 64,
        "mlp_activation": "elu",
    }


def _make_policy() -> RENetActorCritic:
    num_actor_obs = 10 * 78 + 11 * 17 + 2 * 18 * 32 + 1
    return RENetActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=85 * 10 + 187,
        num_actions=23,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        CnnMlp=_depth_cnn_cfg(),
        LidarCnnMlp=_lidar_cnn_cfg(),
        single_proprio_dim=78,
        estimator_mask_dim=1,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[32],
        fusion_type="attention",
        attention_num_heads=1,
        lp_encoder_dims=[32],
        vp_encoder_dims=[32],
        use_vel_estimation=True,
        vel_hidden_dim=[16],
        use_terrain_recon=True,
        terrain_hidden_dim=[32],
        lp_terrain_hidden_dim=[32],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        use_feet_height_prediction=True,
        feet_height_hidden_dim=[16],
    )


def _obs(policy: RENetActorCritic, batch_size: int = 2, mask: float = 1.0) -> torch.Tensor:
    observations = 0.05 * torch.randn(batch_size, policy.num_actor_obs)
    observations[:, -1] = mask
    return observations


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def _has_zero_grad(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.parameters()
    )


def test_config_defines_lp_vp_training_without_terrain_forcing():
    source = _source(CFG_PATH)
    expected_rewards = [
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "ang_vel_xy_l2",
        "energy",
        "dof_acc_l2",
        "action_rate_l2",
        "undesired_contacts",
        "fly",
        "body_orientation_l2",
        "flat_orientation_l2",
        "feet_slide",
        "feet_force",
        "feet_stumble",
        "dof_pos_limits",
        "joint_deviation_hip",
        "joint_deviation_ankle",
        "joint_deviation_arms",
        "joint_deviation_waist",
        "joint_deviation_legs",
        "termination_penalty",
        "alive",
        "feet_air_time",
    ]
    assert _class_assignments(CFG_PATH, "Reward") == expected_rewards
    assert "alternate_interval_iters: int = 20" in source
    assert "lp_probability: float = 0.5" in source
    assert "force_vp_terrain_names" not in source
    assert "force_vp_terrain_level" not in source
    assert "LidarCnnMlp" in source
    assert "lp_terrain_recon_coef: float = 0.5" in source
    assert "input_dim=(11,17)" in source
    assert "max_pool=[True,False]" in source


def test_actor_layout_mask_slots_and_full_lp_decoder():
    torch.manual_seed(7)
    policy = _make_policy()
    assert policy.num_actor_obs == 2120
    assert policy.proprio_actor_dim == 780
    assert policy.lidar_flat_dim == 187
    assert policy.depth_flat_dim == 1152
    assert policy.actor[0].in_features == 206
    assert policy.lp_terrain_output_dim == 187
    assert policy.terrain_output_dim == 99
    assert not any("op_" in name for name, _ in policy.named_parameters())

    observations = _obs(policy, batch_size=2)
    observations[1] = observations[0]
    observations[:, -1] = torch.tensor([0.0, 1.0])
    actor_input = policy._process_actor_obs(observations)

    assert actor_input.shape == (2, 206)
    torch.testing.assert_close(actor_input[0, 78:142], torch.zeros(64))
    torch.testing.assert_close(actor_input[1, 142:206], torch.zeros(64))
    assert torch.count_nonzero(actor_input[0, 142:206]) > 0
    assert torch.count_nonzero(actor_input[1, 78:142]) > 0
    assert policy.predict_lp_terrain().shape == (2, 187)
    assert policy.predict_terrain().shape == (2, 99)
    assert policy.predict_feet_height().shape == (2, 2)


def test_lidar_changes_only_lp_actions_and_depth_changes_only_vp_actions():
    torch.manual_seed(11)
    policy = _make_policy().eval()
    base = _obs(policy, batch_size=1)
    lidar_changed = base.clone()
    lidar_changed[:, policy.proprio_actor_dim : policy.proprio_actor_dim + policy.lidar_flat_dim] += 0.75
    depth_changed = base.clone()
    depth_start = policy.proprio_actor_dim + policy.lidar_flat_dim
    depth_changed[:, depth_start : depth_start + policy.depth_flat_dim] += 0.75

    with torch.inference_mode():
        base[:, -1] = 1.0
        lidar_changed[:, -1] = 1.0
        depth_changed[:, -1] = 1.0
        lp_base = policy.act_inference(base)
        lp_lidar = policy.act_inference(lidar_changed)
        lp_depth = policy.act_inference(depth_changed)

        base[:, -1] = 0.0
        lidar_changed[:, -1] = 0.0
        depth_changed[:, -1] = 0.0
        vp_base = policy.act_inference(base)
        vp_lidar = policy.act_inference(lidar_changed)
        vp_depth = policy.act_inference(depth_changed)

    assert not torch.allclose(lp_base, lp_lidar)
    torch.testing.assert_close(lp_base, lp_depth)
    torch.testing.assert_close(vp_base, vp_lidar)
    assert not torch.allclose(vp_base, vp_depth)


def test_mask_routes_policy_gradient_but_both_auxiliary_branches_can_train():
    torch.manual_seed(19)
    policy = _make_policy()

    lp_obs = _obs(policy, mask=1.0)
    policy.zero_grad(set_to_none=True)
    policy.act_inference(lp_obs).sum().backward()
    assert _has_nonzero_grad(policy.lp_gru)
    assert _has_zero_grad(policy.vp_gru)

    vp_obs = _obs(policy, mask=0.0)
    policy.zero_grad(set_to_none=True)
    policy.act_inference(vp_obs).sum().backward()
    assert _has_zero_grad(policy.lp_gru)
    assert _has_nonzero_grad(policy.vp_gru)

    policy.zero_grad(set_to_none=True)
    policy._process_actor_obs(vp_obs)
    velocity = policy.predict_velocity()
    auxiliary_loss = (
        velocity["lp"].square().mean()
        + velocity["vp"].square().mean()
        + policy.predict_lp_terrain().square().mean()
        + policy.predict_terrain().square().mean()
        + policy.predict_feet_height().square().mean()
    )
    auxiliary_loss.backward()
    assert _has_nonzero_grad(policy.lp_gru)
    assert _has_nonzero_grad(policy.vp_gru)
    assert _has_nonzero_grad(policy.lp_terrain_decoder)
    assert _has_nonzero_grad(policy.terrain_decoder)


def test_lp_target_is_fixed_and_confidence_is_diagnostic_only():
    algorithm_source = _source(ALGORITHM_PATH)
    env_source = _source(ENV_PATH)

    lp_block = algorithm_source.split("lp_terrain_recon_loss = torch.tensor", 1)[1].split(
        "feet_height_loss =", 1
    )[0]
    assert "critic_obs_batch[:, -self.terrain_scan_dim :]" in lp_block
    assert "lp_terrain_target.clamp" in lp_block
    assert "target_mean" not in lp_block
    assert "target_std" not in lp_block

    append_block = env_source.split("def append_renet_training_inputs", 1)[1].split(
        "def update_lidar_elevation_map_diagnostics", 1
    )[0]
    assert "flat_lidar_height" in append_block
    assert "lidar_confidence" not in append_block
    assert 'observations["lidar_confidence"]' in env_source


def test_global_twenty_iteration_schedule_and_boundary_refresh():
    expected = {0: 1.0, 19: 1.0, 20: 0.0, 39: 0.0, 40: 1.0}
    for iteration, mask in expected.items():
        actual = 1.0 if (iteration // 20) % 2 == 0 else 0.0
        assert actual == mask

    env_source = _source(ENV_PATH)
    runner_source = _source(RUNNER_PATH)
    assert "use_lp = (self.renet_training_iteration // interval) % 2 == 0" in env_source
    assert "def apply_renet_estimator_mask" in env_source
    assert "obs = self.env.apply_renet_estimator_mask(obs)" in runner_source
    assert runner_source.index("self.env.set_training_iteration(start_iter)") < runner_source.index(
        "obs, extras = self.env.get_observations()",
        runner_source.index("def learn"),
    )


def test_training_sources_are_lp_vp_only():
    actor_source = _source(ACTOR_PATH)
    algorithm_source = _source(ALGORITHM_PATH)
    assert "_last_op_latent" not in actor_source
    assert "op_vel_estimator" not in actor_source
    assert 'return {"lp": lp_vel, "vp": vp_vel, "active": active_vel}' in actor_source
    assert 'loss_dict["vel_estimation/lp"]' in algorithm_source
    assert 'loss_dict["terrain_recon/lp_full"]' in algorithm_source
    assert 'loss_dict["renet/lp_supervised"]' in algorithm_source
