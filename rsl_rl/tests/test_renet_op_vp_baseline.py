from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from rsl_rl.modules.renet_actor_critic import RENetActorCritic
from rsl_rl.utils.motion_loader import AMPLoader


REPO_ROOT = Path(__file__).parents[2]
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
ACTOR_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "modules" / "renet_actor_critic.py"
ALGORITHM_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "algorithms" / "renet_amp_ppo.py"
RUNNER_PATH = REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "renet_amp_on_policy_runner.py"
PLAY_PATH = REPO_ROOT / "legged_lab" / "scripts" / "play_new_renet.py"
SIM2SIM_PATH = REPO_ROOT / "legged_lab" / "scripts" / "sim2sim_renet.py"
EXPORT_PATH = REPO_ROOT / "legged_lab" / "scripts" / "export.py"


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


def _make_policy() -> RENetActorCritic:
    depth_frames, height, width = 2, 4, 4
    num_actor_obs = 10 * 78 + depth_frames * height * width + 1
    return RENetActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=85 * 10,
        num_actions=23,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        CnnMlp={
            "input_dim": (height, width),
            "input_channels": depth_frames,
            "output_channels": [2],
            "kernel_size": [3],
            "stride": [1],
            "dilation": [1],
            "padding": "zeros",
            "norm": "none",
            "activation": "relu",
            "max_pool": [False],
            "global_pool": "none",
            "flatten": True,
            "mlp_hidden_dim": [16],
            "mlp_output_dim": 64,
            "mlp_activation": "elu",
        },
        single_proprio_dim=78,
        estimator_mask_dim=1,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[32],
        fusion_type="attention",
        attention_num_heads=1,
        op_encoder_dims=[32],
        vp_encoder_dims=[32],
        use_vel_estimation=False,
        use_terrain_recon=False,
        use_feet_height_prediction=False,
    )


def test_renet_config_matches_the_initial_op_vp_contract():
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
    assert "max_episode_length_s=20.0" in source
    assert "add_camera_noise=False" in source
    assert 'force_vp_terrain_names: list[str] = ["gap", "high_platform", "hf_stepping_stones", "star_terrain"]' in source
    assert "estimator_mask_dim: int = 1" in source
    assert "amp_reward_coef = 0.4" in source
    assert "amp_task_reward_lerp = 0.6" in source
    assert source.count("motion_amp_expert_no_ankle/") == 4
    assert "fallAndGetUp" not in source
    assert "recovery" not in source.lower()


def test_actor_has_one_mask_and_a_206d_internal_input():
    torch.manual_seed(7)
    policy = _make_policy()
    assert policy.estimator_mask_dim == 1
    assert policy.actor[0].in_features == 206
    assert not hasattr(policy, "actor_control_dim")
    assert all("recovery" not in name.lower() for name, _ in policy.named_parameters())

    observations = torch.randn(2, policy.num_actor_obs)
    observations[1] = observations[0]
    mask_index = policy.proprio_actor_dim + policy.depth_flat_dim
    observations[:, mask_index] = torch.tensor([0.0, 1.0])
    actor_input = policy._process_actor_obs(observations)

    assert actor_input.shape == (2, 206)
    torch.testing.assert_close(actor_input[0, 78:142], torch.zeros(64))
    torch.testing.assert_close(actor_input[1, 142:206], torch.zeros(64))
    assert torch.count_nonzero(actor_input[0, 142:206]) > 0
    assert torch.count_nonzero(actor_input[1, 78:142]) > 0


def test_renet_uses_one_50d_amp_pipeline(tmp_path):
    frames = np.arange(3 * 50, dtype=np.float32).reshape(3, 50)
    motion_path = tmp_path / "locomotion.txt"
    motion_path.write_text(
        json.dumps(
            {
                "LoopMode": "Wrap",
                "FrameDuration": 0.05,
                "MotionWeight": 1.0,
                "Frames": frames.tolist(),
            }
        ),
        encoding="utf-8",
    )
    loader = AMPLoader(device="cpu", time_between_frames=0.05, motion_files=[motion_path])
    assert loader.observation_dim == 50

    runner_source = _source(RUNNER_PATH)
    assert runner_source.count("AMPLoader(") == 1
    assert runner_source.count("Discriminator(") == 1
    assert "recovery" not in runner_source.lower()
    assert '"discriminator_state_dict"' in runner_source
    assert '"amp_normalizer"' in runner_source

    algorithm_source = _source(ALGORITHM_PATH)
    assert "class RENetAMPPPO(AMPPPO)" in algorithm_source
    assert "recovery" not in algorithm_source.lower()


def test_environment_uses_direct_locomotion_action_reward_and_reset_paths():
    source = _source(ENV_PATH)
    assert "from isaaclab.managers import EventManager, RewardManager" in source
    assert "self.reward_manager = RewardManager(self.cfg.reward, self)" in source
    assert "self.action * self.action_scale" in source
    assert "+ self.robot.data.default_joint_pos" in source
    assert "self.episode_length_buf >= self.max_episode_length" in source
    assert "reset_buf |= time_out_buf" in source
    assert "terminal_amp_states" in source
    assert "recovery" not in source.lower()


def test_deployment_inputs_are_op_vp_only():
    for path in (PLAY_PATH, SIM2SIM_PATH):
        source = _source(path).lower()
        assert "recovery" not in source
        assert "beta" not in source
        assert "actor_mode" not in source
        assert "estimator_mask" in source

    actor_source = _source(ACTOR_PATH).lower()
    assert "recovery" not in actor_source
    assert "beta" not in actor_source
    assert "mode 2" not in actor_source


def test_exported_policy_keeps_the_estimator_mask_dynamic():
    spec = importlib.util.spec_from_file_location("renet_export", EXPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    policy = _make_policy().eval()
    exporter = module.RENetPolicyExporter(policy).eval()
    observations = torch.randn(2, policy.num_actor_obs)
    observations[1] = observations[0]
    mask_index = policy.proprio_actor_dim + policy.depth_flat_dim
    observations[:, mask_index] = torch.tensor([0.0, 1.0])

    traced = torch.jit.trace(
        exporter,
        torch.zeros(1, policy.num_actor_obs),
        strict=False,
    )
    with torch.inference_mode():
        expected = exporter(observations)
        actual = traced(observations)
    torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-5)
