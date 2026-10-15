"""Independent VP parity, supervised updates, terminal alignment and export."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision.transforms as T

from rsl_rl.algorithms import AMPPPO
from rsl_rl.modules import Discriminator, G1DepthActorCritic, RENetActorCritic
from rsl_rl.runners import DepthAmpOnPolicyRunner
from legged_lab.scripts.export_depth import export_depth_policy
from test_renet_deployed_checkpoint import production_policy_kwargs


ROOT = Path(__file__).parents[2]


def depth_kwargs():
    tree = ast.parse((ROOT / "legged_lab/envs/g1/g1_depth_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1DEPTHAGENTCFG")
    call = next(n.value for n in cls.body if isinstance(n, ast.Assign) and n.targets[0].id == "policy")
    scope = {"G1_DEPTH_VP_PROPRIO_HISTORY_FRAMES": 10, "CnnMlpCfg": lambda **kw: kw}
    values = {n.arg: eval(compile(ast.Expression(n.value), "config", "eval"), scope) for n in call.keywords}
    values.pop("class_name")
    values["CnnMlp"].pop("num_heads", None)
    values["CnnMlp"].pop("embed_dim", None)
    return values


def policy():
    torch.set_num_threads(1)
    kw = depth_kwargs()
    kw.update(actor_hidden_dims=[32], critic_hidden_dims=[32], proprio_embed_dims=[32], vp_encoder_dims=[32])
    return G1DepthActorCritic(1932, 1037, 23, **kw)


def observations(batch=2):
    return torch.randn(batch, 1932, generator=torch.Generator().manual_seed(22)) * 0.1


def test_extracted_vp_matches_current_renet_including_actor_effective_columns():
    new = policy().eval()
    kw = production_policy_kwargs(output_type="multi_head")
    kw.update(actor_hidden_dims=[32], critic_hidden_dims=[32], proprio_embed_dims=[32], vp_encoder_dims=[32])
    old = RENetActorCritic(2120, 1037, 23, **kw).eval()
    source = old.state_dict()
    target = new.state_dict()
    for name in target:
        if name == "actor.0.weight":
            target[name] = torch.cat((source[name][:, :78], source[name][:, 142:]), dim=1)
        else:
            target[name] = source[name].clone()
    new.load_state_dict(target, strict=True)
    for batch in (1, 7):
        x = observations(batch)
        routed = torch.cat((x[:, :780], torch.randn(batch, 187), x[:, 780:], torch.zeros(batch, 1)), dim=1)
        with torch.no_grad():
            expected = old.act_inference(routed)
            actual = new.act_inference(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(new._last_vp_latent, old._last_vp_latent)
            torch.testing.assert_close(new.predict_velocity(), old.predict_velocity()["vp"])
            torch.testing.assert_close(new.predict_feet_height(), old.predict_feet_height())
            torch.testing.assert_close(new.predict_terrain(), old.predict_terrain())
            critic = torch.randn(batch, 1037)
            torch.testing.assert_close(new.evaluate(critic), old.evaluate(critic, routed))
    assert new.actor[0].in_features == 147 and new.predict_terrain().shape == (7, 99)
    assert not any(name.startswith(("lp_", "lidar")) for name in new.state_dict())
    assert not hasattr(new, "estimator_mask_dim")


def test_vae_sampling_does_not_resample_actor_context_or_log_probabilities():
    p = policy()
    x = observations()
    actions = p.act(x)
    mean, logprob = p.action_mean.clone(), p.get_actions_log_prob(actions).clone()
    first, mu, logvar = p.predict_next_proprio()
    second, _, _ = p.predict_next_proprio()
    assert first.shape == (2, 78) and mu.shape == logvar.shape == (2, 32)
    assert not torch.equal(first, second)
    p.act(x)
    torch.testing.assert_close(p.action_mean, mean, rtol=0, atol=0)
    torch.testing.assert_close(p.get_actions_log_prob(actions), logprob, rtol=0, atol=0)


class Expert:
    observation_dim = 58
    joint_names = None
    FRAME_LAYOUT = "g1_23dof_full"

    def __init__(self, *args, **kwargs):
        pass

    def feed_forward_generator(self, num_mini_batches, mini_batch_size):
        for _ in range(num_mini_batches):
            yield torch.randn(mini_batch_size, 58), torch.randn(mini_batch_size, 58)


def algorithm(p):
    return AMPPPO(
        p, Discriminator(116, .3, [16, 8], "cpu", .7), Expert(), None,
        amp_replay_buffer_size=32, num_learning_epochs=1, num_mini_batches=1,
        vel_estimation_coef=1., terrain_recon_coef=.5, feet_height_coef=.2,
        terrain_recon_target_normalization="fixed", single_critic_dim=85, critic_history_len=10,
        vel_in_critic_offset=78, feet_height_in_critic_offset=83,
        use_feet_height_valid_mask=True, use_terrain_height_valid_mask=True,
        vp_vae_coef=1., vp_vae_beta=.01,
    )


def test_validity_and_done_targets_remain_aligned_and_all_vp_heads_update():
    p = policy()
    a = algorithm(p)
    a.init_storage("rl", 2, 2, [1932], [1037], [23])
    with torch.no_grad():
        for step in range(2):
            feet = torch.tensor([[True, False], [True, True]])
            terrain = torch.ones(2, 187, dtype=torch.bool)
            terrain[:, :17] = False
            a.act(observations(), torch.randn(2, 1037), torch.randn(2, 58), feet_height_valid=feet, terrain_height_valid=terrain)
            target = torch.full((2, 78), float(step + 1))
            a.process_env_step(torch.ones(2), torch.tensor([False, True]), {}, torch.randn(2, 58), next_proprio=target)
            target.fill_(10000.)
        a.compute_returns(torch.randn(2, 1037))
    assert a.storage.next_proprio_valid.flatten().tolist() == [True, False, True, False]
    assert a.storage.next_proprio[:, 0, 0].tolist() == [1., 2.]
    assert not a.storage.next_proprio[:, 1].any()
    before = {n: t.clone() for n, t in p.named_parameters()}
    losses = a.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    for prefix in ("cnn", "proprio_embedding", "vp_attention", "vp_encoder", "vp_gru", "actor", "critic",
                   "vp_velocity_head", "vp_feet_height_head", "vp_context_head", "vp_terrain_latent_head",
                   "vp_terrain_decoder", "vp_next_proprio_decoder"):
        assert any(not torch.equal(t, before[n]) for n, t in p.named_parameters() if n.startswith(prefix + ".")), prefix
    assert not any("lp" in key for key in losses)


class Environment:
    device = "cpu"
    num_envs = 2
    num_actions = 23
    step_dt = .02
    max_episode_length = 1000
    amp_joint_names = [f"joint_{i}" for i in range(23)]
    depth_observation_contract = {"version": 1, "task": "g1_depth", "actor_observation_dim": 1932}

    def __init__(self):
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.reset_env_ids = torch.empty(0, dtype=torch.long)
        self.training_iteration = 0
        self.ticks = 0

    def set_training_iteration(self, iteration):
        self.training_iteration = iteration

    def get_observations(self):
        critic = torch.full((2, 1037), float(self.ticks))
        return observations(), {"observations": {"critic": critic, "feet_height_valid": torch.ones(2, 2).bool(),
                                                "terrain_height_valid": torch.ones(2, 187).bool()}}

    def get_amp_obs_for_expert_trans(self):
        return torch.zeros(2, 58)

    def step(self, actions):
        self.ticks += 1
        self.reset_env_ids = torch.tensor([1])
        obs, extra = self.get_observations()
        extra["terminal_amp_states"] = torch.full((1, 58), 7.)
        extra["time_outs"] = torch.zeros(2).bool()
        return obs, torch.ones(2), torch.tensor([False, True]), extra


def runner_config():
    kw = depth_kwargs()
    kw.update(class_name="G1DepthActorCritic", actor_hidden_dims=[32], critic_hidden_dims=[32],
              proprio_embed_dims=[32], vp_encoder_dims=[32])
    return {
        "policy": kw, "algorithm": {"class_name": "AMPPPO", "num_learning_epochs": 1, "num_mini_batches": 1,
            "single_critic_dim": 85, "critic_history_len": 10, "vel_in_critic_offset": 78,
            "feet_height_in_critic_offset": 83, "vp_vae_coef": 1., "vp_vae_beta": .01,
            "vel_estimation_coef": 1., "terrain_recon_coef": .5, "feet_height_coef": .2,
            "use_feet_height_valid_mask": True, "use_terrain_height_valid_mask": True,
            "terrain_recon_target_normalization": "fixed", "amp_replay_buffer_size": 32},
        "amp_joint_count": 23, "amp_observation_dim": 58, "amp_data_class_name": "G1AnkleAMPLoader",
        "amp_num_preload_transitions": 1, "amp_motion_files": [], "amp_reward_coef": .3,
        "amp_discr_hidden_dims": [16, 8], "amp_task_reward_lerp": .7, "min_normalized_std": [.05] * 23,
        "num_steps_per_env": 2, "save_interval": 1, "empirical_normalization": False, "logger": "tensorboard",
    }


def test_runner_terminal_amp_clean_targets_strict_resume_and_rejection(monkeypatch, tmp_path):
    import rsl_rl.runners.depth_amp_on_policy_runner as module
    monkeypatch.setattr(module, "G1AnkleAMPLoader", Expert)
    env = Environment()
    runner = DepthAmpOnPolicyRunner(env, runner_config(), str(tmp_path / "first"))
    captures = []
    original = runner.alg.process_env_step

    def capture(rewards, dones, infos, amp_obs, next_proprio=None):
        captures.append((amp_obs.clone(), next_proprio.clone()))
        return original(rewards, dones, infos, amp_obs, next_proprio=next_proprio)

    monkeypatch.setattr(runner.alg, "process_env_step", capture)
    runner.learn(1)
    runner.writer.flush()
    assert captures[0][0][1].tolist() == [7.] * 58
    assert captures[0][0][0].tolist() == [0.] * 58
    assert captures[0][1][0].tolist() == [1.] * 78
    checkpoint = tmp_path / "first/model_1.pt"
    saved = torch.load(checkpoint, weights_only=False)
    restored = DepthAmpOnPolicyRunner(Environment(), runner_config(), str(tmp_path / "second"))
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == restored.alg.current_iteration == restored.env.training_iteration == 1
    assert saved["optimizer_state_dict"]["state"]
    for key, value in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(restored.alg.policy.state_dict()[key], value, rtol=0, atol=0)
    restored.learn(1)
    assert restored.current_learning_iteration == 2 and (tmp_path / "second/model_2.pt").is_file()
    before = copy.deepcopy(restored.alg.policy.state_dict())
    for key, replacement in (("task_name", "g1_renet"), ("network_configuration", {}), ("amp_observation_contract", {})):
        damaged = dict(saved)
        damaged[key] = replacement
        path = tmp_path / "incompatible.pt"
        torch.save(damaged, path)
        with pytest.raises(ValueError, match="no weights have been loaded"):
            restored.load(str(path))
        for name, value in restored.alg.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    runner.writer.close()
    restored.writer.close()


def depth_environment_methods():
    source = ast.parse((ROOT / "legged_lab/envs/g1/g1_depth_env.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in (
        "get_processed_deepcamera", "get_deepcamera_history", "_depth_noise_is_active", "append_depth_training_inputs")]
    extracted = ast.Module(body=[ast.ClassDef(name="DepthSensors", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    namespace = {"torch": torch, "T": T}
    exec(compile(ast.fix_missing_locations(extracted), "depth_sensors", "exec"), namespace)
    return namespace["DepthSensors"]()


def test_camera_capture_history_reset_and_raw_validity():
    env = depth_environment_methods()
    env.num_envs, env.device, env.step_dt = 2, "cpu", .02
    env.training_iteration = 0
    env.cfg = SimpleNamespace(
        depth_noise_force_active=False, depth_noise_start_iteration=0, depth_blur_probability=.5,
        robot=SimpleNamespace(depth_crop=(18, 0, 16, 16), depth_max=2.5, depth_update_interval=5),
        scene=SimpleNamespace(camera=SimpleNamespace(add_camera_noise=False,
            camera=SimpleNamespace(pattern_cfg=SimpleNamespace(height=36, width=64), min_distance=.1))))
    raw = torch.ones(2, 36, 64, 1)
    raw[0, 18, 16, 0] = float("nan")
    env.camera = SimpleNamespace(current_timestamps=torch.zeros(2), data=SimpleNamespace(output={"distance_to_image_plane": raw}))
    env.depth_history_frames, env.depth_initialized = 2, torch.zeros(2).bool()
    env.depth_last_capture_time = torch.full((2,), -torch.inf)
    env.depth_buffer = torch.zeros(2, 2, 18, 32)
    env.depth_valid_buffer = torch.zeros_like(env.depth_buffer)
    env.depth_current_valid = torch.zeros(2, 18, 32)
    env.extras = {}
    first = env.get_deepcamera_history().clone()
    torch.testing.assert_close(first[:, 0], first[:, 1])
    assert first[0, 0, 0, 0] == 1 and env.depth_valid_buffer[0, 0, 0, 0] == 0
    raw.fill_(2.)
    env.camera.current_timestamps.fill_(.02)
    torch.testing.assert_close(env.get_deepcamera_history(), first)
    env.camera.current_timestamps.fill_(.1)
    history = env.get_deepcamera_history().clone()
    torch.testing.assert_close(history[:, 0], first[:, 1])
    torch.testing.assert_close(history[:, 1], torch.full((2, 18, 32), .8))
    env.depth_initialized[0] = False  # reset invalidates only this sensor row
    raw[0].fill_(.5)
    torch.testing.assert_close(env.get_deepcamera_history()[0], torch.full((2, 18, 32), .2))
    result = env.append_depth_training_inputs(torch.zeros(2, 780))
    assert result.shape == (2, 1932) and not any("lp" in n for n in env.extras["observations"])


def test_depth_export_reloads_both_formats_and_restores_attention_settings(tmp_path):
    p = policy().eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    errors = export_depth_policy(p, torch.nn.Identity(), tmp_path,
                                 {"task": "g1_depth", "actor_observation_dim": 1932})
    assert max(errors.values()) < 1e-5
    assert torch.backends.mha.get_fastpath_enabled() == fastpath
    assert (tmp_path / "policy.pt").is_file() and (tmp_path / "policy.onnx").is_file()
