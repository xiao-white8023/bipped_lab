"""Exercise both depth AMP layouts with production features and expert files."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners import DepthAmpOnPolicyRunner
from rsl_rl.utils import AMPLoader, G1AnkleAMPLoader
from test_g1_depth import Environment, ROOT, runner_config


def playback_amp_configurator():
    path = ROOT / "legged_lab/scripts/play_depth.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_checkpoint_amp")
    scope = {"torch": torch, "os": os, "REPO_ROOT": str(ROOT)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


@pytest.mark.parametrize("dimension,configured", [(50, 58), (58, 50), (50, 50), (58, 58)])
def test_playback_selects_saved_amp_layout_without_changing_training_defaults(tmp_path, dimension, configured):
    env_defaults, agent_defaults = amp_configs(configured)
    env_cfg = SimpleNamespace(**copy.deepcopy(env_defaults))
    agent_cfg = SimpleNamespace(**copy.deepcopy(agent_defaults))
    path = tmp_path / "model.pt"
    torch.save({"task_name": "g1_depth", "amp_observation_contract": {"dimension": dimension}}, path)
    playback_amp_configurator()(env_cfg, agent_cfg, path)
    expected_env, expected_agent = amp_configs(dimension)
    assert vars(env_cfg) == expected_env
    assert vars(agent_cfg) == expected_agent
    assert (env_defaults, agent_defaults) == tuple(amp_configs(configured))


@pytest.mark.parametrize("metadata", [
    {"task_name": "g1_renet", "amp_observation_contract": {"dimension": 50}},
    {"task_name": "g1_depth"},
    {"task_name": "g1_depth", "amp_observation_contract": {"dimension": 51}},
    {"task_name": "g1_depth", "amp_observation_contract": {"dimension": 50.0}},
])
def test_playback_rejects_missing_or_incompatible_amp_identity_before_mutation(tmp_path, metadata):
    env_defaults, agent_defaults = amp_configs(58)
    env_cfg = SimpleNamespace(**copy.deepcopy(env_defaults))
    agent_cfg = SimpleNamespace(**copy.deepcopy(agent_defaults))
    path = tmp_path / "bad.pt"
    torch.save(metadata, path)
    with pytest.raises(ValueError, match="g1_depth"):
        playback_amp_configurator()(env_cfg, agent_cfg, path)
    assert vars(env_cfg) == env_defaults
    assert vars(agent_cfg) == agent_defaults


def amp_configs(dimension):
    """Evaluate the real switch and AMP fields without importing Isaac Kit."""
    path = ROOT / "legged_lab/envs/g1/g1_depth_cfg.py"
    tree = ast.parse(path.read_text())
    scope = {"LEGGED_LAB_ROOT": str(ROOT / "legged_lab")}
    for source in tree.body:
        node = copy.deepcopy(source)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if not (name.startswith("G1_DEPTH_AMP_") or name == "G1_DEPTH_TRAINING_DIR"):
                continue
            if name == "G1_DEPTH_AMP_DIM":
                node.value = ast.Constant(dimension)
        elif not (isinstance(node, ast.If) and "G1_DEPTH_AMP_DIM" in ast.unparse(node.test)):
            continue
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    fields = {"amp_observation_dim", "amp_joint_count", "amp_data_class_name",
              "amp_motion_files", "amp_motion_files_display"}
    configs = []
    for name in ("G1DEPTHENVCFG", "G1DEPTHAGENTCFG"):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        cfg = {}
        for node in cls.body:
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AnnAssign) else None
            if isinstance(target, ast.Name) and target.id in fields:
                cfg[target.id] = eval(compile(ast.Expression(node.value), str(path), "eval"), scope)
        configs.append(cfg)
    return configs


def environment_methods():
    path = ROOT / "legged_lab/envs/g1/g1_depth_env.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1DepthEnv")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_buffers")
    first = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "self.amp_observation_dim")
    last = next(i for i, n in enumerate(init.body[first:], first) if isinstance(n, ast.If)
                and "len(self.amp_joint_ids)" in ast.unparse(n.test))
    select = compile(ast.Module(body=init.body[first:last + 1], type_ignores=[]), str(path), "exec")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_amp_obs_for_expert_trans")

    def quat_apply(q, v):
        uv = torch.cross(q[..., 1:], v, dim=-1)
        return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))

    scope = {"torch": torch, "quat_apply": quat_apply,
             "quat_conjugate": lambda q: q * q.new_tensor([1, -1, -1, -1])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return select, scope[method.name]


class AmpEnvironment(Environment):
    def __init__(self, dimension):
        super().__init__()
        self.cfg = SimpleNamespace(amp_observation_dim=dimension)
        _, full_cfg = amp_configs(58)
        full = json.loads(Path(full_cfg["amp_motion_files"][0]).read_text())
        robot_names = full["AMPJointNames"][::-1]  # Simulator order differs from AMP order.
        self.local_ends = torch.tensor([[.15, .2, -.1], [.15, -.2, -.1], [0., .1, -.7], [0., -.1, -.7]])
        root = torch.zeros(2, 13)
        root[:, :3] = torch.tensor([[1., -2., 3.], [-3., 2., 1.]])
        root[0, 3] = 1.
        root[1, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        bodies = torch.zeros(2, 4, 13)
        bodies[0, :, :3] = self.local_ends + root[0, :3]
        rotated = self.local_ends.clone()
        rotated[:, 0], rotated[:, 1] = -self.local_ends[:, 1], self.local_ends[:, 0]
        bodies[1, :, :3] = rotated + root[1, :3]
        positions = torch.arange(46, dtype=torch.float32).reshape(2, 23) / 50.

        def find_joints(names, preserve_order):
            assert preserve_order
            return [robot_names.index(name) for name in names], list(names)

        self.robot = SimpleNamespace(find_joints=find_joints, data=SimpleNamespace(
            joint_pos=positions, joint_vel=-2 * positions, root_state_w=root, body_state_w=bodies))
        self.wrist_link_ids, self.ankle_link_ids = [0, 1], [2, 3]
        select, self.amp_method = environment_methods()
        exec(select, {"self": self})

    def get_amp_obs_for_expert_trans(self):
        return self.amp_method(self)

    def step(self, actions):
        obs, rewards, dones, extra = super().step(actions)
        extra["terminal_amp_states"] = torch.full((1, self.amp_observation_dim), 7.)
        return obs, rewards, dones, extra


@pytest.mark.parametrize("dimension", [50, 58])
def test_depth_switch_dataset_and_environment_features(dimension):
    env_cfg, agent_cfg = amp_configs(dimension)
    count = (dimension - 12) // 2
    assert env_cfg["amp_observation_dim"] == agent_cfg["amp_observation_dim"] == dimension
    assert agent_cfg["amp_joint_count"] == count
    loader_class = G1AnkleAMPLoader if dimension == 58 else AMPLoader
    assert agent_cfg["amp_data_class_name"] == loader_class.__name__
    group = "amp_full" if dimension == 58 else "amp_no_ankle"
    assert all(Path(path).parent.name == group for path in agent_cfg["amp_motion_files"])
    assert [Path(path).stem for path in agent_cfg["amp_motion_files"]] == [
        "walk_backward_20s", "walk_left_20s", "walk_right_20s", "walk_turns_40s"]
    loader = loader_class("cpu", .02, motion_files=agent_cfg["amp_motion_files"])
    assert loader.observation_dim == dimension
    env = AmpEnvironment(dimension)
    actual = env.get_amp_obs_for_expert_trans()
    assert actual.shape == (2, dimension) and env.num_actions == 23
    assert len(env.amp_joint_names) == count
    assert sum("ankle" in name for name in env.amp_joint_names) == (4 if dimension == 58 else 0)
    torch.testing.assert_close(actual[:, :count], env.robot.data.joint_pos[:, env.amp_joint_ids])
    torch.testing.assert_close(actual[:, count:2 * count], env.robot.data.joint_vel[:, env.amp_joint_ids])
    torch.testing.assert_close(actual[:, 2 * count:], env.local_ends.flatten().expand(2, -1), atol=1e-6, rtol=1e-6)
    for path in agent_cfg["amp_motion_files"]:
        assert json.loads(Path(path).read_text())["AMPJointNames"] == env.amp_joint_names
    for path in env_cfg["amp_motion_files_display"]:
        payload = json.loads(Path(path).read_text())
        assert Path(path).parent.name == "visualization"
        assert len(payload["Frames"][0]) == 58 and "FrameLayout" not in payload


def test_depth_50_is_exact_projection_of_same_58_motions_and_robot_states():
    full, reduced = AmpEnvironment(58), AmpEnvironment(50)
    indices = [full.amp_joint_names.index(name) for name in reduced.amp_joint_names]
    columns = indices + [i + 23 for i in indices] + list(range(46, 58))
    torch.testing.assert_close(reduced.get_amp_obs_for_expert_trans(),
                               full.get_amp_obs_for_expert_trans()[:, columns], atol=0, rtol=0)
    _, cfg58 = amp_configs(58)
    _, cfg50 = amp_configs(50)
    for p58, p50 in zip(cfg58["amp_motion_files"], cfg50["amp_motion_files"]):
        data58, data50 = json.loads(Path(p58).read_text()), json.loads(Path(p50).read_text())
        assert data58["FrameDuration"] == data50["FrameDuration"]
        torch.testing.assert_close(torch.tensor(data50["Frames"]), torch.tensor(data58["Frames"])[:, columns], atol=0, rtol=0)


@pytest.mark.parametrize("dimension", [49, 51, 57, 59, 58.0, True])
def test_depth_switch_rejects_invalid_dimensions(dimension):
    with pytest.raises(ValueError, match="G1_DEPTH_AMP_DIM must be 50 or 58"):
        amp_configs(dimension)


def train_config(dimension):
    config = runner_config()
    config.update(amp_configs(dimension)[1])
    config["amp_num_preload_transitions"] = 32
    return config


@pytest.mark.parametrize("dimension", [50, 58])
def test_depth_runner_updates_resumes_and_rejects_other_amp_layout(dimension, tmp_path):
    torch.set_num_threads(1)
    runner = DepthAmpOnPolicyRunner(AmpEnvironment(dimension), train_config(dimension), str(tmp_path / "first"))
    assert isinstance(runner.alg.amp_data, G1AnkleAMPLoader if dimension == 58 else AMPLoader)
    assert runner.alg.discriminator.input_dim == 2 * dimension
    assert runner.alg.amp_normalizer.mean.shape == (dimension,)
    assert runner.alg.amp_storage.states.shape[-1] == dimension
    assert runner.amp_observation_contract["dimension"] == dimension
    before = copy.deepcopy(runner.alg.discriminator.state_dict())
    runner.learn(1)
    assert any(not torch.equal(before[name], value) for name, value in runner.alg.discriminator.state_dict().items())
    checkpoint = tmp_path / "first/model_1.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["amp_observation_contract"]["dimension"] == dimension
    assert saved["optimizer_state_dict"]["state"]
    assert runner.alg.amp_normalizer.count > 1
    restored = DepthAmpOnPolicyRunner(AmpEnvironment(dimension), train_config(dimension), str(tmp_path / "resumed"))
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == restored.env.training_iteration == 1
    torch.testing.assert_close(torch.tensor(restored.alg.amp_normalizer.mean), torch.tensor(runner.alg.amp_normalizer.mean))
    for name, value in runner.alg.discriminator.state_dict().items():
        torch.testing.assert_close(restored.alg.discriminator.state_dict()[name], value, atol=0, rtol=0)
    restored.learn(1)
    assert (tmp_path / "resumed/model_2.pt").is_file()
    opposite = 108 - dimension
    incompatible = DepthAmpOnPolicyRunner(AmpEnvironment(opposite), train_config(opposite))
    before = copy.deepcopy(incompatible.alg.policy.state_dict())
    with pytest.raises(ValueError, match="amp_observation_contract; no weights have been loaded"):
        incompatible.load(str(checkpoint))
    for name, value in incompatible.alg.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    runner.writer.close()
    restored.writer.close()
