"""CPU policy/observation checks; configuration imports do not start Isaac Sim."""

import ast
import copy
import math
import os
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "legged_lab/envs/g1/g1_film_cfg.py"
ENV = ROOT / "legged_lab/envs/g1/g1_film_env.py"


class _Config:
    """Only constructor/default-copy semantics, not an Isaac simulation mock."""

    def __init__(self, *args, **kwargs):
        for cls in reversed(type(self).__mro__):
            for key, value in vars(cls).items():
                if not key.startswith("_") and not callable(value):
                    setattr(self, key, copy.deepcopy(value))
        if args:
            self.name = args[0]
        self.__dict__.update(kwargs)
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()

    def to_dict(self):
        def convert(value):
            if isinstance(value, _Config) or getattr(type(value), "_test_config", False):
                return {key: convert(val) for key, val in vars(value).items()}
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(convert(val) for val in value)
            return value
        return convert(self)


_Config.OffsetCfg = _Config


def _configclass(cls):
    cls._test_config = True
    cls.__init__ = _Config.__init__
    cls.to_dict = _Config.to_dict
    return cls


class _ConfigTypes:
    def __getattr__(self, name):
        return type(name, (_Config,), {"inverted": "Inverted" in name})


def _load_config():
    """Execute the actual declarations with inert config types and imports removed."""
    tree = ast.parse(CFG.read_text())
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                namespace[alias.asname or alias.name] = _Config
    namespace.update(
        __name__="film_configuration_contract",
        math=math,
        copy=copy,
        os=os,
        configclass=_configclass,
        terrain_gen=_ConfigTypes(),
        perlin=_ConfigTypes(),
        mdp=types.SimpleNamespace(),
        legged_lab=types.SimpleNamespace(__file__=str(ROOT / "legged_lab/__init__.py")),
        get_link_prim_targets=lambda links: [],
        G1_CFG=_Config(),
        G1_29CFG=_Config(),
        G1_29DOF_LINKS=[],
        TIENKUNG2LITE_CFG=_Config(),
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "mdp":
            setattr(namespace["mdp"], node.attr, "mdp." + node.attr)
    # These imported terrain instances only belong to the legacy import list.
    for name in ["GRAVEL_TERRAINS_CFG", "ROUGH_TERRAINS_CFG", "flex_terrain_CFG", "ROUGH_PERLIN_TERRAINS_CFG"]:
        namespace[name] = _Config()
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(CFG), "exec"), namespace)
    return namespace


def test_switch_changes_both_policy_heads_and_keeps_configs_independent():
    ns = _load_config()
    agent_type = ns["G129MOE_FILMAGENTENV"]
    mlp, moe = agent_type(), agent_type(use_moe=True)
    assert mlp.policy.class_name == "ActorCritic"
    assert moe.policy.class_name == "MoeActorCritic"
    assert mlp.experiment_name == "g1_noamp_wo_moe"
    assert moe.experiment_name == "g1_noamp"
    a, b = mlp.to_dict(), moe.to_dict()
    for value in (a, b):
        for key in ["use_moe", "experiment_name", "run_name"]:
            value.pop(key)
        value["policy"].pop("class_name")
    assert a == b
    mlp.policy.single_proprio_dim = 7
    assert moe.policy.single_proprio_dim == 100
    assert agent_type().policy.single_proprio_dim == 100


def test_june_configuration_contract():
    ns = _load_config()
    env = ns["G129MOE_FILMENVCFG"]()
    agent = ns["G129MOE_FILMAGENTENV"]()
    assert agent.policy.single_proprio_dim == 100
    assert agent.policy.num_experts == 5
    assert agent.policy.use_gru and agent.policy.use_film_cnn
    assert not agent.policy.use_separate_moe_gate_input
    assert agent.policy.use_vel_estimation and agent.policy.use_terrain_recon
    assert agent.algorithm.single_critic_dim == 124
    assert agent.algorithm.vel_in_critic_offset == 102
    assert not agent.algorithm.use_moe_balance_loss
    assert agent.algorithm.moe_balance_coef == 0
    assert not agent.resume
    terrain = env.scene.terrain_generator
    assert (terrain.num_rows, terrain.num_cols) == (10, 20)
    assert len(terrain.sub_terrains) == 7
    assert terrain.sub_terrains["gap"].proportion == 0.15
    assert terrain.sub_terrains["gap"].gap_width_range == (0.05, 0.4)
    assert terrain.sub_terrains["up_slope_terrain"].slope_range == (0.0837, 0.55)
    assert env.commands.ranges.lin_vel_y == (-0.5, 0.5)
    assert env.commands.ranges.ang_vel_z == (-1.57, 1.57)
    assert env.reward.gait_phase_contact.func == "mdp.gait_phase_contact"
    assert set(env.reward.gait_phase_contact.params) == {"sensor_cfg", "stance_threshold"}
    assert env.reward.joint_deviation_waist_roll is None
    assert env.reward.joint_deviation_waist_yaw is None


def test_actual_observation_function_keeps_velocity_supervision_aligned():
    tree = ast.parse(ENV.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1MOEFILMEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute_current_observations")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENV), "exec"), namespace)
    batch = 2
    velocity = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    data = types.SimpleNamespace(
        root_ang_vel_b=torch.zeros(batch, 3), projected_gravity_b=torch.zeros(batch, 3),
        joint_pos=torch.zeros(batch, 29), default_joint_pos=torch.zeros(batch, 29),
        joint_vel=torch.zeros(batch, 29), default_joint_vel=torch.zeros(batch, 29),
        root_lin_vel_b=velocity, root_pos_w=torch.zeros(batch, 3),
    )
    env = types.SimpleNamespace(
        robot=types.SimpleNamespace(data=data), num_envs=batch,
        contact_sensor=types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w_history=torch.zeros(batch, 1, 2, 3))),
        command_generator=types.SimpleNamespace(command=torch.zeros(batch, 3)),
        action_buffer=types.SimpleNamespace(_circular_buffer=types.SimpleNamespace(buffer=torch.zeros(batch, 1, 29))),
        obs_scales=types.SimpleNamespace(**{k: 1. for k in ["ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions", "lin_vel", "feet_pos", "feet_vel", "contact_force"]}),
        leg_phase=torch.tensor([[0., .5], [.25, .75]]), feet_cfg=types.SimpleNamespace(body_ids=[0, 1]),
        cfg=types.SimpleNamespace(scene=types.SimpleNamespace(privileged_info=types.SimpleNamespace(enable_feet_info=True, enable_feet_contact_force=True, enable_root_height=True))),
        feet_pos_in_body=torch.zeros(batch, 2, 3), feet_vel_in_body=torch.zeros(batch, 2, 3),
        _compute_feet_state=lambda: None,
    )
    actor, critic = namespace["compute_current_observations"](env)
    assert actor.shape == (batch, 100)
    assert critic.shape == (batch, 124)
    torch.testing.assert_close(critic[:, 102:105], velocity)
    torch.testing.assert_close(actor[:, 96:98], torch.sin(2 * torch.pi * env.leg_phase))
    torch.testing.assert_close(actor[:, 98:100], torch.cos(2 * torch.pi * env.leg_phase))


@pytest.mark.parametrize("use_moe", [False, True])
def test_real_policy_forward_auxiliary_gradients_and_checkpoint_roundtrip(use_moe, tmp_path):
    from rsl_rl.modules import ActorCritic, MoeActorCritic
    from rsl_rl.modules.moe import MoeLayer

    torch.set_num_threads(1)
    config = _load_config()["G129MOE_FILMAGENTENV"](use_moe=use_moe).policy.to_dict()
    name = config.pop("class_name")
    config["CnnMlp"].pop("num_heads")
    config["CnnMlp"].pop("embed_dim")
    if not use_moe:
        config.pop("num_experts")
        config.pop("gate_hidden_dim")
    policy_type = {"ActorCritic": ActorCritic, "MoeActorCritic": MoeActorCritic}[name]
    model = policy_type(3304, 1427, 29, **config)
    assert isinstance(model.actor, MoeLayer) is use_moe
    assert isinstance(model.critic, MoeLayer) is use_moe
    if use_moe:
        assert len(model.actor.experts) == len(model.critic.experts) == 5
    obs = torch.randn(2, 3304)
    actions = model.act_inference(obs)
    values = model.evaluate(torch.randn(2, 1427))
    velocity, terrain = model.predict_velocity(), model.predict_terrain()
    assert actions.shape == (2, 29) and values.shape == (2, 1)
    assert velocity.shape == (2, 3) and terrain.shape == (2, 99)
    loss = actions.square().mean() + values.square().mean() + velocity.square().mean() + terrain.square().mean()
    loss.backward()
    for module in [model.cnn, model.history_encoder, model.depth_gru, model.vel_estimator, model.terrain_decoder]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters())
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = policy_type(3304, 1427, 29, **config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.act_inference(obs), actions)


def test_mlp_runner_completes_ppo_update_with_both_auxiliary_losses():
    from rsl_rl.runners.film_on_policy_runner import FilmOnPolicyRunner

    torch.set_num_threads(1)
    config = _load_config()["G129MOE_FILMAGENTENV"]().to_dict()
    config["num_steps_per_env"] = 2
    config["algorithm"]["num_learning_epochs"] = 1
    config["algorithm"]["num_mini_batches"] = 1
    env = types.SimpleNamespace(
        num_actions=29,
        num_envs=2,
        get_observations=lambda: (
            torch.randn(2, 3304),
            {"observations": {"critic": torch.randn(2, 1427)}},
        ),
    )
    runner = FilmOnPolicyRunner(env, config, device="cpu")
    runner.alg.current_iteration = 500
    for _ in range(2):
        with torch.inference_mode():
            runner.alg.act(torch.randn(2, 3304), torch.randn(2, 1427))
            runner.alg.process_env_step(torch.ones(2), torch.zeros(2), {})
    with torch.inference_mode():
        runner.alg.compute_returns(torch.randn(2, 1427))
    losses = runner.alg.update()
    assert losses["vel_estimation"] > 0
    assert losses["terrain_recon"] > 0
    assert all(math.isfinite(value) for value in losses.values())
