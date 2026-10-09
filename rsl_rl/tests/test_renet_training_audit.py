from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
import torch

from test_renet_auxiliary_updates import ROOT, _extract_function


@pytest.mark.parametrize('side,body_id,other_body_id', [('L', 1, 3), ('R', 3, 1)])
def test_edge_penalty_counts_only_contacting_foot_and_missing_support(side, body_id, other_body_id):
    reward_fn = _extract_function(ROOT / 'legged_lab/mdp/rewards.py', 'single_foot_contact_area_penalty')
    tree = ast.parse((ROOT / 'legged_lab/envs/g1/RENet_cfg.py').read_text())
    rewards = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Reward')
    term = next(node.value for node in rewards.body if isinstance(node, ast.Assign)
                and node.targets[0].id == f'stair_edge_penalty_{side}')
    cfg = eval(compile(ast.Expression(term), 'RENet_cfg.py', 'eval'), {
        'RewTerm': lambda **kwargs: SimpleNamespace(**kwargs),
        'SceneEntityCfg': lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        'mdp': SimpleNamespace(single_foot_contact_area_penalty=reward_fn),
    })
    name = 'left' if side == 'L' else 'right'
    assert cfg.params['contact_sensor_cfg'].body_names == f'{name}_ankle_roll_link'
    assert cfg.params['ray_sensor_cfg'].name == f'{name}_feet_ray_caster'
    cfg.params['contact_sensor_cfg'].body_ids = [body_id]
    assert cfg.weight < 0 and cfg.params['threshold'] == .04
    forces = torch.zeros(6, 3, 4, 3)
    forces[[0, 1, 3, 4], 0, body_id, 2] = 20.
    forces[5, 0, other_body_id, 2] = 20.  # Opposite-foot contact must not trigger this term.
    forces[2, 1, body_id, 2] = 20.       # Old contact must not punish the current swing phase.
    distances = torch.tensor([[.01, .01, .01, .01], [.01, .01, .10, .10],
                              [.01, .01, .10, .10], [.10, .10, .10, .10],
                              [float('inf'), float('nan'), float('inf'), float('nan')],
                              [.10, .10, .10, .10]])
    hits = torch.zeros(6, 4, 3)
    hits[..., 2] = -distances
    sensors = {'contact_sensor': SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces)),
               f'{name}_feet_ray_caster': SimpleNamespace(data=SimpleNamespace(pos_w=torch.zeros(6, 3), ray_hits_w=hits))}
    env = SimpleNamespace(scene=SimpleNamespace(sensors=sensors))
    actual = cfg.func(env, **cfg.params)
    assert actual.shape == (6,) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.tensor([0., .5, 0., 1., 1., 0.]))


def test_calf_contacts_are_counted_once_and_foot_contacts_are_exempt():
    reward_fn = _extract_function(ROOT / "legged_lab/mdp/rewards.py", "undesired_contacts")
    tree = ast.parse((ROOT / "legged_lab/envs/g1/RENet_cfg.py").read_text())
    rewards = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Reward")
    term = next(node.value for node in rewards.body if isinstance(node, ast.Assign)
                and node.targets[0].id == "undesired_contacts")
    cfg = eval(compile(ast.Expression(term), "RENet_cfg.py", "eval"), {
        "RewTerm": lambda **kwargs: SimpleNamespace(**kwargs),
        "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        "mdp": SimpleNamespace(undesired_contacts=reward_fn),
    })
    urdf = ET.parse(ROOT / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.urdf")
    body_names = [link.attrib["name"] for link in urdf.findall("link")]
    sensor_cfg = cfg.params["sensor_cfg"]
    selected = [i for i, name in enumerate(body_names)
                if any(re.fullmatch(pattern, name) for pattern in sensor_cfg.body_names)]
    original = [i for i, name in enumerate(body_names) if re.fullmatch("(?!.*ankle.*).*", name)]
    assert set(selected) == set(original)
    assert all(sum(bool(re.fullmatch(pattern, body_names[i])) for pattern in sensor_cfg.body_names) == 1
               for i in selected)
    sensor_cfg.body_ids = selected
    forces = torch.zeros(5, 3, len(body_names), 3)
    forces[0, 0, body_names.index("left_knee_link"), 2] = 2.0
    forces[1, 0, body_names.index("right_knee_link"), 2] = 2.0
    for name in ("left_knee_link", "right_knee_link"):
        forces[2, 0, body_names.index(name), 2] = 2.0
        # A sub-threshold calf contact is not penalized.
        forces[4, 0, body_names.index(name), 2] = 0.5
    for name in ("left_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"):
        forces[3, 0, body_names.index(name), 2] = 20.0
    env = SimpleNamespace(scene=SimpleNamespace(sensors={
        "contact_sensor": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))}))
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([1, 1, 2, 0, 0]))


@pytest.mark.parametrize("terminated", [False, True])
def test_step_rewards_executed_command_then_returns_next_command_and_true_terminal_amp(terminated):
    step = _extract_function(ROOT / "legged_lab/envs/g1/RENet_env.py", "step", "G1RENetEnv")
    calls = []
    command = torch.tensor([[.5, 0., 0.], [-.5, 0., 0.]])
    original_command = command.clone()
    robot_data = SimpleNamespace(
        default_joint_pos=torch.zeros(2, 23),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        root_lin_vel_w=command.clone(),
        body_lin_vel_w=torch.zeros(2, 2, 3))
    amp = torch.arange(116, dtype=torch.float).reshape(2, 58)
    terminal_amp = amp[1:2].clone()
    env = SimpleNamespace(
        num_envs=2, num_actions=23, device="cpu", headless=True,
        renet_estimator_mask=torch.tensor([[1.], [0.]]),
        command_generator=SimpleNamespace(command=command), extras={"observations": {}},
        action_buffer=SimpleNamespace(compute=lambda actions: actions,
                                      time_lags=torch.zeros(2, dtype=torch.int), max_time_lag=0),
        clip_actions=100., action_scale=.25,
        feet_cfg=SimpleNamespace(body_ids=[0, 1]), ankle_link_ids=[0, 1],
        robot=SimpleNamespace(data=robot_data, set_joint_position_target=lambda actions: None),
        sim=SimpleNamespace(step=lambda **kwargs: None),
        scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None),
        contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 2, 3))),
        cfg=SimpleNamespace(sim=SimpleNamespace(decimation=1),
                            renet=SimpleNamespace(domain_rand_start_iteration=30000)), sim_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long), physics_dt=.02, step_dt=.02,
        event_manager=SimpleNamespace(available_modes=[]),
        _domain_rand_is_active=lambda: False,
        get_amp_obs_for_expert_trans=lambda: amp,
        check_reset=lambda: (torch.tensor([False, terminated]), torch.zeros(2, dtype=torch.bool)),
        append_renet_training_inputs=lambda obs: obs)

    def reward(dt):
        calls.append("reward")
        torch.testing.assert_close(command, original_command)
        return -(robot_data.root_lin_vel_w - command).square().sum(-1)

    def reset(env_ids):
        calls.append("reset")
        amp[env_ids] = -100.

    def next_command(dt):
        calls.append("command")
        command.fill_(.8)

    def observations():
        calls.append("observations")
        return command.clone(), torch.zeros(2, 1037)

    env.reward_manager = SimpleNamespace(compute=reward)
    env.reset = reset
    env.command_generator.compute = next_command
    env.compute_observations = observations
    obs, rewards, dones, extras = step(env, torch.zeros(2, 23))
    assert calls == ["reward", "reset", "command", "observations"]
    torch.testing.assert_close(rewards, torch.zeros(2))
    torch.testing.assert_close(obs, torch.full((2, 3), .8))
    torch.testing.assert_close(env._renet_reward_command, original_command)
    assert dones.tolist() == [False, terminated]
    if terminated:
        torch.testing.assert_close(extras["terminal_amp_states"], terminal_amp)
        assert (amp[1] == -100.).all()
    else:
        assert extras["terminal_amp_states"].shape == (0, 58)
