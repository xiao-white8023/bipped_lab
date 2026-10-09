from __future__ import annotations

import ast
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rsl_rl.runners import AmpOnPolicyRunner, RENetAmpOnPolicyRunner
from rsl_rl.utils import AMPLoader, G1AnkleAMPLoader
from test_renet_lp_vp_training import _depth_cnn_cfg, _lidar_cnn_cfg


ROOT = Path(__file__).parents[2]
TASKS = {
    'g1_renet': ('RENet_cfg.py', 'RENet_env.py', 'G1_RENET_AMP_', 'G1RENETENVCFG', 'G1RENETAGENTCFG', 'G1RENetEnv'),
    'g1_rough': ('g1_rough_cfg.py', 'g1_rough_env.py', 'G1_ROUGH_AMP_', 'G129WALK_ROUGHENVCFG', 'G129WALK_ROUGHAGENTENV', 'G1ROUGHEnv'),
}
FIELDS = {'amp_observation_dim', 'amp_joint_count', 'amp_data_class_name',
          'amp_motion_files', 'amp_motion_files_display'}


def _task_cfg(task, dim):
    """Evaluate the production AMP cfg fields without importing Kit decorators."""
    filename, _, prefix, env_class, agent_class, _ = TASKS[task]
    path = ROOT / 'legged_lab/envs/g1' / filename
    tree = ast.parse(path.read_text())
    scope = {'LEGGED_LAB_ROOT': str(ROOT / 'legged_lab')}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith(prefix):
            node = copy.deepcopy(node)
            if node.targets[0].id == prefix + 'DIM':
                node.value = ast.Constant(dim)
        elif not (isinstance(node, ast.If) and prefix + 'DIM' in ast.unparse(node.test)):
            continue
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    configs = []
    for class_name in (env_class, agent_class):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        cfg = {}
        for node in cls.body:
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AnnAssign) else None
            if isinstance(target, ast.Name) and target.id in FIELDS:
                cfg[target.id] = eval(compile(ast.Expression(node.value), str(path), 'eval'), scope)
        configs.append(cfg)
    return configs


def _select_joints(task, proxy):
    _, filename, _, _, _, class_name = TASKS[task]
    path = ROOT / 'legged_lab/envs/g1' / filename
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_buffers')
    first = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'self.amp_observation_dim')
    last = next(i for i, n in enumerate(init.body[first:], first) if isinstance(n, ast.If)
                and 'len(self.amp_joint_ids)' in ast.unparse(n.test))
    scope = {'self': proxy}
    exec(compile(ast.Module(body=init.body[first:last + 1], type_ignores=[]), str(path), 'exec'), scope)


def _quat_apply(q, v):
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def _amp_method(task):
    _, filename, _, _, _, class_name = TASKS[task]
    path = ROOT / 'legged_lab/envs/g1' / filename
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_amp_obs_for_expert_trans')
    scope = {'torch': torch, 'quat_apply': _quat_apply,
             'quat_conjugate': lambda q: q * q.new_tensor([1, -1, -1, -1])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method.name]


def _env(task, dim, critic_frames=None):
    if critic_frames is None:
        critic_frames = 5 if task == 'g1_renet' else 10
    full_names = json.loads((ROOT / 'legged_lab/envs/g1/datasets/g1_23dof_locomotion_v3/manifest.json').read_text())['amp_joint_order']
    robot_names = full_names[::-1]  # The simulator's action order need not match AMP order.
    local_ends = torch.tensor([[.15, .2, -.1], [.15, -.2, -.1], [0., .1, -.7], [0., -.1, -.7]])
    root = torch.zeros(2, 13)
    root[:, :3] = torch.tensor([[1., -2., 3.], [-3., 2., 1.]])
    root[0, 3] = 1.
    root[1, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    bodies = torch.zeros(2, 4, 13)
    bodies[0, :, :3] = local_ends + root[0, :3]
    rotated = local_ends.clone()
    rotated[:, 0], rotated[:, 1] = -local_ends[:, 1], local_ends[:, 0]
    bodies[1, :, :3] = rotated + root[1, :3]
    positions = torch.arange(46, dtype=torch.float32).reshape(2, 23) / 50.
    data = SimpleNamespace(joint_pos=positions, joint_vel=-2 * positions,
                           root_state_w=root, body_state_w=bodies)
    def find_joints(names, preserve_order):
        assert preserve_order
        return [robot_names.index(name) for name in names], list(names)
    proxy = SimpleNamespace(cfg=SimpleNamespace(amp_observation_dim=dim),
        robot=SimpleNamespace(data=data, find_joints=find_joints),
        wrist_link_ids=[0, 1], ankle_link_ids=[2, 3])
    _select_joints(task, proxy)
    method = _amp_method(task)
    proxy.get_amp_obs_for_expert_trans = lambda: method(proxy)
    proxy.num_envs, proxy.num_actions, proxy.step_dt = 2, 23, .02
    proxy.get_observations = lambda: (torch.zeros(2, 2120), {
        'observations': {'critic': torch.zeros(2, 85 * critic_frames + 187)}})
    proxy.local_ends = local_ends
    return proxy


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('dim', [50, 58])
def test_cfg_switch_matches_dataset_loader_and_real_environment_feature_method(task, dim):
    env_cfg, agent_cfg = _task_cfg(task, dim)
    assert env_cfg['amp_observation_dim'] == agent_cfg['amp_observation_dim'] == dim
    count = 19 if dim == 50 else 23
    assert agent_cfg['amp_joint_count'] == count
    assert agent_cfg['amp_data_class_name'] == ('AMPLoader' if dim == 50 else 'G1AnkleAMPLoader')
    motion_files = agent_cfg['amp_motion_files']
    if task == 'g1_rough' or dim == 50:
        dataset = 'motion_amp_expert_no_ankle' if dim == 50 else 'motion_amp_expert'
        assert all(Path(path).parent.name == dataset for path in motion_files)
        assert all('motion_visualization/original_pkl/' in path
                   for path in env_cfg['amp_motion_files_display'])
        expected_names = ['stand_to_walk', 'walk_turn_around', 'walk_turn_left', 'walk_turn_right']
    else:
        dataset = 'g1_23dof_locomotion_v3'
        assert all(f'{dataset}/amp/' in path for path in motion_files)
        assert all(f'{dataset}/visualization/' in path for path in env_cfg['amp_motion_files_display'])
        expected_names = ['walk_forward_turns', 'walk_backward', 'walk_left', 'walk_right']
    assert [Path(path).stem for path in motion_files] == expected_names
    loader_class = AMPLoader if dim == 50 else G1AnkleAMPLoader
    loader = loader_class('cpu', .02, motion_files=agent_cfg['amp_motion_files'])
    assert loader.observation_dim == dim
    env = _env(task, dim)
    actual = env.get_amp_obs_for_expert_trans()
    assert actual.shape == (2, dim)
    assert len(env.amp_joint_names) == count
    assert sum('ankle' in name for name in env.amp_joint_names) == (0 if dim == 50 else 4)
    torch.testing.assert_close(actual[:, :count], env.robot.data.joint_pos[:, env.amp_joint_ids])
    torch.testing.assert_close(actual[:, count:2 * count], env.robot.data.joint_vel[:, env.amp_joint_ids])
    torch.testing.assert_close(actual[:, 2 * count:], env.local_ends.reshape(1, 12).expand(2, -1), atol=1e-6, rtol=1e-6)
    if loader.joint_names is not None:
        assert loader.joint_names == env.amp_joint_names


@pytest.mark.parametrize('task', TASKS)
def test_50_dim_environment_is_exact_non_ankle_projection_of_58(task):
    full, old = _env(task, 58), _env(task, 50)
    indices = [full.amp_joint_names.index(name) for name in old.amp_joint_names]
    full_obs = full.get_amp_obs_for_expert_trans()
    projected = torch.cat((full_obs[:, indices], full_obs[:, [i + 23 for i in indices]], full_obs[:, 46:]), -1)
    torch.testing.assert_close(old.get_amp_obs_for_expert_trans(), projected, atol=0, rtol=0)


@pytest.mark.parametrize('dim', [50, 58])
def test_original_g1_rough_clips_preserve_all_frames_fps_and_source_features(dim):
    env_cfg, agent_cfg = _task_cfg('g1_rough', dim)
    paths = [Path(path) for path in agent_cfg['amp_motion_files']]
    manifest = json.loads((paths[0].parent / 'manifest.json').read_text())
    assert list(manifest['clips']) == ['stand_to_walk', 'walk_turn_around', 'walk_turn_left', 'walk_turn_right']
    assert manifest['amp_dimension'] == dim
    indices = [i for i in range(23) if i not in (4, 5, 10, 11)]
    unique = []
    for path, preview_path in zip(paths, env_cfg['amp_motion_files_display']):
        source = pickle.loads((ROOT / 'legged_lab/envs/g1/datasets/pkl' / (path.stem + '.pkl')).read_bytes())
        payload = json.loads(path.read_text())
        frames = np.asarray(payload['Frames'])
        preview = np.asarray(json.loads(Path(preview_path).read_text())['Frames'])
        q = np.asarray(source['dof_pos'])
        dt = 1 / float(source['fps'])
        velocity = np.empty_like(q)
        velocity[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
        velocity[0] = (-3*q[0] + 4*q[1] - q[2]) / (2 * dt)
        velocity[-1] = (3*q[-1] - 4*q[-2] + q[-3]) / (2 * dt)
        selected = indices if dim == 50 else list(range(23))
        count = len(selected)
        assert frames.shape == (len(q), dim)
        assert np.isfinite(frames).all() and abs(payload['FrameDuration'] - dt) < 1e-12
        assert payload['LoopMode'] == 'Clamp' and payload['MotionWeight'] == .5
        np.testing.assert_allclose(frames[:, :count], q[:, selected], atol=6e-9, rtol=0)
        np.testing.assert_allclose(frames[:, count:2*count], velocity[:, selected], atol=6e-9, rtol=0)
        np.testing.assert_allclose(preview[:, :3], source['root_pos'], atol=6e-9, rtol=0)
        full_path = ROOT / 'legged_lab/envs/g1/datasets/motion_amp_expert' / path.name
        full = np.asarray(json.loads(full_path.read_text())['Frames'])
        if dim == 50:
            np.testing.assert_array_equal(frames, np.c_[full[:, indices], full[:, np.asarray(indices) + 23], full[:, 46:]])
        unique.append(frames)
    assert all(a.shape != b.shape or not np.array_equal(a, b)
               for i, a in enumerate(unique) for b in unique[i+1:])


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('dim', [49, 51, 57, 59, 58.0, True])
def test_cfg_rejects_unsupported_amp_dimensions(task, dim):
    with pytest.raises(ValueError, match='AMP_DIM must be 50 or 58'):
        _task_cfg(task, dim)


def _train_cfg(task, dim, critic_frames=None, vp_critic_frames=5):
    if critic_frames is None:
        critic_frames = 5 if task == 'g1_renet' else 10
    _, amp_cfg = _task_cfg(task, dim)
    policy = {'class_name': 'ActorCritic', 'actor_hidden_dims': [32], 'critic_hidden_dims': [32]}
    if task == 'g1_renet':
        policy.update(class_name='RENetActorCritic', CnnMlp=_depth_cnn_cfg(), LidarCnnMlp=_lidar_cnn_cfg(),
                      single_proprio_dim=78, estimator_mask_dim=1, vp_proprio_history_frames=5,
                      single_critic_dim=85, critic_history_frames=critic_frames, vp_critic_history_frames=vp_critic_frames,
                      proprio_embed_dims=[32], lp_encoder_dims=[32], vp_encoder_dims=[32],
                      lp_encoder_type='ame_point_cross_attention')
    algorithm = {'class_name': 'RENetAMPPPO' if task == 'g1_renet' else 'AMPPPO',
                 'num_learning_epochs': 1, 'num_mini_batches': 2, 'amp_replay_buffer_size': 64,
                 'terrain_recon_target_normalization': 'fixed',
                 'single_critic_dim': 85, 'critic_history_len': critic_frames}
    return {**amp_cfg, 'policy': policy, 'algorithm': algorithm,
            'amp_num_preload_transitions': 64, 'amp_reward_coef': .3, 'amp_task_reward_lerp': .7,
            'amp_discr_hidden_dims': [16, 8], 'min_normalized_std': [.05] * 23,
            'num_steps_per_env': 2, 'save_interval': 100, 'empirical_normalization': False}


def _update_runner(runner, env):
    with torch.no_grad():
        for step in range(2):
            obs, extras = env.get_observations()
            obs[:, -1] = float(step)
            amp = env.get_amp_obs_for_expert_trans()
            runner.alg.act(obs, extras['observations']['critic'], amp)
            env.robot.data.joint_pos += .005
            runner.alg.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {}, env.get_amp_obs_for_expert_trans())
        runner.alg.compute_returns(torch.zeros_like(extras['observations']['critic']), last_actor_obs=obs)
    losses = runner.alg.update()
    assert all(np.isfinite(v) for v in losses.values())


def _assert_optimizer_equal(actual, expected):
    assert actual['param_groups'] == expected['param_groups']
    assert actual['state'].keys() == expected['state'].keys()
    for key, states in expected['state'].items():
        assert actual['state'][key].keys() == states.keys()
        for name, value in states.items():
            torch.testing.assert_close(actual['state'][key][name], value, atol=0, rtol=0)


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('dim', [50, 58])
def test_both_runners_load_sample_update_and_restore_both_layouts(task, dim, tmp_path):
    env = _env(task, dim)
    runner_class = RENetAmpOnPolicyRunner if task == 'g1_renet' else AmpOnPolicyRunner
    runner = runner_class(env, _train_cfg(task, dim), device='cpu')
    assert runner.alg.amp_data.observation_dim == dim
    assert runner.alg.discriminator.input_dim == 2 * dim
    before = {k: v.clone() for k, v in runner.alg.discriminator.state_dict().items()}
    _update_runner(runner, env)
    assert any(not torch.equal(value, before[k]) for k, value in runner.alg.discriminator.state_dict().items())
    assert runner.alg.amp_normalizer.mean.shape == (dim,)
    checkpoint = tmp_path / f'{task}_{dim}.pt'
    runner.logger_type = 'tensorboard'
    runner.save(str(checkpoint))
    probe, _ = env.get_observations()
    with torch.no_grad():
        expected = runner.alg.policy.act_inference(probe)
    restored = runner_class(env, _train_cfg(task, dim), device='cpu')
    restored.load(str(checkpoint))
    with torch.no_grad():
        torch.testing.assert_close(restored.alg.policy.act_inference(probe), expected, atol=0, rtol=0)
    for k, value in runner.alg.discriminator.state_dict().items():
        torch.testing.assert_close(restored.alg.discriminator.state_dict()[k], value, atol=0, rtol=0)
    for attr in ('mean', 'var', 'count'):
        np.testing.assert_array_equal(getattr(restored.alg.amp_normalizer, attr),
                                      getattr(runner.alg.amp_normalizer, attr))
    _assert_optimizer_equal(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict())


@pytest.mark.parametrize('task', TASKS)
def test_environment_agent_mismatch_is_rejected_before_preloading(task):
    runner_class = RENetAmpOnPolicyRunner if task == 'g1_renet' else AmpOnPolicyRunner
    with pytest.raises(ValueError, match='environment/cfg mismatch'):
        runner_class(_env(task, 50), _train_cfg(task, 58), device='cpu')


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('field,value', [('amp_observation_dim', 50), ('amp_joint_count', 19),
                                       ('amp_data_class_name', 'AMPLoader')])
def test_inconsistent_agent_cfg_is_rejected_before_preloading(task, field, value):
    cfg = _train_cfg(task, 58)
    cfg[field] = value
    runner_class = RENetAmpOnPolicyRunner if task == 'g1_renet' else AmpOnPolicyRunner
    with pytest.raises(ValueError, match='AMP cfg'):
        runner_class(_env(task, 58), cfg, device='cpu')


@pytest.mark.parametrize('old_dim,new_dim', [(50, 58), (58, 50)])
def test_ordinary_runner_rejects_other_layout_before_mutating_policy(old_dim, new_dim, tmp_path):
    old = AmpOnPolicyRunner(_env('g1_rough', old_dim), _train_cfg('g1_rough', old_dim), device='cpu')
    old.logger_type = 'tensorboard'
    checkpoint = tmp_path / 'old.pt'
    old.save(str(checkpoint))
    new = AmpOnPolicyRunner(_env('g1_rough', new_dim), _train_cfg('g1_rough', new_dim), device='cpu')
    before = {k: v.clone() for k, v in new.alg.policy.state_dict().items()}
    with pytest.raises(ValueError, match='No policy weights have been loaded'):
        new.load(str(checkpoint))
    for name, value in before.items():
        torch.testing.assert_close(new.alg.policy.state_dict()[name], value, atol=0, rtol=0)
    assert not new.alg.optimizer.state_dict()['state']


@pytest.mark.parametrize('old_dim,new_dim', [(50, 58), (58, 50)])
def test_renet_runner_loads_other_layout_preserving_policy_and_its_adam(old_dim, new_dim, tmp_path):
    env = _env('g1_renet', old_dim)
    old = RENetAmpOnPolicyRunner(env, _train_cfg('g1_renet', old_dim), device='cpu')
    _update_runner(old, env)
    old.logger_type = 'tensorboard'
    checkpoint = tmp_path / 'old.pt'
    old.save(str(checkpoint))
    new = RENetAmpOnPolicyRunner(_env('g1_renet', new_dim), _train_cfg('g1_renet', new_dim), device='cpu')
    before = {k: v.clone() for k, v in new.alg.discriminator.state_dict().items()}
    new.load(str(checkpoint))
    for name, value in old.alg.policy.state_dict().items():
        torch.testing.assert_close(new.alg.policy.state_dict()[name], value, atol=0, rtol=0)
    old_params = dict(old.alg.policy.named_parameters())
    for name, parameter in new.alg.policy.named_parameters():
        for key, value in old.alg.optimizer.state.get(old_params[name], {}).items():
            torch.testing.assert_close(new.alg.optimizer.state[parameter][key], value, atol=0, rtol=0)
    for name, value in before.items():
        torch.testing.assert_close(new.alg.discriminator.state_dict()[name], value, atol=0, rtol=0)
    assert all(not new.alg.optimizer.state.get(p) for p in new.alg.discriminator.parameters())
    np.testing.assert_array_equal(new.alg.amp_normalizer.mean, np.zeros(new_dim))
    assert new.amp_observation_contract['dimension'] == new_dim


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('dim', [50, 58])
def test_playback_export_can_be_loaded_by_the_selected_amp_loader(task, dim, tmp_path):
    path = ROOT / 'legged_lab/scripts/play_amp_animation.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_motion_file')
    scope = {'Path': Path, 'np': np, 'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    env = _env(task, dim)
    frames = env.get_amp_obs_for_expert_trans().numpy()
    output = tmp_path / 'export.txt'
    scope['_write_motion_file'](str(output), frames, .02, 1., 0., .02, env.amp_joint_names)
    loader_class = AMPLoader if dim == 50 else G1AnkleAMPLoader
    loader = loader_class('cpu', .02, motion_files=[str(output)])
    np.testing.assert_array_equal(loader.all_trajectories_full.numpy(), frames)
    payload = json.loads(output.read_text())
    assert payload['JointCount'] == (dim - 12) // 2
    assert payload['AMPJointNames'] == env.amp_joint_names
