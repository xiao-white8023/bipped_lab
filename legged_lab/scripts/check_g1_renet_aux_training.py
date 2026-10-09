from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'rsl_rl'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description='Short IsaacLab integration check for RENet AUX training; not a performance benchmark.')
parser.add_argument('--report', type=Path, default=ROOT / 'work/g1_renet_aux_20261003/isaac_validation.json')
parser.add_argument('--vp_frames', type=int, choices=range(1, 11), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from legged_lab.envs import *
from legged_lab.utils import task_registry
from rsl_rl.runners import RENetAmpOnPolicyRunner


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs('g1_renet')
    if args.vp_frames is not None:
        agent_cfg.policy.vp_proprio_history_frames = args.vp_frames
    env_cfg.scene.num_envs = 2
    env_cfg.scene.terrain_generator.num_rows = 2
    env_cfg.scene.terrain_generator.num_cols = 3
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.commands.debug_vis = False
    default_action_delay_enabled = env_cfg.domain_rand.action_delay.enable
    assert env_cfg.renet.lp_height_source == 'pgtt_full_atten' and env_cfg.scene.lidar is None
    env = task_registry.get_task_class('g1_renet')(env_cfg, headless=True)
    assert 'lidar' not in env.scene.sensors
    assert env.pgtt_height_scanner.num_rays == 187
    assert not hasattr(env, 'lidar_elevation_map') and not hasattr(env, 'lidar_noise_model')
    initial_action_delays = env.action_buffer.time_lags.clone()
    assert ((initial_action_delays >= env.action_delay_min_steps) &
            (initial_action_delays <= env.action_delay_max_steps)).all()
    if not default_action_delay_enabled:
        assert not initial_action_delays.any()
    assert not env._pgtt_noise_is_active() and env._domain_rand_is_active()
    assert env._depth_noise_is_active()
    assert env_cfg.renet.domain_rand_start_iteration == env_cfg.depth_noise_start_iteration == 0
    assert env_cfg.lp_height.noise.start_iteration == 30000
    assert env_cfg.domain_rand.events.physics_material.mode == env_cfg.domain_rand.events.add_base_mass.mode == 'startup'
    def clean_depth_reference():
        depth = env.camera.data.output['distance_to_image_plane'].squeeze(-1)
        crop = env_cfg.robot.depth_crop
        pattern = env_cfg.scene.camera.camera.pattern_cfg
        depth = depth[:, crop[0]:pattern.height-crop[1], crop[2]:pattern.width-crop[3]]
        depth = torch.where(torch.isfinite(depth), depth, env_cfg.robot.depth_max)
        return depth.clamp(env_cfg.scene.camera.camera.min_distance,
                           env_cfg.robot.depth_max) / env_cfg.robot.depth_max
    startup_depth = env.get_processed_deepcamera()
    assert torch.isfinite(startup_depth).all()
    assert (startup_depth-clean_depth_reference()).abs().mean().item() > 1e-6
    initial_mass = env.robot.root_physx_view.get_masses().clone()
    initial_material = env.robot.root_physx_view.get_material_properties().clone()
    assert torch.isfinite(initial_mass).all() and (initial_mass > 0).all()
    assert not torch.allclose(initial_mass, env.robot.data.default_mass.cpu())
    event_calls = []
    native_event_apply = env.event_manager.apply
    def capture_event_apply(*event_args, **event_kwargs):
        event_calls.append(event_kwargs['mode'])
        return native_event_apply(*event_args, **event_kwargs)
    env.event_manager.apply = capture_event_apply
    push_calls = []
    push_cfg = env.event_manager.get_term_cfg('push_robot')
    native_push = push_cfg.func
    def capture_push(push_env, env_ids, **push_params):
        push_calls.append(env_ids.tolist())
        return native_push(push_env, env_ids, **push_params)
    push_cfg.func = capture_push
    assert not hasattr(env_cfg, 'gait')
    assert not hasattr(env, '_calculate_gait_para')
    assert not any(hasattr(env, name) for name in (
        'phase', 'phase_left', 'phase_right', 'leg_phase',
        'avg_feet_force_per_step', 'avg_feet_speed_per_step'))
    air_time_term = getattr(env_cfg.reward, 'feet_air_time', None)
    air_time_enabled = air_time_term is not None and air_time_term.weight != 0.
    assert env.contact_sensor.cfg.track_air_time == air_time_enabled
    assert ('feet_air_time' in env.reward_manager.active_terms) == air_time_enabled
    if air_time_enabled:
        assert env.contact_sensor.data.current_air_time is not None
        assert env.contact_sensor.data.current_contact_time is not None
    contact_term = env.reward_manager.get_term_cfg('undesired_contacts')
    contact_cfg = contact_term.params['sensor_cfg']
    contact_names = [env.contact_sensor.body_names[i] for i in contact_cfg.body_ids]
    edge_terms = {}
    for side, name in (('L', 'left'), ('R', 'right')):
        term_name = f'stair_edge_penalty_{side}'
        assert term_name in env.reward_manager.active_terms
        term = env.reward_manager.get_term_cfg(term_name)
        selected = term.params['contact_sensor_cfg'].body_ids
        assert len(selected) == 1
        assert env.contact_sensor.body_names[selected[0]] == f'{name}_ankle_roll_link'
        assert term.params['ray_sensor_cfg'].name == f'{name}_feet_ray_caster'
        assert term.weight < 0 and term.params['threshold'] == .04
        rays = env.scene.sensors[term.params['ray_sensor_cfg'].name]
        assert rays.data.ray_hits_w.shape[1] > 100
        pattern = rays.cfg.pattern_cfg
        assert pattern.resolution == .01 and pattern.edge_margin == .002
        expected_starts, expected_directions = pattern.func(pattern, env.device)
        expected_starts += torch.tensor(rays.cfg.offset.pos, device=env.device)
        torch.testing.assert_close(rays.ray_starts[0], expected_starts)
        torch.testing.assert_close(rays.ray_directions[0], expected_directions)
        assert torch.isfinite(term.func(env, **term.params)).all()
        edge_terms[term_name] = {'weight': term.weight, 'height_threshold_m': term.params['threshold'],
                                'body_name': f'{name}_ankle_roll_link',
                                'footprint_ray_count': rays.data.ray_hits_w.shape[1],
                                'grid_resolution_m': pattern.resolution,
                                'sole_inset_m': pattern.edge_margin,
                                'local_xy_min_m': rays.ray_starts[0, :, :2].amin(0).tolist(),
                                'local_xy_max_m': rays.ray_starts[0, :, :2].amax(0).tolist()}
    assert {'left_knee_link', 'right_knee_link'} <= set(contact_names)
    assert set(contact_names) == {
        name for name in env.contact_sensor.body_names if re.fullmatch('(?!.*ankle.*).*', name)}
    # Probe the real reward with resolved simulator body indices, without injecting
    # artificial forces into the live physics scene.
    forces = torch.zeros(3, 3, len(env.contact_sensor.body_names), 3, device=env.device)
    for row, name in enumerate(('left_knee_link', 'right_knee_link', 'left_ankle_roll_link')):
        forces[row, 0, env.contact_sensor.body_names.index(name), 2] = 2.0
    probe_env = SimpleNamespace(scene=SimpleNamespace(sensors={
        'contact_sensor': SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))}))
    torch.testing.assert_close(contact_term.func(probe_env, **contact_term.params),
                               torch.tensor([1, 1, 0], device=env.device))
    train_cfg = agent_cfg.to_dict()
    train_cfg['num_steps_per_env'] = 4
    train_cfg['amp_num_preload_transitions'] = 64
    train_cfg['algorithm']['num_learning_epochs'] = 1
    train_cfg['algorithm']['num_mini_batches'] = 2
    scratch = tempfile.TemporaryDirectory(prefix='g1_renet_aux_smoke_')
    smoke_log_dir = scratch.name
    runner = RENetAmpOnPolicyRunner(env, train_cfg, log_dir=smoke_log_dir, device=env_cfg.device)
    runner.git_status_repos = []
    runner.logger_type = 'tensorboard'
    assert runner.alg.terrain_recon_target_normalization == 'fixed'
    assert runner.alg.use_terrain_height_valid_mask
    assert runner.amp_observation_contract['gradient_penalty_space'] == 'normalized_discriminator_inputs'
    assert not runner.alg.policy.use_lp_feet_height_prediction
    configured_vp_frames = agent_cfg.policy.vp_proprio_history_frames
    assert runner.alg.policy.vp_proprio_history_frames == configured_vp_frames
    assert runner.vp_encoder_contract['proprio_history_frames'] == configured_vp_frames
    assert runner.alg.policy.vp_critic_history_frames == 10
    assert runner.alg.policy.critic_history_frames == 10
    assert env.critic_obs_buffer.max_length == 10
    assert runner.alg.critic_history_len == 10
    assert runner.alg.vel_obs_start_idx == 9 * 85 + 78
    assert runner.alg.feet_height_obs_start_idx == 9 * 85 + 83
    assert runner.critic_observation_contract['vp_history_frames'] == 10
    vae_transition_checks = []
    is_multi_head = runner.alg.policy.use_vp_vae
    if is_multi_head:
        assert runner.alg.policy.vp_feature_dim == 69
        assert runner.alg.policy.actor[0].in_features == 211
        assert runner.alg.use_vp_vae
        real_process = runner.alg.process_env_step
        def capture_process(rewards, dones, infos, amp_obs, next_proprio=None):
            raw_target = infos['observations']['critic'][:, 9 * 85:9 * 85 + 78]
            torch.testing.assert_close(next_proprio, raw_target, rtol=0, atol=0)
            real_process(rewards, dones, infos, amp_obs, next_proprio=next_proprio)
            step = runner.alg.storage.step - 1
            expected_valid = (~dones.bool()).reshape(-1, 1) & torch.isfinite(raw_target).all(dim=1, keepdim=True)
            torch.testing.assert_close(runner.alg.storage.next_proprio_valid[step], expected_valid)
            torch.testing.assert_close(runner.alg.storage.next_proprio[step], torch.where(expected_valid, raw_target, 0.))
            vae_transition_checks.append({'valid': int(expected_valid.sum()), 'dones': int(dones.sum())})
        runner.alg.process_env_step = capture_process
    vp_gru_lengths = []
    vp_gru_hook = runner.alg.policy.vp_gru.register_forward_pre_hook(
        lambda module, inputs: vp_gru_lengths.append(inputs[0].shape[1]))
    amp_dim = env_cfg.amp_observation_dim
    assert agent_cfg.amp_observation_dim == amp_dim
    assert runner.alg.amp_data.observation_dim == amp_dim
    assert runner.alg.discriminator.input_dim == 2 * amp_dim
    assert env.get_amp_obs_for_expert_trans().shape == (2, amp_dim)
    assert 'vp_idle_when_commanded' in env.reward_manager.active_terms
    contact_updates = []
    real_contact_update = env.contact_sensor._update_buffers_impl
    def capture_contact_update(*update_args, **update_kwargs):
        contact_updates.append(1)
        return real_contact_update(*update_args, **update_kwargs)
    env.contact_sensor._update_buffers_impl = capture_contact_update
    reports = {}
    real_update = runner.alg.update
    def capture_update():
        losses = real_update()
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
        reports[env.cfg.renet.mask_mode] = losses
        return losses
    runner.alg.update = capture_update
    stage_reports = {}
    for iteration, mode in ((29999, 'lp'), (29999, 'vp'), (30000, 'lp'), (30000, 'vp')):
        env.cfg.renet.mask_mode = mode
        runner.current_learning_iteration = iteration
        runner.alg.current_iteration = iteration
        if iteration == 30000 and mode == 'lp':
            # Exercise the actual runner checkpoint load before the first rollout
            # at the LP noise threshold; VP and physical randomization are already active.
            resume_checkpoint = str(Path(smoke_log_dir) / 'resume_boundary.pt')
            runner.save(resume_checkpoint)
            runner.current_learning_iteration = 0
            runner.alg.current_iteration = 0
            runner.load(resume_checkpoint)
            assert runner.current_learning_iteration == runner.alg.current_iteration == 30000
            assert env._domain_rand_stage_active
        if is_multi_head and mode == 'vp' and iteration == 29999:
            env.episode_length_buf[0] = env.max_episode_length - 1
        before_vae = ([p.detach().clone() for p in runner.alg.policy.vp_next_proprio_decoder.parameters()]
                      if is_multi_head else [])
        runner.learn(1)
        if is_multi_head:
            assert reports[mode]['vp_vae/reconstruction'] > 0
            assert reports[mode]['vp_vae/kl'] >= 0
            assert any(not torch.equal(before, after) for before, after in
                       zip(before_vae, runner.alg.policy.vp_next_proprio_decoder.parameters(), strict=True))
        obs, extras = env.get_observations()
        lp_active = iteration >= env_cfg.lp_height.noise.start_iteration
        assert env._pgtt_noise_is_active() == lp_active
        assert env._domain_rand_is_active() and env._depth_noise_is_active()
        assert extras['log']['LP/noise_enabled'] == float(lp_active)
        assert extras['log']['VP/noise_enabled'] == 1.
        assert extras['log']['DomainRand/enabled'] == 1.
        processed_depth = env.get_processed_deepcamera()
        clean_depth = clean_depth_reference()
        assert torch.isfinite(processed_depth).all()
        depth_difference = (processed_depth - clean_depth).abs().mean().item()
        assert depth_difference > 1e-6
        actual_mass = env.robot.root_physx_view.get_masses().clone()
        actual_material = env.robot.root_physx_view.get_material_properties().clone()
        assert event_calls.count('randomization_start') == 0
        assert 'interval' in event_calls
        assert ((env.action_buffer.time_lags >= env.action_delay_min_steps) &
                (env.action_buffer.time_lags <= env.action_delay_max_steps)).all()
        if not default_action_delay_enabled:
            assert not env.action_buffer.time_lags.any()
        assert torch.isfinite(actual_mass).all() and (actual_mass > 0).all()
        assert ((actual_material[..., 0] >= .6) & (actual_material[..., 0] <= 1.)).all()
        assert ((actual_material[..., 1] >= .4) & (actual_material[..., 1] <= .8)).all()
        assert ((actual_material[..., 2] >= 0.) & (actual_material[..., 2] <= .005)).all()
        torch.testing.assert_close(actual_mass, initial_mass)
        torch.testing.assert_close(actual_material, initial_material)
        stage_reports[f'{mode}_{iteration}'] = {
            'lp_noise_enabled': lp_active, 'vp_noise_enabled': True,
            'depth_difference_from_clean_mean': depth_difference,
            'domain_randomization_enabled': True,
            'action_delay_steps': env.action_buffer.time_lags.tolist(),
            'deferred_startup_call_count': event_calls.count('randomization_start'),
            'optimizer_losses': dict(reports[mode]),
        }
        assert obs.shape == (2, 2120)
        assert extras['observations']['critic'].shape == (2, 1037)
        assert extras['observations']['feet_height_valid'].shape == (2, 2)
        assert extras['observations']['feet_height_valid'].any()
        assert extras['observations']['terrain_height_valid'].shape == (2, 187)
        assert extras['observations']['terrain_height_valid'].dtype == torch.bool
        assert torch.isfinite(extras['observations']['critic']).all()
        assert reports[mode]['terrain_height/valid_fraction'] > 0
        assert reports[mode].get('feet_height/lp_coef_current', 0.) == 0.
        assert reports[mode].get('feet_height/lp', 0.) == 0.
        assert reports[mode]['feet_height/vp'] > 0
        reward_cost = env.reward_manager.get_term_cfg('vp_idle_when_commanded').func(env)
        if air_time_enabled:
            air_time_reward = air_time_term.func(env, **env.reward_manager.get_term_cfg('feet_air_time').params)
            assert torch.isfinite(air_time_reward).all()
        for term_name in edge_terms:
            term = env.reward_manager.get_term_cfg(term_name)
            penalty = term.func(env, **term.params)
            assert penalty.shape == (2,) and torch.isfinite(penalty).all()
            assert ((penalty >= 0.) & (penalty <= 1.)).all()
        if mode == 'lp':
            assert torch.count_nonzero(reward_cost) == 0
    assert not push_calls  # Less than 10 s have elapsed since activation.
    assert event_calls.count('randomization_start') == 0
    training_contact_update_count = len(contact_updates)
    vp_gru_hook.remove()
    assert vp_gru_lengths and set(vp_gru_lengths) == {configured_vp_frames}
    # Match the rollout context: IsaacLab pose caches are inference tensors.
    with torch.inference_mode():
        # Advance the real event manager's interval clock without a long rollout.
        env.event_manager.apply(mode='interval', dt=16.)
        assert len(push_calls) == 1 and set(push_calls[0]) == set(range(env.num_envs))
        assert torch.isfinite(env.robot.data.root_vel_w).all()
        # Exercise real observation assembly with NaN/missed rays. Restore the
        # cached sensor hits before any subsequent simulation step.
        hits = env.height_scanner.data.ray_hits_w
        saved_hits = hits.clone()
        try:
            hits[0].fill_(float('inf'))
            hits[1, 0].fill_(float('nan'))
            _, invalid_extras = env.get_observations()
            assert torch.isfinite(invalid_extras['observations']['critic']).all()
            terrain_valid = invalid_extras['observations']['terrain_height_valid']
            assert not terrain_valid[0].any() and not terrain_valid[1, 0]
            assert not invalid_extras['observations']['critic'][0, -187:].any()
        finally:
            hits.copy_(saved_hits)
            env.get_observations()
        # Verify actual physics targets use queued actions, then check reset and
        # disable semantics. These are separate checks after the eight-step update.
        ids = torch.arange(env.num_envs, device=env.device)
        env.cfg.domain_rand.action_delay.enable = True
        env.action_delay_min_steps = env.action_delay_max_steps = 2
        env.reset(ids)
        assert (env.action_buffer.time_lags == 2).all()
        native_set_target = env.robot.set_joint_position_target
        recorded_targets = []
        def capture_target(target, *target_args, **target_kwargs):
            # Reset events also set subsets of joints; audit the full physics commands.
            if target.shape == env.robot.data.default_joint_pos.shape:
                recorded_targets.append(target.clone())
            return native_set_target(target, *target_args, **target_kwargs)
        env.robot.set_joint_position_target = capture_target
        # The untrained policy can leave a falling robot. Isolate queue timing
        # from automatic resets; native reset behavior is tested separately below.
        native_check_reset = env.check_reset
        env.check_reset = lambda: (torch.zeros_like(env.reset_buf), torch.zeros_like(env.time_out_buf))
        try:
            for k in range(4):
                env.step(torch.full((env.num_envs, env.num_actions), .1 * k, device=env.device))
        finally:
            env.robot.set_joint_position_target = native_set_target
            env.check_reset = native_check_reset
        observed_offsets = []
        for k, expected in enumerate((0., 0., 0., .1)):
            targets = recorded_targets[k * env.cfg.sim.decimation:(k + 1) * env.cfg.sim.decimation]
            assert len(targets) == env.cfg.sim.decimation
            for target in targets:
                torch.testing.assert_close(target, env.robot.data.default_joint_pos + expected * env.action_scale)
            observed_offsets.append(expected)
        # Native reset clears the history only for the reset environments.
        env.action_buffer.set_time_lag(3, batch_ids=ids[1:])
        env.reset(ids[:1])
        assert env.action_buffer.time_lags.tolist() == [2, 3]
        env.cfg.domain_rand.action_delay.enable = False
        env.reset(ids)
        assert not env.action_buffer.time_lags.any()
        current_action = torch.full((env.num_envs, env.num_actions), .7, device=env.device)
        torch.testing.assert_close(env.action_buffer.compute(current_action), current_action)
        # Restore the task's default range and enable state for the saved report.
        env.cfg.domain_rand.action_delay.enable = default_action_delay_enabled
        env.action_delay_min_steps = env_cfg.domain_rand.action_delay.params['min_delay']
        env.action_delay_max_steps = env_cfg.domain_rand.action_delay.params['max_delay']
        env.reset(ids)
    checkpoint_path = str(Path(smoke_log_dir) / 'checkpoint.pt')
    runner.save(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert saved['vp_encoder_contract'] == runner.vp_encoder_contract
    assert saved['critic_observation_contract'] == runner.critic_observation_contract
    assert saved['auxiliary_training_contract'] == runner.alg.get_auxiliary_training_contract()
    runner.load(checkpoint_path)
    assert training_contact_update_count >= 16 * env.cfg.sim.decimation
    result = {'passed': True, 'device': env_cfg.device, 'envs': 2, 'rollout_steps_per_mode': 4,
              'actor_dim': 2120, 'critic_dim': 1037, 'feet_height_valid': env.feet_height_valid.tolist(),
              'amp_dimension': amp_dim, 'discriminator_input_dimension': 2 * amp_dim,
              'amp_preprocessing_contract': runner.amp_observation_contract,
              'vp_encoder_contract': runner.vp_encoder_contract,
              'critic_observation_contract': runner.critic_observation_contract,
              'vp_gru_observed_sequence_lengths': sorted(set(vp_gru_lengths)),
              'terrain_validity': {'full_dimension': 187, 'invalid_ray_observation_probe_passed': True},
              'modes': reports, 'checkpoint_roundtrip': True,
              'randomization_schedule': {
                  'start_iteration': 0, 'lp_noise_start_iteration': 30000,
                  'vp_noise_start_iteration': 0, 'stage_rollouts': stage_reports,
                  'startup_randomization_enabled': True,
                  'randomized_mass_and_material_stable': True,
                  'mass_change_kg': (initial_mass - env.robot.data.default_mass.cpu()).tolist(),
                  'deferred_startup_call_count': event_calls.count('randomization_start'),
                  'checkpoint_resume_at_boundary': True,
                  'push_probe_advanced_interval_seconds': 16., 'push_probe_env_ids': push_calls[0],
              },
              'terrain': 'generator_2x3', 'interpretation': 'integration only; not a trained-policy performance evaluation'}
    if is_multi_head:
        from rsl_rl.modules import RENetActorCritic
        from legged_lab.scripts.export import export_policy_as_jit
        policy_cfg = dict(train_cfg['policy'])
        policy_cfg.pop('class_name', None)
        inference_policy = RENetActorCritic(2120, 1037, 23, **policy_cfg).eval()
        inference_policy.load_state_dict(saved['model_state_dict'], strict=True)
        export_path = export_policy_as_jit(inference_policy, None, str(args.report.parent / f'exported_vp{configured_vp_frames}'),
                                          metadata={'vp_encoder_contract': runner.vp_encoder_contract})
        exported = torch.jit.load(export_path)
        with torch.inference_mode():
            probe = obs.cpu()
            expected = inference_policy.act_inference(probe)
            actual = exported(probe)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert sum(check['dones'] for check in vae_transition_checks) > 0
        result['vp_beta_vae'] = {'transition_checks': vae_transition_checks,
                                 'feature_dim': 69, 'actor_input_dim': 211, 'next_target_dim': 78,
                                 'auxiliary_contract': runner.alg.get_auxiliary_training_contract(),
                                 'export_max_abs_error': float((actual - expected).abs().max())}
    hit_z = env.height_scanner.data.ray_hits_w[..., 2]
    result['training_audit'] = {
        'gait_config_and_phase_removed': True,
        'feet_air_time_reward_enabled': air_time_enabled,
        'feet_air_time_tracking_enabled': env.contact_sensor.cfg.track_air_time,
        'contact_force_updates_during_16_step_check': training_contact_update_count,
        'minimum_expected_contact_force_updates': 16 * env.cfg.sim.decimation,
        'undesired_contact_body_names': contact_names,
        'calf_contact_probe_counts': [1, 1, 0],
        'action_delay_config': env_cfg.domain_rand.action_delay.to_dict(),
        'actual_action_delay_steps': env.action_buffer.time_lags.tolist(),
        'action_delay_verification': {
            'initial_default_samples': initial_action_delays.tolist(),
            'fixed_delay_steps_tested': 2,
            'physics_probe_steps': 4,
            'input_action_values': [0., .1, .2, .3],
            'executed_action_values': observed_offsets,
            'partial_reset_preserves_other_environment_delay': True,
            'disable_executes_current_action': True,
        },
        'configured_physx': env_cfg.sim.physx.to_dict(),
        'actual_physx': {name: getattr(env.sim.cfg.physx, name) for name in (
            'gpu_max_rigid_patch_count', 'gpu_max_rigid_contact_count', 'gpu_heap_capacity')},
        'terrain_hit_valid_fraction': torch.isfinite(hit_z).float().mean().item(),
        'active_reward_terms': env.reward_manager.active_terms,
        'edge_penalty_terms': edge_terms,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    runner.writer.flush()
    runner.writer.close()
    scratch.cleanup()
    print('G1_RENET_AUX_SMOKE_PASS', flush=True)


main()
app.close(wait_for_replicator=False, skip_cleanup=True)
