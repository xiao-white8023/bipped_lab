"""Training-contract checks, not a replacement for play/export/sim2sim.

Example: python legged_lab/scripts/check_renet_lp.py --headless --source pgtt_full_atten
         --terrain flat --num_envs 1 --steps 100 --iterations 1
Use separate processes for different scenes (Isaac singleton context).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
from isaaclab.app import AppLauncher


def parse_lp_height_source(value):
    if value != "pgtt_full_atten":
        raise argparse.ArgumentTypeError("g1_renet supports only pgtt_full_atten")
    return value


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=parse_lp_height_source,
                    metavar="{pgtt_full_atten}", default="pgtt_full_atten")
parser.add_argument("--terrain", choices=("flat", "stairs", "gap", "bottomless", "mixed"), default="flat")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--iterations", type=int, default=0)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--no-height-noise", action="store_true")
parser.add_argument("--log-dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.terrains as terrain_gen
from isaaclab.utils.io import dump_yaml
from legged_lab.envs.g1.RENet_cfg import (
    G1RENETENVCFG, G1RENETAGENTCFG, configure_agent_for_lp_source,
)
from legged_lab.envs.g1.RENet_env import G1RENetEnv
from legged_lab.terrains.mesh_gap_terrain import MeshGapWithFloorTerrainCfg
from legged_lab.terrains.ray_caster import RayCaster as TerrainRayCaster
from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner


def main():
    cfg, agent = G1RENETENVCFG(), G1RENETAGENTCFG()
    cfg.scene.num_envs = args.num_envs
    cfg.renet.lp_height_source = args.source
    cfg.lp_height.noise.enable = not args.no_height_noise
    if args.terrain != "mixed":
        # Test overrides only. Normal training commands/rewards/randomization
        # remain unchanged in RENet_cfg.py.
        terrain = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(),
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(step_height_range=(.1, .1), step_width=.3, platform_width=1.),
            "gap": MeshGapWithFloorTerrainCfg(gap_width_range=(.5, .5), gap_depth_range=(.6, .6), platform_width=1.),
            "bottomless": terrain_gen.MeshGapTerrainCfg(gap_width_range=(.5, .5), platform_width=1.),
        }[args.terrain]
        cfg.scene.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            curriculum=False, size=(8., 8.), num_rows=1, num_cols=1,
            border_width=0., use_cache=False, sub_terrains={args.terrain: terrain})
        cfg.scene.max_init_terrain_level = 0
        cfg.domain_rand.events.reset_base.params["pose_range"] = {key: (0., 0.) for key in ("x", "y", "yaw")}
        cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
        cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1., 1.)
    log_dir = args.log_dir or tempfile.mkdtemp(prefix=f"renet_{args.source}_{args.terrain}_")
    configure_agent_for_lp_source(cfg, agent)
    # Same network/optimizer/AMP losses. Smaller preload only for the diagnostic.
    agent.amp_num_preload_transitions = 4096
    env = G1RENetEnv(cfg, args.headless)
    runner = RENetAmpOnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=cfg.device)
    if args.checkpoint:
        runner.load(args.checkpoint, load_optimizer=False)
    dump_yaml(str(Path(log_dir)/"env.yaml"), cfg)
    dump_yaml(str(Path(log_dir)/"agent.yaml"), agent)
    dump_yaml(str(Path(log_dir)/"lp_observation_contract.yaml"), env.lp_observation_contract)
    policy = runner.get_inference_policy()
    lp_cells = env.lp_height_history_frames * math.prod(env.lp_observation_shape)
    expected = 780 + lp_cells + 1152 + 1
    observations, _ = env.get_observations()
    assert observations.shape == (args.num_envs, expected)
    assert cfg.scene.lidar is None and "lidar" not in env.scene.sensors
    assert isinstance(env.pgtt_height_scanner, TerrainRayCaster)
    assert env.pgtt_height_scanner.num_rays == 187
    assert env.lp_observation_shape == (11, 17) and expected == 2120
    for iteration, mask in ((0, 1), (19, 1), (20, 0), (39, 0), (40, 1)):
        env.set_training_iteration(iteration)
        updated = env.apply_renet_estimator_mask(observations.clone())
        assert (updated[:, -1] == mask).all()
        with torch.inference_mode():
            assert torch.isfinite(policy(updated)).all()
    start = time.perf_counter()
    initial_valid = env.extras["observations"]["lidar_valid_ratio"].mean().item()
    # Exercise clocks while the runner-style episode counter is randomized.
    env.episode_length_buf[:] = torch.randint(0, 100, (args.num_envs,), device=env.device)
    phase_commands = ((0., 0., 0.), (.3, 0., 0.), (-.3, 0., 0.), (0., .3, 0.), (0., 0., .5))
    for step in range(args.steps):
        env.set_training_iteration(0 if step < args.steps//2 else 20)
        command = phase_commands[min(4, step*5//max(1, args.steps))]
        env.command_generator.vel_command_b[:] = torch.tensor(command, device=env.device)
        observations, _ = env.get_observations()
        observations = env.apply_renet_estimator_mask(observations)
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, extras = env.step(actions)
        assert torch.isfinite(observations).all() and torch.isfinite(rewards).all()
        assert observations.shape[-1] == expected
        max_delay = cfg.lp_height.noise.max_delay_packets if cfg.lp_height.noise.enable else 0
        assert env.lp_history.height.shape[1] <= env.lp_height_history_frames + max_delay
        confidence = extras["observations"]["lidar_confidence"]
        heights = extras["observations"]["lidar_height_map"]
        assert torch.isfinite(confidence).all() and torch.isfinite(heights).all()
        assert torch.all(heights[confidence == 0] == env.lp_runtime_cfg.invalid_height)
        if step % 25 == 0:
            print("[LP_CHECK]", json.dumps(dict(step=step, source=args.source,
                valid=extras["observations"]["lidar_valid_ratio"].mean().item(),
                ages=extras["observations"]["lp_frame_age"].mean(0).tolist(),
                scan_count=env.lp_sensor.scan_counts[0].item())), flush=True)
    if args.num_envs > 1:
        other_history = env.lp_history.height[1:].clone()
        with torch.inference_mode():
            env.reset(torch.tensor([0], device=env.device))
        torch.testing.assert_close(other_history, env.lp_history.height[1:])
    step_seconds = time.perf_counter()-start
    print(f"[LP_CHECK] {args.steps} control steps passed in {step_seconds:.2f}s", flush=True)
    train_seconds = None
    if args.iterations:
        start = time.perf_counter()
        runner.current_learning_iteration = 0
        runner.learn(args.iterations, init_at_random_ep_len=True)
        train_seconds = time.perf_counter()-start
        print(f"[LP_CHECK] {args.iterations} PPO iterations passed in {train_seconds:.2f}s", flush=True)
        runner.writer.flush()
        runner.writer.close()
    result = dict(status="passed", source=args.source, terrain=args.terrain, num_envs=args.num_envs,
                  steps=args.steps, iterations=args.iterations, actor_dim=expected,
                  step_seconds=step_seconds, train_seconds=train_seconds,
                  initial_valid_ratio=initial_valid,
                  final_valid_ratio=env.extras["observations"]["lidar_valid_ratio"].mean().item(),
                  height_frames=env.lp_height_history_frames, checkpoint=args.checkpoint,
                  log_dir=log_dir)
    (Path(log_dir)/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"[LP_CHECK] PASS source={args.source} terrain={args.terrain} envs={args.num_envs} logs={log_dir}", flush=True)


success = False
try:
    main()
    success = True
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    # Fast successful teardown avoids a Kit stage-cleanup stall on this host.
    # Never use immediate exit on failure: preserve the exception/exit status.
    app.close(wait_for_replicator=False, skip_cleanup=success)
