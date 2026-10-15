"""Re-sample a legacy RENet checkpoint and measure LP/VP PPO gradients.

No optimizer step is taken. This is a controlled checkpoint re-evaluation,
not a reconstruction of historical gradients. Raw on-policy batches are saved
for CPU replay. Run with IsaacLab's Python environment and a working GPU.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path,
                        default=ROOT / "logs/g1_renet/2026-09-04_14-11-25/model_60000.pt")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "work/g1_renet_mode_gradients_20261014")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--vp_seed_offset", type=int, default=10000,
                        help="Independent VP action noise; use0 only for common-noise sensitivity tests.")
    parser.add_argument("--reuse_lp_from", type=Path, default=None,
                        help="Reuse matching saved LP batches without resampling or updating weights.")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--speed", type=float, default=.3)
    parser.add_argument("--directions", nargs="+", default=["forward", "backward", "left", "right"],
                        choices=["forward", "backward", "left", "right"])
    parser.add_argument("--analyze_only", action="store_true")
    if "--analyze_only" in sys.argv:
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--headless", action="store_true")
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 24 or args.steps % 24 or args.warmup_steps < 10:
        parser.error("Use >=2 environments, a positive multiple of 24 steps, and >=10 warmup steps.")
    if not 0 < args.speed <= .5 or not args.seeds:
        parser.error("Use a speed in (0,.5] and at least one seed.")
    return args


def gradient_analyzer():
    # Avoid utils/__init__.py's Isaac imports during standalone CPU replay.
    spec = importlib.util.spec_from_file_location(
        "renet_gradient_audit", ROOT / "legged_lab/utils/renet_gradient_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze_paired_ppo_gradients


def historical_settings():
    # Evidence recovered from the old run's Git diff. Never import this config.
    path = ROOT / "work/g1_renet_lp_vp_direction_audit_20261014/archived_20260904_RENet_cfg.py"
    tree = ast.parse(path.read_text())
    reward_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Reward")
    weights = {}
    for node in reward_class.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            weight = next((kw.value for kw in node.value.keywords if kw.arg == "weight"), None)
            if weight is not None:
                weights[node.targets[0].id] = ast.literal_eval(weight)
    agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1RENETAGENTCFG")
    settings = {}
    for node in agent.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("amp_reward_coef", "amp_task_reward_lerp"):
            settings[node.targets[0].id] = ast.literal_eval(node.value)
    return weights, settings


def build_policy(args):
    import torch
    from legged_lab.selector.frozen_policy import build_frozen_policy, sha256, validate_reference_weights, validation_inputs
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    policy = build_frozen_policy().cpu().eval()
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "obs_norm_state_dict" in checkpoint:
        raise ValueError("This audit expects the archived identity observation normalization.")
    reference_path = args.checkpoint_path.parent / "exported/policy.pt"
    parity = None
    if reference_path.exists() and checkpoint["iter"] == 60000:
        reference = torch.jit.load(str(reference_path), map_location="cpu").eval()
        validate_reference_weights(policy, reference)
        parity = {}
        with torch.no_grad():
            for mode, mask in (("lp", 1.), ("vp", 0.)):
                errors = []
                for probe in validation_inputs():
                    probe[:, -1] = mask
                    a, b = policy.act_inference(probe), reference(probe)
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
                    errors.append(float((a-b).abs().max()))
                parity[mode] = max(errors)
    # Only actor/std gradients are needed; feature extraction stays unchanged.
    policy.requires_grad_(False)
    identity = {"path": str(args.checkpoint_path.resolve()), "sha256": sha256(args.checkpoint_path),
                "iteration": checkpoint["iter"], "reference_action_max_abs_error": parity}
    return policy, checkpoint, identity


def summarize(records):
    import statistics
    summary = {}
    for direction in sorted({r["direction"] for r in records}):
        selected = [r for r in records if r["direction"] == direction]
        values = [r["gradients"]["surrogate_only"]["shared_actor"]["cosine"] for r in selected]
        values = [v for v in values if v is not None]
        summary[direction] = {"pairs": len(selected), "valid_cosines": len(values),
            "mean_shared_actor_cosine": statistics.mean(values) if values else None,
            "min_shared_actor_cosine": min(values) if values else None,
            "max_shared_actor_cosine": max(values) if values else None,
            "negative_pairs": sum(v < 0 for v in values),
            "negative_fraction": sum(v < 0 for v in values)/len(values) if values else None,
            "performance": {mode: {key: statistics.mean(r["performance"][mode][key] for r in selected)
                                   for key in selected[0]["performance"][mode]}
                            for mode in ("lp", "vp")}}
    return summary


def analyze_saved(args):
    import torch
    torch.set_num_threads(2)
    analyze_paired_ppo_gradients = gradient_analyzer()
    policy, _, identity = build_policy(args)
    records = []
    for path in sorted((args.output_dir / "rollouts").glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["checkpoint_sha256"] != identity["sha256"]:
            raise ValueError(f"Rollout/checkpoint mismatch: {path}")
        result = analyze_paired_ppo_gradients(policy, data["lp"], data["vp"], normalize_advantages=False)
        records.append({"seed": data["seed"], "direction": data["direction"],
                        "performance": data["performance"], "gradients": result})
    if not records:
        raise FileNotFoundError("No saved paired rollouts found.")
    result = {"checkpoint": identity, "records": records, "summary": summarize(records)}
    (args.output_dir / "cpu_reanalysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


def configure_environment(args, historical_weights):
    from legged_lab.utils import task_registry
    from legged_lab.terrains import Flat_terrain
    cfg, _ = task_registry.get_cfgs("g1_renet")
    cfg = copy.deepcopy(cfg)
    cfg.device = args.device
    cfg.amp_observation_dim = 50
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain_generator = copy.deepcopy(Flat_terrain)
    cfg.scene.terrain_generator.num_rows = 2
    cfg.scene.terrain_generator.num_cols = 4
    cfg.scene.terrain_generator.seed = args.seeds[0]
    cfg.scene.max_init_terrain_level = 0
    cfg.scene.height_scanner.debug_vis = False
    cfg.scene.camera.camera.debug_vis = False
    cfg.scene.camera.add_camera_noise = False
    cfg.depth_noise_force_active = False
    cfg.noise.add_noise = False
    cfg.lp_height.noise.enable = False
    cfg.lp_height.noise.force_active = False
    cfg.domain_rand.action_delay.enable = False
    events = cfg.domain_rand.events
    for name in list(vars(events)):
        term = getattr(events, name)
        if name.startswith("_"):
            continue
        if name not in ("reset_base", "reset_robot_joints") and getattr(term, "mode", None) is not None:
            setattr(events, name, None)
    events.reset_base.params["pose_range"] = {"x": (0., 0.), "y": (0., 0.), "yaw": (0., 0.)}
    events.reset_base.params["velocity_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    events.reset_robot_joints.params["position_range"] = (1., 1.)
    events.reset_robot_joints.params["velocity_range"] = (0., 0.)
    cfg.commands.heading_command = False
    cfg.commands.rel_standing_envs = 0.
    cfg.commands.rel_heading_envs = 0.
    cfg.commands.debug_vis = False
    cfg.commands.resampling_time_range = (1e6, 1e6)
    for name in list(vars(cfg.reward)):
        term = getattr(cfg.reward, name)
        if getattr(term, "weight", None) is not None:
            if name in historical_weights:
                term.weight = historical_weights[name]
            else:
                setattr(cfg.reward, name, None)
    return cfg


def collect(env, policy, discriminator, normalizer, args, mode, command, seed):
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.cfg.renet.mask_mode = mode
    ranges = env.command_generator.cfg.ranges
    ranges.lin_vel_x = (command[0], command[0])
    ranges.lin_vel_y = (command[1], command[1])
    ranges.ang_vel_z = (0., 0.)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset(ids)
    env.command_generator.compute(0.)
    env.set_training_iteration(60000)
    obs, info = env.get_observations()
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            obs, _, _, info = env.step(policy.act_inference(obs))
        rows = {k: [] for k in ("observations", "actions", "old_log_probs", "values", "next_values", "rewards", "dones")}
        actual_velocity, task_rewards, style_rewards, falls = [], [], [], []
        amp = env.get_amp_obs_for_expert_trans().clone()
        for _ in range(args.steps):
            critic = info["observations"]["critic"]
            value = policy.evaluate(critic).squeeze(-1)
            actions = policy.act(obs)
            rows["observations"].append(obs.clone())
            rows["actions"].append(actions.clone())
            rows["old_log_probs"].append(policy.get_actions_log_prob(actions).clone())
            rows["values"].append(value.clone())
            obs, task_reward, done, info = env.step(actions)
            next_amp = env.get_amp_obs_for_expert_trans().clone()
            true_next = next_amp.clone()
            if env.reset_env_ids.numel():
                true_next[env.reset_env_ids] = info["terminal_amp_states"]
            mixed, _ = discriminator.predict_amp_reward(amp, true_next, task_reward, normalizer)
            # Match the runner's timeout bootstrap. Normal trials are shorter than20s.
            mixed = mixed + .99*value*info["time_outs"].float()
            rows["rewards"].append(mixed.clone())
            rows["dones"].append(done.clone())
            rows["next_values"].append(policy.evaluate(info["observations"]["critic"]).squeeze(-1).clone())
            amp = next_amp
            actual_velocity.append(quat_apply_inverse(yaw_quat(env.robot.data.root_quat_w),
                                                     env.robot.data.root_lin_vel_w)[:, :2].clone())
            task_rewards.append(task_reward.clone())
            style_rewards.append((mixed - .6*task_reward).clone())
            falls.append((done.bool() & ~info["time_outs"].bool()).clone())
        data = {k: torch.stack(v) for k, v in rows.items()}
        raw_adv = torch.empty_like(data["rewards"])
        normalized_adv = torch.empty_like(raw_adv)
        # Original rollout length24, independent GAE/normalization for each block.
        for begin in range(0, args.steps, 24):
            gae = torch.zeros(env.num_envs, device=env.device)
            for t in reversed(range(begin, begin+24)):
                alive = (~data["dones"][t].bool()).float()
                delta = data["rewards"][t] + .99*data["next_values"][t]*alive - data["values"][t]
                gae = delta + .99*.95*alive*gae
                raw_adv[t] = gae
            block = raw_adv[begin:begin+24]
            normalized_adv[begin:begin+24] = (block-block.mean())/(block.std()+1e-8)
        velocity = torch.stack(actual_velocity)
        requested = velocity.new_tensor(command[:2])
        batch = {k: data[k].flatten(0, 1).cpu() for k in ("observations", "actions", "old_log_probs")}
        batch.update(advantages=normalized_adv.flatten().cpu(), raw_advantages=raw_adv.flatten().cpu(),
                     gae_inputs={k: data[k].cpu() for k in ("values", "next_values", "rewards", "dones")})
        # Exclude reset samples from speed measurement: their velocities are reset.
        valid = ~data["dones"].bool()
        performance = {"mean_vx": float(velocity[..., 0][valid].mean()),
            "mean_vy": float(velocity[..., 1][valid].mean()),
            "xy_tracking_mse": float(((velocity-requested)**2).sum(-1)[valid].mean()),
            "fall_count": int(torch.stack(falls).sum()),
            "mean_task_reward": float(torch.stack(task_rewards).mean()),
            "mean_mixed_style_reward": float(torch.stack(style_rewards).mean())}
    return batch, performance


def run(args):
    import numpy as np
    import torch
    from legged_lab.envs import G1RENetEnv  # registers tasks after AppLauncher
    analyze_paired_ppo_gradients = gradient_analyzer()
    from legged_lab.selector.frozen_policy import sha256
    from rsl_rl.modules import Discriminator
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seeds[0]); np.random.seed(args.seeds[0]); torch.manual_seed(args.seeds[0])
    policy, checkpoint, identity = build_policy(args)
    weights, amp_settings = historical_settings()
    cfg = configure_environment(args, weights)
    env = G1RENetEnv(cfg, headless=True)
    policy = policy.to(env.device)
    discriminator = Discriminator(100, amp_settings["amp_reward_coef"], [1024, 512, 256],
                                  env.device, amp_settings["amp_task_reward_lerp"]).to(env.device)
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=True)
    normalizer = checkpoint["amp_normalizer"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "rollouts"
    raw_dir.mkdir(exist_ok=True)
    if args.reuse_lp_from is not None:
        previous = json.loads((args.reuse_lp_from / "results.json").read_text())
        expected = {"num_envs": args.num_envs, "steps_per_mode": args.steps,
                    "warmup_steps": args.warmup_steps, "speed": args.speed,
                    "reward_weights": weights, "amp_settings": amp_settings}
        if not previous.get("completed") or not previous.get("weights_unchanged"):
            raise ValueError("Only completed, unchanged-policy LP data can be reused.")
        for key, value in expected.items():
            if previous["conditions"].get(key) != value:
                raise ValueError(f"Reused LP condition mismatch: {key}")
    commands = {"forward": (args.speed, 0., 0.), "backward": (-args.speed, 0., 0.),
                "left": (0., args.speed, 0.), "right": (0., -args.speed, 0.)}
    report = {"scope": "fresh controlled IsaacLab reevaluation; historical gradients unavailable",
        "checkpoint": identity, "conditions": {"terrain": "flat", "noise": False, "randomization": False,
        "initial_pose": "default", "warmup_action": "deterministic mean", "collection_action": "original stochastic policy",
        "num_envs": args.num_envs, "steps_per_mode": args.steps, "gae_rollout_steps": 24,
        "warmup_steps": args.warmup_steps, "seeds": args.seeds, "vp_seed_offset": args.vp_seed_offset,
        "reuse_lp_from": str(args.reuse_lp_from) if args.reuse_lp_from else None, "speed": args.speed,
        "reward_weights": weights, "amp_settings": amp_settings, "gamma": .99, "lambda": .95,
        "policy_updated": False, "environment": "current RENet_env, historical reward weights, clean sensors",
        "source_hashes": {name: sha256(ROOT/name) for name in ["legged_lab/envs/g1/RENet_env.py",
         "legged_lab/mdp/rewards.py", "legged_lab/scripts/check_renet_mode_gradients.py",
         "legged_lab/utils/renet_gradient_audit.py"]}}, "records": []}
    path = args.output_dir / "results.json"
    for direction in args.directions:
        for seed in args.seeds:
            batches, performance = {}, {}
            for mode in ("lp", "vp"):
                if mode == "lp" and args.reuse_lp_from is not None:
                    source = args.reuse_lp_from / "rollouts" / f"{direction}_{seed}.pt"
                    saved = torch.load(source, map_location="cpu", weights_only=True)
                    if saved["checkpoint_sha256"] != identity["sha256"]:
                        raise ValueError(f"Reused LP checkpoint mismatch: {source}")
                    batches[mode], performance[mode] = saved[mode], saved["performance"][mode]
                    continue
                actual_seed = seed + (args.vp_seed_offset if mode == "vp" else 0)
                batches[mode], performance[mode] = collect(env, policy, discriminator, normalizer,
                                                          args, mode, commands[direction], actual_seed)
            torch.save({**batches, "performance": performance, "direction": direction, "seed": seed,
                        "vp_seed": seed+args.vp_seed_offset, "checkpoint_sha256": identity["sha256"]},
                       raw_dir/f"{direction}_{seed}.pt")
            gradients = analyze_paired_ppo_gradients(policy, batches["lp"], batches["vp"], normalize_advantages=False)
            record = {"direction": direction, "seed": seed, "performance": performance, "gradients": gradients}
            report["records"].append(record)
            report["summary"] = summarize(report["records"])
            path.write_text(json.dumps(report, indent=2))
            shared = gradients["surrogate_only"]["shared_actor"]
            print(f"[GRADIENT] {direction} seed={seed} shared_cos={shared['cosine']:.6f} "
                  f"lp_mse={performance['lp']['xy_tracking_mse']:.5f} vp_mse={performance['vp']['xy_tracking_mse']:.5f}", flush=True)
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), checkpoint["model_state_dict"][name], rtol=0, atol=0)
    assert sha256(args.checkpoint_path) == identity["sha256"]
    report.update(completed=True, weights_unchanged=True)
    path.write_text(json.dumps(report, indent=2))
    print("[GRADIENT] COMPLETE " + str(path), flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.analyze_only:
        analyze_saved(args)
    else:
        from isaaclab.app import AppLauncher
        args.enable_cameras = True
        app = AppLauncher(args).app
        failed = True
        try:
            run(args)
            failed = False
        finally:
            app.close(skip_cleanup=True)
