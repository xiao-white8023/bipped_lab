"""Collect paired LP/VP selector labels in a completely separate Isaac process."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
LOCAL_RSL_RL = REPO_ROOT / "rsl_rl"
if str(LOCAL_RSL_RL) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect G1 RENet LP/VP selector data.")
parser.add_argument("--task", default="g1_renet")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument(
    "--checkpoint-path",
    type=Path,
    default=None,
    help="Frozen checkpoint path; defaults to DEFAULT_CHECKPOINT_PATH in selector/config.py.",
)
parser.add_argument(
    "--reference-policy",
    type=Path,
    default=None,
    help="Deployed policy.pt used for strict weight and LP/VP action validation.",
)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--target-samples", type=int, default=None)
parser.add_argument("--terrain-mode", choices=("main", "ood"), default="main")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fault-seed", type=int, default=None)
parser.add_argument("--behavior-seed", type=int, default=None)
parser.add_argument("--base-dataset", type=Path, default=None)
parser.add_argument("--supplement-split", choices=("train", "validation", "test"), default=None)
parser.add_argument("--fault-profile", choices=("baseline", "mixed", "lp_map", "vp_hard", "dual", "recovery", "directed_lp", "directed_vp", "directed_dual"), default=None,
                    help="Default: baseline for original collection, mixed for supplements.")
parser.add_argument("--fault-source", choices=("any", "lp_only", "vp_only", "dual"), default=None,
                    help="Validate injected source and observable effects; inferred for directed profiles.")
quota_args = parser.add_mutually_exclusive_group()
quota_args.add_argument("--pair-quotas", nargs=4, type=int, default=None,
                        metavar=("BOTH_SAFE", "LP_FAIL", "VP_FAIL", "BOTH_FAIL"))
quota_args.add_argument("--no-outcome-quotas", action="store_true",
                        help="Pilot source banks/stress sets: sample faults without result balancing.")
parser.add_argument("--max-samples-per-event", type=int, default=8)
parser.add_argument("--min-event-spacing-s", type=float, default=.2)
parser.add_argument("--max-no-progress-batches", type=int, default=1000)
parser.add_argument("--warmup-seconds", type=float, default=None,
                    help="Real faulted behavior burn-in; production supplements default to 15 s.")
parser.add_argument("--min-collection-span-seconds", type=float, default=None,
                    help="Release sample slots over at least this real duration; supplement default 30 s.")
parser.add_argument("--label-interval-frames", type=int, default=None,
                    help="Label every N real 10 Hz frames; supplement default 10. Features remain 10 Hz.")
phase_args = parser.add_mutually_exclusive_group()
phase_args.add_argument("--randomize-label-phase", action="store_true", dest="randomize_label_phase", default=None,
                        help="Randomize the 10 Hz label tick inside each interval; supplement default.")
phase_args.add_argument("--fixed-label-phase", action="store_false", dest="randomize_label_phase", default=None)
parser.add_argument("--max-samples-per-labelled-batch", type=int, default=0,
                    help="Optional per-fork admission cap; 0 leaves only the time-release budget.")
parser.add_argument("--heartbeat-seconds", type=float, default=30.)
parser.add_argument("--analysis-timeout-seconds", type=float, default=600.)
parser.add_argument("--full-sim-cleanup", action="store_true",
                    help="Opt into slow full Kit teardown instead of fast success exit after saving.")
parser.add_argument("--terrain-histogram", type=Path, default=None)
parser.add_argument("--prepass-steps", type=int, default=5000)
parser.add_argument("--prepass-burn-in", type=int, default=4000)
parser.add_argument("--terrain-prepass-only", action="store_true")
parser.add_argument(
    "--replay-check-only", action="store_true",
    help="Diagnose restored LP/VP rollouts in both orders; write no dataset or policy export.",
)
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Allow a small non-production dataset to validate the complete paired-rollout path.",
)
parser.add_argument("--skip-dual-branch-export", action="store_true")
parser.add_argument("--fault-min-duration-frames", type=int, default=None)
parser.add_argument("--fault-max-duration-frames", type=int, default=None)
parser.add_argument("--fault-max-start-delay-frames", type=int, default=None)
parser.add_argument("--fault-min-recovery-frames", type=int, default=None)
parser.add_argument("--fault-max-recovery-frames", type=int, default=None)
parser.add_argument("--fault-max-dual-stagger-frames", type=int, default=None)
parser.add_argument(
    "--fault-severity-probabilities", nargs=3, type=float, default=None,
    metavar=("LIGHT", "MEDIUM", "HEAVY"),
    help="Scheduled fault-plan sampling probabilities, default: 0.10 0.30 0.60.",
)
parser.add_argument("--behavior-min-dwell-frames", type=int, default=10)
parser.add_argument("--behavior-max-dwell-frames", type=int, default=50)
parser.add_argument("--max-candidate-batches", type=int, default=100000)
parser.add_argument(
    "--unbalanced-scenarios",
    action="store_true",
    help="Diagnostic only: disable accepted-sample scenario quotas",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.replay_check_only and not args_cli.terrain_prepass_only:
    analysis_files = (
        CURRENT_DIR / "analyze_selector_collection.py",
        REPO_ROOT / "legged_lab" / "selector" / "analysis.py",
    )
    missing_analysis = [str(path) for path in analysis_files if not path.is_file()]
    if missing_analysis:
        parser.error("missing selector analysis files: " + ", ".join(missing_analysis)
                     + "; restore these files before launching collection. "
                       "Already saved datasets can be analyzed separately without recollecting.")
args_cli.enable_cameras = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(line_buffering=True, write_through=True)
print("[selector stage] launch_simulation", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.selector.collector import CounterfactualCollector  # noqa: E402
from legged_lab.selector.artifacts import export_dual_branch_artifact  # noqa: E402
from legged_lab.selector.config import (  # noqa: E402
    CollectionConfig,
    FaultConfig,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_REFERENCE_POLICY_PATH,
    SelectorConfig,
)
from legged_lab.selector.dataset import SelectorDatasetWriter  # noqa: E402
from legged_lab.selector.collection_progress import CollectionProgress, close_simulation  # noqa: E402
from legged_lab.selector.collection_physics import create_collection_environment, collection_physics_settings  # noqa: E402
from legged_lab.selector.faults import SelectorFaultManager  # noqa: E402
from legged_lab.selector.supplement import (  # noqa: E402
    fault_profile, SupplementConfig, scaled_pair_quotas, DIRECTED_PROFILES, resolve_fault_source,
)
from legged_lab.selector.supplement_dataset import (  # noqa: E402
    base_collection_partition, restrict_terrain_distribution, compatible_manifests,
)
from legged_lab.selector.frozen_policy import (  # noqa: E402
    load_frozen_adapter,
    validate_actions,
)
from legged_lab.selector.terrain import (  # noqa: E402
    TerrainDistribution,
    apply_frozen_distribution,
)
from legged_lab.utils import task_registry  # noqa: E402

progress = None


PROTECTED_FILES = (
    REPO_ROOT / "legged_lab/envs/g1/RENet_env.py",
    REPO_ROOT / "legged_lab/envs/g1/RENet_cfg.py",
    REPO_ROOT / "rsl_rl/rsl_rl/modules/renet_actor_critic.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_run_info(seed: int) -> dict:
    def git_value(arguments):
        try:
            return subprocess.run(
                ["git", *arguments], cwd=REPO_ROOT, check=True,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    source_paths = sorted((REPO_ROOT / "legged_lab/selector").glob("*.py"))
    source_paths += [Path(__file__).resolve()]
    return {
        "collection_run_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "seed": seed,
        "fault_seed": args_cli.fault_seed if args_cli.fault_seed is not None else seed + 2000,
        "behavior_seed": args_cli.behavior_seed if args_cli.behavior_seed is not None else seed,
        "terrain_generation_seed": getattr(args_cli, "terrain_generation_seed", seed),
        "terrain_sampling_seed": seed + 1000,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_tracked_changes": git_value(["status", "--porcelain", "--untracked-files=no"]),
        "selector_source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in source_paths},
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }


def patch_mask(
    observations: torch.Tensor, env, policy, mask_value: float
) -> torch.Tensor:
    observations = observations.clone()
    index = policy.proprio_actor_dim + policy.lidar_flat_dim + policy.depth_flat_dim
    observations[:, index : index + 1] = mask_value
    env.renet_estimator_mask.fill_(mask_value)
    env.cfg.renet.mask_mode = "lp" if mask_value > 0.5 else "vp"
    return observations


@torch.no_grad()
def run_terrain_prepass(
    env,
    adapter,
    *,
    steps: int,
    burn_in: int,
    mask_block_steps: int,
    output: Path,
) -> TerrainDistribution:
    terrain = env.scene.terrain
    rows, columns = terrain.terrain_origins.shape[:2]
    distribution = TerrainDistribution.empty(rows, columns, env.device)
    observations, _ = env.get_observations()
    for step in range(steps):
        mask_value = 1.0 if (step // mask_block_steps) % 2 == 0 else 0.0
        observations = patch_mask(observations, env, adapter.policy, mask_value)
        branches = adapter.evaluate(observations)
        actions = branches.lp_action if mask_value > 0.5 else branches.vp_action
        observations, _, _, _ = env.step(actions)
        if step >= burn_in and step % 5 == 0:
            distribution.observe(terrain.terrain_levels, terrain.terrain_types)
        if step and step % 500 == 0:
            print(
                f"[SelectorTerrain] step={step}/{steps}, "
                f"observed_mean={distribution.mean_level:.3f}, samples={distribution.sample_count}"
            )
    if distribution.sample_count == 0:
        raise RuntimeError("terrain prepass did not record any level 0-6 samples")
    distribution.save(output)
    print(
        f"[SelectorTerrain] wrote {output}; mean={distribution.mean_level:.3f}, "
        f"samples={distribution.sample_count}"
    )
    return distribution


@torch.no_grad()
def export_dual_branch(
    adapter, sample: torch.Tensor, output_dir: Path, frozen_metadata: dict
) -> Path:
    return export_dual_branch_artifact(
        adapter, output_dir / "renet_dual_branch.pt", frozen_metadata, sample=sample
    )


def run_replay_diagnostics(collector, output: Path, metadata: dict) -> dict:
    """Persist all available probes even if a later probe raises an exception."""
    reports = {}
    order = (("lp_before_vp", 1.0), ("vp_after_lp", 0.0),
             ("vp_before_lp", 0.0), ("lp_after_vp", 1.0))
    try:
        for name, mask in order:
            reports[name] = collector.validate_replay(mask_value=mask)
            summary = {key: reports[name].get(key) for key in (
                "passed", "failed_checks", "label_agreement", "label_mismatch_count",
                "initial_restore_max_abs_error", "first_divergent_step", "physics_settings")}
            print(f"[selector] replay {name}: {summary}", flush=True)
    finally:
        report = {
            **metadata,
            "diagnostic_only": True,
            "passed": len(reports) == len(order) and all(r["passed"] for r in reports.values()),
            "probes": reports,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[selector] replay diagnostics: {output}")
    return report


def collection_fault_config(arguments):
    """Pure configuration entry point, testable without starting a simulator."""
    name = arguments.fault_profile or ("mixed" if arguments.supplement_split else "baseline")
    cfg = fault_profile(name)
    overrides = {key: getattr(arguments, "fault_"+key) for key in (
        "min_duration_frames", "max_duration_frames", "max_start_delay_frames",
        "min_recovery_frames", "max_recovery_frames", "max_dual_stagger_frames")
        if getattr(arguments, "fault_"+key) is not None}
    if arguments.fault_severity_probabilities is not None:
        overrides["severity_probabilities"] = tuple(arguments.fault_severity_probabilities)
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def main() -> None:
    global progress
    progress = CollectionProgress(interval_s=args_cli.heartbeat_seconds).start()
    progress.announce("validate_arguments")
    if not math.isfinite(args_cli.analysis_timeout_seconds) or args_cli.analysis_timeout_seconds <= 0:
        raise ValueError("analysis timeout must be positive and finite")
    if args_cli.terrain_prepass_only and args_cli.replay_check_only:
        raise ValueError("choose only one of --terrain-prepass-only and --replay-check-only")
    if args_cli.task != "g1_renet":
        raise ValueError("selector collection is isolated to task='g1_renet'")
    if args_cli.num_envs != 4096 and not args_cli.smoke_test:
        raise ValueError(
            "the approved selector collection uses exactly 4096 environments"
        )
    supplement_mode = args_cli.supplement_split is not None
    args_cli.fault_profile = args_cli.fault_profile or ("mixed" if supplement_mode else "baseline")
    if (args_cli.base_dataset is not None) != supplement_mode:
        raise ValueError("use --base-dataset and --supplement-split together")
    if not supplement_mode and (args_cli.pair_quotas is not None or args_cli.no_outcome_quotas):
        raise ValueError("outcome quota options require supplement mode")
    if not supplement_mode and (args_cli.fault_source is not None or args_cli.fault_profile in DIRECTED_PROFILES):
        raise ValueError("directed profiles and source filters require supplement mode")
    args_cli.fault_source = resolve_fault_source(args_cli.fault_profile, args_cli.fault_source)
    if not supplement_mode and (args_cli.warmup_seconds is not None or args_cli.min_collection_span_seconds is not None
                               or args_cli.label_interval_frames is not None or args_cli.randomize_label_phase is not None
                               or args_cli.max_samples_per_labelled_batch):
        raise ValueError("temporal sampling options require supplement mode")
    base_index, allowed_groups, allowed_pairs = None, None, None
    terrain_seed = args_cli.seed
    if supplement_mode:
        if args_cli.terrain_mode != "main" or args_cli.terrain_histogram is None:
            raise ValueError("supplements require main terrain mode and the original --terrain-histogram")
        progress.announce("read_base_partition")
        base_index, allowed_groups, terrain_seed, allowed_pairs = base_collection_partition(
            args_cli.base_dataset, args_cli.supplement_split)
        base_info = base_index["manifest"]["metadata"].get("run_info", {})
        base_fault_seed = base_info.get("fault_seed", int(base_info.get("seed", terrain_seed))+2000)
        new_fault_seed = args_cli.fault_seed if args_cli.fault_seed is not None else args_cli.seed+2000
        if (int(new_fault_seed) & 0xFFFFFFFF) == (int(base_fault_seed) & 0xFFFFFFFF):
            raise ValueError("supplement needs a new --seed or --fault-seed, independent of the base run")
        args_cli.terrain_generation_seed = terrain_seed
    checkpoint_path = (
        (args_cli.checkpoint_path or DEFAULT_CHECKPOINT_PATH).expanduser().resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        args_cli.prepass_burn_in >= args_cli.prepass_steps
        and args_cli.terrain_histogram is None
    ):
        raise ValueError("prepass-burn-in must be smaller than prepass-steps")
    default_output = (
        REPO_ROOT
        / "outputs/g1_renet_selector_v4_1s_normal"
        / ("dataset" if args_cli.terrain_mode == "main" else "dataset_ood_7_9")
    )
    output = (args_cli.output or default_output).expanduser().resolve()
    if not args_cli.terrain_prepass_only and not args_cli.replay_check_only:
        if (output / "manifest.json").exists() or any(output.glob("*/shard_*.pt")):
            raise FileExistsError(
                f"refusing to overwrite an existing selector dataset: {output}"
            )
        if (
            not args_cli.skip_dual_branch_export
            and (output.parent / "exported/renet_dual_branch.pt").exists()
        ):
            raise FileExistsError(
                "dual-branch export exists; use --skip-dual-branch-export or a new run directory"
            )
    progress.status_path = output / "collection_status.json"
    fault_cfg = collection_fault_config(args_cli)

    target_samples = args_cli.target_samples
    if target_samples is None:
        if args_cli.smoke_test:
            target_samples = args_cli.num_envs
        else:
            target_samples = (sum(args_cli.pair_quotas) if args_cli.pair_quotas is not None
                else 262_144 if supplement_mode and args_cli.supplement_split == "train"
                else 32_768 if supplement_mode
                else 1_048_576 if args_cli.terrain_mode == "main" else 131_072)

    supplement_cfg = None
    if supplement_mode:
        is_train = args_cli.supplement_split == "train"
        if not is_train and args_cli.pair_quotas is not None:
            raise ValueError("stress validation/test may not be balanced by outcomes")
        quotas = (tuple(args_cli.pair_quotas) if args_cli.pair_quotas is not None
                  else scaled_pair_quotas(target_samples) if is_train and not args_cli.no_outcome_quotas else None)
        production_temporal = not args_cli.smoke_test and not args_cli.replay_check_only
        supplement_cfg = SupplementConfig(split=args_cli.supplement_split, pair_quotas=quotas,
            max_samples_per_event=args_cli.max_samples_per_event,
            min_event_spacing_s=args_cli.min_event_spacing_s,
            max_no_progress_batches=args_cli.max_no_progress_batches,
            warmup_s=(args_cli.warmup_seconds if args_cli.warmup_seconds is not None
                      else 15. if production_temporal else 0.),
            min_collection_span_s=(args_cli.min_collection_span_seconds if args_cli.min_collection_span_seconds is not None
                                   else 30. if production_temporal else 0.),
            label_interval_frames=(args_cli.label_interval_frames if args_cli.label_interval_frames is not None
                                   else 10 if production_temporal else 1),
            randomize_label_phase=(args_cli.randomize_label_phase if args_cli.randomize_label_phase is not None
                                   else production_temporal),
            max_samples_per_labelled_batch=args_cli.max_samples_per_labelled_batch,
            fault_source=args_cli.fault_source)
        supplement_cfg.validate(target_samples)
        if supplement_cfg.min_collection_span_s >= args_cli.max_candidate_batches*.1:
            raise ValueError("candidate budget is shorter than the requested real collection span")

    collection_cfg = CollectionConfig(
        checkpoint_path=str(checkpoint_path),
        num_envs=args_cli.num_envs,
        target_samples=target_samples,
        allow_nonproduction_num_envs=args_cli.smoke_test,
        behavior_min_dwell_frames=args_cli.behavior_min_dwell_frames,
        behavior_max_dwell_frames=args_cli.behavior_max_dwell_frames,
        balanced_scenarios=not args_cli.unbalanced_scenarios and not supplement_mode,
        max_candidate_batches=args_cli.max_candidate_batches,
    )
    collection_cfg.validate()
    progress.announce("load_frozen_policy")
    protected_before = {str(path): sha256(path) for path in PROTECTED_FILES}
    adapter, reference, frozen_metadata = load_frozen_adapter(
        checkpoint_path,
        args_cli.reference_policy or DEFAULT_REFERENCE_POLICY_PATH,
    )
    if supplement_mode:
        temporary_manifest = dict(base_index["manifest"])
        temporary_manifest["metadata"] = {**temporary_manifest["metadata"], "frozen_policy": frozen_metadata}
        compatible_manifests(base_index["manifest"], temporary_manifest)
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg, agent_cfg = copy.deepcopy(env_cfg), copy.deepcopy(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = terrain_seed
    agent_cfg.seed = args_cli.seed
    env_cfg.device = args_cli.device
    # Match the task's sole AME height-input contract.
    env_cfg.renet.lp_height_source = "pgtt_full_atten"
    env_cfg.renet.mask_mode = "vp"
    env_cfg.robot.actor_obs_history_length = 10
    env_cfg.robot.critic_obs_history_length = 10
    env_cfg.scene.camera.add_camera_noise = True
    # Keep foot scanners: the current task's active stair-edge rewards use them.
    env_cfg.commands.debug_vis = False
    env_cfg.sim.physx.enable_enhanced_determinism = True
    if hasattr(env_cfg, "lp_height"):
        env_cfg.lp_height.noise.enable = True
        env_cfg.lp_height.noise.force_active = True
    mask_block_steps = max(
        1,
        int(env_cfg.renet.alternate_interval_iters) * int(agent_cfg.num_steps_per_env),
    )
    env_class = task_registry.get_task_class(args_cli.task)
    env_class.seed(terrain_seed)
    progress.announce("create_environment", num_envs=args_cli.num_envs)
    env = create_collection_environment(env_class, env_cfg, args_cli.headless)
    print(f"[selector physics] {json.dumps(collection_physics_settings(env))}", flush=True)
    fault_cfg = replace(fault_cfg, lp_resolution_m=float(env.lp_runtime_cfg.resolution))
    fault_cfg.validate()
    adapter = adapter.to(env.device).eval()
    progress.announce("policy_parity")

    parity_observations, parity_extras = env.get_observations()
    if (
        parity_observations.shape[1] != 2120
        or parity_extras["observations"]["critic"].shape[1] != 1037
    ):
        raise ValueError(
            "archived RENet requires actor=2120 and critic=1037 observations"
        )
    if not abs(env.step_dt - 0.02) < 1.0e-8 or env.num_actions != 23:
        raise ValueError("archived RENet requires 50 Hz control and 23 actions")
    lp_error, vp_error = validate_actions(
        adapter, reference, [parity_observations[: min(32, env.num_envs)]]
    )
    parity = {"lp": lp_error, "vp": vp_error}
    print(f"[selector] frozen-policy parity passed: {parity}")

    histogram_path = (
        args_cli.terrain_histogram.expanduser().resolve()
        if args_cli.terrain_histogram is not None
        else output.parent / "terrain_distribution.json"
    )
    if args_cli.terrain_histogram is not None:
        progress.announce("load_terrain_distribution")
        if supplement_mode and sha256(histogram_path) != base_index["manifest"]["metadata"].get("terrain_distribution_sha256"):
            raise ValueError("supplements must use the unchanged original terrain histogram")
        distribution = TerrainDistribution.load(histogram_path, env.device)
    else:
        progress.announce("terrain_prepass")
        distribution = run_terrain_prepass(
            env,
            adapter,
            steps=args_cli.prepass_steps,
            burn_in=args_cli.prepass_burn_in,
            mask_block_steps=mask_block_steps,
            output=histogram_path,
        )
    if args_cli.terrain_prepass_only:
        return
    if supplement_mode:
        restrict_terrain_distribution(distribution, allowed_pairs)

    terrain_generator = env.cfg.scene.terrain_generator
    if terrain_generator is not None:
        terrain_generator.curriculum = False
    sampling_generator = torch.Generator(device=torch.device(env.device).type)
    sampling_generator.manual_seed(args_cli.seed + 1000)
    apply_frozen_distribution(
        env,
        distribution,
        generator=sampling_generator,
        ood_levels=(7, 8, 9) if args_cli.terrain_mode == "ood" else None,
    )
    observations, _ = env.get_observations()
    faults = SelectorFaultManager(
        env.num_envs,
        lp_shape=tuple(env.lp_observation_shape),
        vp_shape=(env.depth_history_frames, env.camera_height, env.camera_width),
        device=env.device,
        seed=args_cli.fault_seed if args_cli.fault_seed is not None else args_cli.seed + 2000,
        cfg=fault_cfg,
    )
    if args_cli.replay_check_only:
        progress.announce("replay_diagnostics")
        collector = CounterfactualCollector(
            env, adapter, None, faults=faults, selector_cfg=SelectorConfig(),
            collection_cfg=collection_cfg, seed=args_cli.seed,
            terrain_seed=terrain_seed, behavior_seed=args_cli.behavior_seed,
            supplement_cfg=supplement_cfg, allowed_groups=allowed_groups,
            progress=progress,
        )
        diagnostic_path = output.parent / "replay_diagnostics.json"
        report = run_replay_diagnostics(collector, diagnostic_path, {
            "run_info": collection_run_info(args_cli.seed),
            "fault_config": asdict(fault_cfg),
            "collection_config": asdict(collection_cfg),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "terrain_distribution": str(histogram_path),
            "terrain_distribution_sha256": sha256(histogram_path),
        })
        if not report["passed"]:
            raise RuntimeError(f"replay diagnostics failed; inspect {diagnostic_path}")
        return
    if not args_cli.skip_dual_branch_export:
        artifact = export_dual_branch(
            adapter,
            observations,
            output.parent / "exported",
            frozen_metadata,
        )
        print(f"[selector] exported {artifact}")

    selector_cfg = SelectorConfig()
    if (output / "manifest.json").exists() or any(output.glob("*/shard_*.pt")):
        raise FileExistsError(
            f"refusing to overwrite an existing selector dataset: {output}"
        )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "lp_height_source": env.lp_height_source,
        "lp_observation_contract": env.lp_observation_contract,
        "lp_encoder_contract": frozen_metadata["lp_encoder_contract"],
        "frozen_policy": frozen_metadata,
        "terrain_distribution": str(histogram_path),
        "terrain_distribution_sha256": sha256(histogram_path),
        "terrain_observed_mean": distribution.mean_level,
        "terrain_mode": args_cli.terrain_mode,
        "smoke_test": args_cli.smoke_test,
        "terrain_levels": [0, 1, 2, 3, 4, 5, 6]
        if args_cli.terrain_mode == "main"
        else [7, 8, 9],
        "frozen_policy_action_parity": parity,
        "protected_source_sha256": protected_before,
        "fault_config": asdict(fault_cfg),
        "collection_config": asdict(collection_cfg),
        "run_info": collection_run_info(args_cli.seed),
        "collection_complete": False,
    }
    if supplement_mode:
        metadata["supplement_source"] = {"version": 1, "split": args_cli.supplement_split,
            "base_manifest_sha256": base_index["sha256"], "base_dataset": str(base_index["root"]),
            "allowed_terrain_groups": sorted(allowed_groups), "terrain_generation_seed": terrain_seed,
            "config": asdict(supplement_cfg), "fault_profile": args_cli.fault_profile}
    writer = SelectorDatasetWriter(
        output,
        collection_cfg=collection_cfg,
        selector_cfg=selector_cfg,
        metadata=metadata,
        fixed_split=args_cli.supplement_split or ("test" if args_cli.terrain_mode == "ood" else None),
    )
    (output / "collection_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    collector = CounterfactualCollector(
        env,
        adapter,
        writer,
        faults=faults,
        selector_cfg=selector_cfg,
        collection_cfg=collection_cfg,
        seed=args_cli.seed,
        terrain_seed=terrain_seed, behavior_seed=args_cli.behavior_seed,
        supplement_cfg=supplement_cfg, allowed_groups=allowed_groups,
        progress=progress,
    )
    try:
        collector.collect()
    finally:
        collection_failed = sys.exc_info()[0] is not None
        progress.announce("finalize_collection_audit", collection_failed=collection_failed)
        collector.finalize_audit()
        progress.announce("save_dataset_manifest")
        manifest = writer.close()
        progress.announce("dataset_saved", manifest=str(manifest),
                          collection_complete=writer.metadata.get("collection_complete", False))
        try:
            progress.announce("analyze_saved_dataset", timeout_s=args_cli.analysis_timeout_seconds)
            # CPU work runs outside the Kit/PhysX thread pools. The parent
            # heartbeat remains active if loading/reporting is slow.
            analysis_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
            subprocess.run([sys.executable, str(CURRENT_DIR / "analyze_selector_collection.py"),
                            "--dataset", str(output), "--threads", "1"],
                           check=True, env=analysis_env, timeout=args_cli.analysis_timeout_seconds)
            print(f"[selector] collection analysis: {output / 'collection_analysis.json'}", flush=True)
        except Exception:
            if not collection_failed:
                raise
            print("[selector] partial analysis failed; preserving original collection error", file=sys.stderr)
            traceback.print_exc()
    progress.announce("verify_frozen_sources")
    protected_after = {str(path): sha256(path) for path in PROTECTED_FILES}
    if protected_before != protected_after:
        raise RuntimeError(
            "protected RENet source files changed during selector collection"
        )
    progress.announce("dataset_and_analysis_complete", manifest=str(manifest))
    print(f"[selector] dataset complete: {manifest}", flush=True)


if __name__ == "__main__":
    failed = False
    try:
        main()
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        try:
            if progress is not None:
                progress.announce("shutdown_simulation", failed=failed,
                                  requested_full_cleanup=args_cli.full_sim_cleanup)
            sys.stdout.flush()
            sys.stderr.flush()
            # All synchronous dataset writes and the CPU analysis process have
            # finished before successful fast exit. Failures preserve shell status.
            close_simulation(simulation_app, failed=failed,
                             full_cleanup=args_cli.full_sim_cleanup)
        finally:
            if progress is not None:
                progress.stop()
