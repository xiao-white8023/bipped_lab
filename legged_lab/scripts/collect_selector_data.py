"""Collect paired LP/VP selector labels in a completely separate Isaac process."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import sys
import traceback


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
parser.add_argument("--terrain-histogram", type=Path, default=None)
parser.add_argument("--prepass-steps", type=int, default=5000)
parser.add_argument("--prepass-burn-in", type=int, default=4000)
parser.add_argument("--terrain-prepass-only", action="store_true")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Allow a small non-production dataset to validate the complete paired-rollout path.",
)
parser.add_argument("--skip-dual-branch-export", action="store_true")
parser.add_argument("--fault-min-duration-frames", type=int, default=3)
parser.add_argument("--fault-max-duration-frames", type=int, default=50)
parser.add_argument("--fault-max-start-delay-frames", type=int, default=30)
parser.add_argument("--fault-min-recovery-frames", type=int, default=5)
parser.add_argument("--fault-max-recovery-frames", type=int, default=30)
parser.add_argument("--fault-max-dual-stagger-frames", type=int, default=10)
parser.add_argument("--behavior-min-dwell-frames", type=int, default=10)
parser.add_argument("--behavior-max-dwell-frames", type=int, default=50)
parser.add_argument("--max-candidate-batches", type=int, default=100000)
parser.add_argument(
    "--unbalanced-scenarios",
    action="store_true",
    help="Diagnostic only: disable accepted-sample scenario quotas",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
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
from legged_lab.selector.faults import SelectorFaultManager  # noqa: E402
from legged_lab.selector.frozen_policy import (  # noqa: E402
    load_frozen_adapter,
    validate_actions,
)
from legged_lab.selector.terrain import (  # noqa: E402
    TerrainDistribution,
    apply_frozen_distribution,
)
from legged_lab.utils import task_registry  # noqa: E402


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


def main() -> None:
    if args_cli.task != "g1_renet":
        raise ValueError("selector collection is isolated to task='g1_renet'")
    if args_cli.num_envs != 4096 and not args_cli.smoke_test:
        raise ValueError(
            "the approved selector collection uses exactly 4096 environments"
        )
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
        / "outputs/g1_renet_selector"
        / ("dataset" if args_cli.terrain_mode == "main" else "dataset_ood_7_9")
    )
    output = (args_cli.output or default_output).expanduser().resolve()
    if not args_cli.terrain_prepass_only:
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
    fault_cfg = FaultConfig(
        min_duration_frames=args_cli.fault_min_duration_frames,
        max_duration_frames=args_cli.fault_max_duration_frames,
        max_start_delay_frames=args_cli.fault_max_start_delay_frames,
        min_recovery_frames=args_cli.fault_min_recovery_frames,
        max_recovery_frames=args_cli.fault_max_recovery_frames,
        max_dual_stagger_frames=args_cli.fault_max_dual_stagger_frames,
    )
    fault_cfg.validate()

    target_samples = args_cli.target_samples
    if target_samples is None:
        if args_cli.smoke_test:
            target_samples = args_cli.num_envs
        else:
            target_samples = 1_048_576 if args_cli.terrain_mode == "main" else 131_072

    collection_cfg = CollectionConfig(
        checkpoint_path=str(checkpoint_path),
        num_envs=args_cli.num_envs,
        target_samples=target_samples,
        allow_nonproduction_num_envs=args_cli.smoke_test,
        behavior_min_dwell_frames=args_cli.behavior_min_dwell_frames,
        behavior_max_dwell_frames=args_cli.behavior_max_dwell_frames,
        balanced_scenarios=not args_cli.unbalanced_scenarios,
        max_candidate_batches=args_cli.max_candidate_batches,
    )
    collection_cfg.validate()
    protected_before = {str(path): sha256(path) for path in PROTECTED_FILES}
    adapter, reference, frozen_metadata = load_frozen_adapter(
        checkpoint_path,
        args_cli.reference_policy or DEFAULT_REFERENCE_POLICY_PATH,
    )
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg, agent_cfg = copy.deepcopy(env_cfg), copy.deepcopy(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = args_cli.seed
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
    env_class.seed(args_cli.seed)
    env = env_class(env_cfg, args_cli.headless)
    fault_cfg = replace(fault_cfg, lp_resolution_m=float(env.lp_runtime_cfg.resolution))
    fault_cfg.validate()
    adapter = adapter.to(env.device).eval()

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
        distribution = TerrainDistribution.load(histogram_path, env.device)
    else:
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
        "terrain_observed_mean": distribution.mean_level,
        "terrain_mode": args_cli.terrain_mode,
        "smoke_test": args_cli.smoke_test,
        "terrain_levels": [0, 1, 2, 3, 4, 5, 6]
        if args_cli.terrain_mode == "main"
        else [7, 8, 9],
        "frozen_policy_action_parity": parity,
        "protected_source_sha256": protected_before,
        "fault_config": asdict(fault_cfg),
        "collection_complete": False,
    }
    writer = SelectorDatasetWriter(
        output,
        collection_cfg=collection_cfg,
        selector_cfg=selector_cfg,
        metadata=metadata,
        fixed_split="test" if args_cli.terrain_mode == "ood" else None,
    )
    faults = SelectorFaultManager(
        env.num_envs,
        lp_shape=tuple(env.lp_observation_shape),
        vp_shape=(env.depth_history_frames, env.camera_height, env.camera_width),
        device=env.device,
        seed=args_cli.seed + 2000,
        cfg=fault_cfg,
    )
    collector = CounterfactualCollector(
        env,
        adapter,
        writer,
        faults=faults,
        selector_cfg=selector_cfg,
        collection_cfg=collection_cfg,
        seed=args_cli.seed,
    )
    try:
        collector.collect()
    finally:
        manifest = writer.close()
    protected_after = {str(path): sha256(path) for path in PROTECTED_FILES}
    if protected_before != protected_after:
        raise RuntimeError(
            "protected RENet source files changed during selector collection"
        )
    print(f"[selector] dataset complete: {manifest}")


if __name__ == "__main__":
    failed = False
    try:
        main()
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        # A prepass-only process owns no dataset writer and exits immediately;
        # full Kit teardown can take minutes on large cloned scenes.
        simulation_app.close(
            skip_cleanup=failed or args_cli.terrain_prepass_only or args_cli.smoke_test
        )