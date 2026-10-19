"""Replay synthetic 10 Hz packets and record configuration/seed/fault coverage.

This is a module check, not an Isaac Sim rollout or a physical safety result.
Run from the repository root with PYTHONPATH pointing to the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from legged_lab.high_level.high_level_faults import HighLevelSensorFaultPipeline, SensorFrame


def replay(*, num_envs=32, frames=1200, seed=42):
    pipeline = HighLevelSensorFaultPipeline(num_envs, seed=seed)
    yy, xx = torch.meshgrid(torch.linspace(-.5, .5, 11), torch.linspace(-.8, .8, 17), indexing="ij")
    lp_surface = .1 * xx + .05 * yy
    image_y, image_x = torch.meshgrid(torch.linspace(0, 1, 18), torch.linspace(0, 1, 32), indexing="ij")
    vp_surface = .3 + 1.8 * image_x + .3 * image_y
    for frame in range(frames):
        stamp = torch.full((num_envs,), frame * .1, dtype=torch.float64)
        sequence = torch.full((num_envs,), frame, dtype=torch.long)
        lp_data = lp_surface[None].expand(num_envs, -1, -1).clone()
        vp_data = (vp_surface[None] + .02 * torch.sin(torch.tensor(frame * .1))).expand(num_envs, -1, -1)
        lp = SensorFrame(lp_data, torch.ones_like(lp_data), stamp, stamp, sequence)
        vp = SensorFrame(vp_data, torch.ones_like(vp_data), stamp, stamp, sequence)
        pose = torch.tensor([[0., 0., .8, 0.]]).repeat(num_envs, 1)
        observations = pipeline.update(lp, vp, pose)
        assert torch.isfinite(observations.lp).all() and torch.isfinite(observations.vp).all()
        # Read at the two downstream rates without advancing the fault clock.
        for _ in range(4):
            pipeline.read()
    audit = pipeline.audit()

    def distribution(counts, names):
        counts = counts.sum(0)
        total = max(int(counts.sum()), 1)
        return {str(name): {"count": int(count), "fraction": float(count) / total}
                for name, count in zip(names, counts)}

    return dict(
        scope="synthetic sensor packets; no simulator, policy or physical fall-rate evaluation",
        seed=seed, num_envs=num_envs, sensor_frames_per_env=frames,
        seconds_per_env=frames * .1, config=audit["config"],
        plans=distribution(audit["plans"], audit["scenarios"]),
        lp_types=distribution(audit["lp_events"], audit["lp_kinds"]),
        vp_types=distribution(audit["vp_events"], audit["vp_modes"]),
        planned_injection_frames=distribution(audit["planned_frames"], audit["scenarios"]),
        observed_current_frames=distribution(audit["observed_frames"], audit["scenarios"]),
        observed_history_frames=distribution(audit["visible_frames"], audit["scenarios"]),
        lp_residual_frames_without_active_injection=int(audit["residual_frames"].sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 1 or args.num_envs < 1:
        parser.error("frames and num-envs must be positive")
    torch.set_num_threads(1)
    result = replay(num_envs=args.num_envs, frames=args.frames, seed=args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("seed", "num_envs", "sensor_frames_per_env", "plans")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
