#!/usr/bin/env python3
"""Stage 2 v2: select clean get-up clips from one or more whole-body NPZ files.

Compared with v1, this version explicitly rejects clips that transition into
fast locomotion after standing and crops away the preceding fall/run phase.

Detection
---------
Fallen state:
    pelvis height <= fallen_height
    torso local +Z projected onto world Z <= fallen_up_z
    held for fallen_hold_s

Upright state:
    pelvis height >= upright_height
    torso up_z >= upright_up_z
    held for upright_hold_s

Clean get-up clip:
    start = 0.5 s before the robot LEAVES the fallen state
    end   = 0.5 s after stable-upright onset

Additional anti-running filters:
    - upright state must persist for at least min_upright_duration_s
    - median/max horizontal pelvis speed during the first post-upright window
      must remain below thresholds
    - max horizontal pelvis speed during the pre-getup window must remain
      below a threshold

This script never recomputes FK or velocities. It only slices Stage-1 data.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


CORE_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class Run:
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    source: Path
    raw_index: int
    fallen_start: int
    fallen_end: int
    upright_start: int
    upright_end: int
    clip_start: int
    clip_end: int
    pre_speed_median: float
    pre_speed_max: float
    post_speed_median: float
    post_speed_max: float
    upright_duration_s: float
    keep: bool
    reject_reason: str


def _scalar_float(value: np.ndarray, name: str) -> float:
    a = np.asarray(value)
    if a.size != 1:
        raise ValueError(f"{name} must contain one scalar, got {a.shape}.")
    v = float(a.reshape(-1)[0])
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be positive and finite, got {v}.")
    return v


def _string_list(value: np.ndarray, name: str) -> list[str]:
    a = np.asarray(value)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {a.shape}.")
    out = []
    for item in a.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        out.append(str(item))
    return out


def _find_runs(mask: np.ndarray, min_frames: int) -> list[Run]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        Run(int(s), int(e))
        for s, e in zip(starts, ends)
        if int(e - s) >= min_frames
    ]


def _torso_up_z(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    _, x, y, _ = q.T
    return 1.0 - 2.0 * (x * x + y * y)


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with np.load(resolved, allow_pickle=False) as data:
        missing = [k for k in CORE_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{resolved.name}: missing keys {missing}.")
        payload = {k: np.asarray(data[k]) for k in data.files}

    fps = _scalar_float(payload["fps"], "fps")
    t = payload["joint_pos"].shape[0]
    for key in CORE_KEYS[1:]:
        if payload[key].shape[0] != t:
            raise ValueError(f"{resolved.name}: {key} frame count mismatch.")
        if not np.isfinite(payload[key]).all():
            raise ValueError(f"{resolved.name}: {key} contains NaN/Inf.")

    payload["_fps_scalar"] = np.asarray(fps, dtype=np.float64)
    payload["_path"] = np.asarray(str(resolved), dtype=np.str_)
    return payload


def _resolve_indexes(payload: dict[str, np.ndarray]) -> tuple[int, int]:
    names = _string_list(payload["body_names"], "body_names")
    root = (
        int(np.asarray(payload["root_body_index"]).reshape(-1)[0])
        if "root_body_index" in payload
        else names.index("pelvis")
    )
    torso = names.index("torso_link")
    return root, torso


def detect(
    source: Path,
    payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[Candidate]:
    fps = float(payload["_fps_scalar"])
    root_idx, torso_idx = _resolve_indexes(payload)

    root_pos = payload["body_pos_w"][:, root_idx].astype(np.float64)
    root_vel = payload["body_lin_vel_w"][:, root_idx].astype(np.float64)
    horizontal_speed = np.linalg.norm(root_vel[:, :2], axis=1)
    up_z = _torso_up_z(payload["body_quat_w"][:, torso_idx])

    fallen = (root_pos[:, 2] <= args.fallen_height) & (up_z <= args.fallen_up_z)
    upright = (root_pos[:, 2] >= args.upright_height) & (up_z >= args.upright_up_z)

    fallen_runs = _find_runs(fallen, max(1, int(math.ceil(args.fallen_hold_s * fps))))
    upright_runs = _find_runs(upright, max(1, int(math.ceil(args.upright_hold_s * fps))))

    max_gap = int(round(args.max_gap_s * fps))

    # Map each stable-upright event to the latest fallen run preceding it.
    # This avoids duplicate clips when several low-state runs lead into one success.
    mapping: dict[int, tuple[Run, Run]] = {}
    for fallen_run in fallen_runs:
        future = [
            u for u in upright_runs
            if u.start >= fallen_run.end and u.start - fallen_run.end <= max_gap
        ]
        if not future:
            continue
        upright_run = future[0]
        old = mapping.get(upright_run.start)
        if old is None or fallen_run.end > old[0].end:
            mapping[upright_run.start] = (fallen_run, upright_run)

    result: list[Candidate] = []
    raw_idx = 0
    pre_frames = int(round(args.pre_getup_s * fps))
    post_frames = int(round(args.post_upright_s * fps))
    speed_window_frames = int(round(args.speed_check_s * fps))

    for upright_key in sorted(mapping):
        raw_idx += 1
        fallen_run, upright_run = mapping[upright_key]

        clip_start = max(fallen_run.start, fallen_run.end - pre_frames)
        clip_end = min(upright_run.end, upright_run.start + post_frames)

        pre_start = max(fallen_run.start, fallen_run.end - speed_window_frames)
        post_end = min(len(horizontal_speed), upright_run.start + speed_window_frames)

        pre_slice = horizontal_speed[pre_start:fallen_run.end]
        post_slice = horizontal_speed[upright_run.start:post_end]

        pre_med = float(np.median(pre_slice)) if len(pre_slice) else 0.0
        pre_max = float(np.max(pre_slice)) if len(pre_slice) else 0.0
        post_med = float(np.median(post_slice)) if len(post_slice) else 0.0
        post_max = float(np.max(post_slice)) if len(post_slice) else 0.0
        upright_duration = (upright_run.end - upright_run.start) / fps

        reasons = []
        if upright_duration < args.min_upright_duration_s:
            reasons.append(f"upright<{args.min_upright_duration_s:g}s")
        if pre_max > args.max_pre_speed:
            reasons.append(f"pre_speed_max>{args.max_pre_speed:g}")
        if post_med > args.max_post_speed_median:
            reasons.append(f"post_speed_median>{args.max_post_speed_median:g}")
        if post_max > args.max_post_speed:
            reasons.append(f"post_speed_max>{args.max_post_speed:g}")

        result.append(
            Candidate(
                source=source,
                raw_index=raw_idx,
                fallen_start=fallen_run.start,
                fallen_end=fallen_run.end,
                upright_start=upright_run.start,
                upright_end=upright_run.end,
                clip_start=clip_start,
                clip_end=clip_end,
                pre_speed_median=pre_med,
                pre_speed_max=pre_max,
                post_speed_median=post_med,
                post_speed_max=post_max,
                upright_duration_s=upright_duration,
                keep=not reasons,
                reject_reason=";".join(reasons),
            )
        )
    return result


def _slice_payload(
    payload: dict[str, np.ndarray],
    candidate: Candidate,
) -> dict[str, np.ndarray]:
    start, end = candidate.clip_start, candidate.clip_end
    fps = float(payload["_fps_scalar"])

    out: dict[str, np.ndarray] = {
        "fps": np.asarray([fps], dtype=np.float32),
    }
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        out[key] = np.asarray(payload[key][start:end], dtype=np.float32)

    for key in (
        "joint_names",
        "body_names",
        "root_body_index",
        "root_body_name",
        "quaternion_convention",
        "velocity_frame",
        "body_state_frame",
    ):
        if key in payload:
            out[key] = np.asarray(payload[key])

    out["source_file"] = np.asarray(str(candidate.source), dtype=np.str_)
    out["source_start_frame"] = np.asarray(start, dtype=np.int64)
    out["source_end_frame_exclusive"] = np.asarray(end, dtype=np.int64)
    out["source_start_s"] = np.asarray(start / fps, dtype=np.float64)
    out["source_end_s"] = np.asarray(end / fps, dtype=np.float64)
    out["split_points"] = np.asarray([0, end - start], dtype=np.int64)
    return out


def write_report(
    path: Path,
    all_candidates: Sequence[tuple[Candidate, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "raw_candidate",
        "keep",
        "reject_reason",
        "clip_start_s",
        "clip_end_s",
        "duration_s",
        "fallen_start_s",
        "fallen_end_s",
        "upright_start_s",
        "upright_end_s",
        "upright_duration_s",
        "pre_speed_median",
        "pre_speed_max",
        "post_speed_median",
        "post_speed_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c, fps in all_candidates:
            writer.writerow({
                "source": c.source.name,
                "raw_candidate": c.raw_index,
                "keep": int(c.keep),
                "reject_reason": c.reject_reason,
                "clip_start_s": f"{c.clip_start/fps:.3f}",
                "clip_end_s": f"{c.clip_end/fps:.3f}",
                "duration_s": f"{(c.clip_end-c.clip_start)/fps:.3f}",
                "fallen_start_s": f"{c.fallen_start/fps:.3f}",
                "fallen_end_s": f"{c.fallen_end/fps:.3f}",
                "upright_start_s": f"{c.upright_start/fps:.3f}",
                "upright_end_s": f"{c.upright_end/fps:.3f}",
                "upright_duration_s": f"{c.upright_duration_s:.3f}",
                "pre_speed_median": f"{c.pre_speed_median:.4f}",
                "pre_speed_max": f"{c.pre_speed_max:.4f}",
                "post_speed_median": f"{c.post_speed_median:.4f}",
                "post_speed_max": f"{c.post_speed_max:.4f}",
            })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build clean Recovery get-up clips.")
    p.add_argument("--input_npz", type=Path, nargs="+", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--report_csv", type=Path, default=None)
    p.add_argument("--prefix", type=str, default="recovery_getup")

    p.add_argument("--fallen_height", type=float, default=0.35)
    p.add_argument("--fallen_up_z", type=float, default=0.80)
    p.add_argument("--upright_height", type=float, default=0.70)
    p.add_argument("--upright_up_z", type=float, default=0.90)
    p.add_argument("--fallen_hold_s", type=float, default=0.50)
    p.add_argument("--upright_hold_s", type=float, default=0.50)
    p.add_argument("--max_gap_s", type=float, default=8.0)

    # Crop around the get-up itself, not the preceding fall/run.
    p.add_argument("--pre_getup_s", type=float, default=0.50)
    p.add_argument("--post_upright_s", type=float, default=0.50)

    # Anti-running filters.
    p.add_argument("--speed_check_s", type=float, default=0.50)
    p.add_argument("--min_upright_duration_s", type=float, default=1.0)
    p.add_argument("--max_pre_speed", type=float, default=0.70)
    p.add_argument("--max_post_speed_median", type=float, default=0.50)
    p.add_argument("--max_post_speed", type=float, default=0.80)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.report_csv or (args.output_dir / "stage2_v2_selection.csv")

    all_candidates: list[tuple[Candidate, float]] = []
    payloads: dict[Path, dict[str, np.ndarray]] = {}

    for input_path in args.input_npz:
        resolved = input_path.expanduser().resolve()
        payload = _load(resolved)
        payloads[resolved] = payload
        fps = float(payload["_fps_scalar"])
        candidates = detect(resolved, payload, args)
        all_candidates.extend((c, fps) for c in candidates)

    write_report(report, all_candidates)

    kept = [item for item in all_candidates if item[0].keep]
    print(f"Kept {len(kept)} / {len(all_candidates)} detected candidates.")
    print(f"Report: {report.resolve()}")

    for global_index, (candidate, fps) in enumerate(kept, start=1):
        payload = payloads[candidate.source]
        stem = candidate.source.stem.replace("_whole_body", "")
        output = args.output_dir / (
            f"{args.prefix}_{global_index:02d}_{stem}_"
            f"{candidate.clip_start/fps:07.2f}-{candidate.clip_end/fps:07.2f}s.npz"
        )
        np.savez(output, **_slice_payload(payload, candidate))
        print(
            f"[{global_index:02d}] {candidate.source.name} raw#{candidate.raw_index}: "
            f"{candidate.clip_start/fps:.2f}s -> {candidate.clip_end/fps:.2f}s "
            f"({(candidate.clip_end-candidate.clip_start)/fps:.2f}s) "
            f"post_speed_med={candidate.post_speed_median:.2f}, "
            f"post_speed_max={candidate.post_speed_max:.2f}"
        )


if __name__ == "__main__":
    main()
