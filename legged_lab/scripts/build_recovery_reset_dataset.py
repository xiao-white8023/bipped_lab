#!/usr/bin/env python3
"""Build one-state-per-file G1 Recovery reset datasets from Stage-1 NPZ data.

This is a pure NumPy selection tool. It never starts IsaacLab, recomputes
kinematics/velocities, or transforms the recorded world-frame state.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "legged_lab"
    / "envs"
    / "g1"
    / "datasets"
    / "npz"
    / "recovery_reset_states_v1"
)

ALLOWED_SOURCE_FILENAMES = (
    "fallAndGetUp1_subject1_whole_body.npz",
    "fallAndGetUp1_subject4_whole_body.npz",
    "fallAndGetUp1_subject5_whole_body.npz",
    "fallAndGetUp2_subject2_whole_body.npz",
    "fallAndGetUp2_subject3_whole_body.npz",
    "fallAndGetUp3_subject1_whole_body.npz",
)

CORE_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
METADATA_KEYS = (
    "joint_names",
    "body_names",
    "root_body_index",
    "root_body_name",
    "quaternion_convention",
    "velocity_frame",
    "body_state_frame",
)
REQUIRED_KEYS = ("fps", *CORE_ARRAY_KEYS, *METADATA_KEYS)
DIAGNOSTIC_NAMES = (
    "pelvis_height",
    "torso_up_z",
    "root_lin_speed",
    "root_ang_speed",
    "mean_joint_speed",
)
CSV_FIELDS = (
    "output_file",
    "source_file",
    "source_frame",
    "source_time_s",
    "fallen_run_start_frame",
    "fallen_run_end_frame",
    "pelvis_height",
    "torso_up_z",
    "root_lin_speed",
    "root_ang_speed",
    "mean_joint_speed",
)


@dataclass(frozen=True)
class SelectionConfig:
    fallen_height: float = 0.35
    fallen_up_z: float = 0.80
    min_fallen_duration_s: float = 0.50
    stable_window_s: float = 0.20
    max_root_lin_speed: float = 0.50
    max_root_ang_speed: float = 1.50
    max_mean_joint_speed: float = 1.50
    min_sample_interval_s: float = 0.10

    def validate(self) -> None:
        finite_fields = {
            "fallen_height": self.fallen_height,
            "fallen_up_z": self.fallen_up_z,
            "min_fallen_duration_s": self.min_fallen_duration_s,
            "stable_window_s": self.stable_window_s,
            "max_root_lin_speed": self.max_root_lin_speed,
            "max_root_ang_speed": self.max_root_ang_speed,
            "max_mean_joint_speed": self.max_mean_joint_speed,
            "min_sample_interval_s": self.min_sample_interval_s,
        }
        invalid = [name for name, value in finite_fields.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Selection parameters must be finite: {invalid}.")
        for name in (
            "min_fallen_duration_s",
            "stable_window_s",
            "min_sample_interval_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "max_root_lin_speed",
            "max_root_ang_speed",
            "max_mean_joint_speed",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class FallenRun:
    """A half-open fallen interval ``[start, end)``."""

    start: int
    end: int

    @property
    def num_frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LoadedSource:
    path: Path
    payload: dict[str, np.ndarray]
    fps: float
    num_frames: int
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_body_index: int
    torso_body_index: int


@dataclass(frozen=True)
class FrameDiagnostics:
    pelvis_height: np.ndarray
    torso_up_z: np.ndarray
    root_lin_speed: np.ndarray
    root_ang_speed: np.ndarray
    mean_joint_speed: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in DIAGNOSTIC_NAMES}


@dataclass(frozen=True)
class ResetCandidate:
    source: LoadedSource
    frame: int
    run: FallenRun
    pelvis_height: float
    torso_up_z: float
    root_lin_speed: float
    root_ang_speed: float
    mean_joint_speed: float

    @property
    def source_time_s(self) -> float:
        return self.frame / self.source.fps


@dataclass(frozen=True)
class SourceSelection:
    source: LoadedSource
    fallen_mask: np.ndarray
    fallen_runs: tuple[FallenRun, ...]
    diagnostics: FrameDiagnostics
    stable_candidates: tuple[ResetCandidate, ...]
    candidates_before_thinning: tuple[ResetCandidate, ...]
    saved_candidates: tuple[ResetCandidate, ...]


@dataclass(frozen=True)
class SavedResetState:
    output_path: Path
    candidate: ResetCandidate


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    csv_path: Path
    selections: tuple[SourceSelection, ...]
    saved_states: tuple[SavedResetState, ...]


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got {array.shape}.")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}.")
    return result


def _scalar_int(value: np.ndarray, name: str) -> int:
    scalar = _scalar_float(value, name)
    result = int(scalar)
    if scalar != result:
        raise ValueError(f"{name} must be an integer, got {scalar}.")
    return result


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one string, got {array.shape}.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    result = str(item)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _string_tuple(value: np.ndarray, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    result = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty strings.")
    return tuple(result)


def load_source_npz(path: Path) -> LoadedSource:
    """Load and strictly validate one Stage-1 whole-body NPZ source."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with np.load(resolved, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{resolved.name}: missing required keys {missing}.")
        payload = {key: np.asarray(data[key]).copy() for key in REQUIRED_KEYS}

    fps = _scalar_float(payload["fps"], "fps")
    if not math.isclose(fps, 50.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"{resolved.name}: fps must be 50, got {fps}.")

    joint_pos = payload["joint_pos"]
    joint_vel = payload["joint_vel"]
    body_pos_w = payload["body_pos_w"]
    body_quat_w = payload["body_quat_w"]
    body_lin_vel_w = payload["body_lin_vel_w"]
    body_ang_vel_w = payload["body_ang_vel_w"]

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 23:
        raise ValueError(
            f"{resolved.name}: joint_pos must have shape [T,23], got {joint_pos.shape}."
        )
    num_frames = int(joint_pos.shape[0])
    if num_frames <= 0:
        raise ValueError(f"{resolved.name}: trajectory must contain at least one frame.")
    if joint_vel.shape != (num_frames, 23):
        raise ValueError(
            f"{resolved.name}: joint_vel must have shape [{num_frames},23], "
            f"got {joint_vel.shape}."
        )
    expected_body_shape = (num_frames, 24, 3)
    if body_pos_w.shape != expected_body_shape:
        raise ValueError(
            f"{resolved.name}: body_pos_w must have shape {expected_body_shape}, "
            f"got {body_pos_w.shape}."
        )
    if body_quat_w.shape != (num_frames, 24, 4):
        raise ValueError(
            f"{resolved.name}: body_quat_w must have shape [{num_frames},24,4], "
            f"got {body_quat_w.shape}."
        )
    for key, array in (
        ("body_lin_vel_w", body_lin_vel_w),
        ("body_ang_vel_w", body_ang_vel_w),
    ):
        if array.shape != expected_body_shape:
            raise ValueError(
                f"{resolved.name}: {key} must have shape {expected_body_shape}, "
                f"got {array.shape}."
            )

    for key in CORE_ARRAY_KEYS:
        if not np.issubdtype(payload[key].dtype, np.number):
            raise ValueError(f"{resolved.name}: {key} must be numeric.")
        if not np.isfinite(payload[key]).all():
            raise ValueError(f"{resolved.name}: {key} contains NaN or Inf.")

    quaternion_norm = np.linalg.norm(body_quat_w.astype(np.float64), axis=-1)
    max_quaternion_error = float(np.max(np.abs(quaternion_norm - 1.0)))
    if max_quaternion_error > 1.0e-3:
        raise ValueError(
            f"{resolved.name}: quaternion norm max error "
            f"{max_quaternion_error:.6g} exceeds 1e-3."
        )

    joint_names = _string_tuple(payload["joint_names"], "joint_names")
    body_names = _string_tuple(payload["body_names"], "body_names")
    if len(joint_names) != 23:
        raise ValueError(f"{resolved.name}: expected 23 joint_names, got {len(joint_names)}.")
    if len(body_names) != 24:
        raise ValueError(f"{resolved.name}: expected 24 body_names, got {len(body_names)}.")

    root_body_index = _scalar_int(payload["root_body_index"], "root_body_index")
    root_body_name = _scalar_string(payload["root_body_name"], "root_body_name")
    if not 0 <= root_body_index < len(body_names):
        raise ValueError(
            f"{resolved.name}: root_body_index={root_body_index} is out of range."
        )
    if root_body_name != "pelvis" or body_names[root_body_index] != "pelvis":
        raise ValueError(
            f"{resolved.name}: root must resolve to pelvis, got index={root_body_index}, "
            f"metadata={root_body_name!r}, body={body_names[root_body_index]!r}."
        )
    if "torso_link" not in body_names:
        raise ValueError(f"{resolved.name}: body_names is missing torso_link.")

    quaternion_convention = _scalar_string(
        payload["quaternion_convention"], "quaternion_convention"
    )
    velocity_frame = _scalar_string(payload["velocity_frame"], "velocity_frame")
    body_state_frame = _scalar_string(payload["body_state_frame"], "body_state_frame")
    if quaternion_convention.upper() != "WXYZ":
        raise ValueError(
            f"{resolved.name}: quaternion_convention must be WXYZ, "
            f"got {quaternion_convention!r}."
        )
    if velocity_frame.lower() != "world":
        raise ValueError(
            f"{resolved.name}: velocity_frame must be world, got {velocity_frame!r}."
        )
    if body_state_frame.lower() != "link_world":
        raise ValueError(
            f"{resolved.name}: body_state_frame must be link_world, "
            f"got {body_state_frame!r}."
        )

    return LoadedSource(
        path=resolved,
        payload=payload,
        fps=fps,
        num_frames=num_frames,
        joint_names=joint_names,
        body_names=body_names,
        root_body_index=root_body_index,
        torso_body_index=body_names.index("torso_link"),
    )


def find_continuous_runs(mask: np.ndarray, min_frames: int) -> tuple[FallenRun, ...]:
    """Return half-open true runs whose length is at least ``min_frames``."""

    if min_frames <= 0:
        raise ValueError("min_frames must be positive.")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {mask.shape}.")
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return tuple(
        FallenRun(int(start), int(end))
        for start, end in zip(starts, ends)
        if int(end - start) >= min_frames
    )


def torso_up_z_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Match Stage-2's torso local +Z projected onto world +Z definition."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"quaternion must have shape [T,4], got {quaternion.shape}.")
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    _, x, y, _ = quaternion.T
    return 1.0 - 2.0 * (x * x + y * y)


def compute_frame_diagnostics(source: LoadedSource) -> FrameDiagnostics:
    root_index = source.root_body_index
    torso_index = source.torso_body_index
    payload = source.payload
    return FrameDiagnostics(
        pelvis_height=payload["body_pos_w"][:, root_index, 2].astype(np.float64),
        torso_up_z=torso_up_z_wxyz(payload["body_quat_w"][:, torso_index]),
        root_lin_speed=np.linalg.norm(
            payload["body_lin_vel_w"][:, root_index].astype(np.float64), axis=1
        ),
        root_ang_speed=np.linalg.norm(
            payload["body_ang_vel_w"][:, root_index].astype(np.float64), axis=1
        ),
        mean_joint_speed=np.mean(
            np.abs(payload["joint_vel"].astype(np.float64)), axis=1
        ),
    )


def _make_candidate(
    source: LoadedSource,
    diagnostics: FrameDiagnostics,
    run: FallenRun,
    frame: int,
) -> ResetCandidate:
    return ResetCandidate(
        source=source,
        frame=frame,
        run=run,
        pelvis_height=float(diagnostics.pelvis_height[frame]),
        torso_up_z=float(diagnostics.torso_up_z[frame]),
        root_lin_speed=float(diagnostics.root_lin_speed[frame]),
        root_ang_speed=float(diagnostics.root_ang_speed[frame]),
        mean_joint_speed=float(diagnostics.mean_joint_speed[frame]),
    )


def select_reset_states(source: LoadedSource, config: SelectionConfig) -> SourceSelection:
    """Detect stable fallen states and thin candidates independently per run."""

    config.validate()
    diagnostics = compute_frame_diagnostics(source)
    fallen_mask = (
        (diagnostics.pelvis_height <= config.fallen_height)
        & (diagnostics.torso_up_z <= config.fallen_up_z)
    )
    min_fallen_frames = max(1, int(math.ceil(config.min_fallen_duration_s * source.fps)))
    fallen_runs = find_continuous_runs(fallen_mask, min_fallen_frames)

    stable_window_frames = max(1, int(round(config.stable_window_s * source.fps)))
    left_window_frames = stable_window_frames // 2
    right_window_frames = stable_window_frames - left_window_frames

    stable_candidates: list[ResetCandidate] = []
    velocity_candidates: list[ResetCandidate] = []
    for run in fallen_runs:
        first_frame = run.start + left_window_frames
        last_frame = run.end - right_window_frames
        for frame in range(first_frame, last_frame + 1):
            window_start = frame - left_window_frames
            window_end = frame + right_window_frames
            if not np.all(fallen_mask[window_start:window_end]):
                continue
            candidate = _make_candidate(source, diagnostics, run, frame)
            stable_candidates.append(candidate)
            if (
                candidate.root_lin_speed <= config.max_root_lin_speed
                and candidate.root_ang_speed <= config.max_root_ang_speed
                and candidate.mean_joint_speed <= config.max_mean_joint_speed
            ):
                velocity_candidates.append(candidate)

    min_interval_frames = max(1, int(round(config.min_sample_interval_s * source.fps)))
    saved_candidates: list[ResetCandidate] = []
    for run in fallen_runs:
        run_candidates = [candidate for candidate in velocity_candidates if candidate.run == run]
        last_saved_frame: int | None = None
        for candidate in run_candidates:
            if (
                last_saved_frame is None
                or candidate.frame - last_saved_frame >= min_interval_frames
            ):
                saved_candidates.append(candidate)
                last_saved_frame = candidate.frame

    return SourceSelection(
        source=source,
        fallen_mask=fallen_mask,
        fallen_runs=fallen_runs,
        diagnostics=diagnostics,
        stable_candidates=tuple(stable_candidates),
        candidates_before_thinning=tuple(velocity_candidates),
        saved_candidates=tuple(saved_candidates),
    )


def _metadata_signature(source: LoadedSource) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return source.joint_names, source.body_names


def _output_filename(global_index: int, candidate: ResetCandidate) -> str:
    stem = candidate.source.path.stem.removesuffix("_whole_body")
    return (
        f"reset_state_{global_index:04d}_{stem}_f{candidate.frame:05d}_"
        f"t{candidate.source_time_s:06.2f}s.npz"
    )


def _single_state_payload(candidate: ResetCandidate) -> dict[str, np.ndarray]:
    source = candidate.source
    frame = candidate.frame
    payload: dict[str, np.ndarray] = {
        "fps": np.asarray(source.payload["fps"]).copy(),
    }
    for key in CORE_ARRAY_KEYS:
        payload[key] = np.asarray(source.payload[key][frame : frame + 1]).copy()
    for key in METADATA_KEYS:
        payload[key] = np.asarray(source.payload[key]).copy()

    payload.update(
        {
            "source_file": np.asarray(source.path.name, dtype=np.str_),
            "source_frame": np.asarray(frame, dtype=np.int64),
            "source_time_s": np.asarray(candidate.source_time_s, dtype=np.float64),
            "pelvis_height": np.asarray(candidate.pelvis_height, dtype=np.float64),
            "torso_up_z": np.asarray(candidate.torso_up_z, dtype=np.float64),
            "root_lin_speed": np.asarray(candidate.root_lin_speed, dtype=np.float64),
            "root_ang_speed": np.asarray(candidate.root_ang_speed, dtype=np.float64),
            "mean_joint_speed": np.asarray(candidate.mean_joint_speed, dtype=np.float64),
            "fallen_run_start_frame": np.asarray(candidate.run.start, dtype=np.int64),
            "fallen_run_end_frame": np.asarray(candidate.run.end - 1, dtype=np.int64),
        }
    )
    return payload


def _csv_row(saved_state: SavedResetState) -> dict[str, str | int | float]:
    candidate = saved_state.candidate
    return {
        "output_file": saved_state.output_path.name,
        "source_file": candidate.source.path.name,
        "source_frame": candidate.frame,
        "source_time_s": f"{candidate.source_time_s:.9f}",
        "fallen_run_start_frame": candidate.run.start,
        "fallen_run_end_frame": candidate.run.end - 1,
        "pelvis_height": f"{candidate.pelvis_height:.9f}",
        "torso_up_z": f"{candidate.torso_up_z:.9f}",
        "root_lin_speed": f"{candidate.root_lin_speed:.9f}",
        "root_ang_speed": f"{candidate.root_ang_speed:.9f}",
        "mean_joint_speed": f"{candidate.mean_joint_speed:.9f}",
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {name: math.nan for name in ("min", "median", "p90", "p95", "max")}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _print_distribution(label: str, metrics: dict[str, np.ndarray]) -> None:
    sample_count = next(iter(metrics.values())).size if metrics else 0
    print(f"  {label} (n={sample_count})")
    for name in DIAGNOSTIC_NAMES:
        stats = _distribution(metrics[name])
        print(
            f"    {name:20s} min={stats['min']:.6f} "
            f"median={stats['median']:.6f} p90={stats['p90']:.6f} "
            f"p95={stats['p95']:.6f} max={stats['max']:.6f}"
        )


def _selection_metrics(
    selection: SourceSelection, indices: np.ndarray
) -> dict[str, np.ndarray]:
    mapping = selection.diagnostics.as_mapping()
    return {name: mapping[name][indices] for name in DIAGNOSTIC_NAMES}


def _validate_source_metadata_order(sources: Sequence[LoadedSource]) -> None:
    reference = _metadata_signature(sources[0])
    for source in sources[1:]:
        if _metadata_signature(source) != reference:
            raise ValueError(
                f"{source.path.name}: joint_names/body_names order differs from "
                f"{sources[0].path.name}."
            )


def build_reset_dataset(
    input_paths: Sequence[Path],
    output_dir: Path,
    config: SelectionConfig,
) -> BuildResult:
    """Validate all sources, select states, and write one compressed NPZ per state."""

    config.validate()
    if not input_paths:
        raise ValueError("At least one input NPZ is required.")
    sources = [load_source_npz(path) for path in input_paths]
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("Duplicate input NPZ paths are not allowed.")
    _validate_source_metadata_order(sources)

    selections = tuple(select_reset_states(source, config) for source in sources)
    output_dir = Path(output_dir).expanduser().resolve()
    csv_path = output_dir / "reset_selection.csv"
    existing_outputs = list(output_dir.glob("reset_state_*.npz")) if output_dir.exists() else []
    if existing_outputs or csv_path.exists():
        raise FileExistsError(
            f"Output directory already contains a reset dataset: {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_states: list[SavedResetState] = []
    global_index = 0
    for selection in selections:
        for candidate in selection.saved_candidates:
            global_index += 1
            output_path = output_dir / _output_filename(global_index, candidate)
            np.savez_compressed(output_path, **_single_state_payload(candidate))
            saved_states.append(SavedResetState(output_path, candidate))

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(saved_state) for saved_state in saved_states)

    for selection in selections:
        saved_indices = np.asarray(
            [candidate.frame for candidate in selection.saved_candidates], dtype=np.int64
        )
        fallen_indices = np.flatnonzero(selection.fallen_mask)
        print(f"\nSOURCE {selection.source.path.name}")
        print(
            f"  total frames={selection.source.num_frames}, "
            f"fallen runs={len(selection.fallen_runs)}, "
            f"candidate frames before thinning="
            f"{len(selection.candidates_before_thinning)}, "
            f"saved reset states={len(selection.saved_candidates)}"
        )
        _print_distribution(
            "BEFORE: all basic fallen frames",
            _selection_metrics(selection, fallen_indices),
        )
        _print_distribution(
            "AFTER: saved reset states",
            _selection_metrics(selection, saved_indices),
        )

    total_candidates = sum(
        len(selection.candidates_before_thinning) for selection in selections
    )
    print("\n" + "=" * 80)
    print(f"TOTAL candidate frames before thinning: {total_candidates}")
    print(f"TOTAL saved reset states: {len(saved_states)}")
    print(f"TOTAL removed by temporal thinning: {total_candidates - len(saved_states)}")
    print(f"Output directory: {output_dir}")
    print(f"Selection CSV: {csv_path}")

    return BuildResult(
        output_dir=output_dir,
        csv_path=csv_path,
        selections=selections,
        saved_states=tuple(saved_states),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build stable fallen G1 Recovery reset states from Stage-1 NPZ files."
    )
    parser.add_argument("--input_npz", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fallen_height", type=float, default=0.35)
    parser.add_argument("--fallen_up_z", type=float, default=0.80)
    parser.add_argument("--min_fallen_duration_s", type=float, default=0.50)
    parser.add_argument("--stable_window_s", type=float, default=0.20)
    parser.add_argument("--max_root_lin_speed", type=float, default=0.50)
    parser.add_argument("--max_root_ang_speed", type=float, default=1.50)
    parser.add_argument("--max_mean_joint_speed", type=float, default=1.50)
    parser.add_argument("--min_sample_interval_s", type=float, default=0.10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    unexpected = sorted(
        {path.name for path in args.input_npz} - set(ALLOWED_SOURCE_FILENAMES)
    )
    if unexpected:
        parser.error(
            "Stage 4.2a only permits the six approved whole-body sources; "
            f"unexpected files: {unexpected}."
        )
    config = SelectionConfig(
        fallen_height=args.fallen_height,
        fallen_up_z=args.fallen_up_z,
        min_fallen_duration_s=args.min_fallen_duration_s,
        stable_window_s=args.stable_window_s,
        max_root_lin_speed=args.max_root_lin_speed,
        max_root_ang_speed=args.max_root_ang_speed,
        max_mean_joint_speed=args.max_mean_joint_speed,
        min_sample_interval_s=args.min_sample_interval_s,
    )
    print("Selection thresholds:")
    for name, value in vars(config).items():
        print(f"  {name}={value}")
    build_reset_dataset(args.input_npz, args.output_dir, config)


if __name__ == "__main__":
    main()
