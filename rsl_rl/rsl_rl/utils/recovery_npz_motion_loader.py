"""Whole-body expert-motion loader for the standalone G1 recovery task.

Each NPZ file remains an independent trajectory.  This module deliberately
loads raw whole-body state only; construction of AMP observations belongs to a
later stage of the recovery task.

提供专家数据
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


_STATE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class RecoveryMotionClip:
    """One independently sampled recovery motion."""

    name: str
    path: Path
    fps: float
    frame_duration: float
    num_frames: int
    duration: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor


class RecoveryNPZMotionLoader:
    """Load, validate, sample, and interpolate whole-body Recovery NPZ clips."""

    REQUIRED_FIELDS = frozenset(
        {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
            "root_body_index",
            "root_body_name",
            "quaternion_convention",
            "velocity_frame",
            "body_state_frame",
        }
    )
    SAMPLING_MODES = frozenset({"uniform_frame", "uniform_motion"})

    def __init__(
        self,
        motion_dir=None,
        motion_files=None,
        device="cpu",
        expected_fps=50.0,
        sampling="uniform_frame",
        quaternion_atol=1e-3,
    ):
        if (motion_dir is None) == (motion_files is None):
            raise ValueError("Exactly one of motion_dir or motion_files must be provided.")
        if sampling not in self.SAMPLING_MODES:
            raise ValueError(
                f"sampling must be one of {sorted(self.SAMPLING_MODES)}, got {sampling!r}."
            )

        self.device = torch.device(device)
        self.sampling = sampling
        self.expected_fps = self._validate_expected_fps(expected_fps)
        self.quaternion_atol = self._validate_quaternion_atol(quaternion_atol)
        self.motion_files = self._resolve_motion_files(motion_dir, motion_files)

        self.clips: list[RecoveryMotionClip] = []
        canonical_metadata = None
        for path in self.motion_files:
            clip, metadata = self._load_clip(path)
            if canonical_metadata is None:
                canonical_metadata = metadata
            elif metadata != canonical_metadata:
                self._raise_metadata_mismatch(path, canonical_metadata, metadata)
            self.clips.append(clip)

        # ``motions`` is a descriptive alias useful to downstream callers.
        self.motions = self.clips
        joint_names, body_names, root_body_index, root_body_name = canonical_metadata
        self.joint_names = joint_names
        self.body_names = body_names
        self.root_body_index = root_body_index
        self.root_body_name = root_body_name
        self.num_joints = len(self.joint_names)
        self.num_bodies = len(self.body_names)
        self.num_motions = len(self.clips)

        self._joint_name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        self._body_name_to_index = {name: index for index, name in enumerate(self.body_names)}

        self.total_frames = sum(clip.num_frames for clip in self.clips)
        self.total_duration = sum(clip.duration for clip in self.clips)
        self.motion_num_frames = torch.tensor(
            [clip.num_frames for clip in self.clips], dtype=torch.long, device=self.device
        )
        self.motion_frame_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=self.device),
                torch.cumsum(self.motion_num_frames, dim=0)[:-1],
            )
        )
        self.motion_frame_ends = torch.cumsum(self.motion_num_frames, dim=0)
        self.motion_durations = torch.tensor(
            [clip.duration for clip in self.clips], dtype=torch.float64, device=self.device
        )
        self.motion_frame_durations = torch.tensor(
            [clip.frame_duration for clip in self.clips],
            dtype=torch.float64,
            device=self.device,
        )

        fps_values = {clip.fps for clip in self.clips}
        self.fps = self.clips[0].fps if len(fps_values) == 1 else tuple(sorted(fps_values))

        if sampling == "uniform_frame":
            weights = torch.tensor(
                [max(clip.num_frames - 1, 1) for clip in self.clips],
                dtype=torch.float64,
                device=self.device,
            )
        else:
            weights = torch.ones(self.num_motions, dtype=torch.float64, device=self.device)
        self.motion_sampling_weights = weights / weights.sum()

    @staticmethod
    def _validate_expected_fps(expected_fps):
        if expected_fps is None:
            return None
        if isinstance(expected_fps, bool):
            raise ValueError(f"expected_fps must be positive and finite, got {expected_fps!r}.")
        try:
            value = float(expected_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"expected_fps must be positive and finite, got {expected_fps!r}."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"expected_fps must be positive and finite, got {expected_fps!r}.")
        return value

    @staticmethod
    def _validate_quaternion_atol(quaternion_atol):
        if isinstance(quaternion_atol, bool):
            raise ValueError(
                f"quaternion_atol must be non-negative and finite, got {quaternion_atol!r}."
            )
        try:
            value = float(quaternion_atol)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"quaternion_atol must be non-negative and finite, got {quaternion_atol!r}."
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"quaternion_atol must be non-negative and finite, got {quaternion_atol!r}."
            )
        return value

    @staticmethod
    def _resolve_motion_files(motion_dir, motion_files) -> list[Path]:
        if motion_dir is not None:
            directory = Path(motion_dir).expanduser()
            if not directory.is_dir():
                raise ValueError(f"motion_dir is not a directory: {directory}.")
            paths = sorted(directory.glob("*.npz"), key=lambda path: path.name)
        else:
            if isinstance(motion_files, (str, bytes, Path)):
                raise ValueError("motion_files must be an iterable of NPZ paths, not one path.")
            try:
                paths = [Path(path).expanduser() for path in motion_files]
            except TypeError as exc:
                raise ValueError("motion_files must be an iterable of NPZ paths.") from exc

        if not paths:
            raise ValueError("No Recovery NPZ motion files were provided or found.")
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Recovery motion file does not exist: {path}.")
        return paths

    @staticmethod
    def _scalar(array: np.ndarray, field: str, path: Path):
        value = np.asarray(array)
        if value.size != 1:
            raise ValueError(f"{field} must contain exactly one value in {path}, got {value.shape}.")
        return value.reshape(-1)[0].item()

    @classmethod
    def _string_scalar(cls, array: np.ndarray, field: str, path: Path) -> str:
        value = cls._scalar(array, field, path)
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{field} is not valid UTF-8 in {path}.") from exc
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string in {path}, got {type(value).__name__}.")
        return value.strip()

    @staticmethod
    def _names(array: np.ndarray, field: str, path: Path) -> Tuple[str, ...]:
        values = np.asarray(array)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{field} must be a non-empty 1-D array in {path}, got {values.shape}.")

        names = []
        for value in values.tolist():
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{field} contains invalid UTF-8 in {path}.") from exc
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must contain non-empty strings in {path}.")
            names.append(value)
        if len(set(names)) != len(names):
            raise ValueError(f"{field} contains duplicate names in {path}.")
        return tuple(names)

    @staticmethod
    def _numeric_array(data, field: str, path: Path) -> np.ndarray:
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise ValueError(f"{field} must be numeric in {path}, got dtype {array.dtype}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{field} contains NaN or Inf values in {path}.")
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{field} is not finite after float32 conversion in {path}.")
        return array

    def _load_clip(self, path: Path):
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read Recovery NPZ motion {path}: {exc}") from exc

        with archive:
            missing = sorted(self.REQUIRED_FIELDS.difference(archive.files))
            if missing:
                raise ValueError(f"Missing required fields {missing} in {path}.")

            fps_array = np.asarray(archive["fps"])
            fps_value = self._scalar(fps_array, "fps", path)
            if (
                isinstance(fps_value, (bool, np.bool_))
                or not np.issubdtype(fps_array.dtype, np.number)
                or np.issubdtype(fps_array.dtype, np.complexfloating)
            ):
                raise ValueError(f"fps must be numeric in {path}, got {fps_value!r}.")
            fps = float(fps_value)
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"fps must be positive and finite in {path}, got {fps!r}.")
            if self.expected_fps is not None and fps != self.expected_fps:
                raise ValueError(
                    f"fps must equal expected_fps={self.expected_fps} in {path}, got {fps}."
                )

            joint_names = self._names(archive["joint_names"], "joint_names", path)
            body_names = self._names(archive["body_names"], "body_names", path)
            arrays = {
                field: self._numeric_array(archive[field], field, path) for field in _STATE_FIELDS
            }
            self._validate_shapes(arrays, joint_names, body_names, path)
            self._validate_quaternions(arrays["body_quat_w"], path)
            self._validate_frame_metadata(archive, path)
            root_body_index, root_body_name = self._resolve_root_metadata(
                archive, body_names, path
            )

        num_frames = int(arrays["joint_pos"].shape[0])
        frame_duration = 1.0 / fps
        duration = (num_frames - 1) / fps
        tensors = {
            field: torch.as_tensor(array, dtype=torch.float32, device=self.device)
            for field, array in arrays.items()
        }
        clip = RecoveryMotionClip(
            name=path.stem,
            path=path,
            fps=fps,
            frame_duration=frame_duration,
            num_frames=num_frames,
            duration=duration,
            **tensors,
        )
        metadata = (joint_names, body_names, root_body_index, root_body_name)
        return clip, metadata

    @staticmethod
    def _validate_shapes(arrays, joint_names, body_names, path: Path) -> None:
        joint_pos = arrays["joint_pos"]
        joint_vel = arrays["joint_vel"]
        body_pos = arrays["body_pos_w"]
        body_quat = arrays["body_quat_w"]
        body_lin_vel = arrays["body_lin_vel_w"]
        body_ang_vel = arrays["body_ang_vel_w"]

        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must have shape [T, J] in {path}, got {joint_pos.shape}.")
        if joint_pos.shape[0] < 1:
            raise ValueError(f"Recovery motion must contain at least one frame: {path}.")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"joint_vel shape must equal joint_pos shape in {path}: "
                f"{joint_vel.shape} != {joint_pos.shape}."
            )

        expected_body_shapes = {
            "body_pos_w": (3, body_pos),
            "body_quat_w": (4, body_quat),
            "body_lin_vel_w": (3, body_lin_vel),
            "body_ang_vel_w": (3, body_ang_vel),
        }
        for field, (width, array) in expected_body_shapes.items():
            if array.ndim != 3 or array.shape[2] != width:
                raise ValueError(
                    f"{field} must have shape [T, B, {width}] in {path}, got {array.shape}."
                )

        num_frames = joint_pos.shape[0]
        frame_counts = {field: array.shape[0] for field, array in arrays.items()}
        if any(count != num_frames for count in frame_counts.values()):
            raise ValueError(
                f"All motion arrays must have the same frame count in {path}, got {frame_counts}."
            )

        body_shape = body_pos.shape[:2]
        for field in ("body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if arrays[field].shape[:2] != body_shape:
                raise ValueError(
                    f"All body arrays must share [T, B] in {path}: "
                    f"{field} has {arrays[field].shape[:2]}, expected {body_shape}."
                )
        if joint_pos.shape[1] != len(joint_names):
            raise ValueError(
                f"joint_pos has {joint_pos.shape[1]} joints but joint_names has "
                f"{len(joint_names)} entries in {path}."
            )
        if body_pos.shape[1] != len(body_names):
            raise ValueError(
                f"body_pos_w has {body_pos.shape[1]} bodies but body_names has "
                f"{len(body_names)} entries in {path}."
            )

    def _validate_quaternions(self, body_quat: np.ndarray, path: Path) -> None:
        norms = np.linalg.norm(body_quat, axis=-1)
        max_error = float(np.max(np.abs(norms - 1.0)))
        if max_error > self.quaternion_atol:
            raise ValueError(
                f"body_quat_w quaternion norm error exceeds quaternion_atol="
                f"{self.quaternion_atol} in {path}: max error {max_error}."
            )

    @classmethod
    def _validate_frame_metadata(cls, archive, path: Path) -> None:
        if "quaternion_convention" in archive.files:
            value = cls._string_scalar(
                archive["quaternion_convention"], "quaternion_convention", path
            )
            if value.lower() != "wxyz":
                raise ValueError(
                    f"quaternion_convention must be 'WXYZ' in {path}, got {value!r}."
                )

        expected = {"velocity_frame": "world", "body_state_frame": "link_world"}
        for field, expected_value in expected.items():
            if field not in archive.files:
                continue
            value = cls._string_scalar(archive[field], field, path)
            if value != expected_value:
                raise ValueError(
                    f"{field} must be {expected_value!r} in {path}, got {value!r}."
                )

    @classmethod
    def _resolve_root_metadata(cls, archive, body_names, path: Path):
        has_index = "root_body_index" in archive.files
        has_name = "root_body_name" in archive.files

        root_index = None
        root_name = None
        if has_index:
            value = cls._scalar(archive["root_body_index"], "root_body_index", path)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"root_body_index must be an integer in {path}, got {value!r}.")
            root_index = int(value)
            if root_index < 0 or root_index >= len(body_names):
                raise ValueError(
                    f"root_body_index={root_index} is outside body_names in {path}."
                )
        if has_name:
            root_name = cls._string_scalar(archive["root_body_name"], "root_body_name", path)
            if root_name not in body_names:
                raise ValueError(f"root_body_name={root_name!r} is not in body_names in {path}.")

        if root_index is None and root_name is None:
            if "pelvis" not in body_names:
                raise ValueError(
                    f"Root metadata is absent and body_names does not contain 'pelvis' in {path}."
                )
            root_name = "pelvis"
            root_index = body_names.index(root_name)
        elif root_index is None:
            root_index = body_names.index(root_name)
        elif root_name is None:
            root_name = body_names[root_index]

        if body_names[root_index] != root_name:
            raise ValueError(
                f"root_body_index={root_index} resolves to {body_names[root_index]!r}, not "
                f"root_body_name={root_name!r} in {path}."
            )
        return root_index, root_name

    @staticmethod
    def _raise_metadata_mismatch(path, canonical, current):
        field_names = ("joint_names", "body_names", "root_body_index", "root_body_name")
        mismatches = [
            field for field, expected, actual in zip(field_names, canonical, current) if expected != actual
        ]
        raise ValueError(
            f"Dataset metadata mismatch for {mismatches} in {path}; all Recovery NPZ files "
            "must use exactly the same joint/body names, ordering, and root body."
        )

    def joint_index(self, name: str) -> int:
        """Return the canonical joint index for ``name``."""

        try:
            return self._joint_name_to_index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown joint name: {name!r}.") from exc

    def body_index(self, name: str) -> int:
        """Return the canonical body index for ``name``."""

        try:
            return self._body_name_to_index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown body name: {name!r}.") from exc

    def body_indices(self, names: Iterable[str]) -> torch.Tensor:
        """Return canonical body indices on the loader device."""

        if isinstance(names, (str, bytes)):
            raise TypeError("body_indices expects an iterable of body names, not one string.")
        return torch.tensor(
            [self.body_index(name) for name in names], dtype=torch.long, device=self.device
        )

    @staticmethod
    def _validate_batch_size(batch_size) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}.")
        value = int(batch_size)
        if value < 0:
            raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}.")
        return value

    def sample_motion_ids(self, batch_size) -> torch.Tensor:
        """Sample motion IDs according to the configured dataset weighting."""

        batch_size = self._validate_batch_size(batch_size)
        if batch_size == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.multinomial(
            self.motion_sampling_weights, num_samples=batch_size, replacement=True
        )

    def _motion_ids_tensor(self, motion_ids) -> torch.Tensor:
        ids = torch.as_tensor(motion_ids, device=self.device)
        if ids.ndim != 1:
            raise ValueError(f"motion_ids must have shape [B], got {tuple(ids.shape)}.")
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError("motion_ids must contain integers.")
        ids = ids.to(dtype=torch.long)
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_motions)):
            raise ValueError(f"motion_ids must be between 0 and {self.num_motions - 1}.")
        return ids

    def _frame_ids_tensor(self, frame_ids, motion_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(frame_ids, device=self.device)
        batch_size = motion_ids.shape[0]
        if ids.ndim != 1 or ids.shape[0] != batch_size:
            raise ValueError(
                f"frame_ids must have shape [{batch_size}], got {tuple(ids.shape)}."
            )
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError("frame_ids must contain integers.")
        ids = ids.to(dtype=torch.long)
        if ids.numel():
            selected_num_frames = self.motion_num_frames[motion_ids]
            invalid = (ids < 0) | (ids >= selected_num_frames)
            if torch.any(invalid):
                first_invalid = int(torch.nonzero(invalid, as_tuple=False)[0].item())
                raise IndexError(
                    "frame_ids contains an out-of-range index at batch position "
                    f"{first_invalid}: motion_id={int(motion_ids[first_invalid].item())}, "
                    f"frame_id={int(ids[first_invalid].item())}, valid range="
                    f"[0, {int(selected_num_frames[first_invalid].item()) - 1}]."
                )
        return ids

    def get_frame_batch(self, motion_ids, frame_ids) -> Dict[str, torch.Tensor]:
        """Read exact integer frames without continuous-time interpolation."""

        motion_ids = self._motion_ids_tensor(motion_ids)
        frame_ids = self._frame_ids_tensor(frame_ids, motion_ids)
        batch_size = motion_ids.shape[0]
        result = {
            "motion_ids": motion_ids.clone(),
            "frame_ids": frame_ids.clone(),
            "joint_pos": torch.empty(
                (batch_size, self.num_joints), dtype=torch.float32, device=self.device
            ),
            "joint_vel": torch.empty(
                (batch_size, self.num_joints), dtype=torch.float32, device=self.device
            ),
            "body_pos_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
            "body_quat_w": torch.empty(
                (batch_size, self.num_bodies, 4), dtype=torch.float32, device=self.device
            ),
            "body_lin_vel_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
            "body_ang_vel_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
        }

        for motion_id, clip in enumerate(self.clips):
            mask = motion_ids == motion_id
            if not torch.any(mask):
                continue
            selected_frames = frame_ids[mask]
            for field in _STATE_FIELDS:
                result[field][mask] = getattr(clip, field)[selected_frames]
        return result

    def sample_frame_batch(self, batch_size) -> Dict[str, torch.Tensor]:
        """Sample every stored trajectory frame with equal probability."""

        batch_size = self._validate_batch_size(batch_size)
        flat_frame_ids = torch.randint(
            self.total_frames,
            (batch_size,),
            dtype=torch.long,
            device=self.device,
        )
        motion_ids = torch.searchsorted(
            self.motion_frame_ends, flat_frame_ids, right=True
        )
        frame_ids = flat_frame_ids - self.motion_frame_offsets[motion_ids]
        return self.get_frame_batch(motion_ids, frame_ids)

    @staticmethod
    def _validate_transition_dt(transition_dt) -> float:
        if isinstance(transition_dt, bool):
            raise ValueError(f"transition_dt must be non-negative and finite, got {transition_dt!r}.")
        try:
            value = float(transition_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"transition_dt must be non-negative and finite, got {transition_dt!r}."
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"transition_dt must be non-negative and finite, got {transition_dt!r}.")
        return value

    def sample_times(self, motion_ids, transition_dt=0.0) -> torch.Tensor:
        """Sample times whose optional transition endpoint stays in each source clip."""

        ids = self._motion_ids_tensor(motion_ids)
        transition_dt = self._validate_transition_dt(transition_dt)
        latest_times = self.motion_durations[ids] - transition_dt
        if torch.any(latest_times < 0.0):
            shortest = float(self.motion_durations[ids].min().item())
            raise ValueError(
                f"transition_dt={transition_dt} exceeds a selected motion duration "
                f"(shortest selected duration: {shortest})."
            )
        times = torch.rand(ids.shape, dtype=torch.float64, device=self.device) * latest_times
        return torch.minimum(times, latest_times)

    @staticmethod
    def _normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(quaternions.dtype).eps
        return quaternions / torch.linalg.vector_norm(
            quaternions, dim=-1, keepdim=True
        ).clamp_min(eps)

    @classmethod
    def _slerp(cls, start: torch.Tensor, end: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Shortest-path WXYZ quaternion SLERP with a normalized lerp fallback."""

        start = cls._normalize_quaternions(start)
        end = cls._normalize_quaternions(end)
        dot = torch.sum(start * end, dim=-1, keepdim=True)
        end = torch.where(dot < 0.0, -end, end)
        dot = torch.abs(dot).clamp(0.0, 1.0)

        linear = cls._normalize_quaternions(start + blend * (end - start))
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        denominator = sin_theta.clamp_min(torch.finfo(start.dtype).eps)
        spherical = (
            torch.sin((1.0 - blend) * theta) / denominator * start
            + torch.sin(blend * theta) / denominator * end
        )
        result = torch.where(dot > 0.9995, linear, spherical)
        return cls._normalize_quaternions(result)

    def _times_tensor(self, times_s, batch_size: int) -> torch.Tensor:
        times = torch.as_tensor(times_s, dtype=torch.float64, device=self.device)
        if times.ndim != 1 or times.shape[0] != batch_size:
            raise ValueError(
                f"times_s must have shape [{batch_size}], got {tuple(times.shape)}."
            )
        if not torch.isfinite(times).all():
            raise ValueError("times_s contains NaN or Inf values.")
        return times

    def get_state_at_time(self, motion_id, time_s) -> Dict[str, torch.Tensor]:
        """Interpolate one clip at one time, clamping time to the clip bounds."""

        if isinstance(motion_id, (bool, np.bool_)):
            raise ValueError(f"motion_id must be an integer, got {motion_id!r}.")
        if isinstance(motion_id, torch.Tensor):
            if motion_id.numel() != 1 or motion_id.dtype == torch.bool:
                raise ValueError(f"motion_id must be one integer, got {motion_id!r}.")
            if motion_id.is_floating_point() or motion_id.is_complex():
                raise ValueError(f"motion_id must be an integer, got {motion_id!r}.")
            motion_id = motion_id.item()
        if not isinstance(motion_id, (int, np.integer)):
            raise ValueError(f"motion_id must be an integer, got {motion_id!r}.")
        motion_id = int(motion_id)
        states = self.get_state_at_time_batch([motion_id], [time_s])
        return {field: value[0] for field, value in states.items()}

    def get_state_at_time_batch(self, motion_ids, times_s) -> Dict[str, torch.Tensor]:
        """Interpolate a batch of raw whole-body states without crossing clip boundaries."""

        ids = self._motion_ids_tensor(motion_ids)
        times = self._times_tensor(times_s, ids.shape[0])
        batch_size = ids.shape[0]
        result = {
            "joint_pos": torch.empty(
                (batch_size, self.num_joints), dtype=torch.float32, device=self.device
            ),
            "joint_vel": torch.empty(
                (batch_size, self.num_joints), dtype=torch.float32, device=self.device
            ),
            "body_pos_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
            "body_quat_w": torch.empty(
                (batch_size, self.num_bodies, 4), dtype=torch.float32, device=self.device
            ),
            "body_lin_vel_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
            "body_ang_vel_w": torch.empty(
                (batch_size, self.num_bodies, 3), dtype=torch.float32, device=self.device
            ),
        }

        for motion_id, clip in enumerate(self.clips):
            mask = ids == motion_id
            if not torch.any(mask):
                continue
            clip_times = torch.clamp(times[mask], min=0.0, max=clip.duration)
            frame_positions = clip_times * clip.fps
            index_low = torch.floor(frame_positions).to(dtype=torch.long)
            index_low = torch.clamp(index_low, max=clip.num_frames - 1)
            at_end = clip_times >= clip.duration
            index_low = torch.where(
                at_end, torch.full_like(index_low, clip.num_frames - 1), index_low
            )
            index_high = torch.clamp(index_low + 1, max=clip.num_frames - 1)
            blend = torch.where(
                index_high > index_low,
                frame_positions - index_low.to(dtype=frame_positions.dtype),
                torch.zeros_like(frame_positions),
            ).to(dtype=torch.float32)

            for field in _STATE_FIELDS:
                values = getattr(clip, field)
                start = values[index_low]
                end = values[index_high]
                expanded_blend = blend.reshape(
                    (blend.shape[0],) + (1,) * (start.ndim - 1)
                )
                if field == "body_quat_w":
                    interpolated = self._slerp(start, end, expanded_blend)
                else:
                    interpolated = torch.lerp(start, end, expanded_blend)
                result[field][mask] = interpolated
        return result

    def sample_state_batch(self, batch_size) -> Dict[str, torch.Tensor]:
        """Sample interpolated raw states from the expert dataset."""

        motion_ids = self.sample_motion_ids(batch_size)
        times = self.sample_times(motion_ids)
        return self.get_state_at_time_batch(motion_ids, times)

    def sample_transition_batch(self, batch_size, transition_dt):
        """Sample state pairs separated by ``transition_dt`` within each clip."""

        transition_dt = self._validate_transition_dt(transition_dt)
        motion_ids = self.sample_motion_ids(batch_size)
        times = self.sample_times(motion_ids, transition_dt=transition_dt)
        state = self.get_state_at_time_batch(motion_ids, times)
        next_state = self.get_state_at_time_batch(motion_ids, times + transition_dt)
        return state, next_state

    def summary(self) -> dict:
        """Return a compact dataset summary."""

        return {
            "num_motions": self.num_motions,
            "total_frames": self.total_frames,
            "total_duration": self.total_duration,
            "num_joints": self.num_joints,
            "num_bodies": self.num_bodies,
            "device": str(self.device),
            "sampling": self.sampling,
            "root_body": self.root_body_name,
            "fps": self.fps,
        }


__all__ = ["RecoveryMotionClip", "RecoveryNPZMotionLoader"]
