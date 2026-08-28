"""Preloaded physical reset states for the standalone G1 Recovery environment."""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch


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
OPTIONAL_STATE_METADATA_KEYS = (
    "source_file",
    "source_frame",
    "source_time_s",
    "pelvis_height",
    "torso_up_z",
    "root_lin_speed",
    "root_ang_speed",
    "mean_joint_speed",
)


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain a real numeric scalar.")
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
    if array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{name} must contain a string scalar.")
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
    if array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{name} must contain strings.")
    result = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty strings.")
    return tuple(result)


def _optional_scalar(data, key: str):
    if key not in data.files:
        return None
    array = np.asarray(data[key])
    if array.size != 1:
        raise ValueError(
            f"Optional reset metadata {key} must contain one scalar, "
            f"got {array.shape}."
        )
    item = array.reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


class RecoveryResetStateLoader:
    """Validate and preload single-state whole-body NPZ reset files."""

    def __init__(self, state_dir, device="cpu"):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.device = torch.device(device)
        if not self.state_dir.is_dir():
            raise FileNotFoundError(
                f"Recovery reset-state directory not found: {self.state_dir}. "
                "Run legged_lab/scripts/build_recovery_reset_dataset.py first."
            )

        self.state_files = tuple(sorted(self.state_dir.glob("*.npz")))
        if not self.state_files:
            raise FileNotFoundError(
                f"No single-state .npz files found in {self.state_dir}. "
                "Run legged_lab/scripts/build_recovery_reset_dataset.py first."
            )

        root_pos_w = []
        root_quat_w = []
        root_lin_vel_w = []
        root_ang_vel_w = []
        joint_pos = []
        joint_vel = []
        state_metadata = []

        reference_joint_names: tuple[str, ...] | None = None
        reference_body_names: tuple[str, ...] | None = None
        reference_root_body_index: int | None = None
        for state_file in self.state_files:
            state = self._load_and_validate_file(state_file)
            if reference_joint_names is None:
                reference_joint_names = state["joint_names"]
                reference_body_names = state["body_names"]
                reference_root_body_index = state["root_body_index"]
            else:
                if state["joint_names"] != reference_joint_names:
                    raise ValueError(
                        f"{state_file.name}: joint_names order differs from "
                        f"{self.state_files[0].name}."
                    )
                if state["body_names"] != reference_body_names:
                    raise ValueError(
                        f"{state_file.name}: body_names order differs from "
                        f"{self.state_files[0].name}."
                    )
                if state["root_body_index"] != reference_root_body_index:
                    raise ValueError(
                        f"{state_file.name}: root_body_index differs from "
                        f"{self.state_files[0].name}."
                    )

            root_index = state["root_body_index"]
            root_pos_w.append(state["body_pos_w"][0, root_index])
            root_quat_w.append(state["body_quat_w"][0, root_index])
            root_lin_vel_w.append(state["body_lin_vel_w"][0, root_index])
            root_ang_vel_w.append(state["body_ang_vel_w"][0, root_index])
            joint_pos.append(state["joint_pos"][0])
            joint_vel.append(state["joint_vel"][0])
            state_metadata.append(state["metadata"])

        assert reference_joint_names is not None
        assert reference_body_names is not None
        assert reference_root_body_index is not None
        self.joint_names = reference_joint_names
        self.body_names = reference_body_names
        self.root_body_index = reference_root_body_index
        self.root_body_name = "pelvis"
        self.num_states = len(self.state_files)
        self.state_metadata = tuple(
            MappingProxyType(metadata) for metadata in state_metadata
        )

        self.root_pos_w = self._to_device_tensor(root_pos_w)
        self.root_quat_w = self._to_device_tensor(root_quat_w)
        self.root_lin_vel_w = self._to_device_tensor(root_lin_vel_w)
        self.root_ang_vel_w = self._to_device_tensor(root_ang_vel_w)
        self.joint_pos = self._to_device_tensor(joint_pos)
        self.joint_vel = self._to_device_tensor(joint_vel)

    def _to_device_tensor(self, values) -> torch.Tensor:
        array = np.stack(values, axis=0)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @staticmethod
    def _load_and_validate_file(path: Path) -> dict[str, np.ndarray | tuple | int]:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                raise KeyError(f"{path.name}: missing required keys {missing}.")
            for key in data.files:
                array = np.asarray(data[key])
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(
                    array
                ).all():
                    raise ValueError(
                        f"{path.name}: numeric array {key} contains NaN or Inf."
                    )
            payload = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}

            fps = _scalar_float(payload["fps"], "fps")
            if not math.isclose(fps, 50.0, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError(f"{path.name}: fps must be 50, got {fps}.")

            expected_shapes = {
                "joint_pos": (1, 23),
                "joint_vel": (1, 23),
                "body_pos_w": (1, 24, 3),
                "body_quat_w": (1, 24, 4),
                "body_lin_vel_w": (1, 24, 3),
                "body_ang_vel_w": (1, 24, 3),
            }
            for key, expected_shape in expected_shapes.items():
                array = payload[key]
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path.name}: {key} must have shape {expected_shape}, "
                        f"got {array.shape}."
                    )
                if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(
                    array
                ):
                    raise ValueError(f"{path.name}: {key} must be real numeric.")
                if not np.isfinite(array).all():
                    raise ValueError(f"{path.name}: {key} contains NaN or Inf.")

            quaternion_norm = np.linalg.norm(
                payload["body_quat_w"].astype(np.float64), axis=-1
            )
            max_quaternion_error = float(
                np.max(np.abs(quaternion_norm - 1.0))
            )
            if max_quaternion_error > 1.0e-3:
                raise ValueError(
                    f"{path.name}: quaternion norm max error "
                    f"{max_quaternion_error:.6g} exceeds 1e-3."
                )

            joint_names = _string_tuple(payload["joint_names"], "joint_names")
            body_names = _string_tuple(payload["body_names"], "body_names")
            if len(joint_names) != 23:
                raise ValueError(
                    f"{path.name}: expected 23 joint_names, got {len(joint_names)}."
                )
            if len(body_names) != 24:
                raise ValueError(
                    f"{path.name}: expected 24 body_names, got {len(body_names)}."
                )

            root_body_index = _scalar_int(
                payload["root_body_index"], "root_body_index"
            )
            root_body_name = _scalar_string(
                payload["root_body_name"], "root_body_name"
            )
            if "pelvis" not in body_names:
                raise ValueError(f"{path.name}: body_names is missing pelvis.")
            pelvis_index = body_names.index("pelvis")
            if root_body_name != "pelvis" or root_body_index != pelvis_index:
                raise ValueError(
                    f"{path.name}: root must be pelvis at index {pelvis_index}, "
                    f"got name={root_body_name!r}, index={root_body_index}."
                )

            quaternion_convention = _scalar_string(
                payload["quaternion_convention"], "quaternion_convention"
            )
            velocity_frame = _scalar_string(
                payload["velocity_frame"], "velocity_frame"
            )
            body_state_frame = _scalar_string(
                payload["body_state_frame"], "body_state_frame"
            )
            if quaternion_convention.upper() != "WXYZ":
                raise ValueError(
                    f"{path.name}: quaternion_convention must be WXYZ, "
                    f"got {quaternion_convention!r}."
                )
            if velocity_frame.lower() != "world":
                raise ValueError(
                    f"{path.name}: velocity_frame must be world, "
                    f"got {velocity_frame!r}."
                )
            if body_state_frame.lower() != "link_world":
                raise ValueError(
                    f"{path.name}: body_state_frame must be link_world, "
                    f"got {body_state_frame!r}."
                )
            metadata = {
                key: _optional_scalar(data, key)
                for key in OPTIONAL_STATE_METADATA_KEYS
            }

        return {
            "joint_pos": payload["joint_pos"],
            "joint_vel": payload["joint_vel"],
            "body_pos_w": payload["body_pos_w"],
            "body_quat_w": payload["body_quat_w"],
            "body_lin_vel_w": payload["body_lin_vel_w"],
            "body_ang_vel_w": payload["body_ang_vel_w"],
            "joint_names": joint_names,
            "body_names": body_names,
            "root_body_index": root_body_index,
            "metadata": metadata,
        }

    @staticmethod
    def _validate_batch_size(batch_size) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}."
            )
        return batch_size

    def sample_state_ids(self, batch_size) -> torch.Tensor:
        batch_size = self._validate_batch_size(batch_size)
        return torch.randint(
            self.num_states,
            (batch_size,),
            dtype=torch.long,
            device=self.device,
        )

    def get_state_batch(self, state_ids) -> dict[str, torch.Tensor]:
        if isinstance(state_ids, torch.Tensor) and state_ids.dtype == torch.bool:
            raise ValueError("state_ids must contain integer indexes, not booleans.")
        state_ids = torch.as_tensor(
            state_ids, dtype=torch.long, device=self.device
        )
        if state_ids.ndim != 1:
            raise ValueError(
                f"state_ids must have shape [B], got {tuple(state_ids.shape)}."
            )
        if state_ids.numel() == 0:
            raise ValueError("state_ids must not be empty.")
        if torch.any(state_ids < 0) or torch.any(state_ids >= self.num_states):
            raise IndexError(
                f"state_ids must be in [0, {self.num_states - 1}]."
            )
        return {
            "state_ids": state_ids,
            "root_pos_w": self.root_pos_w[state_ids],
            "root_quat_w": self.root_quat_w[state_ids],
            "root_lin_vel_w": self.root_lin_vel_w[state_ids],
            "root_ang_vel_w": self.root_ang_vel_w[state_ids],
            "joint_pos": self.joint_pos[state_ids],
            "joint_vel": self.joint_vel[state_ids],
        }

    def sample_state_batch(self, batch_size) -> dict[str, torch.Tensor]:
        return self.get_state_batch(self.sample_state_ids(batch_size))

    def get_metadata(self, state_ids):
        state_ids = torch.as_tensor(
            state_ids, dtype=torch.long, device=self.device
        )
        if state_ids.ndim != 1:
            raise ValueError(
                f"state_ids must have shape [B], got {tuple(state_ids.shape)}."
            )
        if state_ids.numel() == 0:
            raise ValueError("state_ids must not be empty.")
        if torch.any(state_ids < 0) or torch.any(state_ids >= self.num_states):
            raise IndexError(
                f"state_ids must be in [0, {self.num_states - 1}]."
            )
        return tuple(
            self.state_metadata[state_id]
            for state_id in state_ids.detach().cpu().tolist()
        )


__all__ = ["RecoveryResetStateLoader"]
