"""Strict loader for G1RecoveryResetV1 physical reset states.

This loader is intentionally independent from AMP loaders.  The 59-D frames
are simulator states, while Recovery AMP files are 53-D discriminator-only
features and must never be accepted here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch


RECOVERY_RESET_FORMAT = "G1RecoveryResetV1"
RECOVERY_RESET_FRAME_SIZE = 59
RECOVERY_RESET_ROOT_XY_MODE = "zeroed_for_env_origin_placement"
RECOVERY_RESET_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


class RecoveryResetMotionLoader:
    """Preload and uniformly sample G1RecoveryResetV1 motion crops."""

    def __init__(
        self,
        motion_files: Sequence[str | Path],
        device: str | torch.device = "cpu",
        quaternion_atol: float = 1.0e-4,
    ):
        if isinstance(motion_files, (str, Path)):
            raise TypeError("motion_files must be a sequence of files, not one path string.")
        if not motion_files:
            raise ValueError("At least one Recovery reset motion file is required.")
        if quaternion_atol <= 0.0:
            raise ValueError("quaternion_atol must be positive.")

        self.device = torch.device(device)
        self.motion_files = tuple(Path(path).expanduser().resolve() for path in motion_files)
        if len(set(self.motion_files)) != len(self.motion_files):
            raise ValueError("Recovery reset motion file paths must be unique.")

        self.joint_names = list(RECOVERY_RESET_JOINT_NAMES)
        self.motion_frames = tuple(
            self._load_motion(path, quaternion_atol) for path in self.motion_files
        )
        self.trajectory_num_frames = tuple(frame.shape[0] for frame in self.motion_frames)
        self.num_motions = len(self.motion_frames)

    def _load_motion(self, path: Path, quaternion_atol: float) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Recovery reset motion file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in Recovery reset motion '{path}': {error}") from error

        if not isinstance(payload, dict):
            raise TypeError(f"Recovery reset motion '{path}' must contain a JSON object.")

        expected_metadata = {
            "Format": RECOVERY_RESET_FORMAT,
            "FrameSize": RECOVERY_RESET_FRAME_SIZE,
            "QuaternionConvention": "WXYZ",
            "RootLinearVelocityFrame": "world",
            "RootAngularVelocityFrame": "world",
            "RootXYMode": RECOVERY_RESET_ROOT_XY_MODE,
        }
        for key, expected in expected_metadata.items():
            actual = payload.get(key)
            if actual != expected:
                raise ValueError(
                    f"Recovery reset motion '{path}' has {key}={actual!r}; "
                    f"expected {expected!r}."
                )

        joint_names = payload.get("JointNames")
        if not isinstance(joint_names, list):
            raise TypeError(f"Recovery reset motion '{path}' JointNames must be a list.")
        if len(joint_names) != 23:
            raise ValueError(
                f"Recovery reset motion '{path}' must contain exactly 23 JointNames, "
                f"got {len(joint_names)}."
            )
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"Recovery reset motion '{path}' JointNames contains duplicates.")
        if tuple(joint_names) != RECOVERY_RESET_JOINT_NAMES:
            raise ValueError(
                f"Recovery reset motion '{path}' JointNames do not match "
                "the G1RecoveryResetV1 23-joint metadata."
            )

        try:
            frames = torch.tensor(payload.get("Frames"), dtype=torch.float32, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Recovery reset motion '{path}' Frames cannot be converted to float32."
            ) from error
        if frames.ndim != 2:
            raise ValueError(
                f"Recovery reset motion '{path}' Frames must be 2-D, got {tuple(frames.shape)}."
            )
        if frames.shape[0] == 0:
            raise ValueError(f"Recovery reset motion '{path}' contains no frames.")
        if frames.shape[1] != RECOVERY_RESET_FRAME_SIZE:
            raise ValueError(
                f"Recovery reset motion '{path}' Frames width must be "
                f"{RECOVERY_RESET_FRAME_SIZE}, got {frames.shape[1]}."
            )
        if not torch.isfinite(frames).all():
            raise ValueError(f"Recovery reset motion '{path}' Frames contains NaN or Inf.")
        if not torch.allclose(
            frames[:, 0:2],
            torch.zeros_like(frames[:, 0:2]),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"Recovery reset motion '{path}' declares {RECOVERY_RESET_ROOT_XY_MODE!r} "
                "but contains non-zero root X/Y."
            )

        quaternion_norm = torch.linalg.vector_norm(frames[:, 3:7], dim=1)
        if not torch.allclose(
            quaternion_norm,
            torch.ones_like(quaternion_norm),
            atol=quaternion_atol,
            rtol=quaternion_atol,
        ):
            raise ValueError(
                f"Recovery reset motion '{path}' root quaternion norm is not approximately 1; "
                f"range=[{quaternion_norm.min().item():.7f}, "
                f"{quaternion_norm.max().item():.7f}]."
            )
        return frames

    def sample(
        self,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
        return_indices: bool = False,
    ):
        """Sample crop uniformly first, then sample a frame within that crop."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")
        if num_samples < 0:
            raise ValueError("num_samples cannot be negative.")

        motion_ids = torch.randint(
            self.num_motions,
            (num_samples,),
            device=self.device,
            generator=generator,
        )
        frame_ids = torch.empty(num_samples, dtype=torch.long, device=self.device)
        samples = torch.empty(
            num_samples,
            RECOVERY_RESET_FRAME_SIZE,
            dtype=torch.float32,
            device=self.device,
        )
        for motion_id, motion_frames in enumerate(self.motion_frames):
            selected_rows = torch.nonzero(motion_ids == motion_id, as_tuple=False).flatten()
            if selected_rows.numel() == 0:
                continue
            selected_frame_ids = torch.randint(
                motion_frames.shape[0],
                (selected_rows.numel(),),
                device=self.device,
                generator=generator,
            )
            frame_ids[selected_rows] = selected_frame_ids
            samples[selected_rows] = motion_frames[selected_frame_ids]

        if return_indices:
            return samples, motion_ids, frame_ids
        return samples
