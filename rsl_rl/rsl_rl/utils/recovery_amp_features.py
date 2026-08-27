"""Pure-Torch whole-body AMP features for the standalone G1 Recovery task.

The same :class:`RecoveryAMPFeatureBuilder` is intended for both expert NPZ
states and live IsaacLab articulation states.  This module therefore has no
IsaacLab or Legged Lab dependency.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch


G1_RECOVERY_AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
)
G1_RECOVERY_AMP_ANCHOR_BODY_NAME = "torso_link"

NUM_TARGET_BODIES = len(G1_RECOVERY_AMP_BODY_NAMES)
BODY_POS_DIM = NUM_TARGET_BODIES * 3
BODY_ORI_DIM = NUM_TARGET_BODIES * 6
BODY_LIN_VEL_DIM = NUM_TARGET_BODIES * 3
BODY_ANG_VEL_DIM = NUM_TARGET_BODIES * 3
FEATURE_DIM = BODY_POS_DIM + BODY_ORI_DIM + BODY_LIN_VEL_DIM + BODY_ANG_VEL_DIM

BODY_POS_SLICE = slice(0, BODY_POS_DIM)
BODY_ORI_6D_SLICE = slice(BODY_POS_SLICE.stop, BODY_POS_SLICE.stop + BODY_ORI_DIM)
BODY_LIN_VEL_SLICE = slice(
    BODY_ORI_6D_SLICE.stop, BODY_ORI_6D_SLICE.stop + BODY_LIN_VEL_DIM
)
BODY_ANG_VEL_SLICE = slice(
    BODY_LIN_VEL_SLICE.stop, BODY_LIN_VEL_SLICE.stop + BODY_ANG_VEL_DIM
)


def _validate_quaternion_tensor(quaternion: torch.Tensor, name: str) -> None:
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not quaternion.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype, got {quaternion.dtype}.")
    if quaternion.ndim < 1 or quaternion.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [..., 4], got {tuple(quaternion.shape)}.")
    if not torch.isfinite(quaternion).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def quat_normalize_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize WXYZ quaternions with any leading batch shape."""

    _validate_quaternion_tensor(quaternion, "quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion.dtype).tiny):
        raise ValueError("Cannot normalize a zero or near-zero quaternion.")
    return quaternion / norm


def quat_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of normalized WXYZ quaternions."""

    quaternion = quat_normalize_wxyz(quaternion)
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _promote_pair(left: torch.Tensor, right: torch.Tensor, operation: str):
    if left.device != right.device:
        raise ValueError(
            f"Quaternion inputs to {operation} must share a device: "
            f"{left.device} != {right.device}."
        )
    dtype = torch.promote_types(left.dtype, right.dtype)
    return left.to(dtype=dtype), right.to(dtype=dtype)


def quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product of rotation quaternions in WXYZ convention."""

    left = quat_normalize_wxyz(left)
    right = quat_normalize_wxyz(right)
    left, right = _promote_pair(left, right, "quat_mul_wxyz")
    left, right = torch.broadcast_tensors(left, right)

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    result = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    return quat_normalize_wxyz(result)


def quat_apply_inverse_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply the inverse WXYZ quaternion rotation to vectors."""

    quaternion = quat_normalize_wxyz(quaternion)
    if not isinstance(vector, torch.Tensor):
        raise TypeError("vector must be a torch.Tensor.")
    if not vector.is_floating_point():
        raise ValueError(f"vector must have a floating dtype, got {vector.dtype}.")
    if vector.ndim < 1 or vector.shape[-1] != 3:
        raise ValueError(f"vector must have shape [..., 3], got {tuple(vector.shape)}.")
    if not torch.isfinite(vector).all():
        raise ValueError("vector contains NaN or Inf values.")

    quaternion, vector = _promote_pair(
        quaternion, vector, "quat_apply_inverse_wxyz"
    )
    quaternion_vector = quaternion[..., 1:]
    quaternion_vector, vector = torch.broadcast_tensors(quaternion_vector, vector)
    scalar = torch.broadcast_to(quaternion[..., :1], quaternion_vector.shape[:-1] + (1,))
    first_cross = torch.cross(quaternion_vector, vector, dim=-1)
    second_cross = torch.cross(quaternion_vector, first_cross, dim=-1)
    return vector - 2.0 * scalar * first_cross + 2.0 * second_cross


def matrix_from_quat_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized WXYZ quaternions to rotation matrices."""

    quaternion = quat_normalize_wxyz(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0

    row_0 = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
        ),
        dim=-1,
    )
    row_1 = torch.stack(
        (
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
        ),
        dim=-1,
    )
    row_2 = torch.stack(
        (
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return torch.stack((row_0, row_1, row_2), dim=-2)


def _validated_names(names: Sequence[str], field: str) -> Tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of names, not one string.")
    try:
        result = tuple(names)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of names.") from exc
    if not result:
        raise ValueError(f"{field} must not be empty.")
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{field} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate names.")
    return result


class RecoveryAMPFeatureBuilder:
    """Build the shared 195D whole-body Recovery AMP observation."""

    feature_dim = FEATURE_DIM

    def __init__(
        self,
        body_names,
        target_body_names=G1_RECOVERY_AMP_BODY_NAMES,
        anchor_body_name=G1_RECOVERY_AMP_ANCHOR_BODY_NAME,
    ):
        self.body_names = _validated_names(body_names, "body_names")
        self.target_body_names = _validated_names(
            target_body_names, "target_body_names"
        )
        if not isinstance(anchor_body_name, str) or not anchor_body_name:
            raise ValueError("anchor_body_name must be a non-empty string.")

        body_index = {name: index for index, name in enumerate(self.body_names)}
        if anchor_body_name not in body_index:
            raise ValueError(f"anchor body {anchor_body_name!r} is missing from body_names.")
        missing_targets = [
            name for name in self.target_body_names if name not in body_index
        ]
        if missing_targets:
            raise ValueError(f"Target bodies are missing from body_names: {missing_targets}.")

        self.anchor_body_name = anchor_body_name
        self.anchor_body_index = body_index[anchor_body_name]
        self.target_body_indices = tuple(
            body_index[name] for name in self.target_body_names
        )
        self.num_target_bodies = len(self.target_body_names)
        if self.num_target_bodies != NUM_TARGET_BODIES:
            raise ValueError(
                f"Recovery AMP requires exactly {NUM_TARGET_BODIES} target bodies, "
                f"got {self.num_target_bodies}."
            )

    def _validate_inputs(
        self,
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
        body_ang_vel_w,
    ):
        tensors = {
            "body_pos_w": (body_pos_w, 3),
            "body_quat_w": (body_quat_w, 4),
            "body_lin_vel_w": (body_lin_vel_w, 3),
            "body_ang_vel_w": (body_ang_vel_w, 3),
        }
        expected_batch_size = None
        expected_device = None
        common_dtype = None
        for field, (tensor, width) in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field} must be a torch.Tensor.")
            if tensor.ndim != 3 or tensor.shape[1:] != (len(self.body_names), width):
                raise ValueError(
                    f"{field} must have shape [B, {len(self.body_names)}, {width}], "
                    f"got {tuple(tensor.shape)}."
                )
            if not tensor.is_floating_point():
                raise ValueError(f"{field} must have a floating dtype, got {tensor.dtype}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{field} contains NaN or Inf values.")

            if expected_batch_size is None:
                expected_batch_size = tensor.shape[0]
                expected_device = tensor.device
                common_dtype = tensor.dtype
            else:
                if tensor.shape[0] != expected_batch_size:
                    raise ValueError("All Recovery AMP inputs must have the same batch size.")
                if tensor.device != expected_device:
                    raise ValueError("All Recovery AMP inputs must be on the same device.")
                common_dtype = torch.promote_types(common_dtype, tensor.dtype)

        return tuple(
            tensor.to(dtype=common_dtype) for tensor, _ in tensors.values()
        )

    def build(
        self,
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
        body_ang_vel_w,
    ) -> torch.Tensor:
        """Build ``[body_pos_b, body_ori_6d, body_lin_vel_b, body_ang_vel_b]``."""

        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = self._validate_inputs(
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )
        body_quat_w = quat_normalize_wxyz(body_quat_w)
        batch_size = body_pos_w.shape[0]

        anchor_pos_w = body_pos_w[:, self.anchor_body_index]
        anchor_quat_w = body_quat_w[:, self.anchor_body_index]
        target_pos_w = body_pos_w[:, self.target_body_indices]
        target_quat_w = body_quat_w[:, self.target_body_indices]
        target_lin_vel_w = body_lin_vel_w[:, self.target_body_indices]
        target_ang_vel_w = body_ang_vel_w[:, self.target_body_indices]

        relative_pos_w = target_pos_w - anchor_pos_w.unsqueeze(1)
        body_pos_b = quat_apply_inverse_wxyz(
            anchor_quat_w.unsqueeze(1), relative_pos_w
        )

        relative_quat = quat_mul_wxyz(
            quat_conjugate_wxyz(anchor_quat_w).unsqueeze(1), target_quat_w
        )
        relative_rotation = matrix_from_quat_wxyz(relative_quat)
        body_ori_6d = relative_rotation[..., :, :2]

        body_lin_vel_b = quat_apply_inverse_wxyz(target_quat_w, target_lin_vel_w)
        body_ang_vel_b = quat_apply_inverse_wxyz(target_quat_w, target_ang_vel_w)

        feature = torch.cat(
            (
                body_pos_b.reshape(batch_size, BODY_POS_DIM),
                body_ori_6d.reshape(batch_size, BODY_ORI_DIM),
                body_lin_vel_b.reshape(batch_size, BODY_LIN_VEL_DIM),
                body_ang_vel_b.reshape(batch_size, BODY_ANG_VEL_DIM),
            ),
            dim=-1,
        )
        if feature.shape != (batch_size, FEATURE_DIM):
            raise RuntimeError(
                f"Internal Recovery AMP feature shape error: {tuple(feature.shape)}."
            )
        if not torch.isfinite(feature).all():
            raise ValueError("Recovery AMP feature contains NaN or Inf values.")
        return feature

    def build_from_state(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Build features from a Stage-3.1 raw whole-body state mapping."""

        required = (
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping of raw whole-body tensors.")
        missing = [field for field in required if field not in state]
        if missing:
            raise ValueError(f"Raw Recovery state is missing fields: {missing}.")
        return self.build(*(state[field] for field in required))

    def build_transition(self, state, next_state):
        """Build separate current and next 195D features without concatenating them."""

        return self.build_from_state(state), self.build_from_state(next_state)


__all__ = [
    "G1_RECOVERY_AMP_BODY_NAMES",
    "G1_RECOVERY_AMP_ANCHOR_BODY_NAME",
    "NUM_TARGET_BODIES",
    "BODY_POS_DIM",
    "BODY_ORI_DIM",
    "BODY_LIN_VEL_DIM",
    "BODY_ANG_VEL_DIM",
    "FEATURE_DIM",
    "BODY_POS_SLICE",
    "BODY_ORI_6D_SLICE",
    "BODY_LIN_VEL_SLICE",
    "BODY_ANG_VEL_SLICE",
    "quat_normalize_wxyz",
    "quat_conjugate_wxyz",
    "quat_mul_wxyz",
    "quat_apply_inverse_wxyz",
    "matrix_from_quat_wxyz",
    "RecoveryAMPFeatureBuilder",
]
