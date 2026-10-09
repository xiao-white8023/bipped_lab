"""Exact architecture and validation for the deployed 2026-09-04 RENet policy.

No Isaac Sim or PPO runner is imported here. Collection and export use the
same reconstruction, including the ten-frame VP and critic windows.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import DEFAULT_CHECKPOINT_PATH, DEFAULT_REFERENCE_POLICY_PATH
from .policy_adapter import FrozenRENetAdapter, validate_checkpoint_metadata


OBS_DIM = 2120
PROPRIO_DIM = 10 * 78
LIDAR_DIM = 11 * 17
DEPTH_DIM = 2 * 18 * 32
MASK_INDEX = PROPRIO_DIM + LIDAR_DIM + DEPTH_DIM
ACTION_DIM = 23


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_policy() -> nn.Module:
    """Reconstruct model_60000 without history or auxiliary-head migrations."""

    from rsl_rl.modules import RENetActorCritic

    cnn_config = {
        "input_dim": (18, 32),
        "input_channels": 2,
        "output_channels": [8, 16, 32],
        "kernel_size": [3, 3, 3],
        "stride": [1, 1, 1],
        "dilation": [1, 1, 1],
        "padding": "zeros",
        "norm": "none",
        "activation": "relu",
        "max_pool": [True, True, False],
        "global_pool": "none",
        "flatten": True,
        "mlp_hidden_dim": [256, 128],
        "mlp_output_dim": 64,
        "mlp_activation": "elu",
    }
    lidar_config = {
        "input_dim": (11, 17),
        "input_channels": 1,
        "output_channels": [8, 16],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "dilation": [1, 1],
        "padding": "zeros",
        "norm": "none",
        "activation": "elu",
        "max_pool": [True, False],
        "global_pool": "none",
        "flatten": True,
        "mlp_hidden_dim": [128],
        "mlp_output_dim": 64,
        "mlp_activation": "elu",
    }
    return RENetActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=1037,
        num_actions=ACTION_DIM,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        CnnMlp=cnn_config,
        LidarCnnMlp=lidar_config,
        vp_proprio_history_frames=10,
        vp_output_type="legacy_latent",
        single_critic_dim=85,
        critic_history_frames=10,
        vp_critic_history_frames=10,
        single_proprio_dim=78,
        estimator_mask_dim=1,
        estimator_latent_dim=64,
        proprio_embed_dim=64,
        proprio_embed_dims=[256, 128],
        fusion_type="attention",
        attention_num_heads=1,
        lp_encoder_type="ame_point_cross_attention",
        ame_lp_attention_num_heads=16,
        ame_lp_query_history_frames=5,
        lp_encoder_dims=[128],
        lp_height_history_frames=1,
        vp_encoder_dims=[128],
        use_vel_estimation=True,
        vel_activation="elu",
        vel_hidden_dim=[32],
        vel_dim=3,
        use_terrain_recon=True,
        terrain_activation="elu",
        terrain_hidden_dim=[256, 128],
        lp_terrain_hidden_dim=[256, 128],
        terrain_scan_dim=187,
        terrain_recon_front_only=True,
        lp_terrain_recon_front_only=False,
        terrain_recon_grid_cols=17,
        terrain_recon_grid_rows=11,
        terrain_recon_x_min=0.0,
        use_feet_height_prediction=True,
        use_lp_feet_height_prediction=False,
        feet_height_activation="elu",
        feet_height_hidden_dim=[64],
        feet_height_dim=2,
    )


def validate_reference_weights(policy: nn.Module, reference: torch.jit.ScriptModule) -> None:
    """Prove that the reference policy and reconstructed checkpoint are identical."""

    reference_state = reference.state_dict()
    missing = []
    unequal = []
    for name, value in policy.state_dict().items():
        reference_name = f"policy.{name}"
        if reference_name not in reference_state:
            missing.append(reference_name)
        elif not torch.equal(value.cpu(), reference_state[reference_name].cpu()):
            unequal.append(reference_name)
    if missing or unequal:
        raise RuntimeError(
            "reference policy does not match the source checkpoint: "
            f"missing={missing[:5]}, unequal={unequal[:5]}"
        )


def validation_inputs() -> list[torch.Tensor]:
    """Create deterministic, shape-diverse policy probes."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260930)

    zero = torch.zeros(1, OBS_DIM)
    small = 0.05 * torch.randn(7, OBS_DIM, generator=generator)
    structured = torch.zeros(19, OBS_DIM)
    structured[:, :PROPRIO_DIM] = 0.25 * torch.randn(
        19, PROPRIO_DIM, generator=generator
    )
    structured[:, PROPRIO_DIM : PROPRIO_DIM + LIDAR_DIM] = (
        2.1 * torch.rand(19, LIDAR_DIM, generator=generator) - 1.1
    )
    structured[:, PROPRIO_DIM + LIDAR_DIM : MASK_INDEX] = torch.rand(
        19, DEPTH_DIM, generator=generator
    )
    structured[:, MASK_INDEX] = torch.arange(19) % 2
    return [zero, small, structured]


@torch.inference_mode()
def validate_actions(
    dual_model: nn.Module,
    reference: torch.jit.ScriptModule,
    probes: list[torch.Tensor],
) -> tuple[float, float]:
    """Compare dual actions against both forced-mask outputs of policy.pt."""

    vp_max_error = 0.0
    lp_max_error = 0.0
    for observations in probes:
        outputs = dual_model(observations)
        if len(outputs) != 6:
            raise RuntimeError(f"dual model must return six tensors, got {len(outputs)}")
        lp_action = outputs[4]
        vp_action = outputs[5]
        expected_shape = (observations.shape[0], ACTION_DIM)
        if tuple(lp_action.shape) != expected_shape or tuple(vp_action.shape) != expected_shape:
            raise RuntimeError(
                f"dual action shape mismatch: LP={tuple(lp_action.shape)}, "
                f"VP={tuple(vp_action.shape)}, expected={expected_shape}"
            )

        vp_input = observations.clone()
        vp_input[:, MASK_INDEX] = 0.0
        lp_input = observations.clone()
        lp_input[:, MASK_INDEX] = 1.0
        # The deployed CPU trace contains a literal CPU GRU hidden-state
        # allocation. Moving its weights to CUDA cannot rewrite that graph.
        expected_vp = reference(vp_input.cpu()).to(vp_action.device)
        expected_lp = reference(lp_input.cpu()).to(lp_action.device)

        vp_error = float((vp_action - expected_vp).abs().max().item())
        lp_error = float((lp_action - expected_lp).abs().max().item())
        vp_max_error = max(vp_max_error, vp_error)
        lp_max_error = max(lp_max_error, lp_error)
        if not torch.allclose(vp_action, expected_vp, rtol=1.0e-4, atol=1.0e-5):
            raise RuntimeError(f"VP parity validation failed: max error={vp_error:.9g}")
        if not torch.allclose(lp_action, expected_lp, rtol=1.0e-4, atol=1.0e-5):
            raise RuntimeError(f"LP parity validation failed: max error={lp_error:.9g}")
    return lp_max_error, vp_max_error


def policy_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate and retain the identity needed by datasets and selector artifacts."""

    required = {
        "source_checkpoint_sha256", "reference_policy_sha256", "source_iteration",
        "actor_observation_dim", "critic_observation_dim", "action_dim",
        "lp_query_history_frames", "vp_proprio_history_frames", "normalization", "inference_precision",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"missing frozen RENet identity fields: {sorted(missing)}; recollect the dataset")
    expected = {
        "source_iteration": 60000, "actor_observation_dim": OBS_DIM,
        "critic_observation_dim": 1037, "action_dim": ACTION_DIM,
        "lp_query_history_frames": 5, "vp_proprio_history_frames": 10,
        "normalization": "identity",
        "inference_precision": "float32_no_tf32",
    }
    for name, value in expected.items():
        if metadata[name] != value:
            raise ValueError(f"frozen RENet {name} must be {value!r}, got {metadata[name]!r}")
    for name in ("source_checkpoint_sha256", "reference_policy_sha256"):
        digest = metadata[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid frozen RENet hash: {name}")
    return copy.deepcopy(metadata)


def assert_same_policy(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Compare behavior identity independently of relocated source paths."""

    first, second = policy_identity(first), policy_identity(second)
    for name in (
        "source_checkpoint_sha256", "reference_policy_sha256", "source_iteration",
        "actor_observation_dim", "critic_observation_dim", "action_dim",
        "lp_query_history_frames", "vp_proprio_history_frames", "normalization", "inference_precision",
    ):
        if first[name] != second[name]:
            raise ValueError(f"selector/dataset frozen RENet mismatch: {name}")


def dataset_policy_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject old datasets whose latent/action provenance cannot be established."""

    return policy_identity(manifest.get("metadata", {}).get("frozen_policy", {}))


def load_frozen_adapter(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    reference_path: str | Path = DEFAULT_REFERENCE_POLICY_PATH,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FrozenRENetAdapter, nn.Module, dict[str, Any]]:
    """Strictly load the archived weights and prove parity with deployed policy.pt."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    for path in (checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("iter") != 60000:
        raise ValueError(f"expected model_60000, got iteration {checkpoint.get('iter')!r}")
    if "obs_norm_state_dict" in checkpoint:
        raise ValueError("the archived deployed policy uses identity observation normalization")
    # Match the robot's FP32/no-TF32 engine, including convolution and GRU
    # kernels. cuDNN otherwise permits TF32 even when matmul TF32 is disabled.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = build_frozen_policy().cpu().eval()
    validate_checkpoint_metadata(checkpoint, policy)
    for key, current in (
        ("vp_encoder_contract", policy.get_vp_encoder_contract()),
        ("critic_observation_contract", policy.get_critic_observation_contract()),
    ):
        if key in checkpoint and checkpoint[key] != current:
            raise ValueError(f"checkpoint {key} differs from the archived architecture")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference = torch.jit.load(str(reference_path), map_location="cpu").eval()
    validate_reference_weights(policy, reference)
    adapter = FrozenRENetAdapter(policy, nn.Identity(), copy_policy=False).eval()
    lp_error, vp_error = validate_actions(adapter, reference, validation_inputs())
    metadata = policy_identity({
        "version": 1,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "reference_policy": str(reference_path),
        "reference_policy_sha256": sha256(reference_path),
        "source_iteration": 60000,
        "actor_observation_dim": OBS_DIM,
        "critic_observation_dim": 1037,
        "action_dim": ACTION_DIM,
        "lp_query_history_frames": 5,
        "vp_proprio_history_frames": 10,
        "normalization": "identity",
        "inference_precision": "float32_no_tf32",
        "lp_encoder_contract": policy.get_lp_encoder_contract(),
        "vp_encoder_contract": policy.get_vp_encoder_contract(),
        "critic_observation_contract": policy.get_critic_observation_contract(),
        "lp_observation_contract": checkpoint.get("lp_observation_contract"),
        "checkpoint_lp_height_source": checkpoint.get("lp_height_source"),
        "validation": {"lp_max_abs_error": lp_error, "vp_max_abs_error": vp_error},
    })
    return adapter.to(device).eval(), reference, metadata
