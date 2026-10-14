"""One CPU dual-branch artifact contract for collection and robot deployment."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from .frozen_policy import OBS_DIM, policy_identity, validate_actions, validation_inputs


DUAL_CONTRACT_FILE = "renet_dual_branch_contract.json"
DUAL_OUTPUTS = (
    "lp_latent_64", "vp_latent_64", "lp_velocity_3", "vp_velocity_3",
    "lp_action_23", "vp_action_23",
)
DUAL_OUTPUT_DIMS = (64, 64, 3, 3, 23, 23)
DUAL_INPUT = {
    "shape": ["batch", OBS_DIM],
    "layout": ["proprio_history_10x78", "elevation_11x17",
               "depth_history_2x18x32", "compatibility_mask_unused"],
}


def dual_branch_contract(
    frozen_metadata: dict[str, Any], *, validation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the version-2 contract already accepted by unitree_deploy_ws."""
    frozen = policy_identity(frozen_metadata)
    return {
        "artifact": "g1_renet_dual_branch",
        "version": 2,
        "frozen_policy": frozen,
        "source_checkpoint": frozen.get("source_checkpoint", ""),
        "source_checkpoint_sha256": frozen["source_checkpoint_sha256"],
        "source_iteration": frozen["source_iteration"],
        "reference_policy": frozen.get("reference_policy", ""),
        "reference_policy_sha256": frozen["reference_policy_sha256"],
        "input": copy.deepcopy(DUAL_INPUT),
        "branch_history": {"lp_raw_frames": 5, "vp_embedded_frames": 10},
        "outputs": list(DUAL_OUTPUTS),
        "reference_mask_contract": {"vp": 0, "lp": 1},
        "validation": copy.deepcopy(validation or {}),
    }


def validate_dual_branch_contract(value: Any) -> dict[str, Any]:
    """Reject legacy/incomplete schemas and conflicting identity fields."""
    if not isinstance(value, dict):
        raise ValueError("dual-branch artifact must embed a contract")
    if value.get("artifact") != "g1_renet_dual_branch" or value.get("version") != 2:
        raise ValueError("dual-branch artifact requires g1_renet_dual_branch version 2; re-export")
    frozen = policy_identity(value.get("frozen_policy", {}))
    canonical = dual_branch_contract(frozen)
    for key in ("input", "outputs", "branch_history", "reference_mask_contract",
                "source_checkpoint_sha256", "source_iteration", "reference_policy_sha256"):
        if value.get(key) != canonical[key]:
            raise ValueError(f"dual-branch contract mismatch: {key}; re-export")
    return copy.deepcopy(value)


@torch.inference_mode()
def _compare_outputs(actual, expected, batch_size: int) -> dict[str, float]:
    if not isinstance(actual, (tuple, list)) or len(actual) != 6:
        raise ValueError("dual-branch model must return all six tensors in contract order")
    errors = {}
    for name, dim, observed, reference in zip(DUAL_OUTPUTS, DUAL_OUTPUT_DIMS, actual, expected):
        if (not isinstance(observed, torch.Tensor) or observed.shape != (batch_size, dim)
                or observed.dtype != torch.float32 or not torch.isfinite(observed).all()):
            raise ValueError(f"invalid dual-branch output: {name}")
        if not torch.allclose(observed, reference, rtol=1.0e-4, atol=1.0e-5):
            raise RuntimeError(f"dual-branch TorchScript parity validation failed: {name}")
        errors[name] = float((observed - reference).abs().max())
    return errors


@torch.inference_mode()
def export_dual_branch_artifact(
    adapter: nn.Module,
    destination: str | Path,
    frozen_metadata: dict[str, Any],
    *,
    sample: torch.Tensor | None = None,
    reference: nn.Module | None = None,
) -> Path:
    """Export, reload and validate six outputs at batches 1, 7 and 19 on CPU.

    Never move the deployed reference trace to CUDA. Publish the file only after
    validation, and use an exclusive link so an existing artifact cannot be lost.
    """
    destination = Path(destination).expanduser().resolve()
    frozen = policy_identity(frozen_metadata)
    for key in ("source_checkpoint", "reference_policy"):
        if frozen.get(key) and destination == Path(frozen[key]).expanduser().resolve():
            raise ValueError("dual-branch output must not overwrite a source artifact")
    if destination.exists():
        raise FileExistsError(destination)
    cpu_adapter = copy.deepcopy(adapter).cpu().eval()
    probes = validation_inputs()
    trace_input = probes[1][:2] if sample is None else sample[:2].detach().cpu().float()
    if trace_input.ndim != 2 or trace_input.shape[1] != OBS_DIM or trace_input.shape[0] < 1:
        raise ValueError("dual-branch trace input must have shape [batch, 2120]")
    traced = torch.jit.trace(cpu_adapter, trace_input, strict=False, check_trace=False)
    errors = dict.fromkeys(DUAL_OUTPUTS, 0.0)
    for probe in probes:
        measured = _compare_outputs(traced(probe), cpu_adapter(probe), probe.shape[0])
        errors = {name: max(errors[name], measured[name]) for name in DUAL_OUTPUTS}
    validation = {"all_output_max_abs_errors": errors,
                  "batch_sizes": [int(probe.shape[0]) for probe in probes]}
    if reference is not None:
        eager_lp, eager_vp = validate_actions(cpu_adapter, reference, probes)
        traced_lp, traced_vp = validate_actions(traced, reference, probes)
        validation.update({"eager_lp_max_abs_error": eager_lp, "eager_vp_max_abs_error": eager_vp,
                           "traced_lp_max_abs_error": traced_lp, "traced_vp_max_abs_error": traced_vp})
    metadata = dual_branch_contract(frozen, validation=validation)
    validate_dual_branch_contract(metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".renet_export_", dir=destination.parent) as temporary:
        staged = Path(temporary) / "renet_dual_branch.pt"
        torch.jit.save(traced, str(staged), _extra_files={
            DUAL_CONTRACT_FILE: json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
        })
        extra = {DUAL_CONTRACT_FILE: ""}
        reloaded = torch.jit.load(str(staged), map_location="cpu", _extra_files=extra).eval()
        validate_dual_branch_contract(json.loads(extra[DUAL_CONTRACT_FILE]))
        for probe in probes:
            _compare_outputs(reloaded(probe), cpu_adapter(probe), probe.shape[0])
        if reference is not None:
            validate_actions(reloaded, reference, probes)
        os.link(staged, destination)
    return destination
