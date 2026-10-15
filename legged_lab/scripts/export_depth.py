"""Export and reload the complete independent g1_depth inference graph."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from legged_lab.scripts.export import FullPolicyExporter, export_policy_as_jit, export_policy_as_onnx


@torch.inference_mode()
def export_depth_policy(policy, normalizer, path, observation_contract):
    """Validate TorchScript batches 1/7 and fixed-batch ONNX on CPU."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    if observation_contract.get("task") != "g1_depth" or policy.num_actor_obs != 1932:
        raise ValueError("g1_depth export requires its independent 1932-D observation contract.")
    destination = Path(path)
    metadata = {
        "version": 1, "task": "g1_depth", "input_shape": ["batch", 1932], "output_shape": ["batch", 23],
        "observation_contract": copy.deepcopy(observation_contract),
        "network_contract": policy.get_network_contract(), "vp_encoder_contract": policy.get_vp_encoder_contract(),
    }
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    probes = []
    for size in (1, 7):
        sample = 0.05 * torch.randn(size, 1932, generator=generator)
        sample[:, 780:] = torch.rand(size, 1152, generator=generator)
        probes.append(sample)
    reference = FullPolicyExporter(policy, normalizer).cpu().eval()
    errors = {}
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # Export the same explicit attention operations on all batch sizes.
        torch.backends.mha.set_fastpath_enabled(False)
        jit_path = export_policy_as_jit(
            policy, normalizer, str(destination), metadata=metadata, metadata_filename="depth_contract.json",
        )
        extra = {"depth_contract.json": ""}
        reloaded = torch.jit.load(jit_path, map_location="cpu", _extra_files=extra).eval()
        if json.loads(extra["depth_contract.json"]) != metadata:
            raise RuntimeError("Exported depth identity contract changed during serialization.")
        for sample in probes:
            expected, actual = reference(sample), reloaded(sample)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            if actual.shape != (len(sample), 23) or not torch.isfinite(actual).all():
                raise RuntimeError("Invalid exported g1_depth actions.")
            errors[f"torchscript_batch_{len(sample)}"] = float((actual - expected).abs().max())
        export_policy_as_onnx(policy, normalizer, str(destination))
        onnx_path = destination / "policy.onnx"
        graph = onnx.load(str(onnx_path))
        onnx.helper.set_model_props(graph, {"depth_contract.json": json.dumps(metadata, sort_keys=True)})
        onnx.save(graph, str(onnx_path))
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        evaluator = ReferenceEvaluator(str(onnx_path))
        for sample in (probes[0], torch.zeros(1, 1932)):
            actual = evaluator.run(None, {"obs": sample.numpy()})[0]
            expected = reference(sample).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            errors["onnx_batch_1"] = max(errors.get("onnx_batch_1", 0.0), float(np.abs(actual - expected).max()))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    (destination / "depth_contract.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (destination / "validation.json").write_text(json.dumps({"max_abs_errors": errors, "rtol": 1e-4, "atol": 1e-5}, indent=2))
    return errors
