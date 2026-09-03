from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
NOISE_PATH = REPO_ROOT / "legged_lab" / "sensors" / "camera_noise" / "camera_noise.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_noise_functions():
    """Load the pure PyTorch functions without importing the Isaac Sim package."""
    tree = ast.parse(_source(NOISE_PATH))
    function_names = {
        "compute_distance_dependent_gaussian_std",
        "distance_dependent_gaussian_noise",
    }
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected_nodes],
        type_ignores=[],
    )
    namespace = {
        "math": math,
        "torch": torch,
        "Sequence": Sequence,
    }
    exec(compile(ast.fix_missing_locations(module), str(NOISE_PATH), "exec"), namespace)
    return (
        namespace["compute_distance_dependent_gaussian_std"],
        namespace["distance_dependent_gaussian_noise"],
    )


compute_distance_dependent_gaussian_std, distance_dependent_gaussian_noise = _load_noise_functions()


def test_distance_dependent_std_uses_metric_depth_and_quadratic_growth():
    depth = torch.tensor([0.1, 1.3, 2.5], dtype=torch.float32)
    sigma = compute_distance_dependent_gaussian_std(
        depth,
        min_distance=0.1,
        max_distance=2.5,
        near_std=0.0,
        far_std=0.10,
        exponent=2.0,
    )

    torch.testing.assert_close(sigma, torch.tensor([0.0, 0.025, 0.10]))


def test_distance_dependent_noise_preserves_shape_dtype_and_device(monkeypatch):
    depth = torch.tensor(
        [[[0.1, 1.3], [2.5, 3.0]]],
        dtype=torch.float64,
    )
    cfg = SimpleNamespace(near_std=0.0, far_std=0.10, distance_exponent=2.0)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    noisy = distance_dependent_gaussian_noise(
        depth,
        cfg,
        torch.tensor([0]),
        min_distance=0.1,
        max_distance=2.5,
    )

    assert noisy.shape == depth.shape
    assert noisy.dtype == depth.dtype
    assert noisy.device == depth.device
    torch.testing.assert_close(noisy - depth, torch.tensor([[[0.0, 0.025], [0.10, 0.10]]], dtype=depth.dtype))


def test_distance_dependent_noise_validates_parameters_and_batch_shape():
    depth = torch.ones((2, 3, 4), dtype=torch.float32)
    cfg = SimpleNamespace(near_std=0.0, far_std=0.10, distance_exponent=2.0)

    with pytest.raises(ValueError, match="max_distance"):
        distance_dependent_gaussian_noise(
            depth,
            cfg,
            torch.tensor([0, 1]),
            min_distance=1.0,
            max_distance=1.0,
        )
    with pytest.raises(ValueError, match="env_ids"):
        distance_dependent_gaussian_noise(
            depth,
            cfg,
            torch.tensor([0]),
            min_distance=0.1,
            max_distance=2.5,
        )


def test_renet_enables_only_distance_dependent_random_depth_noise():
    cfg_source = _source(CFG_PATH)
    env_source = _source(ENV_PATH)
    env_tree = ast.parse(env_source)
    method = next(
        node
        for node in ast.walk(env_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_processed_deepcamera"
    )
    method_source = ast.get_source_segment(env_source, method)
    assert method_source is not None

    assert "add_camera_noise=True" in cfg_source
    assert "DistanceDependentGaussianNoiseCfg" in cfg_source
    assert "near_std=0.0" in cfg_source
    assert "far_std=0.10" in cfg_source
    assert "distance_exponent=2.0" in cfg_source
    assert method_source.count("distance_dependent_gaussian_noise(") == 1
    assert method_source.index("distance_dependent_gaussian_noise(") < method_source.index("depth_clipped =")
    assert method_source.index("distance_dependent_gaussian_noise(") < method_source.index("depth_normalized =")
    assert "structured" not in method_source.lower()
    assert "dropout" not in method_source.lower()
