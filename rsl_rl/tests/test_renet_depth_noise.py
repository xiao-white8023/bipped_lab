from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchvision.transforms as T


REPO_ROOT = Path(__file__).parents[2]
NOISE_PATH = REPO_ROOT / "legged_lab" / "utils" / "camera_noise" / "camera_noise.py"
NOISE_CFG_PATH = REPO_ROOT / "legged_lab" / "utils" / "camera_noise" / "camera_noise_cfg.py"
RENET_CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
RENET_ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"


def _load_functions(path, *function_names, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for name in function_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    globals_dict = {
        "torch": torch,
        "math": math,
        "Sequence": Sequence,
        "DistanceDependentGaussianNoiseCfg": object,
        "RangeBasedGaussianNoiseCfg": object,
    }
    if namespace:
        globals_dict.update(namespace)
    exec(compile(module, str(path), "exec"), globals_dict)
    return tuple(globals_dict[name] for name in function_names)


def _load_env_methods(*method_names):
    tree = ast.parse(RENET_ENV_PATH.read_text(encoding="utf-8"), filename=str(RENET_ENV_PATH))
    class_body = next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    selected = []
    for method_name in method_names:
        method = next(
            node for node in class_body if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method.decorator_list = []
        selected.append(method)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"torch": torch, "T": T}
    exec(compile(module, str(RENET_ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _noise_functions():
    return _load_functions(
        NOISE_PATH,
        "compute_distance_dependent_gaussian_std",
        "distance_dependent_gaussian_noise",
    )


def test_distance_dependent_sigma_endpoints_and_clamped_outside_values():
    compute_sigma, _ = _noise_functions()
    depth = torch.tensor([0.0, 0.1, 2.5, 3.0])
    sigma = compute_sigma(depth, 0.1, 2.5, 0.0, 0.10, 2.0)
    torch.testing.assert_close(sigma, torch.tensor([0.0, 0.0, 0.10, 0.10]))


def test_distance_dependent_sigma_is_monotonic_and_matches_quadratic_formula():
    compute_sigma, _ = _noise_functions()
    depth = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    sigma = compute_sigma(depth, 0.1, 2.5, 0.0, 0.10, 2.0)
    expected = 0.10 * ((depth - 0.1) / 2.4).square()
    torch.testing.assert_close(sigma, expected)
    assert torch.all(sigma[1:] >= sigma[:-1])


def test_zero_distance_dependent_noise_is_exact_identity():
    _, add_noise = _noise_functions()
    depth = torch.linspace(0.1, 2.5, 24).reshape(2, 3, 4, 1)
    cfg = SimpleNamespace(near_std=0.0, far_std=0.0, distance_exponent=2.0)
    output = add_noise(
        depth,
        cfg,
        torch.tensor([0, 1]),
        min_distance=0.1,
        max_distance=2.5,
    )
    assert torch.equal(output, depth)
    assert output.data_ptr() != depth.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distance_dependent_noise_preserves_shape_device_dtype_and_is_finite(dtype):
    _, add_noise = _noise_functions()
    depth = torch.linspace(0.1, 2.5, 24, dtype=dtype).reshape(2, 3, 4, 1)
    cfg = SimpleNamespace(near_std=0.0, far_std=0.10, distance_exponent=2.0)
    output = add_noise(
        depth,
        cfg,
        torch.tensor([5, 9]),
        min_distance=0.1,
        max_distance=2.5,
    )
    assert output.shape == depth.shape
    assert output.device == depth.device
    assert output.dtype == depth.dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_distance": 1.0, "max_distance": 1.0}, "max_distance"),
        ({"near_std": -0.01}, "near_std"),
        ({"near_std": 0.2, "far_std": 0.1}, "far_std"),
        ({"exponent": 0.0}, "exponent"),
    ],
)
def test_distance_dependent_sigma_rejects_invalid_parameters(kwargs, message):
    compute_sigma, _ = _noise_functions()
    parameters = {
        "min_distance": 0.1,
        "max_distance": 2.5,
        "near_std": 0.0,
        "far_std": 0.10,
        "exponent": 2.0,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        compute_sigma(torch.ones(1), **parameters)


def test_legacy_range_based_gaussian_noise_keeps_constant_std_mask_semantics():
    (range_noise,) = _load_functions(NOISE_PATH, "range_based_gaussian_noise")
    data = torch.tensor([[[[0.0], [1.0], [2.0], [3.0]]]])
    cfg = SimpleNamespace(noise_std=0.25, min_value=1.0, max_value=2.0)
    torch.manual_seed(17)
    expected_noise = torch.randn(data.shape) * cfg.noise_std
    apply_mask = (data >= cfg.min_value) & (data <= cfg.max_value)
    torch.manual_seed(17)
    output = range_noise(data, cfg, torch.tensor([0]))
    torch.testing.assert_close(output, data + expected_noise * apply_mask)


def _depth_pipeline_env(apply_noise):
    raw_depth = torch.full((2, 36, 64, 1), 1.25)
    return SimpleNamespace(
        camera=SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": raw_depth})),
        cfg=SimpleNamespace(
            robot=SimpleNamespace(depth_crop=(18, 0, 16, 16), depth_max=2.5),
            scene=SimpleNamespace(
                camera=SimpleNamespace(
                    camera=SimpleNamespace(
                        min_distance=0.1,
                        pattern_cfg=SimpleNamespace(height=36, width=64),
                    )
                )
            ),
        ),
        depth_min_distance=0.1,
        depth_max_distance=2.5,
        _apply_distance_dependent_depth_noise=apply_noise,
    )


def test_renet_depth_pipeline_passes_clean_metric_depth_to_noise_before_normalization():
    (get_processed_depth,) = _load_env_methods("get_processed_deepcamera")
    observed = []

    def record_clean_depth(depth, _env_ids):
        observed.append(depth.clone())
        return depth

    env = _depth_pipeline_env(record_clean_depth)
    output = get_processed_depth(env, env_ids=torch.tensor([1]))
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], torch.full((1, 18, 32), 1.25))
    torch.testing.assert_close(output, torch.full((1, 18, 32), 0.5))


def test_renet_depth_pipeline_clips_again_after_noise():
    (get_processed_depth,) = _load_env_methods("get_processed_deepcamera")

    def force_out_of_range(depth, _env_ids):
        output = depth.clone()
        output[:, :, : output.shape[2] // 2] = -100.0
        output[:, :, output.shape[2] // 2 :] = 100.0
        return output

    env = _depth_pipeline_env(force_out_of_range)
    normalized = get_processed_depth(env)
    metric_depth = normalized * env.depth_max_distance
    assert metric_depth.min().item() == pytest.approx(0.1)
    assert metric_depth.max().item() == pytest.approx(2.5)
    assert torch.isfinite(normalized).all()


def test_depth_history_samples_only_on_camera_update_and_repeats_one_reset_frame():
    (get_history,) = _load_env_methods("get_deepcamera_history")
    calls = []

    def get_processed_depth(env_ids):
        calls.append(env_ids.clone())
        values = (env_ids.float() + 10.0).view(-1, 1, 1)
        return values.expand(-1, 2, 2).clone()

    original = torch.arange(24, dtype=torch.float).reshape(3, 2, 2, 2)
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([1, 5, 0]),
        cfg=SimpleNamespace(robot=SimpleNamespace(depth_update_interval=5)),
        depth_buffer=original.clone(),
        depth_history_frames=2,
        get_processed_deepcamera=get_processed_depth,
    )
    output = get_history(env)
    assert len(calls) == 1
    assert torch.equal(calls[0], torch.tensor([1, 2]))
    torch.testing.assert_close(output[0], original[0])
    torch.testing.assert_close(output[1, 0], original[1, 1])
    torch.testing.assert_close(output[1, 1], torch.full((2, 2), 11.0))
    torch.testing.assert_close(output[2, 0], output[2, 1])
    torch.testing.assert_close(output[2, 0], torch.full((2, 2), 12.0))

    cached = output.clone()
    env.episode_length_buf = torch.tensor([2, 6, 1])
    output = get_history(env)
    assert len(calls) == 1
    torch.testing.assert_close(output, cached)


def test_renet_config_enables_only_fixed_distance_dependent_gaussian_noise():
    cfg_tree = ast.parse(NOISE_CFG_PATH.read_text(encoding="utf-8"))
    noise_cfg_class = next(
        node
        for node in cfg_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DistanceDependentGaussianNoiseCfg"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in noise_cfg_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert defaults == {"near_std": 0.0, "far_std": 0.10, "distance_exponent": 2.0}

    renet_source = RENET_CFG_PATH.read_text(encoding="utf-8")
    assert "add_camera_noise=True" in renet_source
    assert "depth_gaussian_noise: DistanceDependentGaussianNoiseCfg" in renet_source
    assert "near_std=0.0" in renet_source
    assert "far_std=0.10" in renet_source
    assert "distance_exponent=2.0" in renet_source


def test_renet_depth_path_has_no_structured_failure_or_noise_curriculum_state():
    source = RENET_ENV_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "structured_depth_failure_model",
        "depth_noise_strength",
        "depth_noise_return_cv",
        "depth_noise_return_window",
        "update_depth_noise_curriculum_once",
        "failure_probability",
    ):
        assert forbidden not in source
    for diagnostic in (
        "DepthNoise/enabled",
        "DepthNoise/near_std",
        "DepthNoise/far_std",
        "DepthNoise/distance_exponent",
    ):
        assert diagnostic in source
