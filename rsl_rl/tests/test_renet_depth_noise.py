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


def _depth_processing_env(probability=.5, add_noise=True, count=4, *, iteration=0, force_active=False,
                          start_iteration=None):
    import torchvision.transforms as transforms

    tree = ast.parse(_source(ENV_PATH))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'G1RENetEnv')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('get_processed_deepcamera', 'get_deepcamera_history', '_depth_noise_is_active')]
    namespace = {'torch': torch, 'T': transforms,
                 'distance_dependent_gaussian_noise': distance_dependent_gaussian_noise}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(ENV_PATH), 'exec'), namespace)
    depth = torch.ones(count, 36, 64, 1)
    depth[:, 20, 20] = 2.4
    if start_iteration is None:
        cfg_tree = ast.parse(_source(CFG_PATH))
        cfg = next(node for node in cfg_tree.body if isinstance(node, ast.ClassDef) and node.name == 'G1RENETENVCFG')
        start_iteration = ast.literal_eval(next(node.value for node in cfg.body if isinstance(node, ast.AnnAssign)
                                               and node.target.id == 'depth_noise_start_iteration'))
    env = SimpleNamespace(
        cfg=SimpleNamespace(depth_blur_probability=probability,
            depth_noise_start_iteration=start_iteration, depth_noise_force_active=force_active,
            depth_noise=SimpleNamespace(near_std=0., far_std=0., distance_exponent=2.),
            robot=SimpleNamespace(depth_crop=(18, 0, 16, 16), depth_max=2.5, depth_update_interval=5),
            scene=SimpleNamespace(camera=SimpleNamespace(add_camera_noise=add_noise,
                camera=SimpleNamespace(pattern_cfg=SimpleNamespace(height=36, width=64), min_distance=.1)))),
        camera=SimpleNamespace(data=SimpleNamespace(output={'distance_to_image_plane': depth}),
                               current_timestamps=torch.zeros(count)),
        step_dt=.02, depth_history_frames=2, renet_training_iteration=iteration,
        depth_buffer=torch.zeros(count, 2, 18, 32),
        depth_last_capture_time=torch.full((count,), -torch.inf),
        depth_initialized=torch.zeros(count, dtype=torch.bool))
    env.get_processed_deepcamera = lambda: namespace['get_processed_deepcamera'](env)
    env.get_deepcamera_history = lambda: namespace['get_deepcamera_history'](env)
    env._depth_noise_is_active = lambda: namespace['_depth_noise_is_active'](env)
    return env


@pytest.mark.parametrize('iteration,force_active,add_noise,active', [
    (0, False, True, True), (29999, False, True, True),
    (30000, False, True, True), (42000, False, True, True),
    (0, True, True, True), (30000, False, False, False), (0, True, False, False),
])
def test_depth_noise_and_blur_start_immediately_and_honor_camera_noise_switch(
        monkeypatch, iteration, force_active, add_noise, active):
    import torchvision.transforms as transforms

    env = _depth_processing_env(probability=1., iteration=iteration,
                                force_active=force_active, add_noise=add_noise)
    env.cfg.depth_noise.near_std = env.cfg.depth_noise.far_std = .025
    draws = []
    def gaussian_draw(value):
        draws.append(1)
        return torch.ones_like(value)
    monkeypatch.setattr(torch, 'randn_like', gaussian_draw)
    cropped = env.camera.data.output['distance_to_image_plane'][:, 18:, 16:48, 0]
    if active:
        expected = transforms.GaussianBlur(3, sigma=1.0)(cropped.unsqueeze(1)).squeeze(1) + .025
    else:
        expected = cropped
    assert env._depth_noise_is_active() == active
    torch.testing.assert_close(env.get_processed_deepcamera(), expected / 2.5, rtol=0, atol=0)
    assert len(draws) == int(active)


def test_boundary_keeps_clean_history_until_next_capture_and_does_not_reprocess_it():
    # Explicit delayed overrides still work; the production default is zero.
    env = _depth_processing_env(probability=1., iteration=29999, start_iteration=30000)
    clean = env.get_deepcamera_history().clone()
    env.renet_training_iteration = 30000
    env.camera.current_timestamps.fill_(.02)
    torch.testing.assert_close(env.get_deepcamera_history(), clean, rtol=0, atol=0)
    env.camera.current_timestamps.fill_(.1)
    noisy = env.get_deepcamera_history().clone()
    torch.testing.assert_close(noisy[:, 0], clean[:, 1], rtol=0, atol=0)
    assert not torch.equal(noisy[:, 1], clean[:, 1])
    env.camera.current_timestamps.fill_(.12)
    torch.testing.assert_close(env.get_deepcamera_history(), noisy, rtol=0, atol=0)


def test_probability_blur_selects_environments_independently(monkeypatch):
    import torchvision.transforms as transforms

    env = _depth_processing_env()
    source = env.camera.data.output['distance_to_image_plane'].clone()
    monkeypatch.setattr(torch, 'rand', lambda count, **kwargs: torch.tensor([.1, .9, .49, .51], device=kwargs['device']))
    actual = env.get_processed_deepcamera()
    cropped = source[:, 18:, 16:48, 0]
    blurred = transforms.GaussianBlur(3, sigma=1.0)(cropped.unsqueeze(1)).squeeze(1) / 2.5
    for row in (0, 2):
        torch.testing.assert_close(actual[row], blurred[row])
        assert not torch.equal(actual[row], cropped[row] / 2.5)
    for row in (1, 3):
        torch.testing.assert_close(actual[row], cropped[row] / 2.5, rtol=0, atol=0)
    torch.testing.assert_close(env.camera.data.output['distance_to_image_plane'], source, rtol=0, atol=0)


@pytest.mark.parametrize('probability,add_noise,expect_blur', [(0., True, False), (1., True, True), (1., False, False)])
def test_blur_probability_endpoints_and_camera_noise_switch(probability, add_noise, expect_blur):
    import torchvision.transforms as transforms

    env = _depth_processing_env(probability, add_noise)
    cropped = env.camera.data.output['distance_to_image_plane'][:, 18:, 16:48, 0]
    expected = transforms.GaussianBlur(3, sigma=1.0)(cropped.unsqueeze(1)).squeeze(1) if expect_blur else cropped
    torch.testing.assert_close(env.get_processed_deepcamera(), expected / 2.5)


@pytest.mark.parametrize('probability', [-.01, 1.01, float('nan')])
def test_invalid_blur_probability_is_rejected(probability):
    with pytest.raises(ValueError, match='depth_blur_probability'):
        _depth_processing_env(probability).get_processed_deepcamera()


def test_blur_decision_is_retained_in_history_until_next_capture(monkeypatch):
    env = _depth_processing_env()
    draws = iter([torch.tensor([.1, .9, .1, .9]), torch.tensor([.9, .1, .9, .1])])
    calls = []
    def draw(count, **kwargs):
        calls.append(count)
        return next(draws).to(kwargs['device'])
    monkeypatch.setattr(torch, 'rand', draw)
    first = env.get_deepcamera_history().clone()
    torch.testing.assert_close(first[:, 0], first[:, 1])
    env.camera.current_timestamps.fill_(.02)
    torch.testing.assert_close(env.get_deepcamera_history(), first, rtol=0, atol=0)
    assert len(calls) == 1
    env.camera.current_timestamps.fill_(.1)
    second = env.get_deepcamera_history()
    torch.testing.assert_close(second[:, 0], first[:, 1], rtol=0, atol=0)
    assert all(not torch.equal(second[row, 1], first[row, 1]) for row in range(4))
    assert len(calls) == 2


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


def test_renet_uses_distance_dependent_depth_noise_before_clipping():
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
