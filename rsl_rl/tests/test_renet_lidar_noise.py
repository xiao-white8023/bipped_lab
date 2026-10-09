from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
NOISE_PATH = REPO_ROOT / "legged_lab" / "sensors" / "lidar" / "lidar_noise.py"
CFG_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_cfg.py"
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
PLAY_PATH = REPO_ROOT / "legged_lab" / "scripts" / "play_new_renet.py"
EVAL_PATH = REPO_ROOT / "legged_lab" / "scripts" / "eval.py"


def _load_noise_module():
    spec = importlib.util.spec_from_file_location("renet_lidar_noise", NOISE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lidar_noise = _load_noise_module()
LidarNoiseModel = lidar_noise.LidarNoiseModel
validate_lidar_noise_cfg = lidar_noise.validate_lidar_noise_cfg


def _cfg(
    *,
    enable=True,
    range_std=(0.0, 0.0),
    dropout=(0.0, 0.0),
    translation=(0.0, 0.0),
    rotation=(0.0, 0.0),
):
    return SimpleNamespace(
        enable=enable,
        range_std_range_m=range_std,
        point_dropout_prob_range=dropout,
        translation_bias_range_m=translation,
        rotation_bias_range_deg=rotation,
    )


def _inputs(num_envs=1, num_rays=1, dtype=torch.float32):
    hits = torch.zeros((num_envs, num_rays, 3), dtype=dtype)
    sensor = torch.zeros((num_envs, 3), dtype=dtype)
    torso = torch.zeros((num_envs, 3), dtype=dtype)
    quaternion = torch.zeros((num_envs, 4), dtype=dtype)
    quaternion[:, 0] = 1.0
    scan_count = torch.ones(num_envs, dtype=torch.long)
    return hits, sensor, torso, quaternion, scan_count


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("range_std_range_m", (-0.01, 0.02), "cannot be negative"),
        ("point_dropout_prob_range", (0.0, 1.01), r"inside \[0, 1\]"),
        ("translation_bias_range_m", (float("nan"), 0.0), "finite"),
        ("rotation_bias_range_deg", (0.2, -0.2), "increasing"),
    ],
)
def test_config_validation_rejects_invalid_ranges(field, value, message):
    cfg = _cfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=message):
        validate_lidar_noise_cfg(cfg)


def test_disabled_model_returns_original_tensors_without_copying():
    model = LidarNoiseModel(_cfg(enable=False), num_envs=1, device="cpu")
    hits, sensor, torso, quaternion, scan_count = _inputs()
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.0])

    returned_hits, returned_sensor, new_ids = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=5.0,
    )
    assert returned_hits is hits
    assert returned_sensor is sensor
    assert new_ids.tolist() == [0]

    _, _, repeated_ids = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=5.0,
    )
    assert repeated_ids.numel() == 0


def test_range_noise_moves_only_along_ray_and_clamps(monkeypatch):
    model = LidarNoiseModel(
        _cfg(range_std=(0.2, 0.2)),
        num_envs=1,
        device="cpu",
    )
    hits, sensor, torso, quaternion, scan_count = _inputs(num_rays=3, dtype=torch.float64)
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.0], dtype=hits.dtype)
    hits[0, 1] = torch.tensor([0.0, 1.9, 0.0], dtype=hits.dtype)
    hits[0, 2] = 2.0  # The grouped ray caster's no-return sentinel.
    original = hits.clone()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.ones_like(value))

    noisy_hits, noisy_sensor, _ = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=2.0,
    )

    torch.testing.assert_close(noisy_hits[0, 0], torch.tensor([1.2, 0.0, 0.0], dtype=hits.dtype))
    torch.testing.assert_close(noisy_hits[0, 1], torch.tensor([0.0, 2.0, 0.0], dtype=hits.dtype))
    torch.testing.assert_close(noisy_hits[0, 2], original[0, 2])
    torch.testing.assert_close(noisy_sensor, sensor)
    torch.testing.assert_close(hits, original)
    assert noisy_hits.dtype == hits.dtype


def test_dropout_removes_only_valid_returns_and_preserves_source(monkeypatch):
    model = LidarNoiseModel(_cfg(dropout=(1.0, 1.0)), num_envs=1, device="cpu")
    hits, sensor, torso, quaternion, scan_count = _inputs(num_rays=3)
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    hits[0, 1] = 5.0
    hits[0, 2] = torch.tensor([float("nan"), 0.0, 0.0])
    original = hits.clone()
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.zeros_like(value))

    noisy_hits, _, _ = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=5.0,
    )

    assert torch.isnan(noisy_hits[0, 0]).all()
    torch.testing.assert_close(noisy_hits[0, 1], torch.full((3,), 5.0))
    assert torch.isnan(noisy_hits[0, 2, 0])
    torch.testing.assert_close(hits, original, equal_nan=True)


def test_finite_out_of_range_return_cannot_become_valid_after_translation_bias(monkeypatch):
    model = LidarNoiseModel(_cfg(), num_envs=1, device="cpu")
    model.translation_bias_body_m[0] = torch.tensor([-0.005, 0.0, 0.0])
    hits, sensor, torso, quaternion, scan_count = _inputs()
    hits[0, 0] = torch.tensor([5.002, 0.0, 0.0])
    original = hits.clone()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.ones_like(value))

    noisy_hits, _, _ = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=5.0,
    )

    assert torch.isnan(noisy_hits[0, 0]).all()
    torch.testing.assert_close(hits, original)


def test_body_frame_extrinsic_bias_transforms_origin_and_ray(monkeypatch):
    model = LidarNoiseModel(_cfg(), num_envs=1, device="cpu")
    model.translation_bias_body_m[0] = torch.tensor([1.0, 0.0, 0.0])
    model.rotation_bias_rpy_rad[0] = torch.tensor([0.0, 0.0, math.pi / 2.0])
    hits, sensor, torso, quaternion, scan_count = _inputs()
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.ones_like(value))

    noisy_hits, noisy_sensor, _ = model.apply_new_scans(
        hits,
        sensor,
        torso,
        quaternion,
        scan_count,
        min_distance=0.1,
        max_distance=5.0,
    )

    torch.testing.assert_close(noisy_sensor[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(noisy_hits[0, 0], torch.tensor([1.0, 1.0, 0.0]), atol=1.0e-6, rtol=0.0)


def test_partial_reset_only_resamples_selected_environments():
    torch.manual_seed(7)
    model = LidarNoiseModel(
        _cfg(
            range_std=(0.01, 0.03),
            dropout=(0.0, 0.05),
            translation=(-0.005, 0.005),
            rotation=(-0.2, 0.2),
        ),
        num_envs=2,
        device="cpu",
    )
    model.last_scan_count[:] = torch.tensor([3, 4])
    env_one_state = (
        model.range_std_m[1].clone(),
        model.point_dropout_prob[1].clone(),
        model.translation_bias_body_m[1].clone(),
        model.rotation_bias_rpy_rad[1].clone(),
    )

    model.reset(torch.tensor([0]))

    torch.testing.assert_close(model.range_std_m[1], env_one_state[0])
    torch.testing.assert_close(model.point_dropout_prob[1], env_one_state[1])
    torch.testing.assert_close(model.translation_bias_body_m[1], env_one_state[2])
    torch.testing.assert_close(model.rotation_bias_rpy_rad[1], env_one_state[3])
    assert model.last_scan_count.tolist() == [-1, 4]
    assert 0.01 <= model.range_std_m[0] <= 0.03
    assert 0.0 <= model.point_dropout_prob[0] <= 0.05


def test_same_scan_is_not_randomized_twice(monkeypatch):
    model = LidarNoiseModel(_cfg(range_std=(0.1, 0.1)), num_envs=1, device="cpu")
    hits, sensor, torso, quaternion, scan_count = _inputs()
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    calls = 0

    def counted_randn(value):
        nonlocal calls
        calls += 1
        return torch.ones_like(value)

    monkeypatch.setattr(torch, "randn_like", counted_randn)
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.ones_like(value))

    first_hits, _, first_ids = model.apply_new_scans(
        hits, sensor, torso, quaternion, scan_count, min_distance=0.1, max_distance=5.0
    )
    repeated_hits, _, repeated_ids = model.apply_new_scans(
        hits, sensor, torso, quaternion, scan_count, min_distance=0.1, max_distance=5.0
    )
    next_hits, _, next_ids = model.apply_new_scans(
        hits, sensor, torso, quaternion, scan_count + 1, min_distance=0.1, max_distance=5.0
    )

    assert calls == 2
    assert first_ids.tolist() == [0]
    assert repeated_ids.numel() == 0
    assert repeated_hits is hits
    assert next_ids.tolist() == [0]
    torch.testing.assert_close(first_hits, next_hits)


def test_seeded_models_are_reproducible():
    cfg = _cfg(
        range_std=(0.01, 0.03),
        dropout=(0.0, 0.05),
        translation=(-0.005, 0.005),
        rotation=(-0.2, 0.2),
    )
    hits, sensor, torso, quaternion, scan_count = _inputs(num_envs=2, num_rays=4)
    hits[..., 0] = 1.0

    torch.manual_seed(123)
    first = LidarNoiseModel(cfg, num_envs=2, device="cpu")
    first_hits, first_sensor, _ = first.apply_new_scans(
        hits, sensor, torso, quaternion, scan_count, min_distance=0.1, max_distance=5.0
    )
    torch.manual_seed(123)
    second = LidarNoiseModel(cfg, num_envs=2, device="cpu")
    second_hits, second_sensor, _ = second.apply_new_scans(
        hits, sensor, torso, quaternion, scan_count, min_distance=0.1, max_distance=5.0
    )

    torch.testing.assert_close(first_hits, second_hits, equal_nan=True)
    torch.testing.assert_close(first_sensor, second_sensor)


def test_renet_uses_ame_packet_noise_without_physical_lidar_noise():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    play_source = PLAY_PATH.read_text(encoding="utf-8")
    eval_source = EVAL_PATH.read_text(encoding="utf-8")
    assert "lidar=None" in cfg_source
    assert "LidarNoiseCfg" not in cfg_source
    assert "LidarNoiseModel" not in env_source
    assert "update_from_point_cloud(" not in env_source
    assert "PgttMapNoise(" in env_source
    assert '"--lp_noise"' in play_source
    assert "env_cfg.lp_height.noise.enable = bool(args_cli.lp_noise)" in play_source
    assert '"--eval_lp_noise"' in eval_source
    assert "env_cfg.lp_height.noise.force_active = bool(args_cli.eval_lp_noise)" in eval_source
