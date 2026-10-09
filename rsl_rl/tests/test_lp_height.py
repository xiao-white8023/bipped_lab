import ast
import json
from pathlib import Path

import pytest
import torch

from legged_lab.perception.lp_height import (
    HeightSnapshotHistory,
    LPHeightCfg,
    PgttMapNoise,
    grid_points,
    observation_contract,
    resolve_pgtt_height_cfg,
)


def _cfg(source="pgtt_full_atten"):
    return resolve_pgtt_height_cfg(LPHeightCfg(), source)


def _noise(source="pgtt_full_atten", num_envs=2, **kwargs):
    cfg = _cfg(source)
    cfg.noise.block_probability = 0
    for key, value in kwargs.items():
        setattr(cfg.noise, key, value)
    return PgttMapNoise(num_envs, cfg, "cpu")


def _apply(noise, count=1, valid=None, z=None, enable_noise=None):
    num_envs = noise.last_height.shape[0]
    num_rays = len(noise.nominal)
    if valid is None:
        valid = torch.ones(num_envs, num_rays, dtype=torch.bool)
    if z is None:
        z = torch.zeros(num_envs, num_rays)
    return noise.apply(
        z,
        valid,
        torch.ones(num_envs),
        torch.full((num_envs,), count),
        height_offset=0.5,
        height_scale=1.0,
        enable_noise=enable_noise,
    )


def test_full_grid_coordinates_shape_count_and_order():
    cfg = _cfg("pgtt_full_atten")
    xy = grid_points(cfg)
    assert cfg.shape == (11, 17)
    assert xy.shape == (11, 17, 2)
    torch.testing.assert_close(xy[0, 0], torch.tensor([-0.8, -0.5]))
    torch.testing.assert_close(xy[-1, -1], torch.tensor([0.8, 0.5]))
    torch.testing.assert_close(xy[0, :, 0], torch.linspace(-0.8, 0.8, 17))
    torch.testing.assert_close(xy[:, 0, 1], torch.linspace(-0.5, 0.5, 11))
    assert xy.numel() // 2 == 187




def test_retired_pgtt_name_and_invalid_value_are_rejected():
    with pytest.raises(ValueError, match="pgtt_full_atten"):
        resolve_pgtt_height_cfg(LPHeightCfg(), "pgtt")
    cfg = LPHeightCfg(invalid_height=-1.0)
    with pytest.raises(ValueError, match="outside"):
        resolve_pgtt_height_cfg(cfg, "pgtt_full_atten")


@pytest.mark.parametrize(
    "source,shape",
    [
        ("pgtt_full_atten", (11, 17)),
    ],
)
def test_disabled_noise_is_exact_and_no_hit_is_minus_1p1(source, shape):
    noise = _noise(source, enable=False)
    valid = torch.ones(2, len(noise.nominal), dtype=torch.bool)
    valid[0, :10] = False
    z = torch.zeros_like(valid, dtype=torch.float)
    z[0, :10] = float("inf")
    height, confidence = _apply(noise, valid=valid, z=z)
    assert height.shape == (2, *shape)
    assert torch.all(height.flatten(1)[0, :10] == -1.1)
    assert torch.all(confidence.flatten(1)[0, :10] == 0)
    assert torch.all(height.flatten(1)[1] == 0.5)
    assert torch.isfinite(height).all()
    assert torch.all(noise.delay == 0)
    torch.testing.assert_close(noise.xy[0], noise.nominal)


def test_same_packet_noise_is_not_resampled_and_block_persists():
    torch.manual_seed(2)
    noise = _noise(
        block_probability=1.0,
        block_duration_packets=(3, 3),
        dropout_range=(0.05, 0.05),
    )
    height, confidence = (value.clone() for value in _apply(noise, count=1))
    remaining = noise.block_remaining.clone()
    again, again_confidence = _apply(noise, count=1)
    torch.testing.assert_close(height, again)
    torch.testing.assert_close(confidence, again_confidence)
    torch.testing.assert_close(remaining, noise.block_remaining)
    _apply(noise, count=2)
    torch.testing.assert_close(noise.block_remaining, remaining - 1)


def test_clean_training_stage_bypasses_all_configured_pgtt_noise():
    torch.manual_seed(4)
    noise = _noise(
        enable=True,
        block_probability=1.0,
        dropout_range=(1.0, 1.0),
        height_std_range=(0.03, 0.03),
        height_bias_range=(0.02, 0.02),
        max_delay_packets=2,
    )
    valid = torch.ones(2, len(noise.nominal), dtype=torch.bool)
    z = torch.zeros_like(valid, dtype=torch.float)
    height, confidence = _apply(
        noise, count=1, valid=valid, z=z, enable_noise=False
    )
    torch.testing.assert_close(height, torch.full_like(height, 0.5))
    torch.testing.assert_close(confidence, torch.ones_like(confidence))

    noisy_height, noisy_confidence = _apply(
        noise, count=2, valid=valid, z=z, enable_noise=True
    )
    assert torch.any(noisy_height != 0.5)
    assert torch.any(noisy_confidence == 0)


def test_pgtt_noise_is_active_from_training_start_and_play_force_switch():
    from types import SimpleNamespace

    source = Path(__file__).parents[2] / "legged_lab/envs/g1/RENet_env.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pgtt_noise_is_active"
    )
    scope = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        scope,
    )
    enabled = scope["_pgtt_noise_is_active"]
    env = SimpleNamespace(
        pgtt_noise=object(),
        lp_runtime_cfg=SimpleNamespace(
            noise=SimpleNamespace(enable=True, force_active=False, start_iteration=0)
        ),
        renet_training_iteration=0,
    )
    assert enabled(env) is True
    env.lp_runtime_cfg.noise.enable = False
    assert enabled(env) is False
    env.lp_runtime_cfg.noise.enable = True
    env.lp_runtime_cfg.noise.force_active = True
    assert enabled(env) is True


def test_partial_noise_reset_preserves_other_environment():
    noise = _noise()
    _apply(noise)
    other = {
        name: value[1].clone()
        for name, value in vars(noise).items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2
    }
    noise.reset(torch.tensor([0]))
    for name, value in other.items():
        torch.testing.assert_close(getattr(noise, name)[1], value)


def test_delay_keeps_private_packets_but_policy_output_is_one_frame():
    cfg = _cfg("pgtt_full_atten")
    history = HeightSnapshotHistory(2, cfg, "cpu", max_delay=2)
    delay = torch.tensor([0, 2])
    for scan in range(1, 9):
        height = torch.full((2, 11, 17), float(scan))
        now = torch.full((2,), scan * 0.1)
        counts = torch.full((2,), scan)
        assert len(history.push(height, torch.ones_like(height), now, torch.zeros(2, 7), counts)) == 2
        assert len(history.push(height + 100, torch.ones_like(height), now, torch.zeros(2, 7), counts)) == 0
        output, confidence, ages = history.read(now, delay)
        assert output.shape == (2, 1, 11, 17)
        assert confidence.shape == output.shape and ages.shape == (2, 1)
        expected = torch.tensor([scan, max(1, scan - 2)], dtype=torch.float)
        torch.testing.assert_close(output[:, 0, 0, 0], expected)
        torch.testing.assert_close(ages[:, 0], (scan - expected) * 0.1)
    assert history.height.shape[1] == 3
    other = history.height[1].clone()
    history.reset(torch.tensor([0]))
    torch.testing.assert_close(history.height[1], other)




def test_depth_packet_clock_ignores_episode_randomization_and_duplicate_reads():
    # Execute the real environment method without importing Isaac Sim.
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    source = Path(__file__).parents[2] / "legged_lab/envs/g1/RENet_env.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_deepcamera_history")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)

    class FakeEnv:
        get_deepcamera_history = scope["get_deepcamera_history"]
        step_dt = 0.02
        depth_history_frames = 2

        def get_processed_deepcamera(self):
            self.reads += 1
            return torch.full((2, 3, 4), float(self.reads))

    env = FakeEnv()
    env.reads = 0
    env.cfg = SimpleNamespace(robot=SimpleNamespace(depth_update_interval=5))
    env.camera = SimpleNamespace(current_timestamps=torch.zeros(2))
    env.depth_last_capture_time = torch.full((2,), -torch.inf)
    env.depth_initialized = torch.zeros(2, dtype=torch.bool)
    env.depth_buffer = torch.zeros(2, 2, 3, 4)
    env.episode_length_buf = torch.tensor([711, 0])
    first = env.get_deepcamera_history().clone()
    assert env.reads == 1
    torch.testing.assert_close(first, torch.ones_like(first))
    env.episode_length_buf[:] = 20
    torch.testing.assert_close(first, env.get_deepcamera_history())
    assert env.reads == 1
    env.camera.current_timestamps += 0.1
    env.get_deepcamera_history()
    assert env.reads == 2
    assert torch.all(env.depth_buffer[:, 0] == 1) and torch.all(env.depth_buffer[:, 1] == 2)






@pytest.mark.parametrize("source", ["pgtt_front", "pgtt_full", "mid360", "pgtt", "unknown"])
def test_removed_height_sources_are_rejected(source):
    with pytest.raises(ValueError, match="pgtt_full_atten"):
        resolve_pgtt_height_cfg(LPHeightCfg(), source)
    with pytest.raises(ValueError, match="pgtt_full_atten"):
        observation_contract(LPHeightCfg(), source, height_offset=.5, height_scale=1.)


def test_atten_observation_contract_preserves_full_grid_and_target():
    contract = observation_contract(_cfg(), "pgtt_full_atten", height_offset=.5, height_scale=1.)
    assert json.loads(json.dumps(contract)) == contract
    assert contract["source"] == "pgtt_full_atten"
    assert contract["shape"] == [11, 17]
    assert contract["history_frames"] == 1
    assert contract["layout"] == "rectangular_full"
    assert contract["lp_target"] == "critic_11x17_full_187_fixed_clip"
    assert len(contract["sample_xy"]) == 11 and len(contract["sample_xy"][0]) == 17
