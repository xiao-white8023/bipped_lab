import math

import torch

from legged_lab.perception import RollingElevationMap


MAX_DISTANCE = 5.0


def _make_map(num_envs=1):
    return RollingElevationMap(
        num_envs=num_envs,
        device="cpu",
        history_frames=8,
        history_seconds=0.8,
        resolution=0.1,
        internal_size=(3.2, 3.2),
        output_size=(1.6, 1.0),
        invalid_height=-1.0,
        height_clip=(-1.0, 1.0),
        reprojection_chunk_size=2,
    )


def _misses(num_envs, num_rays):
    return torch.full((num_envs, num_rays, 3), MAX_DISTANCE)


def _poses(num_envs, *, x=0.0, y=0.0, z=1.0, yaw=0.0):
    torso = torch.tensor([[x, y, z]], dtype=torch.float32).repeat(num_envs, 1)
    sensor = torso.clone()
    sensor[:, 2] += 0.3
    yaws = torch.full((num_envs,), yaw, dtype=torch.float32)
    return torso, sensor, yaws


def _update(mapper, hits, torso, sensor, yaws, scan_count, timestamp):
    num_envs = hits.shape[0]
    return mapper.update_from_point_cloud(
        hits,
        sensor,
        torso,
        yaws,
        torch.full((num_envs,), scan_count, dtype=torch.long),
        torch.full((num_envs,), timestamp, dtype=torch.float32),
        min_distance=0.1,
        max_distance=MAX_DISTANCE,
    )


def _observe(mapper, torso, yaws, timestamp):
    return mapper.get_observation(
        torso,
        yaws,
        torch.full((torso.shape[0],), timestamp, dtype=torch.float32),
        height_offset=0.5,
        height_scale=1.0,
    )


def test_gap_and_miss_sentinels_remain_invalid_and_dangerous():
    mapper = _make_map()
    torso, sensor, yaws = _poses(1)
    hits = _misses(1, 4)
    hits[0, 1] = torch.tensor([float("nan"), 0.0, 0.0])
    hits[0, 2] = torch.tensor([float("inf"), 0.0, 0.0])
    hits[0, 3] = torch.tensor([0.0, 0.0, -10.0])

    _update(mapper, hits, torso, sensor, yaws, scan_count=1, timestamp=0.0)
    height, confidence = _observe(mapper, torso, yaws, timestamp=0.0)

    assert height.shape == (1, 11, 17)
    assert confidence.shape == (1, 11, 17)
    assert torch.all(height == -1.0)
    assert torch.count_nonzero(confidence) == 0
    assert torch.isfinite(height).all()
    assert torch.isfinite(confidence).all()


def test_ground_and_robot_self_return_share_max_height_aggregation():
    mapper = _make_map()
    torso, sensor, yaws = _poses(1)
    hits = _misses(1, 4)
    hits[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    hits[0, 1] = torch.tensor([0.01, 0.01, 0.8])

    _update(mapper, hits, torso, sensor, yaws, scan_count=1, timestamp=0.0)
    height, confidence = _observe(mapper, torso, yaws, timestamp=0.0)

    # The unclassified self-return at z=0.8 wins over ground z=0.0.
    assert torch.isclose(height[0, 5, 8], torch.tensor(-0.3), atol=1.0e-6)
    assert confidence[0, 5, 8] == 1.0


def test_same_sensor_packet_is_not_ingested_twice():
    mapper = _make_map()
    torso, sensor, yaws = _poses(1)
    first_hits = _misses(1, 1)
    first_hits[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    second_hits = _misses(1, 1)
    second_hits[0, 0] = torch.tensor([0.0, 0.0, 0.8])

    first_ids = _update(mapper, first_hits, torso, sensor, yaws, scan_count=1, timestamp=0.0)
    repeated_ids = _update(mapper, second_hits, torso, sensor, yaws, scan_count=1, timestamp=0.02)
    height, _ = _observe(mapper, torso, yaws, timestamp=0.02)

    assert first_ids.tolist() == [0]
    assert repeated_ids.numel() == 0
    assert mapper.frame_count.item() == 1
    assert torch.isclose(height[0, 5, 8], torch.tensor(0.5), atol=1.0e-6)


def test_translation_and_yaw_reproject_static_world_height():
    mapper = _make_map()
    torso_0, sensor_0, yaw_0 = _poses(1)
    hits = _misses(1, 1)
    hits[0, 0] = torch.tensor([0.5, 0.0, 0.0])
    _update(mapper, hits, torso_0, sensor_0, yaw_0, scan_count=1, timestamp=0.0)

    torso_1, sensor_1, yaw_1 = _poses(1, x=0.2, yaw=math.pi / 2.0)
    _update(
        mapper,
        _misses(1, 1),
        torso_1,
        sensor_1,
        yaw_1,
        scan_count=2,
        timestamp=0.1,
    )
    height, confidence = _observe(mapper, torso_1, yaw_1, timestamp=0.1)

    # World point (0.5, 0.0) is (x=0.0, y=-0.3) in the new torso-yaw frame.
    assert torch.isclose(height[0, 2, 8], torch.tensor(0.5), atol=1.0e-6)
    assert torch.isclose(confidence[0, 2, 8], torch.tensor(0.875), atol=1.0e-6)


def test_self_return_ghost_expires_when_ninth_packet_overwrites_ring_slot():
    mapper = _make_map()
    torso, sensor, yaws = _poses(1)
    for scan_index in range(1, 9):
        hits = _misses(1, 1)
        z = 0.8 if scan_index == 1 else 0.0
        hits[0, 0] = torch.tensor([0.0, 0.0, z])
        _update(
            mapper,
            hits,
            torso,
            sensor,
            yaws,
            scan_count=scan_index,
            timestamp=(scan_index - 1) * 0.1,
        )

    height_before, confidence_before = _observe(mapper, torso, yaws, timestamp=0.7)
    assert torch.isclose(height_before[0, 5, 8], torch.tensor(-0.3), atol=1.0e-6)
    assert torch.isclose(confidence_before[0, 5, 8], torch.tensor(0.125), atol=1.0e-6)

    ninth_hits = _misses(1, 1)
    ninth_hits[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    _update(mapper, ninth_hits, torso, sensor, yaws, scan_count=9, timestamp=0.8)
    height_after, confidence_after = _observe(mapper, torso, yaws, timestamp=0.8)

    assert mapper.frame_count.item() == 8
    assert torch.isclose(height_after[0, 5, 8], torch.tensor(0.5), atol=1.0e-6)
    assert confidence_after[0, 5, 8] == 1.0


def test_reset_is_per_environment():
    mapper = _make_map(num_envs=2)
    torso, sensor, yaws = _poses(2)
    hits = _misses(2, 1)
    hits[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    _update(mapper, hits, torso, sensor, yaws, scan_count=1, timestamp=0.0)

    mapper.reset(torch.tensor([0]))
    height, confidence = _observe(mapper, torso, yaws, timestamp=0.0)

    assert torch.all(height[0] == -1.0)
    assert torch.count_nonzero(confidence[0]) == 0
    assert torch.isclose(height[1, 5, 8], torch.tensor(0.5), atol=1.0e-6)
    assert confidence[1, 5, 8] == 1.0

