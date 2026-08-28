from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_DIR = REPO_ROOT / "legged_lab" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_recovery_reset_dataset import (  # noqa: E402
    CORE_ARRAY_KEYS,
    METADATA_KEYS,
    FallenRun,
    SelectionConfig,
    build_reset_dataset,
    load_source_npz,
    select_reset_states,
)


@pytest.fixture
def synthetic_source(tmp_path):
    num_frames = 90
    joint_names = np.asarray([f"joint_{index:02d}" for index in range(23)])
    body_names = np.asarray(
        ["pelvis", "body_01", "body_02", "torso_link"]
        + [f"body_{index:02d}" for index in range(4, 24)]
    )

    joint_pos = np.arange(num_frames * 23, dtype=np.float32).reshape(num_frames, 23)
    joint_vel = np.zeros((num_frames, 23), dtype=np.float32)
    body_pos_w = np.zeros((num_frames, 24, 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = 0.80
    body_quat_w = np.zeros((num_frames, 24, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 24, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, 24, 3), dtype=np.float32)

    # One valid 50-frame run and one 10-frame run that is shorter than 0.5 s.
    fallen_frames = np.r_[10:60, 70:80]
    body_pos_w[fallen_frames, 0, 2] = 0.20
    half_sqrt = np.float32(2.0**-0.5)
    body_quat_w[fallen_frames, 3] = np.asarray(
        [half_sqrt, half_sqrt, 0.0, 0.0], dtype=np.float32
    )

    # Individually reject stable frames with each velocity threshold.
    body_lin_vel_w[25, 0, 0] = 0.75
    body_ang_vel_w[30, 0, 1] = 2.00
    joint_vel[35] = 2.00

    payload = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": joint_names,
        "body_names": body_names,
        "root_body_index": np.asarray(0, dtype=np.int64),
        "root_body_name": np.asarray("pelvis"),
        "quaternion_convention": np.asarray("wxyz"),
        "velocity_frame": np.asarray("world"),
        "body_state_frame": np.asarray("link_world"),
    }
    source_path = tmp_path / "synthetic_whole_body.npz"
    np.savez(source_path, **payload)
    return source_path, payload


def test_fallen_runs_stable_window_velocity_filter_and_thinning(synthetic_source):
    source_path, _ = synthetic_source
    source = load_source_npz(source_path)
    selection = select_reset_states(source, SelectionConfig())

    assert selection.fallen_runs == (FallenRun(10, 60),)
    assert len(selection.stable_candidates) == 41
    assert len(selection.candidates_before_thinning) == 38
    velocity_frames = {
        candidate.frame for candidate in selection.candidates_before_thinning
    }
    assert not {25, 30, 35} & velocity_frames

    saved_frames = [candidate.frame for candidate in selection.saved_candidates]
    assert saved_frames == [15, 20, 26, 31, 36, 41, 46, 51]
    assert np.all(np.diff(saved_frames) >= 5)
    for frame in saved_frames:
        assert selection.fallen_mask[frame - 5 : frame + 5].all()


def test_one_state_per_file_t_dimension_and_metadata_preservation(
    synthetic_source, tmp_path
):
    source_path, source_payload = synthetic_source
    output_dir = tmp_path / "reset_states"
    result = build_reset_dataset(
        [source_path], output_dir=output_dir, config=SelectionConfig()
    )

    state_files = sorted(output_dir.glob("reset_state_*.npz"))
    assert len(state_files) == 8
    assert len(result.saved_states) == 8
    assert result.csv_path == output_dir.resolve() / "reset_selection.csv"

    with result.csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(state_files)
    assert [row["output_file"] for row in rows] == [path.name for path in state_files]

    expected_shapes = {
        "joint_pos": (1, 23),
        "joint_vel": (1, 23),
        "body_pos_w": (1, 24, 3),
        "body_quat_w": (1, 24, 4),
        "body_lin_vel_w": (1, 24, 3),
        "body_ang_vel_w": (1, 24, 3),
    }
    for state_path in state_files:
        with np.load(state_path, allow_pickle=False) as state:
            source_frame = int(np.asarray(state["source_frame"]).reshape(-1)[0])
            for key in CORE_ARRAY_KEYS:
                assert state[key].shape == expected_shapes[key]
                np.testing.assert_array_equal(
                    state[key][0], source_payload[key][source_frame]
                )
            np.testing.assert_array_equal(state["fps"], source_payload["fps"])
            for key in METADATA_KEYS:
                np.testing.assert_array_equal(state[key], source_payload[key])
            assert str(np.asarray(state["source_file"]).reshape(-1)[0]) == source_path.name
            assert int(state["fallen_run_start_frame"]) == 10
            assert int(state["fallen_run_end_frame"]) == 59


def test_builder_rejects_inconsistent_name_order(synthetic_source, tmp_path):
    source_path, payload = synthetic_source
    second_payload = {key: np.asarray(value).copy() for key, value in payload.items()}
    second_payload["joint_names"][[0, 1]] = second_payload["joint_names"][[1, 0]]
    second_path = tmp_path / "second_whole_body.npz"
    np.savez(second_path, **second_payload)

    with pytest.raises(ValueError, match="order differs"):
        build_reset_dataset(
            [source_path, second_path],
            output_dir=tmp_path / "invalid_output",
            config=SelectionConfig(),
        )
