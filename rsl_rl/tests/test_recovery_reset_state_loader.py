from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


LOADER_PATH = (
    Path(__file__).parents[2]
    / "legged_lab"
    / "utils"
    / "recovery_reset_state_loader.py"
)
LOADER_SPEC = importlib.util.spec_from_file_location(
    "test_recovery_reset_state_loader_module", LOADER_PATH
)
assert LOADER_SPEC is not None and LOADER_SPEC.loader is not None
LOADER_MODULE = importlib.util.module_from_spec(LOADER_SPEC)
LOADER_SPEC.loader.exec_module(LOADER_MODULE)
RecoveryResetStateLoader = LOADER_MODULE.RecoveryResetStateLoader


JOINT_NAMES = np.asarray([f"joint_{index:02d}" for index in range(23)])
BODY_NAMES = np.asarray(
    ["pelvis", "body_01", "body_02", "torso_link"]
    + [f"body_{index:02d}" for index in range(4, 24)]
)


def _state_payload(value=0.0, num_frames=1):
    joint_pos = np.full((num_frames, 23), value, dtype=np.float32)
    joint_vel = np.full((num_frames, 23), value + 0.1, dtype=np.float32)
    body_pos_w = np.zeros((num_frames, 24, 3), dtype=np.float32)
    body_pos_w[:, 0] = np.asarray([value, value + 1.0, value + 0.2])
    body_quat_w = np.zeros((num_frames, 24, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 24, 3), dtype=np.float32)
    body_lin_vel_w[:, 0] = np.asarray([value + 0.3, 0.0, 0.0])
    body_ang_vel_w = np.zeros((num_frames, 24, 3), dtype=np.float32)
    body_ang_vel_w[:, 0] = np.asarray([0.0, value + 0.4, 0.0])
    return {
        "fps": np.asarray([50.0], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": JOINT_NAMES.copy(),
        "body_names": BODY_NAMES.copy(),
        "root_body_index": np.asarray(0, dtype=np.int64),
        "root_body_name": np.asarray("pelvis"),
        "quaternion_convention": np.asarray("wxyz"),
        "velocity_frame": np.asarray("world"),
        "body_state_frame": np.asarray("link_world"),
    }


def _write_state(path: Path, payload) -> None:
    np.savez(path, **payload)


def test_preloads_multiple_single_states_and_extracts_root(tmp_path):
    _write_state(tmp_path / "reset_state_0002.npz", _state_payload(2.0))
    _write_state(tmp_path / "reset_state_0001.npz", _state_payload(1.0))
    _write_state(tmp_path / "reset_state_0003.npz", _state_payload(3.0))

    loader = RecoveryResetStateLoader(tmp_path, device="cpu")
    assert loader.num_states == 3
    assert [path.name for path in loader.state_files] == [
        "reset_state_0001.npz",
        "reset_state_0002.npz",
        "reset_state_0003.npz",
    ]
    assert loader.joint_names == tuple(JOINT_NAMES.tolist())
    assert loader.body_names == tuple(BODY_NAMES.tolist())
    assert loader.root_body_index == 0
    assert loader.root_body_name == "pelvis"
    assert not hasattr(loader, "body_pos_w")
    assert not hasattr(loader, "body_quat_w")

    state = loader.get_state_batch(torch.tensor([2, 0]))
    expected_shapes = {
        "state_ids": (2,),
        "root_pos_w": (2, 3),
        "root_quat_w": (2, 4),
        "root_lin_vel_w": (2, 3),
        "root_ang_vel_w": (2, 3),
        "joint_pos": (2, 23),
        "joint_vel": (2, 23),
    }
    for key, expected_shape in expected_shapes.items():
        assert state[key].shape == expected_shape
        assert state[key].device.type == "cpu"
    assert state["root_pos_w"].dtype == torch.float32
    torch.testing.assert_close(
        state["root_pos_w"],
        torch.tensor([[3.0, 4.0, 3.2], [1.0, 2.0, 1.2]]),
    )
    torch.testing.assert_close(state["joint_pos"][0], torch.full((23,), 3.0))
    torch.testing.assert_close(state["joint_vel"][1], torch.full((23,), 1.1))


def test_uniform_sampling_uses_torch_randint(tmp_path, monkeypatch):
    for index in range(4):
        _write_state(
            tmp_path / f"reset_state_{index:04d}.npz", _state_payload(float(index))
        )
    loader = RecoveryResetStateLoader(tmp_path)
    calls = []

    def _fixed_randint(high, size, dtype, device):
        calls.append((high, size, dtype, device))
        return torch.tensor([3, 0, 2, 1, 3], dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randint", _fixed_randint)
    sample = loader.sample_state_batch(5)
    assert calls == [(4, (5,), torch.long, loader.device)]
    assert sample["state_ids"].tolist() == [3, 0, 2, 1, 3]
    assert sample["root_pos_w"].shape == (5, 3)
    assert sample["joint_pos"].shape == (5, 23)


def test_rejects_t_greater_than_one(tmp_path):
    _write_state(tmp_path / "reset_state_0001.npz", _state_payload(num_frames=2))
    with pytest.raises(ValueError, match=r"\(1, 23\)"):
        RecoveryResetStateLoader(tmp_path)


def test_rejects_wrong_quaternion_convention(tmp_path):
    payload = _state_payload()
    payload["quaternion_convention"] = np.asarray("xyzw")
    _write_state(tmp_path / "reset_state_0001.npz", payload)
    with pytest.raises(ValueError, match="quaternion_convention"):
        RecoveryResetStateLoader(tmp_path)


def test_rejects_non_unit_quaternion(tmp_path):
    payload = _state_payload()
    payload["body_quat_w"][0, 5] = np.asarray([2.0, 0.0, 0.0, 0.0])
    _write_state(tmp_path / "reset_state_0001.npz", payload)
    with pytest.raises(ValueError, match="quaternion norm"):
        RecoveryResetStateLoader(tmp_path)


def test_rejects_nan(tmp_path):
    payload = _state_payload()
    payload["joint_vel"][0, 7] = np.nan
    _write_state(tmp_path / "reset_state_0001.npz", payload)
    with pytest.raises(ValueError, match="NaN or Inf"):
        RecoveryResetStateLoader(tmp_path)


@pytest.mark.parametrize("metadata_key", ["joint_names", "body_names"])
def test_rejects_metadata_mismatch(tmp_path, metadata_key):
    first = _state_payload(1.0)
    second = _state_payload(2.0)
    second[metadata_key][[1, 2]] = second[metadata_key][[2, 1]]
    _write_state(tmp_path / "reset_state_0001.npz", first)
    _write_state(tmp_path / "reset_state_0002.npz", second)
    with pytest.raises(ValueError, match=rf"{metadata_key} order differs"):
        RecoveryResetStateLoader(tmp_path)


def test_rejects_nan_in_source_metadata(tmp_path):
    payload = _state_payload()
    payload["source_score"] = np.asarray(np.nan)
    _write_state(tmp_path / "reset_state_0001.npz", payload)
    with pytest.raises(ValueError, match="source_score contains NaN or Inf"):
        RecoveryResetStateLoader(tmp_path)


def test_optional_metadata_accessor_is_read_only_and_missing_is_none(tmp_path):
    first = _state_payload(1.0)
    first.update(
        {
            "source_file": np.asarray("fall_subject_1.npz"),
            "source_frame": np.asarray(321, dtype=np.int64),
            "source_time_s": np.asarray(6.42),
            "pelvis_height": np.asarray(0.24),
        }
    )
    _write_state(tmp_path / "reset_state_0001.npz", first)
    _write_state(tmp_path / "reset_state_0002.npz", _state_payload(2.0))

    loader = RecoveryResetStateLoader(tmp_path)
    metadata = loader.get_metadata(torch.tensor([0, 1]))
    assert metadata[0]["source_file"] == "fall_subject_1.npz"
    assert metadata[0]["source_frame"] == 321
    assert metadata[0]["source_time_s"] == pytest.approx(6.42)
    assert metadata[0]["pelvis_height"] == pytest.approx(0.24)
    assert metadata[1]["source_file"] is None
    assert metadata[1]["mean_joint_speed"] is None
    with pytest.raises(TypeError):
        metadata[0]["source_file"] = "changed.npz"


def test_missing_or_empty_directory_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_recovery_reset_dataset"):
        RecoveryResetStateLoader(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="build_recovery_reset_dataset"):
        RecoveryResetStateLoader(tmp_path)
