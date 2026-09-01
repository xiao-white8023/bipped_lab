from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

import pytest
import torch

from rsl_rl.utils import RecoveryNPZMotionLoader


MOTION_DIR = (
    REPO_ROOT
    / "legged_lab"
    / "envs"
    / "g1"
    / "datasets"
    / "npz"
    / "recovery_candidates_v3"
)
REQUIRED_BODIES = (
    "pelvis",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)
EXPECTED_SHAPES = {
    "joint_pos": (512, 23),
    "joint_vel": (512, 23),
    "body_pos_w": (512, 24, 3),
    "body_quat_w": (512, 24, 4),
    "body_lin_vel_w": (512, 24, 3),
    "body_ang_vel_w": (512, 24, 3),
}
STATE_FIELDS = tuple(EXPECTED_SHAPES)


@pytest.fixture(scope="module")
def loader():
    return RecoveryNPZMotionLoader(motion_dir=MOTION_DIR, device="cpu")


def test_loads_real_recovery_dataset_and_metadata(loader):
    motion_files = sorted(MOTION_DIR.glob("*.npz"))

    assert loader.num_motions == len(motion_files)
    assert loader.num_joints == 23
    assert loader.num_bodies == 24
    assert loader.fps == 50.0
    assert loader.root_body_name == "pelvis"
    assert [clip.path for clip in loader.clips] == motion_files

    indices = loader.body_indices(REQUIRED_BODIES)
    assert indices.shape == (len(REQUIRED_BODIES),)
    assert indices.dtype == torch.long
    assert indices.device == loader.device
    assert [loader.body_names[index] for index in indices.tolist()] == list(REQUIRED_BODIES)


def test_get_frame_batch_reads_exact_integer_frames_including_last(loader):
    motion_ids = torch.tensor(
        [0, loader.num_motions - 1, 0], dtype=torch.long, device=loader.device
    )
    frame_ids = torch.tensor(
        [0, loader.clips[-1].num_frames - 1, loader.clips[0].num_frames - 1],
        dtype=torch.long,
        device=loader.device,
    )

    sample = loader.get_frame_batch(motion_ids, frame_ids)
    torch.testing.assert_close(sample["motion_ids"], motion_ids, rtol=0.0, atol=0.0)
    torch.testing.assert_close(sample["frame_ids"], frame_ids, rtol=0.0, atol=0.0)
    for batch_index, (motion_id, frame_id) in enumerate(
        zip(motion_ids.tolist(), frame_ids.tolist())
    ):
        clip = loader.clips[motion_id]
        for field in STATE_FIELDS:
            torch.testing.assert_close(
                sample[field][batch_index],
                getattr(clip, field)[frame_id],
                rtol=0.0,
                atol=0.0,
            )


def test_get_frame_batch_rejects_invalid_ids(loader):
    with pytest.raises(ValueError, match="motion_ids"):
        loader.get_frame_batch([loader.num_motions], [0])
    with pytest.raises(IndexError, match="out-of-range"):
        loader.get_frame_batch([0], [loader.clips[0].num_frames])
    with pytest.raises(IndexError, match="out-of-range"):
        loader.get_frame_batch([0], [-1])
    with pytest.raises(ValueError, match="shape"):
        loader.get_frame_batch([0, 1], [0])


def test_sample_frame_batch_uses_global_integer_frame_indexing(loader, monkeypatch):
    def deterministic_randint(high, size, *, dtype, device):
        assert high == loader.total_frames
        assert size == (loader.total_frames,)
        assert dtype == torch.long
        assert device == loader.device
        return torch.arange(loader.total_frames, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randint", deterministic_randint)
    sample = loader.sample_frame_batch(loader.total_frames)
    expected_motion_ids = torch.repeat_interleave(
        torch.arange(loader.num_motions, device=loader.device),
        loader.motion_num_frames,
    )
    expected_frame_ids = torch.cat(
        [
            torch.arange(clip.num_frames, device=loader.device)
            for clip in loader.clips
        ]
    )
    torch.testing.assert_close(
        sample["motion_ids"], expected_motion_ids, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        sample["frame_ids"], expected_frame_ids, rtol=0.0, atol=0.0
    )
    assert torch.equal(
        sample["frame_ids"][loader.motion_frame_ends - 1],
        loader.motion_num_frames - 1,
    )


@pytest.mark.parametrize("transition_dt", [0.01, 0.02])
def test_sample_transition_batch_shapes_finiteness_and_quaternions(loader, transition_dt):
    state, next_state = loader.sample_transition_batch(512, transition_dt=transition_dt)

    for sample in (state, next_state):
        assert set(sample) == set(EXPECTED_SHAPES)
        for field, expected_shape in EXPECTED_SHAPES.items():
            assert sample[field].shape == expected_shape
            assert sample[field].dtype == torch.float32
            assert sample[field].device == loader.device
            assert torch.isfinite(sample[field]).all()

        quaternion_error = torch.max(
            torch.abs(torch.linalg.vector_norm(sample["body_quat_w"], dim=-1) - 1.0)
        )
        assert quaternion_error.item() < 1e-4


if __name__ == "__main__":
    smoke_loader = RecoveryNPZMotionLoader(motion_dir=MOTION_DIR, device="cpu")
    test_loads_real_recovery_dataset_and_metadata(smoke_loader)
    for smoke_transition_dt in (0.01, 0.02):
        test_sample_transition_batch_shapes_finiteness_and_quaternions(
            smoke_loader, smoke_transition_dt
        )
    print(smoke_loader.summary())
    print("Recovery NPZ motion loader smoke test passed for transition_dt=0.01 and 0.02.")
