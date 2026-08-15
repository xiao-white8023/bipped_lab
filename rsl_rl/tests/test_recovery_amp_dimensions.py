from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.utils.motion_loader import AMPLoader


REPO_ROOT = Path(__file__).parents[2]
REMOVE_ANKLES_PATH = REPO_ROOT / "legged_lab" / "scripts" / "remove_locked_ankles.py"


def _load_remove_ankles_module():
    spec = importlib.util.spec_from_file_location("remove_locked_ankles", REMOVE_ANKLES_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_motion(path: Path, frames: np.ndarray, frame_duration: float = 0.1) -> None:
    path.write_text(
        json.dumps(
            {
                "LoopMode": "Wrap",
                "FrameDuration": frame_duration,
                "MotionWeight": 1.0,
                "Frames": frames.tolist(),
            }
        ),
        encoding="utf-8",
    )


def test_remove_locked_ankles_supports_58d_and_61d(tmp_path):
    module = _load_remove_ankles_module()
    rng = np.random.default_rng(7)

    locomotion_frames = rng.normal(size=(10, 58)).astype(np.float32)
    loco_input = tmp_path / "loco_input.txt"
    loco_output = tmp_path / "loco_output.txt"
    _write_motion(loco_input, locomotion_frames)
    module.convert_file(loco_input, loco_output)
    loco_reduced = np.asarray(json.loads(loco_output.read_text())["Frames"], dtype=np.float32)
    assert loco_reduced.shape == (10, 50)

    recovery_frames = rng.normal(size=(10, 61)).astype(np.float32)
    recovery_gravity = rng.normal(size=(10, 3)).astype(np.float32)
    recovery_gravity /= np.linalg.norm(recovery_gravity, axis=1, keepdims=True)
    recovery_frames[:, 58:61] = recovery_gravity
    rec_input = tmp_path / "rec_input.txt"
    rec_output = tmp_path / "rec_output.txt"
    _write_motion(rec_input, recovery_frames)
    module.convert_file(rec_input, rec_output)
    rec_reduced = np.asarray(json.loads(rec_output.read_text())["Frames"], dtype=np.float32)
    assert rec_reduced.shape == (10, 53)
    np.testing.assert_array_equal(rec_reduced[:, 50:53], recovery_frames[:, 58:61])


def test_amp_loader_preserves_requested_width_and_normalizes_recovery_interpolation(tmp_path):
    loco_frames = np.arange(3 * 50, dtype=np.float32).reshape(3, 50)
    loco_path = tmp_path / "loco.txt"
    _write_motion(loco_path, loco_frames)
    loco_loader = AMPLoader(
        device="cpu",
        time_between_frames=0.05,
        motion_files=[loco_path],
    )
    assert loco_loader.observation_dim == 50
    assert loco_loader.get_frame_at_time(0, 0.05).shape == (50,)
    assert loco_loader.get_frame_at_time_batch(np.array([0, 0]), np.array([0.05, 0.15])).shape == (2, 50)

    rec_frames = np.zeros((3, 53), dtype=np.float32)
    rec_frames[:, :50] = np.arange(3 * 50, dtype=np.float32).reshape(3, 50)
    rec_frames[:, 50:53] = np.asarray(
        [[0.0, 0.0, -1.0], [0.6, 0.0, -0.8], [0.0, 0.8, -0.6]],
        dtype=np.float32,
    )
    rec_path = tmp_path / "recovery.txt"
    _write_motion(rec_path, rec_frames)
    rec_loader = AMPLoader(
        device="cpu",
        time_between_frames=0.05,
        motion_files=[rec_path],
        frame_size=53,
        preload_transitions=True,
        num_preload_transitions=16,
    )
    assert rec_loader.observation_dim == 53
    samples = (
        rec_loader.get_frame_at_time(0, 0.05).unsqueeze(0),
        rec_loader.get_full_frame_at_time(0, 0.15).unsqueeze(0),
        rec_loader.get_frame_at_time_batch(np.array([0, 0]), np.array([0.05, 0.15])),
        rec_loader.get_full_frame_at_time_batch(np.array([0, 0]), np.array([0.05, 0.15])),
        rec_loader.preloaded_s,
        rec_loader.preloaded_s_next,
    )
    for sample in samples:
        assert sample.shape[-1] == 53
        torch.testing.assert_close(
            torch.linalg.vector_norm(sample[:, 50:53], dim=1),
            torch.ones(sample.shape[0]),
            atol=1e-6,
            rtol=1e-6,
        )


class _DummyRecoveryCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_critic = nn.Linear(1, 1)
        self.amp_critic = nn.Linear(1, 1)
        self.reg_critic = nn.Linear(1, 1)

    def forward(self, observations):
        return {
            "task": self.task_critic(observations),
            "amp": self.amp_critic(observations),
            "reg": self.reg_critic(observations),
        }


def test_renet_amp_accepts_100d_and_106d_discriminators_and_routes_disjoint_replay():
    algorithm = RENetAMPPPO(
        policy=nn.Linear(2, 1),
        discriminator=Discriminator(100, 0.3, [4], "cpu"),
        amp_data=object(),
        amp_normalizer=object(),
        recovery_discriminator=Discriminator(106, 1.0, [4], "cpu"),
        recovery_amp_data=object(),
        recovery_amp_normalizer=object(),
        recovery_critic=_DummyRecoveryCritic(),
        amp_replay_buffer_size=32,
        device="cpu",
    )
    assert algorithm.amp_storage_loco.states.shape == (32, 50)
    assert algorithm.amp_storage_recovery.states.shape == (32, 53)

    current = {
        "loco": torch.arange(8 * 50, dtype=torch.float32).reshape(8, 50),
        "recovery": torch.arange(8 * 53, dtype=torch.float32).reshape(8, 53),
    }
    next_bundle = {key: value + 1.0 for key, value in current.items()}
    recovery_mask = torch.tensor([False, False, False, False, True, True, True, True])
    algorithm._store_amp_transition(current, next_bundle, recovery_mask)

    assert algorithm.amp_storage_loco.num_samples == 4
    assert algorithm.amp_storage_recovery.num_samples == 4
    torch.testing.assert_close(algorithm.amp_storage_loco.states[:4], current["loco"][:4])
    torch.testing.assert_close(algorithm.amp_storage_recovery.states[:4], current["recovery"][4:])
