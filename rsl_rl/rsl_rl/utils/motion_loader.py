"""AMP expert-motion loader for the Unitree G1 23-DoF configuration.

Expected expert frame layout (58 values):
    G1 joint positions (23)
    G1 joint velocities (23)
    end-effector positions in the root frame (12)

The 12 end-effector values are expected in the same order produced by the
current G1 environment:
    left hand, right hand, left foot, right foot.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch


class AMPLoader:
    """G1 AMP loader.

    Frame layout:
        joint positions: 19
        joint velocities: 19
        end-effector positions: 12
        total: 50
    """

    JOINT_POS_SIZE = 19
    JOINT_VEL_SIZE = 19
    END_EFFECTOR_POS_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = (
        JOINT_POSE_START_IDX + JOINT_POS_SIZE
    )  # 19

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = (
        JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    )  # 38

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = (
        END_POS_START_IDX + END_EFFECTOR_POS_SIZE
    )  # 50

    FRAME_SIZE = END_POS_END_IDX

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1_000_000,
        motion_files=None,
    ):
        self.device = device
        self.time_between_frames = float(time_between_frames)

        if motion_files is None:
            search_root = data_dir or "datasets/motion_amp_expert"
            motion_files = sorted(glob.glob(str(Path(search_root) / "*")))
        else:
            motion_files = list(motion_files)

        if not motion_files:
            raise ValueError("No AMP expert motion files were provided or found.")

        self.trajectories: list[torch.Tensor] = []
        self.trajectories_full: list[torch.Tensor] = []
        self.trajectory_names: list[str] = []
        self.trajectory_idxs: list[int] = []
        self.trajectory_lens: list[float] = []
        self.trajectory_weights: list[float] = []
        self.trajectory_frame_durations: list[float] = []
        self.trajectory_num_frames: list[int] = []

        for motion_idx, motion_file in enumerate(motion_files):
            path = Path(motion_file)
            with path.open("r", encoding="utf-8") as file:
                motion_json = json.load(file)

            motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)
            self._validate_motion_data(motion_data, path)

            frame_duration = float(motion_json["FrameDuration"])
            if not np.isfinite(frame_duration) or frame_duration <= 0.0:
                raise ValueError(
                    f"FrameDuration must be positive and finite, got {frame_duration} in {path}."
                )

            trajectory = torch.as_tensor(motion_data, dtype=torch.float32, device=device)
            num_frames = int(motion_data.shape[0])
            trajectory_len = (num_frames - 1) * frame_duration

            self.trajectories.append(trajectory)
            self.trajectories_full.append(trajectory)
            self.trajectory_names.append(path.stem)
            self.trajectory_idxs.append(motion_idx)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_lens.append(trajectory_len)
            self.trajectory_num_frames.append(num_frames)

            print(
                f"Loaded AMP motion '{path}': frames={num_frames}, dim={motion_data.shape[1]}, "
                f"duration={trajectory_len:.6f}s, fps={1.0 / frame_duration:.6f}."
            )

        weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("MotionWeight values must be finite, non-negative, and sum to more than zero.")

        self.trajectory_weights = weights / weights.sum()
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float64)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float64)
        self.trajectory_num_frames = np.asarray(self.trajectory_num_frames, dtype=np.int64)

        self.preload_transitions = bool(preload_transitions)
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} AMP transitions.")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
            print("Finished preloading AMP transitions.")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    @classmethod
    def _validate_motion_data(cls, motion_data: np.ndarray, path: Path) -> None:
        if motion_data.ndim != 2:
            raise ValueError(f"Motion Frames must be a 2-D array, got {motion_data.shape} in {path}.")
        if motion_data.shape[0] < 2:
            raise ValueError(f"Motion must contain at least two frames: {path}.")
        if motion_data.shape[1] != cls.FRAME_SIZE:
            raise ValueError(
                f"G1 23-DoF AMP expert frames must contain {cls.FRAME_SIZE} values, "
                f"but {path} contains {motion_data.shape[1]}."
            )
        if not np.isfinite(motion_data).all():
            raise ValueError(f"Motion contains NaN or Inf values: {path}.")

    def weighted_traj_idx_sample(self):
        return int(np.random.choice(self.trajectory_idxs, p=self.trajectory_weights))

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(
            self.trajectory_idxs,
            size=int(size),
            p=self.trajectory_weights,
            replace=True,
        ).astype(np.int64)

    def traj_time_sample(self, traj_idx):
        traj_idx = int(traj_idx)
        latest_start = max(0.0, self.trajectory_lens[traj_idx] - self.time_between_frames)
        return float(np.random.uniform(0.0, latest_start)) if latest_start > 0.0 else 0.0

    def traj_time_sample_batch(self, traj_idxs):
        traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
        latest_starts = np.maximum(0.0, self.trajectory_lens[traj_idxs] - self.time_between_frames)
        return np.random.uniform(size=traj_idxs.shape[0]) * latest_starts

    @staticmethod
    def slerp(frame1, frame2, blend):
        # These frames contain scalar joint/Cartesian features, not quaternions.
        return (1.0 - blend) * frame1 + blend * frame2

    def _frame_indices(self, traj_idx: int, time: float) -> tuple[int, int, float]:
        traj_idx = int(traj_idx)
        frame_duration = float(self.trajectory_frame_durations[traj_idx])
        num_frames = int(self.trajectory_num_frames[traj_idx])
        max_time = float(self.trajectory_lens[traj_idx])

        clipped_time = float(np.clip(float(time), 0.0, max_time))
        frame_position = clipped_time / frame_duration
        idx_low = min(int(np.floor(frame_position)), num_frames - 1)
        idx_high = min(idx_low + 1, num_frames - 1)
        blend = float(frame_position - idx_low) if idx_high > idx_low else 0.0
        return idx_low, idx_high, blend

    def _frame_indices_batch(self, traj_idxs, times):
        traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        if traj_idxs.shape != times.shape:
            raise ValueError(f"traj_idxs and times must have the same shape: {traj_idxs.shape} != {times.shape}")

        frame_durations = self.trajectory_frame_durations[traj_idxs]
        num_frames = self.trajectory_num_frames[traj_idxs]
        max_times = self.trajectory_lens[traj_idxs]
        clipped_times = np.clip(times, 0.0, max_times)

        frame_positions = clipped_times / frame_durations
        idx_low = np.floor(frame_positions).astype(np.int64)
        idx_low = np.minimum(idx_low, num_frames - 1)
        idx_high = np.minimum(idx_low + 1, num_frames - 1)
        blend = np.where(idx_high > idx_low, frame_positions - idx_low, 0.0)
        return idx_low, idx_high, blend.astype(np.float32)

    def get_trajectory(self, traj_idx):
        return self.trajectories_full[int(traj_idx)]

    def get_frame_at_time(self, traj_idx, time):
        idx_low, idx_high, blend = self._frame_indices(traj_idx, time)
        frame_start = self.trajectories[int(traj_idx)][idx_low]
        frame_end = self.trajectories[int(traj_idx)][idx_high]
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        return self._get_frames_at_time_batch(self.trajectories, traj_idxs, times)

    def get_full_frame_at_time(self, traj_idx, time):
        idx_low, idx_high, blend = self._frame_indices(traj_idx, time)
        frame_start = self.trajectories_full[int(traj_idx)][idx_low]
        frame_end = self.trajectories_full[int(traj_idx)][idx_high]
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        return self._get_frames_at_time_batch(self.trajectories_full, traj_idxs, times)

    def _get_frames_at_time_batch(self, trajectories, traj_idxs, times):
        traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
        idx_low, idx_high, blend = self._frame_indices_batch(traj_idxs, times)
        batch_size = int(traj_idxs.shape[0])

        frame_starts = torch.empty((batch_size, self.FRAME_SIZE), dtype=torch.float32, device=self.device)
        frame_ends = torch.empty_like(frame_starts)

        for traj_idx in np.unique(traj_idxs):
            mask = traj_idxs == traj_idx
            batch_indices = np.flatnonzero(mask)
            batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
            low_indices_t = torch.as_tensor(idx_low[mask], dtype=torch.long, device=self.device)
            high_indices_t = torch.as_tensor(idx_high[mask], dtype=torch.long, device=self.device)
            trajectory = trajectories[int(traj_idx)]
            frame_starts[batch_indices_t] = trajectory[low_indices_t]
            frame_ends[batch_indices_t] = trajectory[high_indices_t]

        blend_tensor = torch.as_tensor(blend, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return self.slerp(frame_starts, frame_ends, blend_tensor)

    def get_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        return self.get_frame_at_time(traj_idx, self.traj_time_sample(traj_idx))

    def get_full_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        return self.get_full_frame_at_time(traj_idx, self.traj_time_sample(traj_idx))

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=int(num_frames))
            return self.preloaded_s[idxs]
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        joint_pos = self.slerp(self.get_joint_pose(frame0), self.get_joint_pose(frame1), blend)
        joint_vel = self.slerp(self.get_joint_vel(frame0), self.get_joint_vel(frame1), blend)
        end_pos = self.slerp(self.get_end_pos(frame0), self.get_end_pos(frame1), blend)
        return torch.cat([joint_pos, joint_vel, end_pos], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Yield batches of expert transitions ``(s, s_next)``."""
        for _ in range(int(num_mini_batch)):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=int(mini_batch_size))
                s = self.preloaded_s[idxs]
                s_next = self.preloaded_s_next[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            yield s, s_next

    @property
    def observation_dim(self):
        return self.FRAME_SIZE

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    @staticmethod
    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]