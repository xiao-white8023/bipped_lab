import glob
import json

import numpy as np
import torch

class AMPLoaderDisplay:
    '''
    # tienkung 
    JOINT_POS_SIZE = 26

    JOINT_VEL_SIZE = 26

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    ROOT_STATES_NUM = 6
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE 
    '''
    
    '''G1 29dof'''
    JOINT_POS_SIZE = 35

    JOINT_VEL_SIZE = 35

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    ROOT_STATES_NUM = 6
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE 

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"), # glob 模块用于匹配文件路径，这里会获取该目录下所有文件的完整路径，作为默认的运动数据文件列表
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames # 两帧之间的间隔

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds. 单位是时间
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = [] # 总的帧数

        for i, motion_file in enumerate(motion_files):
        # 这两行代码是在 遍历所有运动数据文件的同时，提取每个文件的「轨迹名称」（去掉文件后缀），并将其存储到类的实例列表 self.trajectory_names 中
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"]) # 转化为2维数组

                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, : AMPLoaderDisplay.JOINT_VEL_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : AMPLoaderDisplay.JOINT_VEL_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration # 计算总时长
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f" 数据集总共是 {traj_len}s. 模型文件是 {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        # 归一化后的权重数组之和为 1.0，每个值对应「该轨迹被采样的概率」
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        # 每一个文件的帧率 转化为数组的形式 方便后续使用
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        # 每一个文件的总时长 转化为数组的形式
        self.trajectory_lens = np.array(self.trajectory_lens)
        # 每一个文件的总帧数 转化为数组的形式
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print("Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")
        # 按行拼接成一个大的张量
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    '''
    它的核心作用是根据「轨迹索引」和「指定时间点」，通过「时间归一化→计算相邻帧索引→边界保护→线性插值混合」，
    返回一个平滑的运动帧数据，解决了「指定时间点不恰好对应轨迹中整帧」的问题，让获取的帧数据更连续、无跳变
    '''
    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        # 传入的是时间在总时间中的占比
        p = float(time) / self.trajectory_lens[traj_idx]
        # 数据集的总帧数
        n = self.trajectories_full[traj_idx].shape[0]
        # p * n 可以得到在该时间下的 数据对应的是哪一帧 如果对应的帧数恰好是小数 比如 50.5帧
        # np.floor(p * n) 向下取整 变成第50帧   np.ceil(p * n) 向上取整 变成第51帧 为后续的插值做处理
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        # 防止帧数越界
        idx_low = min(idx_low, n - 1)
        idx_high = min(idx_high, n - 1)
        # 开始帧的数据
        frame_start = self.trajectories_full[traj_idx][idx_low]
        # 结束帧的数据
        frame_end = self.trajectories_full[traj_idx][idx_high]
        # 用当前帧 减 前一帧 得到一个小数 这个小数 就是在插值的过程中 前后帧的比重
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs),
            AMPLoaderDisplay.JOINT_VEL_END_IDX - AMPLoaderDisplay.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs),
            AMPLoaderDisplay.JOINT_VEL_END_IDX - AMPLoaderDisplay.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """
        joints0, joints1 = AMPLoaderDisplay.get_joint_pose(frame0), AMPLoaderDisplay.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoaderDisplay.get_joint_vel(frame0), AMPLoaderDisplay.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]
                s_next = self.preloaded_s_next[
                    idxs, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX
                ]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))

                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoaderDisplay.JOINT_VEL_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoaderDisplay.JOINT_VEL_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]
