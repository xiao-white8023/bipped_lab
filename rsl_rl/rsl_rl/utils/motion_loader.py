import glob
import json

import numpy as np
import torch


class AMPLoader:
    '''
    这里的位置和env里的visualize_motion的位置完全一样 关节位置————>关节速度————>末端执行器
    '''
    JOINT_POS_SIZE = 29 # 29

    JOINT_VEL_SIZE =29 # 29

    END_EFFECTOR_POS_SIZE = 12 # 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),  # glob.glob("datasets/motion_amp_expert/*") 会返回该目录下所有文件的路径列表
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        '''
        enumerate(motion_files)  enumerate:在遍历的时候同时给数据加上从0开始的编号 
        '''
        for i, motion_file in enumerate(motion_files):
            '''
            split(".") 会以 . 为分隔符，将字符串切分为列表
            '''
            self.trajectory_names.append(motion_file.split(".")[0])  # 将文件名存入对应的存储区
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(
                    torch.tensor(motion_data[:, : AMPLoader.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(motion_data[:, : AMPLoader.END_POS_END_IDX], dtype=torch.float32, device=device)
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"])) # 有可能有多个文件，这个权重就是每个文件被抽到的概率
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration  # 计算有多少秒
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0])) # 统计一共有多少帧

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        # 所有的数据集都已经保存完了  把每一个数据集的权重进行归一化处理
        '''
        walk.json: 0.5
        run.json: 0.5
        jump.json: 1.0
        
        归一化后（总和为 0.5+0.5+1.0=2.0），实际采样概率为：
        walk.json: 0.5/2.0 = 25%
        run.json: 0.5/2.0 = 25%
        jump.json: 1.0/2.0 = 50%

        则walk被抽到的概率是0.25 run是0.25 jump是0.5
        '''
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights) 
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)  # 转化为数组
        self.trajectory_lens = np.array(self.trajectory_lens)  #转化为数组
        self.trajectory_num_frames = np.array(self.trajectory_num_frames) # 转化为数组

        # Preload transitions.
        '''
        作用是提前将强化学习训练所需的 “当前状态 - 下一状态” 对（s, s'）批量加载到内存 / 显存中，避免训练时反复采样计算，大幅提升效率。
        '''
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    # 这个就是看选哪个数据文件中的数据
    '''
    例如有两个数据集 self.trajectory_idxs=[0,1,2]
    则这个函数返回的就是：
    [0,2,1,2,0,1,0,1,2,...]  长度就是size 代表了 第一条数据从第一个数据集中选取 第二条数据从第三个数据集中产生
    '''
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

        '''
        np.random.uniform(size=len(traj_idxs)) 生成和traj_idxs一样长度的在（0，1）之间的随机数
        '''
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

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs] # 把 “第几秒” 转换成 “在轨迹中的百分比位置
        n = self.trajectory_num_frames[traj_idxs]   # 每一行数据集对应的总的帧数
        
        # 将百分比转化为对应的帧  分别是向下取整 向上取正
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device
        )
        # traj_idxs可能是[0,1,0,2,1] 则经过set(几何操作)之后 就变成了{0,1,2} 去重
        for traj_idx in set(traj_idxs): 
            trajectory = self.trajectories_full[traj_idx]
            # 对 traj_idxs 数组中的每一个元素，逐个和 traj_idx 比较是否相等，最终生成一个和 traj_idxs 长度相同的布尔数组（True 表示相等，False 表示不相等
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
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

        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX]
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
        return self.trajectories[0].shape[1] # 如果这样写 那就要保证所有数据集的维度都一样

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]
