# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import numpy as np
import yaml

class Config:
    def __init__(self, file_path) -> None:
            with open(file_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

                self.control_dt = config["control_dt"]

                self.msg_type = config["msg_type"]
                self.imu_type = config["imu_type"]

                self.weak_motor = []
                if "weak_motor" in config:
                    self.weak_motor = config["weak_motor"]

                self.lowcmd_topic = config["lowcmd_topic"]
                self.lowstate_topic = config["lowstate_topic"]
                self.right_arm_lowcmd = config["right_arm_topic"]

                self.policy_path = config["policy_path"]

                self.action2motor_idx = config["action2motor_idx"]

                # 15维 action 在 IsaacLab 29维关节顺序中对应的位置
                # 用于把 policy 输出的 15维动作塞回 29维 target_q
                self.action_joint_ids = np.array(config["action_joint_ids"], dtype=np.int64)

                # IsaacLab 29维关节顺序 -> 真机 motor index
                self.joint2motor_idx = config["joint2motor_idx"]

                self.kps = config["kps"]
                self.kds = config["kds"]
                self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

                if "torso_idx" in config:
                    self.torso_idx = config["torso_idx"]

                self.ang_vel_scale = config["ang_vel_scale"]
                self.dof_pos_scale = config["dof_pos_scale"]
                self.dof_vel_scale = config["dof_vel_scale"]
                self.action_scale = config["action_scale"]
                self.command_scale = np.array(config["command_scale"], dtype=np.float32)
                
                self.num_joints= config["num_joints"]
                self.num_actions = config["num_actions"] # 15
                self.num_obs = config["num_obs"] # 单帧观测 81

                self.history_length = config["obs_history_length"]
                self.command_range = config["command_range"]
                # 高度命令参数
                height_cfg = config["height_command"]
                self.stand_height = float(height_cfg["stand_height"])
                self.squat_height = float(height_cfg["squat_height"])
                self.max_delta_per_s = float(height_cfg["max_delta_per_s"])
                
                self.num_policy_obs=config["num_policy_obs"]




            
