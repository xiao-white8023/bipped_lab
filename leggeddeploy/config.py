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

                self.policy_path = config["policy_path"]

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

                self.num_actions = config["num_actions"]
                self.num_obs = config["num_obs"]

                self.history_length = config["obs_history_length"]
                self.command_range = config["command_range"]
                
                # 步态相位配置 (可选)
                self.gait_phase_enable = False
                self.gait_phase_period = 0.8
                self.gait_phase_offset = 0.5
                if "gait_phase" in config:
                    gait_cfg = config["gait_phase"]
                    self.gait_phase_enable = gait_cfg.get("enable", False)
                    self.gait_phase_period = gait_cfg.get("period", 0.8)
                    self.gait_phase_offset = gait_cfg.get("offset", 0.5)

                self.num_policy_obs=config["num_policy_obs"]

                # 深度图配置
                self.depth_enable=config["depth"]["enable"]
                self.robot_ip=config["depth"]["robot_ip"]
                self.TCP_port = config["depth"]["port"]
                self.depth_min = config["depth"]["depth_min"]
                self.depth_max = config["depth"]["depth_max"]
                self.raw_h= config["depth"]["raw_h"]
                self.raw_w=config["depth"]["raw_w"]
                self.crop_up=config["depth"]["crop_up"]
                self.crop_down=config["depth"]["crop_down"]
                self.crop_left=config["depth"]["crop_left"]
                self.crop_right=config["depth"]["crop_right"]
                self.history_frames=config["depth"]["history_frames"]
                self.CMD_RAW=config["depth"]["CMD_RAW"]
                self.CMD_INTERRUPT=config["depth"]["CMD_INTERRUPT"]
                self.depth_update_interval=config["depth"]["depth_update_interval"]


                # 深度图显示配置
                self.show_raw_depth = config["depth"].get("show_raw_depth", False)
                self.show_pre_norm_depth = config["depth"].get("show_pre_norm_depth", False)
                self.show_policy_depth = config["depth"].get("show_policy_depth", False)

                self.raw_depth_window_name = config["depth"].get("raw_depth_window_name", "robot_raw_depth")
                self.pre_norm_depth_window_name = config["depth"].get("pre_norm_depth_window_name", "pre_norm_depth_m")
                self.policy_depth_window_name = config["depth"].get("policy_depth_window_name", "policy_depth_norm")

                self.raw_depth_display_scale = config["depth"].get("raw_depth_display_scale", 1)
                self.pre_norm_depth_display_scale = config["depth"].get("pre_norm_depth_display_scale", 10)
                self.policy_depth_display_scale = config["depth"].get("policy_depth_display_scale", 10)


            
