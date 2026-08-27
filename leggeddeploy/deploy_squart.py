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

import sys
import time
from threading import Lock

import numpy as np
import torch

# Unitree SDK2 的 DDS 通信接口
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher, # 创建一个发布器，用来往机器人发 LowCmd 控制命令
    ChannelSubscriber, # 创建一个订阅器，用来接收机器人的 LowState 状态，并且每次收到状态包时调用 self.LowStateHandler
)

'''
unitree_hg_msg_dds__LowCmd_()
        ↓
创建一包具体的 LowCmd 数据 要发布的信息会写进去
        ↓
self.low_cmd

LowCmdHG  （要发布的数据的数据类型）
        ↓
告诉 DDS：这个 topic 的数据类型是 hg LowCmd

ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        ↓
创建 rt/lowcmd 的发布通道
        ↓
self.lowcmd_publisher_

self.lowcmd_publisher_.Write(self.low_cmd)
        ↓
把 self.low_cmd 这包具体数据发出去


'''
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_, # 默认初始化好的消息结构构造函数
    unitree_go_msg_dds__LowState_, # 机器人当前状态
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config_squart import Config
from depth.depth_client import DepthClient
from squat_action_filter import FirstOrderLowPassFilter
from squat_policy_adapter import resolve_squat_policy_layout


RIGHT_ARM_MOTOR_IDS = range(22, 29)


class Controller:
    def __init__(self, config: Config, net: str) -> None:

        ChannelFactoryInitialize(0, net) # 初始化 Unitree SDK2  初始化全局 DDS 环境的通信系统，并指定用哪张网卡连接机器人。初始化 Unitree SDK2 的 DDS 通信环境，让你的电脑准备好通过指定网卡和机器人收发消息。 然后后面的发布通道，接受通道都是建立在这个全局初始化通信上的

        self.first_run = True # 标志位，第一次运行 policy 时，还没有历史观测，所以要特殊处理。
        self.config = config
        self.remote_controller = RemoteController() # 创建遥控器对象

        self.policy = torch.jit.load(config.policy_path, map_location="cpu").eval() # eval()的意思是把策略转成推理模式
        self.policy_layout = resolve_squat_policy_layout(
            self.policy,
            history_length=config.history_length,
            num_actions=config.num_actions,
            requested_layout=config.policy_obs_layout,
            configured_single_obs_dim=config.num_obs,
        )
        self.single_obs_dim = self.policy_layout.single_obs_dim
        self.policy_obs_dim = self.policy_layout.policy_obs_dim

        policy_kind = "CENet" if self.policy_layout.has_cenet else "ActorCritic"
        print(
            f"[INFO] Squat policy: {policy_kind}, layout={self.policy_layout.name}, "
            f"input={config.history_length}x{self.single_obs_dim}={self.policy_obs_dim}, "
            f"output={config.num_actions}."
        )
        if config.num_obs != self.single_obs_dim or config.num_policy_obs != self.policy_obs_dim:
            print(
                "[WARN] YAML observation dimensions do not match the exported policy; "
                f"using detected values num_obs={self.single_obs_dim}, "
                f"num_policy_obs={self.policy_obs_dim}."
            )
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)  # 100Hz/50Hz #创建一个周期线程 每隔 control_dt 秒，执行一次 self.run()  self.run是回调函数  策略每秒钟执行50次
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)  # self.publish_thread  因为 policy 没必要 500Hz 推理，太耗算力；但是底层命令最好高频持续发送，机器人控制更稳定
        self.cmd_lock = Lock() # 这是线程锁。# 因为现在有两个线程都可能访问 self.low_cmd：
        self.right_arm_lock = Lock()
        self.latest_right_arm_lowcmd = None
        self.right_arm_cmd_received = False
        self.external_right_arm_enabled_logged = False
        self.control_start_time = None
 
        self.joint_pos = np.zeros(config.num_joints, dtype=np.float32) # 29
        self.joint_vel = np.zeros(config.num_joints, dtype=np.float32) # 29 
        self.action = np.zeros(config.num_actions, dtype=np.float32) # 动作是15维度
        self.action_filter = FirstOrderLowPassFilter(
            size=config.num_actions,
            control_dt=config.control_dt,
            cutoff_hz=config.action_filter_cutoff_hz,
            enabled=config.action_filter_enabled,
        )
        if self.action_filter.enabled:
            print(
                "[INFO] Squat action low-pass filter enabled: "
                f"cutoff={self.action_filter.cutoff_hz:g} Hz, "
                f"alpha={self.action_filter.alpha:.6f}."
            )
        else:
            print("[INFO] Squat action low-pass filter disabled.")
        # 当前高度命令，初始为站立高度
        # 后面按遥控器输入缓慢变化，作为策略的 1 维 command
        self.height_cmd = float(config.stand_height)

        self.current_obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        self.current_obs_history = np.zeros(
            (config.history_length, self.single_obs_dim),
            dtype=np.float32,
        )

        self.clip_max_command = np.array(
            [
                self.config.command_range["height"][1],
            ],
            dtype=np.float32,
        )
        self.clip_min_command = np.array(
            [
                self.config.command_range["height"][0],
            ],
            dtype=np.float32,
        )
        # ===============================================================
        # 策略预热
        # 新 CENet 策略仍接收原始 800 维历史观测；v_z 和 latent 在导出策略内部生成。
        dummy_policy_obs = np.zeros((1, self.policy_obs_dim), dtype=np.float32)
        for _ in range(50):
            with torch.inference_mode():
                self.policy(torch.from_numpy(dummy_policy_obs))
        # ===============================================================
        
        if config.msg_type == "hg":
            '''
            low_cmd中有如下几个命令: 
                        mode_pr # 代表电机的控制模式
                        mode_machine # 是机器人当前机器模式相关字段。
                        motor_cmd # 这是电机命令对象 是一个有35个元素的列表 每一个元素就是一个电机对象 电机对象里面又有如下几个属性：       
                                                                                                                        mode
                                                                                                                        q
                                                                                                                        dq
                                                                                                                        tau
                                                                                                                        kp
                                                                                                                        kd
                                                                                                                        reserve
                        reserve
                        crc 校验码
            '''
            self.low_cmd = unitree_hg_msg_dds__LowCmd_() # 是在创建一包具体的命令消息。后面会往里面填内容
            
            '''
            low_state中有如下几个属性:
                        version: types.array[types.uint32, 2]
                        mode_pr: types.uint8                    # 协议/固件版本信息,
                        mode_machine: types.uint8               # 当前机器人机器状态/模式,
                        tick: types.uint32                      # 状态包计数/时间 tick 初始值为0
                        imu_state: 'unitree_sdk2py.idl.unitree_hg.msg.dds_.IMUState_'        # IMU状态包 里面储存的是相应的IMU的状态
                        motor_state: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorState_', 35] # 电机状态包
                        wireless_remote: types.array[types.uint8, 40] # 遥控器原始数据
                        reserve: types.array[types.uint32, 4]
                        crc: types.uint32                             # 校验码
            '''
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR

            # 创建发布者
            '''
            意思是我要创建一个发布通道；
            这个通道的名字叫 rt/lowcmd；
            这个通道里发送的数据类型是 HG 版本的 LowCmd。
            '''
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG) 
            self.lowcmd_publisher_.Init() # 初始化发布者
            
            # 创建订阅者
            '''
            我要创建一个订阅通道；
            这个通道的名字叫 rt/lowstate；
            这个通道里接收的数据类型是 HG 版本的 LowState。
            '''
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.right_arm_subscriber = ChannelSubscriber(config.right_arm_topic, LowCmdHG)
            '''
            每次来消息就要调用self.LowStateHandler这个回调函数
            10的意思是：机器人消息发送太快的话 可以做多储存10次的消息
            '''
            # 调用这一句之后，订阅器就开始监听机器人状态 topic 了
            self.lowstate_subscriber.Init(self.LowStateHandler, 10) # 初始化订阅者 #
            self.right_arm_subscriber.Init(self.RightArmLowCmdHandler, 10)
            print(f"Right arm LowCmd subscriber initialized: {config.right_arm_topic}")
        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()

        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_) # 初始化cmd

        self.publish_thread.Start()
        
        # 真正开始 policy 控制前，确保至少有一帧真实深度
        self.wait_for_start()

        self.move_to_default_pos()
        self.wait_for_control()

        self.control_start_time = time.monotonic()
        print("Start Control!")
        self.run_thread.Start()
    
    # 没有看懂
    def LowStateHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def RightArmLowCmdHandler(self, msg: LowCmdHG):
        first_message = False
        with self.right_arm_lock:
            self.latest_right_arm_lowcmd = msg
            if not self.right_arm_cmd_received:
                self.right_arm_cmd_received = True
                first_message = True

        if first_message:
            print("Received first right arm LowCmd.")

    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def wait_for_start(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def update_height_command(self):
        """
        更新 1 维高度命令。

        左摇杆上下控制：
        - 摇杆不动：保持站立高度 stand_height
        - 摇杆上推：保持站立高度 stand_height
        - 摇杆下拉：根据下拉幅度逐渐下蹲到 squat_height

        注意：
        remote_controller.ly 是左摇杆上下方向。
        一般情况下：
            ly > 0 表示上推
            ly < 0 表示下拉
        如果真机测试发现方向反了，把下面的 ly 改成 -ly。
        """

        # 左摇杆上下方向
        ly = float(self.remote_controller.ly)

        # 摇杆死区，避免轻微漂移导致机器人误下蹲
        deadzone = 0.10

        # ==========================================================
        # 目标高度 raw_target_height
        #
        # ly >= -deadzone:
        #   包括摇杆不动、轻微漂移、上推
        #   都保持站立高度
        #
        # ly < -deadzone:
        #   说明摇杆下拉
        #   下拉越多，目标高度越接近 squat_height
        # ==========================================================
        if ly >= -deadzone:
            target_height = self.config.stand_height
        else:
            # ly 从 -deadzone 到 -1.0
            # alpha 从 0 到 1
            alpha = (-ly - deadzone) / (1.0 - deadzone)
            alpha = np.clip(alpha, 0.0, 1.0)

            target_height = (
                self.config.stand_height
                + alpha * (self.config.squat_height - self.config.stand_height)
            )

        # ==========================================================
        # 高度命令限速滤波
        # 防止从站立高度突然跳到下蹲高度
        # ==========================================================
        max_delta = self.config.max_delta_per_s * self.config.control_dt
        delta = target_height - self.height_cmd
        delta = np.clip(delta, -max_delta, max_delta)

        self.height_cmd += float(delta)

        # 限幅，确保高度命令在 YAML 的 command_range 内
        height_cmd = np.array([self.height_cmd], dtype=np.float32)
        height_cmd = np.clip(height_cmd, self.clip_min_command, self.clip_max_command)

        self.height_cmd = float(height_cmd[0])
        return height_cmd
    
    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint2motor_idx
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            time.sleep(self.config.control_dt) 

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def run(self):
        # ==========================================================
        # 1. 读取 29 个关节状态
        # self.joint_pos / self.joint_vel 存储的是 IsaacLab 关节顺序
        # joint2motor_idx: IsaacLab joint index -> 真机 motor index
        # ==========================================================
        for i in range(self.config.num_joints):
            motor_idx = self.config.joint2motor_idx[i]
            # joint_pos joint_vel isaaclab的位置
            self.joint_pos[i] = self.low_state.motor_state[motor_idx].q
            self.joint_vel[i] = self.low_state.motor_state[motor_idx].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(
            self.low_state.imu_state.gyroscope,
            dtype=np.float32,
        ).reshape(3)



        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq

            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel.reshape(1, 3),
            )

            ang_vel = np.array(ang_vel, dtype=np.float32).reshape(3)

        # ==========================================================
        # 3. 计算重力方向和关节观测
        # ==========================================================
        gravity_orientation = get_gravity_orientation(quat).astype(np.float32)

        joint_pos = (
            self.joint_pos - self.config.default_joint_pos
        ) * self.config.dof_pos_scale
        joint_pos = joint_pos.astype(np.float32)

        joint_vel = self.joint_vel * self.config.dof_vel_scale
        joint_vel = joint_vel.astype(np.float32)

        ang_vel = ang_vel * self.config.ang_vel_scale
        ang_vel = ang_vel.astype(np.float32)

        # ==========================================================
        # 4. 生成 1 维高度命令
        # 按住 B 下蹲，松开 B 站起
        # ==========================================================
        command = self.update_height_command()

        # command_scale 现在应该是 [1.0]
        command = command * self.config.command_scale[0]

        # ==========================================================
        # 5. 拼当前训练使用的 80 维单帧 Actor 观测
        # ang_vel(3)
        # + gravity(3)
        # + height command(1)
        # + joint_pos(29)
        # + joint_vel(29)
        # + last_action(15)
        # = 80
        #
        # root_height 现在只属于 Critic 特权观测，不送入 Actor。仅当自动检测到
        # 旧 810 维 TorchScript 时，才保留原先的 height_cmd 近似值作为第 81 维。
        # ==========================================================
        idx = 0

        self.current_obs[idx : idx + 3] = ang_vel
        idx += 3

        self.current_obs[idx : idx + 3] = gravity_orientation
        idx += 3

        self.current_obs[idx : idx + 1] = command
        idx += 1

        self.current_obs[idx : idx + self.config.num_joints] = joint_pos
        idx += self.config.num_joints

        self.current_obs[idx : idx + self.config.num_joints] = joint_vel
        idx += self.config.num_joints

        self.current_obs[idx : idx + self.config.num_actions] = self.action
        idx += self.config.num_actions

        if self.policy_layout.includes_root_height:
            self.current_obs[idx : idx + 1] = self.height_cmd
            idx += 1

        if idx != self.single_obs_dim:
            raise RuntimeError(
                f"obs dim error: got {idx}, expected {self.single_obs_dim}"
            )
        

        # ==========================================================
        # 7. 维护 10 帧历史观测
        # current_obs_history shape:
        #   current policy: (obs_history_length, num_obs) = (10, 80)
        # policy_obs shape:
        #   current policy: (1, num_policy_obs) = (1, 800)
        # ==========================================================
        if self.first_run:
            # 第一次运行时，用当前观测填满整个历史
            # 避免历史帧一开始全是 0
            self.current_obs_history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            # 滚动历史：
            # 丢掉最旧的一帧，把当前观测放到最后
            self.current_obs_history = np.concatenate(
                (
                    self.current_obs_history[1:],
                    self.current_obs.reshape(1, -1),
                ),
                axis=0,
            )

        policy_obs = self.current_obs_history.reshape(1, -1).astype(np.float32)

        # ==============================
        # 4. 检查维度
        # ==============================
        if policy_obs.shape[1] != self.policy_obs_dim:
            raise RuntimeError(
                f"policy_obs dim mismatch: got {policy_obs.shape[1]}, "
                f"expected {self.policy_obs_dim}"
            )

        # ==============================
        # 5. policy 推理
        # ==============================
        with torch.inference_mode():
            action_tensor = self.policy(
                torch.from_numpy(policy_obs).clip(-100, 100)
            ).clip(-100, 100)
        if tuple(action_tensor.shape) != (1, self.config.num_actions):
            raise RuntimeError(
                f"action dim mismatch: "
                f"got {tuple(action_tensor.shape)}, "
                f"expected (1, {self.config.num_actions})"
            )
        raw_action = action_tensor[0].detach().cpu().numpy()
        # self.action 始终代表真机实际执行的滤波后动作。下一控制周期的
        # last_action 观测因此与发送给双腿/腰部的目标关节角保持一致。
        self.action = self.action_filter.update(raw_action)
        # ==========================================================
        # 9. 构造 29 维目标角
        #
        # policy 只输出 15 维动作：
        #   左腿 6 + 右腿 6 + 腰 3
        #
        # 但真机 LowCmd 仍然发 29 个关节：
        #   腿+腰：default + action * action_scale
        #   胳膊：保持 default_joint_pos
        # ==========================================================
        target_q = self.config.default_joint_pos.copy()

        target_q[self.config.action_joint_ids] = (
            self.config.default_joint_pos[self.config.action_joint_ids]
            + self.action * self.config.action_scale
        )

        right_arm_cmd_snapshot = None

        # ==========================================================
        # 10. 发送 29 个关节命令
        #
        # 注意：
        # 这里 29 个关节全部写 q/dq/kp/kd/tau。
        # 所以胳膊不会软掉，而是保持默认位置。
        # ==========================================================
        with self.cmd_lock:
            for i in range(self.config.num_joints):
                motor_idx = self.config.joint2motor_idx[i]

                self.low_cmd.motor_cmd[motor_idx].q = float(target_q[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            # 先按原逻辑写完全部 29 个关节，再只覆盖真机右臂 motor 22~28。
            # LowCmd 顶层的 mode_pr/mode_machine/reserve/crc 均保持本程序生成的值。
            elapsed = time.monotonic() - self.control_start_time
            if elapsed >= 0.5:
                # DDS callback 与控制线程并发运行。right_arm_lock 只用于
                # 快速复制最新一帧的右臂 motor command 控制字段。
                with self.right_arm_lock:
                    if self.right_arm_cmd_received:
                        right_arm_cmd_snapshot = tuple(
                            (
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].mode,
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].q,
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].dq,
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].kp,
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].kd,
                                self.latest_right_arm_lowcmd.motor_cmd[motor_idx].tau,
                            )
                            for motor_idx in RIGHT_ARM_MOTOR_IDS
                        )

            if right_arm_cmd_snapshot is not None:
                for motor_idx, motor_values in zip(
                    RIGHT_ARM_MOTOR_IDS, right_arm_cmd_snapshot
                ):
                    motor_cmd = self.low_cmd.motor_cmd[motor_idx]
                    (
                        motor_cmd.mode,
                        motor_cmd.q,
                        motor_cmd.dq,
                        motor_cmd.kp,
                        motor_cmd.kd,
                        motor_cmd.tau,
                    ) = motor_values

        if (
            right_arm_cmd_snapshot is not None
            and not self.external_right_arm_enabled_logged
        ):
            self.external_right_arm_enabled_logged = True
            print("Switched to external right arm LowCmd after 0.5 seconds.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp4s0", help="network interface")
    parser.add_argument("--config_path", type=str, default="config/g1_squart.yaml", help="configuration file path")
    args = parser.parse_args()

    config = Config(args.config_path)
    controller = Controller(config, args.net)

    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select Button detected, Exit!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()

        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)

        time.sleep(0.2)
        print("Exit")
