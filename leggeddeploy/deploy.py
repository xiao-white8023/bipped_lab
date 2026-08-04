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
from threading import Lock,Thread

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
from config import Config
from depth.depth_client import DepthClient

class Controller:
    def __init__(self, config: Config, net: str) -> None:

        ChannelFactoryInitialize(0, net) # 初始化 Unitree SDK2  初始化全局 DDS 环境的通信系统，并指定用哪张网卡连接机器人。初始化 Unitree SDK2 的 DDS 通信环境，让你的电脑准备好通过指定网卡和机器人收发消息。 然后后面的发布通道，接受通道都是建立在这个全局初始化通信上的

        self.first_run = True # 标志位，第一次运行 policy 时，还没有历史观测，所以要特殊处理。
        self.config = config
        self.remote_controller = RemoteController() # 创建遥控器对象

        self.policy = torch.jit.load(config.policy_path).eval() # eval()的意思是把策略转成推理模式
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)  # 100Hz/50Hz #创建一个周期线程 每隔 control_dt 秒，执行一次 self.run()  self.run是回调函数  策略每秒钟执行50次
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)  # self.publish_thread  因为 policy 没必要 500Hz 推理，太耗算力；但是底层命令最好高频持续发送，机器人控制更稳定
        self.cmd_lock = Lock() # 这是线程锁。# 因为现在有两个线程都可能访问 self.low_cmd：
 
        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)  # 本体观测
        self.current_obs_history = np.zeros((config.history_length, config.num_obs), dtype=np.float32) # 本体历史观测

        # =========================================================
        # 视觉深度输入相关
        self.policy_obs_debug_printed = False
        # depth_client.py 只负责输出单帧 policy_depth: (18, 32)
        # deploy.py 负责把单帧整合成 depth_history: (4, 18, 32)
        self.depth_client = None

        # 深度线程相关
        self.depth_thread = None
        self.depth_running = False
        
        # 深度缓存需要单独的锁
        # 因为后面 depth_thread 会写，run_thread 会读
        self.depth_lock = Lock()

        # 计算 crop 后的单帧深度尺寸
        self.policy_depth_h = (
            self.config.raw_h
            - self.config.crop_up
            - self.config.crop_down
        )
        self.policy_depth_w = (
            self.config.raw_w
            - self.config.crop_left
            - self.config.crop_right
        )

        if self.policy_depth_h <= 0 or self.policy_depth_w <= 0:
            raise ValueError(
                f"Invalid depth crop size: "
                f"raw=({self.config.raw_h}, {self.config.raw_w}), "
                f"crop_up={self.config.crop_up}, "
                f"crop_down={self.config.crop_down}, "
                f"crop_left={self.config.crop_left}, "
                f"crop_right={self.config.crop_right}"
            )
        
        # 初始化 depth history
        # shape = (history_frames, 18, 32)
        self.depth_history = np.ones(
            (
                self.config.history_frames,
                self.policy_depth_h,
                self.policy_depth_w,
            ),
            dtype=np.float32,
        )
        # 是否已经收到过真实深度
        self.has_depth = False

        # 创建 DepthClient，但这里先不启动线程
        if self.config.depth_enable:
            self.depth_client = DepthClient(self.config)

        # ============================================================

        # 步态相位时间追踪
        self.gait_phase_time = 0.0

        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )

        # ===============================================================
        # 策略预热
        # 视觉 policy 的输入维度：
        # proprio_history: history_length * num_obs
        # depth_history: history_frames * policy_depth_h * policy_depth_w
        proprio_dim = self.config.history_length * self.config.num_obs  # 本体历史感知维度
        depth_dim = (
            self.config.history_frames
            * self.policy_depth_h
            * self.policy_depth_w
        )
        policy_obs_dim = proprio_dim + depth_dim

        if hasattr(self.config, "num_policy_obs"):
            if policy_obs_dim != self.config.num_policy_obs:
                raise RuntimeError(
                    f"policy obs dim mismatch: "
                    f"computed={policy_obs_dim}, config.num_policy_obs={self.config.num_policy_obs}"
                )

        dummy_policy_obs = np.zeros((1, policy_obs_dim), dtype=np.float32)
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
            
            #创建订阅者
            '''
            我要创建一个订阅通道；
            这个通道的名字叫 rt/lowstate；
            这个通道里接收的数据类型是 HG 版本的 LowState。
            '''
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            '''
            每次来消息就要调用self.LowStateHandler这个回调函数
            10的意思是：机器人消息发送太快的话 可以做多储存10次的消息
            '''
            # 调用这一句之后，订阅器就开始监听机器人状态 topic 了
            self.lowstate_subscriber.Init(self.LowStateHandler, 10) # 初始化订阅者 # 
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()

        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_) # 初始化cmd
        elif config.msg_type == "go":
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.publish_thread.Start()
        
        # 启动深度线程，让相机数据先热起来
        self.start_depth_thread()

        # 真正开始 policy 控制前，确保至少有一帧真实深度
        self.wait_for_start()

        self.move_to_default_pos()
        self.wait_for_control()
        self.wait_for_depth()
        print("Start Control!")
        self.run_thread.Start()
    
    # 没有看懂
    def LowStateHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

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

    def wait_for_depth(self):
        """
        等待第一帧真实深度到达。

        视觉策略不要在 depth_history 还是全 1.0 的时候开始控制。
        """

        if not self.config.depth_enable:
            return

        print("[DepthThread] waiting for first depth frame...")

        while not self.has_depth:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()

            time.sleep(self.config.control_dt)

        print("[DepthThread] first depth frame received.")

    def depth_loop(self):
        """
        后台深度线程。

        这个线程只做一件事：
            从 depth_client 取单帧 policy_depth
            然后更新 self.depth_history

        注意：
            不在这里调用 policy。
            不在这里发电机命令。
        """
        # depth_update_interval 表示隔多少个 control_dt 更新一次深度
        # 比如 control_dt=0.02, depth_update_interval=5
        # 那么 depth_dt=0.1s，也就是 10Hz
        depth_update_interval = getattr(self.config, "depth_update_interval", 5)
        depth_dt = self.config.control_dt * depth_update_interval

        while self.depth_running:
            loop_start = time.time()

            try:
                # depth_client.update_once() 返回单帧 policy_depth: (18, 32)
                policy_depth = self.depth_client.update_once()

                # deploy.py 负责把单帧整合成 depth_history: (4, 18, 32)
                self.update_depth_history(policy_depth)

            except Exception as e:
                print(f"[DepthThread] depth update failed: {e}")

            elapsed = time.time() - loop_start
            sleep_time = depth_dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def update_depth_history(self, policy_depth: np.ndarray):
        """
        把单帧 policy_depth 更新到 depth_history 里。

        输入:
            policy_depth:
                shape = (policy_depth_h, policy_depth_w)
                dtype = float32
                数值范围约为 [0, 1]

        更新后:
            self.depth_history:
                shape = (history_frames, policy_depth_h, policy_depth_w)
                时间顺序是 [旧, ..., 新]
        """

        expected_shape = (self.policy_depth_h, self.policy_depth_w)

        if policy_depth.shape != expected_shape:
            raise RuntimeError(
                f"policy_depth shape mismatch: "
                f"expected {expected_shape}, got {policy_depth.shape}"
            )
        
        policy_depth = policy_depth.astype(np.float32)
        with self.depth_lock:
            if not self.has_depth:
                # 第一帧来了以后，用这一帧填满整个历史
                self.depth_history[:] = policy_depth.reshape(
                    1,
                    self.policy_depth_h,
                    self.policy_depth_w,
                )
                self.has_depth = True
            else:
                # 丢掉最老的一帧，把最新一帧放到最后
                self.depth_history[:-1] = self.depth_history[1:]
                self.depth_history[-1] = policy_depth

    def get_depth_obs(self) -> np.ndarray:
        """
            返回展平后的深度历史。

            输出:
                depth_obs:
                    shape = (history_frames * policy_depth_h * policy_depth_w,)
                    如果是 4×18×32，就是 2304 维。
        """
        with self.depth_lock:
            depth_obs = self.depth_history.copy()

        return depth_obs.reshape(-1).astype(np.float32)

    def start_depth_thread(self):
        """
        启动深度接收线程。
        """

        if not self.config.depth_enable:
            print("[DepthThread] depth is disabled.")
            return

        if self.depth_client is None:
            raise RuntimeError("[DepthThread] depth_client is None.")

        print("[DepthThread] connecting to depth server...")
        self.depth_client.connect()

        self.depth_running = True
        self.depth_thread = Thread(target=self.depth_loop)
        self.depth_thread.daemon = True
        self.depth_thread.start()
        print("[DepthThread] started.")

    def stop_depth_thread(self):
        """
        停止深度线程，并关闭 TCP 连接。
        """

        if not self.config.depth_enable:
            return

        self.depth_running = False

        if self.depth_thread is not None:
            self.depth_thread.join(timeout=1.0)

        if self.depth_client is not None:
            self.depth_client.close()
        print("[DepthThread] stopped.")  

    
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

    def compute_gait_phase(self) -> np.ndarray:
        """计算步态相位 (4 维: sin_left, cos_left, sin_right, cos_right)
        
        左右腿相位差 offset=0.5，实现交替行走。
        """
        period = self.config.gait_phase_period  # 0.8s
        offset = self.config.gait_phase_offset  # 0.5
        
        # 左腿相位
        phase_left = (self.gait_phase_time % period) / period  # [0, 1)
        
        # 右腿相位 (偏移 50%)
        phase_right = ((self.gait_phase_time / period) + offset) % 1.0
        
        # 转换为 sin/cos
        sin_left = np.sin(2 * np.pi * phase_left)
        cos_left = np.cos(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_right = np.cos(2 * np.pi * phase_right)
        
        return np.array([sin_left, sin_right, cos_left, cos_right], dtype=np.float32)


    def run(self):
        for i in range(len(self.config.joint2motor_idx)):
            # self.joint_pos中存储的就是isaaclab中的关节顺序
            self.joint_pos[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32) # 角速度

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel
            )

        gravity_orientation = get_gravity_orientation(quat)
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        command = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx], dtype=np.float32
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)

        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command
        self.current_obs[9 : 9 + num_actions] = joint_pos
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = joint_vel
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        # 添加步态相位 (如果启用)
        if self.config.gait_phase_enable:
            gait_phase = self.compute_gait_phase()
            obs_base_end = 9 + num_actions * 3  # 96
            self.current_obs[obs_base_end : obs_base_end + 4] = gait_phase
            # 更新步态相位时间
            self.gait_phase_time += self.config.control_dt
        
        # 当是第一次运行时，将当前时刻的观测完全填充到历史帧中
        if self.first_run:
            self.current_obs_history[:] = self.current_obs.reshape(1, -1) 
            self.first_run = False
        else:
            self.current_obs_history = np.concatenate(
                (self.current_obs_history[1:], self.current_obs.reshape(1, -1)), axis=0
            )

        # ==============================
        # 1. 本体历史展平
        # shape: (history_length * num_obs,)
        # 例如 10 * 100 = 1000
        # ==============================
        proprio_obs = self.current_obs_history.reshape(-1).astype(np.float32)

        # ==============================
        # 2. 深度历史展平
        # shape: (history_frames * 18 * 32,)
        # 例如 4 * 18 * 32 = 2304
        # ==============================
        if self.config.depth_enable:
            depth_obs = self.get_depth_obs()
        else:
            depth_obs = np.zeros(0, dtype=np.float32)

        # ==============================
        # 3. 拼接成视觉 policy 输入
        # 训练注释写的是：
        # 10帧本体观测历史 + 4帧展平后的深度图
        # 所以这里顺序用 [proprio_obs, depth_obs]
        # ==============================
        policy_obs = np.concatenate(
            [proprio_obs, depth_obs],
            axis=0,
        ).reshape(1, -1).astype(np.float32)

        # ==============================
        # 4. 检查维度
        # ==============================
        if hasattr(self.config, "num_policy_obs"):
            if policy_obs.shape[1] != self.config.num_policy_obs:
                raise RuntimeError(
                    f"policy_obs dim mismatch: "
                    f"got {policy_obs.shape[1]}, "
                    f"expected {self.config.num_policy_obs}"
                )

        # ==============================
        # 5. policy 推理
        # ==============================
        self.action = (
            self.policy(torch.from_numpy(policy_obs).clip(-100, 100))
            .clip(-100, 100)
            .detach()
            .numpy()
            .squeeze()
        )
        if not self.policy_obs_debug_printed:
            print(
                "[PolicyObs] "
                f"proprio_dim={proprio_obs.shape[0]}, "
                f"depth_dim={depth_obs.shape[0]}, "
                f"total_dim={policy_obs.shape[1]}, "
                f"proprio_min={proprio_obs.min():.3f}, "
                f"proprio_max={proprio_obs.max():.3f}, "
                f"depth_min={depth_obs.min():.3f}, "
                f"depth_max={depth_obs.max():.3f}, "
                f"depth_mean={depth_obs.mean():.3f}"
            )
        self.policy_obs_debug_printed = True
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale
        with self.cmd_lock:
            for i in range(len(self.config.joint2motor_idx)):
                self.low_cmd.motor_cmd[self.config.joint2motor_idx[i]].q = target_dof_pos[i]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp4s0", help="network interface")
    parser.add_argument("--config_path", type=str, default="config/g1_film_phase.yaml", help="configuration file path")
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
        controller.stop_depth_thread()

        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)

        time.sleep(0.2)
        print("Exit")
