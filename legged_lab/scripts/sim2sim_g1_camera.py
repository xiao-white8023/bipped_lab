import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
from pynput import keyboard
import torchvision.transforms as T
from collections import deque
from joystick import JoyStick, JoystickButton
# ==================== 关节顺序定义 ====================
# MuJoCo XML 中的关节顺序 (29 DOF)
MUJOCO_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

# Isaac Lab 中的关节顺序 (按 URDF actuator 定义)
LAB_DOF_NAMES = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]

class G1FlatSim2SimCfg:
    """G1 Flat Sim2Sim 配置类 - 与训练配置保持一致"""

    class sim:
        sim_duration = 100.0
        num_actions = 29
        num_obs_per_step = 100 # 96
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # G1 初始高度
        init_height = 0.793
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85

        depth_update_interval = 5
        depth_history_frames = 8
        camera_height = 36
        camera_width = 64

class G1FlatMujocoRunner:
    """
    G1 Flat Sim2Sim 运行器
    
    加载策略和 MuJoCo 模型，运行实时仿真控制。

    Args:
        cfg: 配置对象
        policy_path: TorchScript 导出的策略路径
        model_path: MuJoCo XML 模型路径
    """

    def __init__(self, cfg: G1FlatSim2SimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        
        # 加载 MuJoCo 模型
        print(f"[INFO] 加载 MuJoCo 模型: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        # 加载策略
        print(f"[INFO] 加载策略: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # 加入深度相机的参数
        print(f"[INFO] 初始化深度相机渲染器 ({self.cfg.robot.camera_width}x{self.cfg.robot.camera_height})")
        self.renderer = mujoco.Renderer(self.model, height=self.cfg.robot.camera_height, width=self.cfg.robot.camera_width)
        self.renderer.enable_depth_rendering()

        # 深度图转换物理距离参数
        extent = self.model.stat.extent
        self.depth_near = self.model.vis.map.znear * extent
        self.depth_far = self.model.vis.map.zfar * extent
        # 初始化双端队列作为深度图历史缓存
        self.depth_buffer = deque(
            [torch.zeros((18, 32), dtype=torch.float32) for _ in range(self.cfg.robot.depth_history_frames)], 
            maxlen=self.cfg.robot.depth_history_frames
        )
        # 高斯模糊处理器
        self.blur_transform = T.GaussianBlur(kernel_size=3, sigma=1.0)
        
        
        # 初始化变量
        self.init_variables()

        # ===== Matplotlib 初始化 =====
        plt.ion() # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.im = None
        self.ax.set_title("G1 Depth Camera")

        self.build_joint_mappings()
        self.set_initial_pose()
        
        print(f"[INFO] 控制频率: {1.0 / (cfg.sim.dt * cfg.sim.decimation):.1f} Hz")
        print(f"[INFO] 观测维度: {cfg.sim.num_obs_per_step} x {cfg.sim.actor_obs_history_length} = {cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length}")

        try:
            self.joystick = JoyStick()
            self.use_joystick = True
            print("[INFO] Xbox 手柄连接成功！")
        except RuntimeError as e:
            print(f"[WARNING] 手柄初始化失败: {e}。将无法使用速度控制。")
            self.use_joystick = False

    def init_variables(self) -> None:
        """初始化仿真变量"""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions
        
        # 关节状态
        self.dof_pos = np.zeros(self.num_actions)
        self.dof_vel = np.zeros(self.num_actions)
        
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        # 动作 (Isaac Lab 顺序)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # 默认关节位置 (MuJoCo 顺序) - 与 G1_CFG init_state.joint_pos 一致
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 左腿
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 右腿
            0.0, 0.0, 0.0,                        # 腰部
            0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # 左臂
            0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0 # 右臂
        ], dtype=np.float32)
        
        # PD 增益 (MuJoCo 顺序) - 与 G1_CFG 训练配置一致
        # MuJoCo 顺序: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, ...
        self.kps = np.array([
            200, 150, 150, 200, 20, 20,    # 左腿: 200, 150, 150, 200, 20, 20,
            200, 150, 150, 200, 20, 20,    # 右腿200, 150, 150, 200, 20, 20, 
            200, 200, 200,                  # 腰部: waist_yaw, waist_roll, waist_pitch
            100, 100, 50, 50, 15, 15,15,  # 左臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_*
            100, 100, 50, 50, 15, 15, 15   # 右臂
        ], dtype=np.float32)
        
        self.kds = np.array([
            5, 5, 5, 5, 2, 2,              # 左腿 5, 5, 5, 5, 2, 2,
            5, 5, 5, 5, 2, 2,              # 右腿 # 5, 5, 5, 5, 2, 2, 
            5, 5, 5,                        # 腰部
            2, 2, 2, 2, 2, 2, 2,           # 左臂
            2, 2, 2, 2, 2, 2, 2            # 右臂
        ], dtype=np.float32)
        
        self.episode_length_buf = 0
        
        # 速度命令
        self.command_vel = np.array([0, 0.0, 0.0], dtype=np.float32)
        
        # 观测历史
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32
        )
        # 提取 MuJoCo 解析出来的真实活动关节名称 (排除 floating_base)
        real_mujoco_names = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != 0: # 0 是 free joint (基座)
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                real_mujoco_names.append(name)
                
        print("\n========== 验证 MuJoCo 真实顺序 ==========")
        for i, (code_name, real_name) in enumerate(zip(MUJOCO_DOF_NAMES, real_mujoco_names)):
            if code_name != real_name:
                print(f"⚠️ 警告: 索引 {i} 不匹配! 代码写的是 '{code_name}', 真实是 '{real_name}'")
        print("========== 验证结束 ==========\n")

    def build_joint_mappings(self) -> None:
        """建立关节映射索引"""
        # MuJoCo -> Isaac Lab 索引映射
        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        self.mujoco_to_isaac_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
        
        # Isaac Lab -> MuJoCo 索引映射
        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.isaac_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        
        # 默认关节位置转换为 Isaac Lab 顺序
        self.default_dof_pos_isaac = self.default_dof_pos[self.mujoco_to_isaac_idx]
        
        print(f"[INFO] 关节映射建立完成")
        print(f"[INFO] 默认关节位置 (MuJoCo 前6个): {self.default_dof_pos[:6]}")

    def set_initial_pose(self) -> None:
        """设置初始姿态"""
        # 基座位置
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # (w, x, y, z)
        
        # 关节位置
        self.data.qpos[7:7 + self.num_actions] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] 初始高度: {self.data.qpos[2]:.3f}m")

    def get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        """计算投影重力向量
        
        Args:
            quat: MuJoCo 四元数 (w, x, y, z)
        
        Returns:
            投影重力向量 (3,)
        """
        qw, qx, qy, qz = quat
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """四元数逆旋转 - 将向量从世界坐标系转到 body 坐标系
        
        与 sim2sim.py 保持一致的实现
        
        Args:
            q: 四元数 (x, y, z, w) 格式
            v: 世界坐标系中的向量
        
        Returns:
            body 坐标系中的向量
        """
        q_w = q[-1]  # w 在最后
        q_vec = q[:3]  # xyz
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def mj29_to_lab29(self, array_mj: np.ndarray) -> np.ndarray:
        """将 MuJoCo 顺序数组转换为 Isaac Lab 顺序"""
        return array_mj[self.mujoco_to_isaac_idx]

    def lab29_to_mj29(self, array_lab: np.ndarray) -> np.ndarray:
        """将 Isaac Lab 顺序数组转换为 MuJoCo 顺序"""
        return array_lab[self.isaac_to_mujoco_idx]

    def update_depth_vision(self, noise_active: bool = False):
        """完全复刻 Isaac Lab 中的深度图处理

        Args:
            noise_active: 当 True 时，向深度图注入真实立体相机噪声
        """
        # 1. 渲染深度
        self.renderer.update_scene(self.data, camera="depth_camera")
        raw_depth = self.renderer.render()
        print(f"[DEBUG] Raw Depth Max: {raw_depth.max():.3f}, Min: {raw_depth.min():.3f}")

        vis_depth = np.clip(raw_depth, 0.1, 2.5)
        vis_depth = (vis_depth - 0.1) / (2.5 - 0.1)
        # 翻转颜色，0-1 浮点数即可
        vis_img = 1.0 - vis_depth
        
        if self.im is None:
            self.im = self.ax.imshow(vis_img, cmap='gray', vmin=0, vmax=1)
        else:
            self.im.set_data(vis_img)

        # 根据噪声状态更新标题颜色
        if noise_active:
            self.ax.set_title("G1 Depth Camera [NOISE ON]", color='red', fontweight='bold')
        else:
            self.ax.set_title("G1 Depth Camera", color='black', fontweight='normal')

        self.fig.canvas.flush_events() # 刷新画面

        depth_tensor = torch.tensor(raw_depth.copy(), dtype=torch.float32)

        # 3. 裁剪 (crop: up=18, down=0, left=16, right=16) 
        # 原尺寸 36x64 -> 结果 [18:36, 16:48] 即 18x32
        depth_cropped = depth_tensor[18:36, 16:48]
        
        # 4. 去除非法值
        depth_cropped[torch.isinf(depth_cropped)] = 2.5
        depth_cropped[torch.isnan(depth_cropped)] = 2.5

        # 4.5 噪声注入 (按住 R1 时触发)
        if noise_active:
            depth_cropped = self._apply_depth_noise(depth_cropped)

        # 5. 高斯模糊
        depth_cropped = depth_cropped.unsqueeze(0) # [1, H, W]
        depth_blurred = self.blur_transform(depth_cropped).squeeze(0)
        
        # 6. 截断与归一化
        depth_clipped = torch.clip(depth_blurred, min=0.1, max=2.5)
        depth_normalized = depth_clipped / 2.5
        
        print(f"[VISION DEBUG] Depth Min: {depth_normalized.min().item():.3f}, Max: {depth_normalized.max().item():.3f}, Mean: {depth_normalized.mean().item():.3f}")

        # 7. 更新历史缓存
        self.depth_buffer.append(depth_normalized)


    def _apply_depth_noise(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """模拟真实立体视觉深度相机的噪声

        参考 Intel RealSense D435i 在复杂场景下的典型噪声模式:
        - 距离相关的高斯噪声 (立体匹配误差与距离平方成正比)
        - 随机像素丢失 (纹理不足/反光导致的匹配失败)
        - 块状伪影 (镜面反射/遮挡边界)
        - 边缘噪声 (深度不连续处的过曝/欠曝)

        Args:
            depth_tensor: 裁剪后的深度图 (H, W)，已移除 inf/nan

        Returns:
            带噪声的深度图 (H, W)
        """
        H, W = depth_tensor.shape
        device = depth_tensor.device

        # 1. 距离依赖的高斯噪声 (立体深度误差 ∝ z²/fb)
        noise_std = 0.005 * (depth_tensor ** 2)
        gaussian_noise = torch.randn(H, W, device=device) * noise_std
        depth_tensor = (depth_tensor + gaussian_noise).clamp(min=0.05)

        # 2. 随机像素丢失 — 模拟纹理不足区域的立体匹配失败
        dropout_mask = torch.rand(H, W, device=device) < 0.03
        depth_tensor[dropout_mask] = torch.rand(
            (dropout_mask.sum().item(),), device=device
        ) * 2.5

        # 3. 随机块状伪影 — 模拟镜面反射/强光区域的传感器致盲
        num_blocks = torch.randint(0, 6, (1,)).item()
        for _ in range(num_blocks):
            bh = torch.randint(2, 7, (1,)).item()
            bw = torch.randint(2, 7, (1,)).item()
            y = torch.randint(0, max(1, H - bh), (1,)).item()
            x = torch.randint(0, max(1, W - bw), (1,)).item()
            block_val = torch.rand(1, device=device).item() * 2.5
            depth_tensor[y:y + bh, x:x + bw] = block_val

        # 4. 边缘噪声 — 使用 Sobel 检测深度不连续处并注入额外抖动
        depth_2d = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        sobel_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=device).view(1, 1, 3, 3)
        sobel_v = sobel_h.transpose(-1, -2)
        edge_h = torch.abs(torch.nn.functional.conv2d(depth_2d, sobel_h, padding=1))
        edge_v = torch.abs(torch.nn.functional.conv2d(depth_2d, sobel_v, padding=1))
        edge_map = (edge_h + edge_v).squeeze()  # (H, W)
        edge_mask = edge_map > 0.15
        depth_tensor[edge_mask] += torch.randn(
            (edge_mask.sum().item(),), device=device
        ) * 0.15

        # 5. 强光照噪声 — 模拟极端光照对深度相机的毁灭性影响
        # 生成低频光照图，通过平方拉大对比度模拟强聚光灯+浓重阴影
        light_lowres = torch.rand(4, 6, device=device)
        light_lowres = light_lowres ** 2  # 平方拉大对比度，暗区更暗亮区更亮
        light_lowres = torch.nn.functional.interpolate(
            light_lowres.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()  # (H, W)，强对比光照分布

        # 5a. 暗区噪声暴增 — 光照 < 0.4 的阴影区，SNR 崩溃
        dark_mask = light_lowres < 0.4
        dark_boost = torch.randn(H, W, device=device) * 0.8 + 0.5
        depth_tensor[dark_mask] = (
            depth_tensor[dark_mask] + dark_boost[dark_mask]
        ).clamp(min=0.0)

        # 5b. 暗区大面积像素丢失 — 阴影中匹配几乎完全失效
        dark_dropout = torch.rand(H, W, device=device) < 0.35
        dark_dropout = dark_dropout & dark_mask
        depth_tensor[dark_dropout] = torch.rand(
            (dark_dropout.sum().item(),), device=device
        ) * 2.5

        # 5c. 过曝区域 — 强光直射导致传感器饱和，深度直接清零
        bright_mask = light_lowres > 0.6
        depth_tensor[bright_mask] = 0.0

        # 5d. 光照过渡带散斑噪声 — 半影区强烈激光散斑
        penumbra_mask = (light_lowres >= 0.2) & (light_lowres <= 0.5)
        speckle = torch.randn(H, W, device=device) * 0.3
        depth_tensor[penumbra_mask] += speckle[penumbra_mask]

        return depth_tensor.clamp(min=0.05, max=3.0)

    def get_obs(self) -> np.ndarray:
        """
        计算当前观测向量
        
        Returns:
            归一化和裁剪后的观测历史
        """
        # 读取关节状态 (MuJoCo 顺序)
        dof_pos_mj = self.data.qpos[7:7 + self.num_actions].copy()
        dof_vel_mj = self.data.qvel[6:6 + self.num_actions].copy()
        
        # 基座状态
        # MuJoCo 的 d.qvel[3:6] 本身就是 body frame 角速度，无需转换
        ang_vel_body = self.data.qvel[3:6].copy()
        
        # 投影重力 - 使用 get_gravity_orientation (与 deploy_mujoco.py 一致)
        quat = self.data.qpos[3:7].copy()  # MuJoCo: (w, x, y, z)
        projected_gravity = self.get_gravity_orientation(quat)
        
        # 转换为 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        
        # 构建观测 (96 维) - 与 g1_env.py compute_current_observations 一致
        obs = np.concatenate([
            ang_vel_body,                     # 3: 角速度 (body frame, 直接从 qvel[3:6])
            projected_gravity,                # 3: 投影重力 (body frame)
            self.command_vel,                 # 3: 速度命令
            joint_pos_isaac,                  # 29: 关节位置偏差
            joint_vel_isaac,                  # 29: 关节速度
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 29: 上一步动作
            # np.sin(2 * np.pi * self.gait_phase),  # 2
            # np.cos(2 * np.pi * self.gait_phase),  # 2
            # self.phase_ratio,  # 2
        ], axis=0).astype(np.float32)
        
        # 更新观测历史 (滚动更新)
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs.copy()
        
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def position_control(self) -> np.ndarray:
        """
        计算目标关节位置 (MuJoCo 顺序)
        
        Returns:
            目标关节位置
        """
        # action 是 Isaac Lab 顺序，转换为 MuJoCo 顺序
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab29_to_mj29(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        """
        PD 控制器计算力矩
        
        Args:
            target_q: 目标关节位置 (MuJoCo 顺序)
        
        Returns:
            关节力矩
        """
        q = self.data.qpos[7:7 + self.num_actions]
        dq = self.data.qvel[6:6 + self.num_actions]
        return (target_q - q) * self.kps + (0 - dq) * self.kds

    def run(self) -> None:
        """运行仿真循环"""
        
        # 创建 viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.stabilize_robot(duration=0.05)
        
        # 初始化观测历史
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision()
        print("\n[INFO] 键盘控制: 8/2=前后, 4/6=左右, 7/9=转向, 5=停止")
        print("[INFO] Xbox: 左摇杆=移动, 右摇杆=转向, R1=注入深度噪声, Select=退出")
        print("[INFO] 按 Ctrl+C 退出\n")

        debug_counter = 0

        try:
            while self.viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                # 检测 R1 是否按下用于噪声注入
                depth_noise_active = (
                    self.use_joystick and
                    self.joystick.is_button_pressed(JoystickButton.R1)
                )
                if self.episode_length_buf % self.cfg.robot.depth_update_interval == 0:
                    self.update_depth_vision(noise_active=depth_noise_active)
                if self.use_joystick:
                    self.joystick.update()
                    
                    # 按下 Select 键退出仿真
                    if self.joystick.is_button_pressed(JoystickButton.SELECT):
                        print("[INFO] 收到手柄退出指令")
                        break
                    
                    # 读取摇杆轴数据
                    # 轴 1: 左摇杆上下 (Pygame中上是负，下是正)
                    # 轴 0: 左摇杆左右 (左是负，右是正)
                    # 轴 3: 右摇杆左右 (用于控制 Yaw 转向)
                    ax1 = self.joystick.get_axis_value(1)
                    ax0 = self.joystick.get_axis_value(0)
                    ax3 = self.joystick.get_axis_value(3)
                    
                    # 设置死区（例如 0.1），防止轻微漂移
                    deadzone = 0.1
                    ax1 = 0.0 if abs(ax1) < deadzone else ax1
                    ax0 = 0.0 if abs(ax0) < deadzone else ax0
                    ax3 = 0.0 if abs(ax3) < deadzone else ax3
                    
                    # 映射到速度命令限制：vx=1.0, vy=0.5, yaw_rate=1.57
                    self.command_vel[0] = -ax1 * 1.0   # 前推为负，所以取负号变为正速度
                    self.command_vel[1] = -ax0 * 0.5   
                    self.command_vel[2] = -ax3 * 1.57
                # 获取观测并执行策略
                obs = self.get_obs()
                proprio_tensor = torch.tensor(obs, dtype=torch.float32)
                # ================= 2. 拼装多模态观测 =================
                # 提取视觉历史 (大小: 8 x 18 x 32 = 4608)
                flat_depth = torch.stack(list(self.depth_buffer)).view(-1)
                # flat_depth = torch.ones_like(flat_depth) * 1.0
                obs = torch.cat([proprio_tensor, flat_depth], dim=-1).unsqueeze(0)
                self.action[:] = self.policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()[:self.num_actions]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)# 调试输出
                debug_counter += 1
                if debug_counter <= 3:
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs shape: {obs.shape}")
    
                    # 提取本体感知特征的最后 102 维（最新的一帧）
                    latest_obs = obs[0, 1020-102 : 1020] 
    
                    print(f"  current_obs (first 9): {latest_obs[:9]}") 
                    print(f"  command: {self.command_vel}")
                    print(f"  action (first 6, Isaac order): {self.action[:6]}")
                    print(f"  dof_pos_obs (first 6): {latest_obs[9:15]}") 
                    print(f"  dof_vel_obs (first 6): {latest_obs[38:44]}")
                    # 执行 decimation 步
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    
                    # 计算目标位置并执行 PD 控制
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:self.num_actions] = tau
                    
                    # 物理步进
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()
                    
                    # 时间控制
                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)
                
                self.episode_length_buf += 1
                self.calculate_gait_para()

                # 定期打印状态
                if self.episode_length_buf % 100 == 0:
                    print(f"[INFO] t={self.data.time:.1f}s, cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], h={self.data.qpos[2]:.3f}m")
                    
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断")
        finally:
            self.viewer.close()
            plt.close('all')
            print("[INFO] 仿真结束")

    def stabilize_robot(self, duration: float = 2.0) -> None:
        target_pos = self.default_dof_pos.copy()
        num_steps = int(duration / self.cfg.sim.dt)
        
        # 上帝之手：锁定基座
        for i in range(num_steps):
            self.data.qpos[0:2] = 0.0  
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  
            self.data.qvel[0:6] = 0.0  
            
            tau = self.pd_control(target_pos)
            self.data.ctrl[:self.num_actions] = tau
            mujoco.mj_step(self.model, self.data)
            
        # 🔥 恢复自然记忆：连续获取 10 帧真实的静止状态，不要清零！
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()
            
        for _ in range(self.cfg.robot.depth_history_frames):
            self.update_depth_vision()

        print(f"[INFO] 稳定后高度: {self.data.qpos[2]:.4f}m，释放上帝之手，策略接管！")



def get_available_scenes(mjcf_dir: str) -> dict:
    """获取可用的场景文件
    
    约定：场景文件以 _scene.xml 结尾，或者是特殊名称如 stairs.xml
    """
    scenes = {}
    
    # 搜索 mjcf 目录下的场景文件
    if os.path.isdir(mjcf_dir):
        for f in os.listdir(mjcf_dir):
            if f.endswith('.xml'):
                # 特殊场景文件
                if f in ['scene.xml', 'stairs.xml', 'flat_scene.xml', 'rough_scene.xml', 'slope_scene.xml']:
                    name = f.replace('.xml', '').replace('_scene', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
                # 通用 *_scene.xml 文件
                elif f.endswith('_scene.xml'):
                    name = f.replace('_scene.xml', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
    
    return scenes


def main():
    LEGGED_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    MJCF_DIR = os.path.join(LEGGED_LAB_ROOT, "legged_lab/assets/g1/g1_29dof")
    
    # 默认路径
    default_policy = os.path.join(LEGGED_LAB_ROOT, "logs/g1_rough/2026-03-21_20-27-18/exported/policy.pt")
    default_model = os.path.join(MJCF_DIR, "legged_lab/assets/g1/g1_29dof/stair_scene.xml")
    
    # 获取可用场景
    available_scenes = get_available_scenes(MJCF_DIR)
    scene_names = list(available_scenes.keys())
    
    parser = argparse.ArgumentParser(
        description="G1 Flat Sim2Sim - 支持加载不同地形场景",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--policy", type=str, default=default_policy, help="策略文件路径")
    parser.add_argument("--model", type=str, default=default_model, 
                        help="MuJoCo XML 模型路径 (当未指定 --scene 时使用)")
    parser.add_argument("--scene", type=str, default=None, 
                        choices=scene_names if scene_names else None,
                        help=f"场景名称，可用场景: {', '.join(scene_names) if scene_names else '无'}\n"
                             f"场景文件位于: {MJCF_DIR}")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="直接指定场景文件路径 (优先级高于 --scene)")
    parser.add_argument("--duration", type=float, default=100.0, help="仿真时长 (秒)")
    parser.add_argument("--list-scenes", action="store_true", help="列出所有可用场景")
    args = parser.parse_args()
    
    # 列出可用场景
    if args.list_scenes:
        print("\n可用场景:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"场景文件目录: {MJCF_DIR}")
        print("\n使用示例:")
        print(f"  python {sys.argv[0]} --scene stairs")
        print(f"  python {sys.argv[0]} --scene-file /path/to/custom_scene.xml")
        sys.exit(0)
    
    # 检查策略文件
    if not os.path.isfile(args.policy):
        print(f"[ERROR] 策略文件不存在: {args.policy}")
        sys.exit(1)
    
    # 确定要加载的模型/场景文件
    if args.scene_file:
        # 直接指定场景文件
        model_path = args.scene_file
        if not os.path.isfile(model_path):
            print(f"[ERROR] 场景文件不存在: {model_path}")
            sys.exit(1)
    elif args.scene:
        # 使用预定义场景
        if args.scene not in available_scenes:
            print(f"[ERROR] 未知场景: {args.scene}")
            print(f"[INFO] 可用场景: {', '.join(scene_names)}")
            sys.exit(1)
        model_path = available_scenes[args.scene]
    else:
        # 使用默认模型
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"[ERROR] MuJoCo 模型不存在: {model_path}")
            sys.exit(1)
    
    print("=" * 60)
    print("G1 Flat Sim2Sim")
    print("=" * 60)
    print(f"策略: {args.policy}")
    print(f"MuJoCo 模型/场景: {model_path}")
    if args.scene:
        print(f"场景: {args.scene}")
    print("=" * 60)
    
    cfg = G1FlatSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    
    runner = G1FlatMujocoRunner(
        cfg=cfg,
        policy_path=args.policy,
        model_path=model_path,
    )
    runner.run()


if __name__ == "__main__":
    main()