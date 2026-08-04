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
        num_dofs=29
        num_actions = 15
        num_obs_per_step = 81 
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        # G1 初始高度
        init_height = 0.793
        stand_height = 0.77
        squat_height = 0.26


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
    
        # 初始化变量
        self.init_variables()

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
        self.num_dofs = self.cfg.sim.num_dofs          # 29
        self.num_actions = self.cfg.sim.num_actions    # 15
        # 策略输出的 15 维动作
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        # 1 维高度命令
        self.height_cmd = np.array([self.cfg.robot.stand_height], dtype=np.float32)
        # 观测历史：81 * 10 = 810
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32,
        )
        self.dof_pos = np.zeros(self.num_dofs)
        self.dof_vel = np.zeros(self.num_dofs)

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
            200, 150, 150, 200, 20, 20,    # 左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            200, 150, 150, 200, 20, 20,    # 右腿
            200, 200, 200,                  # 腰部: waist_yaw, waist_roll, waist_pitch
            100, 100, 50, 50, 40, 40,40,  # 左臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_*
            100, 100, 50, 50, 40, 40, 40   # 右臂
        ], dtype=np.float32)
        
        self.kds = np.array([
            5, 5, 5, 5, 2, 2,              # 左腿
            5, 5, 5, 5, 2, 2,              # 右腿
            5, 5, 5,                        # 腰部
            2, 2, 2, 2, 2, 2, 2,           # 左臂
            2, 2, 2, 2, 2, 2, 2            # 右臂
        ], dtype=np.float32)
        
        self.episode_length_buf = 0
        
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
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qpos[7:7 + self.num_dofs] = self.default_dof_pos.copy()
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

   
    def get_obs(self) -> np.ndarray:
        # 读取 29 个关节状态，MuJoCo 顺序
        dof_pos_mj = self.data.qpos[7:7 + self.num_dofs].copy()
        dof_vel_mj = self.data.qvel[6:6 + self.num_dofs].copy()

        # MuJoCo free joint:
        # qvel[0:3] 通常是 base 线速度
        # qvel[3:6] 通常是 base 角速度
        ang_vel_body = self.data.qvel[3:6].copy()

        # 投影重力
        quat = self.data.qpos[3:7].copy()  # MuJoCo: w, x, y, z
        projected_gravity = self.get_gravity_orientation(quat)

        # 关节状态转 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)

        # 高度命令，1 维
        height_cmd = self.height_cmd.astype(np.float32)  # shape: (1,)

        # 上一时刻动作，15 维
        last_action = self.action.astype(np.float32)

        # root height，1 维
        root_height = np.array([self.data.qpos[2]], dtype=np.float32)

        obs = np.concatenate(
            [
                ang_vel_body.astype(np.float32),          # 3
                projected_gravity.astype(np.float32),     # 3
                height_cmd,                               # 1
                joint_pos_isaac.astype(np.float32),       # 29
                joint_vel_isaac.astype(np.float32),       # 29
                last_action.astype(np.float32),           # 15
                root_height,                              # 1
            ],
            axis=0,
        ).astype(np.float32)

        assert obs.shape[0] == self.cfg.sim.num_obs_per_step, (
            f"obs dim wrong: {obs.shape[0]} != {self.cfg.sim.num_obs_per_step}"
        )

        self.obs_history = np.roll(
            self.obs_history,
            shift=-self.cfg.sim.num_obs_per_step,
        )
        self.obs_history[-self.cfg.sim.num_obs_per_step:] = obs.copy()

        return np.clip(
            self.obs_history,
            -self.cfg.sim.clip_observations,
            self.cfg.sim.clip_observations,
        )

    def position_control(self) -> np.ndarray:
        # 29 维目标位置，默认全身姿态
        target_q = self.default_dof_pos.copy()

        # 15 维 action：左腿6 + 右腿6 + 腰3
        actions_scaled = self.action * self.cfg.sim.action_scale

        # MuJoCo 前 15 个关节就是腿+腰
        target_q[:self.num_actions] = (
            self.default_dof_pos[:self.num_actions] + actions_scaled
        )

        # 手臂保持默认位置，不由策略控制
        return target_q

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        q = self.data.qpos[7:7 + self.num_dofs]
        dq = self.data.qvel[6:6 + self.num_dofs]

        tau = (target_q - q) * self.kps + (0.0 - dq) * self.kds
        return tau

    def run(self) -> None:
        """运行仿真循环"""
        
        # 创建 viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.stabilize_robot(duration=0.05)
        
        # 初始化观测历史
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()

        debug_counter = 0

        try:
            while self.viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:


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

                    # 设置死区（例如 0.1），防止轻微漂移
                    deadzone = 0.1
                    ax1 = 0.0 if abs(ax1) < deadzone else ax1

                    ax1 = self.joystick.get_axis_value(1)

                    deadzone = 0.1
                    ax1 = 0.0 if abs(ax1) < deadzone else ax1

                    # Pygame 一般：上推负，下拉正
                    # 不动/上推 -> 站立
                    # 下拉 -> 下蹲
                    squat_ratio = np.clip(ax1, 0.0, 1.0)

                    raw_height_cmd = (
                        self.cfg.robot.stand_height
                        - squat_ratio * (self.cfg.robot.stand_height - self.cfg.robot.squat_height)
                    )

                    # 可选：限速滤波
                    max_delta_per_s = 0.15
                    max_delta = max_delta_per_s * self.dt
                    delta = np.clip(raw_height_cmd - self.height_cmd[0], -max_delta, max_delta)

                    self.height_cmd[0] += delta

                # 获取观测并执行策略
                obs_np = self.get_obs()
                obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)

                action_tensor = self.policy(obs_tensor)

                self.action[:] = action_tensor.detach().cpu().numpy()[0, :self.num_actions]
                self.action = np.clip(
                    self.action,
                    -self.cfg.sim.clip_actions,
                    self.cfg.sim.clip_actions,
                )
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)# 调试输出
                debug_counter += 1
                if debug_counter <= 3:
                    latest_obs = obs_tensor[0, -self.cfg.sim.num_obs_per_step:]

                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs_tensor shape: {obs_tensor.shape}")
                    print(f"  latest_obs dim: {latest_obs.shape[0]}")
                    print(f"  current_obs first 7: {latest_obs[:7]}")
                    print(f"  height_cmd: {self.height_cmd[0]:.3f}")
                    print(f"  root_height: {self.data.qpos[2]:.3f}")
                    print(f"  action first 6: {self.action[:6]}")
                    print(f"  joint_pos_obs first 6: {latest_obs[7:13]}")
                    print(f"  joint_vel_obs first 6: {latest_obs[36:42]}")
                    # 执行 decimation 步
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    
                    # 计算目标位置并执行 PD 控制
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:self.num_dofs] = tau
                    
                    # 物理步进
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()
                    
                    # 时间控制
                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)
                
                self.episode_length_buf += 1
               

                # 定期打印状态

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"h_cmd={self.height_cmd[0]:.3f}, "
                        f"root_h={self.data.qpos[2]:.3f}, "
                        f"err={self.data.qpos[2] - self.height_cmd[0]:+.3f}"
                    )
                    
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
            self.data.ctrl[:self.num_dofs] = tau
            mujoco.mj_step(self.model, self.data)
            
        # 🔥 恢复自然记忆：连续获取 10 帧真实的静止状态，不要清零！
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()
            

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