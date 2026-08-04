import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pprint
import warnings
warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")
# --- 修复 1: 路径置顶 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
local_rsl_rl_path = os.path.abspath(os.path.join(current_dir, "../../rsl_rl"))
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

# --- 基础配置导入 (绝对安全) ---
from isaaclab.app import AppLauncher

# ==========================================
# 阶段 1：基础参数解析与引擎启动
# ==========================================
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--terrain", type=str, default=None, help="Terrain used for the environment")
# 先只添加 AppLauncher 的参数 (比如 --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args() # 第一次解析：忽略目前还不认识的 RSL-RL 参数

# Start camera rendering
if args_cli.task and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# 启动底层引擎 (在此之前绝对不能导入 legged_lab.utils 下的任何东西)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# 阶段 2：延迟导入与二次解析
# ==========================================
# 引擎启动后，此时再触发 utils.__init__.py 就是安全的了
import legged_lab.utils.cli_args as cli_args  # isort: skip
from legged_lab.utils import task_registry

# 补充 RSL-RL 参数，并进行第二次解析 (覆盖更新 args_cli)
cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --- 导入其他依赖包 ---
import torch
from export import  export_policy_as_jit, export_policy_as_onnx,export_g1_student_policy_as_jit
from isaaclab_tasks.utils import get_checkpoint_path

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner,MoeAmpOnPolicyRunner,DWAQAmpOnPolicyRunner,MoeOnPolicyRunner,FilmOnPolicyRunner
from rsl_rl.runners.g1_student_on_policy_runner import G1StudentOnPolicyRunner
from legged_lab.envs import * # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG,flex_terrain_CFG,ROUGH_PERLIN_TERRAINS_CFG,Flat_terrain


def play():
    runner: OnPolicyRunner | AmpOnPolicyRunner | MoeAmpOnPolicyRunner|DWAQAmpOnPolicyRunner|MoeOnPolicyRunner|G1StudentOnPolicyRunner|FilmOnPolicyRunner
    
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # 渲染与仿真参数微调
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50 if args_cli.num_envs is None else args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.height = (0.46, 0.76)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if "rough" == args_cli.terrain:
        Terrain=ROUGH_TERRAINS_CFG
    elif "rough_perlin" == args_cli.terrain:
        Terrain=ROUGH_PERLIN_TERRAINS_CFG
    elif "gravel" == args_cli.terrain:
        Terrain=GRAVEL_TERRAINS_CFG
    else:
        Terrain=Flat_terrain
    env_cfg.scene.terrain_generator = Terrain
    env_cfg.scene.terrain_type = "generator"

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    print("训练时的真实关节顺序:", env.robot.joint_names)
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # 动态加载 Runner
    runner_class: OnPolicyRunner | AmpOnPolicyRunner | MoeAmpOnPolicyRunner | DWAQAmpOnPolicyRunner | MoeOnPolicyRunner | G1StudentOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    # 导出模型 (建议保留，以便 Sim-to-Real)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
   
    if isinstance(runner, G1StudentOnPolicyRunner):
        export_g1_student_policy_as_jit(runner, path=export_model_dir, filename="policy.pt")
    else:
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )


    # ==============================
    # 手柄控制高度 h_cmd
    # 目标：什么也不动 -> 正常站立高度；左摇杆下拉 -> 下蹲
    # ==============================
    gamepad = None

    # 和训练命令范围保持一致
    stand_height = float(env_cfg.commands.stand_height)      # 通常 0.76
    squat_height = float(env_cfg.commands.ranges.height[0])  # 通常 0.46

    if not args_cli.headless:
        import evdev
        import threading

        class XboxController:
            """Linux evdev Xbox 手柄控制器.

            左摇杆上下：控制目标高度 h_cmd
                不动：站立高度
                上推：仍然保持站立高度
                下拉：下蹲，越往下越低

            A / Start：重置所有环境
            """

            AXIS_LEFT_Y = 1
            BTN_A = 304
            BTN_START = 315

            DEAD_ZONE = 0.08

            def __init__(self, device_path, reset_cb, squat_height, stand_height):
                self._reset_cb = reset_cb
                self._axes = {}
                self._running = True
                self._dev = evdev.InputDevice(device_path)

                self.SQUAT_HEIGHT = float(squat_height)
                self.STAND_HEIGHT = float(stand_height)

                # 读取轴范围，兼容 [-32768, 32767] 和 [0, 65535] 两种手柄轴格式
                self._abs_info = {}
                try:
                    self._abs_info[self.AXIS_LEFT_Y] = self._dev.absinfo(self.AXIS_LEFT_Y)
                except Exception:
                    self._abs_info[self.AXIS_LEFT_Y] = None

                print(f"[INFO] 手柄已连接: {self._dev.name} ({self._dev.path})")

                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()

            def _axis_center(self, code):
                info = self._abs_info.get(code)
                if info is not None:
                    return 0.5 * (info.min + info.max)
                return 0.0

            def _map_axis(self, code, raw):
                """把摇杆原始值映射到 [-1, 1]，并加入死区。"""
                info = self._abs_info.get(code)

                if info is not None:
                    center = 0.5 * (info.min + info.max)
                    half_range = 0.5 * (info.max - info.min)
                    if half_range <= 1e-6:
                        v = 0.0
                    else:
                        v = (float(raw) - center) / half_range
                else:
                    # 兼容老写法
                    v = float(raw) / 32767.0

                v = max(-1.0, min(1.0, v))
                return 0.0 if abs(v) < self.DEAD_ZONE else v

            def _read_loop(self):
                try:
                    for event in self._dev.read_loop():
                        if not self._running:
                            break

                        if event.type == evdev.ecodes.EV_ABS:
                            self._axes[event.code] = event.value

                        elif event.type == evdev.ecodes.EV_KEY:
                            if event.value == 1:
                                if event.code in (self.BTN_A, self.BTN_START):
                                    self._reset_cb()

                except Exception as e:
                    print(f"[WARN] 手柄读取线程退出: {e}")

            def get_target_height(self, device):
                """左摇杆下拉才下蹲；不动或上推都保持站立高度。"""

                # 如果没有收到任何摇杆事件，就使用轴中心值，保证“不动=站立”
                raw_y = self._axes.get(self.AXIS_LEFT_Y, self._axis_center(self.AXIS_LEFT_Y))

                # 一般情况下：
                #   上推 -> 负方向
                #   下拉 -> 正方向
                # 这里取负号后：
                #   上推 -> stick_y > 0
                #   不动 -> stick_y = 0
                #   下拉 -> stick_y < 0
                stick_y = -self._map_axis(self.AXIS_LEFT_Y, raw_y)

                # 只把“下拉”映射成下蹲比例：
                #   不动 stick_y=0       -> squat_ratio=0 -> stand_height
                #   上推 stick_y>0       -> squat_ratio=0 -> stand_height
                #   下拉 stick_y<0       -> squat_ratio>0 -> squat
                squat_ratio = torch.clamp(
                    torch.tensor(-stick_y, dtype=torch.float32, device=device),
                    min=0.0,
                    max=1.0,
                )

                target_height = self.STAND_HEIGHT - squat_ratio * (
                    self.STAND_HEIGHT - self.SQUAT_HEIGHT
                )

                return target_height

        # 自动查找 Xbox 手柄
        for p in evdev.list_devices():
            dev = evdev.InputDevice(p)

            if "x-box" in dev.name.lower() or "xbox" in dev.name.lower():
                try:
                    gamepad = XboxController(
                        p,
                        lambda: env.episode_length_buf.copy_(
                            torch.ones_like(env.episode_length_buf) * 1e6
                        ),
                        squat_height=squat_height,
                        stand_height=stand_height,
                    )
                except Exception as e:
                    print(f"[WARN] 无法打开手柄 {p}: {e}")

                break

            dev.close()

        if gamepad is None:
            print("[WARN] 未找到 Xbox 手柄，将使用固定站立高度运行。")

    # ==============================
    # 初始化观测和高度命令
    # ==============================
    obs, _ = env.get_observations()
    step_counter = 0

    filtered_height_cmd = torch.full(
        (env.num_envs,),
        stand_height,
        dtype=torch.float32,
        device=env.device,
    )

    # 高度命令限速，单位 m/s
    # 如果想响应更快，可改成 0.20；如果想更柔，可改成 0.10
    max_delta_per_s = 0.15

    # 先强制写入站立高度，确保刚开始就是正常站立命令
    env.command_generator.command[:, 0] = filtered_height_cmd
    obs, _ = env.get_observations()

    # ==============================
    # 主循环
    # ==============================
    while simulation_app.is_running():
        with torch.inference_mode():

            # 1. 读取目标高度
            if gamepad is not None:
                raw_target_height = gamepad.get_target_height(env.device)
            else:
                # 没有手柄时，默认一直站立
                raw_target_height = torch.tensor(
                    stand_height,
                    dtype=torch.float32,
                    device=env.device,
                )

            # 限制高度范围，避免异常输入
            raw_target_height = torch.clamp(
                raw_target_height,
                min=squat_height,
                max=stand_height,
            )

            # 2. 对高度命令做限速滤波
            max_delta = max_delta_per_s * env.step_dt

            delta = raw_target_height - filtered_height_cmd
            delta = torch.clamp(delta, -max_delta, max_delta)

            filtered_height_cmd += delta

            # 3. 写入高度命令
            env.command_generator.command[:, 0] = filtered_height_cmd

            # 4. 重新计算 obs，保证 policy 看到的是新的 h_cmd
            obs, _ = env.get_observations()

            # 5. 策略推理 + 环境 step
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if step_counter % 100 == 0:
                print(
                    f"[DEBUG] step={step_counter}, "
                    f"raw_target_height={raw_target_height.item():.3f}, "
                    f"filtered_height={filtered_height_cmd[0].item():.3f}, "
                    f"env_cmd={env.command_generator.command[0].cpu().numpy()}"
                )

            step_counter += 1

if __name__ == "__main__":
    play()
    simulation_app.close()