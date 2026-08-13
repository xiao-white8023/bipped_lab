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

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner,MoeAmpOnPolicyRunner,DWAQAmpOnPolicyRunner,MoeOnPolicyRunner,FilmOnPolicyRunner, RENetAmpOnPolicyRunner
from rsl_rl.runners.g1_student_on_policy_runner import G1StudentOnPolicyRunner
from legged_lab.envs import * # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.attention_height_scan_visualizer import HeightScanAttentionVisualizer
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG,flex_terrain_CFG,ROUGH_PERLIN_TERRAINS_CFG


def play():
    runner: OnPolicyRunner | AmpOnPolicyRunner | RENetAmpOnPolicyRunner | MoeAmpOnPolicyRunner|DWAQAmpOnPolicyRunner|MoeOnPolicyRunner|G1StudentOnPolicyRunner|FilmOnPolicyRunner
    
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # 渲染与仿真参数微调
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50 if args_cli.num_envs is None else args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if "rough" == args_cli.terrain:
        Terrain=ROUGH_TERRAINS_CFG
    elif "rough_perlin" == args_cli.terrain:
        Terrain=ROUGH_PERLIN_TERRAINS_CFG
    elif "gravel" == args_cli.terrain:
        Terrain=GRAVEL_TERRAINS_CFG
    else:
        Terrain=env_cfg.scene.terrain_generator
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
    attention_visualizer = None
    if (
        not args_cli.headless
        and env_cfg.scene.height_scanner.enable_height_scan
        and hasattr(runner.alg.policy, "get_attention_weights")
    ):
        attention_visualizer = HeightScanAttentionVisualizer(env, runner.alg.policy, env_id=0)
        print("[INFO] Height-scan attention visualization enabled: blue=scan, yellow/red=attention focus.")

    # 导出模型 (建议保留，以便 Sim-to-Real)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
   
    if isinstance(runner, G1StudentOnPolicyRunner):
        export_g1_student_policy_as_jit(runner, path=export_model_dir, filename="policy.pt")
    else:
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
    gamepad = None

    if not args_cli.headless:
        import evdev
        import threading
        
        class XboxController:
            """Linux evdev Xbox 手柄控制器.

            左摇杆上下: v_x (前进/后退)
            左摇杆左右: v_y (横向移动)
            右摇杆左右: omega_z (转向)
            A 键: 重置所有环境
            """
            AXIS_LEFT_Y = 1   # 左摇杆 Y -> v_x
            AXIS_LEFT_X = 0   # 左摇杆 X -> v_y
            AXIS_RIGHT_X = 3  # 右摇杆 X -> omega_z
            BTN_A = 304       # evdev BTN_SOUTH
            BTN_START = 315   # evdev BTN_START
            DEAD_ZONE = 0.08
            VX_SENSITIVITY = 1.0
            VY_SENSITIVITY = 1.0
            OMEGA_SENSITIVITY = 1.0

            def __init__(self, device_path, reset_cb):
                self._reset_cb = reset_cb
                self._axes = {}
                self._running = True
                self._dev = evdev.InputDevice(device_path)
                print(f"[INFO] 手柄已连接: {self._dev.name} ({self._dev.path})")
                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()

            def _map_axis(self, raw):
                v = raw / 32767.0
                return 0.0 if abs(v) < self.DEAD_ZONE else v

            def _read_loop(self):
                try:
                    for event in self._dev.read_loop():
                        if not self._running:
                            break
                        if event.type == evdev.ecodes.EV_ABS:
                            self._axes[event.code] = event.value
                        elif event.type == evdev.ecodes.EV_KEY:
                            if event.value == 1:  # press
                                if event.code in (self.BTN_A, self.BTN_START):
                                    self._reset_cb()
                except Exception as e:
                    print(f"[WARN] 手柄读取线程退出: {e}")

            def advance(self):
                return torch.tensor([
                    -self._map_axis(self._axes.get(self.AXIS_LEFT_Y, 0)) * self.VX_SENSITIVITY,
                    -self._map_axis(self._axes.get(self.AXIS_LEFT_X, 0)) * self.VY_SENSITIVITY,
                    -self._map_axis(self._axes.get(self.AXIS_RIGHT_X, 0)) * self.OMEGA_SENSITIVITY,
                ], dtype=torch.float32, device=env.device)

        gamepad = None
        # 自动查找 Xbox 手柄
        for p in evdev.list_devices():
            dev = evdev.InputDevice(p)
            if "x-box" in dev.name.lower() or "xbox" in dev.name.lower():
                try:
                    gamepad = XboxController(p, lambda: env.episode_length_buf.copy_(
                        torch.ones_like(env.episode_length_buf) * 1e6))
                except Exception as e:
                    print(f"[WARN] 无法打开手柄 {p}: {e}")
                break
            dev.close()
        if gamepad is None:
            print("[WARN] 未找到 Xbox 手柄，速度命令保持为零。")

    def apply_gamepad_command():
        if gamepad is None:
            return None
        gamepad_cmd = gamepad.advance()
        env.command_generator.command[:] = gamepad_cmd.unsqueeze(0).expand(env.num_envs, -1)
        return gamepad_cmd

    apply_gamepad_command()
    obs, _ = env.get_observations()
    step_counter = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            gamepad_cmd = apply_gamepad_command()
            if gamepad_cmd is not None:
                obs, _ = env.get_observations()
                if step_counter % 100 == 0:
                    print(
                        f"[DEBUG] step={step_counter}, "
                        f"gamepad_cmd={gamepad_cmd.cpu().numpy()}, "
                        f"env_cmd={env.command_generator.command[0].cpu().numpy()}"
                    )
            actions = policy(obs)
            if attention_visualizer is not None:
                attention_visualizer.update(step_counter)
            obs, _, _, _ = env.step(actions)
            if gamepad is not None:
                apply_gamepad_command()
                obs, _ = env.get_observations()

            step_counter += 1

if __name__ == "__main__":
    play()
    simulation_app.close()
