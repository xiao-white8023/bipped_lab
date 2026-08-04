import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")

current_dir = os.path.dirname(os.path.abspath(__file__))
local_rsl_rl_path = os.path.abspath(os.path.join(current_dir, "../../rsl_rl"))
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained g1_rough policy with Xbox controller commands.")
parser.add_argument("--task", type=str, default="g1_rough", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    choices=["flat", "rough", "rough_perlin", "gravel", "flex"],
    help="Optional terrain override. Omit to use the task config.",
)
parser.add_argument(
    "--no_export",
    action="store_true",
    help="Skip exporting policy.pt and policy.onnx after loading the checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import legged_lab.utils.cli_args as cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

import torch  # noqa: E402
from export import export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.terrains import (  # noqa: E402
    Flat_terrain,
    GRAVEL_TERRAINS_CFG,
    ROUGH_PERLIN_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    flex_terrain_CFG,
)
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import AmpOnPolicyRunner  # noqa: E402


TERRAINS = {
    "flat": Flat_terrain,
    "rough": ROUGH_TERRAINS_CFG,
    "rough_perlin": ROUGH_PERLIN_TERRAINS_CFG,
    "gravel": GRAVEL_TERRAINS_CFG,
    "flex": flex_terrain_CFG,
}


def play():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50 if args_cli.num_envs is None else args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.debug_vis = False

    if args_cli.terrain is not None:
        env_cfg.scene.terrain_generator = TERRAINS[args_cli.terrain]
        env_cfg.scene.terrain_type = "generator"

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.amp_num_preload_transitions = 1
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    print("训练时的真实关节顺序:", env.robot.joint_names)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    if not args_cli.no_export:
        export_model_dir = os.path.join(log_dir, "exported")
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    def apply_gamepad_command(command: torch.Tensor):
        env.command_generator.command[:, :3] = command.unsqueeze(0).expand(env.num_envs, -1)

    gamepad = None
    if not args_cli.headless:
        try:
            import evdev
            import threading
        except ImportError as exc:
            print(f"[WARN] 未安装 evdev，无法启用手柄控制，继续使用环境自动采样命令: {exc}")
        else:
            name_keywords = ("x-box", "xbox", "gamepad", "controller")

            class XboxController:
                """Linux evdev Xbox controller.

                Left stick up/down: vx
                Left stick left/right: vy
                Right stick left/right: wz
                A or Start: reset all environments
                """

                AXIS_LEFT_Y = evdev.ecodes.ABS_Y
                AXIS_LEFT_X = evdev.ecodes.ABS_X
                AXIS_RIGHT_X = evdev.ecodes.ABS_RX
                BTN_A = evdev.ecodes.BTN_SOUTH
                BTN_START = evdev.ecodes.BTN_START
                DEAD_ZONE = 0.08
                VX_SENSITIVITY = 1.0
                VY_SENSITIVITY = 1.0
                OMEGA_SENSITIVITY = 1.0

                def __init__(self, device_path, reset_cb):
                    self._reset_cb = reset_cb
                    self._axes = {}
                    self._axis_info = {}
                    self._running = True
                    self._dev = evdev.InputDevice(device_path)
                    for code, info in self._dev.capabilities(absinfo=True).get(evdev.ecodes.EV_ABS, []):
                        self._axis_info[code] = info
                        self._axes[code] = info.value
                    print(f"[INFO] 手柄已连接: {self._dev.name} ({self._dev.path})")
                    self._thread = threading.Thread(target=self._read_loop, daemon=True)
                    self._thread.start()

                def _map_axis(self, code):
                    raw = self._axes.get(code, 0)
                    info = self._axis_info.get(code)
                    if info is not None and info.max != info.min:
                        center = 0.5 * (info.max + info.min)
                        half_range = 0.5 * (info.max - info.min)
                        value = (raw - center) / half_range if half_range > 0 else 0.0
                    else:
                        value = raw / 32767.0
                    value = max(-1.0, min(1.0, value))
                    return 0.0 if abs(value) < self.DEAD_ZONE else value

                def _read_loop(self):
                    try:
                        for event in self._dev.read_loop():
                            if not self._running:
                                break
                            if event.type == evdev.ecodes.EV_ABS:
                                self._axes[event.code] = event.value
                            elif event.type == evdev.ecodes.EV_KEY and event.value == 1:
                                if event.code in (self.BTN_A, self.BTN_START):
                                    self._reset_cb()
                    except Exception as exc:
                        print(f"[WARN] 手柄读取线程退出: {exc}")

                def advance(self):
                    return torch.tensor(
                        [
                            -self._map_axis(self.AXIS_LEFT_Y) * self.VX_SENSITIVITY,
                            -self._map_axis(self.AXIS_LEFT_X) * self.VY_SENSITIVITY,
                            -self._map_axis(self.AXIS_RIGHT_X) * self.OMEGA_SENSITIVITY,
                        ],
                        dtype=torch.float32,
                        device=env.device,
                    )

            for device_path in evdev.list_devices():
                try:
                    device = evdev.InputDevice(device_path)
                    device_name = device.name
                    device.close()
                except OSError as exc:
                    print(f"[WARN] 跳过输入设备 {device_path}: {exc}")
                    continue
                if any(keyword in device_name.lower() for keyword in name_keywords):
                    try:
                        gamepad = XboxController(
                            device_path,
                            lambda: env.episode_length_buf.copy_(torch.ones_like(env.episode_length_buf) * 1e6),
                        )
                    except Exception as exc:
                        print(f"[WARN] 无法打开手柄 {device_path}: {exc}")
                    break

            if gamepad is None:
                print("[WARN] 未找到可用手柄，继续使用环境自动采样命令。")
    else:
        print("[WARN] headless 模式下不启用手柄，继续使用环境自动采样命令。")

    obs, _ = env.get_observations()
    step_counter = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if gamepad is not None:
                gamepad_cmd = gamepad.advance()
                apply_gamepad_command(gamepad_cmd)
                obs, _ = env.get_observations()

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if gamepad is not None:
                apply_gamepad_command(gamepad_cmd)

            if step_counter % 100 == 0:
                if gamepad is not None:
                    print(
                        f"[DEBUG] step={step_counter}, "
                        f"gamepad_cmd={gamepad_cmd.detach().cpu().numpy()}, "
                        f"env_cmd={env.command_generator.command[0].detach().cpu().numpy()}"
                    )
                else:
                    print(f"[DEBUG] step={step_counter}, cmd={env.command_generator.command[0].detach().cpu().numpy()}")

            step_counter += 1


if __name__ == "__main__":
    play()
    simulation_app.close()
