"""Play the current g1_renet policy in Isaac Lab.

The RENet actor observation layout is::

    proprio_history | depth_history | actor_mode | recovery_beta

Normal locomotion uses actor mode 0 (VP) or 1 (OP).  The environment recovery
state machine owns actor mode 2, so switching the locomotion estimator never
overrides an active Recovery episode.
"""

import argparse
import os
import sys
import traceback
import warnings


warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
LOCAL_RSL_RL_PATH = os.path.join(REPO_ROOT, "rsl_rl")
if LOCAL_RSL_RL_PATH not in sys.path:
    sys.path.insert(0, LOCAL_RSL_RL_PATH)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a trained g1_renet policy.")
parser.add_argument("--task", type=str, default="g1_renet", help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--vx", type=float, default=0.5, help="Fixed forward velocity without a gamepad.")
parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral velocity without a gamepad.")
parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw velocity without a gamepad.")
parser.add_argument(
    "--estimator",
    type=str,
    default="vp",
    choices=("op", "vp"),
    help="Initial locomotion estimator: OP uses proprioception, VP also uses depth.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    choices=("flat", "rough", "rough_perlin", "gravel", "flex"),
    help="Optional terrain override; the task terrain is used when omitted.",
)
parser.add_argument("--depth_noise", action="store_true", help="Keep the configured training-time depth noise.")
parser.add_argument("--depth_debug_vis", action="store_true", help="Visualize depth-camera rays.")
parser.add_argument(
    "--no_export",
    action="store_true",
    help="Skip exporting policy.pt and policy.onnx after loading the checkpoint.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    help="Export policy.pt and policy.onnx, then exit without entering the play loop.",
)
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Export directory (default: <checkpoint_run>/exported).",
)
parser.add_argument(
    "--dedicated_recovery",
    action="store_true",
    help="Keep dedicated Recovery reset environments; disabled for normal playback.",
)
parser.add_argument("--no_gamepad", action="store_true", help="Disable Linux evdev gamepad input.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# g1_renet always consumes a depth observation in VP mode.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Imports below this point may initialize Isaac/Omniverse modules.
import torch  # noqa: E402
from export import export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import legged_lab.utils.cli_args as cli_args  # noqa: E402
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
from rsl_rl.runners import RENetAmpOnPolicyRunner  # noqa: E402


cli_args.add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

TERRAINS = {
    "flat": Flat_terrain,
    "rough": ROUGH_TERRAINS_CFG,
    "rough_perlin": ROUGH_PERLIN_TERRAINS_CFG,
    "gravel": GRAVEL_TERRAINS_CFG,
    "flex": flex_terrain_CFG,
}


def configure_playback(env_cfg):
    """Turn the training configuration into a deterministic playback setup."""

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.noise.add_noise = False

    if hasattr(env_cfg.domain_rand.events, "push_robot"):
        env_cfg.domain_rand.events.push_robot = None

    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)

    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.camera.add_camera_noise = bool(args_cli.depth_noise)
    env_cfg.scene.camera.camera.debug_vis = bool(args_cli.depth_debug_vis)

    # Dedicated rows are a training sampler.  Natural falls can still enter
    # Recovery through the state machine when this sampler is disabled.
    env_cfg.recovery.dedicated_training_enable = bool(args_cli.dedicated_recovery)

    # Playback owns the OP/VP choice.  Do not silently force individual terrain
    # cells back to VP when the user explicitly selected OP.
    env_cfg.renet.mask_mode = args_cli.estimator
    env_cfg.renet.force_vp_terrain_names = []
    env_cfg.renet.force_vp_terrain_level = -1

    if args_cli.terrain is not None:
        env_cfg.scene.terrain_generator = TERRAINS[args_cli.terrain]
        env_cfg.scene.terrain_type = "generator"

    terrain_generator = env_cfg.scene.terrain_generator
    if terrain_generator is not None and hasattr(terrain_generator, "curriculum"):
        terrain_generator.curriculum = False

    return env_cfg


def create_gamepad(env):
    """Create an evdev controller, returning ``None`` when unavailable."""

    if args_cli.headless or args_cli.no_gamepad:
        return None

    try:
        import evdev
        import threading
    except ImportError as exc:
        print(f"[WARN] 未安装 evdev，使用固定速度命令: {exc}")
        return None

    class XboxController:
        """Xbox-style controller with command, reset, and RENet mode inputs."""

        AXIS_LEFT_Y = evdev.ecodes.ABS_Y
        AXIS_LEFT_X = evdev.ecodes.ABS_X
        AXIS_RIGHT_X = evdev.ecodes.ABS_RX
        BTN_RESET = {evdev.ecodes.BTN_SOUTH, evdev.ecodes.BTN_START}
        BTN_OP = getattr(evdev.ecodes, "BTN_TL", 310)
        BTN_VP = getattr(evdev.ecodes, "BTN_TR", 311)
        DEAD_ZONE = 0.08
        VX_SENSITIVITY = 1.0
        VY_SENSITIVITY = 1.0
        WZ_SENSITIVITY = 1.0

        def __init__(self, device_path):
            self._axes = {}
            self._axis_info = {}
            self._reset_requested = False
            self._estimator_requested = None
            self._lock = threading.Lock()
            self._running = True
            self._device = evdev.InputDevice(device_path)
            for code, info in self._device.capabilities(absinfo=True).get(evdev.ecodes.EV_ABS, []):
                self._axis_info[code] = info
                self._axes[code] = info.value
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[INFO] 手柄已连接: {self._device.name} ({self._device.path})")
            print("[INFO] 左摇杆=vx/vy，右摇杆=wz，L1=OP，R1=VP，A/Start=重置。")

        def _read_loop(self):
            try:
                for event in self._device.read_loop():
                    if not self._running:
                        break
                    with self._lock:
                        if event.type == evdev.ecodes.EV_ABS:
                            self._axes[event.code] = event.value
                        elif event.type == evdev.ecodes.EV_KEY and event.value == 1:
                            if event.code in self.BTN_RESET:
                                self._reset_requested = True
                            elif event.code == self.BTN_OP:
                                self._estimator_requested = "op"
                            elif event.code == self.BTN_VP:
                                self._estimator_requested = "vp"
            except (OSError, ValueError) as exc:
                if self._running:
                    print(f"[WARN] 手柄读取线程退出: {exc}")

        def _map_axis(self, code):
            raw = self._axes.get(code, 0)
            info = self._axis_info.get(code)
            if info is not None and info.max != info.min:
                center = 0.5 * (info.max + info.min)
                half_range = 0.5 * (info.max - info.min)
                value = (raw - center) / half_range
            else:
                value = raw / 32767.0
            value = max(-1.0, min(1.0, value))
            return 0.0 if abs(value) < self.DEAD_ZONE else value

        def poll(self):
            with self._lock:
                command = torch.tensor(
                    [
                        -self._map_axis(self.AXIS_LEFT_Y) * self.VX_SENSITIVITY,
                        -self._map_axis(self.AXIS_LEFT_X) * self.VY_SENSITIVITY,
                        -self._map_axis(self.AXIS_RIGHT_X) * self.WZ_SENSITIVITY,
                    ],
                    dtype=torch.float32,
                    device=env.device,
                )
                reset_requested = self._reset_requested
                estimator_requested = self._estimator_requested
                self._reset_requested = False
                self._estimator_requested = None
            return command, reset_requested, estimator_requested

        def close(self):
            self._running = False
            self._device.close()

    name_keywords = ("x-box", "xbox", "gamepad", "controller")
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
                return XboxController(device_path)
            except OSError as exc:
                print(f"[WARN] 无法打开手柄 {device_path}: {exc}")

    print("[WARN] 未找到可用手柄，使用固定速度命令。")
    return None


def patch_current_command(obs, command, env, policy_model):
    """Update the command without appending a duplicate observation history row."""

    env.command_generator.command[:, :3] = command.unsqueeze(0).expand(env.num_envs, -1)
    current_proprio_start = policy_model.proprio_actor_dim - policy_model.single_proprio_dim
    command_start = current_proprio_start + 6
    obs[:, command_start : command_start + 3] = command.unsqueeze(0).expand(env.num_envs, -1)
    return obs


def patch_estimator_mode(obs, estimator, env, policy_model):
    """Switch OP/VP in both the environment scheduler and current raw obs."""

    mask_value = 1.0 if estimator == "op" else 0.0
    env.cfg.renet.mask_mode = estimator
    env.renet_estimator_mask.fill_(mask_value)

    actor_mode_index = policy_model.proprio_actor_dim + policy_model.depth_flat_dim
    locomotion_mode = torch.full(
        (env.num_envs, 1), mask_value, dtype=obs.dtype, device=obs.device
    )
    recovery_mode = torch.full_like(locomotion_mode, 2.0)
    obs[:, actor_mode_index : actor_mode_index + 1] = torch.where(
        env.recovery_mask.unsqueeze(-1), recovery_mode, locomotion_mode
    )
    print(f"[INFO] RENet locomotion estimator switched to {estimator.upper()} (mask={mask_value:.1f}).")
    return obs


def play():
    if args_cli.task != "g1_renet":
        raise ValueError(f"play_new_renet.py only supports task 'g1_renet', got {args_cli.task!r}.")
    if args_cli.export_only and args_cli.no_export:
        raise ValueError("--export_only and --no_export cannot be used together.")

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg = configure_playback(env_cfg)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    # The runner constructs AMP components even for inference.  One transition
    # is enough to validate each expert-data interface and avoids a large,
    # unnecessary playback allocation.
    agent_cfg.amp_num_preload_transitions = 1
    agent_cfg.recovery_amp_num_preload_transitions = 1
    agent_cfg.num_steps_per_env = 1

    if agent_cfg.runner_class_name != "RENetAmpOnPolicyRunner":
        raise ValueError(
            "g1_renet must use RENetAmpOnPolicyRunner, got "
            f"{agent_cfg.runner_class_name!r}."
        )

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    print(f"[INFO] Isaac joint order: {env.robot.joint_names}")

    log_root_path = os.path.join(REPO_ROOT, "logs", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint: {resume_path}")

    runner = RENetAmpOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    policy_model = runner.alg.policy

    if not args_cli.no_export:
        export_model_dir = (
            os.path.abspath(os.path.expanduser(args_cli.export_dir))
            if args_cli.export_dir is not None
            else os.path.join(log_dir, "exported")
        )
        export_policy_as_jit(
            policy_model,
            runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            policy_model,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        print(f"[INFO] RENet deployment models exported to: {export_model_dir}")

    if args_cli.export_only:
        print("[INFO] Export-only mode completed; skipping the simulation loop.")
        return

    fixed_command = torch.tensor(
        [args_cli.vx, args_cli.vy, args_cli.wz], dtype=torch.float32, device=env.device
    )
    gamepad = create_gamepad(env)
    estimator = args_cli.estimator

    env.command_generator.command[:, :3] = fixed_command.unsqueeze(0).expand(env.num_envs, -1)
    obs, _ = env.get_observations()
    obs = patch_estimator_mode(obs, estimator, env, policy_model)
    step_counter = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if gamepad is None:
                    command = fixed_command
                    reset_requested = False
                    estimator_requested = None
                else:
                    command, reset_requested, estimator_requested = gamepad.poll()

                if reset_requested:
                    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
                    env.reset(env_ids)
                    env.command_generator.command[:, :3] = command.unsqueeze(0).expand(env.num_envs, -1)
                    obs, _ = env.get_observations()
                    print("[INFO] 已重置全部环境。")

                if estimator_requested is not None and estimator_requested != estimator:
                    estimator = estimator_requested
                    obs = patch_estimator_mode(obs, estimator, env, policy_model)

                obs = patch_current_command(obs, command, env, policy_model)
                actions = policy(obs)
                obs, _, _, extras = env.step(actions)

                if step_counter % 100 == 0:
                    actor_mode = extras["observations"]["actor_mode"].squeeze(-1)
                    vp_count = int((actor_mode == 0.0).sum().item())
                    op_count = int((actor_mode == 1.0).sum().item())
                    recovery_count = int((actor_mode == 2.0).sum().item())
                    print(
                        f"[DEBUG] step={step_counter}, cmd={command.detach().cpu().numpy()}, "
                        f"requested={estimator.upper()}, "
                        f"modes(VP/OP/Recovery)={vp_count}/{op_count}/{recovery_count}"
                    )

                step_counter += 1
    finally:
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    failed = False
    try:
        play()
    except BaseException:
        # Isaac Sim cleanup can take a long time after a partially initialized
        # scene.  Print the real failure before cleanup so it is not mistaken
        # for a hang at the environment's last status message.
        failed = True
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=failed or args_cli.export_only)
