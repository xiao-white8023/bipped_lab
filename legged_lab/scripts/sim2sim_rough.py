import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from joystick import JoyStick, JoystickButton  # noqa: E402
    JOYSTICK_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    JoyStick = None
    JoystickButton = None
    JOYSTICK_IMPORT_ERROR = exc


MUJOCO_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# Isaac Lab joint order used by the current g1_rough task.
LAB_DOF_NAMES = ['left_hip_pitch_joint', 
                 'right_hip_pitch_joint', 
                 'waist_yaw_joint', 
                 'left_hip_roll_joint', 
                 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 
                 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
                 'left_knee_joint', 
                 'right_knee_joint', 
                 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
                 'left_elbow_joint', 
                 'right_elbow_joint', 
                 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_roll_joint', 
                 'right_wrist_roll_joint']



class G1RoughSim2SimCfg:
    class sim:
        sim_duration = 100.0
        num_actions = 23
        num_obs_per_step = 78
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class gait:
        enable = True
        period = 0.8
        offset = 0.5

    class command:
        lin_vel_x = (-0.6, 1.0)
        lin_vel_y = (-0.5, 0.5)
        ang_vel_z = (-1.57, 1.57)
        deadzone = 0.1

    class robot:
        init_height = 0.793
        joint_armature = 0.1
        joint_damping = 0.001
        joint_frictionloss = 0.1


class G1RoughMujocoRunner:
    def __init__(self, cfg: G1RoughSim2SimCfg, policy_path: str, model_path: str):
        self.cfg = cfg

        print(f"[INFO] Loading MuJoCo model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        self.init_joystick()

        control_hz = 1.0 / (self.cfg.sim.dt * self.cfg.sim.decimation)
        obs_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        print(f"[INFO] Control frequency: {control_hz:.1f} Hz")
        print(f"[INFO] Actor obs: {self.cfg.sim.num_obs_per_step} x {self.cfg.sim.actor_obs_history_length} = {obs_dim}")

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions

        if self.model.nu < self.num_actions:
            raise RuntimeError(f"MuJoCo model has {self.model.nu} actuators, expected at least {self.num_actions}.")

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.command_vel = np.zeros(3, dtype=np.float32)
        self.episode_length_buf = 0

        self.phase = 0.0
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.leg_phase = np.zeros(2, dtype=np.float32)

        self.default_dof_pos = np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.42,
                -0.23,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.42,
                -0.23,
                0.0,
                0.0,
                0.35,
                0.18,
                0.0,
                0.87,
                0.0,
                0.35,
                -0.18,
                0.0,
                0.87,
                0.0,
            ],
            dtype=np.float32,
        )

        self.kps = np.array(
            [
                200.0,
                150.0,
                150.0,
                200.0,
                20.0,
                20.0,
                200.0,
                150.0,
                150.0,
                200.0,
                20.0,
                20.0,
                200.0,
                100.0,
                100.0,
                50.0,
                50.0,
                15.0,
                100.0,
                100.0,
                50.0,
                50.0,
                15.0,
            ],
            dtype=np.float32,
        )
        self.kds = np.array(
            [
                5.0,
                5.0,
                5.0,
                5.0,
                2.0,
                2.0,
                5.0,
                5.0,
                5.0,
                5.0,
                2.0,
                2.0,
                5.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
            dtype=np.float32,
        )
        self.torque_limits = np.array(
            [
                88.0,
                139.0,
                88.0,
                139.0,
                35.0,
                35.0,
                88.0,
                139.0,
                88.0,
                139.0,
                35.0,
                35.0,
                88.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
            ],
            dtype=np.float32,
        )

        self.obs_history = np.zeros(
            self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,
            dtype=np.float32,
        )

        self.viewer = None
        self.joystick = None
        self.use_joystick = False

    def init_joystick(self) -> None:
        if JoyStick is None:
            self.use_joystick = False
            print(f"[WARNING] Controller support is unavailable: {JOYSTICK_IMPORT_ERROR}. Command will stay zero.")
            return

        try:
            self.joystick = JoyStick()
            self.use_joystick = True
            print("[INFO] Xbox controller connected.")
        except RuntimeError as exc:
            self.use_joystick = False
            print(f"[WARNING] Controller init failed: {exc}. Command will stay zero.")

    def build_joint_mappings(self) -> None:
        real_mujoco_names = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                real_mujoco_names.append(name)

        if real_mujoco_names != MUJOCO_DOF_NAMES:
            print("\n========== MuJoCo joint order mismatch ==========")
            for idx, (expected, actual) in enumerate(zip(MUJOCO_DOF_NAMES, real_mujoco_names)):
                if expected != actual:
                    print(f"[WARN] index {idx}: expected {expected}, got {actual}")
            if len(real_mujoco_names) != len(MUJOCO_DOF_NAMES):
                print(f"[WARN] expected {len(MUJOCO_DOF_NAMES)} joints, got {len(real_mujoco_names)}")
            print("================================================\n")

        missing = sorted(set(LAB_DOF_NAMES) - set(real_mujoco_names))
        if missing:
            raise RuntimeError(f"Model is missing joints required by g1_rough: {missing}")

        self.mujoco_joint_ids = []
        self.qpos_addr_mj = []
        self.qvel_addr_mj = []
        for name in MUJOCO_DOF_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Model is missing MuJoCo joint: {name}")
            self.mujoco_joint_ids.append(joint_id)
            self.qpos_addr_mj.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_addr_mj.append(int(self.model.jnt_dofadr[joint_id]))

        self.mujoco_joint_ids = np.asarray(self.mujoco_joint_ids, dtype=np.int32)
        self.qpos_addr_mj = np.asarray(self.qpos_addr_mj, dtype=np.int32)
        self.qvel_addr_mj = np.asarray(self.qvel_addr_mj, dtype=np.int32)
        self.model.dof_armature[self.qvel_addr_mj] = self.cfg.robot.joint_armature
        self.model.dof_damping[self.qvel_addr_mj] = self.cfg.robot.joint_damping
        self.model.dof_frictionloss[self.qvel_addr_mj] = self.cfg.robot.joint_frictionloss

        actuator_by_joint = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None and joint_name not in actuator_by_joint:
                actuator_by_joint[joint_name] = actuator_id

        missing_actuators = sorted(set(MUJOCO_DOF_NAMES) - set(actuator_by_joint))
        if missing_actuators:
            raise RuntimeError(f"Model is missing actuators for joints: {missing_actuators}")
        self.ctrl_addr_mj = np.asarray([actuator_by_joint[name] for name in MUJOCO_DOF_NAMES], dtype=np.int32)

        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.mujoco_to_lab_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
        self.lab_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        self.default_dof_pos_lab = self.mj_to_lab(self.default_dof_pos)

        print("[INFO] Joint mapping ready.")
        print(f"[INFO] MuJoCo first 6 joints: {MUJOCO_DOF_NAMES[:6]}")
        print(f"[INFO] Isaac first 6 joints: {LAB_DOF_NAMES[:6]}")
        print(f"[INFO] qpos addr first 6: {self.qpos_addr_mj[:6].tolist()}")
        print(f"[INFO] qvel addr first 6: {self.qvel_addr_mj[:6].tolist()}")
        print(f"[INFO] ctrl addr first 6: {self.ctrl_addr_mj[:6].tolist()}")
        print(
            "[INFO] MuJoCo joint dynamics: "
            f"armature={self.cfg.robot.joint_armature}, "
            f"damping={self.cfg.robot.joint_damping}, "
            f"frictionloss={self.cfg.robot.joint_frictionloss}"
        )
        print("[INFO] Isaac action -> MuJoCo joint mapping:")
        for lab_idx, joint_name in enumerate(LAB_DOF_NAMES):
            mj_idx = mujoco_indices[joint_name]
            print(
                f"  action[{lab_idx:02d}] {joint_name} -> "
                f"mj[{mj_idx:02d}], qpos[{self.qpos_addr_mj[mj_idx]}], ctrl[{self.ctrl_addr_mj[mj_idx]}]"
            )

    def set_initial_pose(self) -> None:
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_addr_mj] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] Initial height: {self.data.qpos[2]:.3f} m")

    def mj_to_lab(self, array_mj: np.ndarray) -> np.ndarray:
        return array_mj[self.mujoco_to_lab_idx]

    def lab_to_mj(self, array_lab: np.ndarray) -> np.ndarray:
        return array_lab[self.lab_to_mujoco_idx]

    def get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        return np.array(
            [
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ],
            dtype=np.float32,
        )

    def get_obs(self) -> np.ndarray:
        dof_pos_mj = self.data.qpos[self.qpos_addr_mj].copy()
        dof_vel_mj = self.data.qvel[self.qvel_addr_mj].copy()

        ang_vel_body = self.data.qvel[3:6].copy()
        projected_gravity = self.get_gravity_orientation(self.data.qpos[3:7].copy())

        joint_pos_lab = self.mj_to_lab(dof_pos_mj - self.default_dof_pos)
        joint_vel_lab = self.mj_to_lab(dof_vel_mj)

        obs = np.concatenate(
            [
                ang_vel_body,
                projected_gravity,
                self.command_vel,
                joint_pos_lab,
                joint_vel_lab,
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.cfg.sim.num_obs_per_step:
            raise RuntimeError(f"Expected single obs dim {self.cfg.sim.num_obs_per_step}, got {obs.shape[0]}.")

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def calculate_gait_para(self) -> None:
        if not self.cfg.gait.enable:
            return

        t = self.episode_length_buf * self.dt
        self.phase = (t % self.cfg.gait.period) / self.cfg.gait.period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.cfg.gait.offset) % 1.0
        self.leg_phase[0] = self.phase_left
        self.leg_phase[1] = self.phase_right

    def update_command_from_joystick(self) -> bool:
        if not self.use_joystick:
            return False

        self.joystick.update()
        if self.joystick.is_button_pressed(JoystickButton.SELECT):
            print("[INFO] Controller SELECT pressed, exiting.")
            return True

        if self.joystick.is_button_pressed(JoystickButton.A):
            self.command_vel[:] = 0.0
            return False

        if self.joystick.is_button_released(JoystickButton.START):
            print("[INFO] Controller START released, resetting robot.")
            self.reset_robot()
            return False

        ax1 = self.apply_deadzone(self.joystick.get_axis_value(1))
        ax0 = self.apply_deadzone(self.joystick.get_axis_value(0))
        ax3 = self.apply_deadzone(self.joystick.get_axis_value(3))

        self.command_vel[0] = np.clip(-ax1 * 1.0, *self.cfg.command.lin_vel_x)
        self.command_vel[1] = np.clip(-ax0 * 0.5, *self.cfg.command.lin_vel_y)
        self.command_vel[2] = np.clip(-ax3 * 1.57, *self.cfg.command.ang_vel_z)
        return False

    def apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.cfg.command.deadzone else value

    def position_control(self) -> np.ndarray:
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab_to_mj(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.qpos_addr_mj]
        dq = self.data.qvel[self.qvel_addr_mj]
        tau = (target_q - q) * self.kps - dq * self.kds
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def reset_robot(self) -> None:
        self.action[:] = 0.0
        self.command_vel[:] = 0.0
        self.episode_length_buf = 0
        self.leg_phase[:] = 0.0
        self.obs_history[:] = 0.0
        self.set_initial_pose()
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()

    def stabilize_robot(self, duration: float = 2.0) -> None:
        target_pos = self.default_dof_pos.copy()
        num_steps = int(duration / self.cfg.sim.dt)
        for i in range(num_steps):
            self.data.qpos[0:2] = 0.0
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[0:6] = 0.0

            tau = self.pd_control(target_pos)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.ctrl_addr_mj] = tau
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                if i % 10 == 0:
                    self.viewer.sync()
                time.sleep(self.cfg.sim.dt)

        self.obs_history[:] = 0.0
        for _ in range(self.cfg.sim.actor_obs_history_length):
            self.get_obs()

        print(f"[INFO] Stabilized height: {self.data.qpos[2]:.4f} m")

    def run(self) -> None:
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.stabilize_robot()

        print("\n[INFO] Xbox: left stick = vx/vy, right stick X = yaw rate, A = stop, START = reset, SELECT = exit")
        print("[INFO] Press Ctrl+C in terminal to exit.\n")

        debug_counter = 0
        try:
            while self.viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                if self.update_command_from_joystick():
                    break

                obs = self.get_obs()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.inference_mode():
                    action_tensor = self.policy(obs_tensor)
                self.action[:] = action_tensor.squeeze(0).detach().cpu().numpy()[: self.num_actions]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

                debug_counter += 1
                if debug_counter <= 3:
                    latest_obs = obs[-self.cfg.sim.num_obs_per_step :]
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs shape: {obs_tensor.shape}")
                    print(f"  latest obs first 9: {latest_obs[:9]}")
                    print(f"  command: {self.command_vel}")
                    print(f"  leg_phase: {self.leg_phase}")
                    print(f"  action first 6 (Isaac order): {self.action[:6]}")

                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:] = 0.0
                    self.data.ctrl[self.ctrl_addr_mj] = tau
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()

                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)

                self.episode_length_buf += 1
                self.calculate_gait_para()

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], "
                        f"phase=[{self.leg_phase[0]:.2f}, {self.leg_phase[1]:.2f}], "
                        f"h={self.data.qpos[2]:.3f}m"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self.viewer.close()
            print("[INFO] Simulation finished.")


def find_latest_exported_policy(log_root: str) -> str:
    candidates = []
    if os.path.isdir(log_root):
        for run_name in os.listdir(log_root):
            policy_path = os.path.join(log_root, run_name, "exported", "policy.pt")
            if os.path.isfile(policy_path):
                candidates.append(policy_path)
    if not candidates:
        return os.path.join(log_root, "exported", "policy.pt")
    return max(candidates, key=os.path.getmtime)


def get_available_scenes(mjcf_dir: str) -> dict:
    scenes = {}
    if os.path.isdir(mjcf_dir):
        for filename in os.listdir(mjcf_dir):
            if filename in ["scene.xml", "flat_scene.xml", "rough_scene.xml", "slope_scene.xml"]:
                scenes[filename.replace(".xml", "").replace("_scene", "")] = os.path.join(mjcf_dir, filename)
            elif filename.endswith("_scene.xml"):
                scenes[filename.replace("_scene.xml", "")] = os.path.join(mjcf_dir, filename)
    return scenes


def main() -> None:
    legged_lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    mjcf_dir = os.path.join(legged_lab_root, "legged_lab/assets/g1/g1_23dof")

    default_policy = find_latest_exported_policy(os.path.join(legged_lab_root, "logs/g1_rough"))
    default_model = os.path.join(mjcf_dir, "g1_23dof_rev_1_0.xml")
    available_scenes = get_available_scenes(mjcf_dir)
    scene_names = list(available_scenes.keys())

    parser = argparse.ArgumentParser(
        description="G1 rough 23DOF Sim2Sim with Xbox controller.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--policy", type=str, default=default_policy, help="Path to exported policy.pt")
    parser.add_argument("--model", type=str, default=default_model, help="MuJoCo XML path used when --scene is omitted")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=scene_names if scene_names else None,
        help=f"Scene name. Available: {', '.join(scene_names) if scene_names else 'none'}",
    )
    parser.add_argument("--scene-file", type=str, default=None, help="Custom scene XML path, higher priority than --scene")
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument("--list-scenes", action="store_true", help="List available scene XML files")
    parser.add_argument("--check-mapping", action="store_true", help="Print joint/actuator mapping and exit")
    args = parser.parse_args()

    if args.list_scenes:
        print("\nAvailable scenes:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"Scene directory: {mjcf_dir}")
        return

    if args.scene_file:
        model_path = args.scene_file
    elif args.scene:
        model_path = available_scenes[args.scene]
    else:
        model_path = args.model

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file does not exist: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"[ERROR] MuJoCo XML does not exist: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("G1 Rough 23DOF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"MuJoCo XML: {model_path}")
    if args.scene:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    cfg = G1RoughSim2SimCfg()
    cfg.sim.sim_duration = args.duration

    runner = G1RoughMujocoRunner(cfg=cfg, policy_path=args.policy, model_path=model_path)
    if args.check_mapping:
        print("[INFO] Mapping check finished; viewer was not launched.")
        return

    runner.run()


if __name__ == "__main__":
    main()
