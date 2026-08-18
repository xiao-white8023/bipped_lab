"""MuJoCo sim-to-sim runner for the current ``g1_renet`` task.

The exported policy observation is exactly::

    proprio_history[10 x 78] | depth_history[2 x 18 x 32]
    | actor_mode[1] | recovery_beta[1]

``actor_mode`` is 0 for VP, 1 for OP, and 2 for Recovery.  Normal locomotion
actions target ``default_q + 0.25 * action``.  Recovery actions intentionally
use the task's different interpretation, ``current_q + beta * action``.

This module reuses the tested MJCF camera patching, depth preprocessing, and
23-DoF Isaac/MuJoCo joint mapping from ``sim2sim_renet.py`` while supplying the
mode-aware deployment loop required by the new RENet checkpoint/exporter.
"""

from __future__ import annotations

import argparse
import math
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

from sim2sim_renet import (  # noqa: E402
    G1RENetMujocoRunner,
    G1RENetSim2SimCfg,
    JoystickButton,
    find_latest_exported_policy,
    get_available_scenes,
)


EXPECTED_PROPRIO_DIM = 10 * 78
EXPECTED_DEPTH_DIM = 2 * 18 * 32
EXPECTED_POLICY_OBS_DIM = EXPECTED_PROPRIO_DIM + EXPECTED_DEPTH_DIM + 2
EXPECTED_ACTION_DIM = 23


class G1NewRENetSim2SimCfg(G1RENetSim2SimCfg):
    """Deployment settings matching ``RENet_cfg.py``."""

    class sim(G1RENetSim2SimCfg.sim):
        headless = False
        realtime = True
        stabilization_duration = 2.0

    class renet(G1RENetSim2SimCfg.renet):
        actor_mode_vp = 0.0
        actor_mode_op = 1.0
        actor_mode_recovery = 2.0
        normal_beta = 0.25

    class recovery:
        enable = True
        max_duration_s = 6.0
        upright_threshold = 0.93
        success_height_ratio = 0.80
        torso_force_threshold = 1.0
        beta = 0.25

    class input:
        enable_joystick = True


class G1NewRENetMujocoRunner(G1RENetMujocoRunner):
    """RENet MuJoCo runner with automatic locomotion/Recovery routing."""

    def init_variables(self) -> None:
        # Called polymorphically by the parent constructor.
        super().init_variables()
        self.recovery_active = False
        self.recovery_steps = 0
        self.recovery_trigger_armed = True
        self.manual_recovery_requested = False
        self.last_torso_contact_force = 0.0
        self.nominal_torso_height = 0.0
        self.torso_body_id = -1
        self.robot_root_body_id = -1
        self._terrain_geom_ids = np.empty(0, dtype=np.int32)
        self._terrain_geom_group_mask = np.zeros(6, dtype=np.uint8)
        self._terrain_geom_group_mask[5] = 1

    def build_joint_mappings(self) -> None:
        super().build_joint_mappings()

        torso_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso_link",
        )
        if torso_body_id < 0:
            raise RuntimeError("Current g1_renet Recovery requires MuJoCo body 'torso_link'.")
        self.torso_body_id = int(torso_body_id)

        free_joint_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_joint_ids.size != 1:
            raise RuntimeError(
                "Expected exactly one floating-base joint, got "
                f"{free_joint_ids.size}."
            )
        self.robot_root_body_id = int(self.model.jnt_bodyid[int(free_joint_ids[0])])

        geom_body_ids = np.asarray(self.model.geom_bodyid, dtype=np.int32)
        geom_root_ids = np.asarray(self.model.body_rootid[geom_body_ids], dtype=np.int32)
        self._terrain_geom_ids = np.flatnonzero(
            geom_root_ids != self.robot_root_body_id
        ).astype(np.int32)
        if self._terrain_geom_ids.size == 0:
            raise RuntimeError("No non-robot terrain geometry was found in the MuJoCo scene.")

        # MuJoCo ray queries filter by one of six visualization groups.  Put
        # non-robot geoms in a dedicated group so the height ray cannot hit the
        # robot itself. This does not change collision properties.
        self.model.geom_group[self._terrain_geom_ids] = 5

    def set_initial_pose(self) -> None:
        super().set_initial_pose()
        if self.torso_body_id < 0:
            return
        torso_height = self._torso_height_above_terrain()
        if torso_height is None:
            torso_height = float(self.data.xpos[self.torso_body_id, 2])
            print(
                "[WARNING] Initial terrain-height ray missed; using world-Z "
                f"torso height {torso_height:.3f} m."
            )
        if not math.isfinite(torso_height) or torso_height <= 0.0:
            raise RuntimeError(f"Invalid nominal torso height: {torso_height}.")
        self.nominal_torso_height = torso_height

    def init_joystick(self) -> None:
        if not self.cfg.input.enable_joystick:
            self.joystick = None
            self.use_joystick = False
            print("[INFO] Joystick disabled; using fixed --vx/--vy/--wz commands.")
            return
        super().init_joystick()

    @property
    def actor_mode(self) -> float:
        if self.recovery_active:
            return self.cfg.renet.actor_mode_recovery
        if self.estimator == "op":
            return self.cfg.renet.actor_mode_op
        return self.cfg.renet.actor_mode_vp

    @property
    def active_beta(self) -> float:
        if self.recovery_active:
            return float(self.cfg.recovery.beta)
        return float(self.cfg.renet.normal_beta)

    def current_mode_text(self) -> str:
        if self.recovery_active:
            return f"Recovery mode=2 beta={self.active_beta:.2f}"
        return f"{self.estimator.upper()} mode={int(self.actor_mode)} beta={self.active_beta:.2f}"

    def build_policy_obs(self) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        proprio_history = self.get_proprio_obs()
        proprio_tensor = torch.from_numpy(proprio_history).float()
        depth_flat = torch.stack(list(self.depth_buffer)).reshape(-1).float()
        control = torch.tensor(
            [self.actor_mode, self.active_beta],
            dtype=torch.float32,
        )
        policy_obs = torch.cat(
            [proprio_tensor, depth_flat, control],
            dim=-1,
        ).unsqueeze(0)

        if proprio_tensor.numel() != EXPECTED_PROPRIO_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_PROPRIO_DIM} proprio values, got {proprio_tensor.numel()}."
            )
        if depth_flat.numel() != EXPECTED_DEPTH_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_DEPTH_DIM} depth values, got {depth_flat.numel()}."
            )
        if policy_obs.shape != (1, EXPECTED_POLICY_OBS_DIM):
            raise RuntimeError(
                "Expected g1_renet policy observation shape "
                f"(1, {EXPECTED_POLICY_OBS_DIM}), got {tuple(policy_obs.shape)}."
            )
        return policy_obs, proprio_history, depth_flat

    def validate_deployment_policy(self) -> None:
        """Exercise every exported mode before starting physics control."""

        observations = torch.zeros(1, EXPECTED_POLICY_OBS_DIM, dtype=torch.float32)
        actions_by_mode = []
        with torch.inference_mode():
            for actor_mode in (
                self.cfg.renet.actor_mode_vp,
                self.cfg.renet.actor_mode_op,
                self.cfg.renet.actor_mode_recovery,
            ):
                observations[:, -2] = actor_mode
                observations[:, -1] = (
                    self.cfg.recovery.beta
                    if actor_mode == self.cfg.renet.actor_mode_recovery
                    else self.cfg.renet.normal_beta
                )
                actions = self.policy(observations)
                if actions.shape != (1, EXPECTED_ACTION_DIM):
                    raise RuntimeError(
                        "Exported policy must return shape "
                        f"(1, {EXPECTED_ACTION_DIM}), got {tuple(actions.shape)}."
                    )
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"Exported policy returned non-finite actions for mode {actor_mode}.")
                actions_by_mode.append(actions.clone())

        # A legacy trace made only in VP mode often produces identical mode
        # outputs because OP/Recovery were removed from the graph.
        for first in range(len(actions_by_mode)):
            for second in range(first + 1, len(actions_by_mode)):
                if torch.allclose(actions_by_mode[first], actions_by_mode[second]):
                    raise RuntimeError(
                        "Exported policy does not preserve dynamic VP/OP/Recovery routing. "
                        "Re-export it with play_new_renet.py --export_only."
                    )
        print(
            "[INFO] Deployment policy validated: obs=(1, 1934), actions=(1, 23), "
            "VP/OP/Recovery routing active."
        )

    def update_command_from_joystick(self) -> tuple[bool, bool]:
        should_exit, depth_noise_active = super().update_command_from_joystick()
        if self.use_joystick and self.joystick.is_button_released(JoystickButton.B):
            self.manual_recovery_requested = True
            print("[INFO] Controller B released: manual Recovery requested.")
        return should_exit, depth_noise_active

    def position_control(self) -> np.ndarray:
        action_mj = self.lab_to_mj(self.action)
        if self.recovery_active:
            current_q = self.data.qpos[self.qpos_addr_mj].copy()
            return current_q + self.active_beta * action_mj
        return self.default_dof_pos + self.cfg.sim.action_scale * action_mj

    def _torso_contact_force(self) -> float:
        """Return the largest current contact force involving torso_link."""

        maximum_force = 0.0
        force_torque = np.zeros(6, dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if self.torso_body_id not in (body1, body2):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_id, force_torque)
            maximum_force = max(maximum_force, float(np.linalg.norm(force_torque[:3])))
        return maximum_force

    def _torso_height_above_terrain(self) -> float | None:
        """Cast the training-equivalent vertical terrain ray below torso_link."""

        if self.torso_body_id < 0:
            return None
        ray_origin = np.asarray(self.data.xpos[self.torso_body_id], dtype=np.float64).copy()
        ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            ray_origin,
            ray_direction,
            self._terrain_geom_group_mask,
            1,
            -1,
            geom_id,
        )
        if distance < 0.0 or geom_id[0] < 0 or not math.isfinite(distance):
            return None
        return float(distance)

    def _recovery_ready(self) -> tuple[bool, float, float | None]:
        projected_gravity = self.get_gravity_orientation(self.data.qpos[3:7].copy())
        upright_value = float(-projected_gravity[2])
        torso_height = self._torso_height_above_terrain()
        height_threshold = self.cfg.recovery.success_height_ratio * self.nominal_torso_height
        ready = (
            upright_value >= self.cfg.recovery.upright_threshold
            and torso_height is not None
            and torso_height >= height_threshold
        )
        return ready, upright_value, torso_height

    def _enter_recovery(self, reason: str) -> None:
        self.recovery_active = True
        self.recovery_steps = 0
        self.recovery_trigger_armed = False
        print(
            f"[RECOVERY] Entered mode=2 ({reason}); beta={self.active_beta:.2f}, "
            f"torso_force={self.last_torso_contact_force:.2f} N."
        )

    def _update_recovery_state(self, torso_force: float) -> None:
        self.last_torso_contact_force = torso_force
        locomotion_failure = torso_force > self.cfg.recovery.torso_force_threshold

        if not self.cfg.recovery.enable:
            self.manual_recovery_requested = False
            return

        if not self.recovery_active:
            if self.manual_recovery_requested:
                self._enter_recovery("manual controller request")
            elif self.recovery_trigger_armed and locomotion_failure:
                self._enter_recovery("torso contact")
            elif not locomotion_failure:
                self.recovery_trigger_armed = True
            self.manual_recovery_requested = False
            return

        self.manual_recovery_requested = False
        self.recovery_steps += 1
        ready, upright_value, torso_height = self._recovery_ready()
        if ready:
            self.recovery_active = False
            self.recovery_steps = 0
            # Match the training state machine: only a later clear locomotion
            # step re-arms another natural Recovery trigger.
            self.recovery_trigger_armed = False
            print(
                "[RECOVERY] Success; returning to "
                f"{self.estimator.upper()} (upright={upright_value:.3f}, "
                f"torso_height={torso_height:.3f} m)."
            )
            return

        max_steps = max(1, math.ceil(self.cfg.recovery.max_duration_s / self.dt))
        if self.recovery_steps >= max_steps:
            height_text = "invalid" if torso_height is None else f"{torso_height:.3f} m"
            print(
                "[RECOVERY] Timed out; resetting robot "
                f"(upright={upright_value:.3f}, torso_height={height_text})."
            )
            self.reset_robot()

    def reset_robot(self) -> None:
        super().reset_robot()
        self.recovery_active = False
        self.recovery_steps = 0
        self.recovery_trigger_armed = True
        self.manual_recovery_requested = False
        self.last_torso_contact_force = 0.0

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.fig = None
        if self._temp_mjcf_dir is not None:
            self._temp_mjcf_dir.cleanup()
            self._temp_mjcf_dir = None

    def run(self) -> None:
        if not self.cfg.sim.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.stabilize_robot(duration=self.cfg.sim.stabilization_duration)
        run_start_time = float(self.data.time)

        print(
            "\n[INFO] Xbox: left stick=vx/vy, right stick X=yaw, A=stop, "
            "START=reset, SELECT=exit"
        )
        print("[INFO] RENet: L1=OP, R1=VP, B release=manual Recovery, Y=collapsed VP depth.")
        print("[INFO] Automatic Recovery: torso contact -> mode 2; upright + height -> locomotion.")
        print(f"[INFO] Current mode: {self.current_mode_text()}")
        print("[INFO] Press Ctrl+C in terminal to exit.\n")

        debug_counter = 0
        try:
            while float(self.data.time) - run_start_time < self.cfg.sim.sim_duration:
                if self.viewer is not None and not self.viewer.is_running():
                    break

                should_exit, depth_noise_active = self.update_command_from_joystick()
                if should_exit:
                    break

                if self.episode_length_buf % self.cfg.robot.depth_update_interval == 0:
                    self.update_depth_vision(noise_active=depth_noise_active)

                policy_obs, proprio_history, depth_flat = self.build_policy_obs()
                with torch.inference_mode():
                    action_tensor = self.policy(policy_obs)
                action_np = action_tensor.squeeze(0).detach().cpu().numpy()
                if action_np.shape != (self.num_actions,) or not np.isfinite(action_np).all():
                    raise RuntimeError(
                        f"Invalid policy action: shape={action_np.shape}, finite={np.isfinite(action_np).all()}."
                    )
                self.action[:] = np.clip(
                    action_np,
                    -self.cfg.sim.clip_actions,
                    self.cfg.sim.clip_actions,
                )

                debug_counter += 1
                if debug_counter <= 3:
                    latest_obs = proprio_history[-self.cfg.sim.num_obs_per_step :]
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  policy obs shape: {tuple(policy_obs.shape)}")
                    print(f"  mode: {self.current_mode_text()}")
                    print(f"  latest proprio first 9: {latest_obs[:9]}")
                    print(f"  depth: shape={tuple(depth_flat.shape)}, mean={depth_flat.mean().item():.3f}")
                    print(f"  command: {self.command_vel}")
                    print(f"  action first 6 (Isaac order): {self.action[:6]}")

                # Recovery targets are relative to q at the beginning of this
                # control step. Keep this target fixed for all decimation steps.
                target_pos = self.position_control()
                maximum_torso_force = 0.0
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.perf_counter()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:] = 0.0
                    self.data.ctrl[self.ctrl_addr_mj] = tau
                    mujoco.mj_step(self.model, self.data)
                    maximum_torso_force = max(
                        maximum_torso_force,
                        self._torso_contact_force(),
                    )
                    if self.viewer is not None:
                        self.viewer.sync()

                    if self.cfg.sim.realtime:
                        remaining = self.cfg.sim.dt - (time.perf_counter() - step_start)
                        if remaining > 0.0:
                            time.sleep(remaining)

                self.episode_length_buf += 1
                self.calculate_gait_para()
                self._update_recovery_state(maximum_torso_force)

                if self.episode_length_buf > 0 and self.episode_length_buf % 100 == 0:
                    torso_height = self._torso_height_above_terrain()
                    height_text = "invalid" if torso_height is None else f"{torso_height:.3f}"
                    print(
                        f"[INFO] t={self.data.time - run_start_time:.1f}s, "
                        f"mode={self.current_mode_text()}, "
                        f"cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, "
                        f"{self.command_vel[2]:.2f}], "
                        f"torso_h={height_text}m, torso_force={self.last_torso_contact_force:.2f}N"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self.close()
            print("[INFO] Simulation finished.")


def build_argument_parser(
    default_policy: str,
    default_model: str,
    available_scenes: dict[str, str],
) -> argparse.ArgumentParser:
    scene_names = sorted(available_scenes)
    parser = argparse.ArgumentParser(
        description=(
            "G1 new RENet 23DOF Sim2Sim. Observation: proprio history | "
            "depth history | actor_mode | recovery_beta."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", default=default_policy, help="Exported g1_renet TorchScript policy.pt")
    parser.add_argument("--model", default=default_model, help="Robot MJCF used when --scene is omitted")
    parser.add_argument(
        "--scene",
        choices=scene_names if scene_names else None,
        help=f"Bundled scene name; available: {', '.join(scene_names) if scene_names else 'none'}",
    )
    parser.add_argument("--scene-file", help="Custom scene XML; takes precedence over --scene and --model")
    parser.add_argument("--duration", type=float, default=100.0, help="Run duration after stabilization (seconds)")
    parser.add_argument("--stabilization", type=float, default=2.0, help="Initial fixed-base stabilization (seconds)")
    parser.add_argument("--estimator", choices=("vp", "op"), default="vp", help="Initial locomotion estimator")
    parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward command without joystick")
    parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral command without joystick")
    parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw-rate command without joystick")
    parser.add_argument("--recovery-beta", type=float, default=0.25, help="Recovery delta-position action scale")
    parser.add_argument("--disable-recovery", action="store_true", help="Disable automatic/manual Recovery routing")
    parser.add_argument("--no-joystick", action="store_true", help="Use fixed velocity commands only")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true", help="Run physics as fast as possible")
    parser.add_argument("--no-depth-view", action="store_true", help="Do not open the matplotlib depth window")
    parser.add_argument("--debug-depth", action="store_true", help="Print depth min/max statistics")
    parser.add_argument("--no-auto-camera", action="store_true", help="Do not patch missing depth_camera into MJCF")
    parser.add_argument("--check-mapping", action="store_true", help="Validate model, policy, and mapping, then exit")
    parser.add_argument("--list-scenes", action="store_true", help="List bundled scenes and exit")
    return parser


def main() -> None:
    legged_lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    mjcf_dir = os.path.join(legged_lab_root, "legged_lab", "assets", "g1", "g1_23dof")
    available_scenes = get_available_scenes(mjcf_dir)
    default_policy = find_latest_exported_policy(os.path.join(legged_lab_root, "logs", "g1_renet"))
    default_model = os.path.join(mjcf_dir, "g1_23dof_rev_1_0.xml")
    parser = build_argument_parser(default_policy, default_model, available_scenes)
    args = parser.parse_args()

    if args.list_scenes:
        print("Available scenes:")
        for name in sorted(available_scenes):
            print(f"  {name:12s} -> {available_scenes[name]}")
        return

    if args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.stabilization < 0.0:
        parser.error("--stabilization cannot be negative.")
    if args.recovery_beta <= 0.0:
        parser.error("--recovery-beta must be positive.")

    if args.scene_file is not None:
        model_path = os.path.abspath(os.path.expanduser(args.scene_file))
    elif args.scene is not None:
        model_path = available_scenes[args.scene]
    else:
        model_path = os.path.abspath(os.path.expanduser(args.model))
    policy_path = os.path.abspath(os.path.expanduser(args.policy))

    if not os.path.isfile(policy_path):
        parser.error(f"Policy file does not exist: {policy_path}")
    if not os.path.isfile(model_path):
        parser.error(f"MuJoCo XML does not exist: {model_path}")

    cfg = G1NewRENetSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    cfg.sim.stabilization_duration = args.stabilization
    cfg.sim.print_mapping = args.check_mapping
    cfg.sim.headless = args.headless
    cfg.sim.realtime = not args.no_realtime
    cfg.renet.default_estimator = args.estimator
    cfg.recovery.enable = not args.disable_recovery
    cfg.recovery.beta = args.recovery_beta
    cfg.input.enable_joystick = not args.no_joystick
    cfg.robot.show_depth = not args.no_depth_view and not args.headless
    cfg.robot.debug_depth = args.debug_depth
    cfg.robot.auto_patch_camera = not args.no_auto_camera

    print("=" * 72)
    print("G1 new RENet 23DOF Sim2Sim")
    print("=" * 72)
    print(f"Policy: {policy_path}")
    print(f"MuJoCo XML: {model_path}")
    print(f"Observation: [1, {EXPECTED_POLICY_OBS_DIM}], action: [1, {EXPECTED_ACTION_DIM}]")
    print(f"Initial estimator: {cfg.renet.default_estimator.upper()}")
    print(
        f"Recovery: enabled={cfg.recovery.enable}, beta={cfg.recovery.beta:.2f}, "
        f"timeout={cfg.recovery.max_duration_s:.1f}s"
    )
    print("=" * 72)

    runner = G1NewRENetMujocoRunner(
        cfg=cfg,
        policy_path=policy_path,
        model_path=model_path,
    )
    runner.command_vel[:] = np.array(
        [
            np.clip(args.vx, *cfg.command.lin_vel_x),
            np.clip(args.vy, *cfg.command.lin_vel_y),
            np.clip(args.wz, *cfg.command.ang_vel_z),
        ],
        dtype=np.float32,
    )
    runner.validate_deployment_policy()

    if args.check_mapping:
        runner.close()
        print("[INFO] Mapping/policy check completed; simulation was not started.")
        return
    runner.run()


if __name__ == "__main__":
    main()
