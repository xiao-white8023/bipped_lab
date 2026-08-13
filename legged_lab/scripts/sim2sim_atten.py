"""MuJoCo sim-to-sim runner for the ``g1_atten`` task.

The proprioceptive state and 23-DOF control path intentionally reuse the
well-tested ``sim2sim_rough.py`` implementation.  This runner adds the two
observations that are specific to ``AttenEnv``:

* body-frame root linear velocity and two foot-contact flags;
* a yaw-aligned, forward-only 3-D height-scan map with shape ``(3, 11, 13)``.

The exported attention policy therefore receives 83 proprioceptive values
followed by 429 flattened height-scan values (512 values in total).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sim2sim_rough import (  # noqa: E402
    G1RoughMujocoRunner,
    G1RoughSim2SimCfg,
    find_latest_exported_policy,
    get_available_scenes,
    prepare_mjcf_for_custom_scene,
)


class G1AttenSim2SimCfg(G1RoughSim2SimCfg):
    """Deployment settings matching ``AttenCFG`` and ``AttenAGENTENV``."""

    class sim(G1RoughSim2SimCfg.sim):
        num_actions = 23
        num_obs_per_step = 83
        actor_obs_history_length = 1

    class height_scanner:
        resolution = 0.1
        size = (2.4, 1.0)
        history_frames = 1
        channels = 3
        ray_start_height = 20.0
        no_hit_depth = 5.0

        # AttenEnv reshapes the full 11 x 25 grid and retains x >= 0,
        # producing a forward scan of shape 11 x 13.
        rows = int(round(size[1] / resolution)) + 1
        full_cols = int(round(size[0] / resolution)) + 1
        front_col_start = full_cols // 2
        front_cols = full_cols - front_col_start

    class contact:
        history_length = 3
        force_threshold = 0.5


class G1AttenMujocoRunner(G1RoughMujocoRunner):
    """Run the G1 attention policy in MuJoCo."""

    def __init__(self, cfg: G1AttenSim2SimCfg, policy_path: str, model_path: str):
        super().__init__(cfg=cfg, policy_path=policy_path, model_path=model_path)
        self.init_foot_contacts()
        self.init_height_scanner()
        self.validate_policy_interface()

        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        map_dim = (
            self.cfg.height_scanner.history_frames
            * self.cfg.height_scanner.channels
            * self.cfg.height_scanner.rows
            * self.cfg.height_scanner.front_cols
        )
        print(
            "[INFO] Attention policy input: "
            f"proprio={proprio_dim}, height_scan={map_dim} "
            f"({self.cfg.height_scanner.channels}x{self.cfg.height_scanner.rows}x"
            f"{self.cfg.height_scanner.front_cols}), total={proprio_dim + map_dim}"
        )

    def init_foot_contacts(self) -> None:
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        self.foot_body_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_names],
            dtype=np.int32,
        )
        if np.any(self.foot_body_ids < 0):
            missing = [name for name, body_id in zip(foot_names, self.foot_body_ids) if body_id < 0]
            raise RuntimeError(f"MuJoCo model is missing foot bodies required by g1_atten: {missing}")

        self.foot_contact_history = np.zeros(
            (self.cfg.contact.history_length, len(foot_names)), dtype=np.bool_
        )
        self._contact_force = np.zeros(6, dtype=np.float64)
        self._update_foot_contact_history()

    def init_height_scanner(self) -> None:
        scanner_cfg = self.cfg.height_scanner
        full_x = np.linspace(
            -scanner_cfg.size[0] / 2.0,
            scanner_cfg.size[0] / 2.0,
            scanner_cfg.full_cols,
            dtype=np.float64,
        )
        grid_y = np.linspace(
            -scanner_cfg.size[1] / 2.0,
            scanner_cfg.size[1] / 2.0,
            scanner_cfg.rows,
            dtype=np.float64,
        )
        grid_x, grid_y = np.meshgrid(full_x, grid_y, indexing="xy")
        grid_x = grid_x[:, scanner_cfg.front_col_start :]
        grid_y = grid_y[:, scanner_cfg.front_col_start :]
        self.height_scan_offsets = np.stack(
            [grid_x, grid_y, np.zeros_like(grid_x)], axis=-1
        )

        expected_shape = (scanner_cfg.rows, scanner_cfg.front_cols, 3)
        if self.height_scan_offsets.shape != expected_shape:
            raise RuntimeError(
                f"Height-scan grid shape mismatch: expected {expected_shape}, "
                f"got {self.height_scan_offsets.shape}."
            )

        # Isaac Lab ray-casts only against /World/ground.  During each scan the
        # MuJoCo geoms attached to the world body are moved temporarily into an
        # otherwise-unused group, so robot geoms cannot occlude the scan.  The
        # original groups are restored immediately to keep the terrain visible
        # in the passive viewer.
        self.terrain_geom_ids = np.flatnonzero(self.model.geom_bodyid == 0).astype(np.int32)
        if self.terrain_geom_ids.size == 0:
            raise RuntimeError("MuJoCo model has no world-body geoms for the height scanner.")
        robot_geom_groups = set(
            int(group)
            for group in self.model.geom_group[self.model.geom_bodyid != 0]
        )
        free_groups = [group for group in range(5, -1, -1) if group not in robot_geom_groups]
        if not free_groups:
            raise RuntimeError("No free MuJoCo geom group is available for terrain-only ray casting.")
        self.ray_geom_group_index = free_groups[0]
        self.ray_geom_group = np.zeros(6, dtype=np.uint8)
        self.ray_geom_group[self.ray_geom_group_index] = 1
        self._ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._ray_geom_id = np.full(1, -1, dtype=np.int32)

        terrain_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            or f"geom_{geom_id}"
            for geom_id in self.terrain_geom_ids
        ]
        print(
            f"[INFO] Height scanner uses {len(terrain_names)} world geom(s): "
            + ", ".join(terrain_names[:8])
            + (" ..." if len(terrain_names) > 8 else "")
        )

    def validate_policy_interface(self) -> None:
        expected_input_dim = self.policy_input_dim
        with torch.inference_mode():
            try:
                output = self.policy(torch.zeros(1, expected_input_dim, dtype=torch.float32))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"The exported policy does not accept the g1_atten input dimension "
                    f"({expected_input_dim}). Check that --policy points to logs/g1_atten/.../exported/policy.pt."
                ) from exc
        if output.ndim != 2 or output.shape != (1, self.num_actions):
            raise RuntimeError(
                f"Expected policy output shape (1, {self.num_actions}), got {tuple(output.shape)}."
            )

    @property
    def policy_input_dim(self) -> int:
        proprio_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        map_dim = (
            self.cfg.height_scanner.history_frames
            * self.cfg.height_scanner.channels
            * self.cfg.height_scanner.rows
            * self.cfg.height_scanner.front_cols
        )
        return proprio_dim + map_dim

    @staticmethod
    def quat_rotate_inverse(quat_wxyz: np.ndarray, vectors_w: np.ndarray) -> np.ndarray:
        """Rotate one or more world-frame vectors into the quaternion body frame."""

        quat = np.asarray(quat_wxyz, dtype=np.float64)
        vectors = np.asarray(vectors_w, dtype=np.float64)
        quat_w = quat[0]
        quat_vec = quat[1:4]
        dot = np.sum(vectors * quat_vec, axis=-1, keepdims=True)
        return (
            vectors * (2.0 * quat_w * quat_w - 1.0)
            - 2.0 * quat_w * np.cross(quat_vec, vectors)
            + 2.0 * dot * quat_vec
        )

    @staticmethod
    def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
        quat_w, quat_x, quat_y, quat_z = quat_wxyz
        return float(
            np.arctan2(
                2.0 * (quat_w * quat_z + quat_x * quat_y),
                1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z),
            )
        )

    def _sample_current_foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(2, dtype=np.bool_)
        foot_index_by_body = {int(body_id): idx for idx, body_id in enumerate(self.foot_body_ids)}

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1]) if geom1 >= 0 else -1
            body2 = int(self.model.geom_bodyid[geom2]) if geom2 >= 0 else -1

            foot_indices = []
            if body1 in foot_index_by_body:
                foot_indices.append(foot_index_by_body[body1])
            if body2 in foot_index_by_body:
                foot_indices.append(foot_index_by_body[body2])
            if not foot_indices:
                continue

            self._contact_force.fill(0.0)
            mujoco.mj_contactForce(self.model, self.data, contact_id, self._contact_force)
            if np.linalg.norm(self._contact_force[:3]) > self.cfg.contact.force_threshold:
                contacts[foot_indices] = True

        return contacts

    def _update_foot_contact_history(self) -> None:
        self.foot_contact_history = np.roll(self.foot_contact_history, shift=-1, axis=0)
        self.foot_contact_history[-1] = self._sample_current_foot_contacts()

    def get_foot_contacts(self) -> np.ndarray:
        # Include the immediate contact state as well.  This is important after
        # stabilization/reset, before the first controlled physics step.
        return np.logical_or(
            np.any(self.foot_contact_history, axis=0),
            self._sample_current_foot_contacts(),
        ).astype(np.float32)

    def get_proprio_obs(self) -> np.ndarray:
        dof_pos_mj = self.data.qpos[self.qpos_addr_mj].copy()
        dof_vel_mj = self.data.qvel[self.qvel_addr_mj].copy()
        root_quat = self.data.qpos[3:7].copy()

        ang_vel_body = self.data.qvel[3:6].copy()
        projected_gravity = self.get_gravity_orientation(root_quat)
        root_lin_vel_body = self.quat_rotate_inverse(root_quat, self.data.qvel[0:3]).astype(np.float32)
        feet_contact = self.get_foot_contacts()
        joint_pos_lab = self.mj_to_lab(dof_pos_mj - self.default_dof_pos)
        joint_vel_lab = self.mj_to_lab(dof_vel_mj)

        obs = np.concatenate(
            [
                ang_vel_body,
                projected_gravity,
                root_lin_vel_body,
                self.command_vel,
                feet_contact,
                joint_pos_lab,
                joint_vel_lab,
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.cfg.sim.num_obs_per_step:
            raise RuntimeError(
                f"Expected g1_atten proprio dim {self.cfg.sim.num_obs_per_step}, got {obs.shape[0]}."
            )

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(
            self.obs_history,
            -self.cfg.sim.clip_observations,
            self.cfg.sim.clip_observations,
        )

    def get_height_scan_map(self) -> np.ndarray:
        """Return the AttenEnv-compatible forward xyz map in CHW order."""

        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        yaw = self.yaw_from_quat(root_quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        offsets = self.height_scan_offsets.reshape(-1, 3)
        ray_starts = np.empty_like(offsets, dtype=np.float64)
        ray_starts[:, 0] = root_pos[0] + cos_yaw * offsets[:, 0] - sin_yaw * offsets[:, 1]
        ray_starts[:, 1] = root_pos[1] + sin_yaw * offsets[:, 0] + cos_yaw * offsets[:, 1]
        ray_starts[:, 2] = root_pos[2] + self.cfg.height_scanner.ray_start_height

        ray_hits = ray_starts.copy()
        original_terrain_groups = self.model.geom_group[self.terrain_geom_ids].copy()
        self.model.geom_group[self.terrain_geom_ids] = self.ray_geom_group_index
        try:
            for ray_id, ray_start in enumerate(ray_starts):
                self._ray_geom_id[0] = -1
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    ray_start,
                    self._ray_direction,
                    self.ray_geom_group,
                    1,
                    -1,
                    self._ray_geom_id,
                )
                if distance >= 0.0 and np.isfinite(distance):
                    ray_hits[ray_id] = ray_start + distance * self._ray_direction
                else:
                    # This matches AttenEnv's invalid-ray fallback: preserve
                    # x/y and place z five metres below the root.
                    ray_hits[ray_id, 2] = root_pos[2] - self.cfg.height_scanner.no_hit_depth
        finally:
            self.model.geom_group[self.terrain_geom_ids] = original_terrain_groups

        points_body = self.quat_rotate_inverse(root_quat, ray_hits - root_pos)
        points_body = points_body.reshape(
            self.cfg.height_scanner.rows,
            self.cfg.height_scanner.front_cols,
            self.cfg.height_scanner.channels,
        )
        height_scan_chw = points_body.transpose(2, 0, 1).astype(np.float32)
        return height_scan_chw.reshape(-1)

    def get_obs(self) -> np.ndarray:
        proprio_obs = self.get_proprio_obs()
        height_scan = self.get_height_scan_map()
        obs = np.concatenate([proprio_obs, height_scan], axis=0).astype(np.float32)
        if obs.shape[0] != self.policy_input_dim:
            raise RuntimeError(f"Expected policy input dim {self.policy_input_dim}, got {obs.shape[0]}.")
        return obs

    def set_initial_pose(self) -> None:
        super().set_initial_pose()
        if hasattr(self, "foot_contact_history"):
            self.foot_contact_history[:] = False
            self._update_foot_contact_history()

    def run(self) -> None:
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.stabilize_robot()

        print(
            "\n[INFO] Xbox: left stick = vx/vy, right stick X = yaw rate, "
            "A = stop, START = reset, SELECT = exit"
        )
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
                    proprio = obs[: self.cfg.sim.num_obs_per_step]
                    scan = obs[self.cfg.sim.num_obs_per_step :].reshape(
                        self.cfg.height_scanner.channels,
                        self.cfg.height_scanner.rows,
                        self.cfg.height_scanner.front_cols,
                    )
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  obs shape: {obs_tensor.shape}")
                    print(f"  proprio first 14: {proprio[:14]}")
                    print(f"  command: {self.command_vel}")
                    print(f"  foot contacts: {proprio[12:14]}")
                    print(
                        f"  scan xyz min: {scan.reshape(3, -1).min(axis=1)}, "
                        f"max: {scan.reshape(3, -1).max(axis=1)}"
                    )
                    print(f"  action first 6 (Isaac order): {self.action[:6]}")

                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.data.ctrl[:] = 0.0
                    self.data.ctrl[self.ctrl_addr_mj] = tau
                    mujoco.mj_step(self.model, self.data)
                    self._update_foot_contact_history()
                    self.viewer.sync()

                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0.0:
                        time.sleep(self.cfg.sim.dt - elapsed)

                self.episode_length_buf += 1
                self.calculate_gait_para()

                if self.episode_length_buf % 100 == 0:
                    print(
                        f"[INFO] t={self.data.time:.1f}s, "
                        f"cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, "
                        f"{self.command_vel[2]:.2f}], "
                        f"contacts={self.get_foot_contacts().astype(np.int32).tolist()}, "
                        f"h={self.data.qpos[2]:.3f}m"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self.viewer.close()
            print("[INFO] Simulation finished.")


def main() -> None:
    legged_lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    mjcf_dir = os.path.join(legged_lab_root, "legged_lab/assets/g1/g1_23dof")

    default_policy = find_latest_exported_policy(os.path.join(legged_lab_root, "logs/g1_atten"))
    default_model = os.path.join(mjcf_dir, "g1_23dof_rev_1_0.xml")
    available_scenes = get_available_scenes(mjcf_dir)
    scene_names = list(available_scenes.keys())

    parser = argparse.ArgumentParser(
        description="G1 attention 23DOF Sim2Sim with MuJoCo height scanning and Xbox control.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--policy", type=str, default=default_policy, help="Path to exported policy.pt")
    parser.add_argument(
        "--model", type=str, default=default_model, help="MuJoCo XML path used when --scene is omitted"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=scene_names if scene_names else None,
        help=f"Scene name. Available: {', '.join(scene_names) if scene_names else 'none'}",
    )
    parser.add_argument(
        "--scene-file",
        type=str,
        default=None,
        help="Custom scene XML path, higher priority than --scene",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument("--list-scenes", action="store_true", help="List available scene XML files")
    parser.add_argument(
        "--check-mapping",
        action="store_true",
        help="Validate joints, scan map, and policy I/O, then exit",
    )
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
    print("G1 Attention 23DOF Sim2Sim")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"MuJoCo XML: {model_path}")
    if args.scene:
        print(f"Scene: {args.scene}")
    print("=" * 60)

    cfg = G1AttenSim2SimCfg()
    cfg.sim.sim_duration = args.duration

    load_model_path, temp_mjcf_dir = prepare_mjcf_for_custom_scene(model_path)
    try:
        runner = G1AttenMujocoRunner(cfg=cfg, policy_path=args.policy, model_path=load_model_path)
        if args.check_mapping:
            obs = runner.get_obs()
            print(
                f"[INFO] Mapping/scan/policy check finished: obs={obs.shape}, "
                f"finite={bool(np.isfinite(obs).all())}; viewer was not launched."
            )
            return
        runner.run()
    finally:
        if temp_mjcf_dir is not None:
            temp_mjcf_dir.cleanup()


if __name__ == "__main__":
    main()
