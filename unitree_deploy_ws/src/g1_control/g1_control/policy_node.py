"""Run the verified dual-branch RENet model at 50 Hz and publish joint targets."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from g1_policy_msgs.msg import (
    DepthPolicyInput,
    ElevationPolicyInput,
    PolicyJointTarget,
    ProprioPolicyInput,
)

from .contracts import (
    ACTION_SCALE,
    CLIP_ACTIONS,
    DEFAULT_Q_POLICY,
    DEPTH_HISTORY_DIM,
    ELEVATION_DIM,
    JOINT_COUNT,
    POLICY_INPUT_DIM,
)
from .observation import ProprioHistory, build_proprio_frame, projected_gravity_wxyz
from .remote import decode_remote, map_axis_to_command_range, rising_buttons
from .tensorrt_runner import TensorRTPolicyRunner
from .transition import ActionTransition, EstimatorMode


def _deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class G1RENetPolicy(Node):
    """The policy process does inference only; it never publishes Unitree LowCmd."""

    def __init__(self) -> None:
        super().__init__("g1_renet_policy")
        self.declare_parameter(
            "policy_path",
            "/home/unitree/cjh/models/renet_dual_branch.pt",
        )
        self.declare_parameter(
            "expected_policy_sha256",
            "c512197e3c5e9951c3f183d5be4317777a29bba33be3535370b02d8282b2fb4e",
        )
        self.declare_parameter("inference_backend", "tensorrt")
        self.declare_parameter(
            "policy_engine_path",
            "/home/unitree/cjh/models/renet_policy_fp32_notf32.engine",
        )
        self.declare_parameter("expected_policy_engine_sha256", "")
        self.declare_parameter("proprio_topic", "/g1/proprio/policy_input")
        self.declare_parameter("depth_topic", "/g1/depth/policy_input")
        self.declare_parameter("elevation_topic", "/g1/elevation/policy_input")
        self.declare_parameter("target_topic", "/g1/policy/joint_target")
        self.declare_parameter("event_topic", "/g1/control/event")
        self.declare_parameter("torch_num_threads", 1)
        self.declare_parameter("initial_estimator", "vp")
        self.declare_parameter("remote_commands_enabled", False)
        self.declare_parameter("remote_mode_switch_enabled", False)
        self.declare_parameter("command_deadzone", 0.10)
        self.declare_parameter("command_x_range", [-0.4, 0.8])
        self.declare_parameter("command_y_range", [-0.4, 0.4])
        self.declare_parameter("command_yaw_range", [-1.57, 1.57])
        self.declare_parameter("command_x_axis_sign", 1.0)
        self.declare_parameter("command_y_axis_sign", -1.0)
        self.declare_parameter("command_yaw_axis_sign", -1.0)
        self.declare_parameter("transition_steps", 5)
        self.declare_parameter("report_every", 50)

        self._command_ranges = tuple(
            tuple(float(bound) for bound in self.get_parameter(name).value)
            for name in ("command_x_range", "command_y_range", "command_yaw_range")
        )
        self._command_axis_signs = tuple(
            float(self.get_parameter(name).value)
            for name in (
                "command_x_axis_sign",
                "command_y_axis_sign",
                "command_yaw_axis_sign",
            )
        )
        for command_range, axis_sign in zip(
            self._command_ranges, self._command_axis_signs
        ):
            map_axis_to_command_range(0.0, command_range, axis_sign=axis_sign)

        self._backend = str(
            self.get_parameter("inference_backend").value
        ).lower()
        self._model = None
        self._trt_runner = None
        if self._backend == "torchscript":
            import torch

            torch.set_num_threads(
                int(self.get_parameter("torch_num_threads").value)
            )
            path = Path(
                str(self.get_parameter("policy_path").value)
            ).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"dual-branch policy not found: {path}"
                )
            expected_digest = str(
                self.get_parameter("expected_policy_sha256").value
            )
            actual_digest = _sha256(path)
            if expected_digest and actual_digest != expected_digest:
                raise RuntimeError(
                    "policy SHA-256 mismatch: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            extra_files = {"renet_dual_branch_contract.json": ""}
            self._model = torch.jit.load(
                str(path), map_location="cpu", _extra_files=extra_files
            ).eval()
            contract = json.loads(
                extra_files["renet_dual_branch_contract.json"]
            )
            if (
                contract.get("artifact") != "g1_renet_dual_branch"
                or contract.get("version") != 2
            ):
                raise RuntimeError(
                    f"unexpected embedded policy contract: {contract}"
                )
            artifact_path = path
        elif self._backend == "tensorrt":
            engine_path = Path(
                str(self.get_parameter("policy_engine_path").value)
            ).expanduser()
            if not engine_path.is_file():
                raise FileNotFoundError(
                    f"TensorRT policy engine not found: {engine_path}"
                )
            expected_digest = str(
                self.get_parameter("expected_policy_engine_sha256").value
            )
            actual_digest = _sha256(engine_path)
            if expected_digest and actual_digest != expected_digest:
                raise RuntimeError(
                    "policy engine SHA-256 mismatch: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            self._trt_runner = TensorRTPolicyRunner(
                engine_path, POLICY_INPUT_DIM, JOINT_COUNT
            )
            artifact_path = engine_path
        else:
            raise ValueError(
                "inference_backend must be 'tensorrt' or 'torchscript'"
            )

        initial = str(self.get_parameter("initial_estimator").value).lower()
        if initial not in ("vp", "lp"):
            raise ValueError("initial_estimator must be 'vp' or 'lp'")
        self._mode = EstimatorMode.VP if initial == "vp" else EstimatorMode.LP
        self._history = ProprioHistory()
        self._transition = ActionTransition(
            blend_steps=int(self.get_parameter("transition_steps").value)
        )
        self._previous_action = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._active = False
        self._activation_transition_pending = False
        self._proprio = None
        self._depth = None
        self._elevation = None
        self._sequence = 0
        self._previous_remote_keys = 0

        proprio_topic = str(self.get_parameter("proprio_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        elevation_topic = str(self.get_parameter("elevation_topic").value)
        target_topic = str(self.get_parameter("target_topic").value)
        event_topic = str(self.get_parameter("event_topic").value)
        sensor_qos = copy.copy(qos_profile_sensor_data)
        sensor_qos.depth = 1
        self.create_subscription(
            ProprioPolicyInput,
            proprio_topic,
            self._on_proprio,
            sensor_qos,
        )
        self.create_subscription(
            DepthPolicyInput, depth_topic, self._on_depth, sensor_qos
        )
        self.create_subscription(
            ElevationPolicyInput,
            elevation_topic,
            self._on_elevation,
            sensor_qos,
        )
        self._publisher = self.create_publisher(PolicyJointTarget, target_topic, 10)
        self._event_publisher = self.create_publisher(String, event_topic, 10)
        self._event_subscription = self.create_subscription(
            String, event_topic, self._on_control_event, 10
        )
        self.get_logger().info(
            "RENet policy ready: "
            f"backend={self._backend}, model={artifact_path}, "
            f"proprio={proprio_topic} (snapshot-triggered), "
            f"initial={self._mode.name}, "
            "activation=policy_requested event, "
            f"remote_commands={self.get_parameter('remote_commands_enabled').value}, "
            f"remote_switch={self.get_parameter('remote_mode_switch_enabled').value}"
        )

    def _publish_event(self, event: str) -> None:
        message = String()
        message.data = event
        self._event_publisher.publish(message)

    def _reset_runtime_state(self) -> None:
        """Start an execution epoch without any pre-activation policy state."""

        self._history.reset()
        self._transition.reset()
        # The bridge holds DEFAULT_Q_POLICY before policy activation. Anchor
        # the first five policy targets at the action that is really being
        # executed, rather than at an output inferred before A was pressed.
        self._transition.set_executed(
            np.zeros(JOINT_COUNT, dtype=np.float32)
        )
        self._previous_action.fill(0.0)
        self._sequence = 0
        self._activation_transition_pending = True
        if self._proprio is not None:
            remote = decode_remote(self._proprio.wireless_remote)
            self._previous_remote_keys = remote.keys

    def _on_control_event(self, message: String) -> None:
        if message.data == "policy_requested":
            self._reset_runtime_state()
            self._active = True
            self.get_logger().info(
                "fresh policy epoch requested: cleared proprio history and "
                "previous_action; waiting for the next 50 Hz snapshot"
            )
        elif message.data == "policy_exit":
            self._active = False
            self._history.reset()
            self._transition.reset()
            self._previous_action.fill(0.0)
            self._activation_transition_pending = False
            self.get_logger().info(
                "policy epoch stopped after SELECT; inference is idle"
            )

    def _on_proprio(self, message: ProprioPolicyInput) -> None:
        self._proprio = message
        if not self._active:
            # Track the current key state while idle so a held L1/R1 cannot be
            # mistaken for a new mode request when a later epoch starts.
            self._previous_remote_keys = decode_remote(
                message.wireless_remote
            ).keys
            return
        self._step()

    def _on_depth(self, message: DepthPolicyInput) -> None:
        self._depth = np.asarray(message.history, dtype=np.float32)

    def _on_elevation(self, message: ElevationPolicyInput) -> None:
        self._elevation = np.asarray(message.height, dtype=np.float32)

    def _remote_command_and_switch(self) -> tuple[np.ndarray, bool]:
        remote = decode_remote(self._proprio.wireless_remote)
        rising = rising_buttons(self._previous_remote_keys, remote.keys)
        self._previous_remote_keys = remote.keys
        switched = False
        if bool(self.get_parameter("remote_mode_switch_enabled").value):
            requested = self._mode
            event = None
            if "L1" in rising and "R1" in rising:
                self.get_logger().warning(
                    "L1 and R1 rose together; ignoring ambiguous estimator request"
                )
            elif "L1" in rising:
                requested = EstimatorMode.LP
                event = "lidar_mode"
            elif "R1" in rising:
                requested = EstimatorMode.VP
                event = "visual_mode"
            if requested != self._mode:
                old = self._mode
                self._mode = requested
                switched = True
                self.get_logger().info(f"manual estimator switch: {old.name} -> {self._mode.name}")
            elif event is not None:
                self.get_logger().info(
                    f"manual estimator request keeps {self._mode.name} active"
                )
            if event is not None:
                self._publish_event(event)

        command = np.zeros(3, dtype=np.float32)
        if bool(self.get_parameter("remote_commands_enabled").value):
            deadzone = float(self.get_parameter("command_deadzone").value)
            for index, raw_axis in enumerate((remote.ly, remote.lx, remote.rx)):
                command[index] = map_axis_to_command_range(
                    _deadzone(raw_axis, deadzone),
                    self._command_ranges[index],
                    axis_sign=self._command_axis_signs[index],
                )
        return command, switched

    def _step(self) -> None:
        if self._proprio is None or self._depth is None or self._elevation is None:
            return
        if self._depth.shape != (DEPTH_HISTORY_DIM,):
            raise RuntimeError(f"depth input must have {DEPTH_HISTORY_DIM} values")
        if self._elevation.shape != (ELEVATION_DIM,):
            raise RuntimeError(f"elevation input must have {ELEVATION_DIM} values")

        q_policy = np.asarray(self._proprio.joint_position, dtype=np.float32)
        dq_policy = np.asarray(self._proprio.joint_velocity, dtype=np.float32)
        command, switched = self._remote_command_and_switch()
        frame = build_proprio_frame(
            np.asarray(self._proprio.angular_velocity, dtype=np.float32),
            projected_gravity_wxyz(self._proprio.quaternion_wxyz),
            command,
            q_policy,
            dq_policy,
            self._previous_action,
        )
        history = self._history.append(frame)
        observation = np.concatenate(
            (history, self._elevation, self._depth, np.zeros(1, dtype=np.float32))
        ).astype(np.float32, copy=False)
        if observation.shape != (POLICY_INPUT_DIM,):
            raise RuntimeError(f"policy observation has unexpected shape {observation.shape}")
        inference_start = time.perf_counter()
        if self._backend == "tensorrt":
            lp_action, vp_action = self._trt_runner.infer_dual(observation)
        else:
            import torch

            with torch.inference_mode():
                outputs = self._model(
                    torch.from_numpy(observation).unsqueeze(0)
                )
            if len(outputs) != 6:
                raise RuntimeError(
                    f"dual model must return six tensors, got {len(outputs)}"
                )
            lp_action = (
                outputs[4]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            vp_action = (
                outputs[5]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        if lp_action.shape != (JOINT_COUNT,) or vp_action.shape != (JOINT_COUNT,):
            raise RuntimeError(
                f"dual action shapes must both be {(JOINT_COUNT,)}, "
                f"got LP={lp_action.shape}, VP={vp_action.shape}"
            )
        selected = lp_action if self._mode == EstimatorMode.LP else vp_action
        selected = np.clip(selected, -CLIP_ACTIONS, CLIP_ACTIONS)
        action, progress = self._transition.step(
            selected, switched or self._activation_transition_pending
        )
        self._activation_transition_pending = False
        q_target = DEFAULT_Q_POLICY + ACTION_SCALE * action
        self._transition.set_executed(action)
        self._previous_action = action.astype(np.float32, copy=True)

        message = PolicyJointTarget()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "g1_renet_policy_joint_order"
        message.sequence = self._sequence
        message.estimator_mode = int(self._mode)
        message.transition_progress = float(progress)
        message.action = action.tolist()
        message.q_target = q_target.tolist()
        self._publisher.publish(message)
        self._sequence += 1
        report_every = int(self.get_parameter("report_every").value)
        if report_every > 0 and self._sequence % report_every == 0:
            self.get_logger().info(
                f"policy frame #{self._sequence}: mode={self._mode.name}, "
                f"command={command.tolist()}, transition={progress:.2f}, "
                f"action_range=[{float(action.min()):.3f}, "
                f"{float(action.max()):.3f}], "
                f"proprio_sequence={self._proprio.sequence}, "
                f"inference={inference_ms:.2f}ms"
            )

    def destroy_node(self):
        if self._trt_runner is not None:
            self._trt_runner.close()
            self._trt_runner = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = G1RENetPolicy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
