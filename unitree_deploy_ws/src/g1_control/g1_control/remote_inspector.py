"""Read and print the real G1 remote payload; this node never publishes control."""

from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from unitree_hg.msg import LowState

from .remote import decode_remote, rising_buttons


class RemoteInspector(Node):
    def __init__(self) -> None:
        super().__init__("g1_remote_inspector")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("report_rate", 10.0)
        self._report_period = 1.0 / float(self.get_parameter("report_rate").value)
        self._last_report = 0.0
        self._previous_keys = 0
        topic = str(self.get_parameter("lowstate_topic").value)
        self.create_subscription(LowState, topic, self._on_lowstate, 10)
        self.get_logger().info(
            f"read-only G1 remote inspector: input={topic}, "
            f"report_rate={1.0 / self._report_period:.1f}Hz; no publisher is created"
        )

    def _on_lowstate(self, message: LowState) -> None:
        remote = decode_remote(message.wireless_remote)
        rising = rising_buttons(self._previous_keys, remote.keys)
        self._previous_keys = remote.keys
        now = time.monotonic()
        if rising:
            self.get_logger().info(
                f"button rising={','.join(rising)} keys=0x{remote.keys:04x}"
            )
        if now - self._last_report >= self._report_period:
            self._last_report = now
            pressed = ",".join(remote.pressed_names) if remote.pressed_names else "none"
            self.get_logger().info(
                "axes raw: "
                f"lx={remote.lx:+.4f} ly={remote.ly:+.4f} "
                f"rx={remote.rx:+.4f} ry={remote.ry:+.4f}; "
                f"pressed={pressed}; keys=0x{remote.keys:04x}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteInspector()
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
