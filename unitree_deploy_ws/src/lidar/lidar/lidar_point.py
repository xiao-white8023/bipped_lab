"""Inspect the G1 MID-360 PointCloud2 stream without commanding the robot."""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


'''
read_xyz_numpy()：把二进制 blob 零拷贝解析成 (N, 3) 数组
'''
def read_xyz_numpy(msg: PointCloud2) -> np.ndarray:
    """Decode x/y/z directly from PointCloud2 on ROS 2 Foxy and newer."""
    '''
    等价于：
        fields = {}
        for field in msg.fields:
            fields[field.name] = field
    '''
    fields = {field.name: field for field in msg.fields}
    missing = {"x", "y", "z"} - fields.keys()
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {sorted(missing)}")

    xyz_fields = [fields[name] for name in ("x", "y", "z")]
    if any(field.datatype != PointField.FLOAT32 or field.count != 1 for field in xyz_fields):
        raise ValueError("x, y, and z must be scalar FLOAT32 fields")

    byte_order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (byte_order + "f4",) * 3,
            "offsets": tuple(field.offset for field in xyz_fields),
            "itemsize": msg.point_step,
        }
    )
    points = np.ndarray(
        shape=(msg.height, msg.width),
        dtype=dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    ).reshape(-1)
    return np.column_stack((points["x"], points["y"], points["z"]))


class G1LidarProbe(Node):
    def __init__(self):
        super().__init__("g1_lidar_probe")
        self.declare_parameter("topic", "/livox/lidar")
        self.declare_parameter("report_every", 10)

        topic = self.get_parameter("topic").value
        self._report_every = int(self.get_parameter("report_every").value)
        if self._report_every < 1:
            raise ValueError("report_every must be at least 1")

        self._count = 0
        self._first_arrival = None
        self._subscription = self.create_subscription(
            PointCloud2, topic, self._on_cloud, qos_profile_sensor_data
        )
        self.get_logger().info(f"Waiting for PointCloud2 on {topic}")

    def _on_cloud(self, msg: PointCloud2):
        now = time.monotonic()
        self._count += 1
        if self._first_arrival is None:
            self._first_arrival = now
            fields = ", ".join(
                f"{field.name}@{field.offset}:type{field.datatype}" for field in msg.fields
            )
            self.get_logger().info(
                f"First cloud: frame={msg.header.frame_id!r}, "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                f"size={msg.width}x{msg.height}, point_step={msg.point_step}, "
                f"fields=[{fields}]"
            )

        if self._count != 1 and self._count % self._report_every != 0:
            return

        names = {field.name for field in msg.fields}
        if not {"x", "y", "z"}.issubset(names):
            self.get_logger().error("PointCloud2 is missing x, y, or z")
            return

        xyz = read_xyz_numpy(msg)
        finite = np.isfinite(xyz).all(axis=1)
        valid_xyz = xyz[finite]
        elapsed = now - self._first_arrival
        rate = (self._count - 1) / elapsed if elapsed > 0 else 0.0
        if valid_xyz.size:
            distance = np.linalg.norm(valid_xyz, axis=1)
            self.get_logger().info(
                f"clouds={self._count}, rate={rate:.1f} Hz, "
                f"finite_xyz={len(valid_xyz)}/{len(xyz)}, "
                f"range={distance.min():.2f}/{np.median(distance):.2f}/"
                f"{distance.max():.2f} m (min/median/max)"
            )
        else:
            self.get_logger().warn(
                f"clouds={self._count}, rate={rate:.1f} Hz, no finite xyz points"
            )


def main(args=None):
    rclpy.init(args=args)
    node = G1LidarProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()