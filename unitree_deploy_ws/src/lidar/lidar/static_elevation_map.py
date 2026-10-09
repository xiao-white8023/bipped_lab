"""
Build a stationary G1 elevation grid from torso-frame MID-360 clouds.

This node is the static validation stage before odometry and rolling-map
integration.  It assumes every packet is expressed in one unchanged
``torso_link`` frame, so the robot must remain stationary while it runs.
"""

from collections import deque
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import (
    Float32,
    Float32MultiArray,
    Header,
    MultiArrayDimension,
    UInt8MultiArray,
)

from .cloud_preprocessor import xyz_view


def cell_count(extent, resolution):
    """Return endpoint-inclusive cell count for an exactly divisible extent."""
    intervals = float(extent) / float(resolution)
    rounded = round(intervals)
    if not np.isclose(intervals, rounded, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            'map extents must be exact multiples of the resolution: '
            f'extent={extent}, resolution={resolution}'
        )
    return int(rounded) + 1


def multi_array_layout(rows, cols):
    """Create the policy row-major [y, x] layout metadata."""
    return [
        MultiArrayDimension(label='rows_y', size=rows, stride=rows * cols),
        MultiArrayDimension(label='cols_x', size=cols, stride=cols),
    ]


class StaticElevationMap(Node):
    """Fuse stationary torso-frame scans and publish the 11x17 policy grid."""

    def __init__(self):
        super().__init__('g1_static_elevation_map')
        self.declare_parameter('input_topic', '/g1/lidar/elevation_input')
        self.declare_parameter('target_frame', 'torso_link')
        self.declare_parameter('history_frames', 8)
        self.declare_parameter('history_seconds', 0.8)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('internal_size_x', 3.2)
        self.declare_parameter('internal_size_y', 3.2)
        self.declare_parameter('output_size_x', 1.6)
        self.declare_parameter('output_size_y', 1.0)
        self.declare_parameter('height_offset', 0.5)
        self.declare_parameter('height_scale', 1.0)
        self.declare_parameter('height_clip_min', -1.0)
        self.declare_parameter('height_clip_max', 1.0)
        self.declare_parameter('invalid_height', -1.1)
        self.declare_parameter('report_every', 10)

        self._input_topic = self.get_parameter('input_topic').value
        self._target_frame = self.get_parameter('target_frame').value
        self._history_frames = int(self.get_parameter('history_frames').value)
        self._history_seconds = float(self.get_parameter('history_seconds').value)
        self._resolution = float(self.get_parameter('resolution').value)
        self._internal_size_x = float(self.get_parameter('internal_size_x').value)
        self._internal_size_y = float(self.get_parameter('internal_size_y').value)
        self._output_size_x = float(self.get_parameter('output_size_x').value)
        self._output_size_y = float(self.get_parameter('output_size_y').value)
        self._height_offset = float(self.get_parameter('height_offset').value)
        self._height_scale = float(self.get_parameter('height_scale').value)
        self._height_clip = (
            float(self.get_parameter('height_clip_min').value),
            float(self.get_parameter('height_clip_max').value),
        )
        self._invalid_height = float(self.get_parameter('invalid_height').value)
        self._report_every = int(self.get_parameter('report_every').value)
        self._validate_configuration()

        self._internal_cols = cell_count(self._internal_size_x, self._resolution)
        self._internal_rows = cell_count(self._internal_size_y, self._resolution)
        self._output_cols = cell_count(self._output_size_x, self._resolution)
        self._output_rows = cell_count(self._output_size_y, self._resolution)
        if (self._output_rows, self._output_cols) != (11, 17):
            raise ValueError(
                'pgtt_full_atten requires an 11x17 output grid, got '
                f'{self._output_rows}x{self._output_cols}'
            )

        self._internal_x = np.linspace(
            -self._internal_size_x / 2.0,
            self._internal_size_x / 2.0,
            self._internal_cols,
            dtype=np.float64,
        )
        self._internal_y = np.linspace(
            -self._internal_size_y / 2.0,
            self._internal_size_y / 2.0,
            self._internal_rows,
            dtype=np.float64,
        )
        self._grid_x, self._grid_y = np.meshgrid(self._internal_x, self._internal_y)

        self._output_col_start = self._aligned_start(
            -self._output_size_x / 2.0, self._internal_x, 'output x'
        )
        self._output_row_start = self._aligned_start(
            -self._output_size_y / 2.0, self._internal_y, 'output y'
        )
        self._output_col_slice = slice(
            self._output_col_start, self._output_col_start + self._output_cols
        )
        self._output_row_slice = slice(
            self._output_row_start, self._output_row_start + self._output_rows
        )

        self._history = deque()
        self._last_stamp = None
        self._count = 0
        self._last_frame_warning = 0.0

        self._height_pub = self.create_publisher(
            Float32MultiArray, '/g1/elevation/policy_heights', 10
        )
        self._valid_pub = self.create_publisher(
            UInt8MultiArray, '/g1/elevation/policy_valid', 10
        )
        self._valid_ratio_pub = self.create_publisher(
            Float32, '/g1/elevation/valid_ratio', 10
        )
        self._internal_cloud_pub = self.create_publisher(
            PointCloud2, '/g1/elevation/internal_cloud', qos_profile_sensor_data
        )
        self._policy_cloud_pub = self.create_publisher(
            PointCloud2, '/g1/elevation/policy_cloud', qos_profile_sensor_data
        )
        self._sub = self.create_subscription(
            PointCloud2,
            self._input_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Static elevation map input={self._input_topic}, frame={self._target_frame}, '
            f'history={self._history_frames} packets/{self._history_seconds:.2f}s, '
            f'internal={self._internal_rows}x{self._internal_cols}, '
            f'policy={self._output_rows}x{self._output_cols}, '
            'order=y_ascending/x_ascending, source=pgtt_full_atten'
        )
        self.get_logger().warning(
            'Static mapper assumes the robot and torso_link remain stationary; '
            'do not use this output while the robot is moving.'
        )

    def _validate_configuration(self):
        if self._history_frames < 1:
            raise ValueError('history_frames must be at least 1')
        if self._history_seconds <= 0.0:
            raise ValueError('history_seconds must be positive')
        if self._resolution <= 0.0:
            raise ValueError('resolution must be positive')
        if min(
            self._internal_size_x,
            self._internal_size_y,
            self._output_size_x,
            self._output_size_y,
        ) <= 0.0:
            raise ValueError('map sizes must be positive')
        if self._output_size_x > self._internal_size_x:
            raise ValueError('output_size_x must fit inside internal_size_x')
        if self._output_size_y > self._internal_size_y:
            raise ValueError('output_size_y must fit inside internal_size_y')
        if self._height_clip[0] >= self._height_clip[1]:
            raise ValueError('height clipping bounds must be increasing')
        if self._height_clip[0] <= self._invalid_height <= self._height_clip[1]:
            raise ValueError('invalid_height must lie outside the valid clipping interval')
        if self._report_every < 1:
            raise ValueError('report_every must be at least 1')

    def _aligned_start(self, first_value, internal_coordinates, name):
        index = int(round((first_value - internal_coordinates[0]) / self._resolution))
        if index < 0 or index >= len(internal_coordinates):
            raise ValueError(f'{name} lies outside the internal map')
        if not np.isclose(
            internal_coordinates[index], first_value, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f'{name} is not aligned to the internal map resolution')
        return index

    @staticmethod
    def _stamp_seconds(stamp):
        value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        return value if value > 0.0 else time.monotonic()

    def _rasterize(self, xyz):
        height = np.full(
            (self._internal_rows, self._internal_cols), -np.inf, dtype=np.float64
        )
        finite = np.isfinite(xyz).all(axis=1)
        points = xyz[finite]
        if not len(points):
            return height, np.zeros_like(height, dtype=bool)

        cols = np.rint(
            (points[:, 0] + self._internal_size_x / 2.0) / self._resolution
        ).astype(np.int64)
        rows = np.rint(
            (points[:, 1] + self._internal_size_y / 2.0) / self._resolution
        ).astype(np.int64)
        inside = (
            (cols >= 0)
            & (cols < self._internal_cols)
            & (rows >= 0)
            & (rows < self._internal_rows)
        )
        if np.any(inside):
            flat_indices = rows[inside] * self._internal_cols + cols[inside]
            np.maximum.at(height.reshape(-1), flat_indices, points[inside, 2])
        return height, np.isfinite(height)

    def _push_and_fuse(self, stamp, packet_height, packet_valid):
        if self._last_stamp is not None and stamp < self._last_stamp - 1.0e-6:
            self.get_logger().warning('Point-cloud time moved backwards; clearing map history')
            self._history.clear()
        self._last_stamp = stamp
        self._history.append((stamp, packet_height, packet_valid))

        while len(self._history) > self._history_frames:
            self._history.popleft()
        while self._history and stamp - self._history[0][0] >= self._history_seconds:
            self._history.popleft()

        masked_packets = [
            np.where(valid, height, -np.inf)
            for _, height, valid in self._history
        ]
        fused_height = np.maximum.reduce(masked_packets)
        fused_valid = np.isfinite(fused_height)
        hit_count = np.sum(
            np.stack([valid for _, _, valid in self._history], axis=0), axis=0
        )
        confidence = hit_count.astype(np.float32) / float(len(self._history))
        return fused_height, fused_valid, confidence

    @staticmethod
    def _debug_cloud(stamp, frame_id, x, y, z, confidence, valid):
        count = int(valid.sum())
        records = np.empty(
            count,
            dtype=np.dtype(
                [
                    ('x', '<f4'),
                    ('y', '<f4'),
                    ('z', '<f4'),
                    ('intensity', '<f4'),
                ]
            ),
        )
        records['x'] = x[valid].astype(np.float32)
        records['y'] = y[valid].astype(np.float32)
        records['z'] = z[valid].astype(np.float32)
        records['intensity'] = confidence[valid].astype(np.float32)

        cloud = PointCloud2()
        cloud.header = Header(stamp=stamp, frame_id=frame_id)
        cloud.height = 1
        cloud.width = count
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = 16 * count
        cloud.data = records.tobytes()
        cloud.is_dense = True
        return cloud

    def _publish_policy_arrays(self, policy_height, policy_valid):
        height_msg = Float32MultiArray()
        height_msg.layout.dim = multi_array_layout(self._output_rows, self._output_cols)
        height_msg.layout.data_offset = 0
        height_msg.data = policy_height.astype(np.float32).reshape(-1).tolist()
        self._height_pub.publish(height_msg)

        valid_msg = UInt8MultiArray()
        valid_msg.layout.dim = multi_array_layout(self._output_rows, self._output_cols)
        valid_msg.layout.data_offset = 0
        valid_msg.data = policy_valid.astype(np.uint8).reshape(-1).tolist()
        self._valid_pub.publish(valid_msg)

    def _on_cloud(self, msg):
        if msg.header.frame_id != self._target_frame:
            now = time.monotonic()
            if now - self._last_frame_warning >= 2.0:
                self.get_logger().error(
                    f'Expected frame {self._target_frame!r}, got {msg.header.frame_id!r}; '
                    'run cloud_preprocessor and use /g1/lidar/elevation_input'
                )
                self._last_frame_warning = now
            return

        try:
            xyz_records, _, _ = xyz_view(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        xyz = np.column_stack(
            (xyz_records['x'], xyz_records['y'], xyz_records['z'])
        ).astype(np.float64, copy=False)

        packet_height, packet_valid = self._rasterize(xyz)
        stamp = self._stamp_seconds(msg.header.stamp)
        fused_height, fused_valid, confidence = self._push_and_fuse(
            stamp, packet_height, packet_valid
        )

        rs = self._output_row_slice
        cs = self._output_col_slice
        policy_raw_height = fused_height[rs, cs]
        policy_valid = fused_valid[rs, cs]
        policy_height = np.full(
            (self._output_rows, self._output_cols),
            self._invalid_height,
            dtype=np.float64,
        )
        relative_height = (
            -policy_raw_height[policy_valid] - self._height_offset
        ) * self._height_scale
        policy_height[policy_valid] = np.clip(
            relative_height, self._height_clip[0], self._height_clip[1]
        )
        self._publish_policy_arrays(policy_height, policy_valid)

        valid_ratio = float(policy_valid.mean())
        self._valid_ratio_pub.publish(Float32(data=valid_ratio))
        internal_cloud = self._debug_cloud(
            msg.header.stamp,
            self._target_frame,
            self._grid_x,
            self._grid_y,
            fused_height,
            confidence,
            fused_valid,
        )
        self._internal_cloud_pub.publish(internal_cloud)
        policy_cloud = self._debug_cloud(
            msg.header.stamp,
            self._target_frame,
            self._grid_x[rs, cs],
            self._grid_y[rs, cs],
            policy_raw_height,
            confidence[rs, cs],
            policy_valid,
        )
        self._policy_cloud_pub.publish(policy_cloud)

        self._count += 1
        if self._count == 1 or self._count % self._report_every == 0:
            internal_count = int(fused_valid.sum())
            policy_count = int(policy_valid.sum())
            if policy_count:
                valid_values = policy_height[policy_valid]
                value_summary = (
                    f', policy_height={valid_values.min():.3f}/'
                    f'{np.median(valid_values):.3f}/{valid_values.max():.3f}'
                )
            else:
                value_summary = ', policy_height=no valid cells'
            self.get_logger().info(
                f'clouds={self._count}, history={len(self._history)}, '
                f'internal_valid={internal_count}/{self._internal_rows * self._internal_cols}, '
                f'policy_valid={policy_count}/{self._output_rows * self._output_cols} '
                f'({valid_ratio:.1%}){value_summary}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = StaticElevationMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
