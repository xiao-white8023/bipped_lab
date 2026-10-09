"""
Build a motion-compensated rolling elevation map from FAST-LIO outputs.

Every registered scan is filtered in the gravity-aligned torso-yaw frame,
reduced to one representative point per map cell, and stored in ``g1_world``.
The bounded source-time history is reprojected into the current torso-yaw
frame before publishing the 11x17 policy terrain contract.
"""

from collections import deque
import copy
import math
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import (
    Float32,
    Float32MultiArray,
    Header,
    MultiArrayDimension,
    UInt8MultiArray,
)
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from .cloud_preprocessor import quaternion_matrix, xyz_view
from .rolling_history import highest_point_per_cell, TimedPointHistory


def cell_count(extent, resolution):
    """Return endpoint-inclusive cell count for an exactly divisible extent."""
    intervals = float(extent) / float(resolution)
    rounded = round(intervals)
    if not np.isclose(intervals, rounded, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            'map extents must be exact multiples of the resolution: '
            f'extent={extent}, resolution={resolution}'
        )
    return int(rounded) + 1  # 端点数


def multi_array_layout(rows, cols):
    """Create row-major [y, x] metadata for the policy arrays."""
    return [
        MultiArrayDimension(label='rows_y', size=rows, stride=rows * cols),
        MultiArrayDimension(label='cols_x', size=cols, stride=cols),
    ]


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def pose_matrix(position, orientation):
    """Return a homogeneous parent<-child matrix from a ROS pose/transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    matrix[:3, 3] = (position.x, position.y, position.z)
    return matrix


def parameter_matrix(translation, quaternion):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(*quaternion)
    matrix[:3, 3] = translation
    return matrix  # 得到其次变换矩阵


def yaw_from_rotation(rotation):
    """Return torso heading after projecting its x axis onto world XY."""
    heading = rotation[:2, 0]
    if np.linalg.norm(heading) < 1.0e-6:
        raise ValueError('torso x axis cannot define a horizontal heading')
    return math.atan2(heading[1], heading[0])


class RollingElevationMap(Node):
    """Fuse motion-compensated FAST-LIO scans into a current torso map."""

    def __init__(
        self, node_name='g1_rolling_elevation_map', uses_rolling_history=True
    ):
        super().__init__(node_name)
        self.declare_parameter('cloud_topic', '/cloud_registered') # FAST-LIO2 发布的、已经完成扫描运动补偿和世界配准的点云
        # 雷达中的点
        #     ↓ FAST-LIO外参  就是mid360到imu的位置与姿态 官网有
        # body/IMU中的点
        #     ↓ IMU估计的运动
        # 世界中的点
        self.declare_parameter('odometry_topic', '/Odometry') # FAST-LIO2 发布的机器人 body 位姿 FAST-LIO2 估计的机器人本体  点云描述“环境里的点在哪里”；Odometry 描述“机器人自己在哪里、朝向哪里”。
        self.declare_parameter('world_frame', 'g1_world') # 本项目采用的 Z 向上世界坐标系
        self.declare_parameter('lio_world_frame', 'camera_init') # FAST-LIO2 的初始化世界坐标系
        self.declare_parameter('lio_body_frame', 'body') # FAST-LIO2 里程计中的机器人刚体坐标系
        self.declare_parameter('capture_frame', 'capture_torso_yaw') # 每帧根据当前 torso 位置和 yaw 创建的策略坐标系
        self.declare_parameter('pose_sync_tolerance', 0.03) # 节点对每个点云，在 Odometry 队列中寻找时间戳最接近的一条  ！！！！！！！！！！ 这里有问题
        self.declare_parameter('sync_queue_size', 40)
        self.declare_parameter('input_qos_depth', 5)
        self.declare_parameter('drop_superseded_clouds', False)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('internal_size_x', 3.2)# 表示内部高城图覆盖  -1.6 ～ +1.6 m
        self.declare_parameter('internal_size_y', 3.2)# 表示内部高城图覆盖  -1.6 ～ +1.6 m
        self.declare_parameter('output_size_x', 1.6) # 最终的高城图范围X：-0.8 ～ +0.8 m
        self.declare_parameter('output_size_y', 1.0)#  最终的高城图范围Y：-0.5 ～ +0.5 m
        self.declare_parameter('capture_z_min', -1.5) # 注意这个 \(z\) 是点在 capture_torso_yaw 中的高度，相对于 torso 原点，而不是相对于地面。
        self.declare_parameter('capture_z_max', 0.5)
        self.declare_parameter('history_seconds', 10.0) # 保存最近10秒的代表点  这里使用的是点云消息的源时间戳，不是回调实际执行的系统时间
        self.declare_parameter('history_max_frames', 200) # 除了10秒限制，还规定最多保存200帧。  两个限制同时生效，哪个先达到就由哪个删除旧帧。
        self.declare_parameter('confidence_observations', 3) # 栅格化时，代码统计每个格子里有多少个历史点：hit_count
        # 然后计算confidence = np.clip(
        #           hit_count / confidence_observations,
        #           0.0,
        #           1.0,
        #           )
        self.declare_parameter('head_filter_enabled', True)
        self.declare_parameter('head_x_min', -0.05)
        self.declare_parameter('head_x_max', 0.15)
        self.declare_parameter('head_y_min', -0.25)
        self.declare_parameter('head_y_max', 0.25)
        self.declare_parameter('head_z_min', 0.15)
        self.declare_parameter('head_z_max', 0.50)
        self.declare_parameter('height_offset', 0.5)
        self.declare_parameter('height_scale', 1.0)
        self.declare_parameter('height_clip_min', -1.0)
        self.declare_parameter('height_clip_max', 1.0)
        self.declare_parameter('invalid_height', -1.1) # 无效观测视为-1.1
        self.declare_parameter('report_every', 10)

        # T_body_torso = T_body_lidar * inverse(T_torso_lidar).
        # Defaults use the current FAST-LIO extrinsic and measured head mount.
        # 以下7个参数一起定义了一个固定的坐标转换 把 torso 坐标系中的点，转换到 FAST-LIO 的 body 坐标系
        self.declare_parameter('body_to_torso_x', 0.01060918)
        self.declare_parameter('body_to_torso_y', -0.02326)
        self.declare_parameter('body_to_torso_z', 0.47200879)
        self.declare_parameter('body_to_torso_qx', 0.999673352)
        self.declare_parameter('body_to_torso_qy', 0.0)
        self.declare_parameter('body_to_torso_qz', -0.025557564)
        self.declare_parameter('body_to_torso_qw', 0.0)

        self._cloud_topic = self.get_parameter('cloud_topic').value
        self._odometry_topic = self.get_parameter('odometry_topic').value
        self._world_frame = self.get_parameter('world_frame').value
        self._lio_world_frame = self.get_parameter('lio_world_frame').value
        self._lio_body_frame = self.get_parameter('lio_body_frame').value
        self._capture_frame = self.get_parameter('capture_frame').value
        self._sync_tolerance = float(
            self.get_parameter('pose_sync_tolerance').value
        )
        self._sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self._input_qos_depth = int(
            self.get_parameter('input_qos_depth').value
        )
        self._drop_superseded_clouds = bool(
            self.get_parameter('drop_superseded_clouds').value
        )
        self._resolution = float(self.get_parameter('resolution').value)
        self._internal_size_x = float(self.get_parameter('internal_size_x').value)
        self._internal_size_y = float(self.get_parameter('internal_size_y').value)
        self._output_size_x = float(self.get_parameter('output_size_x').value)
        self._output_size_y = float(self.get_parameter('output_size_y').value)
        self._capture_z = (
            float(self.get_parameter('capture_z_min').value),
            float(self.get_parameter('capture_z_max').value),
        )
        self._history_seconds = float(
            self.get_parameter('history_seconds').value
        )
        self._history_max_frames = int(
            self.get_parameter('history_max_frames').value
        )
        self._confidence_observations = int(
            self.get_parameter('confidence_observations').value
        )
        self._head_filter_enabled = bool(
            self.get_parameter('head_filter_enabled').value
        )
        self._head_bounds = np.array(
            [
                self.get_parameter('head_x_min').value,
                self.get_parameter('head_x_max').value,
                self.get_parameter('head_y_min').value,
                self.get_parameter('head_y_max').value,
                self.get_parameter('head_z_min').value,
                self.get_parameter('head_z_max').value,
            ],
            dtype=np.float64,
        )
        self._height_offset = float(self.get_parameter('height_offset').value)
        self._height_scale = float(self.get_parameter('height_scale').value)
        self._height_clip = (
            float(self.get_parameter('height_clip_min').value),
            float(self.get_parameter('height_clip_max').value),
        )
        self._invalid_height = float(self.get_parameter('invalid_height').value)
        self._report_every = int(self.get_parameter('report_every').value)
        self._validate_configuration()

        translation = np.array(
            [
                self.get_parameter('body_to_torso_x').value,
                self.get_parameter('body_to_torso_y').value,
                self.get_parameter('body_to_torso_z').value,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                self.get_parameter('body_to_torso_qx').value,
                self.get_parameter('body_to_torso_qy').value,
                self.get_parameter('body_to_torso_qz').value,
                self.get_parameter('body_to_torso_qw').value,
            ],
            dtype=np.float64,
        )
        '''
        /cloud_registered'
        # FAST-LIO2 发布的、已经完成扫描运动补偿和世界配准的点云
        # 雷达中的点
        #     ↓ FAST-LIO外参  就是mid360到imu的位置与姿态 官网有
        # body/IMU中的点
        #     ↓ IMU估计的运动
        # 世界中的点
        '''
        self._body_to_torso = parameter_matrix(translation, quaternion) # 这个就是生成了一个把torso坐标系下的点转换到IMU/body坐标系下的齐次变换矩阵

        # 内部地图的列 行  x方向对应列 cols  y方向对应行 rows
        self._internal_cols = cell_count(self._internal_size_x, self._resolution)
        self._internal_rows = cell_count(self._internal_size_y, self._resolution)
        # 输出的高城图的列与行
        self._output_cols = cell_count(self._output_size_x, self._resolution)
        self._output_rows = cell_count(self._output_size_y, self._resolution)
        if (self._output_rows, self._output_cols) != (11, 17):
            raise ValueError(
                'pgtt_full_atten requires an 11x17 output grid, got '
                f'{self._output_rows}x{self._output_cols}'
            )
        '''
        [
            -1.6, -1.5, -1.4, ..., -0.1,
            0.0,
            0.1, ..., 1.4, 1.5, 1.6
        ]
        第0列：机器人后方1.6米
        第16列：机器人中心
        第32列：机器人前方1.6米

        第0行：机器人右侧1.6米
        第16行：机器人中心
        第32行：机器人左侧1.6米
        '''
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
        self._grid_x, self._grid_y = np.meshgrid(
            self._internal_x, self._internal_y
        )  # 将两个一维坐标轴展开成两个 33×33 坐标矩阵。
        col_start = self._aligned_start(
            -self._output_size_x / 2.0, self._internal_x, 'output x'
        )
        row_start = self._aligned_start(
            -self._output_size_y / 2.0, self._internal_y, 'output y'
        )
        self._output_col_slice = slice(col_start, col_start + self._output_cols)
        self._output_row_slice = slice(row_start, row_start + self._output_rows)

        self._cloud_queue = deque(maxlen=self._sync_queue_size) # 用于暂存刚收到、尚未匹配 Odometry 的 /cloud_registered 消息  sync_queue_size = 40  所以最多保存40帧点云 点云进入队列后，程序会尝试寻找时间最接近的 Odometry
        self._odom_queue = deque(maxlen=self._sync_queue_size * 4) # Odometry等待队列  默认最多保存160条 Odometry Odometry 队列比点云队列大，是为了保留更密集、更长一段时间的位姿数据，增加点云找到最近时间位姿的机会。
        '''
        匹配过程是：
        取最早的待处理点云时间
                ↓
        在odom_queue中计算所有时间差
                ↓
        找到时间差最小的一条Odometry
                ↓
        误差不超过30毫秒则配对处理
        '''
        self._history = TimedPointHistory(
            self._history_seconds, self._history_max_frames
        ) # 保存的是已经完成同步、坐标变换、裁剪和压缩的世界坐标代表点  就是已经是从一大堆雷达点中选择出来、用于代表某个栅格高度的真实雷达点。
        self._count = 0 # 统计成功处理并走到发布阶段的点云帧数
        self._dropped_clouds = 0  # 统计因为无法找到时间匹配的 Odometry 而主动丢弃的点云
        self._superseded_clouds = 0
        '''
        1. 收到 /cloud_registered
            得到世界中的当前扫描点

        2. 找到时间最接近的 /Odometry
        得到扫描时刻的机器人位置和姿态

        3. 把世界点变换到当前机器人坐标
        判断点位于机器人前后左右的哪里

        4. 裁剪3.2m×3.2m范围
        删除头部回波

        5. 每个格子保留最高的真实点
        这些就是当前帧代表点

        6. 把代表点重新转换回g1_world

        7. 保存到最近10秒历史
        '''
        self._last_warning = {}

        self._tf_buffer = Buffer() # 创建一个 TF 数据库，用来保存ROS系统中收到的坐标变换
        self._tf_listener = TransformListener(self._tf_buffer, self) # 会监听外界的tf转换，并把受到的关系写入self._tf_buffer
        self._tf_broadcaster = TransformBroadcaster(self) # 它负责把当前节点自己计算出的坐标变换发布到 ROS TF 系统中  g1_world → capture_torso_yaw \({}^{g1\_world}T_{capture}\)
        self._world_from_lio_cache = None

        self._height_pub = self.create_publisher(
            Float32MultiArray, '/g1/elevation/policy_heights', 10
        )
        self._valid_pub = self.create_publisher(
            UInt8MultiArray, '/g1/elevation/policy_valid', 10
        )
        self._valid_ratio_pub = self.create_publisher(
            Float32, '/g1/elevation/valid_ratio', 10
        )
        self._sync_error_pub = self.create_publisher(
            Float32, '/g1/elevation/pose_sync_error', 10
        )
        self._policy_age_pub = self.create_publisher(
            Float32MultiArray, '/g1/elevation/policy_age', 10
        )
        self._history_span_pub = self.create_publisher(
            Float32, '/g1/elevation/history_span', 10
        )
        self._internal_cloud_pub = self.create_publisher(
            PointCloud2, '/g1/elevation/internal_cloud', qos_profile_sensor_data
        )
        self._policy_cloud_pub = self.create_publisher(
            PointCloud2, '/g1/elevation/policy_cloud', qos_profile_sensor_data
        )
        input_qos = copy.copy(qos_profile_sensor_data)
        input_qos.depth = self._input_qos_depth
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            self._cloud_topic,
            self._on_cloud,
            input_qos,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            self._odometry_topic,
            self._on_odometry,
            input_qos,
        )

        if uses_rolling_history:
            storage_summary = (
                f'rolling_history={self._history_seconds:.1f}s/'
                f'{self._history_max_frames} frames'
            )
        else:
            storage_summary = 'storage=provided_by_derived_node'
        self.get_logger().info(
            f'Elevation map cloud={self._cloud_topic}, '
            f'odom={self._odometry_topic}, world={self._world_frame}, '
            f'capture={self._capture_frame}, {storage_summary}, '
            f'internal={self._internal_rows}x{self._internal_cols}, '
            f'policy={self._output_rows}x{self._output_cols}, '
            'order=y_ascending/x_ascending, source=pgtt_full_atten'
        )
        if self._head_filter_enabled:
            b = self._head_bounds
            self.get_logger().info(
                'Head-return filter in capture_torso_yaw: '
                f'x=[{b[0]:.2f}, {b[1]:.2f}], '
                f'y=[{b[2]:.2f}, {b[3]:.2f}], '
                f'z=[{b[4]:.2f}, {b[5]:.2f}] m'
            )
        self.get_logger().warning(
            'Validation stage: keep policy control disabled until pose sync, '
            'TF direction, map motion, and terrain heights have been checked.'
        )

    def _validate_configuration(self):
        if self._sync_tolerance <= 0.0:
            raise ValueError('pose_sync_tolerance must be positive')
        if self._sync_queue_size < 2:
            raise ValueError('sync_queue_size must be at least 2')
        if self._input_qos_depth < 1:
            raise ValueError('input_qos_depth must be at least one')
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
        if self._capture_z[0] >= self._capture_z[1]:
            raise ValueError('capture z bounds must be increasing')
        if self._history_seconds <= 0.0:
            raise ValueError('history_seconds must be positive')
        if self._history_max_frames < 1:
            raise ValueError('history_max_frames must be at least 1')
        if self._confidence_observations < 1:
            raise ValueError('confidence_observations must be at least 1')
        if self._head_filter_enabled and not (
            self._head_bounds[0] < self._head_bounds[1]
            and self._head_bounds[2] < self._head_bounds[3]
            and self._head_bounds[4] < self._head_bounds[5]
        ):
            raise ValueError('head filter bounds must be increasing')
        if self._height_clip[0] >= self._height_clip[1]:
            raise ValueError('height clipping bounds must be increasing')
        if self._height_clip[0] <= self._invalid_height <= self._height_clip[1]:
            raise ValueError('invalid_height must lie outside the clipping interval')
        if self._report_every < 1:
            raise ValueError('report_every must be at least 1')

    def _aligned_start(self, first_value, coordinates, name):
        index = int(round((first_value - coordinates[0]) / self._resolution))
        if index < 0 or index >= len(coordinates):
            raise ValueError(f'{name} lies outside the internal map')
        if not np.isclose(
            coordinates[index], first_value, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f'{name} is not aligned to the map resolution')
        return index

    def _warn_throttled(self, key, message):
        now = time.monotonic()
        if now - self._last_warning.get(key, 0.0) >= 2.0:
            self.get_logger().warning(message)
            self._last_warning[key] = now

    def _on_cloud(self, msg):
        if self._drop_superseded_clouds and self._cloud_queue:
            self._superseded_clouds += len(self._cloud_queue)
            self._cloud_queue.clear()
        self._cloud_queue.append(msg)
        self._drain_sync_queue()

    def _on_odometry(self, msg):
        self._odom_queue.append(msg)
        self._drain_sync_queue()

    def _drain_sync_queue(self):
        while self._cloud_queue and self._odom_queue:
            cloud = self._cloud_queue[0]
            cloud_time = stamp_seconds(cloud.header.stamp)
            odom_times = np.fromiter(
                (stamp_seconds(item.header.stamp) for item in self._odom_queue),
                dtype=np.float64,
            )
            best_index = int(np.argmin(np.abs(odom_times - cloud_time)))
            error = float(abs(odom_times[best_index] - cloud_time))
            if error <= self._sync_tolerance:
                self._cloud_queue.popleft()
                odometry = self._odom_queue[best_index]
                while self._odom_queue and (
                    stamp_seconds(self._odom_queue[0].header.stamp)
                    < cloud_time - self._sync_tolerance
                ):
                    self._odom_queue.popleft()
                self._process_pair(cloud, odometry, error)
                continue

            if odom_times[-1] > cloud_time + self._sync_tolerance:
                self._cloud_queue.popleft()
                self._dropped_clouds += 1
                self._warn_throttled(
                    'sync',
                    f'Dropping cloud without matching odometry: '
                    f'nearest error={error * 1000.0:.1f} ms',
                )
                continue
            break

    def _world_from_lio(self):  # camera_init ---> g1_world
        if self._world_from_lio_cache is not None:
            return self._world_from_lio_cache
        try:
            transform = self._tf_buffer.lookup_transform(
                self._world_frame, self._lio_world_frame, Time()
            ).transform
        except Exception as exc:
            self._warn_throttled(
                'world_tf',
                f'Waiting for TF {self._world_frame} <- '
                f'{self._lio_world_frame}: {exc}',
            )
            return None
        self._world_from_lio_cache = pose_matrix(
            transform.translation, transform.rotation
        )
        return self._world_from_lio_cache  # camera_init ---> g1_world

    def _validate_frames(self, cloud, odometry):
        cloud_frame = cloud.header.frame_id.lstrip('/')
        odom_parent = odometry.header.frame_id.lstrip('/')
        odom_child = odometry.child_frame_id.lstrip('/')
        expected_world = self._lio_world_frame.lstrip('/')
        expected_body = self._lio_body_frame.lstrip('/')
        if cloud_frame != expected_world:
            self._warn_throttled(
                'cloud_frame',
                f'Expected cloud frame {self._lio_world_frame!r}, got '
                f'{cloud.header.frame_id!r}',
            )
            return False
        if odom_parent != expected_world or odom_child != expected_body:
            self._warn_throttled(
                'odom_frame',
                f'Expected odometry {self._lio_world_frame} -> '
                f'{self._lio_body_frame}, got {odometry.header.frame_id} -> '
                f'{odometry.child_frame_id}',
            )
            return False
        return True

    @staticmethod
    def _yaw_frame(world_from_torso):
        yaw = yaw_from_rotation(world_from_torso[:3, :3])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rotation, yaw

    @staticmethod
    def _points_in_capture(points_world, torso_position, capture_rotation):
        return (points_world - torso_position) @ capture_rotation

    def _capture_mask(self, points_local):
        half_x = self._internal_size_x / 2.0
        half_y = self._internal_size_y / 2.0
        return (
            np.isfinite(points_local).all(axis=1)
            & (points_local[:, 0] >= -half_x)
            & (points_local[:, 0] <= half_x)
            & (points_local[:, 1] >= -half_y)
            & (points_local[:, 1] <= half_y)
            & (points_local[:, 2] >= self._capture_z[0])
            & (points_local[:, 2] <= self._capture_z[1])
        )

    def _capture_packet(self, points_local):
        return points_local[self._capture_mask(points_local)].copy()

    def _remove_head_returns(self, points_local):
        if not self._head_filter_enabled or not len(points_local):
            return points_local, 0
        b = self._head_bounds
        head = (
            (points_local[:, 0] >= b[0])
            & (points_local[:, 0] <= b[1])
            & (points_local[:, 1] >= b[2])
            & (points_local[:, 1] <= b[3])
            & (points_local[:, 2] >= b[4])
            & (points_local[:, 2] <= b[5])
        )
        return points_local[~head].copy(), int(head.sum())

    def _rasterize(self, points_local, point_ages):
        shape = (self._internal_rows, self._internal_cols)
        height = np.full(shape, -np.inf, dtype=np.float64)
        freshest_age = np.full(shape, np.inf, dtype=np.float64)
        hit_count = np.zeros(shape, dtype=np.int32)
        if len(points_local) != len(point_ages):
            raise ValueError('point ages must match the point count')
        if not len(points_local):
            valid = np.zeros(shape, dtype=bool)
            confidence = np.zeros(shape, dtype=np.float32)
            return height, valid, confidence, freshest_age

        finite = np.isfinite(points_local).all(axis=1) & np.isfinite(point_ages)
        points = points_local[finite]
        ages = point_ages[finite]
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
            & (points[:, 2] >= self._capture_z[0])
            & (points[:, 2] <= self._capture_z[1])
        )
        if np.any(inside):
            flat_indices = rows[inside] * self._internal_cols + cols[inside]
            np.maximum.at(height.reshape(-1), flat_indices, points[inside, 2])
            np.minimum.at(freshest_age.reshape(-1), flat_indices, ages[inside])
            np.add.at(hit_count.reshape(-1), flat_indices, 1)
        valid = np.isfinite(height)
        confidence = np.clip(
            hit_count.astype(np.float32) / float(self._confidence_observations),
            0.0,
            1.0,
        )
        return height, valid, confidence, freshest_age

    def _publish_capture_tf(self, stamp, torso_position, yaw):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self._world_frame
        transform.child_frame_id = self._capture_frame
        transform.transform.translation.x = float(torso_position[0])
        transform.transform.translation.y = float(torso_position[1])
        transform.transform.translation.z = float(torso_position[2])
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self._tf_broadcaster.sendTransform(transform)

    @staticmethod
    def _debug_cloud(stamp, frame_id, x, y, z, confidence, valid):
        count = int(valid.sum())
        records = np.empty(
            count,
            dtype=np.dtype(
                [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]
            ),
        )
        records['x'] = x[valid].astype(np.float32)
        records['y'] = y[valid].astype(np.float32)
        records['z'] = z[valid].astype(np.float32)
        records['intensity'] = confidence[valid].astype(np.float32)

        output = PointCloud2()
        output.header = Header(stamp=stamp, frame_id=frame_id)
        output.height = 1
        output.width = count
        output.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]
        output.is_bigendian = False
        output.point_step = 16
        output.row_step = 16 * count
        output.data = records.tobytes()
        output.is_dense = True
        return output

    def _publish_policy_arrays(self, policy_height, policy_valid, policy_age):
        height_msg = Float32MultiArray()
        height_msg.layout.dim = multi_array_layout(
            self._output_rows, self._output_cols
        )
        height_msg.layout.data_offset = 0
        height_msg.data = policy_height.astype(np.float32).reshape(-1).tolist()
        self._height_pub.publish(height_msg)

        valid_msg = UInt8MultiArray()
        valid_msg.layout.dim = multi_array_layout(
            self._output_rows, self._output_cols
        )
        valid_msg.layout.data_offset = 0
        valid_msg.data = policy_valid.astype(np.uint8).reshape(-1).tolist()
        self._valid_pub.publish(valid_msg)

        age_msg = Float32MultiArray()
        age_msg.layout.dim = multi_array_layout(
            self._output_rows, self._output_cols
        )
        age_msg.layout.data_offset = 0
        age_msg.data = policy_age.astype(np.float32).reshape(-1).tolist()
        self._policy_age_pub.publish(age_msg)

    def _process_pair(self, cloud, odometry, sync_error):
        if not self._validate_frames(cloud, odometry):
            return
        world_from_lio = self._world_from_lio()
        if world_from_lio is None:
            return
        try:
            xyz_records, _, _ = xyz_view(cloud)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        xyz = np.column_stack(
            (xyz_records['x'], xyz_records['y'], xyz_records['z'])
        ).astype(np.float64, copy=False)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]

        lio_from_body = pose_matrix(
            odometry.pose.pose.position, odometry.pose.pose.orientation
        ) # body ---> camera_init 的齐次坐标转换
        world_from_torso = world_from_lio @ lio_from_body @ self._body_to_torso
        torso_position = world_from_torso[:3, 3]
        try:
            capture_rotation, yaw = self._yaw_frame(world_from_torso)
        except ValueError as exc:
            self._warn_throttled('heading', str(exc))
            return

        points_world = xyz @ world_from_lio[:3, :3].T + world_from_lio[:3, 3]
        points_local = self._points_in_capture(
            points_world, torso_position, capture_rotation
        )
        current_packet = self._capture_packet(points_local)
        current_packet, head_removed = self._remove_head_returns(current_packet)
        representatives = highest_point_per_cell(
            current_packet,
            self._internal_size_x,
            self._internal_size_y,
            self._resolution,
        )
        representative_world = (
            representatives @ capture_rotation.T + torso_position
        )

        source_stamp = stamp_seconds(cloud.header.stamp)
        rewound = self._history.push(source_stamp, representative_world)
        if rewound:
            self.get_logger().warning(
                'Point-cloud source time moved backwards; map history was cleared'
            )

        history_world, history_ages = self._history.points_and_ages(source_stamp)
        history_local = self._points_in_capture(
            history_world, torso_position, capture_rotation
        )
        in_current_window = self._capture_mask(history_local)
        fused_points = history_local[in_current_window]
        fused_ages = history_ages[in_current_window]
        (
            fused_height,
            fused_valid,
            confidence,
            freshest_age,
        ) = self._rasterize(fused_points, fused_ages)

        rs = self._output_row_slice
        cs = self._output_col_slice
        policy_raw_height = fused_height[rs, cs]
        policy_valid = fused_valid[rs, cs]
        policy_age = np.full(
            (self._output_rows, self._output_cols), -1.0, dtype=np.float64
        )
        policy_age[policy_valid] = freshest_age[rs, cs][policy_valid]
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
        self._publish_policy_arrays(policy_height, policy_valid, policy_age)

        valid_ratio = float(policy_valid.mean())
        self._valid_ratio_pub.publish(Float32(data=valid_ratio))
        self._sync_error_pub.publish(Float32(data=float(sync_error)))
        self._history_span_pub.publish(
            Float32(data=float(self._history.span_seconds))
        )
        self._publish_capture_tf(cloud.header.stamp, torso_position, yaw)
        self._internal_cloud_pub.publish(
            self._debug_cloud(
                cloud.header.stamp,
                self._capture_frame,
                self._grid_x,
                self._grid_y,
                fused_height,
                confidence,
                fused_valid,
            )
        )
        self._policy_cloud_pub.publish(
            self._debug_cloud(
                cloud.header.stamp,
                self._capture_frame,
                self._grid_x[rs, cs],
                self._grid_y[rs, cs],
                policy_raw_height,
                confidence[rs, cs],
                policy_valid,
            )
        )

        self._count += 1
        if self._count == 1 or self._count % self._report_every == 0:
            policy_count = int(policy_valid.sum())
            if policy_count:
                values = policy_height[policy_valid]
                ages = policy_age[policy_valid]
                value_summary = (
                    f', policy_height={values.min():.3f}/'
                    f'{np.median(values):.3f}/{values.max():.3f}, '
                    f'freshest_age={ages.min():.2f}/'
                    f'{np.median(ages):.2f}/{ages.max():.2f}s'
                )
            else:
                value_summary = ', policy_height=no valid cells'
            self.get_logger().info(
                f'clouds={self._count}, current_roi={len(current_packet)}, '
                f'head_removed={head_removed}, stored_current={len(representatives)}, '
                f'history={len(self._history)} frames/'
                f'{self._history.point_count} points/'
                f'{self._history.span_seconds:.2f}s, '
                f'current_window_points={len(fused_points)}, '
                f'internal_valid={int(fused_valid.sum())}/'
                f'{self._internal_rows * self._internal_cols}, '
                f'policy_valid={policy_count}/'
                f'{self._output_rows * self._output_cols} ({valid_ratio:.1%}), '
                f'pose_dt={sync_error * 1000.0:.2f} ms, '
                f'dropped={self._dropped_clouds}{value_summary}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RollingElevationMap()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
