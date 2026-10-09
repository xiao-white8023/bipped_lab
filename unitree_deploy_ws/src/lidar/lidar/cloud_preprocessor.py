"""Filter and transform G1 MID-360 clouds for elevation-map diagnostics."""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header
from tf2_ros import Buffer, TransformListener


def quaternion_matrix(x, y, z, w):
    """Return a 3x3 rotation matrix for a normalized xyzw quaternion."""
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero-length transform quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def xyz_view(msg):
    """Return zero-copy structured x/y/z and raw-record views of PointCloud2."""
    fields = {field.name: field for field in msg.fields}
    missing = {"x", "y", "z"} - fields.keys()
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {sorted(missing)}")

    xyz_fields = [fields[name] for name in ("x", "y", "z")]
    if any(field.datatype != PointField.FLOAT32 or field.count != 1 for field in xyz_fields):
        raise ValueError("x, y, and z must be scalar FLOAT32 fields")

    byte_order = ">" if msg.is_bigendian else "<"
    xyz_dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (byte_order + "f4",) * 3,
            "offsets": tuple(field.offset for field in xyz_fields),
            "itemsize": msg.point_step,
        }
    )
    xyz_records = np.ndarray(
        shape=(msg.height, msg.width),
        dtype=xyz_dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    ).reshape(-1)
    raw_records = np.ndarray(
        shape=(msg.height, msg.width, msg.point_step),
        dtype=np.uint8,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step, 1),
    ).reshape(-1, msg.point_step)
    return xyz_records, raw_records, xyz_dtype


class CloudPreprocessor(Node):
    """Remove no-return points, transform the cloud, and crop a local ROI."""

    def __init__(self):
        super().__init__("g1_cloud_preprocessor")
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("filtered_topic", "/g1/lidar/filtered")
        self.declare_parameter("output_topic", "/g1/lidar/elevation_input")
        self.declare_parameter("target_frame", "torso_link")
        self.declare_parameter("min_range", 0.1) # 原始点云下的距离
        self.declare_parameter("max_range", 20.0) # 原始点云下的距离
        self.declare_parameter("x_min", -2.0)
        self.declare_parameter("x_max", 2.0)
        self.declare_parameter("y_min", -1.5)
        self.declare_parameter("y_max", 1.5)
        self.declare_parameter("z_min", -1.5)
        self.declare_parameter("z_max", 0.5)
        self.declare_parameter("report_every", 10)

        self._input_topic = self.get_parameter("input_topic").value
        self._filtered_topic = self.get_parameter("filtered_topic").value
        self._output_topic = self.get_parameter("output_topic").value
        self._target_frame = self.get_parameter("target_frame").value
        self._min_range = float(self.get_parameter("min_range").value)
        self._max_range = float(self.get_parameter("max_range").value)
        # 该roi区域是在躯干坐标系下的
        self._bounds = np.array(
            [
                self.get_parameter("x_min").value,
                self.get_parameter("x_max").value,
                self.get_parameter("y_min").value,
                self.get_parameter("y_max").value,
                self.get_parameter("z_min").value,
                self.get_parameter("z_max").value,
            ],
            dtype=np.float64,
        )
        self._report_every = max(1, int(self.get_parameter("report_every").value))
        if self._min_range < 0.0 or self._max_range <= self._min_range:
            raise ValueError("range limits must satisfy 0 <= min_range < max_range")
        if not (
            self._bounds[0] < self._bounds[1]
            and self._bounds[2] < self._bounds[3]
            and self._bounds[4] < self._bounds[5]
        ):
            raise ValueError("each ROI minimum must be smaller than its maximum")

        # 这两行是在给当前 ROS 2 节点建立一个 TF 坐标变换接收与查询系统
        '''
        translation:
            x = 0.0002835
            y = 0.00003
            z = 0.428434

        rotation:
            roll  = π
            pitch = 0.05112
            yaw   = 0
        这个变换由你运行的静态 TF 节点发布：
            ros2 run tf2_ros static_transform_publisher \
            0.0002835 0.00003 0.428434 \
            0 0.05112069379091391 3.141592653589793 \
            torso_link livox_frame
        '''

        """
        它保存收到的坐标关系，例如：
        torso_link → livox_frame
        odom → torso_link
        map → odom
        动态 TF 还会保存一段时间内的历史变化:
            时间1：机器人位于这里
            时间2：机器人向前移动
            时间3：机器人发生旋转
        """
        self._tf_buffer = Buffer()
        '''
                TransformListener 会为当前节点创建 TF 订阅器，主要监听：/tf  /tf_static    /tf：动态变化的坐标关系；    /tf_static：固定不变的坐标关系
            运行的static_transform_publisher 会把雷达安装关系发布到：/tf_static   TransformListener 收到后，把它写进：self._tf_buffer
        '''

        self._tf_listener = TransformListener(self._tf_buffer, self) # 负责接收 TF 消息 self就表示第二个节点  TransformListener 需要这个节点来创建/tf 订阅器；  /tf_static 订阅器等
        self._filtered_pub = self.create_publisher(
            PointCloud2, self._filtered_topic, qos_profile_sensor_data
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, self._output_topic, qos_profile_sensor_data
        ) # 这个是在经过roi后的点云
        self._valid_ratio_pub = self.create_publisher(
            Float32, "/g1/lidar/valid_ratio", 10
        )
        self._roi_ratio_pub = self.create_publisher(Float32, "/g1/lidar/roi_ratio", 10)
        self._sub = self.create_subscription(
            PointCloud2,
            self._input_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self._count = 0
        self._last_tf_warning = 0.0
        self.get_logger().info(
            f"Filtering {self._input_topic} -> {self._filtered_topic}; "
            f"ROI -> {self._output_topic}, target={self._target_frame}, "
            f"range=[{self._min_range}, {self._max_range}] m, "
            f"ROI=x[{self._bounds[0]}, {self._bounds[1]}] "
            f"y[{self._bounds[2]}, {self._bounds[3]}] "
            f"z[{self._bounds[4]}, {self._bounds[5]}]"
        )

    @staticmethod
    def _make_cloud(source, frame_id, raw_records):
        output = PointCloud2()
        output.header = Header(stamp=source.header.stamp, frame_id=frame_id)
        output.height = 1
        output.width = len(raw_records)
        output.fields = source.fields
        output.is_bigendian = source.is_bigendian
        output.point_step = source.point_step
        output.row_step = source.point_step * output.width
        output.data = raw_records.tobytes()
        output.is_dense = True
        return output

    def _on_cloud(self, msg):
        '''
            | 返回值 | 大致形状 | 含义 |
            |---|---|---|
            | `xyz_records` | `(N,)` | 可通过名称访问XYZ的结构化视图 |
            | `raw_records` | `(N, 26)` | 每个点完整的26个原始字节 |
            | `xyz_dtype` | dtype | 描述XYZ在26字节点记录中的布局 |
        '''
        try:
            xyz_records, raw_records, xyz_dtype = xyz_view(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        '''
        假设：
            xyz_records["x"] = [1.0, 4.0, 7.0]
            xyz_records["y"] = [2.0, 5.0, 8.0]
            xyz_records["z"] = [3.0, 6.0, 9.0]
        那么执行之后：
            xyz = [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                  ]
        '''
        xyz = np.column_stack(
            (xyz_records["x"], xyz_records["y"], xyz_records["z"])
        ).astype(np.float64, copy=False)
        """
        [
            [True,  True,  True ],
            [True,  False, True ],
            [False, True,  True ],
        ]
        """
        finite = np.isfinite(xyz).all(axis=1) # 检查每个坐标是否为有限数 .all(axis=1)表示对每一行检测是否全部为true
        ranges = np.linalg.norm(xyz, axis=1) # 计算每一个点到原点的欧斯距离
        """
        每个点必须同时满足：
            XYZ都是有限数
            距离大于0.1 m
            距离小于等于20 m
        """
        range_valid = finite & (ranges > self._min_range) & (ranges <= self._max_range) # 生成距离掩码

        """
        这段代码把 range_valid 中标记为有效的点，从原始26字节点记录里取出来，重新组成一条过滤后的 PointCloud2，发布到：/g1/lidar/filtered
        """
        source_indices = np.flatnonzero(range_valid)
        filtered_raw = raw_records[source_indices].copy()
        filtered = self._make_cloud(msg, msg.header.frame_id, filtered_raw) # 创建新的点云
        self._filtered_pub.publish(filtered)

        """
        lookup_transform(
            target_frame,
            source_frame,
            time,
        )
        查询如何把 source_frame 中的坐标转换到 target_frame
        TransformStamped
        ├── header
        │   ├── stamp
        │   └── frame_id = torso_link
        ├── child_frame_id = livox_frame
        └── transform
            ├── translation
            │   ├── x
            │   ├── y
            │   └── z
            └── rotation
                ├── x
                ├── y
                ├── z
                └── w
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame, msg.header.frame_id, Time()
            ).transform # 查询的是静态的TF坐标转换
        except Exception as exc:  # tf2 exception classes differ across Foxy/Jazzy.
            now = time.monotonic()
            if now - self._last_tf_warning >= 2.0:
                self.get_logger().warning(
                    f"Waiting for TF {self._target_frame} <- {msg.header.frame_id}: {exc}"
                )
                self._last_tf_warning = now
            return

        q = transform.rotation
        rotation = quaternion_matrix(q.x, q.y, q.z, q.w)
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float64,
        )
        transformed = xyz[range_valid] @ rotation.T + translation
        b = self._bounds
        in_roi = (
            (transformed[:, 0] >= b[0])
            & (transformed[:, 0] <= b[1])
            & (transformed[:, 1] >= b[2])
            & (transformed[:, 1] <= b[3])
            & (transformed[:, 2] >= b[4])
            & (transformed[:, 2] <= b[5])
        )
        selected_indices = source_indices[in_roi]
        selected_xyz = transformed[in_roi]

        selected_raw = raw_records[selected_indices].copy()
        selected_records = np.ndarray(
            shape=(len(selected_raw),), dtype=xyz_dtype, buffer=selected_raw
        )
        if len(selected_records):
            selected_records["x"] = selected_xyz[:, 0].astype(np.float32)
            selected_records["y"] = selected_xyz[:, 1].astype(np.float32)
            selected_records["z"] = selected_xyz[:, 2].astype(np.float32)

        output = self._make_cloud(msg, self._target_frame, selected_raw)
        self._cloud_pub.publish(output)

        """
        这段代码负责统计每一帧点云的处理结果，并发布两个比例指标:
            原始点数
            距离过滤后点数
            ROI区域内点数
            有效点比例
            ROI点比例

        """
        raw_count = len(xyz) # 原始点云的点数
        range_count = int(range_valid.sum()) # 距离过滤后的点数
        roi_count = len(selected_records) # roi之后的点数
        valid_ratio = range_count / raw_count if raw_count else 0.0  # 有效率
        roi_ratio = roi_count / raw_count if raw_count else 0.0
        # ROI点比例
        self._valid_ratio_pub.publish(Float32(data=float(valid_ratio)))
        self._roi_ratio_pub.publish(Float32(data=float(roi_ratio)))

        self._count += 1 # 处理成功的帧数 成功完成TF转换和ROI处理的点云帧数
        # 这个的意思就是说第一帧成功后打印，或者是每成功10帧后开始打印
        if self._count == 1 or self._count % self._report_every == 0:
            self.get_logger().info(
                f"clouds={self._count}, raw={raw_count}, range_valid={range_count} "
                f"({valid_ratio:.1%}), roi={roi_count} ({roi_ratio:.1%})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CloudPreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()