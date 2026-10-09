"""Convert rectified D435i depth into the G1 RENet VP image contract."""

import copy
import math
import time

from g1_policy_msgs.msg import DepthPolicyInput
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo, Image

from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension

from .depth_policy import (
    build_policy_projection,
    decode_depth_image,
    INVALID_DEPTH_MODES,
    PinholeIntrinsics,
    PolicyDepthHistory,
    preprocess_policy_depth,
)


def stamp_seconds(stamp):
    """Convert a ROS builtin time message to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def intrinsics_from_camera_info(message):
    """Read rectified intrinsics from CameraInfo, preferring P over K."""
    if len(message.p) == 12 and message.p[0] > 0.0 and message.p[5] > 0.0:
        fx, fy, cx, cy = (
            message.p[0],
            message.p[5],
            message.p[2],
            message.p[6],
        )
    elif len(message.k) == 9 and message.k[0] > 0.0 and message.k[4] > 0.0:
        fx, fy, cx, cy = (
            message.k[0],
            message.k[4],
            message.k[2],
            message.k[5],
        )
    else:
        raise ValueError('CameraInfo contains no valid P or K intrinsics')
    intrinsics = PinholeIntrinsics(
        width=int(message.width),
        height=int(message.height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    intrinsics.validate()
    return intrinsics


def history_layout(frames, rows, columns):
    """Create row-major [history, rows, columns] metadata."""
    return [
        MultiArrayDimension(
            label='history_oldest_to_newest',
            size=frames,
            stride=frames * rows * columns,
        ),
        MultiArrayDimension(
            label='rows_y',
            size=rows,
            stride=rows * columns,
        ),
        MultiArrayDimension(
            label='cols_x',
            size=columns,
            stride=columns,
        ),
    ]


class DepthPolicyPreprocessor(Node):
    """Publish policy-ready depth frames and a guarded two-frame history."""

    def __init__(self):
        """Configure the fixed policy contract and all ROS interfaces."""
        super().__init__('g1_depth_policy_preprocessor')
        self.declare_parameter(
            'depth_topic', '/camera/depth/image_rect_raw'
        )
        self.declare_parameter(
            'camera_info_topic', '/camera/depth/camera_info'
        )
        self.declare_parameter('policy_frame_topic', '/g1/depth/policy_frame')
        self.declare_parameter(
            'policy_history_topic', '/g1/depth/policy_history'
        )
        self.declare_parameter('policy_input_topic', '/g1/depth/policy_input')
        self.declare_parameter('valid_topic', '/g1/depth/policy_valid')
        self.declare_parameter('source_age_topic', '/g1/depth/source_age')
        self.declare_parameter('valid_ratio_topic', '/g1/depth/valid_ratio')
        self.declare_parameter('input_qos_depth', 1)
        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('capture_period', 0.1)
        self.declare_parameter('history_frames', 2)
        self.declare_parameter('target_width', 64)
        self.declare_parameter('target_height', 36)
        self.declare_parameter('target_horizontal_fov_deg', 89.51)
        self.declare_parameter('target_vertical_fov_deg', 58.29)
        self.declare_parameter('crop_top', 18)
        self.declare_parameter('crop_bottom', 0)
        self.declare_parameter('crop_left', 16)
        self.declare_parameter('crop_right', 16)
        self.declare_parameter('sampling_mode', 'nearest')
        self.declare_parameter('invalid_depth_mode', 'training')
        self.declare_parameter('blur_enable', True)
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 2.5)
        self.declare_parameter('minimum_projection_coverage', 0.99)
        self.declare_parameter('minimum_valid_ratio', 0.50)
        self.declare_parameter('maximum_source_age', 0.20)
        self.declare_parameter('future_stamp_tolerance', 0.05)
        self.declare_parameter('source_timeout', 0.25)
        self.declare_parameter('expected_frame_id', '')
        self.declare_parameter('watchdog_rate', 20.0)
        self.declare_parameter('warning_interval', 2.0)
        self.declare_parameter('report_every', 30)

        self._depth_topic = self.get_parameter('depth_topic').value
        self._camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).value
        self._depth_unit_scale = float(
            self.get_parameter('depth_unit_scale').value
        )
        self._capture_period = float(
            self.get_parameter('capture_period').value
        )
        self._history_frames = int(
            self.get_parameter('history_frames').value
        )
        self._target_width = int(self.get_parameter('target_width').value)
        self._target_height = int(self.get_parameter('target_height').value)
        self._target_horizontal_fov = float(
            self.get_parameter('target_horizontal_fov_deg').value
        )
        self._target_vertical_fov = float(
            self.get_parameter('target_vertical_fov_deg').value
        )
        self._crop = (
            int(self.get_parameter('crop_top').value),
            int(self.get_parameter('crop_bottom').value),
            int(self.get_parameter('crop_left').value),
            int(self.get_parameter('crop_right').value),
        )
        self._sampling_mode = self.get_parameter('sampling_mode').value
        self._invalid_depth_mode = self.get_parameter(
            'invalid_depth_mode'
        ).value
        blur_enable = self.get_parameter('blur_enable').value
        if not isinstance(blur_enable, bool):
            raise ValueError('blur_enable must be a boolean')
        self._blur_enable = blur_enable
        self._depth_min = float(self.get_parameter('depth_min').value)
        self._depth_max = float(self.get_parameter('depth_max').value)
        self._minimum_projection_coverage = float(
            self.get_parameter('minimum_projection_coverage').value
        )
        self._minimum_valid_ratio = float(
            self.get_parameter('minimum_valid_ratio').value
        )
        self._maximum_source_age = float(
            self.get_parameter('maximum_source_age').value
        )
        self._future_stamp_tolerance = float(
            self.get_parameter('future_stamp_tolerance').value
        )
        self._source_timeout = float(
            self.get_parameter('source_timeout').value
        )
        self._expected_frame_id = self.get_parameter(
            'expected_frame_id'
        ).value
        self._warning_interval = float(
            self.get_parameter('warning_interval').value
        )
        self._report_every = int(self.get_parameter('report_every').value)
        self._validate_configuration()

        output_height = self._target_height - self._crop[0] - self._crop[1]
        output_width = self._target_width - self._crop[2] - self._crop[3]
        self._output_shape = (output_height, output_width)
        self._history = PolicyDepthHistory(
            frame_count=self._history_frames,
            capture_period=self._capture_period,
        )
        self._valid_history = PolicyDepthHistory(
            frame_count=self._history_frames, capture_period=self._capture_period,
        )
        self._camera_info = None
        self._projection = None
        self._projection_key = None
        self._last_input_monotonic = None
        self._last_source_stamp = None
        self._last_valid_ratio = 0.0
        self._last_health = False
        self._warnings = {}
        self._published_frames = 0

        policy_qos = copy.copy(qos_profile_sensor_data)
        policy_qos.depth = 1
        self._frame_pub = self.create_publisher(
            Image,
            self.get_parameter('policy_frame_topic').value,
            policy_qos,
        )
        self._history_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('policy_history_topic').value,
            policy_qos,
        )
        self._policy_input_pub = self.create_publisher(
            DepthPolicyInput,
            self.get_parameter('policy_input_topic').value,
            policy_qos,
        )
        self._valid_pub = self.create_publisher(
            Bool, self.get_parameter('valid_topic').value, 1
        )
        self._age_pub = self.create_publisher(
            Float32, self.get_parameter('source_age_topic').value, 1
        )
        self._valid_ratio_pub = self.create_publisher(
            Float32, self.get_parameter('valid_ratio_topic').value, 1
        )

        input_qos = copy.copy(qos_profile_sensor_data)
        input_qos.depth = int(self.get_parameter('input_qos_depth').value)
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._on_camera_info,
            input_qos,
        )
        self._depth_sub = self.create_subscription(
            Image,
            self._depth_topic,
            self._on_depth,
            input_qos,
        )
        watchdog_rate = float(self.get_parameter('watchdog_rate').value)
        self._watchdog_timer = self.create_timer(
            1.0 / watchdog_rate, self._watchdog
        )

        self.get_logger().info(
            'RENet D435i preprocessor ready: '
            f'input={self._depth_topic}, info={self._camera_info_topic}, '
            f'target={self._target_width}x{self._target_height} '
            f'FOV={self._target_horizontal_fov:.2f}x'
            f'{self._target_vertical_fov:.2f}deg, crop={self._crop} -> '
            f'{output_height}x{output_width}, '
            f'history={self._history_frames} at '
            f'{1.0 / self._capture_period:.1f}Hz, '
            f'sampling={self._sampling_mode}, '
            f'blur={"Gaussian(3x3,sigma=1)" if self._blur_enable else "off"}, '
            f'invalid={self._invalid_depth_mode}, '
            'random_noise=off'
        )

    def _validate_configuration(self):
        if self._depth_unit_scale <= 0.0:
            raise ValueError('depth_unit_scale must be positive')
        if self._capture_period <= 0.0:
            raise ValueError('capture_period must be positive')
        if self._history_frames != 2:
            raise ValueError('the exported RENet policy requires 2 frames')
        output_height = self._target_height - self._crop[0] - self._crop[1]
        output_width = self._target_width - self._crop[2] - self._crop[3]
        if (output_height, output_width) != (18, 32):
            raise ValueError(
                'the exported RENet policy requires an 18x32 depth frame'
            )
        if self._sampling_mode not in ('nearest', 'bilinear'):
            raise ValueError('sampling_mode must be nearest or bilinear')
        if self._invalid_depth_mode not in INVALID_DEPTH_MODES:
            raise ValueError(
                'invalid_depth_mode must be one of '
                f'{INVALID_DEPTH_MODES}, got {self._invalid_depth_mode!r}'
            )
        if not 0.0 <= self._minimum_projection_coverage <= 1.0:
            raise ValueError('minimum_projection_coverage must be in [0, 1]')
        if not 0.0 <= self._minimum_valid_ratio <= 1.0:
            raise ValueError('minimum_valid_ratio must be in [0, 1]')
        if not 0.0 <= self._depth_min < self._depth_max:
            raise ValueError('depth limits must satisfy 0 <= min < max')
        if self._maximum_source_age <= 0.0:
            raise ValueError('maximum_source_age must be positive')
        if self._future_stamp_tolerance < 0.0:
            raise ValueError('future_stamp_tolerance must be non-negative')
        if self._source_timeout <= 0.0:
            raise ValueError('source_timeout must be positive')
        if self._warning_interval <= 0.0:
            raise ValueError('warning_interval must be positive')
        if self._report_every < 0:
            raise ValueError('report_every must be non-negative')
        if float(self.get_parameter('watchdog_rate').value) <= 0.0:
            raise ValueError('watchdog_rate must be positive')

    def _warn(self, key, message):
        now = time.monotonic()
        if now - self._warnings.get(key, -math.inf) >= self._warning_interval:
            self.get_logger().warning(message)
            self._warnings[key] = now

    def _on_camera_info(self, message):
        try:
            intrinsics = intrinsics_from_camera_info(message)
        except ValueError as error:
            self._warn('camera_info', f'Invalid CameraInfo: {error}')
            return
        self._camera_info = intrinsics

    def _projection_for(self, image):
        if self._camera_info is None:
            raise ValueError('waiting for CameraInfo')
        if (
            int(image.width) != self._camera_info.width
            or int(image.height) != self._camera_info.height
        ):
            raise ValueError(
                'Image/CameraInfo dimensions differ: '
                f'image={image.width}x{image.height}, '
                f'info={self._camera_info.width}x{self._camera_info.height}'
            )
        key = (
            self._camera_info,
            self._target_width,
            self._target_height,
            self._target_horizontal_fov,
            self._target_vertical_fov,
            self._crop,
            self._sampling_mode,
        )
        if key != self._projection_key:
            self._projection = build_policy_projection(
                self._camera_info,
                target_width=self._target_width,
                target_height=self._target_height,
                horizontal_fov_degrees=self._target_horizontal_fov,
                vertical_fov_degrees=self._target_vertical_fov,
                crop=self._crop,
                sampling_mode=self._sampling_mode,
            )
            self._projection_key = key
            self.get_logger().info(
                'CameraInfo accepted: '
                f'{self._camera_info.width}x{self._camera_info.height}, '
                f'fx={self._camera_info.fx:.3f}, '
                f'fy={self._camera_info.fy:.3f}, '
                f'cx={self._camera_info.cx:.3f}, '
                f'cy={self._camera_info.cy:.3f}, '
                'target-ray coverage='
                f'{100.0 * self._projection.coverage:.2f}%'
            )
        return self._projection

    def _source_age(self, source_stamp):
        if source_stamp <= 0.0:
            return math.inf
        now = self.get_clock().now().nanoseconds * 1.0e-9
        return now - source_stamp

    def _publish_health(self, healthy, source_age, valid_ratio):
        self._last_health = bool(healthy)
        self._last_valid_ratio = float(valid_ratio)
        valid_message = Bool()
        valid_message.data = self._last_health
        self._valid_pub.publish(valid_message)
        age_message = Float32()
        age_message.data = float(source_age)
        self._age_pub.publish(age_message)
        ratio_message = Float32()
        ratio_message.data = self._last_valid_ratio
        self._valid_ratio_pub.publish(ratio_message)

    def _publish_frame(self, source, frame):
        message = Image()
        message.header = source.header
        message.height = frame.shape[0]
        message.width = frame.shape[1]
        message.encoding = '32FC1'
        message.is_bigendian = False
        message.step = frame.shape[1] * np.dtype(np.float32).itemsize
        message.data = np.ascontiguousarray(frame, dtype='<f4').tobytes()
        self._frame_pub.publish(message)

    def _publish_history(
        self, source, history, healthy, source_age, valid_ratio, valid_support
    ):
        flat_history = np.ascontiguousarray(
            history.reshape(-1), dtype=np.float32
        )
        message = Float32MultiArray()
        frames, rows, columns = history.shape
        message.layout.dim = history_layout(frames, rows, columns)
        message.layout.data_offset = 0
        message.data = flat_history.tolist()
        self._history_pub.publish(message)

        policy_input = DepthPolicyInput()
        policy_input.header = source.header
        # Foxy represents Float32MultiArray.data as array.array after its
        # setter runs.  The generated setter for a fixed float32[N] field
        # does not accept that container, but does explicitly accept a
        # correctly typed numpy array.  Assign the canonical flattened array
        # directly instead of round-tripping it through message.data.
        policy_input.history = flat_history
        policy_input.valid_support = np.ascontiguousarray(valid_support.reshape(-1), dtype=np.float32)
        policy_input.valid = bool(healthy)
        policy_input.source_age = float(source_age)
        policy_input.valid_ratio = float(valid_ratio)
        self._policy_input_pub.publish(policy_input)

    def _on_depth(self, message):
        self._last_input_monotonic = time.monotonic()
        source_stamp = stamp_seconds(message.header.stamp)
        if not self._history.is_due(source_stamp):
            return
        try:
            projection = self._projection_for(message)
            depth_metres = decode_depth_image(
                message.data,
                message.height,
                message.width,
                message.step,
                message.encoding,
                message.is_bigendian,
                self._depth_unit_scale,
            )
            frame, stats = preprocess_policy_depth(
                depth_metres,
                projection,
                sampling_mode=self._sampling_mode,
                depth_min=self._depth_min,
                depth_max=self._depth_max,
                invalid_depth_mode=self._invalid_depth_mode,
                blur_enable=self._blur_enable,
            )
            history, rewound = self._history.push(source_stamp, frame)
            valid_support, valid_rewound = self._valid_history.push(source_stamp, stats.valid_support)
            if rewound != valid_rewound or (history is None) != (valid_support is None):
                raise RuntimeError('depth history and raw validity history lost synchronization')
            if history is None:
                return
        except (RuntimeError, ValueError) as error:
            self._warn('depth_processing', f'Depth frame rejected: {error}')
            self._publish_health(False, math.inf, 0.0)
            return

        if rewound:
            self._warn(
                'time_rewind',
                'Depth source time moved backwards; reset the two-frame history',
            )
        source_age = self._source_age(source_stamp)
        frame_matches = (
            not self._expected_frame_id
            or message.header.frame_id == self._expected_frame_id
        )
        healthy = (
            stats.projection_coverage >= self._minimum_projection_coverage
            and stats.valid_ratio >= self._minimum_valid_ratio
            and -self._future_stamp_tolerance
            <= source_age
            <= self._maximum_source_age
            and frame_matches
        )
        if not frame_matches:
            self._warn(
                'frame_id',
                f'Expected frame_id {self._expected_frame_id!r}, got '
                f'{message.header.frame_id!r}',
            )
        if stats.projection_coverage < self._minimum_projection_coverage:
            self._warn(
                'coverage',
                'D435i CameraInfo cannot cover all training rays: '
                f'{100.0 * stats.projection_coverage:.2f}% coverage',
            )
        if stats.valid_ratio < self._minimum_valid_ratio:
            self._warn(
                'valid_ratio',
                'Too few valid D435i depth samples: '
                f'{100.0 * stats.valid_ratio:.1f}%',
            )
        if not -self._future_stamp_tolerance <= source_age <= self._maximum_source_age:
            self._warn(
                'source_age',
                'D435i timestamp is outside the policy freshness window: '
                f'age={source_age:.3f}s',
            )

        self._publish_frame(message, frame)
        self._publish_history(
            message, history, healthy, source_age, stats.valid_ratio, valid_support
        )
        self._publish_health(healthy, source_age, stats.valid_ratio)
        self._last_source_stamp = source_stamp
        self._published_frames += 1
        if (
            self._report_every > 0
            and self._published_frames % self._report_every == 0
        ):
            self.get_logger().info(
                f'VP frame #{self._published_frames}: '
                f'healthy={healthy}, age={source_age:.3f}s, '
                f'valid={100.0 * stats.valid_ratio:.1f}%, '
                f'range=[{stats.minimum_metres:.3f}, '
                f'{stats.maximum_metres:.3f}]m'
            )

    def _watchdog(self):
        if self._last_input_monotonic is None:
            self._publish_health(False, math.inf, 0.0)
            return
        silence = time.monotonic() - self._last_input_monotonic
        source_age = (
            math.inf
            if self._last_source_stamp is None
            else self._source_age(self._last_source_stamp)
        )
        timed_out = silence > self._source_timeout
        stale = source_age > self._maximum_source_age
        future = source_age < -self._future_stamp_tolerance
        if timed_out or stale or future:
            self._publish_health(False, source_age, self._last_valid_ratio)
            if timed_out:
                self._warn(
                    'source_timeout',
                    f'No D435i depth message for {silence:.3f}s',
                )


def main(args=None):
    """Run the D435i policy-depth preprocessor."""
    rclpy.init(args=args)
    node = DepthPolicyPreprocessor()
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
