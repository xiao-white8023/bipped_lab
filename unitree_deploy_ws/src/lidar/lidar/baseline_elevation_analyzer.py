"""Collect quantitative diagnostics for the G1 single-scan elevation map."""

import csv
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2

from std_msgs.msg import Float32, Float32MultiArray, UInt8MultiArray

from .cloud_preprocessor import xyz_view


def finite_summary(values, scale=1.0):
    """Return JSON-safe descriptive statistics for finite scalar values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)] * float(scale)
    if not len(array):
        return {'count': 0}
    return {
        'count': int(len(array)),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'p50': float(np.percentile(array, 50)),
        'p95': float(np.percentile(array, 95)),
        'p99': float(np.percentile(array, 99)),
        'max': float(np.max(array)),
    }


class GridStatistics:
    """Accumulate per-cell validity, missing-run, and height statistics."""

    def __init__(
        self,
        rows=11,
        cols=17,
        invalid_height=-1.1,
        valid_min=-1.0,
        valid_max=1.0,
        tolerance=1.0e-5,
    ):
        """Initialize fixed-size grid accumulators."""
        self.rows = int(rows)
        self.cols = int(cols)
        self.size = self.rows * self.cols
        self.invalid_height = float(invalid_height)
        self.valid_min = float(valid_min)
        self.valid_max = float(valid_max)
        self.tolerance = float(tolerance)

        shape = (self.rows, self.cols)
        self.frames = 0
        self.valid_count = np.zeros(shape, dtype=np.int64)
        self.valid_value_errors = np.zeros(shape, dtype=np.int64)
        self.current_invalid_run = np.zeros(shape, dtype=np.int64)
        self.longest_invalid_run = np.zeros(shape, dtype=np.int64)
        self.height_count = np.zeros(shape, dtype=np.int64)
        self.height_sum = np.zeros(shape, dtype=np.float64)
        self.height_sum_sq = np.zeros(shape, dtype=np.float64)
        self.height_min = np.full(shape, np.inf, dtype=np.float64)
        self.height_max = np.full(shape, -np.inf, dtype=np.float64)
        self.invalid_encoding_errors = np.zeros(shape, dtype=np.int64)
        self.nonfinite_valid_errors = np.zeros(shape, dtype=np.int64)
        self.valid_range_errors = np.zeros(shape, dtype=np.int64)

    def add(self, heights, valid):
        """Consume one flattened policy height/valid pair."""
        height = np.asarray(
            heights, dtype=np.float64
        ).reshape(self.rows, self.cols)
        valid_values = np.asarray(
            valid, dtype=np.int64
        ).reshape(self.rows, self.cols)
        valid_mask = valid_values != 0
        valid_value_error = (valid_values != 0) & (valid_values != 1)
        finite = np.isfinite(height)

        invalid_error = (~valid_mask) & (
            (~finite)
            | (~np.isclose(
                height,
                self.invalid_height,
                rtol=0.0,
                atol=self.tolerance,
            ))
        )
        nonfinite_valid = valid_mask & ~finite
        valid_range_error = valid_mask & finite & (
            (height < self.valid_min - self.tolerance)
            | (height > self.valid_max + self.tolerance)
        )

        self.frames += 1
        self.valid_count += valid_mask
        self.valid_value_errors += valid_value_error
        self.current_invalid_run = np.where(
            valid_mask, 0, self.current_invalid_run + 1
        )
        self.longest_invalid_run = np.maximum(
            self.longest_invalid_run, self.current_invalid_run
        )
        self.invalid_encoding_errors += invalid_error
        self.nonfinite_valid_errors += nonfinite_valid
        self.valid_range_errors += valid_range_error

        usable = valid_mask & finite
        self.height_count += usable
        self.height_sum += np.where(usable, height, 0.0)
        self.height_sum_sq += np.where(usable, height * height, 0.0)
        self.height_min = np.where(
            usable, np.minimum(self.height_min, height), self.height_min
        )
        self.height_max = np.where(
            usable, np.maximum(self.height_max, height), self.height_max
        )

        valid_values = height[usable]
        return {
            'valid_count': int(valid_mask.sum()),
            'valid_ratio': float(valid_mask.mean()),
            'valid_value_errors': int(valid_value_error.sum()),
            'invalid_encoding_errors': int(invalid_error.sum()),
            'nonfinite_valid_errors': int(nonfinite_valid.sum()),
            'valid_range_errors': int(valid_range_error.sum()),
            'height_min': (
                float(np.min(valid_values)) if len(valid_values) else math.nan
            ),
            'height_median': (
                float(np.median(valid_values)) if len(valid_values) else math.nan
            ),
            'height_max': (
                float(np.max(valid_values)) if len(valid_values) else math.nan
            ),
        }

    def arrays(self):
        """Return finalized per-cell arrays."""
        valid_ratio = np.zeros_like(self.height_sum)
        if self.frames:
            valid_ratio = (
                self.valid_count.astype(np.float64) / float(self.frames)
            )

        height_mean = np.full_like(self.height_sum, np.nan)
        height_std = np.full_like(self.height_sum, np.nan)
        observed = self.height_count > 0
        height_mean[observed] = (
            self.height_sum[observed] / self.height_count[observed]
        )
        variance = np.zeros_like(self.height_sum)
        variance[observed] = (
            self.height_sum_sq[observed] / self.height_count[observed]
            - height_mean[observed] ** 2
        )
        height_std[observed] = np.sqrt(
            np.maximum(variance[observed], 0.0)
        )
        return {
            'valid_ratio': valid_ratio,
            'height_mean': height_mean,
            'height_std': height_std,
            'height_min': np.where(observed, self.height_min, np.nan),
            'height_max': np.where(observed, self.height_max, np.nan),
        }


class BaselineElevationAnalyzer(Node):
    """Analyze the current single-scan output without affecting mapping."""

    def __init__(self):
        """Declare inputs and prepare an empty diagnostic run."""
        super().__init__('g1_baseline_elevation_analyzer')
        self.declare_parameter(
            'height_topic', '/g1/elevation/policy_heights'
        )
        self.declare_parameter(
            'valid_topic', '/g1/elevation/policy_valid'
        )
        self.declare_parameter(
            'valid_ratio_topic', '/g1/elevation/valid_ratio'
        )
        self.declare_parameter(
            'sync_error_topic', '/g1/elevation/pose_sync_error'
        )
        self.declare_parameter(
            'policy_cloud_topic', '/g1/elevation/policy_cloud'
        )
        self.declare_parameter(
            'expected_cloud_frame', 'capture_torso_yaw'
        )
        self.declare_parameter('self_hit_x_min', -0.35)
        self.declare_parameter('self_hit_x_max', 0.35)
        self.declare_parameter('self_hit_y_min', -0.35)
        self.declare_parameter('self_hit_y_max', 0.35)
        self.declare_parameter('self_hit_z_min', -0.35)
        self.declare_parameter('self_hit_z_max', 0.5)
        self.declare_parameter('output_root', 'baseline_results')
        self.declare_parameter('run_name', '')
        self.declare_parameter('pair_timeout_sec', 0.05)
        self.declare_parameter('idle_timeout_sec', 0.0)
        self.declare_parameter('duration_sec', 0.0)
        self.declare_parameter('max_frames', 0)
        self.declare_parameter('max_stored_frames', 20000)
        self.declare_parameter('report_every', 50)

        self._rows = 11
        self._cols = 17
        self._size = self._rows * self._cols
        self._invalid_height = -1.1
        self._pair_timeout = float(
            self.get_parameter('pair_timeout_sec').value
        )
        self._idle_timeout = float(
            self.get_parameter('idle_timeout_sec').value
        )
        self._duration = float(self.get_parameter('duration_sec').value)
        self._max_frames = int(self.get_parameter('max_frames').value)
        self._max_stored_frames = int(
            self.get_parameter('max_stored_frames').value
        )
        self._report_every = int(
            self.get_parameter('report_every').value
        )
        self._expected_cloud_frame = self.get_parameter(
            'expected_cloud_frame'
        ).value.lstrip('/')
        self._self_hit_bounds = np.asarray(
            [
                self.get_parameter('self_hit_x_min').value,
                self.get_parameter('self_hit_x_max').value,
                self.get_parameter('self_hit_y_min').value,
                self.get_parameter('self_hit_y_max').value,
                self.get_parameter('self_hit_z_min').value,
                self.get_parameter('self_hit_z_max').value,
            ],
            dtype=np.float64,
        )
        self._validate_parameters()

        output_root = Path(
            self.get_parameter('output_root').value
        ).expanduser()
        run_name = self.get_parameter('run_name').value.strip()
        if not run_name:
            run_name = datetime.now(timezone.utc).strftime(
                '%Y%m%dT%H%M%SZ'
            )
        self._output_dir = output_root / run_name

        self._statistics = GridStatistics()
        self._pending_heights = deque()
        self._pending_valid = deque()
        self._frame_rows = []
        self._cloud_rows = []
        self._stored_heights = []
        self._stored_valid = []
        self._pair_delays = []
        self._published_valid_ratios = []
        self._sync_errors = []
        self._cloud_stamps = []
        self._cloud_arrivals = []
        self._first_pair_time = None
        self._last_pair_time = None
        self._written = False
        self._stop_requested = False
        self._counters = {
            'height_messages': 0,
            'valid_messages': 0,
            'valid_ratio_messages': 0,
            'sync_error_messages': 0,
            'policy_cloud_messages': 0,
            'height_layout_errors': 0,
            'valid_layout_errors': 0,
            'height_length_errors': 0,
            'valid_length_errors': 0,
            'unpaired_height_messages': 0,
            'unpaired_valid_messages': 0,
            'pair_timeout_errors': 0,
            'cloud_frame_errors': 0,
            'cloud_stamp_regressions': 0,
            'cloud_decode_errors': 0,
            'stored_frame_limit_drops': 0,
        }

        # Default QoS is reliable/volatile and works on both Foxy and Jazzy.
        reliable_qos = QoSProfile(depth=100)
        self._height_sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter('height_topic').value,
            self._on_height,
            reliable_qos,
        )
        self._valid_sub = self.create_subscription(
            UInt8MultiArray,
            self.get_parameter('valid_topic').value,
            self._on_valid,
            reliable_qos,
        )
        self._ratio_sub = self.create_subscription(
            Float32,
            self.get_parameter('valid_ratio_topic').value,
            self._on_valid_ratio,
            reliable_qos,
        )
        self._sync_sub = self.create_subscription(
            Float32,
            self.get_parameter('sync_error_topic').value,
            self._on_sync_error,
            reliable_qos,
        )
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('policy_cloud_topic').value,
            self._on_policy_cloud,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(
            0.2, self._check_stop_conditions
        )

        self.get_logger().info(
            'Collecting single-scan baseline diagnostics; '
            f'results will be written under {self._output_dir}'
        )
        self.get_logger().info(
            'This analyzer is read-only and publishes no robot commands.'
        )

    def _validate_parameters(self):
        if self._pair_timeout <= 0.0:
            raise ValueError('pair_timeout_sec must be positive')
        if self._idle_timeout < 0.0 or self._duration < 0.0:
            raise ValueError('timeout parameters must be nonnegative')
        if self._max_frames < 0 or self._max_stored_frames < 0:
            raise ValueError('frame limits must be nonnegative')
        if self._report_every < 1:
            raise ValueError('report_every must be at least one')
        bounds = self._self_hit_bounds
        if not (
            bounds[0] < bounds[1]
            and bounds[2] < bounds[3]
            and bounds[4] < bounds[5]
        ):
            raise ValueError('each self-hit bound minimum must be below maximum')

    def _layout_matches(self, message):
        dimensions = message.layout.dim
        if len(dimensions) != 2:
            return False
        return (
            int(dimensions[0].size) == self._rows
            and int(dimensions[1].size) == self._cols
            and int(message.layout.data_offset) == 0
        )

    def _on_height(self, message):
        now = time.monotonic()
        self._counters['height_messages'] += 1
        if not self._layout_matches(message):
            self._counters['height_layout_errors'] += 1
            self.get_logger().error(
                'Height message does not have an 11x17 layout'
            )
            return
        if len(message.data) != self._size:
            self._counters['height_length_errors'] += 1
            self.get_logger().error(
                f'Height message has {len(message.data)} values; '
                f'expected {self._size}'
            )
            return
        self._pending_heights.append(
            (now, np.asarray(message.data, dtype=np.float32).copy())
        )
        self._drain_array_pairs(now)

    def _on_valid(self, message):
        now = time.monotonic()
        self._counters['valid_messages'] += 1
        if not self._layout_matches(message):
            self._counters['valid_layout_errors'] += 1
            self.get_logger().error(
                'Validity message does not have an 11x17 layout'
            )
            return
        if len(message.data) != self._size:
            self._counters['valid_length_errors'] += 1
            self.get_logger().error(
                f'Validity message has {len(message.data)} values; '
                f'expected {self._size}'
            )
            return

        self._pending_valid.append(
            (now, np.asarray(message.data, dtype=np.uint8).copy())
        )
        self._drain_array_pairs(now)

    def _expire_array_messages(self, now):
        queues = (
            (self._pending_heights, 'unpaired_height_messages'),
            (self._pending_valid, 'unpaired_valid_messages'),
        )
        for queue, counter in queues:
            while queue and now - queue[0][0] > self._pair_timeout:
                queue.popleft()
                self._counters[counter] += 1
                self._counters['pair_timeout_errors'] += 1
            while len(queue) > 100:
                queue.popleft()
                self._counters[counter] += 1

    def _drain_array_pairs(self, now):
        self._expire_array_messages(now)
        while self._pending_heights and self._pending_valid:
            height_time, heights = self._pending_heights.popleft()
            valid_time, valid = self._pending_valid.popleft()
            pair_delay = abs(valid_time - height_time)
            if pair_delay > self._pair_timeout:
                self._counters['pair_timeout_errors'] += 1
                if height_time < valid_time:
                    self._counters['unpaired_height_messages'] += 1
                    self._pending_valid.appendleft((valid_time, valid))
                else:
                    self._counters['unpaired_valid_messages'] += 1
                    self._pending_heights.appendleft((height_time, heights))
                self._expire_array_messages(now)
                continue
            self._consume_array_pair(
                max(height_time, valid_time), pair_delay, heights, valid
            )

    def _consume_array_pair(self, now, pair_delay, heights, valid):
        frame_result = self._statistics.add(heights, valid)
        frame_index = self._statistics.frames
        if self._first_pair_time is None:
            self._first_pair_time = now
        self._last_pair_time = now
        self._pair_delays.append(pair_delay)
        self._frame_rows.append(
            {
                'frame': frame_index,
                'elapsed_sec': now - self._first_pair_time,
                'pair_delay_ms': pair_delay * 1000.0,
                **frame_result,
            }
        )
        if len(self._stored_heights) < self._max_stored_frames:
            self._stored_heights.append(heights)
            self._stored_valid.append(valid.copy())
        else:
            self._counters['stored_frame_limit_drops'] += 1

        if frame_index == 1 or frame_index % self._report_every == 0:
            contract_errors = (
                frame_result['valid_value_errors']
                + frame_result['invalid_encoding_errors']
                + frame_result['nonfinite_valid_errors']
                + frame_result['valid_range_errors']
            )
            self.get_logger().info(
                f'frames={frame_index}, '
                f"valid={frame_result['valid_count']}/{self._size} "
                f"({frame_result['valid_ratio']:.1%}), "
                f'pair_dt={pair_delay * 1000.0:.2f} ms, '
                f'contract_errors={contract_errors}'
            )
        if self._max_frames and frame_index >= self._max_frames:
            self._request_stop('max_frames reached')

    def _on_valid_ratio(self, message):
        self._counters['valid_ratio_messages'] += 1
        self._published_valid_ratios.append(float(message.data))

    def _on_sync_error(self, message):
        self._counters['sync_error_messages'] += 1
        self._sync_errors.append(float(message.data))

    def _on_policy_cloud(self, message):
        self._counters['policy_cloud_messages'] += 1
        now = time.monotonic()
        stamp = float(message.header.stamp.sec) + (
            float(message.header.stamp.nanosec) * 1.0e-9
        )
        frame = message.header.frame_id.lstrip('/')
        if frame != self._expected_cloud_frame:
            self._counters['cloud_frame_errors'] += 1
        if self._cloud_stamps and (
            stamp < self._cloud_stamps[-1] - 1.0e-9
        ):
            self._counters['cloud_stamp_regressions'] += 1
        self._cloud_stamps.append(stamp)
        self._cloud_arrivals.append(now)
        try:
            xyz_records, _, _ = xyz_view(message)
        except ValueError as exc:
            self._counters['cloud_decode_errors'] += 1
            self.get_logger().error(f'Cannot decode policy cloud: {exc}')
            return
        xyz = np.column_stack(
            (xyz_records['x'], xyz_records['y'], xyz_records['z'])
        ).astype(np.float64, copy=False)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        bounds = self._self_hit_bounds
        candidate = (
            (xyz[:, 0] >= bounds[0])
            & (xyz[:, 0] <= bounds[1])
            & (xyz[:, 1] >= bounds[2])
            & (xyz[:, 1] <= bounds[3])
            & (xyz[:, 2] >= bounds[4])
            & (xyz[:, 2] <= bounds[5])
        )
        self._cloud_rows.append(
            {
                'cloud': len(self._cloud_rows) + 1,
                'stamp_sec': stamp,
                'point_count': int(len(xyz_records)),
                'finite_point_count': int(len(xyz)),
                'self_hit_candidate_count': int(candidate.sum()),
                'self_hit_candidate_ratio': (
                    float(candidate.mean()) if len(candidate) else 0.0
                ),
                'z_min': float(np.min(xyz[:, 2])) if len(xyz) else math.nan,
                'z_median': (
                    float(np.median(xyz[:, 2])) if len(xyz) else math.nan
                ),
                'z_max': float(np.max(xyz[:, 2])) if len(xyz) else math.nan,
            }
        )

    def _check_stop_conditions(self):
        now = time.monotonic()
        self._expire_array_messages(now)
        if self._stop_requested or self._first_pair_time is None:
            return
        if self._duration and (
            now - self._first_pair_time >= self._duration
        ):
            self._request_stop('duration_sec reached')
            return
        if self._idle_timeout and (
            now - self._last_pair_time >= self._idle_timeout
        ):
            self._request_stop('input idle timeout reached')

    def _request_stop(self, reason):
        if self._stop_requested:
            return
        self._stop_requested = True
        self.get_logger().info(
            f'Stopping baseline collection: {reason}'
        )
        self.write_report()
        if rclpy.ok():
            rclpy.shutdown()

    def _unique_output_dir(self):
        candidate = self._output_dir
        suffix = 1
        while candidate.exists() and (
            not candidate.is_dir() or any(candidate.iterdir())
        ):
            candidate = Path(f'{self._output_dir}_{suffix:02d}')
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _write_per_cell_csv(self, output_dir, arrays):
        x_coordinates = np.linspace(-0.8, 0.8, self._cols)
        y_coordinates = np.linspace(-0.5, 0.5, self._rows)
        path = output_dir / 'per_cell.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow([
                'row_y', 'col_x', 'x_m', 'y_m', 'frames',
                'valid_count', 'valid_ratio',
                'longest_invalid_run_frames', 'height_count',
                'height_mean', 'height_std', 'height_min', 'height_max',
                'valid_value_errors', 'invalid_encoding_errors',
                'nonfinite_valid_errors', 'valid_range_errors',
            ])
            for row, y_value in enumerate(y_coordinates):
                for col, x_value in enumerate(x_coordinates):
                    writer.writerow([
                        row,
                        col,
                        float(x_value),
                        float(y_value),
                        self._statistics.frames,
                        int(self._statistics.valid_count[row, col]),
                        float(arrays['valid_ratio'][row, col]),
                        int(
                            self._statistics.longest_invalid_run[row, col]
                        ),
                        int(self._statistics.height_count[row, col]),
                        float(arrays['height_mean'][row, col]),
                        float(arrays['height_std'][row, col]),
                        float(arrays['height_min'][row, col]),
                        float(arrays['height_max'][row, col]),
                        int(
                            self._statistics.valid_value_errors[row, col]
                        ),
                        int(
                            self._statistics
                            .invalid_encoding_errors[row, col]
                        ),
                        int(
                            self._statistics
                            .nonfinite_valid_errors[row, col]
                        ),
                        int(
                            self._statistics.valid_range_errors[row, col]
                        ),
                    ])

    def _write_frame_csv(self, output_dir):
        fieldnames = [
            'frame', 'elapsed_sec', 'pair_delay_ms', 'valid_count',
            'valid_ratio', 'valid_value_errors', 'invalid_encoding_errors',
            'nonfinite_valid_errors', 'valid_range_errors',
            'height_min', 'height_median', 'height_max',
        ]
        with (output_dir / 'frames.csv').open(
            'w', newline='', encoding='utf-8'
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._frame_rows)

    def _write_heatmap(self, output_dir, valid_ratio):
        with (output_dir / 'valid_ratio_heatmap.txt').open(
            'w', encoding='utf-8'
        ) as stream:
            stream.write(
                'Rows: y=-0.5..+0.5 m; columns: x=-0.8..+0.8 m.\n'
            )
            stream.write('Values are per-cell valid percentages.\n')
            for row in valid_ratio:
                stream.write(
                    ' '.join(
                        f'{100.0 * value:6.1f}' for value in row
                    )
                )
                stream.write('\n')

    def write_report(self):
        """Write all collected diagnostics exactly once."""
        if self._written:
            return
        self._written = True
        self._expire_array_messages(time.monotonic())
        self._counters['unpaired_height_messages'] += len(
            self._pending_heights
        )
        self._counters['unpaired_valid_messages'] += len(
            self._pending_valid
        )
        self._pending_heights.clear()
        self._pending_valid.clear()
        output_dir = self._unique_output_dir()
        arrays = self._statistics.arrays()
        elapsed = 0.0
        if (
            self._first_pair_time is not None
            and self._last_pair_time is not None
        ):
            elapsed = self._last_pair_time - self._first_pair_time
        frame_ratios = [
            row['valid_ratio'] for row in self._frame_rows
        ]
        valid_ratio = arrays['valid_ratio']

        summary = {
            'contract': {
                'shape': [self._rows, self._cols],
                'cell_count': self._size,
                'x_range_m': [-0.8, 0.8],
                'y_range_m': [-0.5, 0.5],
                'sample_order': 'y_ascending/x_ascending',
                'invalid_height': self._invalid_height,
                'valid_height_range': [-1.0, 1.0],
                'expected_cloud_frame': self._expected_cloud_frame,
                'self_hit_candidate_box_m': {
                    'x': [
                        float(self._self_hit_bounds[0]),
                        float(self._self_hit_bounds[1]),
                    ],
                    'y': [
                        float(self._self_hit_bounds[2]),
                        float(self._self_hit_bounds[3]),
                    ],
                    'z': [
                        float(self._self_hit_bounds[4]),
                        float(self._self_hit_bounds[5]),
                    ],
                },
            },
            'collection': {
                'generated_utc': (
                    datetime.now(timezone.utc).isoformat()
                ),
                'paired_frames': self._statistics.frames,
                'elapsed_sec': elapsed,
                'paired_rate_hz': (
                    float((self._statistics.frames - 1) / elapsed)
                    if self._statistics.frames > 1 and elapsed > 0.0
                    else 0.0
                ),
                'stored_frame_arrays': len(self._stored_heights),
            },
            'transport_counters': self._counters,
            'frame_valid_ratio': finite_summary(frame_ratios),
            'published_valid_ratio': finite_summary(
                self._published_valid_ratios
            ),
            'height_valid_pair_delay_ms': finite_summary(
                self._pair_delays, scale=1000.0
            ),
            'cloud_odom_sync_error_ms': finite_summary(
                self._sync_errors, scale=1000.0
            ),
            'policy_cloud_stamp_period_ms': finite_summary(
                np.diff(self._cloud_stamps), scale=1000.0
            ),
            'policy_cloud_arrival_period_ms': finite_summary(
                np.diff(self._cloud_arrivals), scale=1000.0
            ),
            'per_cell_coverage': {
                'mean': float(np.mean(valid_ratio)),
                'min': float(np.min(valid_ratio)),
                'max': float(np.max(valid_ratio)),
                'never_valid_cells': int(
                    np.sum(self._statistics.valid_count == 0)
                ),
                'always_valid_cells': (
                    int(np.sum(
                        self._statistics.valid_count
                        == self._statistics.frames
                    ))
                    if self._statistics.frames
                    else 0
                ),
            },
            'contract_errors': {
                'valid_not_binary': int(
                    self._statistics.valid_value_errors.sum()
                ),
                'invalid_encoding': int(
                    self._statistics.invalid_encoding_errors.sum()
                ),
                'nonfinite_valid': int(
                    self._statistics.nonfinite_valid_errors.sum()
                ),
                'valid_out_of_range': int(
                    self._statistics.valid_range_errors.sum()
                ),
            },
            'policy_cloud': {
                'point_count': finite_summary(
                    [row['point_count'] for row in self._cloud_rows]
                ),
                'self_hit_candidate_count': finite_summary(
                    [
                        row['self_hit_candidate_count']
                        for row in self._cloud_rows
                    ]
                ),
                'self_hit_candidate_ratio': finite_summary(
                    [
                        row['self_hit_candidate_ratio']
                        for row in self._cloud_rows
                    ]
                ),
                'clouds_with_candidates': int(sum(
                    row['self_hit_candidate_count'] > 0
                    for row in self._cloud_rows
                )),
            },
        }

        with (output_dir / 'summary.json').open(
            'w', encoding='utf-8'
        ) as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write('\n')
        self._write_per_cell_csv(output_dir, arrays)
        self._write_frame_csv(output_dir)
        self._write_cloud_csv(output_dir)
        self._write_heatmap(output_dir, valid_ratio)

        stored_heights = np.asarray(
            self._stored_heights, dtype=np.float32
        )
        stored_valid = np.asarray(
            self._stored_valid, dtype=np.uint8
        )
        if not len(stored_heights):
            stored_heights = np.empty(
                (0, self._size), dtype=np.float32
            )
            stored_valid = np.empty(
                (0, self._size), dtype=np.uint8
            )
        np.savez_compressed(
            output_dir / 'baseline_arrays.npz',
            heights=stored_heights,
            valid=stored_valid,
            valid_count=self._statistics.valid_count,
            valid_value_errors=self._statistics.valid_value_errors,
            valid_ratio=valid_ratio,
            longest_invalid_run=(
                self._statistics.longest_invalid_run
            ),
            height_count=self._statistics.height_count,
            height_mean=arrays['height_mean'],
            height_std=arrays['height_std'],
            height_min=arrays['height_min'],
            height_max=arrays['height_max'],
            invalid_encoding_errors=(
                self._statistics.invalid_encoding_errors
            ),
            nonfinite_valid_errors=(
                self._statistics.nonfinite_valid_errors
            ),
            valid_range_errors=(
                self._statistics.valid_range_errors
            ),
            pair_delay_sec=np.asarray(
                self._pair_delays, dtype=np.float64
            ),
            sync_error_sec=np.asarray(
                self._sync_errors, dtype=np.float64
            ),
            cloud_stamp_sec=np.asarray(
                self._cloud_stamps, dtype=np.float64
            ),
        )
        self.get_logger().info(
            f'Baseline report written to {output_dir}; '
            f'frames={self._statistics.frames}, '
            f"mean_valid={summary['per_cell_coverage']['mean']:.1%}, "
            f'never_valid='
            f"{summary['per_cell_coverage']['never_valid_cells']}/"
            f'{self._size}'
        )

    def _write_cloud_csv(self, output_dir):
        fieldnames = [
            'cloud', 'stamp_sec', 'point_count', 'finite_point_count',
            'self_hit_candidate_count', 'self_hit_candidate_ratio',
            'z_min', 'z_median', 'z_max',
        ]
        with (output_dir / 'clouds.csv').open(
            'w', newline='', encoding='utf-8'
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._cloud_rows)


def main(args=None):
    """Run the read-only baseline analyzer."""
    rclpy.init(args=args)
    node = BaselineElevationAnalyzer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
