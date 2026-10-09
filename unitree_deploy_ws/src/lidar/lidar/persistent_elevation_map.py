"""
Persistent local-world elevation map with covariance-aware fusion.

This Phase 3 node keeps the Phase 2 input, synchronization, coordinate-frame,
filtering, and 11x17 policy contracts.  The only intentional architectural
change is storage: observations are fused into world-anchored elevation cells
instead of being discarded after a fixed ten-second point-history window.
"""

import copy
import math
import threading
import time

from builtin_interfaces.msg import Time as TimeMessage
from g1_policy_msgs.msg import ElevationPolicyInput
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import (
    Float32,
    Float32MultiArray,
    Float64,
    UInt32,
    UInt8MultiArray,
)

from .cloud_preprocessor import xyz_view
from .persistent_grid import PersistentElevationGrid
from .pose_prediction import PosePredictor
from .rolling_elevation_map import (
    multi_array_layout,
    pose_matrix,
    RollingElevationMap,
    stamp_seconds,
)


class PersistentElevationMap(RollingElevationMap):
    """Maintain a bounded, world-anchored local map across long stops."""

    def __init__(self):
        super().__init__(
            'g1_persistent_elevation_map',
            uses_rolling_history=False,
        )

        self.declare_parameter('persistent_resolution', self._resolution)
        self.declare_parameter('retention_radius', 2.5)
        self.declare_parameter('sensor_z_std', 0.02)
        self.declare_parameter('sensor_z_range_std_scale', 0.001)
        self.declare_parameter('measurement_variance_min', 1.0e-5)
        self.declare_parameter('measurement_variance_max', 0.04)
        self.declare_parameter('cell_variance_min', 1.0e-5)
        self.declare_parameter('cell_variance_max', 0.04)
        self.declare_parameter('max_valid_variance', 0.01)
        self.declare_parameter('variance_growth_per_second', 1.0e-6)
        self.declare_parameter('mahalanobis_threshold', 3.0)
        self.declare_parameter('candidate_confirmations', 3)
        self.declare_parameter('candidate_tolerance', 0.05)
        self.declare_parameter('candidate_timeout', 1.0)
        self.declare_parameter('conflict_variance_increase', 0.0004)
        self.declare_parameter('max_cell_age_seconds', 0.0)
        self.declare_parameter('pose_jump_translation', 0.5)
        self.declare_parameter('pose_jump_yaw', math.radians(45.0))
        self.declare_parameter('sampling_hysteresis_enabled', True)
        self.declare_parameter('sampling_hysteresis_m', 0.02)
        self.declare_parameter('policy_publish_rate', 50.0)
        self.declare_parameter(
            'policy_input_topic', '/g1/elevation/policy_input'
        )
        self.declare_parameter('pose_prediction_enabled', True)
        self.declare_parameter('prediction_window_seconds', 0.25)
        self.declare_parameter('prediction_minimum_span_seconds', 0.10)
        self.declare_parameter('prediction_maximum_horizon_seconds', 0.25)
        self.declare_parameter('prediction_maximum_clock_age_seconds', 1.0)
        self.declare_parameter('prediction_linear_deadband', 0.08)
        self.declare_parameter('prediction_yaw_rate_deadband', 0.04)
        self.declare_parameter('prediction_yaw_gain', 0.5)
        self.declare_parameter('prediction_maximum_linear_speed', 1.5)
        self.declare_parameter('prediction_maximum_vertical_speed', 0.5)
        self.declare_parameter('prediction_maximum_yaw_rate', 2.0)
        self.declare_parameter('diagnostic_publish_rate', 10.0)
        self.declare_parameter('debug_cloud_publish_rate', 2.0)

        self._persistent_resolution = float(
            self.get_parameter('persistent_resolution').value
        )
        self._retention_radius = float(
            self.get_parameter('retention_radius').value
        )
        self._sensor_z_std = float(
            self.get_parameter('sensor_z_std').value
        )
        self._sensor_z_range_std_scale = float(
            self.get_parameter('sensor_z_range_std_scale').value
        )
        self._measurement_variance = (
            float(self.get_parameter('measurement_variance_min').value),
            float(self.get_parameter('measurement_variance_max').value),
        )
        self._cell_variance = (
            float(self.get_parameter('cell_variance_min').value),
            float(self.get_parameter('cell_variance_max').value),
        )
        self._max_valid_variance = float(
            self.get_parameter('max_valid_variance').value
        )
        self._variance_growth_per_second = float(
            self.get_parameter('variance_growth_per_second').value
        )
        self._max_cell_age_seconds = float(
            self.get_parameter('max_cell_age_seconds').value
        )
        self._pose_jump_translation = float(
            self.get_parameter('pose_jump_translation').value
        )
        self._pose_jump_yaw = float(
            self.get_parameter('pose_jump_yaw').value
        )
        self._sampling_hysteresis_enabled = bool(
            self.get_parameter('sampling_hysteresis_enabled').value
        )
        self._sampling_hysteresis_m = float(
            self.get_parameter('sampling_hysteresis_m').value
        )
        self._policy_publish_rate = float(
            self.get_parameter('policy_publish_rate').value
        )
        self._pose_prediction_enabled = bool(
            self.get_parameter('pose_prediction_enabled').value
        )
        self._diagnostic_publish_rate = float(
            self.get_parameter('diagnostic_publish_rate').value
        )
        self._debug_cloud_publish_rate = float(
            self.get_parameter('debug_cloud_publish_rate').value
        )
        self._validate_persistent_configuration()

        self._persistent_grid = PersistentElevationGrid(
            resolution=self._persistent_resolution,
            variance_min=self._cell_variance[0],
            variance_max=self._cell_variance[1],
            mahalanobis_threshold=float(
                self.get_parameter('mahalanobis_threshold').value
            ),
            candidate_confirmations=int(
                self.get_parameter('candidate_confirmations').value
            ),
            candidate_tolerance=float(
                self.get_parameter('candidate_tolerance').value
            ),
            candidate_timeout=float(
                self.get_parameter('candidate_timeout').value
            ),
            conflict_variance_increase=float(
                self.get_parameter('conflict_variance_increase').value
            ),
            variance_growth_per_second=self._variance_growth_per_second,
        )
        self._pose_predictor = PosePredictor(
            window_seconds=float(
                self.get_parameter('prediction_window_seconds').value
            ),
            minimum_span_seconds=float(
                self.get_parameter(
                    'prediction_minimum_span_seconds'
                ).value
            ),
            maximum_horizon_seconds=float(
                self.get_parameter(
                    'prediction_maximum_horizon_seconds'
                ).value
            ),
            maximum_clock_age_seconds=float(
                self.get_parameter(
                    'prediction_maximum_clock_age_seconds'
                ).value
            ),
            linear_deadband=float(
                self.get_parameter('prediction_linear_deadband').value
            ),
            yaw_rate_deadband=float(
                self.get_parameter('prediction_yaw_rate_deadband').value
            ),
            yaw_prediction_gain=float(
                self.get_parameter('prediction_yaw_gain').value
            ),
            maximum_linear_speed=float(
                self.get_parameter(
                    'prediction_maximum_linear_speed'
                ).value
            ),
            maximum_vertical_speed=float(
                self.get_parameter(
                    'prediction_maximum_vertical_speed'
                ).value
            ),
            maximum_yaw_rate=float(
                self.get_parameter('prediction_maximum_yaw_rate').value
            ),
        )
        self._pose_predictor_lock = threading.Lock()
        self._map_state_lock = threading.Lock()
        self._last_persistent_position = None
        self._last_persistent_yaw = None
        self._last_source_stamp = None
        self._persistent_reset_count = 0
        self._map_epoch = 0
        self._sampling_grid = None
        self._sampling_indices = None
        rs = self._output_row_slice
        cs = self._output_col_slice
        self._policy_grid_x = self._grid_x[rs, cs].copy()
        self._policy_grid_y = self._grid_y[rs, cs].copy()
        self._last_sampling_key_switches = 0
        self._latest_pose_stamp_msg = None
        self._latest_map_source_stamp = None
        self._latest_map_stamp_msg = None
        self._latest_sync_error = 0.0
        self._latest_processing_time = 0.0
        self._latest_fusion_statistics = None
        self._policy_publish_count = 0
        self._last_policy_publish_clock = None
        self._last_reported_fusion_count = -1
        self._last_diagnostic_publish_time = -math.inf
        self._last_debug_publish_time = -math.inf

        self._variance_pub = self.create_publisher(
            Float32MultiArray, '/g1/elevation/policy_variance', 10
        )
        self._freshest_age_pub = self.create_publisher(
            Float32MultiArray, '/g1/elevation/policy_freshest_age', 10
        )
        self._source_stamp_pub = self.create_publisher(
            Float64, '/g1/elevation/source_stamp', 10
        )
        policy_qos = copy.copy(qos_profile_sensor_data)
        policy_qos.depth = 1
        self._policy_input_pub = self.create_publisher(
            ElevationPolicyInput,
            self.get_parameter('policy_input_topic').value,
            policy_qos,
        )
        self._map_cell_count_pub = self.create_publisher(
            UInt32, '/g1/elevation/map_cell_count', 10
        )
        self._map_reset_count_pub = self.create_publisher(
            UInt32, '/g1/elevation/map_reset_count', 10
        )
        self._processing_time_pub = self.create_publisher(
            Float32, '/g1/elevation/map_processing_time', 10
        )
        self._output_latency_pub = self.create_publisher(
            Float32, '/g1/elevation/source_to_output_latency', 10
        )
        self._prediction_horizon_pub = self.create_publisher(
            Float32, '/g1/elevation/pose_prediction_horizon', 10
        )
        self._sampling_switches_pub = self.create_publisher(
            UInt32, '/g1/elevation/sampling_key_switches', 10
        )
        self._superseded_clouds_pub = self.create_publisher(
            UInt32, '/g1/elevation/superseded_clouds', 10
        )
        self._policy_stop_event = threading.Event()
        self._policy_thread = threading.Thread(
            target=self._policy_publish_loop,
            name='g1_policy_height_publisher',
            daemon=True,
        )

        self.get_logger().info(
            'Phase 3 persistent map enabled: '
            f'resolution={self._persistent_resolution:.3f}m, '
            f'local_radius={self._retention_radius:.2f}m, '
            'hard_age_limit='
            f'{self._max_cell_age_seconds:.1f}s '
            '(0 means disabled), '
            f'max_valid_variance={self._max_valid_variance:.6f}m^2, '
            f'sampling_hysteresis={self._sampling_hysteresis_m:.3f}m, '
            f'policy_rate={self._policy_publish_rate:.1f}Hz, '
            f'diagnostic_rate={self._diagnostic_publish_rate:.1f}Hz, '
            f'pose_prediction={self._pose_prediction_enabled}'
        )
        self._policy_thread.start()

    def _validate_persistent_configuration(self):
        if self._persistent_resolution <= 0.0:
            raise ValueError('persistent_resolution must be positive')
        if not np.isclose(
            self._persistent_resolution,
            self._resolution,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                'persistent_resolution must equal policy-map resolution'
            )
        if self._retention_radius <= 0.0:
            raise ValueError('retention_radius must be positive')
        required_radius = 0.5 * math.hypot(
            self._internal_size_x, self._internal_size_y
        )
        if self._retention_radius + 1.0e-6 < required_radius:
            raise ValueError(
                'retention_radius must cover every corner of the internal '
                f'map (at least {required_radius:.3f} m)'
            )
        if self._sensor_z_std <= 0.0:
            raise ValueError('sensor_z_std must be positive')
        if self._sensor_z_range_std_scale < 0.0:
            raise ValueError('sensor_z_range_std_scale must be non-negative')
        if self._measurement_variance[0] <= 0.0:
            raise ValueError('measurement_variance_min must be positive')
        if self._measurement_variance[1] < self._measurement_variance[0]:
            raise ValueError(
                'measurement_variance_max must be at least its minimum'
            )
        if self._cell_variance[0] <= 0.0:
            raise ValueError('cell_variance_min must be positive')
        if self._cell_variance[1] < self._cell_variance[0]:
            raise ValueError('cell_variance_max must be at least its minimum')
        if not (
            self._cell_variance[0]
            <= self._max_valid_variance
            <= self._cell_variance[1]
        ):
            raise ValueError(
                'max_valid_variance must lie inside the cell variance range'
            )
        if self._variance_growth_per_second < 0.0:
            raise ValueError('variance_growth_per_second must be non-negative')
        if self._max_cell_age_seconds < 0.0:
            raise ValueError('max_cell_age_seconds must be non-negative')
        if self._pose_jump_translation <= 0.0:
            raise ValueError('pose_jump_translation must be positive')
        if not 0.0 < self._pose_jump_yaw <= math.pi:
            raise ValueError('pose_jump_yaw must be in (0, pi]')
        if not 0.0 <= self._sampling_hysteresis_m < (
            0.5 * self._persistent_resolution
        ):
            raise ValueError(
                'sampling_hysteresis_m must lie in [0, half a cell)'
            )
        if self._policy_publish_rate <= 0.0:
            raise ValueError('policy_publish_rate must be positive')
        if self._diagnostic_publish_rate < 0.0:
            raise ValueError(
                'diagnostic_publish_rate must be non-negative'
            )
        if self._debug_cloud_publish_rate < 0.0:
            raise ValueError(
                'debug_cloud_publish_rate must be non-negative'
            )

    @staticmethod
    def _positive_semidefinite(matrix):
        """Sanitize a covariance block before using it as a variance."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(matrix).all():
            return np.zeros_like(matrix)
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _measurement_variances(
        self, points_local, odometry, world_from_lio, capture_rotation
    ):
        """Propagate sensor and FAST-LIO pose uncertainty into world Z."""
        if not len(points_local):
            return np.empty((0,), dtype=np.float64)

        covariance_lio = np.asarray(
            odometry.pose.covariance, dtype=np.float64
        ).reshape(6, 6)
        covariance_lio = self._positive_semidefinite(covariance_lio)
        rotation_world_lio = world_from_lio[:3, :3]
        lio_to_world_adjoint = np.zeros((6, 6), dtype=np.float64)
        lio_to_world_adjoint[:3, :3] = rotation_world_lio
        lio_to_world_adjoint[3:, 3:] = rotation_world_lio
        covariance_world = (
            lio_to_world_adjoint
            @ covariance_lio
            @ lio_to_world_adjoint.T
        )
        world_to_capture_adjoint = np.zeros((6, 6), dtype=np.float64)
        world_to_capture_adjoint[:3, :3] = capture_rotation.T
        world_to_capture_adjoint[3:, 3:] = capture_rotation.T
        covariance_capture = (
            world_to_capture_adjoint
            @ covariance_world
            @ world_to_capture_adjoint.T
        )

        height_jacobian = np.zeros((len(points_local), 6), dtype=np.float64)
        height_jacobian[:, 2] = 1.0
        height_jacobian[:, 3] = points_local[:, 1]
        height_jacobian[:, 4] = -points_local[:, 0]
        pose_variance = np.einsum(
            'ni,ij,nj->n',
            height_jacobian,
            covariance_capture,
            height_jacobian,
        )
        ranges = np.linalg.norm(points_local, axis=1)
        sensor_std = (
            self._sensor_z_std
            + self._sensor_z_range_std_scale * ranges
        )
        variance = (
            sensor_std * sensor_std
            + np.maximum(pose_variance, 0.0)
        )
        return np.clip(
            variance,
            self._measurement_variance[0],
            self._measurement_variance[1],
        )

    @staticmethod
    def _angle_difference(first, second):
        return math.atan2(
            math.sin(first - second), math.cos(first - second)
        )

    def _reset_map(self, reason):
        with self._map_state_lock:
            self._sampling_grid = None
            self._map_epoch += 1
            self._sampling_indices = None
        self._persistent_grid.clear()
        with self._pose_predictor_lock:
            self._pose_predictor.reset()
        self._persistent_reset_count += 1
        self.get_logger().warning(
            f'Persistent elevation map cleared: {reason}'
        )

    def _check_coordinate_epoch(self, position, yaw, source_stamp):
        if (
            self._last_source_stamp is not None
            and source_stamp < self._last_source_stamp - 1.0e-6
        ):
            self._reset_map('point-cloud source time moved backwards')
        elif self._last_persistent_position is not None:
            translation = float(
                np.linalg.norm(position - self._last_persistent_position)
            )
            yaw_change = abs(
                self._angle_difference(yaw, self._last_persistent_yaw)
            )
            if (
                translation > self._pose_jump_translation
                or yaw_change > self._pose_jump_yaw
            ):
                self._reset_map(
                    'FAST-LIO pose discontinuity '
                    f'(translation={translation:.3f}m, '
                    f'yaw={math.degrees(yaw_change):.1f}deg)'
                )
        self._last_source_stamp = source_stamp
        self._last_persistent_position = position.copy()
        self._last_persistent_yaw = float(yaw)

    @staticmethod
    def _capture_rotation_from_yaw(yaw):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _cache_odometry_pose(self, odometry, world_from_lio=None):
        odom_parent = odometry.header.frame_id.lstrip('/')
        odom_child = odometry.child_frame_id.lstrip('/')
        if (
            odom_parent != self._lio_world_frame.lstrip('/')
            or odom_child != self._lio_body_frame.lstrip('/')
        ):
            return False
        if world_from_lio is None:
            world_from_lio = self._world_from_lio()
        if world_from_lio is None:
            return False
        lio_from_body = pose_matrix(
            odometry.pose.pose.position, odometry.pose.pose.orientation
        )
        world_from_torso = (
            world_from_lio @ lio_from_body @ self._body_to_torso
        )
        position = world_from_torso[:3, 3]
        try:
            _, yaw = self._yaw_frame(world_from_torso)
        except ValueError as exc:
            self._warn_throttled('prediction_heading', str(exc))
            return False
        source_stamp = stamp_seconds(odometry.header.stamp)
        with self._pose_predictor_lock:
            updated = self._pose_predictor.update(
                source_stamp, position, yaw
            )
        if updated:
            self._latest_pose_stamp_msg = odometry.header.stamp
        return updated

    def _on_odometry(self, msg):
        """Cache every newest pose before using it for cloud synchronization."""
        self._cache_odometry_pose(msg)
        super()._on_odometry(msg)

    @staticmethod
    def _publish_float_grid(publisher, values):
        rows, cols = values.shape
        message = Float32MultiArray()
        message.layout.dim = multi_array_layout(rows, cols)
        message.layout.data_offset = 0
        message.data = values.astype(np.float32).reshape(-1).tolist()
        publisher.publish(message)

    def _sample_grid(
        self,
        grid_x,
        grid_y,
        torso_position,
        capture_rotation,
        source_stamp,
        previous_indices=None,
        use_hysteresis=False,
        sampling_grid=None,
    ):
        """Sample an arbitrary torso-frame grid from the persistent map."""
        if sampling_grid is None:
            sampling_grid = self._sampling_grid
        if sampling_grid is None:
            raise RuntimeError('persistent map has no sampling snapshot')
        grid_x = np.asarray(grid_x, dtype=np.float64)
        grid_y = np.asarray(grid_y, dtype=np.float64)
        if grid_x.shape != grid_y.shape:
            raise ValueError('grid_x and grid_y must have matching shapes')
        local_query = np.column_stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                np.zeros(grid_x.size, dtype=np.float64),
            )
        )
        world_query = (
            local_query @ capture_rotation.T + torso_position
        )
        nearest_indices = sampling_grid.cell_indices(
            world_query[:, :2]
        )
        selected_indices = nearest_indices
        if use_hysteresis:
            selected_indices = (
                sampling_grid.hysteretic_cell_indices(
                    world_query[:, :2],
                    previous_indices,
                    self._sampling_hysteresis_m,
                )
            )
        sampled = sampling_grid.sample(
            world_query[:, :2],
            current_stamp=source_stamp,
            max_valid_variance=self._max_valid_variance,
            max_age_seconds=self._max_cell_age_seconds,
            cell_indices=selected_indices,
        )
        differs_from_nearest = np.any(
            selected_indices != nearest_indices, axis=1
        )
        fallback_candidates = (~sampled['valid']) & differs_from_nearest
        if np.any(fallback_candidates):
            nearest_sampled = sampling_grid.sample(
                world_query[:, :2],
                current_stamp=source_stamp,
                max_valid_variance=self._max_valid_variance,
                max_age_seconds=self._max_cell_age_seconds,
                cell_indices=nearest_indices,
            )
            fallback = fallback_candidates & nearest_sampled['valid']
            if np.any(fallback):
                for name in sampled:
                    sampled[name][fallback] = nearest_sampled[name][fallback]
                selected_indices[fallback] = nearest_indices[fallback]
        if previous_indices is None:
            key_switches = 0
        else:
            key_switches = int(
                np.any(
                    selected_indices != previous_indices, axis=1
                ).sum()
            )
        shape = grid_x.shape
        height = (
            sampled['height'].reshape(shape) - float(torso_position[2])
        )
        variance = sampled['variance'].reshape(shape)
        height_age = sampled['height_age'].reshape(shape)
        freshest_age = sampled['freshest_age'].reshape(shape)
        observations = sampled['observations'].reshape(shape)
        valid = sampled['valid'].reshape(shape)
        valid &= np.isfinite(height)
        valid &= height >= self._capture_z[0]
        valid &= height <= self._capture_z[1]

        confidence = np.zeros(shape, dtype=np.float32)
        denominator = max(
            1.0e-12,
            self._max_valid_variance - self._cell_variance[0],
        )
        variance_confidence = np.clip(
            1.0
            - (variance - self._cell_variance[0]) / denominator,
            0.0,
            1.0,
        )
        observation_confidence = np.clip(
            observations.astype(np.float64)
            / float(self._confidence_observations),
            0.0,
            1.0,
        )
        confidence[valid] = (
            variance_confidence[valid] * observation_confidence[valid]
        ).astype(np.float32)
        return (
            height,
            variance,
            height_age,
            freshest_age,
            valid,
            confidence,
            selected_indices.copy(),
            key_switches,
        )

    def _process_pair(self, cloud, odometry, sync_error):
        processing_start = time.perf_counter()
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
        )
        world_from_torso = (
            world_from_lio @ lio_from_body @ self._body_to_torso
        )
        torso_position = world_from_torso[:3, 3]
        try:
            capture_rotation, yaw = self._yaw_frame(world_from_torso)
        except ValueError as exc:
            self._warn_throttled('heading', str(exc))
            return

        points_world = (
            xyz @ world_from_lio[:3, :3].T + world_from_lio[:3, 3]
        )
        points_local = self._points_in_capture(
            points_world, torso_position, capture_rotation
        )
        current_packet = self._capture_packet(points_local)
        current_packet, head_removed = self._remove_head_returns(
            current_packet
        )
        packet_world = (
            current_packet @ capture_rotation.T + torso_position
        )
        measurement_variance = self._measurement_variances(
            current_packet,
            odometry,
            world_from_lio,
            capture_rotation,
        )

        source_stamp = stamp_seconds(cloud.header.stamp)
        self._check_coordinate_epoch(torso_position, yaw, source_stamp)
        self._cache_odometry_pose(odometry, world_from_lio)
        counters = self._persistent_grid.update(
            packet_world, measurement_variance, source_stamp
        )
        pruned = self._persistent_grid.prune(
            torso_position[:2],
            self._retention_radius,
            source_stamp,
            max_age_seconds=self._max_cell_age_seconds,
        )
        sampling_grid = self._persistent_grid.sampling_snapshot()
        self._count += 1
        processing_time = (
            time.perf_counter() - processing_start
        )
        fusion_statistics = {
            'current_roi': len(current_packet),
            'head_removed': head_removed,
            'counters': counters,
            'pruned': pruned,
        }
        with self._map_state_lock:
            self._latest_map_source_stamp = source_stamp
            self._latest_map_stamp_msg = cloud.header.stamp
            self._latest_sync_error = float(sync_error)
            self._latest_processing_time = processing_time
            self._latest_fusion_statistics = fusion_statistics
            self._sampling_grid = sampling_grid
        self._processing_time_pub.publish(
            Float32(data=float(processing_time))
        )

    def _predicted_policy_pose(self, now_seconds, map_source_stamp):
        with self._pose_predictor_lock:
            if not self._pose_predictor.ready:
                return None
            latest_stamp = self._pose_predictor.latest_stamp
            target_stamp = (
                now_seconds
                if self._pose_prediction_enabled
                else latest_stamp
            )
            position, yaw, horizon, clock_compatible = (
                self._pose_predictor.predict(target_stamp)
            )
        if not self._pose_prediction_enabled:
            clock_compatible = False
        sample_stamp = max(
            map_source_stamp,
            latest_stamp + horizon,
        )
        return position, yaw, horizon, clock_compatible, sample_stamp

    def _debug_publish_due(self):
        if self._debug_cloud_publish_rate <= 0.0:
            return False
        now = time.monotonic()
        period = 1.0 / self._debug_cloud_publish_rate
        if now - self._last_debug_publish_time < period:
            return False
        self._last_debug_publish_time = now
        return True

    def _policy_publish_loop(self):
        """Run the policy output clock independently of cloud fusion."""
        period = 1.0 / self._policy_publish_rate
        next_deadline = time.monotonic()
        while not self._policy_stop_event.is_set():
            delay = max(0.0, next_deadline - time.monotonic())
            if self._policy_stop_event.wait(delay):
                return
            try:
                self._publish_policy_snapshot()
            except Exception as exc:
                self.get_logger().error(
                    f'Policy publication thread stopped: {exc}'
                )
                self._policy_stop_event.set()
                return
            next_deadline += period
            now = time.monotonic()
            if next_deadline < now - period:
                next_deadline = now + period

    def _diagnostic_publish_due(self):
        if self._diagnostic_publish_rate <= 0.0:
            return False
        now = time.monotonic()
        period = 1.0 / self._diagnostic_publish_rate
        if now - self._last_diagnostic_publish_time < period:
            return False
        self._last_diagnostic_publish_time = now
        return True

    def _publish_realtime_policy_arrays(
        self, policy_height, policy_valid
    ):
        """Publish only the two arrays needed on the 50 Hz control path."""
        height_message = Float32MultiArray()
        height_message.layout.dim = multi_array_layout(
            self._output_rows, self._output_cols
        )
        height_message.layout.data_offset = 0
        height_message.data = (
            policy_height.astype(np.float32).reshape(-1).tolist()
        )
        self._height_pub.publish(height_message)

        valid_message = UInt8MultiArray()
        valid_message.layout.dim = multi_array_layout(
            self._output_rows, self._output_cols
        )
        valid_message.layout.data_offset = 0
        valid_message.data = (
            policy_valid.astype(np.uint8).reshape(-1).tolist()
        )
        self._valid_pub.publish(valid_message)

    @staticmethod
    def _time_message(seconds):
        """Convert finite non-negative seconds to builtin_interfaces/Time."""
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                'policy input timestamp must be finite and non-negative'
            )
        sec = int(math.floor(seconds))
        nanosec = int(round((seconds - sec) * 1_000_000_000.0))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return TimeMessage(sec=sec, nanosec=nanosec)

    def _publish_policy_input(
        self,
        sample_stamp,
        source_stamp,
        policy_height,
        policy_valid,
    ):
        """Publish one timestamped, internally consistent LP policy sample."""
        message = ElevationPolicyInput()
        message.header.stamp = self._time_message(sample_stamp)
        message.header.frame_id = self._capture_frame
        message.source_stamp = self._time_message(source_stamp)
        message.height = (
            policy_height.astype(np.float32).reshape(-1).tolist()
        )
        message.valid = (
            policy_valid.astype(np.uint8).reshape(-1).tolist()
        )
        message.valid_ratio = float(policy_valid.mean())
        self._policy_input_pub.publish(message)

    def _publish_policy_snapshot(self, force=False):
        """Sample the fused world map at a current, latency-compensated pose."""
        with self._map_state_lock:
            sampling_grid = self._sampling_grid
            map_source_stamp = self._latest_map_source_stamp
            map_stamp_msg = self._latest_map_stamp_msg
            sync_error = self._latest_sync_error
            processing_time = self._latest_processing_time
            fusion_statistics = self._latest_fusion_statistics
            map_epoch = self._map_epoch
            sampling_indices = self._sampling_indices
            cloud_count = self._count
        if sampling_grid is None or map_source_stamp is None:
            return
        now = self.get_clock().now()
        now_seconds = float(now.nanoseconds) * 1.0e-9
        if (
            self._last_policy_publish_clock is not None
            and now_seconds < self._last_policy_publish_clock
        ):
            self._last_policy_publish_clock = None
        predicted = self._predicted_policy_pose(
            now_seconds, map_source_stamp
        )
        if predicted is None:
            return
        (
            torso_position,
            yaw,
            prediction_horizon,
            clock_compatible,
            sample_stamp,
        ) = predicted
        capture_rotation = self._capture_rotation_from_yaw(yaw)
        (
            fused_height,
            fused_variance,
            height_age,
            freshest_age,
            fused_valid,
            confidence,
            selected_indices,
            key_switches,
        ) = self._sample_grid(
            self._policy_grid_x,
            self._policy_grid_y,
            torso_position,
            capture_rotation,
            sample_stamp,
            previous_indices=sampling_indices,
            use_hysteresis=self._sampling_hysteresis_enabled,
            sampling_grid=sampling_grid,
        )
        with self._map_state_lock:
            if map_epoch != self._map_epoch:
                return
            self._sampling_indices = selected_indices
            self._last_sampling_key_switches = key_switches

        policy_raw_height = fused_height
        policy_valid = fused_valid
        policy_age = np.full(policy_valid.shape, -1.0, dtype=np.float64)
        policy_freshest_age = np.full(
            policy_valid.shape, -1.0, dtype=np.float64
        )
        policy_variance = np.full(
            policy_valid.shape, -1.0, dtype=np.float64
        )
        policy_age[policy_valid] = height_age[policy_valid]
        policy_freshest_age[policy_valid] = freshest_age[policy_valid]
        policy_variance[policy_valid] = fused_variance[policy_valid]

        policy_height = np.full(
            policy_valid.shape, self._invalid_height, dtype=np.float64
        )
        relative_height = (
            -policy_raw_height[policy_valid] - self._height_offset
        ) * self._height_scale
        policy_height[policy_valid] = np.clip(
            relative_height, self._height_clip[0], self._height_clip[1]
        )
        self._publish_realtime_policy_arrays(policy_height, policy_valid)
        valid_ratio = float(policy_valid.mean())
        self._publish_policy_input(
            sample_stamp,
            map_source_stamp,
            policy_height,
            policy_valid,
        )
        source_latency = now_seconds - map_source_stamp
        if not clock_compatible or source_latency < 0.0:
            source_latency = -1.0
        diagnostic_due = self._diagnostic_publish_due()
        debug_due = self._debug_publish_due()
        output_stamp = None
        if diagnostic_due or debug_due:
            output_stamp = (
                now.to_msg()
                if clock_compatible
                else self._latest_pose_stamp_msg
            )
            if output_stamp is None:
                output_stamp = map_stamp_msg
            self._publish_capture_tf(output_stamp, torso_position, yaw)

        if diagnostic_due:
            self._publish_float_grid(self._policy_age_pub, policy_age)
            self._publish_float_grid(self._variance_pub, policy_variance)
            self._publish_float_grid(
                self._freshest_age_pub, policy_freshest_age
            )
            self._valid_ratio_pub.publish(Float32(data=valid_ratio))
            self._sync_error_pub.publish(
                Float32(data=float(sync_error))
            )
            self._history_span_pub.publish(
                Float32(
                    data=float(
                        sampling_grid.span_seconds(sample_stamp)
                    )
                )
            )
            self._source_stamp_pub.publish(
                Float64(data=map_source_stamp)
            )
            self._map_cell_count_pub.publish(
                UInt32(data=len(sampling_grid))
            )
            self._map_reset_count_pub.publish(
                UInt32(data=self._persistent_reset_count)
            )
            self._output_latency_pub.publish(
                Float32(data=float(source_latency))
            )
            self._prediction_horizon_pub.publish(
                Float32(data=float(prediction_horizon))
            )
            self._sampling_switches_pub.publish(
                UInt32(data=self._last_sampling_key_switches)
            )
            self._superseded_clouds_pub.publish(
                UInt32(data=self._superseded_clouds)
            )

        if debug_due:
            (
                internal_height,
                _,
                _,
                _,
                internal_valid,
                internal_confidence,
                _,
                _,
            ) = self._sample_grid(
                self._grid_x,
                self._grid_y,
                torso_position,
                capture_rotation,
                sample_stamp,
                sampling_grid=sampling_grid,
            )
            self._internal_cloud_pub.publish(
                self._debug_cloud(
                    output_stamp,
                    self._capture_frame,
                    self._grid_x,
                    self._grid_y,
                    internal_height,
                    internal_confidence,
                    internal_valid,
                )
            )
            self._policy_cloud_pub.publish(
                self._debug_cloud(
                    output_stamp,
                    self._capture_frame,
                    self._policy_grid_x,
                    self._policy_grid_y,
                    policy_raw_height,
                    confidence,
                    policy_valid,
                )
            )

        self._policy_publish_count += 1
        self._last_policy_publish_clock = now_seconds
        should_report = (
            cloud_count != self._last_reported_fusion_count
            and (
                cloud_count == 1
                or cloud_count % self._report_every == 0
            )
        )
        if should_report:
            self._report_status(
                policy_height,
                policy_valid,
                policy_age,
                policy_variance,
                valid_ratio,
                prediction_horizon,
                source_latency,
                sampling_grid,
                fusion_statistics,
                sync_error,
                processing_time,
                map_source_stamp,
                cloud_count,
            )
            self._last_reported_fusion_count = cloud_count

    def _report_status(
        self,
        policy_height,
        policy_valid,
        policy_age,
        policy_variance,
        valid_ratio,
        prediction_horizon,
        source_latency,
        sampling_grid,
        statistics,
        sync_error,
        processing_time,
        map_source_stamp,
        cloud_count,
    ):
        policy_count = int(policy_valid.sum())
        if policy_count:
            values = policy_height[policy_valid]
            ages = policy_age[policy_valid]
            variances = policy_variance[policy_valid]
            value_summary = (
                f', policy_height={values.min():.3f}/'
                f'{np.median(values):.3f}/{values.max():.3f}, '
                f'height_age={ages.min():.2f}/'
                f'{np.median(ages):.2f}/{ages.max():.2f}s, '
                f'variance={variances.min():.6f}/'
                f'{np.median(variances):.6f}/'
                f'{variances.max():.6f}m^2'
            )
        else:
            value_summary = ', policy_height=no valid cells'
        counters = statistics['counters']
        self.get_logger().info(
            f'clouds={cloud_count}, '
            f'current_roi={statistics["current_roi"]}, '
            f'head_removed={statistics["head_removed"]}, '
            f'measurements={counters["measurements"]}, '
            f'created/fused/conflict/replaced='
            f'{counters["created"]}/{counters["fused"]}/'
            f'{counters["conflicts"]}/{counters["replaced"]}, '
            f'pruned={statistics["pruned"]}, '
            f'map_cells={len(sampling_grid)}, '
            f'map_span={sampling_grid.span_seconds(map_source_stamp):.2f}s, '
            f'policy_valid={policy_count}/'
            f'{self._output_rows * self._output_cols} '
            f'({valid_ratio:.1%}), '
            f'pose_dt={sync_error * 1000.0:.2f}ms, '
            f'processing={processing_time * 1000.0:.1f}ms, '
            f'source_latency={source_latency * 1000.0:.1f}ms, '
            f'prediction={prediction_horizon * 1000.0:.1f}ms, '
            f'key_switches={self._last_sampling_key_switches}, '
            f'dropped={self._dropped_clouds}, '
            f'superseded={self._superseded_clouds}, '
            f'resets={self._persistent_reset_count}{value_summary}'
        )

    def destroy_node(self):
        """Stop the policy publisher before destroying ROS entities."""
        self._policy_stop_event.set()
        if (
            self._policy_thread.is_alive()
            and threading.current_thread() is not self._policy_thread
        ):
            self._policy_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersistentElevationMap()
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
