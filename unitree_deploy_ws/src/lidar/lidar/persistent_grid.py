"""Persistent world-frame elevation cells with covariance-aware fusion."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class ElevationCell:
    """State retained for one world-anchored XY cell."""

    height: float
    variance: float
    last_seen: float
    last_update: float
    observations: int = 1
    candidate_height: float = np.nan
    candidate_variance: float = np.nan
    candidate_count: int = 0
    candidate_last_seen: float = -np.inf


class PersistentElevationGrid:
    """Sparse robot-local window whose cell keys are fixed in world XY."""

    def __init__(
        self,
        resolution,
        variance_min,
        variance_max,
        mahalanobis_threshold,
        candidate_confirmations,
        candidate_tolerance,
        candidate_timeout,
        conflict_variance_increase,
        variance_growth_per_second=0.0,
    ):
        self.resolution = float(resolution)
        self.variance_min = float(variance_min)
        self.variance_max = float(variance_max)
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.candidate_confirmations = int(candidate_confirmations)
        self.candidate_tolerance = float(candidate_tolerance)
        self.candidate_timeout = float(candidate_timeout)
        self.conflict_variance_increase = float(
            conflict_variance_increase
        )
        self.variance_growth_per_second = float(
            variance_growth_per_second
        )
        self._validate_configuration()
        self._cells = {}
        self._last_stamp = None

    def _validate_configuration(self):
        if self.resolution <= 0.0:
            raise ValueError('resolution must be positive')
        if self.variance_min <= 0.0:
            raise ValueError('variance_min must be positive')
        if self.variance_max < self.variance_min:
            raise ValueError('variance_max must be at least variance_min')
        if self.mahalanobis_threshold <= 0.0:
            raise ValueError('mahalanobis_threshold must be positive')
        if self.candidate_confirmations < 1:
            raise ValueError('candidate_confirmations must be at least one')
        if self.candidate_tolerance <= 0.0:
            raise ValueError('candidate_tolerance must be positive')
        if self.candidate_timeout <= 0.0:
            raise ValueError('candidate_timeout must be positive')
        if self.conflict_variance_increase < 0.0:
            raise ValueError(
                'conflict_variance_increase must be non-negative'
            )
        if self.variance_growth_per_second < 0.0:
            raise ValueError(
                'variance_growth_per_second must be non-negative'
            )

    def __len__(self):
        return len(self._cells)

    def clear(self):
        """Discard all cells and reset source-time ordering."""
        self._cells.clear()
        self._last_stamp = None

    def sampling_snapshot(self):
        """Return an isolated copy safe for concurrent read-only sampling."""
        snapshot = PersistentElevationGrid(
            resolution=self.resolution,
            variance_min=self.variance_min,
            variance_max=self.variance_max,
            mahalanobis_threshold=self.mahalanobis_threshold,
            candidate_confirmations=self.candidate_confirmations,
            candidate_tolerance=self.candidate_tolerance,
            candidate_timeout=self.candidate_timeout,
            conflict_variance_increase=self.conflict_variance_increase,
            variance_growth_per_second=self.variance_growth_per_second,
        )
        snapshot._cells = {
            key: replace(cell) for key, cell in self._cells.items()
        }
        snapshot._last_stamp = self._last_stamp
        return snapshot

    def span_seconds(self, current_stamp):
        """Return the oldest retained height-update age."""
        if not self._cells:
            return 0.0
        oldest = min(cell.last_update for cell in self._cells.values())
        return max(0.0, float(current_stamp) - oldest)

    def cell_indices(self, points_xy):
        """Return nearest world-cell keys for arbitrary XY coordinates."""
        return np.rint(
            np.asarray(points_xy, dtype=np.float64) / self.resolution
        ).astype(np.int64)

    def hysteretic_cell_indices(
        self, points_xy, previous_indices, hysteresis_m
    ):
        """
        Keep previous keys inside an expanded nearest-cell boundary.

        Nearest-neighbour sampling changes key exactly at half a grid cell.
        Millimetre-scale localization jitter can therefore alternate between
        two stable cells at a discontinuity such as a stair edge.  A previous
        key is retained until the query crosses the ordinary boundary plus
        ``hysteresis_m``.  Large real motion still switches directly to the
        nearest current key.
        """
        points_xy = np.asarray(points_xy, dtype=np.float64)
        nearest = self.cell_indices(points_xy)
        if previous_indices is None:
            return nearest
        previous = np.asarray(previous_indices, dtype=np.int64)
        if previous.shape != nearest.shape:
            raise ValueError(
                'previous_indices must match the query XY shape'
            )
        hysteresis_m = float(hysteresis_m)
        if hysteresis_m < 0.0:
            raise ValueError('hysteresis_m must be non-negative')
        previous_centers = previous.astype(np.float64) * self.resolution
        retain = np.abs(points_xy - previous_centers) <= (
            0.5 * self.resolution + hysteresis_m
        )
        return np.where(retain, previous, nearest)

    def _aggregate_scan(self, points_world, variances):
        """Keep the highest measurement in each world cell for this scan."""
        points_world = np.asarray(points_world, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError('points_world must have shape (N, 3)')
        if variances.shape != (len(points_world),):
            raise ValueError('variances must have shape (N,)')
        finite = (
            np.isfinite(points_world).all(axis=1)
            & np.isfinite(variances)
            & (variances > 0.0)
        )
        points = points_world[finite]
        variances = np.clip(
            variances[finite], self.variance_min, self.variance_max
        )
        if not len(points):
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        indices = self.cell_indices(points[:, :2])
        order = np.lexsort((points[:, 2], indices[:, 1], indices[:, 0]))
        sorted_indices = indices[order]
        take = np.empty(len(order), dtype=bool)
        take[:-1] = np.any(
            sorted_indices[:-1] != sorted_indices[1:], axis=1
        )
        take[-1] = True
        selected = order[take]
        return indices[selected], points[selected, 2], variances[selected]

    @staticmethod
    def _clear_candidate(cell):
        cell.candidate_height = np.nan
        cell.candidate_variance = np.nan
        cell.candidate_count = 0
        cell.candidate_last_seen = -np.inf

    def _grow_variance(self, cell, stamp):
        """Materialize uncertainty accumulated since the last observation."""
        elapsed = max(0.0, float(stamp) - cell.last_seen)
        cell.variance = np.clip(
            cell.variance + self.variance_growth_per_second * elapsed,
            self.variance_min,
            self.variance_max,
        )

    def _fuse_primary(self, cell, height, variance, stamp):
        innovation = max(
            self.variance_min, cell.variance + variance
        )
        gain = cell.variance / innovation
        cell.height += gain * (height - cell.height)
        cell.variance = np.clip(
            (1.0 - gain) * cell.variance,
            self.variance_min,
            self.variance_max,
        )
        cell.last_seen = stamp
        cell.last_update = stamp
        cell.observations += 1
        self._clear_candidate(cell)

    def _update_candidate(self, cell, height, variance, stamp):
        candidate_is_current = (
            cell.candidate_count > 0
            and stamp - cell.candidate_last_seen <= self.candidate_timeout
            and abs(height - cell.candidate_height)
            <= self.candidate_tolerance
        )
        if not candidate_is_current:
            cell.candidate_height = height
            cell.candidate_variance = variance
            cell.candidate_count = 1
        else:
            innovation = max(
                self.variance_min,
                cell.candidate_variance + variance,
            )
            gain = cell.candidate_variance / innovation
            cell.candidate_height += gain * (
                height - cell.candidate_height
            )
            cell.candidate_variance = np.clip(
                (1.0 - gain) * cell.candidate_variance,
                self.variance_min,
                self.variance_max,
            )
            cell.candidate_count += 1
        cell.candidate_last_seen = stamp
        cell.last_seen = stamp
        cell.variance = min(
            self.variance_max,
            cell.variance + self.conflict_variance_increase,
        )
        if cell.candidate_count < self.candidate_confirmations:
            return False

        cell.height = cell.candidate_height
        cell.variance = np.clip(
            cell.candidate_variance,
            self.variance_min,
            self.variance_max,
        )
        cell.last_update = stamp
        cell.observations = cell.candidate_count
        self._clear_candidate(cell)
        return True

    def update(self, points_world, variances, stamp):
        """
        Fuse one scan and return update counters.

        A source-time rewind clears the coordinate epoch before inserting the
        new scan. Large pose discontinuities are handled by the ROS node.
        """
        stamp = float(stamp)
        rewound = (
            self._last_stamp is not None
            and stamp < self._last_stamp - 1.0e-6
        )
        if rewound:
            self.clear()
        self._last_stamp = stamp

        indices, heights, scan_variances = self._aggregate_scan(
            points_world, variances
        )
        counters = {
            'rewound': rewound,
            'measurements': int(len(indices)),
            'created': 0,
            'fused': 0,
            'conflicts': 0,
            'replaced': 0,
        }
        threshold_squared = self.mahalanobis_threshold ** 2
        for index, height, variance in zip(
            indices, heights, scan_variances
        ):
            key = (int(index[0]), int(index[1]))
            cell = self._cells.get(key)
            if cell is None:
                self._cells[key] = ElevationCell(
                    height=float(height),
                    variance=float(variance),
                    last_seen=stamp,
                    last_update=stamp,
                )
                counters['created'] += 1
                continue

            self._grow_variance(cell, stamp)
            residual = float(height) - cell.height
            innovation = max(
                self.variance_min, cell.variance + float(variance)
            )
            if residual * residual <= threshold_squared * innovation:
                self._fuse_primary(
                    cell, float(height), float(variance), stamp
                )
                counters['fused'] += 1
                continue

            counters['conflicts'] += 1
            if self._update_candidate(
                cell, float(height), float(variance), stamp
            ):
                counters['replaced'] += 1
        return counters

    def prune(self, center_xy, radius, current_stamp, max_age_seconds=0.0):
        """Remove cells outside the robot-local radius or explicit age cap."""
        center_xy = np.asarray(center_xy, dtype=np.float64)
        if center_xy.shape != (2,):
            raise ValueError('center_xy must have shape (2,)')
        radius = float(radius)
        max_age_seconds = float(max_age_seconds)
        if radius <= 0.0:
            raise ValueError('radius must be positive')
        radius_squared = radius * radius
        remove = []
        for key, cell in self._cells.items():
            world_xy = self.resolution * np.asarray(key, dtype=np.float64)
            outside = np.sum((world_xy - center_xy) ** 2) > radius_squared
            too_old = (
                max_age_seconds > 0.0
                and float(current_stamp) - cell.last_update
                > max_age_seconds
            )
            if outside or too_old:
                remove.append(key)
        for key in remove:
            del self._cells[key]
        return len(remove)

    def sample(
        self,
        query_world_xy,
        current_stamp,
        max_valid_variance,
        max_age_seconds=0.0,
        cell_indices=None,
    ):
        """Sample world cells at query XY or at explicitly supplied keys."""
        query_world_xy = np.asarray(query_world_xy, dtype=np.float64)
        if query_world_xy.shape[-1] != 2:
            raise ValueError('query_world_xy last dimension must be two')
        output_shape = query_world_xy.shape[:-1]
        flat_xy = query_world_xy.reshape(-1, 2)
        if cell_indices is None:
            indices = self.cell_indices(flat_xy)
        else:
            supplied = np.asarray(cell_indices, dtype=np.int64)
            if supplied.shape != query_world_xy.shape:
                raise ValueError(
                    'cell_indices must match the query XY shape'
                )
            indices = supplied.reshape(-1, 2)
        height = np.full(len(flat_xy), np.nan, dtype=np.float64)
        variance = np.full(len(flat_xy), np.inf, dtype=np.float64)
        height_age = np.full(len(flat_xy), -1.0, dtype=np.float64)
        freshest_age = np.full(len(flat_xy), -1.0, dtype=np.float64)
        observations = np.zeros(len(flat_xy), dtype=np.int32)
        valid = np.zeros(len(flat_xy), dtype=bool)
        max_valid_variance = float(max_valid_variance)
        max_age_seconds = float(max_age_seconds)

        for i, index in enumerate(indices):
            cell = self._cells.get((int(index[0]), int(index[1])))
            if cell is None:
                continue
            age = max(0.0, float(current_stamp) - cell.last_update)
            fresh = max(0.0, float(current_stamp) - cell.last_seen)
            age_valid = max_age_seconds <= 0.0 or age <= max_age_seconds
            effective_variance = np.clip(
                cell.variance
                + self.variance_growth_per_second * fresh,
                self.variance_min,
                self.variance_max,
            )
            variance_valid = effective_variance <= max_valid_variance
            if not age_valid or not variance_valid:
                continue
            height[i] = cell.height
            variance[i] = effective_variance
            height_age[i] = age
            freshest_age[i] = fresh
            observations[i] = cell.observations
            valid[i] = True

        return {
            'height': height.reshape(output_shape),
            'variance': variance.reshape(output_shape),
            'height_age': height_age.reshape(output_shape),
            'freshest_age': freshest_age.reshape(output_shape),
            'observations': observations.reshape(output_shape),
            'valid': valid.reshape(output_shape),
        }

    def world_points(self):
        """Return one XYZ point and state vector for every retained cell."""
        count = len(self._cells)
        points = np.empty((count, 3), dtype=np.float64)
        variances = np.empty(count, dtype=np.float64)
        last_updates = np.empty(count, dtype=np.float64)
        observations = np.empty(count, dtype=np.int32)
        for i, (key, cell) in enumerate(self._cells.items()):
            points[i] = (
                key[0] * self.resolution,
                key[1] * self.resolution,
                cell.height,
            )
            variances[i] = cell.variance
            last_updates[i] = cell.last_update
            observations[i] = cell.observations
        return points, variances, last_updates, observations
