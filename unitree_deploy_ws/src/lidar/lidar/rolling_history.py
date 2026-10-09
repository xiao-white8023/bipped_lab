"""Pure NumPy helpers for the motion-compensated rolling map."""

from collections import deque

import numpy as np


def highest_point_per_cell(points, size_x, size_y, resolution):
    """Return the highest actual point in every endpoint-inclusive XY cell."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)

    cols = int(round(float(size_x) / float(resolution))) + 1
    rows = int(round(float(size_y) / float(resolution))) + 1
    col_indices = np.rint(
        (points[:, 0] + float(size_x) / 2.0) / float(resolution)
    ).astype(np.int64)
    row_indices = np.rint(
        (points[:, 1] + float(size_y) / 2.0) / float(resolution)
    ).astype(np.int64)
    inside = (
        np.isfinite(points).all(axis=1)
        & (col_indices >= 0)
        & (col_indices < cols)
        & (row_indices >= 0)
        & (row_indices < rows)
    )
    selected = points[inside]
    if not len(selected):
        return np.empty((0, 3), dtype=np.float64)

    flat_indices = row_indices[inside] * cols + col_indices[inside]
    # lexsort uses the final key as the primary key.  Therefore each cell is
    # contiguous and its final entry has the largest z value.
    order = np.lexsort((selected[:, 2], flat_indices))
    sorted_cells = flat_indices[order]
    take = np.empty(len(order), dtype=bool)
    take[:-1] = sorted_cells[:-1] != sorted_cells[1:]
    take[-1] = True
    return selected[order[take]].copy()


class TimedPointHistory:
    """Bound a deque of world-frame point packets by source time and count."""

    def __init__(self, duration_seconds, max_frames):
        self.duration_seconds = float(duration_seconds)
        self.max_frames = int(max_frames)
        if self.duration_seconds <= 0.0:
            raise ValueError('duration_seconds must be positive')
        if self.max_frames < 1:
            raise ValueError('max_frames must be at least one')
        self._packets = deque()
        self._last_stamp = None

    def __len__(self):
        return len(self._packets)

    @property
    def point_count(self):
        return sum(len(points) for _, points in self._packets)

    @property
    def span_seconds(self):
        if len(self._packets) < 2:
            return 0.0
        return float(self._packets[-1][0] - self._packets[0][0])

    def clear(self):
        """Discard every packet and reset timestamp ordering."""
        self._packets.clear()
        self._last_stamp = None

    def push(self, stamp, points_world):
        """Append one packet; return True when a timestamp rewind reset it."""
        stamp = float(stamp)
        points_world = np.asarray(points_world, dtype=np.float64)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError('points_world must have shape (N, 3)')

        rewound = (
            self._last_stamp is not None
            and stamp < self._last_stamp - 1.0e-6
        )
        if rewound:
            self.clear()

        self._packets.append((stamp, points_world.copy()))
        self._last_stamp = stamp
        cutoff = stamp - self.duration_seconds
        while self._packets and self._packets[0][0] < cutoff:
            self._packets.popleft()
        while len(self._packets) > self.max_frames:
            self._packets.popleft()
        return rewound

    def points_and_ages(self, current_stamp):
        """Return all stored points and a source-time age for every point."""
        nonempty = [packet for packet in self._packets if len(packet[1])]
        if not nonempty:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        points = np.concatenate([packet[1] for packet in nonempty], axis=0)
        ages = np.concatenate(
            [
                np.full(
                    len(packet_points),
                    max(0.0, float(current_stamp) - packet_stamp),
                    dtype=np.float64,
                )
                for packet_stamp, packet_points in nonempty
            ]
        )
        return points, ages
