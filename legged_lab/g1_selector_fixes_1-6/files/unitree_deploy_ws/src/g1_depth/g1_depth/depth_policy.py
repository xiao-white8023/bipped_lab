"""Pure NumPy helpers for the G1 RENet depth-observation contract."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


# ``training`` reproduces the training pipeline exactly: only inf/nan are
# rewritten to ``depth_max`` before the blur, and the ``[depth_min,
# depth_max]`` clip is what turns a measured zero into the near plane.
# ``max`` is the previous deployment behaviour, where every invalid sample
# (including a measured zero) was pushed to the far plane. It is kept only for
# A/B experiments on the real robot.
INVALID_DEPTH_MODES = ('training', 'max')


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Rectified pinhole intrinsics expressed in source-image pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self):
        """Reject intrinsics that cannot define a usable projection."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError('camera dimensions must be positive')
        values = np.asarray(
            [self.fx, self.fy, self.cx, self.cy], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError('camera intrinsics must be finite')
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError('camera focal lengths must be positive')


@dataclass(frozen=True)
class ProjectionMap:
    """Source-image coordinates for every retained policy camera ray."""

    source_u: np.ndarray
    source_v: np.ndarray
    inside_source: np.ndarray
    output_height: int
    output_width: int

    @property
    def coverage(self):
        """Return the fraction of target ray centers inside the source image."""
        return float(np.mean(self.inside_source))


@dataclass(frozen=True)
class DepthFrameStats:
    """Health information computed before invalid pixels are filled."""

    projection_coverage: float
    valid_ratio: float
    minimum_metres: float
    maximum_metres: float
    valid_support: np.ndarray


def decode_depth_image(
    data,
    height,
    width,
    step,
    encoding,
    is_bigendian=False,
    depth_unit_scale=0.001,
):
    """
    Decode a ROS depth-image byte buffer into metres.

    ``16UC1`` and ``mono16`` are scaled by ``depth_unit_scale``.  RealSense
    depth images conventionally use millimetres, hence the 0.001 default.
    ``32FC1`` is interpreted as metres.  Row padding and byte order are both
    honoured instead of assuming a tightly packed little-endian image.
    """
    height = int(height)
    width = int(width)
    step = int(step)
    if height <= 0 or width <= 0:
        raise ValueError('depth image dimensions must be positive')

    normalized_encoding = str(encoding).strip().upper()
    if normalized_encoding in ('16UC1', 'MONO16'):
        dtype = np.dtype('>u2' if is_bigendian else '<u2')
        scale = float(depth_unit_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError('depth_unit_scale must be finite and positive')
    elif normalized_encoding == '32FC1':
        dtype = np.dtype('>f4' if is_bigendian else '<f4')
        scale = 1.0
    else:
        raise ValueError(
            'unsupported depth encoding '
            f'{encoding!r}; expected 16UC1, mono16, or 32FC1'
        )

    packed_width = width * dtype.itemsize
    if step < packed_width:
        raise ValueError(
            f'image step {step} is smaller than packed row {packed_width}'
        )
    raw = bytes(data)
    required = step * height
    if len(raw) < required:
        raise ValueError(
            f'depth buffer has {len(raw)} bytes, expected at least {required}'
        )

    image = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=raw,
        strides=(step, dtype.itemsize),
    )
    return image.astype(np.float32) * np.float32(scale)


def build_policy_projection(
    source_intrinsics,
    target_width=64,
    target_height=36,
    horizontal_fov_degrees=89.51,
    vertical_fov_degrees=58.29,
    crop=(18, 0, 16, 16),
    sampling_mode='nearest',
):
    """
    Map the cropped Isaac/MuJoCo policy rays into a rectified image.

    The target focal lengths are derived from the full 64x36 training image.
    Cropping is then applied in target-ray space.  This is intentionally not a
    resize of the full D435i image: it preserves the angular rays seen during
    training even when the real camera profile or principal point changes.
    """
    source_intrinsics.validate()
    target_width = int(target_width)
    target_height = int(target_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError('target dimensions must be positive')
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError('horizontal FOV must lie in (0, 180) degrees')
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError('vertical FOV must lie in (0, 180) degrees')
    if len(crop) != 4:
        raise ValueError('crop must be (top, bottom, left, right)')
    top, bottom, left, right = (int(value) for value in crop)
    if min(top, bottom, left, right) < 0:
        raise ValueError('crop values must be non-negative')
    output_height = target_height - top - bottom
    output_width = target_width - left - right
    if output_height <= 0 or output_width <= 0:
        raise ValueError('crop removes the entire target image')
    if sampling_mode not in ('nearest', 'bilinear'):
        raise ValueError("sampling_mode must be 'nearest' or 'bilinear'")

    horizontal_half = math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    vertical_half = math.tan(math.radians(vertical_fov_degrees) / 2.0)
    target_fx = target_width / (2.0 * horizontal_half)
    target_fy = target_height / (2.0 * vertical_half)

    columns = np.arange(left, target_width - right, dtype=np.float64)
    rows = np.arange(top, target_height - bottom, dtype=np.float64)
    target_u, target_v = np.meshgrid(columns, rows)
    target_x = (target_u + 0.5 - target_width / 2.0) / target_fx
    target_y = (target_v + 0.5 - target_height / 2.0) / target_fy

    source_u = source_intrinsics.fx * target_x + source_intrinsics.cx
    source_v = source_intrinsics.fy * target_y + source_intrinsics.cy
    if sampling_mode == 'nearest':
        inside = (
            (source_u >= -0.5)
            & (source_u < source_intrinsics.width - 0.5)
            & (source_v >= -0.5)
            & (source_v < source_intrinsics.height - 0.5)
        )
    else:
        inside = (
            (source_u >= 0.0)
            & (source_u <= source_intrinsics.width - 1.0)
            & (source_v >= 0.0)
            & (source_v <= source_intrinsics.height - 1.0)
        )

    return ProjectionMap(
        source_u=source_u,
        source_v=source_v,
        inside_source=inside,
        output_height=output_height,
        output_width=output_width,
    )


def _fill_measured_invalids(samples, invalid_fill, invalid_mode):
    """Replace only the values the training pipeline itself replaces.

    The training pipeline rewrites ``inf``/``nan`` to ``depth_max`` and leaves
    every finite value to the blur and the ``[0.1, 2.5]`` clip, so a measured
    ``0`` (D435i "no return") must survive this step unchanged.
    """

    if invalid_mode not in INVALID_DEPTH_MODES:
        raise ValueError(
            f'invalid_depth_mode must be one of {INVALID_DEPTH_MODES}, '
            f'got {invalid_mode!r}'
        )
    finite = np.isfinite(samples)
    non_positive = finite & (samples <= 0.0)
    samples[~finite] = np.float32(invalid_fill)
    if invalid_mode == 'max':
        # Legacy deployment behaviour: every invalid sample becomes the far
        # plane. Kept only for the A/B comparison against training parity.
        samples[non_positive] = np.float32(invalid_fill)
    return samples


def _sample_nearest(depth_metres, projection, invalid_fill, invalid_mode='training'):
    output = np.full(
        (projection.output_height, projection.output_width),
        invalid_fill,
        dtype=np.float32,
    )
    source_valid = np.zeros(output.shape, dtype=bool)
    inside = projection.inside_source
    source_x = np.floor(projection.source_u[inside] + 0.5).astype(np.int64)
    source_y = np.floor(projection.source_v[inside] + 0.5).astype(np.int64)
    samples = np.array(depth_metres[source_y, source_x], dtype=np.float32, copy=True)
    valid_samples = np.isfinite(samples) & (samples > 0.0)
    # Rays outside the source image keep the far-plane fill: training has no
    # analogue because every training ray either hits or reports max_distance.
    samples = _fill_measured_invalids(samples, invalid_fill, invalid_mode)
    output[inside] = samples
    source_valid[inside] = valid_samples
    return output, source_valid.astype(np.float32)


def _sample_bilinear(depth_metres, projection, invalid_fill, invalid_mode='training'):
    safe_depth = np.asarray(depth_metres, dtype=np.float32).copy()
    raw_valid = np.isfinite(safe_depth) & (safe_depth > 0.0)
    # Only inf/nan take the fill here; measured zeros stay in the mix so the
    # blur and the clip see exactly what the training pipeline would see.
    safe_depth = _fill_measured_invalids(safe_depth, invalid_fill, invalid_mode)

    output = np.full(
        (projection.output_height, projection.output_width),
        invalid_fill,
        dtype=np.float32,
    )
    valid_support = np.zeros(output.shape, dtype=np.float32)
    inside = projection.inside_source
    u = projection.source_u[inside]
    v = projection.source_v[inside]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, depth_metres.shape[1] - 1)
    y1 = np.minimum(y0 + 1, depth_metres.shape[0] - 1)
    wx = (u - x0).astype(np.float32)
    wy = (v - y0).astype(np.float32)
    weights = (
        (1.0 - wx) * (1.0 - wy),
        wx * (1.0 - wy),
        (1.0 - wx) * wy,
        wx * wy,
    )
    neighbours = (
        safe_depth[y0, x0],
        safe_depth[y0, x1],
        safe_depth[y1, x0],
        safe_depth[y1, x1],
    )
    valid_neighbours = (
        raw_valid[y0, x0],
        raw_valid[y0, x1],
        raw_valid[y1, x0],
        raw_valid[y1, x1],
    )
    sampled = sum(weight * value for weight, value in zip(weights, neighbours))
    support = sum(
        weight * valid.astype(np.float32)
        for weight, valid in zip(weights, valid_neighbours)
    )
    output[inside] = sampled
    valid_support[inside] = support
    return output, valid_support


def gaussian_blur_3x3_sigma1(image):
    """Match torchvision GaussianBlur(3, 1.0) with reflection padding."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError('Gaussian blur input must be two-dimensional')
    # torchvision constructs this normalized one-dimensional kernel and
    # applies its outer product after one-pixel reflection padding.
    coordinates = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    kernel = np.exp(-0.5 * coordinates * coordinates).astype(np.float32)
    kernel /= np.sum(kernel, dtype=np.float32)
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(3):
        for column in range(3):
            output += (
                kernel[row]
                * kernel[column]
                * padded[
                    row:row + image.shape[0],
                    column:column + image.shape[1],
                ]
            )
    return output


def preprocess_policy_depth(
    depth_metres,
    projection,
    sampling_mode='nearest',
    depth_min=0.1,
    depth_max=2.5,
    invalid_depth_mode='training',
    blur_enable=True,
):
    """Produce one finite, normalized 18x32 RENet VP frame."""
    depth_metres = np.asarray(depth_metres, dtype=np.float32)
    if depth_metres.ndim != 2:
        raise ValueError('depth image must be two-dimensional')
    if not 0.0 <= depth_min < depth_max:
        raise ValueError('depth limits must satisfy 0 <= min < max')
    if invalid_depth_mode not in INVALID_DEPTH_MODES:
        raise ValueError(
            f'invalid_depth_mode must be one of {INVALID_DEPTH_MODES}, '
            f'got {invalid_depth_mode!r}'
        )
    if sampling_mode == 'nearest':
        sampled, valid_support = _sample_nearest(
            depth_metres, projection, np.float32(depth_max), invalid_depth_mode
        )
    elif sampling_mode == 'bilinear':
        sampled, valid_support = _sample_bilinear(
            depth_metres, projection, np.float32(depth_max), invalid_depth_mode
        )
    else:
        raise ValueError("sampling_mode must be 'nearest' or 'bilinear'")

    # The training and MuJoCo contracts both blur with Gaussian(3x3, sigma=1).
    # blur_enable=False deliberately feeds the unblurred projected frame, which
    # is a documented departure from that contract; it exists so the real-robot
    # effect of the deterministic training blur can be measured on its own.
    if blur_enable:
        sampled = gaussian_blur_3x3_sigma1(sampled)
    metric = np.clip(sampled, depth_min, depth_max).astype(np.float32)
    normalized = (metric / np.float32(depth_max)).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError('processed policy depth contains NaN or Inf')

    stats = DepthFrameStats(
        projection_coverage=projection.coverage,
        valid_ratio=float(np.mean(valid_support)),
        minimum_metres=float(np.min(metric)),
        maximum_metres=float(np.max(metric)),
        valid_support=np.asarray(valid_support, dtype=np.float32).copy(),
    )
    return normalized, stats


class PolicyDepthHistory:
    """Maintain the policy's oldest-to-newest depth history at source time."""

    def __init__(self, frame_count=2, capture_period=0.1, tolerance=1.0e-6):
        """Create a fixed-length history sampled in source timestamp time."""
        if int(frame_count) <= 0:
            raise ValueError('frame_count must be positive')
        if float(capture_period) <= 0.0:
            raise ValueError('capture_period must be positive')
        self.frame_count = int(frame_count)
        self.capture_period = float(capture_period)
        self.tolerance = float(tolerance)
        self._frames = deque(maxlen=self.frame_count)
        self._last_stamp = None
        self._next_capture_stamp = None

    @property
    def last_stamp(self):
        """Return the latest accepted source timestamp, if any."""
        return self._last_stamp

    def is_due(self, stamp):
        """Return whether ``stamp`` starts a new 10 Hz policy capture."""
        stamp = float(stamp)
        if not math.isfinite(stamp):
            return False
        if self._last_stamp is None:
            return True
        delta = stamp - self._last_stamp
        if delta < -self.tolerance:
            return True
        return stamp >= self._next_capture_stamp - self.tolerance

    def push(self, stamp, frame):
        """
        Accept a due frame and return ``(history, clock_rewound)``.

        The first frame is copied into every slot, matching the training
        environment.  Frames arriving before the next scheduled capture return
        ``(None, False)``.  Capture deadlines stay on a fixed source-time grid;
        camera-frame quantization must not accumulate into sampling drift.
        A source-time rewind clears the old time epoch.
        """
        stamp = float(stamp)
        if not math.isfinite(stamp):
            raise ValueError('source timestamp must be finite')
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError('policy depth frame must be two-dimensional')

        rewound = False
        if self._last_stamp is not None:
            delta = stamp - self._last_stamp
            if delta < -self.tolerance:
                self._frames.clear()
                self._last_stamp = None
                self._next_capture_stamp = None
                rewound = True
            elif not self.is_due(stamp):
                return None, False

        if not self._frames:
            for _ in range(self.frame_count):
                self._frames.append(frame.copy())
        else:
            if frame.shape != self._frames[-1].shape:
                raise ValueError('policy depth frame shape changed')
            self._frames.append(frame.copy())
        self._last_stamp = stamp
        if self._next_capture_stamp is None:
            self._next_capture_stamp = stamp + self.capture_period
        else:
            # Advance from the scheduled deadline, not the accepted frame.
            # At 30.08 Hz, three frames span 99.734 ms. Re-anchoring to each
            # accepted frame would always wait for a fourth (7.52 Hz output).
            # Skip expired deadlines after a gap; never emit catch-up frames.
            elapsed_periods = math.floor(
                (stamp - self._next_capture_stamp + self.tolerance)
                / self.capture_period
            )
            self._next_capture_stamp += (
                max(0, elapsed_periods) + 1
            ) * self.capture_period
        return np.stack(tuple(self._frames), axis=0), rewound
