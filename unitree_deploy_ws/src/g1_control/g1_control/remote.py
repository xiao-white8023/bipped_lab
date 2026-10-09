"""Decode the 40-byte G1 wireless remote payload without controlling anything."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable


BUTTON_NAMES = (
    "R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
    "A", "B", "X", "Y", "up", "right", "down", "left",
)
BUTTON_INDEX = {name: index for index, name in enumerate(BUTTON_NAMES)}


@dataclass(frozen=True)
class RemoteState:
    keys: int
    lx: float
    ly: float
    rx: float
    ry: float

    def pressed(self, name: str) -> bool:
        return bool(self.keys & (1 << BUTTON_INDEX[name]))

    @property
    def pressed_names(self) -> tuple[str, ...]:
        return tuple(name for index, name in enumerate(BUTTON_NAMES) if self.keys & (1 << index))


def decode_remote(data: Iterable[int]) -> RemoteState:
    """Decode the Unitree little-endian wireless_remote layout."""

    payload = bytes(data)
    if len(payload) != 40:
        raise ValueError(f"wireless_remote must contain 40 bytes, got {len(payload)}")
    keys = struct.unpack_from("<H", payload, 2)[0]
    lx = struct.unpack_from("<f", payload, 4)[0]
    rx = struct.unpack_from("<f", payload, 8)[0]
    ry = struct.unpack_from("<f", payload, 12)[0]
    ly = struct.unpack_from("<f", payload, 20)[0]
    return RemoteState(keys=keys, lx=lx, ly=ly, rx=rx, ry=ry)


def rising_buttons(previous_keys: int, current_keys: int) -> tuple[str, ...]:
    rising = current_keys & ~previous_keys
    return tuple(name for index, name in enumerate(BUTTON_NAMES) if rising & (1 << index))


def map_axis_to_command_range(
    value: float,
    command_range: Iterable[float],
    *,
    axis_sign: float = 1.0,
) -> float:
    """Map one normalized stick axis onto an asymmetric command range."""

    bounds = tuple(float(bound) for bound in command_range)
    if len(bounds) != 2:
        raise ValueError(
            f"command_range must contain [minimum, maximum], got {bounds}"
        )
    lower, upper = bounds
    sign = float(axis_sign)
    if not all(math.isfinite(item) for item in (value, lower, upper, sign)):
        raise ValueError("stick value, command range, and axis sign must be finite")
    if lower > 0.0 or upper < 0.0 or lower >= upper:
        raise ValueError(
            "command range must satisfy minimum <= 0 <= maximum and minimum < maximum"
        )
    if sign not in (-1.0, 1.0):
        raise ValueError("axis_sign must be either -1 or 1")

    signed = max(-1.0, min(1.0, float(value) * sign))
    scale = upper if signed >= 0.0 else -lower
    return signed * scale
