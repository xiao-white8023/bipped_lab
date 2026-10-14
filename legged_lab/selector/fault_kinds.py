"""Stable fault IDs; new collection families append to existing dataset IDs."""

from enum import IntEnum


class FaultKind(IntEnum):
    NONE = 0
    FREEZE = 1
    DELAY = 2
    VP_OCCLUSION = 3
    VP_SCALE = 4
    VP_BIAS = 5
    VP_SHIFT = 6
    VP_COLLAPSE = 7
    LP_RANDOM_DROPOUT = 8
    LP_BLOCK_DROPOUT = 9
    LP_HEIGHT_BIAS = 10
    LP_REGISTRATION = 11
    LP_DRIFT = 12
    LP_COLLAPSE = 13
    LP_MAP_DRIFT = 14
    LP_PERSON = 15
    LP_COMPOUND = 16
    VP_COMPOUND = 17
    VP_BURST = 18
    VP_LOCAL_DROPOUT = 19
    VP_EDGE_MISMATCH = 20
    VP_SPATIAL_BIAS = 21
    VP_MOTION_WARP = 22
    VP_TIME_JITTER = 23
    VP_FILTER_TRAIL = 24
    VP_DYNAMIC_OCCLUSION = 25


VP_FAULTS = (
    FaultKind.FREEZE, FaultKind.DELAY, FaultKind.VP_OCCLUSION,
    FaultKind.VP_SCALE, FaultKind.VP_BIAS, FaultKind.VP_SHIFT, FaultKind.VP_COLLAPSE,
)
LP_FAULTS = (
    FaultKind.FREEZE, FaultKind.DELAY, FaultKind.LP_RANDOM_DROPOUT,
    FaultKind.LP_BLOCK_DROPOUT, FaultKind.LP_HEIGHT_BIAS,
    FaultKind.LP_REGISTRATION, FaultKind.LP_DRIFT, FaultKind.LP_COLLAPSE,
)
EXTENDED_LP_FAULTS = LP_FAULTS + (
    FaultKind.LP_MAP_DRIFT, FaultKind.LP_PERSON, FaultKind.LP_COMPOUND,
)
EXTENDED_VP_FAULTS = VP_FAULTS + (FaultKind.VP_COMPOUND, FaultKind.VP_BURST)
STRUCTURED_VP_FAULTS = EXTENDED_VP_FAULTS + (
    FaultKind.VP_LOCAL_DROPOUT, FaultKind.VP_EDGE_MISMATCH,
    FaultKind.VP_SPATIAL_BIAS, FaultKind.VP_MOTION_WARP,
    FaultKind.VP_TIME_JITTER, FaultKind.VP_FILTER_TRAIL,
    FaultKind.VP_DYNAMIC_OCCLUSION,
)
