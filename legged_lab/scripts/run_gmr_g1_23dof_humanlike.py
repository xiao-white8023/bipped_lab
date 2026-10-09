#!/usr/bin/env python3
"""Run GMR's SMPL-X retargeter with a human-like G1 23-DoF IK preset.

This wrapper does not modify the GMR checkout.  It derives a temporary IK
configuration from the installed 23-DoF configuration, replaces the heavily
position-biased weights with the upstream G1 balance, and prevents the
one-axis G1 wrists from pulling the shoulders and elbows toward an
unreachable human wrist pose.

All remaining arguments are forwarded to GMR's ``scripts/smplx_to_robot.py``.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import tempfile
from pathlib import Path


GMR_ROOT = Path("/home/cjh/GMR")
BASE_CONFIG = (
    GMR_ROOT
    / "general_motion_retargeting"
    / "ik_configs"
    / "smplx_to_g1_23dof.json"
)


def _set_cost(table: dict, frame: str, position: float, orientation: float) -> None:
    table[frame][1] = position
    table[frame][2] = orientation


def _make_humanlike_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    stage1 = config["ik_match_table1"]
    stage2 = config["ik_match_table2"]

    # Stage 1 establishes the global pose.  These are the upstream G1 costs,
    # mapped to the corresponding 23-DoF ankle/hand body names.
    _set_cost(stage1, "pelvis", 100, 10)
    for frame in (
        "left_hip_roll_link",
        "left_knee_link",
        "right_hip_roll_link",
        "right_knee_link",
        "torso_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ):
        _set_cost(stage1, frame, 0, 10)
    for frame in ("left_ankle_roll_link", "right_ankle_roll_link"):
        _set_cost(stage1, frame, 100, 10)

    # The 23-DoF robot has only wrist roll.  A full 6-D human-wrist target is
    # underactuated and makes the solver bend the elbows to compensate.
    for frame in (
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    ):
        # Keep a numerically negligible task so GMR still constructs the
        # per-human-body offset entry required by its preprocessing code.
        _set_cost(stage1, frame, 0, 1.0e-6)

    # Stage 2 refines body positions without sacrificing torso/limb/foot
    # orientation.  Foot orientation is particularly important for a normal
    # heel-to-toe walking silhouette.
    _set_cost(stage2, "pelvis", 100, 5)
    for frame in (
        "left_hip_roll_link",
        "left_knee_link",
        "right_hip_roll_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ):
        _set_cost(stage2, frame, 10, 5)
    _set_cost(stage2, "torso_link", 0, 10)
    for frame in ("left_ankle_roll_link", "right_ankle_roll_link"):
        _set_cost(stage2, frame, 100, 50)
    for frame in (
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    ):
        _set_cost(stage2, frame, 0, 0)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    _, forwarded = parser.parse_known_args()

    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import params

    config = _make_humanlike_config()
    with tempfile.TemporaryDirectory(prefix="gmr_g1_23dof_humanlike_") as tmp_dir:
        config_path = Path(tmp_dir) / "smplx_to_g1_23dof_humanlike.json"
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)

        params.IK_CONFIG_DICT["smplx"]["unitree_g1_23dof"] = config_path
        sys.argv = [str(GMR_ROOT / "scripts" / "smplx_to_robot.py"), *forwarded]
        runpy.run_path(str(GMR_ROOT / "scripts" / "smplx_to_robot.py"), run_name="__main__")


if __name__ == "__main__":
    main()
