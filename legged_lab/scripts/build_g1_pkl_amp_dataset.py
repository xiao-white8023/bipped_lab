#!/usr/bin/env python3
"""Package the four revised G1 PKLs as 50-D AMP, without changing their poses."""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import mujoco

from build_g1_23dof_amp_dataset import (
    AMP_JOINT_NAMES, END_EFFECTOR_NAMES, JOINT_NAMES_23, _sha256, build_clip,
)

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "legged_lab/envs/g1/datasets/pkl"
DATASET = INPUT.parent / "g1_23dof_pkl_locomotion_v1"
MJCF = REPO / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml"
CLIPS = (
    "stand_to_walk_leg_v1", "walk_turn_around_leg_v1",
    "walk_turn_left_leg_v1", "walk_turn_right_leg_v1",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=DATASET)
    parser.add_argument("--from-snapshots", action="store_true",
                        help="Rebuild using output-dir/sources instead of upstream PKLs.")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    source_dir = output / "sources" if args.from_snapshots else args.input_dir.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    if model.nq != 30 or [model.joint(i).name for i in range(1, model.njnt)] != JOINT_NAMES_23:
        raise ValueError("MJCF does not match the G1 23-DoF joint order")
    clips = {}
    for name in CLIPS:
        source = source_dir / (name + ".pkl")
        snapshot = output / "sources" / source.name
        if source != snapshot:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
        if _sha256(source) != _sha256(snapshot):
            raise RuntimeError(f"Source snapshot differs: {name}")
        sidecar = source.with_suffix(".json")
        if sidecar.exists() and sidecar != snapshot.with_suffix(".json"):
            shutil.copy2(sidecar, snapshot.with_suffix(".json"))
        with snapshot.open("rb") as stream:
            motion = pickle.load(stream)
        metrics = build_clip(model, snapshot, output / "visualization" / (name + ".txt"),
                             output / "amp" / (name + ".txt"), preserve_root_pose=True)
        metrics.update(source_original=str(source),
                       sample_span_s=metrics.pop("duration_s"),
                       ending_alignment=motion.get("ending_alignment"),
                       left_leg_refinement=motion.get("left_leg_refinement"),
                       motion_weight=1.0, normalized_sampling_probability=0.25)
        for key, folder in (("gmr_pkl", "sources"), ("amp_file", "amp"),
                            ("visualization_file", "visualization")):
            metrics[key] = f"{folder}/{name}." + ("pkl" if folder == "sources" else "txt")
        clips[name] = metrics
        print(f"{name}: {metrics['frames']} frames, {metrics['fps']:.6f} fps, "
              f"{metrics['sample_span_s']:.3f}s, AMP 50-D / preview 58-D", flush=True)
    manifest = {
        "dataset": output.name, "format_version": 2, "robot": "Unitree G1 23-DoF",
        "task": "g1_renet", "curation": "Four revised PKLs; no backward/lateral clips",
        "mjcf": str(MJCF.relative_to(REPO)), "mjcf_sha256": _sha256(MJCF),
        "urdf": str(MJCF.with_suffix('.urdf').relative_to(REPO)),
        "urdf_sha256": _sha256(MJCF.with_suffix('.urdf')),
        "joint_order_23": JOINT_NAMES_23, "amp_joint_order": AMP_JOINT_NAMES,
        "amp_joint_order_19": AMP_JOINT_NAMES, "amp_joint_count": 19,
        "amp_dimension": 50, "includes_ankles": False,
        "amp_feature_layout": "g1_amp_joint_pos_vel_end_effectors_v1",
        "amp_end_effector_order": END_EFFECTOR_NAMES,
        "amp_layout": {"joint_pos": [0, 19], "joint_vel": [19, 38],
                       "left_hand_pos_b": [38, 41], "right_hand_pos_b": [41, 44],
                       "left_foot_pos_b": [44, 47], "right_foot_pos_b": [47, 50]},
        "visualization_layout": {"root_xyz": [0, 3], "root_euler_xyz": [3, 6],
                                 "joint_pos": [6, 29], "root_lin_vel_w": [29, 32],
                                 "root_ang_vel_b": [32, 35], "joint_vel": [35, 58]},
        "units": "metres, radians, seconds", "fps": "Per-clip source FPS preserved",
        "frames_total": sum(m['frames'] for m in clips.values()),
        "sample_span_total_s": sum(m['sample_span_s'] for m in clips.values()),
        "training_transition_dt_s": 0.02,
        "sampling": "MotionWeight=1 per clip; uniform valid start time; Clamp, no wrapping.",
        "derivative": "Central finite differences and second-order one-sided endpoints at each source FPS.",
        "pose_conversion": "No resampling, cropping, root height correction, speed scaling, smoothing or new IK. Source joint/root poses and standing endings preserved.",
        "feature_conversion": "19 non-ankle joint positions/velocities; full 23-joint FK for hand/ankle link origins in pelvis frame.",
        "limitations": "Kinematic references. Conversion validates training features, not physical tracking. Root travel direction is not part of 50-D AMP.",
        "clips": clips,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
