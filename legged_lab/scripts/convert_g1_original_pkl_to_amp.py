#!/usr/bin/env python3
"""Convert the four original G1 locomotion PKLs to full/non-ankle AMP files."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from build_g1_23dof_amp_dataset import (
    AMP_JOINT_NAMES, END_EFFECTOR_NAMES, JOINT_NAMES_23, _sha256, build_clip,
)

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "legged_lab/envs/g1/datasets"
MJCF = REPO / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml"
CLIPS = ("stand_to_walk", "walk_turn_around", "walk_turn_left", "walk_turn_right")
MOTION_WEIGHT = 0.5  # Preserve the existing files' weight in shared AMP mixtures.


def load_original_motion(path: Path) -> tuple[dict, dict]:
    """Read flat GMR or a single nested motion; both use XYZW quaternions."""
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a motion dictionary")
    if {"root_pos", "root_rot", "dof_pos"}.issubset(raw):
        motion = raw
        metadata = {"source_layout": "gmr_flat"}
    else:
        candidates = [(key, value) for key, value in raw.items()
                      if isinstance(value, dict)
                      and {"root_trans_offset", "root_rot", "dof"}.issubset(value)]
        if len(candidates) != 1:
            raise ValueError(f"{path}: expected exactly one supported nested motion")
        key, nested = candidates[0]
        motion = {"root_pos": nested["root_trans_offset"],
                  "root_rot": nested["root_rot"], "dof_pos": nested["dof"],
                  "fps": nested.get("fps")}
        metadata = {"source_layout": "nested", "nested_motion_key": str(key)}
    fps = np.asarray(motion.get("fps"))
    if fps.size != 1 or fps.dtype == object:
        raise ValueError(f"{path}: missing or invalid source fps")
    metadata["source_quaternion_order"] = "xyzw"
    return {"fps": float(fps.reshape(-1)[0]),
            **{key: np.asarray(motion[key], dtype=np.float64)
               for key in ("root_pos", "root_rot", "dof_pos")}}, metadata


def _manifest(output: Path, clips: dict, include_ankles: bool) -> dict:
    names = JOINT_NAMES_23 if include_ankles else AMP_JOINT_NAMES
    count = len(names)
    return {
        "dataset": output.name, "format_version": 1,
        "robot": "Unitree G1 23-DoF", "task": "g1_rough",
        "source_selection": "Only stand_to_walk, walk_turn_around, walk_turn_left, walk_turn_right; original PKLs, no *_v1 variants",
        "mjcf": os.path.relpath(MJCF, output), "mjcf_sha256": _sha256(MJCF),
        "urdf": os.path.relpath(MJCF.with_suffix('.urdf'), output),
        "urdf_sha256": _sha256(MJCF.with_suffix('.urdf')),
        "joint_order_23": JOINT_NAMES_23, "amp_joint_order": names,
        "amp_joint_count": count, "amp_dimension": 2 * count + 12,
        "includes_ankles": include_ankles,
        "amp_feature_layout": "g1_amp_joint_pos_vel_end_effectors_v1",
        "amp_end_effector_order": END_EFFECTOR_NAMES,
        "amp_layout": {"joint_pos": [0, count], "joint_vel": [count, 2 * count],
                       "end_effector_pos_b": [2 * count, 2 * count + 12]},
        "visualization_layout": {"root_xyz": [0, 3], "root_euler_xyz": [3, 6],
                                 "joint_pos": [6, 29], "root_lin_vel_w": [29, 32],
                                 "root_ang_vel_b": [32, 35], "joint_vel": [35, 58]},
        "frames_total": sum(meta['frames'] for meta in clips.values()),
        "sample_span_total_s": sum(meta['duration_s'] for meta in clips.values()),
        "training_transition_dt_s": .02, "units": "metres, radians, seconds",
        "sampling": "Equal MotionWeight=0.5 per clip; normalized probability 25% each; Clamp, no wrapping.",
        "derivative": "Central differences; second-order one-sided endpoints at source FPS.",
        "pose_conversion": "All source frames/poses/FPS preserved; no resampling, cropping, grounding, smoothing, appended holds or IK.",
        "feature_conversion": "Full 23-joint FK of hand/ankle link origins in pelvis coordinates; non-ankle format removes only ankle joint position/velocity columns.",
        "clips": clips,
    }


def _readme(manifest: dict) -> str:
    count = manifest['amp_joint_count']
    lines = [f"# 原始 G1 PKL 的 {manifest['amp_dimension']} 维 AMP 数据", "",
             "仅由 `pkl/` 中原始起步、掉头、左转、右转四个 PKL 转换，保留全部源帧和帧率，不使用 `_v1` 版本。", "",
             f"特征：{count} 维关节角 + {count} 维关节速度 + 12 维 pelvis 坐标系手/脚踝 link 位置。",
             "58 维 AMP 与 58 维可视化文件的字段含义不同；可视化文件位于 `../motion_visualization/original_pkl/`。", "",
             "| 片段 | 帧数 | FPS |", "|---|---:|---:|"]
    lines.extend(f"| `{name}` | {meta['frames']} | {meta['fps']:.6f} |"
                 for name, meta in manifest['clips'].items())
    lines.extend(["", "保留各段 MotionWeight=0.5，四段归一化采样概率各 25%；LoopMode=Clamp，不将片段首尾拼接。",
                  "训练加载器按源帧率插值为相隔 20 ms 的状态对。来源、文件哈希、关节顺序和模型记录在 `manifest.json`。",
                  "旧的其他 TXT 可能仍在此目录，g1_rough 仅使用 manifest 中明确列出的这四段。", ""])
    return "\n".join(lines)


def convert(input_dir: Path, full_dir: Path, no_ankle_dir: Path,
            visualization_dir: Path, backup_dir: Path | None = None) -> dict:
    sources = [input_dir / (name + ".pkl") for name in CLIPS]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"Missing original PKLs: {missing}")
    outputs = {"full": full_dir, "no_ankle": no_ankle_dir,
               "visualization": visualization_dir}
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Output directories must be distinct")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    if model.nq != 30 or [model.joint(i).name for i in range(1, model.njnt)] != JOINT_NAMES_23:
        raise ValueError("MJCF does not match the G1 23-DoF joint order")
    source_hashes = {p: _sha256(p) for p in sources}
    clips = {"full": {}, "no_ankle": {}}
    # Validate and generate every clip before replacing any live training file.
    with tempfile.TemporaryDirectory(prefix="g1_original_amp_") as scratch:
        stage = Path(scratch)
        for source in sources:
            motion, metadata = load_original_motion(source)
            canonical = stage / (source.stem + ".pkl")
            with canonical.open("wb") as stream:
                pickle.dump(motion, stream)
            for group, include_ankles in (("full", True), ("no_ankle", False)):
                preview = stage / "visualization" / (source.stem + ".txt")
                amp = stage / group / (source.stem + ".txt")
                metrics = build_clip(model, canonical, preview, amp,
                                     preserve_root_pose=True, include_ankles=include_ankles,
                                     motion_weight=MOTION_WEIGHT)
                metrics.update(metadata)
                metrics.update(gmr_pkl=os.path.relpath(source, outputs[group]),
                               gmr_pkl_sha256=source_hashes[source],
                               amp_file=source.stem + ".txt",
                               visualization_file=os.path.relpath(
                                   visualization_dir / preview.name, outputs[group]),
                               motion_weight=MOTION_WEIGHT, normalized_sampling_probability=1 / len(sources))
                clips[group][source.stem] = metrics
            print(f"{source.name}: {len(motion['dof_pos'])} frames, {motion['fps']:.6f} FPS, AMP 58/50-D", flush=True)
        manifests = {group: _manifest(outputs[group], clips[group], group == "full")
                     for group in clips}
        for group, manifest in manifests.items():
            (stage / group / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            (stage / group / "README.md").write_text(_readme(manifest))
        if any(_sha256(p) != digest for p, digest in source_hashes.items()):
            raise RuntimeError("An input PKL changed during conversion")
        files = [(p, outputs[group] / p.name) for group in outputs
                 for p in sorted((stage / group).iterdir())]
        replacements = [(src, dst) for src, dst in files if dst.exists()
                        and src.read_bytes() != dst.read_bytes()]
        if replacements and backup_dir is None:
            raise ValueError("Existing outputs differ; supply --backup-dir before overwriting them")
        for _, dst in replacements:
            saved = backup_dir / dst.parent.name / dst.name
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                if saved.read_bytes() != dst.read_bytes():
                    raise ValueError(f"{saved}: backup already exists for a different version")
            else:
                shutil.copy2(dst, saved)
        for src, dst in files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or src.read_bytes() != dst.read_bytes():
                shutil.copy2(src, dst)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATASETS / "pkl")
    parser.add_argument("--full-dir", type=Path, default=DATASETS / "motion_amp_expert")
    parser.add_argument("--no-ankle-dir", type=Path, default=DATASETS / "motion_amp_expert_no_ankle")
    parser.add_argument("--visualization-dir", type=Path, default=DATASETS / "motion_visualization/original_pkl")
    parser.add_argument("--backup-dir", type=Path,
                        help="Required if existing output files would be replaced")
    args = parser.parse_args()
    convert(*(getattr(args, name).expanduser().resolve()
              for name in ("input_dir", "full_dir", "no_ankle_dir", "visualization_dir")),
            backup_dir=args.backup_dir.expanduser().resolve() if args.backup_dir else None)


if __name__ == "__main__":
    main()
