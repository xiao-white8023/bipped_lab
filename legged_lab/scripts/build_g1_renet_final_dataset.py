#!/usr/bin/env python3
"""Package the four reviewed final motions, including ankles in RENet AMP."""
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
DATASET = REPO/'legged_lab/envs/g1/datasets/g1_23dof_locomotion_v3'
MJCF = REPO/'legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml'
URDF = MJCF.with_suffix('.urdf')
SOURCES = {
    'walk_forward_turns': ('front8_review/v8_continuity/g1_23_accad_forward_turns_60s.pkl',
                           1800, 'forward_turns', 8, 'ACCAD captured start/stop and walking style; planned stepping turns'),
    'walk_backward': ('backward_coordination_review/v2/g1_23_backward_stand_40s.pkl',
                      1200, 'backward', 2, 'Kimodo SOMA style; planned footstep sequence and coordination refinement'),
    'walk_left': ('side_gait_review/g1_23_side_left_stand_40s.pkl',
                  1200, 'left', None, 'Kimodo SOMA left side-step style; planned noncrossing footstep sequence'),
    'walk_right': ('side_gait_review/g1_23_side_right_stand_40s.pkl',
                   1200, 'right', None, 'Robot mirror of the accepted left side-step sequence'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kimodo-root', type=Path, default=Path('/home/cjh/kimodo'))
    parser.add_argument('--output-dir', type=Path, default=DATASET)
    parser.add_argument('--amp-joint-count', type=int, choices=(19,23), default=23,
                        help='23 includes ankles; 19 rebuilds the legacy non-ankle format.')
    parser.add_argument('--from-snapshots', action='store_true',
                        help='Rebuild from packaged source PKLs without the Kimodo workspace.')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    joint_count = args.amp_joint_count
    amp_dim = 2 * joint_count + 12
    amp_names = JOINT_NAMES_23 if joint_count == 23 else AMP_JOINT_NAMES
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if model.nq != 30 or names != JOINT_NAMES_23:
        raise ValueError('Target model does not match the G1 23-DoF joint contract')
    clips = {}
    for name, (relative, count, direction, revision, origin) in SOURCES.items():
        upstream = args.kimodo_root.expanduser().resolve()/'outputs'/relative
        snapshot = output/'sources'/(name+'.pkl')
        if not args.from_snapshots:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(upstream, snapshot)
            if _sha256(upstream) != _sha256(snapshot):
                raise RuntimeError(f'Source snapshot differs: {name}')
        with snapshot.open('rb') as stream:
            raw = pickle.load(stream)
        composition = raw['sequence_composition']
        if (len(raw['root_pos']) != count or float(raw['fps']) != 30. or
                composition['direction'] != direction or
                (revision is not None and composition.get('revision') != revision)):
            raise ValueError(f'Unexpected source version or timing: {name}')
        if composition['stand_start_s'] != 3 or composition['stand_end_s'] != 3:
            raise ValueError(f'Unexpected standing intervals: {name}')
        result = build_clip(model, snapshot, output/'visualization'/(name+'.txt'),
                            output/'amp'/(name+'.txt'), preserve_root_pose=True,
                            include_ankles=joint_count == 23)
        result.update(source_original=str(upstream), source_origin=origin,
                      source_revision=revision, sequence_composition=composition,
                      sample_span_s=result.pop('duration_s'),
                      playback_duration_s=count/30., motion_weight=1.,
                      normalized_sampling_probability=.25)
        for key, subdir in [('gmr_pkl', 'sources'), ('visualization_file', 'visualization'), ('amp_file', 'amp')]:
            result[key] = f'{subdir}/{name}.'+('pkl' if subdir == 'sources' else 'txt')
        clips[name] = result
        print(f'{name}: {count} frames, {count/30:g}s, AMP {amp_dim}-D / visualization 58-D', flush=True)
    manifest = {
        'dataset': output.name, 'format_version': 3 if joint_count == 23 else 2, 'curation': 'User-accepted four final motion sequences',
        'robot': 'Unitree G1 23-DoF', 'task': 'g1_renet',
        'mjcf': str(MJCF.relative_to(REPO)), 'mjcf_sha256': _sha256(MJCF),
        'urdf': str(URDF.relative_to(REPO)), 'urdf_sha256': _sha256(URDF),
        'joint_order_23': JOINT_NAMES_23, 'amp_joint_order': amp_names,
        f'amp_joint_order_{joint_count}': amp_names, 'amp_joint_count': joint_count,
        'amp_dimension': amp_dim, 'includes_ankles': joint_count == 23,
        'amp_feature_layout': 'g1_amp_joint_pos_vel_end_effectors_v1',
        'amp_end_effector_order': END_EFFECTOR_NAMES,
        'amp_layout': {'joint_pos': [0,joint_count], 'joint_vel': [joint_count,2*joint_count],
                       'left_hand_pos_b': [2*joint_count,2*joint_count+3],
                       'right_hand_pos_b': [2*joint_count+3,2*joint_count+6],
                       'left_foot_pos_b': [2*joint_count+6,2*joint_count+9],
                       'right_foot_pos_b': [2*joint_count+9,amp_dim]},
        'visualization_layout': {'root_xyz': [0,3], 'root_euler_xyz': [3,6],
                                 'joint_pos': [6,29], 'root_lin_vel_w': [29,32],
                                 'root_ang_vel_b': [32,35], 'joint_vel': [35,58]},
        'units': 'metres, radians, seconds', 'fps': 30., 'frames_total': 5400,
        'playback_duration_total_s': 180., 'training_transition_dt_s': .02,
        'sampling': 'Equal MotionWeight=1 per full clip; uniform valid start time within each clip; no wrapping.',
        'derivative': 'numpy.gradient, central differences inside each clip and second-order one-sided endpoints; no smoothing/resampling.',
        'pose_conversion': 'Source joints and root XYZ preserved; normalized XYZW quaternion converted to unwrapped XYZ Euler only for visualization. No grounding, speed scaling, cropping or new IK.',
        'feature_conversion': f'{joint_count} joint positions/velocities; MJCF FK of hand/ankle link origins relative to pelvis, rotated into pelvis frame.',
        'limitations': f'Kinematic motion references. Direction and root speed are not inputs to the {amp_dim}-D AMP frame. Physics tracking and trained-policy style are not validated by dataset conversion.',
        'clips': clips,
    }
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Wrote {output}/manifest.json', flush=True)


if __name__ == '__main__':
    main()
