"""Reduce excessive inward left hip roll/yaw directly, without IK.

Uses the ending-aligned PKLs as inputs. Only left_hip_roll_joint and
left_hip_yaw_joint are modified; the standing hold and zero ankles are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from align_g1_pkl_ending import bottom_height

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml'
DIRECTORY = ROOT / 'legged_lab/envs/g1/datasets/pkl/refined'
NAMES = ('stand_to_walk', 'walk_turn_around', 'walk_turn_left', 'walk_turn_right')


def soften_inward(values, threshold_deg, retained_fraction, width_deg):
    """Smoothly compress only the negative excess past an inward threshold.

    At the threshold, value and first/second derivatives match the unchanged
    side. This avoids a sharp clipping kink or changing outward motion.
    """
    width = np.deg2rad(width_deg)
    excess = np.maximum(np.deg2rad(threshold_deg) - values, 0.)
    return values + (1. - retained_fraction) * (excess - width * np.tanh(excess / width))


def measure(model, data, motion):
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    hip = model.body('left_hip_roll_link').id
    knee = model.body('left_knee_link').id
    foot_positions, yaw, knee_y, bottom = [], [], [], []
    for position, quaternion, angles in zip(motion['root_pos'], motion['root_rot'], motion['dof_pos']):
        data.qpos[:] = np.r_[position, quaternion[[3, 0, 1, 2]], angles]
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all():
            raise ValueError('Nonfinite forward kinematics.')
        root = R.from_quat(quaternion).as_matrix()
        foot = root.T @ data.xmat[feet[0]].reshape(3, 3)
        yaw.append(np.arctan2(foot[1, 0], foot[0, 0]))
        knee_y.append(((data.xpos[knee] - data.xpos[hip]) @ root)[1])
        foot_positions.append(data.xpos[feet].copy())
        bottom.append(bottom_height(model, data, feet))
    return {'foot_positions': np.asarray(foot_positions), 'left_foot_yaw': np.asarray(yaw),
            'left_knee_lateral_from_hip': np.asarray(knee_y), 'shoe_bottoms': np.asarray(bottom)}


def refine(input_path, output_path, args, model, data):
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must be a new file.')
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    with input_path.open('rb') as stream:
        old = pickle.load(stream)
    joints = np.asarray(old['dof_pos'])
    if joints.ndim != 2 or joints.shape[1] != 23 or not np.isfinite(joints).all():
        raise ValueError('Expected finite G1 23-DoF joint angles.')
    names = [model.joint(i).name for i in range(1, model.njnt)]
    roll = names.index('left_hip_roll_joint')
    yaw = names.index('left_hip_yaw_joint')
    if (roll, yaw) != (1, 2):
        raise ValueError('Unexpected GMR G1 PKL joint order.')
    ankle = [names.index(side + '_ankle_' + axis + '_joint')
             for side in ('left', 'right') for axis in ('pitch', 'roll')]
    if np.any(joints[:, ankle] != 0):
        raise ValueError('Expected the reviewed zero-ankle motions.')
    new = dict(old)
    new_joints = joints.copy()
    new_joints[:, roll] = soften_inward(joints[:, roll], -2., args.roll_excess_scale, .5)
    new_joints[:, yaw] = soften_inward(joints[:, yaw], -6., args.yaw_excess_scale, 1.)
    new['dof_pos'] = new_joints
    unchanged = [index for index in range(23) if index not in (roll, yaw)]
    assert np.array_equal(new_joints[:, unchanged], joints[:, unchanged])
    assert np.array_equal(new['root_pos'], old['root_pos'])
    assert np.array_equal(new['root_rot'], old['root_rot'])
    limits = model.jnt_range[1:]
    violation = np.maximum(limits[[roll, yaw], 0] - new_joints[:, [roll, yaw]], 0.)
    violation += np.maximum(new_joints[:, [roll, yaw]] - limits[[roll, yaw], 1], 0.)
    if violation.max() > 1e-6:
        raise ValueError('Modified hip joints exceed model limits.')
    source_frames = int(old['ending_alignment']['source_frames'])
    hold = max(1, int(round(1.5 * float(old['fps']))))
    for key in ('root_pos', 'root_rot', 'dof_pos'):
        assert np.array_equal(new[key][-hold:], old[key][-hold:])
    before = measure(model, data, old)
    after = measure(model, data, new)
    difference = after['foot_positions'] - before['foot_positions']
    report = {
        'input': str(input_path.resolve()), 'input_sha256': source_hash,
        'output': str(output_path.resolve()), 'frames': len(joints), 'fps': float(old['fps']),
        'modified_joints': ['left_hip_roll_joint', 'left_hip_yaw_joint'],
        'inward_roll_threshold_deg': -2., 'inward_yaw_threshold_deg': -6.,
        'inward_roll_excess_retained_fraction': args.roll_excess_scale,
        'inward_yaw_excess_retained_fraction': args.yaw_excess_scale,
        'hip_roll_min_deg_before_after': np.rad2deg([joints[:source_frames, roll].min(), new_joints[:source_frames, roll].min()]).tolist(),
        'hip_yaw_min_deg_before_after': np.rad2deg([joints[:source_frames, yaw].min(), new_joints[:source_frames, yaw].min()]).tolist(),
        'left_foot_yaw_relative_pelvis_min_deg_before_after': np.rad2deg([
            before['left_foot_yaw'][:source_frames].min(), after['left_foot_yaw'][:source_frames].min()]).tolist(),
        'left_knee_lateral_from_hip_min_m_before_after': [
            float(before['left_knee_lateral_from_hip'][:source_frames].min()),
            float(after['left_knee_lateral_from_hip'][:source_frames].min())],
        'maximum_left_ankle_position_change_m': float(np.linalg.norm(difference[:, 0], axis=1).max()),
        'right_ankle_positions_unchanged': bool(np.array_equal(before['foot_positions'][:, 1], after['foot_positions'][:, 1])),
        'minimum_shoe_height_LR_m_before': before['shoe_bottoms'].min(0).tolist(),
        'minimum_shoe_height_LR_m_after': after['shoe_bottoms'].min(0).tolist(),
        'maximum_modified_joint_frame_step_deg_before_after': np.rad2deg([
            np.abs(np.diff(joints[:, [roll, yaw]], axis=0)).max(),
            np.abs(np.diff(new_joints[:, [roll, yaw]], axis=0)).max()]).tolist(),
        'root_pose_unchanged': True, 'other_21_joints_unchanged': True, 'ankle_angles_all_zero': True,
        'standing_hold_unchanged': True, 'modified_joint_limit_violation_rad': float(violation.max()),
        'ik_used': False,
        'note': 'Direct smooth joint-angle attenuation. Foot paths change slightly; contacts are not re-solved. Kinematic preview only.'}
    new['left_leg_refinement'] = report
    new['ending_alignment'] = dict(old['ending_alignment'], source_samples_unchanged=False,
        source_sample_changes=report['modified_joints'], standing_hold_unchanged=True)
    new['motion_name'] = output_path.stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as stream:
        pickle.dump(new, stream)
    output_path.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == source_hash
    print(f'{output_path.name}: left foot inward yaw {report["left_foot_yaw_relative_pelvis_min_deg_before_after"]}; '
          f'left ankle displacement {report["maximum_left_ankle_position_change_m"]:.4f}m; standing hold unchanged.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, nargs='+',
                        default=[DIRECTORY / (name + '_end_aligned.pkl') for name in NAMES])
    parser.add_argument('--output-dir', type=Path, default=DIRECTORY)
    parser.add_argument('--roll-excess-scale', type=float, default=.35,
                        help='Fraction of large inward roll to retain; smaller means less inward roll.')
    parser.add_argument('--yaw-excess-scale', type=float, default=.55,
                        help='Fraction of large inward yaw to retain; smaller means less inward yaw.')
    args = parser.parse_args()
    if any(not np.isfinite(value) or not 0. <= value <= 1.
           for value in (args.roll_excess_scale, args.yaw_excess_scale)):
        parser.error('Excess scales must be finite and between 0 and 1.')
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    for path in args.motion:
        name = path.stem.removesuffix('_end_aligned')
        refine(path, args.output_dir / (name + '_leg_v1.pkl'), args, model, data)


if __name__ == '__main__':
    main()
