"""Append a parallel-foot standing ending without IK or changing source frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml'
DEFAULT_INPUT = ROOT / 'legged_lab/envs/g1/datasets/pkl/walk_turn_left.pkl'


def bottom_height(model, data, feet):
    """Lowest rendered shoe or spherical foot collider in the written pose."""
    heights = []
    for body in feet:
        samples = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                samples.append(data.geom_xpos[geom, 2] - model.geom_size[geom, 0])
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                samples.append((model.mesh_vert[first:first + count] @ data.geom_xmat[geom].reshape(3, 3)[2]
                                + data.geom_xpos[geom, 2]).min())
        heights.append(min(samples))
    return np.asarray(heights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path,
                        default=DEFAULT_INPUT.parent / 'refined/walk_turn_left_end_aligned.pkl')
    parser.add_argument('--transition-duration', type=float, default=.4,
                        help='Short joint-pose closure after the source clip; no walking deceleration.')
    parser.add_argument('--hold-duration', type=float, default=1.5)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Keep the original: output must be a different file.')
    if (not np.isfinite(args.transition_duration) or args.transition_duration < 0
            or not np.isfinite(args.hold_duration) or args.hold_duration <= 0):
        parser.error('Durations must be finite; transition >=0 and hold >0.')
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open('rb') as stream:
        clip = pickle.load(stream)
    fps = float(clip['fps'])
    positions = np.asarray(clip['root_pos'])
    rotations = np.asarray(clip['root_rot'])
    angles = np.asarray(clip['dof_pos'])
    if (not np.isfinite(fps) or fps <= 0 or len(angles) < 2 or angles.shape[1:] != (23,)
            or positions.shape != (len(angles), 3) or rotations.shape != (len(angles), 4)):
        raise ValueError('Expected G1 23-DoF root XYZ, root XYZW and joint-angle arrays.')
    old = np.c_[positions, rotations[:, [3, 0, 1, 2]], angles].astype(float)
    if not np.isfinite(old).all() or not np.allclose(np.linalg.norm(rotations, axis=1), 1., atol=1e-4):
        raise ValueError('Expected finite values and normalized root quaternions.')
    ankle_indices = np.array([4, 5, 10, 11])
    if not np.array_equal(angles[:, ankle_indices], np.zeros((len(angles), 4))):
        raise ValueError('This ending preserves the source zero-ankle convention.')
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    if model.nq != 30 or model.njnt != 24:
        raise ValueError('Expected the G1 floating root plus 23 scalar joints.')
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    last = old[-1].copy()
    stand = last.copy()
    heading = R.from_quat(last[[4, 5, 6, 3]]).as_euler('ZYX')[0]
    stand[3:7] = R.from_euler('z', heading).as_quat(scalar_first=True)
    # Symmetric, slightly flexed knees. Opposed hip/knee pitch gives flat soles
    # with ankle pitch/roll fixed at zero; root/arm XY and arm angles are retained.
    stand[7:19] = 0.
    stand[[7, 13]] = -.12
    stand[[10, 16]] = .12
    data.qpos[:] = stand
    mujoco.mj_forward(model, data)
    stand[2] += .001 - bottom_height(model, data, feet).min()
    transition_count = int(round(args.transition_duration * fps))
    appended = []
    minimum_closure_height = np.inf
    if transition_count:
        rotation_path = Slerp([0., 1.], R.from_quat(np.stack([last[[4, 5, 6, 3]], stand[[4, 5, 6, 3]]])))
        for u in np.linspace(0., 1., transition_count + 1)[1:]:
            blend = u ** 3 * (10 - 15 * u + 6 * u ** 2)
            row = (1 - blend) * last + blend * stand
            row[3:7] = rotation_path(float(blend)).as_quat(scalar_first=True)
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            # A vertical floor offset only. There is no IK or stance-foot fitting.
            row[2] += max(0., .001 - bottom_height(model, data, feet).min())
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            minimum_closure_height = min(minimum_closure_height, float(bottom_height(model, data, feet).min()))
            appended.append(row)
        appended[-1] = stand.copy()
    hold_count = max(1, int(round(args.hold_duration * fps)))
    appended.extend([stand.copy() for _ in range(hold_count)])
    extra = np.asarray(appended)
    updated = dict(clip)
    # Concatenate the original arrays directly to preserve every source sample.
    updated.update(root_pos=np.concatenate([positions, extra[:, :3]]),
                   root_rot=np.concatenate([rotations, extra[:, [4, 5, 6, 3]]]),
                   dof_pos=np.concatenate([angles, extra[:, 7:]]),
                   motion_name=args.output.stem, playback_loop=False)
    for field, source in [('root_pos', positions), ('root_rot', rotations), ('dof_pos', angles)]:
        assert np.array_equal(updated[field][:len(old)], source)
    assert np.array_equal(updated['dof_pos'][:, ankle_indices], np.zeros((len(old) + len(extra), 4)))
    data.qpos[:] = stand
    mujoco.mj_forward(model, data)
    foot_positions = data.xpos[feet].copy()
    foot_rotations = data.xmat[feet].reshape(2, 3, 3).copy()
    separation = (foot_positions[0] - foot_positions[1]) @ R.from_euler('z', heading).as_matrix()
    if abs(separation[0]) > 1e-6 or abs(separation[2]) > 1e-6:
        raise RuntimeError('The final feet are not side by side at the same height.')
    limits = model.jnt_range[1:]
    violation = np.maximum(limits[:, 0] - extra[:, 7:], 0) + np.maximum(extra[:, 7:] - limits[:, 1], 0)
    if violation.max() > 1e-6 or bottom_height(model, data, feet).min() < 0:
        raise RuntimeError('The new ending violates joint limits or penetrates the floor.')
    report = {
        'source': str(args.input.resolve()), 'source_sha256': source_hash,
        'output': str(args.output.resolve()), 'fps': fps, 'source_frames': len(old),
        'source_samples_unchanged': True, 'all_ankle_angles_zero': True,
        'ik_used': False, 'walking_deceleration_added': False,
        'closure_duration_s': transition_count / fps, 'hold_duration_s': hold_count / fps,
        'output_frames': len(old) + len(extra), 'output_duration_s': (len(old) + len(extra) - 1) / fps,
        'final_foot_separation_forward_lateral_vertical_m': separation.tolist(),
        'final_shoe_bottom_heights_m': bottom_height(model, data, feet).tolist(),
        'final_foot_yaw_pitch_roll_deg_LR': R.from_matrix(foot_rotations).as_euler('ZYX', degrees=True).tolist(),
        'minimum_closure_shoe_height_m': float(minimum_closure_height) if transition_count else None,
        'new_joint_limit_violation_rad': float(violation.max()),
        'note': 'Only appended pose closure and static hold; original gait/torso/feet unchanged. Kinematic preview.'}
    updated['ending_alignment'] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        pickle.dump(updated, stream)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == source_hash
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
