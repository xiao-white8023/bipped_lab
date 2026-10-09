#!/usr/bin/env python3
"""Build MuJoCo-only candidates from reviewed captured/style-guided G1 motions."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / 'legged_lab/envs/g1/datasets/g1_23dof_locomotion_v3/sources'
MODEL = REPO / 'legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml'


def save_candidate(path, clip, direction, speed, provenance):
    clip['motion_name'] = f'G1_{direction}_{speed:.2f}mps_review'
    clip['multispeed_review'] = {'status': 'awaiting_user_review', 'nominal_speed_mps': speed,
                                'direction': direction, 'method': provenance,
                                'integration': 'MuJoCo kinematic preview only; AMP configuration not changed'}
    clip['sequence_composition'].update(nominal_speed_mps=speed, candidate_for_review=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        pickle.dump(clip, stream)
    q = np.c_[clip['root_pos'], clip['root_rot'][:, [3, 0, 1, 2]], clip['dof_pos']].astype(float)
    fps = float(clip['fps'])
    velocities = np.gradient(q[:, 7:], 1/fps, axis=0, edge_order=2)
    root_velocity = np.gradient(q[:, :3], 1/fps, axis=0, edge_order=2)
    np.savez_compressed(path.with_suffix('.npz'), fps=fps, root_pos=clip['root_pos'],
                        root_rot_xyzw=clip['root_rot'], joint_pos=clip['dof_pos'],
                        joint_vel=velocities, root_lin_vel_w=root_velocity)
    path.with_suffix('.sequence.json').write_text(json.dumps(clip['sequence_composition'], indent=2)+'\n')
    return {'name': path.stem, 'motion': str(path), 'direction': direction, 'nominal_speed_mps': speed,
            'duration_s': len(q)/fps, 'fps': fps, 'frames': len(q), 'status': 'awaiting_user_review',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def command(args, name, script, parameters):
    log = args.output / 'logs' / (name+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Building', name, flush=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable, str(args.kimodo/script), *map(str, parameters)],
                       cwd=args.kimodo, stdout=stream, stderr=subprocess.STDOUT, check=True)


def backward(args):
    result = []
    source = args.kimodo / 'outputs/natural_gait_review/g1_23_backward_045mps_01_refined_v2.pkl'
    for speed, cycle in ((.2, 1.60), (.4, 1.30), (.6, 1.08)):
        name = f'backward_{speed:.2f}'.replace('.', 'p')
        raw = args.output/'intermediate'/(name+'_plan.pkl')
        output = args.output/'motions'/(name+'.pkl')
        command(args, name+'_plan', Path('compose_g1_23_stand_walk.py'), [
            '--input', source, '--output', raw, '--duration', 40, '--stand-start', 3, '--stand-end', 3,
            '--target-speed', speed, '--cycle-duration', cycle, '--swing-duration', .44*cycle,
            '--walking-height-offset', -.01 if speed < .6 else -.022])
        command(args, name+'_coordination', Path('local_setup/refine_backward_coordination.py'),
                ['--input', raw, '--output', output])
        with output.open('rb') as stream:
            clip = pickle.load(stream)
        result.append(save_candidate(output, clip, 'backward', speed,
            'Reviewed SOMA gait: new cadence/step-length plan, fixed stance feet, leg IK, step-coupled opposed arms'))
        print('Saved', name, flush=True)
    return result


def side(args):
    result = []
    source = args.kimodo/'outputs/side_gait_review/g1_23_left_01_refined.pkl'
    human = args.kimodo/'outputs/side_gait_review/human_left_03mps/human_left_03mps_01.npz'
    for speed, cycle in ((.15, 1.45), (.3, 1.18), (.5, .90)):
        swing = .42*cycle
        # Account for integer cycle count so the steady planned average matches
        # the requested speed, while preserving the exact 40-second duration.
        first, last = 3.2, 37-swing-.5
        pairs = max(2, int(round(((last-first)/(cycle/2)+1)/2)))
        actual_cycle = 2*(last-first)/(pairs*2-1)
        paths = {s: args.output/'motions'/('side_'+s+'_'+f'{speed:.2f}'.replace('.', 'p')+'.pkl')
                 for s in ('left', 'right')}
        command(args, 'side_'+f'{speed:.2f}'.replace('.', 'p'), Path('compose_g1_23_side_walk.py'), [
            '--input', source, '--source-human', human, '--standing-reference', SOURCES/'walk_backward.pkl',
            '--output-left', paths['left'], '--output-right', paths['right'], '--duration', 40,
            '--step-size', speed*actual_cycle, '--cycle-duration', cycle, '--swing-duration', swing,
            '--clearance-m', .04 if speed < .5 else .045,
            '--body-lowering', .015 if speed < .3 else (.023 if speed < .5 else .034),
            '--weight-shift', .018])
        for direction, output in paths.items():
            with output.open('rb') as stream:
                clip = pickle.load(stream)
            result.append(save_candidate(output, clip, direction, speed,
                'Reviewed SOMA swing shape: new leading/closing step plan and cadence, planted-foot IK; right is robot mirror'))
            print('Saved', output.stem, flush=True)
    return result


def forward(args):
    sys.path.insert(0, str(args.kimodo))
    from local_setup import build_forward_v8 as v8
    def orient(q, direction):
        velocity = np.diff(q[:, :2], axis=0)*30
        selected = np.linalg.norm(velocity, axis=1) > .05
        heading = np.arctan2(*velocity[selected][:30].mean(0)[::-1])
        rotation = R.from_euler('z', direction-heading)
        out = q.copy()
        out[:, :3] = rotation.apply(out[:, :3])
        out[:, 3:7] = (rotation*R.from_quat(out[:, [4, 5, 6, 3]])).as_quat(scalar_first=True)
        return out

    v8.orient = orient
    parent = args.kimodo/'outputs/front8_review/v8_continuity'

    class SpeedBuilder(v8.Builder):
        def __init__(self, output, speed):
            super().__init__(output)
            self.speed = speed
            self.source_template = super().template()

        def template(self):
            clip, q, relative, angles, lift, source_speed = self.source_template
            ratio = self.speed/source_speed
            q, relative, angles, lift = [value.copy() for value in (q, relative, angles, lift)]
            relative[:, :, 0] *= ratio
            style_scale = np.clip(np.sqrt(ratio), .55, 1.1)
            q[:, 20:] = q[:, 20:].mean(0)+style_scale*(q[:, 20:]-q[:, 20:].mean(0))
            q[:, 2] = q[:, 2].mean()+style_scale*(q[:, 2]-q[:, 2].mean())
            angles[:, :, 1:] *= style_scale
            lift *= style_scale
            return clip, q, relative, angles, lift, self.speed

        def cap(self, name):
            clip, q = v8.load(parent/(name+'.pkl'))
            q = orient(q, 0.)
            positions, rotations, _ = self.fk(q)
            ratio = self.speed/self.source_template[-1]
            q[:, 0] = q[0, 0]+ratio*(q[:, 0]-q[0, 0])
            for k in range(2):
                positions[:, k, 0] = positions[0, k, 0]+ratio*(positions[:, k, 0]-positions[0, k, 0])
            q = self.refit(q, positions, rotations, name)
            return clip, q

    result = []
    for speed in (.3, .6, 1.):
        name = 'forward_'+f'{speed:.2f}'.replace('.', 'p')
        output_dir = args.output/'intermediate'/name
        output_dir.mkdir(parents=True, exist_ok=True)
        builder = SpeedBuilder(output_dir, speed)
        clips, motions = {}, {}
        for key in ('start', 'stop'):
            clips[key], motions[key] = builder.cap(key)
            builder.save(key, clips[key], motions[key])
        for key, count, angle in [('walk', 430, 0.), ('left', 213, np.pi/2),
                                  ('right', 233, -np.pi/2), ('around', 319, np.pi)]:
            clips[key], motions[key] = builder.stepping_curve(key, count, angle)
            builder.save(key, clips[key], motions[key])
        builder.compose(clips, motions)
        with (output_dir/'g1_23_accad_forward_turns_60s.pkl').open('rb') as stream:
            clip = pickle.load(stream)
        output = args.output/'motions'/(name+'.pkl')
        result.append(save_candidate(output, clip, 'forward_turns', speed,
            'Reviewed ACCAD v8 stride/sole-roll/caps: speed-dependent landing geometry, captured cadence, leg/root IK and gait-phase joins'))
        print('Saved', name, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kimodo', type=Path, default=Path('/home/cjh/kimodo'))
    parser.add_argument('--output', type=Path, default=Path('/home/cjh/kimodo/outputs/multispeed_review_20261003'))
    parser.add_argument('--group', choices=['backward', 'side', 'forward'], required=True)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.kimodo = args.kimodo.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    clips = {'backward': backward, 'side': side, 'forward': forward}[args.group](args)
    (args.output/f'clips_{args.group}.json').write_text(json.dumps(clips, indent=2)+'\n')
    print('MULTISPEED_GROUP_PASS', args.group, flush=True)


if __name__ == '__main__':
    main()
