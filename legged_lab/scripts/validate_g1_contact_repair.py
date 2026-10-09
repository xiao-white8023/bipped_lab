"""Check repaired G1 poses and the actual viewer interpolation against MJCF.

Recomputes foot mesh/sphere extrema in world coordinates and verifies static
COM projection at the standing ending. These checks do not establish dynamic
tracking or measured contact forces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from play_g1_pkl_mujoco import interpolate, load_motion, MODEL


def validate(path, model):
    fps, poses=load_motion(path)
    with path.open('rb') as stream:
        clip=pickle.load(stream)
    metadata=clip['contact_repair']
    source=Path(metadata['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError(f'{source}: source hash changed.')
    data=mujoco.MjData(model)
    feet=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    sphere_ids=[np.flatnonzero((model.geom_bodyid==body)&(model.geom_type==mujoco.mjtGeom.mjGEOM_SPHERE)) for body in feet]
    geometry=[np.flatnonzero(model.geom_bodyid==body) for body in feet]
    labels=np.asarray(clip['inferred_foot_contacts'],dtype=bool)
    if labels.shape!=(len(poses),2):
        raise ValueError('Invalid contact labels.')

    def measure(pose):
        data.qpos[:]=pose
        mujoco.mj_forward(model,data)
        bottoms=[]
        for ids in geometry:
            candidates=[]
            for geom in ids:
                if model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_SPHERE:
                    candidates.append(data.geom_xpos[geom,2]-model.geom_size[geom,0])
                elif model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=model.geom_dataid[geom]
                    a,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
                    candidates.append((model.mesh_vert[a:a+n]@data.geom_xmat[geom].reshape(3,3)[2]+data.geom_xpos[geom,2]).min())
            bottoms.append(min(candidates))
        return np.asarray(bottoms),data.xpos[feet].copy()

    sampled_min=np.inf
    positions=[]
    for pose in poses:
        bottom,xyz=measure(pose)
        sampled_min=min(sampled_min,float(bottom.min()))
        positions.append(xyz)
    positions=np.asarray(positions)
    between_min=np.inf
    between_stance_error=0.
    for i,(a,b) in enumerate(zip(poses[:-1],poses[1:])):
        stance=labels[i]&labels[i+1]
        for fraction in (.25,.5,.75):
            bottom,xyz=measure(interpolate(a,b,fraction))
            between_min=min(between_min,float(bottom.min()))
            if stance.any():
                expected=(1-fraction)*positions[i]+fraction*positions[i+1]
                between_stance_error=max(between_stance_error,float(np.linalg.norm((xyz-expected)[stance,:2],axis=1).max()))
    limits=model.jnt_range[1:]
    violation=np.maximum(limits[:,0]-poses[:,7:],0)+np.maximum(poses[:,7:]-limits[:,1],0)
    step_rate=np.abs(np.diff(poses[:,7:],axis=0))*fps
    root_step=np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1)
    hold=int(round(1.5*fps))
    hold_identical=bool(np.array_equal(poses[-hold:],np.repeat(poses[-1:],hold,axis=0)))
    measure(poses[-1])
    vertices=data.geom_xpos[np.concatenate(sphere_ids),:2]
    hull=ConvexHull(vertices)
    com=data.subtree_com[model.body('pelvis').id,:2]
    signed=hull.equations[:,:2]@com+hull.equations[:,2]
    result=dict(file=path.name,source_sha256_verified=True,frames=len(poses),fps=fps,
                sampled_minimum_foot_height_mm=sampled_min*1000,
                interpolated_minimum_foot_height_mm=between_min*1000,
                interpolated_stance_xy_deviation_mm_max=between_stance_error*1000,
                joint_limit_violation_rad=float(violation.max()),
                maximum_adjacent_joint_rate_radps=float(step_rate.max()),
                maximum_adjacent_root_translation_mm=float(root_step.max()*1000),
                final_1p5_seconds_identical=hold_identical,
                final_com_inside_support_polygon=bool(signed.max()<=0.),
                final_com_support_polygon_minimum_margin_mm=float(-signed.max()*1000),
                checked_interpolation_fractions=[.25,.5,.75])
    failures=[]
    if sampled_min<0. or between_min<-.0005:
        failures.append('Foot floor penetration')
    if between_stance_error>.002:
        failures.append('Interpolated support foot deviation >2 mm')
    if violation.max()>1e-7 or not hold_identical or signed.max()>0:
        failures.append('Joint limits or standing ending')
    if step_rate.max()>6.:
        failures.append('Adjacent joint rate >6 rad/s')
    result['passed']=not failures
    result['failures']=failures
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'envs/g1/datasets/new1009')
    args=parser.parse_args()
    files=sorted(args.directory.glob('*_contact_fixed.pkl'))
    if len(files)!=4:
        parser.error('Expected exactly four repaired motions.')
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    results=[validate(path,model) for path in files]
    report=dict(model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                scope='Sampled and interpolated kinematics; ending static COM projection. No dynamic controller/force-contact test.',
                clips=results,passed=all(item['passed'] for item in results))
    (args.directory/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if not report['passed']:
        raise SystemExit('Validation failed.')


if __name__=='__main__':
    main()
