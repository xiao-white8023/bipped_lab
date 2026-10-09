#!/usr/bin/env python3
"""Audit packaged AMP against source poses, URDF FK and actual RENet AMP code."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from build_g1_23dof_amp_dataset import _sha256
from build_g1_renet_final_dataset import DATASET, REPO


def read_json(path):
    return json.loads(path.read_text())


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_apply(q, v):
    first = 2*torch.cross(q[:,1:], v, dim=-1)
    return v+q[:,:1]*first+torch.cross(q[:,1:], first, dim=-1)


def quat_conjugate(q):
    return torch.cat((q[:,:1],-q[:,1:]),dim=-1)


def env_contract():
    path = REPO/'legged_lab/envs/g1/RENet_env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1RENetEnv')
    names = next(ast.literal_eval(n.value) for n in ast.walk(cls)
                 if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='amp_joint_names' for t in n.targets))
    function = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_amp_obs_for_expert_trans')
    namespace = {'torch':torch, 'quat_apply':quat_apply, 'quat_conjugate':quat_conjugate}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return names,namespace[function.name]


def independent_velocity(q, dt):
    v = np.empty_like(q)
    v[1:-1] = (q[2:]-q[:-2])/(2*dt)
    v[0] = (-3*q[0]+4*q[1]-q[2])/(2*dt)
    v[-1] = (3*q[-1]-4*q[-2]+q[-3])/(2*dt)
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,default=DATASET)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    dataset=args.dataset_dir.resolve();manifest=read_json(dataset/'manifest.json')
    model=mujoco.MjModel.from_xml_path(str(REPO/manifest['urdf']));data=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(model.njnt)]
    assert names==manifest['joint_order_23']
    amp_names,feature=env_contract()
    if manifest.get('amp_dimension',50)==50:
        amp_names=[name for name in amp_names if 'ankle' not in name]
    assert amp_names==manifest.get('amp_joint_order',manifest.get('amp_joint_order_19'))
    joint_count=len(amp_names);amp_dim=2*joint_count+12
    selected=[names.index(name) for name in amp_names]
    bodies=[model.body(name).id for name in manifest['amp_end_effector_order']]
    results={};amp_paths=[];display_paths=[]
    for name,meta in manifest['clips'].items():
        paths={field:dataset/meta[field] for field in ['gmr_pkl','amp_file','visualization_file']}
        for field,path in paths.items():
            assert _sha256(path)==meta[field.replace('_file','')+'_sha256'],(name,field)
        source=pickle.load(paths['gmr_pkl'].open('rb'))
        amp_json=read_json(paths['amp_file']);display_json=read_json(paths['visualization_file'])
        amp=np.asarray(amp_json['Frames']);display=np.asarray(display_json['Frames'])
        joints=np.asarray(source['dof_pos'],dtype=float)
        fps=float(source['fps']);dt=1/fps
        velocity=independent_velocity(joints,dt)
        n=len(joints)
        assert amp.shape==(n,amp_dim) and display.shape==(n,58)
        assert np.isfinite(amp).all() and np.isfinite(display).all()
        assert amp_json['FrameDuration']==display_json['FrameDuration'] and abs(amp_json['FrameDuration']-dt)<1e-12
        assert amp_json['MotionWeight']==1 and amp_json['LoopMode']==display_json['LoopMode']=='Clamp'
        if 'FrameLayout' in amp_json:
            assert amp_json['FrameLayout']=='g1_amp_joint_pos_vel_end_effectors_v1'
            assert amp_json['JointCount']==joint_count and amp_json['AMPJointNames']==amp_names
        pose_error=max(abs(display[:,:3]-source['root_pos']).max(),abs(display[:,6:29]-joints).max())
        ori_error=np.linalg.norm((Rotation.from_euler('XYZ',display[:,3:6]).inv()*Rotation.from_quat(source['root_rot'])).as_rotvec(),axis=1).max()
        joint_error=abs(amp[:,:joint_count]-joints[:,selected]).max()
        velocity_error=abs(amp[:,joint_count:2*joint_count]-velocity[:,selected]).max()
        assert pose_error<1e-8 and ori_error<2e-8 and joint_error<1e-8 and velocity_error<1e-7
        # URDF is independently compiled; its fixed-base world frame is pelvis.
        local=np.empty((n,4,3))
        for i,row in enumerate(joints):
            data.qpos[:]=row;mujoco.mj_forward(model,data);local[i]=data.xpos[bodies]
        rotation=Rotation.from_quat(source['root_rot'])
        world=np.einsum('nij,nkj->nki',rotation.as_matrix(),local)
        world+=np.asarray(source['root_pos'])[:,None,:]
        root=np.c_[source['root_pos'],rotation.as_quat()[:,[3,0,1,2]],np.zeros((n,6))]
        body=np.zeros((n,4,13));body[:,:,:3]=world
        robot=SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor(joints,dtype=torch.float64),joint_vel=torch.tensor(velocity,dtype=torch.float64),
            root_state_w=torch.tensor(root,dtype=torch.float64),body_state_w=torch.tensor(body,dtype=torch.float64)))
        proxy=SimpleNamespace(robot=robot,amp_joint_ids=selected,wrist_link_ids=[0,1],ankle_link_ids=[2,3])
        reconstructed=feature(proxy).numpy();feature_error=abs(reconstructed-amp).max()
        assert feature_error<2e-6,(name,feature_error)
        composition=source.get('sequence_composition',{})
        ending=source.get('ending_alignment',{})
        start_hold=round(fps*composition.get('stand_start_s',0))
        end_hold=round(fps*ending.get('hold_duration_s',composition.get('stand_end_s',0)))
        assert end_hold>=3,(name,'missing standing hold')
        for field in ('root_pos','root_rot','dof_pos'):
            values=np.asarray(source[field])
            if start_hold:
                assert np.array_equal(values[:start_hold],np.repeat(values[:1],start_hold,axis=0))
            assert np.array_equal(values[-end_hold:],np.repeat(values[-1:],end_hold,axis=0))
        if ending:
            assert np.count_nonzero(joints[:,[4,5,10,11]])==0
            feet_offset=local[-1,3]-local[-1,2]
            assert abs(feet_offset[[0,2]]).max()<1e-6 and feet_offset[1]<0
        limits=model.jnt_range
        violation=np.maximum(limits[:,0]-joints,0)+np.maximum(joints-limits[:,1],0)
        assert violation.max()<1e-6
        results[name]={'frames':n,'fps':fps,'sample_span_s':(n-1)/fps,
                       'root_xyz_joint_pose_max_abs_error':float(pose_error),
                       'root_orientation_max_error_rad':float(ori_error),
                       'joint_velocity_max_abs_error_rad_s':float(velocity_error),
                       'urdf_and_actual_env_feature_max_abs_error':float(feature_error),
                       'finite_and_joint_limits_passed':True,'standing_holds_preserved':True}
        amp_paths.append(str(paths['amp_file']));display_paths.append(str(paths['visualization_file']))
    assert _sha256(REPO/manifest['mjcf'])==manifest['mjcf_sha256']
    assert _sha256(REPO/manifest['urdf'])==manifest['urdf_sha256']
    loader_module=module_from_file('renet_dataset_loader',REPO/'rsl_rl/rsl_rl/utils/motion_loader.py')
    loader_cls=loader_module.G1AnkleAMPLoader if joint_count==23 else loader_module.AMPLoader
    display_cls=module_from_file('renet_dataset_display',REPO/'rsl_rl/rsl_rl/utils/motion_loader_for_display.py').AMPLoaderDisplay
    np.random.seed(23);torch.manual_seed(23)
    loader=loader_cls(args.device,time_between_frames=.02,preload_transitions=True,
                      num_preload_transitions=200000,motion_files=amp_paths)
    viewer=display_cls(args.device,time_between_frames=.005,motion_files=display_paths)
    clip_count=len(manifest['clips'])
    assert loader.observation_dim==amp_dim and loader.num_motions==clip_count and len(viewer.trajectories_full)==clip_count
    assert loader.preloaded_s.shape==loader.preloaded_s_next.shape==(200000,amp_dim)
    assert torch.isfinite(loader.preloaded_s).all() and torch.isfinite(loader.preloaded_s_next).all()
    assert np.array_equal(loader.trajectory_weights,np.full(clip_count,1/clip_count))
    # Explicit 20ms pairs at endpoints and fractional source-frame times.
    for k,meta in enumerate(manifest['clips'].values()):
        a=np.asarray(read_json(dataset/meta['amp_file'])['Frames'])
        times=np.array([0.,.007,3.007,5.003,loader.trajectory_lens[k]-.02])
        for offset in [0.,.02]:
            actual=loader.get_full_frame_at_time_batch(np.full(len(times),k),times+offset).cpu().numpy()
            indices=np.arange(len(a))*loader.trajectory_frame_durations[k]
            expected=np.stack([np.interp(times+offset,indices,a[:,j]) for j in range(amp_dim)],axis=1)
            assert abs(actual-expected).max()<3e-6
    discriminator_cls=module_from_file('renet_dataset_discriminator',REPO/'rsl_rl/rsl_rl/modules/discriminator.py').Discriminator
    discriminator=discriminator_cls(2*amp_dim,.3,[1024,512,256],args.device,.7)
    state,next_state=next(loader.feed_forward_generator(1,128))
    logits=discriminator(torch.cat((state,next_state),dim=-1))
    penalty=discriminator.compute_grad_pen(state.clone(),next_state.clone())
    loss=(logits-1).square().mean()+penalty;loss.backward()
    reward,_=discriminator.predict_amp_reward(state,next_state,torch.ones(128,device=args.device))
    assert torch.isfinite(loss) and torch.isfinite(reward).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in discriminator.parameters())
    report={'passed':True,'clips':results,'total_frames':sum(x['frames'] for x in results.values()),
            'training_device_tested':args.device,'policy_control_dt_s':.02,'amp_dimension':amp_dim,'discriminator_input_dimension':2*amp_dim,
            'preloaded_transitions':200000,'sampling_weights':loader.trajectory_weights.tolist(),
            'exact_20ms_pair_interpolation_passed':True,'actual_loader_and_display_passed':True,
            'actual_discriminator_forward_gradient_penalty_backward_reward_passed':True,
            'env_code_sha256':_sha256(REPO/'legged_lab/envs/g1/RENet_env.py'),
            'note':'Full-source URDF FK and actual environment AMP method, evaluated without an Isaac simulation; live simulator parity is a separate check.'}
    (dataset/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
