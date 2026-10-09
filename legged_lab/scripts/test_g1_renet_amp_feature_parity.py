#!/usr/bin/env python3
"""Verify the final dataset against a G1 task's AMP method in IsaacLab FK."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'rsl_rl'))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=('g1_renet', 'g1_rough'), default='g1_renet')
parser.add_argument('--report', type=Path, help='Write results here instead of the default dataset report.')
parser.add_argument('--dataset-dir', type=Path,
                    help='Defaults to the dataset selected by the task AMP cfg switch.')
parser.add_argument('--atol',type=float,default=2e-5)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul

from legged_lab.assets.g1.g1_29 import G1_23CFG
from legged_lab.envs.g1.RENet_env import G1RENetEnv
from legged_lab.utils import task_registry


@configclass
class ParitySceneCfg(InteractiveSceneCfg):
    robot = G1_23CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


def main():
    label = 'RENet' if args.task == 'g1_renet' else args.task
    env_cfg,agent_cfg=task_registry.get_cfgs(args.task)
    if args.dataset_dir is not None:
        dataset = args.dataset_dir.resolve()
    else:
        motion_dir = Path(agent_cfg.amp_motion_files[0]).parent
        dataset = motion_dir if (motion_dir/'manifest.json').is_file() else motion_dir.parent
    manifest=json.loads((dataset/'manifest.json').read_text())
    assert env_cfg.amp_observation_dim == manifest['amp_dimension']
    env_class=task_registry.get_task_class(args.task)
    amp_paths=[str((dataset/meta['amp_file']).resolve()) for meta in manifest['clips'].values()]
    display_paths=[str((dataset/meta['visualization_file']).resolve()) for meta in manifest['clips'].values()]
    if args.task == 'g1_renet':
        assert env_class is G1RENetEnv
    assert agent_cfg.to_dict()['amp_motion_files']==amp_paths
    assert env_cfg.amp_motion_files_display==display_paths
    assert abs(env_cfg.sim.dt*env_cfg.sim.decimation-.02)<1e-10
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device,dt=.005,gravity=(0.,0.,0.)))
    scene=InteractiveScene(ParitySceneCfg(num_envs=1,env_spacing=2.))
    sim.reset();scene.reset();robot=scene['robot']
    joint_ids,joint_names=robot.find_joints(manifest['joint_order_23'],preserve_order=True)
    amp_ids,amp_names=robot.find_joints(manifest['amp_joint_order'],preserve_order=True)
    hands,_=robot.find_bodies(manifest['amp_end_effector_order'][:2],preserve_order=True)
    feet,_=robot.find_bodies(manifest['amp_end_effector_order'][2:],preserve_order=True)
    assert joint_names==manifest['joint_order_23'] and amp_names==manifest['amp_joint_order']
    joint_count=len(amp_names)
    proxy=SimpleNamespace(robot=robot,amp_joint_ids=amp_ids,wrist_link_ids=hands,ankle_link_ids=feet)
    reports={}
    for name,meta in manifest['clips'].items():
        raw=pickle.load((dataset/meta['gmr_pkl']).open('rb'))
        expert=np.asarray(json.loads((dataset/meta['amp_file']).read_text())['Frames'])
        fps=float(raw['fps'])
        q=np.asarray(raw['dof_pos'],dtype=float);v=np.gradient(q,1/fps,axis=0,edge_order=2)
        frames=set(np.linspace(0,len(q)-1,12,dtype=int).tolist())
        for first,last in meta.get('sequence_composition',{}).get('phase_intervals_s',{}).values():
            for fraction in [.25,.65]:frames.add(min(round(fps*(first+fraction*(last-first))),len(q)-1))
        block_errors=np.zeros(4);count=0
        for frame in sorted(frames):
            joints=torch.zeros((1,robot.num_joints),device=args.device)
            speeds=torch.zeros_like(joints)
            joints[:,joint_ids]=torch.as_tensor(q[frame],dtype=torch.float32,device=args.device)
            speeds[:,joint_ids]=torch.as_tensor(v[frame],dtype=torch.float32,device=args.device)
            root=torch.as_tensor(np.r_[raw['root_pos'][frame],raw['root_rot'][frame,[3,0,1,2]]],dtype=torch.float32,device=args.device).unsqueeze(0)
            root[:,3:7]/=torch.linalg.vector_norm(root[:,3:7],dim=-1,keepdim=True)
            expected=torch.as_tensor(expert[frame],dtype=torch.float32,device=args.device).unsqueeze(0)
            for variant in ['source','translated_and_rotated']:
                pose=root.clone()
                if variant!='source':
                    pose[:,:3]+=torch.tensor([1.25,-.6,.8],device=args.device)
                    heading=torch.tensor([[np.cos(.185),0.,0.,np.sin(.185)]],device=args.device,dtype=torch.float32)
                    pose[:,3:7]=quat_mul(heading,pose[:,3:7])
                robot.write_root_link_pose_to_sim(pose)
                robot.write_root_link_velocity_to_sim(torch.zeros((1,6),device=args.device))
                robot.write_joint_state_to_sim(joints,speeds)
                sim.forward();scene.update(.005)
                actual=env_class.get_amp_obs_for_expert_trans(proxy)
                error=abs(actual-expected).squeeze().cpu().numpy()
                blocks=[error[:joint_count].max(),error[joint_count:2*joint_count].max(),
                        error[2*joint_count:2*joint_count+6].max(),error[2*joint_count+6:].max()]
                block_errors=np.maximum(block_errors,blocks);count+=1
                assert error.max()<args.atol,(name,frame,variant,float(error.max()))
        reports[name]={'source_frames_tested':sorted(frames),'world_pose_variants':2,'samples':count,
                       'max_absolute_error_by_block':dict(zip(['joint_pos','joint_vel','hands_pos_b','feet_pos_b'],block_errors.tolist()))}
        print(f'[{label} AMP parity] {name}: {count} poses; max error {block_errors.max():.3g}',flush=True)
    result={'passed':True,'task':args.task,'device':args.device,'atol':args.atol,'task_registration_and_config_paths_passed':True,
            'amp_dimension':2*joint_count+12,'includes_ankles':joint_count==23,
            'samples_total':sum(r['samples'] for r in reports.values()),'clips':reports,
            'note':'Exact pose/velocity writes and FK refresh on the actual G1_23CFG asset; no physics stepping or policy training.'}
    report = args.report or dataset/('isaac_feature_parity.json' if args.task == 'g1_renet'
                                     else 'isaac_g1_rough_feature_parity.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(f'[{label} AMP parity] PASSED',flush=True)


main()
# The report is fully written after successful checks. This short-lived FK
# audit has no pending render/write jobs; use SimulationApp's documented
# immediate shutdown to avoid waiting on unrelated extension teardown.
# Keep this outside finally so assertion failures retain their failure status.
app.close(wait_for_replicator=False, skip_cleanup=True)
