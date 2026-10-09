"""Repair G1 PKL foot contacts and append a contact-aware stopping motion.

Source clips are never overwritten. Contacts are inferred from relative foot
heights, not measured forces. A bounded, Jacobian-based IK solves pelvis XYZ
and all twelve leg joints; torso/arms are retained during the original gait.
The output is a kinematic reference, not a proven torque-feasible controller.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml"
DIRECTORY = ROOT / "legged_lab/envs/g1/datasets/pkl"
NAMES = ("stand_to_walk", "walk_turn_around", "walk_turn_left", "walk_turn_right")


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u ** 3 * (10. - 15. * u + 6. * u ** 2)


def runs(mask):
    edges = np.r_[0, np.flatnonzero(np.diff(mask)) + 1, len(mask)]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


class Geometry:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.feet = [self.model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        self.pelvis = self.model.body("pelvis").id
        self.samples = []
        self.spheres = []
        for body in self.feet:
            points = []
            sphere_ids = []
            for geom in np.flatnonzero(self.model.geom_bodyid == body):
                if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = self.model.geom_dataid[geom]
                    first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                    rotation = R.from_quat(self.model.geom_quat[geom, [1, 2, 3, 0]]).as_matrix()
                    points.extend(self.model.mesh_vert[first:first + count] @ rotation.T + self.model.geom_pos[geom])
                elif self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    sphere_ids.append(int(geom))
            self.samples.append(np.asarray(points))
            self.spheres.append(np.asarray(sphere_ids))
        self.qindices = np.r_[0:3, 7:19]
        self.vindices = np.r_[0:3, 6:18]
        self.low = np.r_[[-np.inf] * 3, self.model.jnt_range[1:13, 0] + 1e-5]
        self.high = np.r_[[np.inf] * 3, self.model.jnt_range[1:13, 1] - 1e-5]
        self.low[[6, 12]] = .15  # Avoid straight-knee singularities/branch changes.

    def set(self, pose):
        self.data.qpos[:] = pose
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def local_bottom(self, side, rotation):
        points = self.samples[side]
        mesh_bottom = float((points @ rotation.T)[:, 2].min())
        ids = self.spheres[side]
        collision_bottom = float(((self.model.geom_pos[ids] @ rotation.T)[:, 2]
                                  - self.model.geom_size[ids, 0]).min())
        return min(mesh_bottom, collision_bottom)

    def measure(self, poses):
        count = len(poses)
        xyz = np.empty((count, 2, 3))
        rotation = np.empty((count, 2, 3, 3))
        bottom = np.empty((count, 2))
        com = np.empty((count, 3))
        for i, pose in enumerate(poses):
            self.set(pose)
            xyz[i] = self.data.xpos[self.feet]
            rotation[i] = self.data.xmat[self.feet].reshape(2, 3, 3)
            com[i] = self.data.subtree_com[self.pelvis]
            bottom[i] = [xyz[i, s, 2] + self.local_bottom(s, rotation[i, s]) for s in range(2)]
        return xyz, rotation, bottom, com

    def solve(self, reference, xyz, rotations, previous=None):
        """High-weight foot pose tasks with soft root and joint regularization."""
        pose = reference.copy()
        if previous is not None:
            pose[self.qindices] = previous[self.qindices]
        xref = reference[self.qindices]
        x = np.clip(pose[self.qindices], self.low, self.high)
        # Position tasks are in metres; rotation tasks are world rotvec radians.
        task_weights = np.tile([1000., 1000., 1000., 200., 200., 200.], 2)
        prior_weights = np.r_[[12., 12., 18.], np.tile([.6, .6, .3, .8, .1, .1], 2)]
        continuity_weights = np.r_[[0., 0., 0.], [.35] * 12] if previous is not None else np.zeros(15)
        xprevious = previous[self.qindices] if previous is not None else xref
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        def residual_and_jacobian(x, want_jacobian=False):
            pose[self.qindices] = x
            self.set(pose)
            residual, jac = [], []
            for side, body in enumerate(self.feet):
                current_r = self.data.xmat[body].reshape(3, 3)
                residual.extend(self.data.xpos[body] - xyz[side])
                phi = R.from_matrix(current_r @ rotations[side].T).as_rotvec()
                residual.extend(phi)
                mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
                jac.extend(jp[:, self.vindices])
                theta = np.linalg.norm(phi)
                skew = np.array([[0., -phi[2], phi[1]], [phi[2], 0., -phi[0]], [-phi[1], phi[0], 0.]])
                coefficient = 1./12. if theta < 1e-4 else (1. - .5*theta/np.tan(.5*theta))/theta**2
                inverse_left = np.eye(3) - .5*skew + coefficient*skew@skew
                jac.extend(inverse_left @ jr[:, self.vindices])
            error = np.asarray(residual)
            a = np.asarray(jac) * task_weights[:, None]
            b = error * task_weights
            if want_jacobian:
                return np.r_[a, np.diag(prior_weights), np.diag(continuity_weights)]
            return np.r_[b, prior_weights * (x - xref), continuity_weights * (x - xprevious)]
        result = least_squares(residual_and_jacobian, x,
                               jac=lambda value:residual_and_jacobian(value,True),
                               bounds=(self.low,self.high),x_scale='jac',
                               ftol=1e-9,xtol=1e-9,gtol=1e-8,max_nfev=80)
        pose[self.qindices] = result.x
        return pose


def infer_contacts(xyz, fps):
    heights = gaussian_filter1d(xyz[:, :, 2], max(.5, .035 * fps), axis=0)
    relative = heights - heights.min(axis=1, keepdims=True)
    contact = relative < .018
    minimum = max(2, int(round(.09 * fps)))
    for side in range(2):
        for _ in range(3):
            segments = runs(contact[:, side])
            for index, (a, b, on) in enumerate(segments):
                if b - a < minimum and 0 < index < len(segments) - 1:
                    contact[a:b, side] = not on
    empty = np.flatnonzero(~contact.any(axis=1))
    contact[empty, np.argmin(heights[empty], axis=1)] = True
    return contact, relative


def foot_targets(geometry, xyz, rotation, contact, fps, clearance):
    target = xyz.copy()
    yaw = np.unwrap(np.arctan2(rotation[:, :, 1, 0], rotation[:, :, 0, 0]), axis=0)
    target_yaw = yaw.copy()
    swing_height = np.zeros(contact.shape)
    segments_report = []
    for side in range(2):
        segments = runs(contact[:, side])
        for a, b, on in segments:
            if on:
                anchor = np.median(xyz[a:b, side, :2], axis=0)
                heading = float(np.median(yaw[a:b, side]))
                target[a:b, side, :2] = anchor
                target_yaw[a:b, side] = heading
                segments_report.append(dict(side=side, start_frame=a, end_frame_exclusive=b,
                                            ankle_anchor_xy_m=anchor.tolist(), yaw_rad=heading))
        for a, b, on in segments:
            if on:
                continue
            left, right = max(0, a - 1), min(len(xyz) - 1, b)
            u = np.linspace(0., 1., right - left + 1)
            blend = smooth(u)
            first = target[left, side, :2].copy()
            last = target[right, side, :2].copy()
            curve = (1. - blend[:, None]) * first + blend[:, None] * last
            # Keep a small source swing curvature, vanishing at both contacts.
            raw_line = ((1. - u[:, None]) * xyz[left, side, :2]
                        + u[:, None] * xyz[right, side, :2])
            curve += .35 * (xyz[left:right + 1, side, :2] - raw_line) * np.sin(np.pi * u)[:, None] ** 2
            target[left:right + 1, side, :2] = curve
            target_yaw[left:right + 1, side] = (1. - blend) * target_yaw[left, side] + blend * target_yaw[right, side]
            raw_relative = xyz[left:right + 1, side, 2] - xyz[left:right + 1, :, 2].min(axis=1)
            peak = float(np.clip(.65 * raw_relative.max(), .04, .065))
            lift = peak * np.sin(np.pi * u) ** 3
            # Clips can begin/end in mid swing. Preserve that endpoint clearance.
            if a == 0:
                lift += raw_relative[0] * (1. - blend)
            if b == len(xyz):
                lift += raw_relative[-1] * blend
            swing_height[left:right + 1, side] = lift
    rotations = R.from_euler("z", target_yaw.ravel()).as_matrix().reshape(-1, 2, 3, 3)
    for frame in range(len(target)):
        for side in range(2):
            target[frame, side, 2] = (clearance + swing_height[frame, side]
                                      - geometry.local_bottom(side, rotations[frame, side]))
    return target, rotations, segments_report


def resample_tail(reference, xyz, rotations, contacts, fps, seconds=1.):
    """Slow the last source second to rest with continuous time derivatives."""
    length = min(len(reference) - 2, int(round(seconds * fps)))
    start = len(reference) - 1 - length
    u = np.linspace(0., 1., 2 * length + 1)
    source_times = start + length * (2. * u - 2. * u ** 3 + u ** 4)
    times = np.r_[np.arange(start), source_times]
    source = np.arange(len(reference))
    ref = np.stack([np.interp(times, source, reference[:, j]) for j in range(30)], axis=1)
    root_rot = Slerp(source, R.from_quat(reference[:, [4, 5, 6, 3]]))(times).as_quat()
    ref[:, 3:7] = root_rot[:, [3, 0, 1, 2]]
    targets = np.stack([np.interp(times, source, xyz[:, s, j])
                        for s in range(2) for j in range(3)], axis=1).reshape(-1, 2, 3)
    foot_rot = np.stack([Slerp(source, R.from_matrix(rotations[:, s]))(times).as_matrix()
                         for s in range(2)], axis=1)
    # Interpolated touchdown boundaries remain designated swing until landing.
    labels = contacts[np.clip(np.floor(times).astype(int), 0, len(source) - 1)].copy()
    for s in range(2):
        flat = np.isclose(targets[:, s, 2], targets[:, s, 2].min(), atol=1e-7)
        labels[:, s] &= flat
    return ref, targets, foot_rot, labels, times


def append_stop(geometry, reference, xyz, rotations, contact, fps, clearance):
    support = int(np.argmin(xyz[-1, :, 2]))
    other = 1 - support
    last = reference[-1].copy()
    last_xyz = xyz[-1].copy()
    last_rotation = rotations[-1].copy()
    heading = float(np.arctan2(last_rotation[support, 1, 0], last_rotation[support, 0, 0]))
    yaw_r = R.from_euler("z", heading).as_matrix()
    stand = last.copy()
    stand[3:7] = R.from_euler("z", heading).as_quat()[[3, 0, 1, 2]]
    stand[7:] = 0.
    stand[[7, 13]] = -.15
    stand[[10, 16]] = .30
    stand[[11, 17]] = -.15
    stand[[21, 26]] = [.10, -.10]
    stand[[23, 28]] = .30
    stand[:3] = 0.
    geometry.set(stand)
    local_feet = geometry.data.xpos[geometry.feet].copy()
    desired = last_xyz.copy()
    desired[other, :2] = last_xyz[support, :2] + (local_feet[other] - local_feet[support])[:2]
    for side in range(2):
        desired[side, 2] = clearance - geometry.local_bottom(side, yaw_r)
    stand[:3] = desired.mean(axis=0) - local_feet.mean(axis=0)
    geometry.set(last)
    com_offset = geometry.data.subtree_com[geometry.pelvis, :2] - last[:2]
    support_root = last.copy()
    support_root[:2] = last_xyz[support, :2] + (yaw_r @ np.array([.035, 0., 0.]))[:2] - com_offset
    support_root[2] = min(last[2], stand[2])
    # Shift over the support foot, land the other foot, then centre the pelvis.
    phases = ((.45, "shift"), (.75, "land"), (.65, "settle"), (1.5, "hold"))
    new_ref, new_xyz, new_rot, labels = [], [], [], []
    for seconds, phase in phases:
        count = max(2, int(round(seconds * fps)))
        for u in np.linspace(0., 1., count + 1)[1:]:
            blend = float(smooth(u))
            feet = last_xyz.copy()
            foot_r = last_rotation.copy()
            on = np.array([False, False])
            on[support] = True
            if phase == "shift":
                pose = (1. - blend) * last + blend * support_root
                pose[3:7] = last[3:7]
            elif phase == "land":
                pose = support_root.copy()
                pose[7:] = (1. - .5 * blend) * last[7:] + .5 * blend * stand[7:]
                pose[3:7] = Slerp([0., 1.], R.from_quat(np.stack([last[[4,5,6,3]], stand[[4,5,6,3]]])))(blend).as_quat()[[3,0,1,2]]
                feet[other] = (1. - blend) * last_xyz[other] + blend * desired[other]
                feet[other, 2] += .055 * 16. * u ** 2 * (1. - u) ** 2
                foot_r[other] = Slerp([0., 1.], R.from_matrix(np.stack([last_rotation[other], yaw_r])))(blend).as_matrix()
                on[other] = u == 1.
            else:
                feet = desired.copy()
                foot_r[:] = yaw_r
                on[:] = True
                if phase == "settle":
                    transition_start = support_root.copy()
                    transition_start[3:7] = stand[3:7]
                    transition_start[7:] = .5 * last[7:] + .5 * stand[7:]
                    pose = (1. - blend) * transition_start + blend * stand
                    pose[3:7] = stand[3:7]
                else:
                    pose = stand.copy()
            new_ref.append(pose)
            new_xyz.append(feet)
            new_rot.append(foot_r)
            labels.append(on)
    return (np.r_[reference, new_ref], np.r_[xyz, new_xyz], np.r_[rotations, new_rot],
            np.r_[contact, labels], dict(support_side=support, durations_s={p:s for s,p in phases},
                                       final_heading_rad=heading, start_frame=len(reference)))


def statistics(geometry, poses, contact, fps):
    xyz, rotations, bottom, com = geometry.measure(poses)
    velocity = np.gradient(poses[:, 7:], 1. / fps, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, 1. / fps, axis=0, edge_order=2)
    foot_speed = np.linalg.norm(np.diff(xyz[:, :, :2], axis=0) * fps, axis=2)
    support_speed = foot_speed[contact[:-1] & contact[1:]]
    limits = geometry.model.jnt_range[1:]
    violation = np.maximum(limits[:, 0] - poses[:, 7:], 0.) + np.maximum(poses[:, 7:] - limits[:, 1], 0.)
    separation = (xyz[-1, 0] - xyz[-1, 1]) @ R.from_quat(poses[-1, [4,5,6,3]]).as_matrix()
    tilt = np.rad2deg(np.arccos(np.clip(rotations[:, :, 2, 2], -1., 1.)))
    report = dict(frames=len(poses), duration_s=(len(poses)-1)/fps,
                  minimum_shoe_or_collider_height_mm=float(bottom.min()*1000),
                  foot_penetration_over_1mm_frames=int(np.count_nonzero(bottom.min(axis=1)<-.001)),
                  both_feet_above_20mm_frames=int(np.count_nonzero(bottom.min(axis=1)>.02)),
                  inferred_support_xy_speed_mps_p50_p95_max=(np.percentile(support_speed,[50,95,100]).tolist() if len(support_speed) else None),
                  inferred_support_sole_tilt_deg_max=float(tilt[contact].max()) if contact.any() else None,
                  peak_joint_speed_radps=float(np.abs(velocity).max()),
                  peak_joint_acceleration_radps2=float(np.abs(acceleration).max()),
                  peak_adjacent_joint_change_deg=float(np.rad2deg(np.abs(np.diff(poses[:,7:],axis=0))).max()),
                  joint_limit_violation_rad=float(violation.max()),
                  final_foot_separation_forward_lateral_vertical_m=separation.tolist(),
                  final_foot_bottom_heights_mm=(bottom[-1]*1000).tolist(),
                  final_foot_tilt_deg=tilt[-1].tolist(),
                  final_root_speed_mps=float(np.linalg.norm(poses[-1,:3]-poses[-2,:3])*fps),
                  final_joint_speed_radps=float(np.abs(poses[-1,7:]-poses[-2,7:]).max()*fps),
                  final_com_xy_m=com[-1,:2].tolist())
    return report, xyz, bottom, tilt


def limit_speed(geometry, poses, xyz, rotations, contacts, source_time, fps, ending):
    """Retiming preserves planted foot targets and slows fast knee extension."""
    joint_rate = np.abs(np.diff(poses[:,7:],axis=0)).max(axis=1)*fps
    root_rate = np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1)*fps
    root_rotation = R.from_quat(poses[:,[4,5,6,3]])
    angular_rate = (root_rotation[:-1].inv()*root_rotation[1:]).magnitude()*fps
    factor = np.maximum.reduce([np.ones(len(joint_rate)),joint_rate/5.,root_rate/1.2,angular_rate/1.8])
    envelope = gaussian_filter1d(maximum_filter1d(factor,size=max(3,int(.16*fps)//2*2+1)),max(.5,.05*fps))
    envelope = 1.+1.12*(envelope-1.)
    envelope = np.maximum(envelope,factor*1.015)
    envelope[factor==1.] = np.maximum(envelope[factor==1.],1.)
    clock=np.r_[0.,np.cumsum(envelope/fps)]
    new_clock=np.linspace(0.,clock[-1],int(np.ceil(clock[-1]*fps))+1)
    sample=np.interp(new_clock,clock,np.arange(len(poses)))
    source=np.arange(len(poses))
    reference=PchipInterpolator(source,poses,axis=0)(sample)
    reference[:,3:7]=Slerp(source,root_rotation)(sample).as_quat()[:,[3,0,1,2]]
    targets=PchipInterpolator(source,xyz,axis=0)(sample)
    foot_rot=np.stack([Slerp(source,R.from_matrix(rotations[:,s]))(sample).as_matrix() for s in range(2)],axis=1)
    low=np.minimum(np.floor(sample).astype(int),len(poses)-1)
    high=np.minimum(low+1,len(poses)-1)
    labels=contacts[low]&contacts[high]
    previous=None
    out=[]
    for i,ref in enumerate(reference):
        previous=geometry.solve(ref,targets[i],foot_rot[i],previous)
        out.append(previous)
        if i%150==0:
            print(f"Speed-limited IK {i}/{len(reference)}",flush=True)
    new_source=np.r_[source_time,np.full(len(poses)-len(source_time),-1.)]
    mapping=np.interp(sample,source,new_source)
    mapping[sample>=len(source_time)]=-1.
    old_start=ending['start_frame']
    ending=dict(ending,start_frame=int(np.searchsorted(sample,old_start)),
                planned_start_frame_before_speed_retiming=old_start)
    return (np.asarray(out),targets,foot_rot,labels,mapping,ending,
            dict(joint_speed_target_radps=5.,root_speed_target_mps=1.2,root_angular_speed_target_radps=1.8,
                 maximum_time_stretch=float(envelope.max()),duration_before_s=(len(poses)-1)/fps,
                 duration_after_s=(len(reference)-1)/fps,
                 method='Smooth local time stretching and bounded IK reprojection; original FPS retained.'))


def repair(path, output_dir, geometry, clearance):
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as stream:
        clip = pickle.load(stream)
    fps = float(clip["fps"])
    root = np.asarray(clip["root_pos"], dtype=float)
    quaternion = np.asarray(clip["root_rot"], dtype=float)
    joints = np.asarray(clip["dof_pos"], dtype=float)
    if fps <= 0 or not np.isfinite(fps) or root.shape != (len(joints),3) or quaternion.shape != (len(joints),4) or joints.shape[1:] != (23,):
        raise ValueError(f"Invalid G1 PKL: {path}")
    original = np.c_[root, quaternion[:,[3,0,1,2]], joints]
    if not np.isfinite(original).all() or len(original)<3 or not np.allclose(np.linalg.norm(quaternion,axis=1),1.,atol=1e-5):
        raise ValueError(f"Invalid poses: {path}")
    xyz, rotation, original_bottom, _ = geometry.measure(original)
    contacts, relative = infer_contacts(xyz, fps)
    target_xyz, target_rotation, segments = foot_targets(geometry, xyz, rotation, contacts, fps, clearance)
    reference = original.copy()
    reference[:, :3] = gaussian_filter1d(reference[:, :3], .035 * fps, axis=0)
    reference[:, 7:] = gaussian_filter1d(reference[:, 7:], .025 * fps, axis=0)
    euler = R.from_quat(quaternion).as_euler('ZYX')
    euler[:, 0] = np.unwrap(euler[:, 0])
    reference[:, 3:7] = R.from_euler('ZYX', gaussian_filter1d(euler, .035 * fps, axis=0)).as_quat()[:, [3,0,1,2]]
    reference[:,2] += gaussian_filter1d(target_xyz[:,:,2].min(axis=1)-xyz[:,:,2].min(axis=1), .055*fps)
    reference[:,2] -= .05
    reference, target_xyz, target_rotation, contacts, source_time = resample_tail(reference, target_xyz, target_rotation, contacts, fps)
    # Solve the last gait pose once before planning its weight shift.
    reference[-1] = geometry.solve(reference[-1],target_xyz[-1],target_rotation[-1])
    reference, target_xyz, target_rotation, contacts, ending = append_stop(geometry,reference,target_xyz,target_rotation,contacts,fps,clearance)
    poses=[]
    previous=None
    for frame, ref in enumerate(reference):
        previous=geometry.solve(ref,target_xyz[frame],target_rotation[frame],previous)
        poses.append(previous)
        if frame%100==0:
            print(f"{path.stem}: IK {frame}/{len(reference)}",flush=True)
    poses=np.asarray(poses)
    poses,target_xyz,target_rotation,contacts,source_time,ending,retiming=limit_speed(
        geometry,poses,target_xyz,target_rotation,contacts,source_time,fps,ending)
    # An identical reference/feet at the hold is written as an identical pose.
    hold=max(2,int(round(1.5*fps)))
    poses[-hold:]=poses[-1]
    report, actual_xyz, bottom, tilt=statistics(geometry,poses,contacts,fps)
    foot_error=np.linalg.norm(actual_xyz-target_xyz,axis=2)
    report.update(source=str(path.resolve()), source_sha256=before_hash, source_frames=len(original),fps=fps,
                  source_minimum_shoe_or_collider_height_mm=float(original_bottom.min()*1000),
                  foot_target_error_mm_p50_p95_max=(np.percentile(foot_error,[50,95,100])*1000).tolist(),
                  inferred_contacts=True, contact_inference="Relative ankle heights <18 mm; remove short interior runs <90 ms; at least one inferred support foot.",
                  stance_segments=segments, ending=ending, tail_time_scaling="Last source second stretched to two seconds with zero final clock velocity and acceleration.",
                  speed_retiming=retiming,
                  ik_used=True, ankle_angles_no_longer_fixed=True, source_modified=True,
                  validation_scope="Kinematics, mesh/collision grounding, inferred stance sliding, joint limits, ending. No dynamics or force-contact validation.")
    if report['minimum_shoe_or_collider_height_mm']<-.5 or report['joint_limit_violation_rad']>1e-7 or foot_error.max()>.003:
        output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output_dir/(path.stem+'_failed_debug.npz'),poses=poses,target_xyz=target_xyz,target_rotation=target_rotation,reference=reference,contact=contacts)
        raise RuntimeError(f"{path.name}: IK/grounding validation failed: {json.dumps({k:v for k,v in report.items() if k!='stance_segments'})}")
    if abs(report['final_foot_separation_forward_lateral_vertical_m'][0])>.003 or max(report['final_foot_tilt_deg'])>.5:
        raise RuntimeError(f"{path.name}: final standing feet are not aligned.")
    output_dir.mkdir(parents=True,exist_ok=True)
    destination=output_dir/(path.stem+"_contact_fixed.pkl")
    if destination.resolve()==path.resolve() or destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    updated=dict(clip)
    for stale in ('local_body_pos','body_pos','body_rot','dof_vel','root_vel','ending_alignment','left_leg_refinement'):
        updated.pop(stale,None)
    updated.update(root_pos=poses[:,:3],root_rot=poses[:,[4,5,6,3]],dof_pos=poses[:,7:],
                   local_body_pos=None,
                   motion_name=destination.stem,playback_loop=False,contact_repair=report,
                   inferred_foot_contacts=contacts,source_frame_times=np.r_[source_time,np.full(len(poses)-len(source_time),-1.)])
    with destination.open('wb') as stream:
        pickle.dump(updated,stream)
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with destination.with_suffix('.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(('frame','time_s','left_inferred_contact','right_inferred_contact','left_bottom_m','right_bottom_m','left_tilt_deg','right_tilt_deg','left_target_error_m','right_target_error_m'))
        for frame in range(len(poses)):
            writer.writerow((frame,frame/fps,*contacts[frame].astype(int),*bottom[frame],*tilt[frame],*foot_error[frame]))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before_hash
    print(json.dumps({k:v for k,v in report.items() if k not in ('stance_segments',)},ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',nargs='+',type=Path,default=[DIRECTORY/(name+'.pkl') for name in NAMES])
    parser.add_argument('--output-dir',type=Path,default=DIRECTORY.parent/'new1009')
    parser.add_argument('--clearance-mm',type=float,default=1.)
    args=parser.parse_args()
    if not np.isfinite(args.clearance_mm) or args.clearance_mm<0:
        parser.error('Clearance must be nonnegative and finite.')
    for path in args.motion:
        if (args.output_dir/(path.stem+'_contact_fixed.pkl')).exists():
            parser.error('Output exists; choose a new directory.')
    geometry=Geometry()
    reports={path.stem:repair(path,args.output_dir,geometry,args.clearance_mm/1000.) for path in args.motion}
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(model=str(MODEL),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),clips=reports),indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()
