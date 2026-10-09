"""Render four repaired G1 clips as a MuJoCo video and standing montage.

For a desktop-free render, run with MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from play_g1_pkl_mujoco import MODEL, interpolate, load_motion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'envs/g1/datasets/new1009')
    parser.add_argument('--video-fps',type=int,default=20)
    args=parser.parse_args()
    names=('stand_to_walk','walk_turn_around','walk_turn_left','walk_turn_right')
    clips=[]
    for name in names:
        fps,poses=load_motion(args.directory/(name+'_contact_fixed.pkl'))
        clips.append((name,fps,poses))
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth=640
    model.vis.global_.offheight=456
    data=mujoco.MjData(model)
    cam=mujoco.MjvCamera()
    cam.distance=2.25
    cam.elevation=-12
    cam.azimuth=180
    with mujoco.Renderer(model,height=456,width=640) as renderer:
        def render(pose,front=False):
            data.qpos[:]=pose
            mujoco.mj_forward(model,data)
            cam.lookat[:]=[*pose[:2],.68]
            cam.azimuth=170+Rotation.from_quat(pose[[4,5,6,3]]).as_euler('ZYX',degrees=True)[0] if front else 180
            renderer.update_scene(data,camera=cam)
            return Image.fromarray(renderer.render())

        montage=Image.new('RGB',(1280,960))
        draw=ImageDraw.Draw(montage)
        for i,(name,fps,poses) in enumerate(clips):
            x,y=i%2*640,i//2*480
            montage.paste(render(poses[-1],front=True),(x,y+24))
            draw.text((x+12,y+7),f'{name} - corrected standing ending',fill='white')
        montage.save(args.directory/'ending_preview.png')
        duration=max((len(poses)-1)/fps for _,fps,poses in clips)
        count=int(np.ceil(duration*args.video_fps))+1
        destination=args.directory/'preview.mp4'
        with imageio.get_writer(str(destination),fps=args.video_fps,codec='libx264',quality=7,macro_block_size=16) as writer:
            for frame in range(count):
                time=frame/args.video_fps
                canvas=Image.new('RGB',(1280,960))
                draw=ImageDraw.Draw(canvas)
                for i,(name,fps,poses) in enumerate(clips):
                    position=min(time*fps,len(poses)-1)
                    low=int(position)
                    pose=interpolate(poses[low],poses[min(low+1,len(poses)-1)],position-low)
                    x,y=i%2*640,i//2*480
                    canvas.paste(render(pose),(x,y+24))
                    draw.text((x+12,y+7),f'{name}  {min(time,(len(poses)-1)/fps):.2f}s',fill='white')
                writer.append_data(np.asarray(canvas))
                if frame%40==0:
                    print(f'Rendering {frame}/{count}',flush=True)
        print(f'Saved {destination}',flush=True)


if __name__=='__main__':
    main()
