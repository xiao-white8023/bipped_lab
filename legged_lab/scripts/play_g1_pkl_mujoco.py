"""Preview G1 23-DoF PKL poses in MuJoCo, with one window per motion.

This replays root/joint poses with mj_forward; it does not run a trained policy.
PKL root_rot is XYZW and dof_pos follows the GMR G1 23-DoF joint order.
"""
from __future__ import annotations

import argparse
import ctypes as C
import ctypes.util
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml'
DATA = ROOT / 'legged_lab/envs/g1/datasets/pkl'
DEFAULTS = [DATA / f'{name}.pkl' for name in (
    'stand_to_walk', 'walk_turn_around', 'walk_turn_left', 'walk_turn_right')]


def load_motion(path):
    with path.open('rb') as stream:
        clip = pickle.load(stream)
    fps = float(clip['fps'])
    positions = np.asarray(clip['root_pos'], dtype=float)
    rotations = np.asarray(clip['root_rot'], dtype=float)
    angles = np.asarray(clip['dof_pos'], dtype=float)
    count = len(positions)
    if (count < 2 or positions.shape != (count, 3)
            or rotations.shape != (count, 4) or angles.shape != (count, 23)):
        raise ValueError(f'{path}: expected root_pos(T,3), root_rot(T,4), dof_pos(T,23).')
    qpos = np.concatenate((positions, rotations[:, [3, 0, 1, 2]], angles), axis=1)
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f'{path}: invalid motion values or FPS.')
    if not np.allclose(np.linalg.norm(rotations, axis=1), 1., atol=1e-4):
        raise ValueError(f'{path}: root rotations must be unit XYZW quaternions.')
    return fps, qpos


def make_model(name):
    xml = ET.parse(MODEL).getroot()
    xml.set('model', f'G1_23DoF__{name}')
    compiler = xml.find('compiler')
    compiler.set('meshdir', str((MODEL.parent / compiler.get('meshdir', '')).resolve()))
    # A visual floor makes swing clearance and the final planted stance visible.
    asset = xml.find('asset')
    ET.SubElement(asset, 'texture', name='motion_preview_grid', type='2d', builtin='checker',
                  rgb1='.78 .80 .82', rgb2='.88 .90 .92', width='512', height='512')
    ET.SubElement(asset, 'material', name='motion_preview_ground', texture='motion_preview_grid',
                  texrepeat='2 2', texuniform='true', reflectance='.05')
    world = xml.find('worldbody')
    ET.SubElement(world, 'geom', name='motion_preview_floor', type='plane', size='20 20 .1',
                  pos='0 0 0', material='motion_preview_ground', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', name='motion_preview_light', pos='0 0 3', dir='-.2 -.3 -1',
                  directional='true', diffuse='.8 .8 .8', ambient='.3 .3 .3')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    if model.nq != 30 or model.njnt != 24:
        raise ValueError('Expected a floating base and 23 scalar G1 joints.')
    return model


def interpolate(first, second, fraction):
    pose = first * (1 - fraction) + second * fraction
    a, b = first[3:7], second[3:7].copy()
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, 0., 1.))
    if dot > .9995:
        quat = a * (1 - fraction) + b * fraction
    else:
        angle = np.arccos(dot)
        quat = (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)
    pose[3:7] = quat / np.linalg.norm(quat)
    return pose


def play(path, args):
    fps, poses = load_motion(path)
    model = make_model(path.stem)
    data = mujoco.MjData(model)
    count = len(poses)
    state = {'frame': 0., 'paused': args.start_paused, 'speed': args.speed}
    mutex = threading.Lock()

    def key_callback(key):
        with mutex:
            if key == 32:  # Space
                state['paused'] = not state['paused']
            elif key == 82:  # R
                state.update(frame=0., paused=False)
            elif key == 83:  # S
                state['speed'] = args.speed / 2 if state['speed'] == args.speed else args.speed
            elif key in (262, 263):
                state['paused'] = True
                state['frame'] = float(np.clip(int(state['frame']) + (1 if key == 262 else -1), 0, count - 1))

    data.qpos[:] = poses[0]
    mujoco.mj_forward(model, data)
    print(f'{path.name}: {count} frames, {fps:g} FPS, {count / fps:.3f}s.', flush=True)
    print('Space: pause/play; R: restart; S: half speed; arrows: step; close window: exit.', flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.azimuth = 180 if args.view == 'front' else 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 2.4
        print('Window ready.', flush=True)
        previous = time.monotonic()
        scheduled_start = None
        waiting_for_start = args.start_file is not None
        while viewer.is_running():
            now = time.monotonic()
            if waiting_for_start and scheduled_start is None and args.start_file.is_file():
                scheduled_start = float(args.start_file.read_text())
            if waiting_for_start and scheduled_start is not None and now >= scheduled_start:
                with mutex:
                    state.update(frame=(now - scheduled_start) * fps * state['speed'], paused=False)
                previous = now
                waiting_for_start = False
            with mutex:
                if not state['paused']:
                    state['frame'] += (now - previous) * fps * state['speed']
                    if args.loop:
                        state['frame'] %= count
                    elif state['frame'] >= count - 1:
                        state.update(frame=float(count - 1), paused=True)
                        print('Holding last frame. Press R to replay.', flush=True)
                frame = int(state['frame'])
                # Repeat by restarting, without blending different end/start locations.
                next_frame = min(frame + 1, count - 1)
                pose = interpolate(poses[frame], poses[next_frame], state['frame'] - frame)
            previous = now
            with viewer.lock():
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = [*pose[:2], .68]
            viewer.sync()
            time.sleep(1 / 60)


def tile_and_start(children):
    """Arrange only the newly launched viewer windows on an X11 desktop."""
    x = C.CDLL(ctypes.util.find_library('X11'))
    signatures = {
        'XOpenDisplay': ([C.c_char_p], C.c_void_p),
        'XDefaultRootWindow': ([C.c_void_p], C.c_ulong),
        'XDefaultScreen': ([C.c_void_p], C.c_int),
        'XDisplayWidth': ([C.c_void_p, C.c_int], C.c_int),
        'XDisplayHeight': ([C.c_void_p, C.c_int], C.c_int),
        'XInternAtom': ([C.c_void_p, C.c_char_p, C.c_int], C.c_ulong),
        'XGetWindowProperty': ([C.c_void_p, C.c_ulong, C.c_ulong, C.c_long, C.c_long, C.c_int,
            C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_int), C.POINTER(C.c_ulong),
            C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ubyte))], C.c_int),
        'XFree': ([C.c_void_p], C.c_int),
        'XMoveResizeWindow': ([C.c_void_p, C.c_ulong, C.c_int, C.c_int, C.c_uint, C.c_uint], C.c_int),
        'XRaiseWindow': ([C.c_void_p, C.c_ulong], C.c_int),
        'XFlush': ([C.c_void_p], C.c_int),
        'XCloseDisplay': ([C.c_void_p], C.c_int),
    }
    for name, (parameters, result) in signatures.items():
        getattr(x, name).argtypes, getattr(x, name).restype = parameters, result
    display = x.XOpenDisplay(None)
    if not display:
        raise RuntimeError('Cannot access the X11 desktop; press Space in each viewer to start.')

    def property32(window, name):
        kind, fmt, count, remaining = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong()
        data = C.POINTER(C.c_ubyte)()
        status = x.XGetWindowProperty(display, window, x.XInternAtom(display, name.encode(), False),
            0, 4096, False, 0, C.byref(kind), C.byref(fmt), C.byref(count), C.byref(remaining), C.byref(data))
        values = []
        if status == 0 and data:
            if fmt.value == 32:
                values = list(C.cast(data, C.POINTER(C.c_ulong))[:count.value])
            x.XFree(data)
        return values

    try:
        root = x.XDefaultRootWindow(display)
        windows = {}
        deadline = time.monotonic() + 20
        while len(windows) < len(children) and time.monotonic() < deadline:
            for window in property32(root, '_NET_CLIENT_LIST'):
                pid = property32(window, '_NET_WM_PID')
                for process, path, log, trigger in children:
                    if pid == [process.pid] and 'Window ready.' in log.read_text():
                        windows[process.pid] = window
            for process, path, log, trigger in children:
                if process.poll() is not None:
                    raise RuntimeError(f'{path.name} exited: {log.read_text()}')
            time.sleep(.1)
        if len(windows) != len(children):
            raise RuntimeError('Some viewer windows could not be identified; press Space to start.')
        screen = x.XDefaultScreen(display)
        columns = 2
        rows = (len(children) + columns - 1) // columns
        width = (x.XDisplayWidth(display, screen) - 120) // columns
        height = (x.XDisplayHeight(display, screen) - 120) // rows - 25
        for index, (process, path, log, trigger) in enumerate(children):
            window = windows[process.pid]
            x.XMoveResizeWindow(display, window, 70 + index % columns * (width + 10),
                               35 + index // columns * (height + 35), width, height)
            x.XRaiseWindow(display, window)
            print(f'{path.name}: PID {process.pid}, window {hex(window)}', flush=True)
        x.XFlush(display)
        # Each child waits at frame zero until the same start time is published.
        start = time.monotonic() + .5
        for process, path, log, trigger in children:
            pending = trigger.with_suffix('.pending')
            pending.write_text(str(start))
            pending.replace(trigger)
    finally:
        x.XCloseDisplay(display)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', nargs='+', type=Path, default=DEFAULTS)
    parser.add_argument('--view', choices=('front', 'side'), default='front')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--loop', action='store_true', help='Repeat clips; default holds the final frame.')
    parser.add_argument('--check', action='store_true', help='Validate every frame without opening windows.')
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--start-paused', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--start-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive.')
    if args.single and len(args.motion) != 1:
        parser.error('--single requires one motion.')
    for path in args.motion:
        fps, poses = load_motion(path)
        print(f'{path.name}: {len(poses)} frames, {fps:g} FPS, {len(poses) / fps:.3f}s.', flush=True)
        if args.check:
            model = make_model(path.stem)
            data = mujoco.MjData(model)
            for pose in poses:
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.xpos).all():
                    raise ValueError(f'{path}: nonfinite forward kinematics.')
    if args.check:
        print('All motions passed model/FK checks.', flush=True)
        return
    if len(args.motion) == 1:
        play(args.motion[0], args)
        return
    logs = Path(tempfile.mkdtemp(prefix='g1_pkl_mujoco_'))
    children = []
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    print(f'Viewer logs: {logs}', flush=True)
    try:
        for index, path in enumerate(args.motion):
            log, trigger = logs / f'{index}_{path.stem}.log', logs / f'{index}.start'
            command = [sys.executable, str(Path(__file__).resolve()), '--single', '--start-paused',
                '--start-file', str(trigger), '--motion', str(path.resolve()), '--view', args.view,
                '--speed', str(args.speed)]
            if args.loop:
                command.append('--loop')
            with log.open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            children.append((process, path, log, trigger))
        tile_and_start(children)
        while any(process.poll() is None for process, _, _, _ in children):
            time.sleep(.5)
    except (OSError, RuntimeError) as error:
        print(f'{error}\nLogs: {logs}', flush=True)
        # Keep successfully opened windows available, even if desktop tiling failed.
        if not any(process.poll() is None for process, _, _, _ in children):
            raise
    except KeyboardInterrupt:
        for process, _, _, _ in children:
            if process.poll() is None:
                process.terminate()


if __name__ == '__main__':
    main()
