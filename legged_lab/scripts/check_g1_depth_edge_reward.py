"""检查 G1 23dof 脚体积点边缘奖励；只步进仿真，不进行策略学习。

固定使用 2 个环境、32 步零动作，不读取 checkpoint，不创建训练器。
用 IsaacLab Python 运行，默认输出到 work/g1_depth_edge_reward_validation。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--disabled", action="store_true", help="验证关闭奖励后不构建边缘几何。")
    parser.add_argument("--debug_vis", action="store_true", help="在仿真窗口显示体积点和边缘圆柱。")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if args.output_dir is None:
        args.output_dir = ROOT / "work/g1_depth_edge_reward_validation"
        if args.disabled:
            args.output_dir = args.output_dir / "disabled"
    return args


def json_value(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def make_config(args):
    from legged_lab.utils import task_registry

    cfg, _ = task_registry.get_cfgs("g1_depth")
    cfg = copy.deepcopy(cfg)
    cfg.device = args.device
    cfg.scene.num_envs = 2
    cfg.scene.seed = 42
    generator = copy.deepcopy(cfg.scene.terrain_generator)
    generator.num_rows, generator.num_cols = 2, 3
    generator.curriculum = False
    generator.seed = 42
    generator.use_cache = False
    cfg.scene.terrain_generator = generator
    cfg.scene.max_init_terrain_level = 0
    cfg.scene.height_scanner.debug_vis = False
    cfg.scene.camera.camera.debug_vis = False
    cfg.scene.camera.add_camera_noise = False
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.commands.debug_vis = False
    cfg.commands.heading_command = False
    cfg.commands.rel_heading_envs = 0.
    cfg.commands.rel_standing_envs = 1.
    cfg.commands.ranges.lin_vel_x = (0., 0.)
    cfg.commands.ranges.lin_vel_y = (0., 0.)
    cfg.commands.ranges.ang_vel_z = (0., 0.)
    cfg.commands.resampling_time_range = (1e6, 1e6)
    cfg.foot_edge.debug_vis = args.debug_vis and not args.disabled
    cfg.foot_edge.enabled = not args.disabled
    events = cfg.domain_rand.events
    for name in list(vars(events)):
        if name.startswith("_"):
            continue
        if name not in ("reset_base", "reset_robot_joints") and getattr(getattr(events, name), "mode", None):
            setattr(events, name, None)
    events.reset_base.params["pose_range"] = {"x": (0., 0.), "y": (0., 0.), "yaw": (0., 0.)}
    events.reset_base.params["velocity_range"] = {
        k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    events.reset_robot_joints.params["position_range"] = (1., 1.)
    events.reset_robot_joints.params["velocity_range"] = (0., 0.)
    return cfg


def verify_reward_manager(env):
    """固定状态下比较启用/关闭边缘项，正值探针验证权重和 dt。"""
    import torch

    manager = env.reward_manager
    name = "foot_edge_penalty"
    original = manager.get_term_cfg(name)
    episode_sums = {key: value.clone() for key, value in manager._episode_sums.items()}
    step_rewards, reward_buffer = manager._step_reward.clone(), manager._reward_buf.clone()
    result = {}
    try:
        for mode in ("actual_cost", "positive_probe"):
            term = copy.copy(original)
            if mode == "positive_probe":
                def positive_probe(_env):
                    return torch.tensor([.125, .25], device=_env.device)
                term.func = positive_probe
                term.params = {}
            manager.set_term_cfg(name, term)
            raw = term.func(env, **term.params).clone()
            enabled = manager.compute(env.step_dt).clone()
            disabled_term = copy.copy(term)
            disabled_term.weight = 0.
            manager.set_term_cfg(name, disabled_term)
            disabled = manager.compute(env.step_dt).clone()
            expected = raw * term.weight * env.step_dt
            torch.testing.assert_close(enabled - disabled, expected, rtol=1e-4, atol=2e-6)
            manager.set_term_cfg(name, term)
            doubled_dt = manager.compute(2 * env.step_dt).clone()
            manager.set_term_cfg(name, disabled_term)
            doubled_disabled = manager.compute(2 * env.step_dt).clone()
            torch.testing.assert_close(doubled_dt - doubled_disabled, 2 * expected, rtol=1e-4, atol=2e-6)
            result[mode] = {"raw_cost": raw, "weight": term.weight, "dt": env.step_dt,
                            "reward_difference": enabled-disabled,
                            "expected_cost_weight_dt": expected}
    finally:
        manager.set_term_cfg(name, original)
        for key, value in episode_sums.items():
            manager._episode_sums[key].copy_(value)
        manager._step_reward.copy_(step_rewards)
        manager._reward_buf.copy_(reward_buffer)
    return result


def verify_kinematics(device):
    import torch
    from legged_lab.perception.foot_edge import point_kinematics

    point = torch.tensor([[.2, 0., 0.]], device=device)
    link = torch.tensor([[[1., 2., 3.]]], device=device)
    quat = torch.tensor([[[1., 0., 0., 0.]]], device=device)
    com = link + torch.tensor([[[.1, 0., 0.]]], device=device)
    zero = torch.zeros_like(link)
    angular = torch.tensor([[[0., 0., 2.]]], device=device)
    positions, velocities = point_kinematics(point, link, quat, com, zero, angular)
    torch.testing.assert_close(positions, torch.tensor([[[[1.2, 2., 3.]]]], device=device))
    torch.testing.assert_close(velocities, torch.tensor([[[[0., .2, 0.]]]], device=device))
    return {"pure_rotation_with_com_offset": "passed", "point_velocity": velocities}


def verify_geometry(geometry, device):
    import numpy as np
    import torch

    edges = np.asarray(geometry.edges)
    assert len(edges), "缩小后的真实地形未检测到锐边，无法验证查询。"
    segment = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    midpoint = (segment[0] + segment[1]) * .5
    far = np.max(edges.reshape(-1, 3), axis=0) + 100.
    points = torch.tensor(np.stack((midpoint, far)), dtype=torch.float32, device=device)
    offsets = geometry.penetration_offsets(points)
    assert torch.isfinite(offsets).all()
    depths = offsets.norm(dim=-1)
    assert depths[0] > 0, "圆柱轴心未得到正的穿入深度。"
    assert depths[1] == 0, "远离全部边缘的点应当没有穿入。"
    return {"edge_count": len(edges), "axis_midpoint_depth": depths[0], "far_point_depth": depths[1]}


def write_overlay(env, directory):
    """离线 HTML 叠加真实脚网格、100点和实际地形圆柱，无外部 JS 依赖。"""
    import numpy as np
    import torch
    import trimesh
    from isaaclab.utils.math import quat_apply

    monitor = env.foot_edge_monitor
    monitor.cost()
    geometry = env.scene.terrain.foot_edge_geometry
    points = monitor.points_world[0].detach().cpu().numpy()
    foot_triangles = []
    local_triangles = []
    mesh_dir = ROOT / "legged_lab/assets/g1/g1_23dof/meshes"
    for foot_index, side in enumerate(("left", "right")):
        mesh = trimesh.load(mesh_dir / f"{side}_ankle_roll_link.STL", force="mesh")
        triangles = np.asarray(mesh.triangles)
        local_triangles.append(triangles.tolist())
        vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device=env.device)
        body_id = int(monitor.body_ids[foot_index])
        quaternion = env.robot.data.body_link_quat_w[0, body_id]
        position = env.robot.data.body_link_pos_w[0, body_id]
        vertices = quat_apply(quaternion.expand(len(vertices), -1), vertices) + position
        foot_triangles.append(vertices.detach().cpu().numpy()[mesh.faces].tolist())
    center = points.reshape(-1, 3).mean(axis=0)
    all_edges = np.asarray(geometry.edges)
    # 保留跨越视窗的长边，不能只按线段中点筛选。
    a, b = all_edges[:, 0], all_edges[:, 1]
    t = np.clip(((center-a)*(b-a)).sum(-1) / np.maximum(((b-a)**2).sum(-1), 1e-12), 0, 1)
    distances = np.linalg.norm(a+t[:, None]*(b-a)-center, axis=-1)
    nearby = all_edges[distances < 2.0]
    cylinders = []
    for segment in nearby:
        mesh = trimesh.creation.cylinder(radius=env.cfg.foot_edge.geometry.cylinder_radius,
                                        segment=segment, sections=12)
        cylinders.extend(np.asarray(mesh.triangles).tolist())
    terrain = env.scene.terrain.foot_edge_mesh
    terrain_triangles = np.asarray(terrain.triangles)
    keep = np.linalg.norm(terrain_triangles.mean(axis=1)-center, axis=-1) < 2.2
    data = {"world": {"center": center, "terrain": terrain_triangles[keep],
                       "feet": foot_triangles, "cylinders": cylinders, "points": points},
            "local": {"center": [0.04, 0, -.015], "feet": local_triangles,
                       "points": monitor.local_points.detach().cpu().numpy()},
            "nearby_edges": len(nearby), "radius": env.cfg.foot_edge.geometry.cylinder_radius}
    payload = json.dumps(json_value(data), separators=(",", ":"))
    html = """<!doctype html><meta charset="utf-8"><title>G1脚体积点与地形锐边</title>
<style>body{font:15px sans-serif;background:#f4f6f8;color:#172b3a;margin:20px}canvas{background:white;border:1px solid #ccd5df;width:100%;height:450px}.row{display:grid;grid-template-columns:1fr 1fr;gap:18px}label{margin-right:16px}h2{font-size:18px}p{line-height:1.6}</style>
<h1>G1 23dof 脚体积点与有限圆柱</h1><p>拖动旋转，滚轮缩放。左侧为第0环境真实世界坐标快照，右侧为左右脚局部坐标叠加（对应网格相近）。
灰色：物理地形；蓝色：左脚真实STL；紫色：右脚真实STL；橙色：惩罚圆柱；红/绿点：两脚体积点。圆柱为训练几何，不是物理碰撞体。</p>
<label><input id="terrain" type="checkbox" checked>地形</label><label><input id="cylinders" type="checkbox" checked>圆柱</label><label><input id="feet" type="checkbox" checked>脚模型</label><label><input id="points" type="checkbox" checked>体积点</label>
<div class="row"><section><h2>实际地形与脚（世界坐标）</h2><canvas id="world"></canvas></section><section><h2>G1脚模型与100点（局部坐标）</h2><canvas id="local"></canvas></section></div>
<p id="note"></p><script>const data=__PAYLOAD__;const views=[];
function create(id){const canvas=document.getElementById(id),d=data[id];const v={canvas,d,yaw:-.6,pitch:.8,scale:id==='world'?130:1900};views.push(v);let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;v.yaw+=(e.clientX-drag[0])*.008;v.pitch=Math.max(-1.5,Math.min(1.5,v.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw(v)};canvas.onwheel=e=>{e.preventDefault();v.scale*=Math.exp(-e.deltaY*.001);draw(v)};draw(v)}
function draw(v){const c=v.canvas;c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;const ctx=c.getContext('2d');ctx.scale(devicePixelRatio,devicePixelRatio);const w=c.clientWidth,h=c.clientHeight;ctx.clearRect(0,0,w,h);function p(q){let[x,y,z]=q.map((t,i)=>t-v.d.center[i]);let a=x*Math.cos(v.yaw)-y*Math.sin(v.yaw),b=x*Math.sin(v.yaw)+y*Math.cos(v.yaw);return[w/2+a*v.scale,h/2-(z*Math.cos(v.pitch)-b*Math.sin(v.pitch))*v.scale,b*Math.cos(v.pitch)+z*Math.sin(v.pitch)]}const faces=[];function mesh(tris,color){for(const tri of tris){const q=tri.map(p);faces.push({q,color,z:q.reduce((s,t)=>s+t[2],0)/3})}}if(v.d.terrain&&document.getElementById('terrain').checked)mesh(v.d.terrain,'#c0c6ce');if(v.d.cylinders&&document.getElementById('cylinders').checked)mesh(v.d.cylinders,'#e7a24c88');if(document.getElementById('feet').checked){mesh(v.d.feet[0],'#3178b5aa');mesh(v.d.feet[1],'#9669b5aa')}faces.sort((a,b)=>b.z-a.z);for(const f of faces){ctx.beginPath();f.q.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle='#53647522';ctx.lineWidth=.25;ctx.stroke()}if(document.getElementById('points').checked){const groups=v.d.terrain?v.d.points:[v.d.points];groups.forEach((pts,i)=>pts.forEach(q=>{const a=p(q);ctx.beginPath();ctx.arc(a[0],a[1],2,0,Math.PI*2);ctx.fillStyle=i?'#1c8e5b':'#d33f3f';ctx.fill()}))}ctx.fillStyle='#172b3a';ctx.font='13px sans-serif';ctx.fillText('拖动旋转 · 滚轮缩放 · 单位 m',12,22)}
create('world');create('local');for(const el of document.querySelectorAll('input'))el.onchange=()=>views.forEach(draw);window.onresize=()=>views.forEach(draw);document.getElementById('note').textContent=`附近锐边 ${data.nearby_edges} 条，圆柱半径 ${data.radius} m。此文件仅验证坐标和几何，不表示策略性能。`;</script>""".replace("__PAYLOAD__", payload)
    path = directory / "foot_edge_overlay.html"
    path.write_text(html)
    return {"path": str(path.resolve()), "nearby_edges": len(nearby), "point_count_per_foot": points.shape[1]}


def run_check(args):
    import torch
    import legged_lab.envs  # noqa: F401 — 仿真应用启动后才注册任务。
    from legged_lab.utils import task_registry
    from isaaclab.utils.io import dump_yaml

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"task": "g1_depth", "seed": 42, "num_envs": 2, "fixed_action_steps": 32,
              "action": "zero_joint_position_offset", "learning_executed": False,
              "checkpoint_loaded": False, "edge_reward_enabled": not args.disabled,
              "scope": "仿真接入检查，不是策略性能验证。"}
    try:
        cfg = make_config(args)
        roundtrip_cfg = copy.deepcopy(cfg)
        roundtrip_cfg.from_dict(cfg.to_dict())
        # BaseSceneCfg仍含由SceneCfg运行时填入的prim_path占位值，不能在此
        # 对整个任务模板调用validate；往返一致性和新配置独立校验在这里验证。
        roundtrip_cfg.foot_edge.validate()
        assert roundtrip_cfg.foot_edge == cfg.foot_edge
        dump_yaml(str(args.output_dir / "env_cfg.yaml"), cfg)
        report["configuration"] = {"to_dict_from_dict_roundtrip": "passed",
                                   "validate": "passed", "yaml": "env_cfg.yaml"}
        report["terrain"] = {"rows": 2, "columns": 3, "curriculum": False,
                             "candidate_types": list(cfg.scene.terrain_generator.sub_terrains),
                             "coverage_note": "6块地形随机采样，不保证覆盖全部候选类型。"}
        environment_class = task_registry.get_task_class("g1_depth")
        environment_class.seed(42)
        env = environment_class(cfg, args.headless)
        env.foot_edge_monitor.save_metadata(args.output_dir)
        geometry = getattr(env.scene.terrain, "foot_edge_geometry", None)
        if args.disabled:
            assert geometry is None
            assert type(env.scene.terrain).__name__ == "TerrainImporter"
        else:
            assert geometry is not None
        construction_count = getattr(geometry, "construction_count", None)
        edges_identity = id(geometry)
        obs, info = env.get_observations()
        assert obs.shape == (2, 1932), obs.shape
        assert info["observations"]["critic"].shape == (2, 1037)
        active = list(env.reward_manager.active_terms)
        assert ("foot_edge_penalty" in active) == (not args.disabled)
        assert not any("support" in name or "stair_edge_penalty" in name for name in active)
        assert env.foot_edge_monitor.local_points.shape == (100, 3)
        report["reward_terms"] = active
        report["observations"] = {"actor": list(obs.shape), "critic": [2, 1037], "actions": env.num_actions}
        report["kinematics"] = verify_kinematics(env.device)
        # 正常 step 会先判断终止再计算奖励；这里复现该顺序，不额外步进物理。
        env.reset_buf, env.time_out_buf = env.check_reset()
        if args.disabled:
            torch.testing.assert_close(env.foot_edge_monitor.cost(), torch.zeros(2, device=env.device))
            report["geometry_query"] = "not_constructed"
            report["reward_manager"] = "edge_term_absent"
            report["overlay"] = "not_constructed"
        else:
            report["geometry_query"] = verify_geometry(geometry, env.device)
            report["reward_manager"] = verify_reward_manager(env)
            report["overlay"] = write_overlay(env, args.output_dir)
            env.foot_edge_monitor.set_debug_vis(True)
            env.foot_edge_monitor.reset()
            env.foot_edge_monitor.cost()  # 实际提交marker位置，不只测试marker构造。
            env.foot_edge_monitor.set_debug_vis(args.debug_vis)
        metrics = []
        action = torch.zeros((2, env.num_actions), device=env.device)
        with torch.no_grad():
            for index in range(32):
                obs, reward, done, info = env.step(action)
                assert obs.shape == (2, 1932)
                assert info["observations"]["critic"].shape == (2, 1037)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                diagnostics = env.foot_edge_monitor.diagnostics()
                assert all(torch.isfinite(torch.as_tensor(value)).all() for value in diagnostics.values())
                if args.disabled:
                    torch.testing.assert_close(env.foot_edge_monitor.cost(), torch.zeros(2, device=env.device))
                    assert getattr(env.scene.terrain, "foot_edge_geometry", None) is None
                metrics.append({"step": index + 1, "reward": reward.clone(), "done": done.clone(),
                                "post_step_diagnostics": json_value(diagnostics),
                                "pre_reset_log": json_value(info.get("log", {}))})
        # 这里显式清缓存和 reset；不增加物理步数，不重建静态地形。
        ids = torch.tensor([0], device=env.device)
        env.foot_edge_monitor.reset(ids)
        env.reset(ids)
        assert getattr(env.scene.terrain, "foot_edge_geometry", None) is geometry
        assert id(geometry) == edges_identity
        if construction_count is not None:
            assert geometry.construction_count == construction_count
        assert torch.isfinite(env.foot_edge_monitor.cost()).all()
        report["reset"] = {"same_geometry_object": True, "construction_count": construction_count,
                            "post_reset_cost_finite": True}
        report["geometry"] = geometry.metadata() if geometry is not None else "not_constructed"
        report["steps"] = metrics
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output_dir / "isaac_validation.json").write_text(
            json.dumps(json_value(report), indent=2, ensure_ascii=False))
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "isaac_validation.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import os
    import traceback

    args = parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        run_check(args)
    except BaseException:
        # Kit 的 headless 快速关闭可能以 0 退出；先输出完整异常并保留失败退出码。
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if args.headless:
            os._exit(1)
        simulation_app.close()
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(skip_cleanup=args.headless)
