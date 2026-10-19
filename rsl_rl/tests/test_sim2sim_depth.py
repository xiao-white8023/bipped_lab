"""Depth sim2sim contracts, native camera geometry, timing and control."""

from __future__ import annotations

import ast
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch


mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from legged_lab.scripts import sim2sim_depth as depth


@pytest.fixture
def contract():
    return {
        "version": 1, "task": "g1_depth", "actor_observation_dim": 1932,
        "critic_observation_dim": 1037, "action_dim": 23,
        "proprio_history_frames": 10, "proprio_frame_dim": 78,
        "proprio_order": depth.PROPRIO_ORDER.copy(), "joint_names": list(depth.ISAAC_JOINT_NAMES),
        "obs_scales": {n: 1.0 for n in depth.SCALE_NAMES}, "depth_shape": [2, 18, 32],
        "depth_capture_period_s": .1, "sequence_order": "oldest_to_current",
        "depth_crop": [18, 0, 16, 16], "depth_min_m": .1, "depth_max_m": 2.5,
        "depth_normalization": "clip_metres_divide_depth_max", "depth_invalid_fill": "nonfinite_to_depth_max",
        "camera_size": [36, 64], "camera_focal_length": 1.0,
        "camera_horizontal_aperture": 2*np.tan(np.deg2rad(89.51)/2),
        "camera_vertical_aperture": 2*np.tan(np.deg2rad(58.29)/2),
        "camera_position": [.0576235, .01753, .42987], "camera_rotation": [.9149595, 0., .4035447, 0.],
        "camera_convention": "world", "camera_ray_alignment": "yaw", "control_dt_s": .02,
        "action_scale": .25, "clip_observations": 100., "clip_actions": 100.,
    }


def metadata(contract):
    return {"version": 1, "task": "g1_depth", "input_shape": ["batch", 1932],
            "output_shape": ["batch", 23], "observation_contract": contract}


class ConstantPolicy(torch.nn.Module):
    def __init__(self, value=0., width=23):
        super().__init__()
        self.value, self.width = value, width

    def forward(self, obs):
        return obs[:, :self.width]*0 + self.value


@pytest.fixture
def policy_file(tmp_path, contract):
    p = tmp_path / "policy.pt"
    model = torch.jit.trace(ConstantPolicy(), torch.zeros(1, 1932))
    torch.jit.save(model, str(p), _extra_files={"depth_contract.json": json.dumps(metadata(contract))})
    return p


@pytest.fixture
def runner(policy_file):
    with depth.G1DepthMujocoRunner(depth.G1DepthSim2SimCfg(headless=True, duration=.24),
                                  policy_file, depth.get_available_scenes()["flat"]) as instance:
        yield instance


def source_asset_config():
    """Read the actual training asset without importing Isaac Sim."""
    p = ROOT / "legged_lab/assets/g1/g1_29.py"
    tree = ast.parse(p.read_text())
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "G1_23CFG")
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    initial = next(kw.value for kw in kwargs["init_state"].keywords if kw.arg == "joint_pos")
    groups = {}
    for name, constructor in zip(kwargs["actuators"].keys, kwargs["actuators"].values):
        groups[ast.literal_eval(name)] = {kw.arg: ast.literal_eval(kw.value) for kw in constructor.keywords}
    return ast.literal_eval(initial), groups


def test_pd_parameters_and_self_return_meshes_match_training_asset():
    defaults, groups = source_asset_config()
    for name in depth.ISAAC_JOINT_NAMES:
        expected_position = 0.0
        for pattern, value in defaults.items():
            if re.fullmatch(pattern, name):
                expected_position = value
        selected = [g for g in groups.values() if any(re.fullmatch(p, name) for p in g["joint_names_expr"])]
        assert len(selected) == 1
        group = selected[0]

        def select(field):
            value = group[field]
            return next(v for p, v in value.items() if re.fullmatch(p, name)) if isinstance(value, dict) else value

        assert depth.joint_control_parameters(name) == (
            expected_position, select("stiffness"), select("damping"), select("effort_limit_sim"), select("armature"))
    tree = ast.parse((ROOT / "legged_lab/assets/g1/g1_29.py").read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "G1_23DOF_LINKS")
    assert depth.SELF_RETURN_BODIES == set(names)


@pytest.mark.parametrize("field,value", [
    ("task", "g1_renet"), ("actor_observation_dim", 2120), ("depth_shape", [2, 36, 64]),
    ("joint_names", ["left_hip_pitch_joint"]*23), ("camera_rotation", [0., 0., 0., 0.]),
    ("camera_position", ["bad", 0., 0.]),
    ("camera_focal_length", float("nan")), ("depth_crop", [0, 0, 0, 0]),
    ("control_dt_s", .01), ("depth_capture_period_s", .02),
])
def test_incompatible_observation_contract_rejected(contract, field, value):
    contract[field] = value
    with pytest.raises(ValueError):
        depth.validate_depth_contract(metadata(contract))


def test_loading_checks_embedded_identity_and_actual_output_shape(tmp_path, policy_file, contract):
    p, c = depth.load_depth_policy(policy_file)
    assert p(torch.zeros(1, 1932)).shape == (1, 23) and c == contract
    model = torch.jit.trace(ConstantPolicy(), torch.zeros(1, 1932))
    path = tmp_path / "wrong.pt"
    torch.jit.save(model, str(path))
    with pytest.raises(ValueError, match="no embedded"):
        depth.load_depth_policy(path)
    bad = metadata(contract)
    bad["task"] = "g1_renet"
    torch.jit.save(model, str(path), _extra_files={"depth_contract.json": json.dumps(bad)})
    with pytest.raises(ValueError, match="g1_depth v1"):
        depth.load_depth_policy(path)
    wrong_shape = torch.jit.trace(ConstantPolicy(width=22), torch.zeros(1, 1932))
    torch.jit.save(wrong_shape, str(path), _extra_files={"depth_contract.json": json.dumps(metadata(contract))})
    with pytest.raises(ValueError, match="23 finite actions"):
        depth.load_depth_policy(path)


def test_named_mapping_survives_joint_policy_and_actuator_reordering(runner):
    xml = depth._read_mjcf(depth.MJCF_DIR / "g1_23dof_rev_1_0.xml")
    compiler = xml.find("compiler")
    compiler.set("meshdir", str(depth.MJCF_DIR / "meshes"))
    world = xml.find("worldbody")
    pelvis = world.find("body")
    # Exchange the two complete leg trees: qpos order changes along with actuators.
    left, right = pelvis.find("body[@name='left_hip_pitch_link']"), pelvis.find("body[@name='right_hip_pitch_link']")
    li, ri = list(pelvis).index(left), list(pelvis).index(right)
    pelvis.remove(left)
    pelvis.remove(right)
    pelvis.insert(li, right)
    pelvis.insert(ri, left)
    actuator = xml.find("actuator")
    actuator[:] = list(actuator)[::-1]
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    names = list(depth.ISAAC_JOINT_NAMES)[::-1]
    mapping = depth.JointMapping.from_model(model, names)
    for i, name in enumerate(names):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert mapping.qpos[i] == model.jnt_qposadr[joint]
        assert mapping.qvel[i] == model.jnt_dofadr[joint]
        assert model.actuator_trnid[mapping.ctrl[i], 0] == joint
    assert not np.array_equal(mapping.ctrl, runner.mapping.ctrl)
    np.testing.assert_allclose(mapping.default, runner.mapping.default[::-1])


def test_invalid_joint_and_actuator_models_are_rejected(runner):
    model = copy.copy(runner.model)
    model.actuator_trnid[0, 0] = model.actuator_trnid[1, 0]
    with pytest.raises(ValueError, match="exactly one"):
        depth.JointMapping.from_model(model)
    model = copy.copy(runner.model)
    model.actuator_gear[0, 0] = 2
    with pytest.raises(ValueError, match="unit-gear"):
        depth.JointMapping.from_model(model)
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match="23 G1"):
        depth.JointMapping.from_model(model)


def test_proprio_frame_matches_body_rotation_scales_and_previous_action(runner):
    r = runner
    axis = np.array([1., 2., -1.]); axis /= np.linalg.norm(axis)
    quat = np.r_[np.cos(.4), axis*np.sin(.4)]
    r.data.qpos[r.mapping.root_qpos+3:r.mapping.root_qpos+7] = quat
    r.data.qvel[r.mapping.root_qvel+3:r.mapping.root_qvel+6] = [.1, -.2, .3]
    r.data.qpos[r.mapping.qpos] += np.linspace(-.03, .03, 23)
    r.data.qvel[r.mapping.qvel] = np.linspace(-.3, .3, 23)
    r.action[:] = np.arange(23) / 100.
    r.contract["obs_scales"].update(ang_vel=2., projected_gravity=3., commands=.5, joint_pos=4., joint_vel=.2, actions=5.)
    mujoco.mj_forward(r.model, r.data)
    x = r.sample_proprio()
    np.testing.assert_allclose(x[:3], np.array([.1, -.2, .3])*2, atol=1e-6)
    np.testing.assert_allclose(x[3:6], depth.quat_wxyz_to_matrix(quat).T @ [0., 0., -3.], atol=1e-6)
    np.testing.assert_allclose(x[6:9], r.command_vel*.5)
    np.testing.assert_allclose(x[9:32], np.linspace(-.03, .03, 23)*4, atol=1e-6)
    np.testing.assert_allclose(x[32:55], np.linspace(-.3, .3, 23)*.2, atol=1e-6)
    np.testing.assert_allclose(x[55:], r.action*5, atol=1e-6)


def test_history_capture_clock_reset_and_applied_action(runner):
    r = runner
    before = r.build_policy_obs().clone()
    assert before.shape == (1, 1932) and r.capture_times == [0.]
    np.testing.assert_array_equal(r.proprio_history, np.repeat(r.proprio_history[:1], 10, axis=0))
    torch.testing.assert_close(r.depth_history[0], r.depth_history[1], atol=0, rtol=0)
    r.build_policy_obs()
    assert r.capture_count == 1
    r.policy = lambda obs: torch.full((1, 23), .3)
    r.step()
    next_obs = r.build_policy_obs()
    np.testing.assert_allclose(r.proprio_history[-1, 55:], .3)
    torch.testing.assert_close(next_obs[0, :702], before[0, 78:780])
    r.build_policy_obs()  # Reading twice never consumes a history slot.
    assert r.capture_count == 1
    for _ in range(10):
        r.step()
    assert r.capture_times == pytest.approx([0., .1, .2], abs=1e-8)
    r.command_vel[:] = [-.2, .1, .3]
    r.reset_robot()
    assert r.capture_times == [0.] and r.data.time == 0 and r.reset_count == 1
    assert not r.action.any() and not r.proprio_history[:, 55:].any()
    np.testing.assert_allclose(r.proprio_history[0, 6:9], r.command_vel)
    np.testing.assert_array_equal(r.proprio_history, np.repeat(r.proprio_history[:1], 10, axis=0))
    torch.testing.assert_close(r.depth_history[0], r.depth_history[1], atol=0, rtol=0)


def test_pixel_centres_and_optical_depth_for_wall_and_floor(contract):
    rays = depth.pinhole_rays(contract)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1., atol=1e-12)
    fx = 64/contract["camera_horizontal_aperture"]
    fy = 36/contract["camera_vertical_aperture"]
    np.testing.assert_allclose(rays[0]/rays[0, 0], [1, 31.5/fx, 17.5/fy])
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" pos="1.1 0 0" size=".1 5 5"/></worldbody></mujoco>')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    distances, geoms = np.empty(len(rays)), np.empty(len(rays), dtype=np.int32)
    mujoco.mj_multiRay(model, data, np.zeros(3), rays.ravel(), None, 1, -1, geoms, distances, len(rays), 10.)
    np.testing.assert_allclose(distances*rays[:, 0], 1., atol=1e-12)
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="0 0 .1"/></worldbody></mujoco>')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    angle = np.pi/4
    rot = depth.quat_wxyz_to_matrix([np.cos(angle/2), 0, np.sin(angle/2), 0])
    directions = np.ascontiguousarray(rays @ rot.T)
    mujoco.mj_multiRay(model, data, np.array([0., 0., 1.]), directions.ravel(), None, 1, -1,
                      geoms, distances, len(rays), 10.)
    hits = distances >= 0
    np.testing.assert_allclose(distances[hits], -1/directions[hits, 2], atol=1e-10)


def test_runner_camera_extrinsics_follow_full_torso_pose_and_keep_axial_depth(runner, monkeypatch):
    r = runner
    axis = np.array([1., -.5, 2.]); axis /= np.linalg.norm(axis)
    quat = np.r_[np.cos(.3), axis*np.sin(.3)]
    r.data.qpos[r.mapping.root_qpos+3:r.mapping.root_qpos+7] = quat
    mujoco.mj_forward(r.model, r.data)
    torso = r.data.xmat[r.torso_id].reshape(3, 3)
    saved = {}

    def raycast(model, data, origin, vectors, groups, static, excluded, geoms, distances, nray, cutoff):
        saved.update(origin=origin.copy(), vectors=vectors.reshape(-1, 3).copy())
        geoms[:] = 0
        distances[:] = 1/r.camera_rays[:, 0]
        geoms[0] = -1
        distances[0] = -1

    monkeypatch.setattr(depth.mujoco, "mj_multiRay", raycast)
    image = r.sample_raw_depth()
    np.testing.assert_allclose(saved["origin"], r.data.xpos[r.torso_id]+torso@r.contract["camera_position"])
    np.testing.assert_allclose(saved["vectors"], r.camera_rays@(torso@r.camera_rotation).T, atol=1e-12)
    np.testing.assert_allclose(image.reshape(-1)[1:], 1., atol=1e-12)
    assert np.isinf(image[0, 0])


def test_sensor_groups_match_training_visuals_and_do_not_change_collision_bits(runner):
    model = copy.copy(runner.model)
    # Runner already grouped this copy, so compare its groups to named ownership.
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if model.geom_group[i] == 1:
            assert name in depth.SELF_RETURN_BODIES
    original = mujoco.MjModel.from_xml_path(str(depth.MJCF_DIR / "g1_23dof_rev_1_0.xml"))
    bits = (original.geom_contype.copy(), original.geom_conaffinity.copy())
    groups = depth.configure_sensor_geom_groups(original, runner.mapping.root_body)
    np.testing.assert_array_equal(original.geom_contype, bits[0])
    np.testing.assert_array_equal(original.geom_conaffinity, bits[1])
    np.testing.assert_array_equal(groups, [1, 1, 0, 0, 0, 0])


def training_depth_preprocessor(raw, contract, noise):
    """Execute the actual training methods with the same global random draws."""
    import torchvision.transforms as T

    envfile = ROOT / "legged_lab/envs/g1/g1_depth_env.py"
    tree = ast.parse(envfile.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1DepthEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_processed_deepcamera")
    functions = ast.parse((ROOT / "legged_lab/sensors/camera_noise/camera_noise.py").read_text())
    functions = [n for n in functions.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("compute_distance_dependent_gaussian_std", "distance_dependent_gaussian_noise")]
    scope = {"torch": torch, "T": T, "math": __import__("math"), "Sequence": list,
             "DistanceDependentGaussianNoiseCfg": SimpleNamespace}
    exec(compile(ast.Module(body=[*functions, method], type_ignores=[]), str(envfile), "exec"), scope)
    cfg = SimpleNamespace(depth_blur_probability=.5, depth_noise=SimpleNamespace(near_std=0., far_std=.1, distance_exponent=2.),
        robot=SimpleNamespace(depth_crop=contract["depth_crop"], depth_max=contract["depth_max_m"]),
        scene=SimpleNamespace(camera=SimpleNamespace(camera=SimpleNamespace(
            pattern_cfg=SimpleNamespace(height=36, width=64), min_distance=contract["depth_min_m"]))))
    env = SimpleNamespace(num_envs=1, device="cpu", cfg=cfg, _depth_noise_is_active=lambda: noise,
                          depth_current_valid=torch.zeros(1, 18, 32),
                          camera=SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": raw[None, ..., None]})))
    return scope[method.name](env)[0]


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_preprocessing_matches_actual_training_blur_gaussian_and_clipping(contract, noise, seed, monkeypatch):
    import torchvision.transforms as T

    # Fix the blur sigma draw at its configured value. torchvision otherwise
    # consumes an extra RNG sample even when its min/max sigma are both 1.0.
    monkeypatch.setattr(T.GaussianBlur, "get_params", staticmethod(lambda low, high: 1.0))
    raw = torch.linspace(.01, 3., 36*64).reshape(36, 64)
    raw[18, 16] = float("nan")
    raw[19, 17] = float("inf")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected = training_depth_preprocessor(raw.clone(), contract, noise)
        torch.manual_seed(seed)
        actual = depth.process_depth(raw, contract, noise=noise)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    assert torch.isfinite(actual).all() and actual.min() >= .04 and actual.max() <= 1


def test_noise_is_reproducible_and_does_not_corrupt_commands_or_previous_action(contract):
    raw = np.full((36, 64), 1.2)
    a = depth.process_depth(raw, contract, noise=True, generator=torch.Generator().manual_seed(8))
    b = depth.process_depth(raw, contract, noise=True, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert a.std() > 0 and not torch.equal(a, depth.process_depth(raw, contract))
    vector = depth.proprio_noise_vector(contract["obs_scales"])
    np.testing.assert_array_equal(vector[6:9], 0)
    np.testing.assert_array_equal(vector[55:], 0)


def test_actual_proprio_noise_matches_scales_and_preserves_clean_fields(runner):
    r = runner
    clean = r.sample_proprio()
    r.cfg.proprio_noise = True
    r.proprio_rng.manual_seed(10)
    first = r.sample_proprio()
    r.proprio_rng.manual_seed(10)
    np.testing.assert_array_equal(r.sample_proprio(), first)
    delta = first-clean
    assert np.any(delta != 0)
    assert (np.abs(delta) <= r.noise_vector+1e-7).all()
    np.testing.assert_array_equal(first[6:9], clean[6:9])
    np.testing.assert_array_equal(first[55:], clean[55:])


@pytest.mark.parametrize("scene", ["flat", "mix", "slope", "gaps"])
def test_scenes_compile_without_source_changes_and_temporary_files_are_cleaned(scene):
    files = list(depth.MJCF_DIR.glob("*.xml"))
    before = {p: p.read_bytes() for p in files}
    prepared, temporary = depth.prepare_mjcf(depth.get_available_scenes()[scene])
    directory = Path(temporary.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(prepared))
        assert model.nu == 23
        floors = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert len(floors) == (1 if scene in ("flat", "slope") else 0)
        if scene == "gaps":
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            distance = mujoco.mj_ray(model, data, np.array([1.65, 0., 1.]), np.array([0., 0., -1.]),
                                     None, 1, -1, np.empty(1, dtype=np.int32))
            assert distance == pytest.approx(1.5)  # Pit floor, not an invisible z=0 plane.
        assert all(p.read_bytes() == data for p, data in before.items())
    finally:
        temporary.cleanup()
    assert not directory.exists()


def test_relative_nested_includes_assets_and_cycles(tmp_path):
    included = tmp_path / "parts"; included.mkdir()
    meshdir = included / "meshes"; meshdir.mkdir()
    mesh = meshdir / "test.obj"
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    (included / "assets.xml").write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh name="m" file="test.obj"/></asset></mujoco>')
    (included / "body.xml").write_text('<mujoco><include file="assets.xml"/><worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>')
    scene = tmp_path / "main.xml"
    scene.write_text('<mujoco><include file="parts/body.xml"/></mujoco>')
    prepared, temporary = depth.prepare_mjcf(scene)
    try:
        assert mujoco.MjModel.from_xml_path(str(prepared)).ngeom == 1
    finally:
        temporary.cleanup()
    scene.write_text('<mujoco><include file="main.xml"/></mujoco>')
    with pytest.raises(ValueError, match="Cyclic"):
        depth.prepare_mjcf(scene)


def test_custom_fixed_body_terrain_replaces_included_flat_floor(tmp_path):
    scene = tmp_path / "body_terrain.xml"
    robot = depth.MJCF_DIR / "g1_23dof_rev_1_0.xml"
    scene.write_text(f'<mujoco><include file="{robot}"/><worldbody><body name="terrain">'
                     '<geom name="platform" type="box" pos="0 0 -.3" size="3 3 .1"/>'
                     '</body></worldbody></mujoco>')
    prepared, temporary = depth.prepare_mjcf(scene)
    try:
        model = mujoco.MjModel.from_xml_path(str(prepared))
        assert not (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any()
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "platform") >= 0
    finally:
        temporary.cleanup()


def test_pd_clips_every_physics_step_and_rejects_nonfinite_actions(runner, monkeypatch):
    r = runner
    r.policy = lambda obs: torch.full((1, 23), 200.)
    captures = []
    original = depth.mujoco.mj_step

    def capture(model, data):
        captures.append(data.ctrl.copy())
        original(model, data)

    monkeypatch.setattr(depth.mujoco, "mj_step", capture)
    r.step()
    assert len(captures) == 4
    assert (r.action == 100).all()
    assert all((np.abs(ctrl[r.mapping.ctrl]) <= r.mapping.torque_limit).all() for ctrl in captures)
    r.reset_robot()
    captures.clear()
    r.policy = lambda obs: torch.full((1, 23), .1)
    r.step()
    assert len(captures) == 4
    assert not np.array_equal(captures[0], captures[-1])  # Recompute PD inside decimation.
    r.policy = lambda obs: torch.full((1, 23), float("nan"))
    before = r.data.time
    with pytest.raises(RuntimeError, match="action"):
        r.step()
    assert r.data.time == before
    r.data.qvel[0] = float("nan")
    with pytest.raises(RuntimeError, match="state"):
        r.build_policy_obs()


def test_headless_run_has_no_gui_imports_and_records_timing(runner, monkeypatch):
    gui = {"glfw", "mujoco.viewer", "pygame", "matplotlib.pyplot"}
    original = __import__("builtins").__import__

    def guard(name, *args, **kwargs):
        if name in gui:
            raise AssertionError(f"Headless mode imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guard)
    report = runner.run()
    assert report["control_steps"] == 12 and report["simulation_time_s"] == pytest.approx(.24)
    assert report["depth_captures"] == 3 and report["real_time_factor"] > 0
    assert report["action_dim"] == 23 and report["observation_dim"] == 1932
    assert np.isfinite(report["tracking_xy_mse"]) and report["wall_time_s"] > 0


def test_display_failure_is_actionable_and_context_cleanup_is_idempotent(runner, monkeypatch):
    runner.cfg.headless = False
    monkeypatch.setitem(sys.modules, "glfw", SimpleNamespace(init=lambda: False))
    with pytest.raises(RuntimeError, match="--headless"):
        runner.init_interfaces()
    directory = runner.model_path.parent
    runner.close()
    runner.close()
    assert not directory.exists()


def test_keyboard_and_joystick_stop_takes_precedence_over_axes(runner):
    r = runner
    r.key_callback(32); r.update_inputs()
    assert not r.command_vel.any() and r._stopped
    r.key_callback(ord("R")); r.update_inputs()
    assert not r._stopped and not r.action.any()
    np.testing.assert_allclose(r.command_vel, r.cfg.command)
    buttons = SimpleNamespace(A=0, START=7, SELECT=6)
    state = {"pressed": {0}, "released": set()}
    r.joystick_buttons = buttons
    r.joystick = SimpleNamespace(update=lambda: None,
        is_button_pressed=lambda b: b in state["pressed"],
        is_button_released=lambda b: b in state["released"], get_axis_value=lambda i: -1.)
    try:
        r.update_inputs()
        assert not r.command_vel.any()
        state["pressed"] = set()
        r.update_inputs()
        np.testing.assert_allclose(r.command_vel, [1., .5, 1.57])
        state["pressed"] = {6}
        r.update_inputs()
        assert r._exit_event.is_set()
    finally:
        r.joystick = None


def test_latest_policy_and_scene_listing_do_not_require_a_model(tmp_path):
    assert depth.find_latest_exported_policy(tmp_path) is None
    p = tmp_path / "run/exported/policy.pt"
    p.parent.mkdir(parents=True); p.write_bytes(b"not loaded for discovery")
    assert depth.find_latest_exported_policy(tmp_path) == p
    assert set(depth.get_available_scenes()) >= {"flat", "mix", "slope", "gaps"}
