from __future__ import annotations

import ast
import importlib.util
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners.renet_amp_on_policy_runner import RENetAmpOnPolicyRunner
from rsl_rl.algorithms.renet_amp_ppo import RENetAMPPPO


REPO_ROOT = Path(__file__).parents[2]
ENV_PATH = REPO_ROOT / "legged_lab" / "envs" / "g1" / "RENet_env.py"
LOADER_PATH = REPO_ROOT / "legged_lab" / "utils" / "recovery_reset_motion_loader.py"


def _load_reset_loader_module():
    spec = importlib.util.spec_from_file_location("test_recovery_reset_loader", LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_env_methods(*method_names):
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    class_body = next(
        node.body
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "G1RENetEnv"
    )
    selected = []
    for method_name in method_names:
        method = next(
            node
            for node in class_body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method.decorator_list = []
        selected.append(method)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"torch": torch, "math": math, "warnings": warnings}
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in method_names)


def _valid_payload(module, num_frames=3):
    frames = torch.zeros(num_frames, module.RECOVERY_RESET_FRAME_SIZE)
    frames[:, 3] = 1.0
    return {
        "Format": module.RECOVERY_RESET_FORMAT,
        "FrameSize": module.RECOVERY_RESET_FRAME_SIZE,
        "QuaternionConvention": "WXYZ",
        "RootLinearVelocityFrame": "world",
        "RootAngularVelocityFrame": "world",
        "RootXYMode": module.RECOVERY_RESET_ROOT_XY_MODE,
        "JointNames": list(module.RECOVERY_RESET_JOINT_NAMES),
        "Frames": frames.tolist(),
    }


def _write_payload(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recovery_reset_loader_loads_valid_59d_and_samples_shape(tmp_path):
    module = _load_reset_loader_module()
    path = _write_payload(tmp_path / "valid.txt", _valid_payload(module, 5))
    loader = module.RecoveryResetMotionLoader([path])

    samples, motion_ids, frame_ids = loader.sample(17, return_indices=True)

    assert samples.shape == (17, 59)
    assert samples[:, 0:7].shape == (17, 7)
    assert samples[:, 7:13].shape == (17, 6)
    assert samples[:, 13:36].shape == (17, 23)
    assert samples[:, 36:59].shape == (17, 23)
    assert torch.isfinite(samples).all()
    assert motion_ids.shape == (17,)
    assert frame_ids.shape == (17,)
    assert torch.count_nonzero(motion_ids) == 0
    assert torch.all((frame_ids >= 0) & (frame_ids < 5))
    assert loader.joint_names == list(module.RECOVERY_RESET_JOINT_NAMES)


def test_repository_recovery_reset_crops_all_load_as_59d():
    module = _load_reset_loader_module()
    reset_dir = (
        REPO_ROOT / "legged_lab" / "envs" / "g1" / "datasets" / "motion_recovery_reset"
    )
    motion_files = sorted(reset_dir.glob("fallAndGetUp2_subject2_reset_crop_*.txt"))

    loader = module.RecoveryResetMotionLoader(motion_files)

    assert loader.num_motions == 8
    assert all(frame.shape[1] == 59 for frame in loader.motion_frames)
    assert loader.trajectory_num_frames == (105, 145, 161, 53, 269, 93, 109, 69)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("FrameSize", 53, "FrameSize"),
        ("Format", "AMP53D", "Format"),
        ("QuaternionConvention", "XYZW", "QuaternionConvention"),
        ("RootAngularVelocityFrame", "body", "RootAngularVelocityFrame"),
        ("RootLinearVelocityFrame", "body", "RootLinearVelocityFrame"),
        ("RootXYMode", "source", "RootXYMode"),
    ],
)
def test_recovery_reset_loader_rejects_invalid_metadata(
    tmp_path, field, invalid_value, message
):
    module = _load_reset_loader_module()
    payload = _valid_payload(module)
    payload[field] = invalid_value
    path = _write_payload(tmp_path / f"bad_{field}.txt", payload)
    with pytest.raises(ValueError, match=message):
        module.RecoveryResetMotionLoader([path])


def test_recovery_reset_loader_rejects_nan_quaternion_and_bad_joint_names(tmp_path):
    module = _load_reset_loader_module()

    nan_payload = _valid_payload(module)
    nan_payload["Frames"][0][0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        module.RecoveryResetMotionLoader(
            [_write_payload(tmp_path / "nan.txt", nan_payload)]
        )

    quat_payload = _valid_payload(module)
    quat_payload["Frames"][0][3] = 2.0
    with pytest.raises(ValueError, match="quaternion norm"):
        module.RecoveryResetMotionLoader(
            [_write_payload(tmp_path / "quat.txt", quat_payload)]
        )

    joints_payload = _valid_payload(module)
    joints_payload["JointNames"] = joints_payload["JointNames"][:-1]
    with pytest.raises(ValueError, match="exactly 23"):
        module.RecoveryResetMotionLoader(
            [_write_payload(tmp_path / "joints.txt", joints_payload)]
        )


def test_recovery_reset_sampling_is_uniform_by_crop_not_total_frame_count(tmp_path):
    module = _load_reset_loader_module()
    paths = []
    for crop_index, num_frames in enumerate((1, 10, 100)):
        payload = _valid_payload(module, num_frames)
        paths.append(_write_payload(tmp_path / f"crop_{crop_index}.txt", payload))
    loader = module.RecoveryResetMotionLoader(paths)
    generator = torch.Generator(device="cpu").manual_seed(1234)

    _, motion_ids, frame_ids = loader.sample(
        12_000,
        generator=generator,
        return_indices=True,
    )

    crop_ratios = torch.bincount(motion_ids, minlength=3).float() / motion_ids.numel()
    torch.testing.assert_close(
        crop_ratios,
        torch.full((3,), 1.0 / 3.0),
        atol=0.025,
        rtol=0.0,
    )
    assert frame_ids[motion_ids == 0].max().item() == 0
    assert frame_ids[motion_ids == 1].max().item() < 10
    assert frame_ids[motion_ids == 2].max().item() < 100


def test_dedicated_identity_is_terrain_stratified_and_persistent():
    build_mask, activate = _load_env_methods(
        "_build_dedicated_recovery_env_mask",
        "_activate_dedicated_recovery_mode",
    )
    terrain_types = torch.repeat_interleave(torch.arange(4), 100)
    generator = torch.Generator(device="cpu").manual_seed(7)
    mask = build_mask(terrain_types, 0.20, True, True, generator)

    assert mask.sum().item() == 80
    for terrain_type in range(4):
        type_rows = terrain_types == terrain_type
        assert mask[type_rows].sum().item() == 20
        assert (~mask[type_rows]).sum().item() == 80

    env = SimpleNamespace(
        dedicated_recovery_env_mask=mask.clone(),
        recovery_mask=torch.zeros(400, dtype=torch.bool),
        recovery_mask_t=torch.zeros(400, dtype=torch.bool),
        recovery_timer=torch.ones(400, dtype=torch.long),
        recovery_attempt_active=torch.zeros(400, dtype=torch.bool),
        recovery_trigger_armed=torch.ones(400, dtype=torch.bool),
        enter_recovery_buf=torch.ones(400, dtype=torch.bool),
        exit_recovery_buf=torch.ones(400, dtype=torch.bool),
        recovery_failed_buf=torch.ones(400, dtype=torch.bool),
    )
    original_identity = env.dedicated_recovery_env_mask.clone()
    dedicated_ids = torch.nonzero(mask, as_tuple=False).flatten()
    for _ in range(100):
        activate(env, dedicated_ids)
    assert torch.equal(env.dedicated_recovery_env_mask, original_identity)
    assert torch.all(env.recovery_mask[dedicated_ids])
    assert torch.all(env.recovery_mask_t[dedicated_ids])
    assert torch.count_nonzero(env.recovery_timer[dedicated_ids]) == 0
    assert torch.all(env.recovery_attempt_active[dedicated_ids])
    assert not torch.any(env.recovery_trigger_armed[dedicated_ids])
    assert not torch.any(env.enter_recovery_buf[dedicated_ids])
    assert not torch.any(env.exit_recovery_buf[dedicated_ids])
    assert not torch.any(env.recovery_failed_buf[dedicated_ids])


def test_dedicated_identity_falls_back_to_one_global_group_without_terrain_types():
    (build_mask,) = _load_env_methods("_build_dedicated_recovery_env_mask")
    fallback_group = torch.zeros(40, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(11)

    mask = build_mask(fallback_group, 0.20, True, False, generator)

    assert mask.dtype == torch.bool
    assert mask.shape == (40,)
    assert mask.sum().item() == 8
    assert (~mask).sum().item() == 32


def test_small_terrain_strata_preserve_the_global_ratio_without_failing():
    (build_mask,) = _load_env_methods("_build_dedicated_recovery_env_mask")
    terrain_types = torch.repeat_interleave(torch.arange(3), 2)
    generator = torch.Generator(device="cpu").manual_seed(13)

    mask = build_mask(terrain_types, 0.20, True, True, generator)

    assert mask.sum().item() == 1
    assert (~mask).sum().item() == 5
    for terrain_type in range(3):
        assert mask[terrain_types == terrain_type].sum().item() <= 1


def test_single_environment_falls_back_to_natural_identity():
    (build_mask,) = _load_env_methods("_build_dedicated_recovery_env_mask")

    singleton_strata_mask = build_mask(
        torch.arange(5),
        0.20,
        True,
        True,
        torch.Generator(device="cpu").manual_seed(17),
    )
    one_env_mask = build_mask(torch.tensor([0]), 0.20, True, True)

    assert singleton_strata_mask.sum().item() == 1
    assert torch.equal(one_env_mask, torch.tensor([False]))


def _boundaries(
    *,
    was_recovery=False,
    fall=False,
    success=False,
    failure=False,
    absolute=False,
    budget=False,
    dedicated=False,
):
    (compute_boundaries,) = _load_env_methods("_compute_recovery_episode_boundaries")
    tensor = lambda value: torch.tensor([value], dtype=torch.bool)
    return compute_boundaries(
        tensor(was_recovery),
        tensor(fall),
        tensor(success),
        tensor(failure),
        tensor(absolute),
        tensor(budget),
        tensor(dedicated),
    )


def test_dedicated_success_and_failure_are_true_non_timeout_terminals():
    success = _boundaries(was_recovery=True, success=True, dedicated=True)
    assert success["dedicated_success"].item() is True
    assert success["reset_buf"].item() is True
    assert success["time_out_buf"].item() is False
    assert success["natural_exit_to_locomotion"].item() is False

    failure = _boundaries(was_recovery=True, failure=True, dedicated=True)
    assert failure["reset_buf"].item() is True
    assert failure["time_out_buf"].item() is False


def test_dedicated_immediate_success_is_a_true_terminal():
    (compute_boundaries,) = _load_env_methods("_compute_recovery_episode_boundaries")
    exit_recovery = torch.tensor([True])
    result = compute_boundaries(
        torch.tensor([True]),
        torch.tensor([False]),
        exit_recovery,
        torch.tensor([False]),
        torch.tensor([False]),
        torch.tensor([False]),
        torch.tensor([True]),
    )
    assert result["reset_buf"].item() is True
    assert result["time_out_buf"].item() is False


def test_dedicated_failure_occurs_at_six_seconds_not_before():
    (compute_failure,) = _load_env_methods("_compute_recovery_attempt_failure")
    was_recovery = torch.tensor([True])
    no_event = torch.tensor([False])
    # step_dt=0.02 -> 6.0 s == 300 Recovery control steps.
    before_limit = compute_failure(was_recovery, torch.tensor([299]), 300, no_event, no_event)
    at_limit = compute_failure(was_recovery, torch.tensor([300]), 300, no_event, no_event)
    assert before_limit.item() is False
    assert at_limit.item() is True
    result = _boundaries(was_recovery=True, failure=at_limit.item(), dedicated=True)
    assert result["reset_buf"].item() is True
    assert result["time_out_buf"].item() is False


def test_natural_success_before_budget_returns_to_locomotion():
    result = _boundaries(was_recovery=True, success=True, budget=False)
    assert result["natural_exit_to_locomotion"].item() is True
    assert result["reset_buf"].item() is False
    assert result["time_out_buf"].item() is False


def test_natural_20s_budget_and_last_step_fall_priority():
    plain_timeout = _boundaries(budget=True)
    assert plain_timeout["natural_locomotion_timeout"].item() is True
    assert plain_timeout["reset_buf"].item() is True
    assert plain_timeout["time_out_buf"].item() is True

    final_step_fall = _boundaries(fall=True, budget=True)
    assert final_step_fall["enter_recovery"].item() is True
    assert final_step_fall["reset_buf"].item() is False
    assert final_step_fall["time_out_buf"].item() is False


def test_natural_post_budget_success_is_terminal_without_timeout():
    result = _boundaries(was_recovery=True, success=True, budget=True)
    assert result["natural_post_budget_success"].item() is True
    assert result["reset_buf"].item() is True
    assert result["time_out_buf"].item() is False
    assert result["natural_exit_to_locomotion"].item() is False


@pytest.mark.parametrize("was_recovery", [False, True])
def test_27s_absolute_timeout_wins_in_both_modes(was_recovery):
    result = _boundaries(
        was_recovery=was_recovery,
        fall=True,
        success=was_recovery,
        absolute=True,
    )
    assert result["reset_buf"].item() is True
    assert result["time_out_buf"].item() is True
    assert result["enter_recovery"].item() is False


class _FakeRobot:
    def __init__(self):
        limits = torch.zeros(2, 23, 2)
        limits[..., 0] = -1.0
        limits[..., 1] = 1.0
        self.data = SimpleNamespace(
            soft_joint_pos_limits=limits,
            # World X/Y already sampled by reset_root_state_uniform around the
            # current terrain origin before the 59-D physical override.
            root_link_pos_w=torch.tensor(
                [[0.0, 0.0, 0.8], [5.25, 5.75, 0.8]]
            ),
        )
        self.calls = {}

    def write_root_link_pose_to_sim(self, value, env_ids):
        self.calls["root_pose"] = (value.clone(), env_ids.clone())

    def write_root_link_velocity_to_sim(self, value, env_ids):
        self.calls["root_velocity"] = (value.clone(), env_ids.clone())

    def write_joint_state_to_sim(self, position, velocity, joint_ids, env_ids):
        self.calls["joints"] = (
            position.clone(),
            velocity.clone(),
            list(joint_ids),
            env_ids.clone(),
        )


def test_dedicated_reset_writes_complete_physical_state_and_clamps_joint_position():
    apply_reset, activate = _load_env_methods(
        "_apply_dedicated_recovery_reset",
        "_activate_dedicated_recovery_mode",
    )
    frame = torch.zeros(1, 59)
    frame[:, 2] = 0.4
    frame[:, 3] = 1.0
    frame[:, 7:13] = torch.arange(6)
    frame[:, 13:15] = 2.0
    frame[:, 36:59] = 0.25
    loader = SimpleNamespace(
        sample=lambda _count, return_indices: (
            frame.clone(),
            torch.tensor([3]),
            torch.tensor([7]),
        )
    )
    env = SimpleNamespace(
        recovery_reset_loader=loader,
        scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [5.0, 6.0, 1.0]])),
        robot=_FakeRobot(),
        recovery_reset_joint_ids=list(range(23)),
        device="cpu",
        dedicated_recovery_joint_clamp_total=torch.zeros((), dtype=torch.long),
        dedicated_recovery_joint_sample_total=torch.zeros((), dtype=torch.long),
        dedicated_recovery_reset_total=torch.zeros((), dtype=torch.long),
        dedicated_last_sample_motion_id=torch.full((2,), -1, dtype=torch.long),
        dedicated_last_sample_frame_id=torch.full((2,), -1, dtype=torch.long),
        _dedicated_joint_clamp_warning_emitted=False,
        recovery_mask=torch.zeros(2, dtype=torch.bool),
        recovery_mask_t=torch.zeros(2, dtype=torch.bool),
        recovery_timer=torch.ones(2, dtype=torch.long),
        recovery_attempt_active=torch.zeros(2, dtype=torch.bool),
        recovery_trigger_armed=torch.ones(2, dtype=torch.bool),
        enter_recovery_buf=torch.ones(2, dtype=torch.bool),
        exit_recovery_buf=torch.ones(2, dtype=torch.bool),
        recovery_failed_buf=torch.ones(2, dtype=torch.bool),
    )
    env._activate_dedicated_recovery_mode = lambda ids: activate(env, ids)

    with pytest.warns(UserWarning, match="clamp ratio"):
        apply_reset(env, torch.tensor([1]))

    root_pose, pose_ids = env.robot.calls["root_pose"]
    root_velocity, velocity_ids = env.robot.calls["root_velocity"]
    joint_pos, joint_vel, joint_ids, joint_env_ids = env.robot.calls["joints"]
    torch.testing.assert_close(root_pose[0, :3], torch.tensor([5.25, 5.75, 1.4]))
    root_xy_from_origin = root_pose[0, :2] - env.scene.env_origins[1, :2]
    assert torch.all(root_xy_from_origin >= -0.5)
    assert torch.all(root_xy_from_origin <= 0.5)
    torch.testing.assert_close(root_pose[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(root_velocity[0], torch.arange(6).float())
    assert joint_pos[0, 0].item() == 1.0
    torch.testing.assert_close(joint_vel, torch.full((1, 23), 0.25))
    assert joint_ids == list(range(23))
    assert torch.equal(pose_ids, torch.tensor([1]))
    assert torch.equal(velocity_ids, pose_ids)
    assert torch.equal(joint_env_ids, pose_ids)
    assert env.dedicated_last_sample_motion_id[1].item() == 3
    assert env.dedicated_last_sample_frame_id[1].item() == 7
    assert env.dedicated_recovery_reset_total.item() == 1
    assert env.recovery_mask[1].item() is True
    assert env.recovery_mask_t[1].item() is True


def test_terrain_curriculum_only_updates_natural_and_dedicated_inherits_same_type():
    update_reset_terrain, inherit = _load_env_methods(
        "_update_reset_terrain",
        "_inherit_dedicated_terrain_levels",
    )
    terrain_origins = torch.zeros(3, 2, 3)
    for level in range(3):
        for terrain_type in range(2):
            terrain_origins[level, terrain_type] = torch.tensor(
                [float(level), float(terrain_type), 0.0]
            )
    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([2, 0, 1, 0]),
        terrain_types=torch.tensor([0, 0, 1, 1]),
        terrain_origins=terrain_origins,
        env_origins=torch.zeros(4, 3),
    )
    captured_natural_ids = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            scene=SimpleNamespace(terrain_generator=SimpleNamespace(curriculum=True)),
            recovery=SimpleNamespace(dedicated_inherit_natural_terrain_level=True),
        ),
        scene=SimpleNamespace(terrain=terrain),
        num_envs=4,
        device="cpu",
        dedicated_recovery_enabled=True,
        natural_env_mask=torch.tensor([True, False, True, False]),
        update_terrain_levels=lambda ids: (
            captured_natural_ids.append(ids.clone()) or {"Curriculum/terrain_levels": torch.tensor(1.0)}
        ),
    )
    env._inherit_dedicated_terrain_levels = lambda ids: inherit(env, ids)
    natural_ids = torch.tensor([0, 2])
    dedicated_ids = torch.tensor([1, 3])

    update_reset_terrain(env, natural_ids, dedicated_ids)

    assert len(captured_natural_ids) == 1
    assert torch.equal(captured_natural_ids[0], natural_ids)
    assert terrain.terrain_levels.tolist() == [2, 2, 1, 1]
    torch.testing.assert_close(terrain.env_origins[1], terrain_origins[2, 0])
    torch.testing.assert_close(terrain.env_origins[3], terrain_origins[1, 1])


def test_initial_progress_randomization_keeps_dedicated_fresh_and_natural_coherent():
    (randomize_progress,) = _load_env_methods("randomize_initial_episode_progress")
    env = SimpleNamespace(
        natural_env_ids=torch.tensor([0, 2]),
        dedicated_recovery_env_ids=torch.tensor([1, 3]),
        locomotion_budget_steps=1000,
        episode_length_buf=torch.full((4,), 9999, dtype=torch.long),
        locomotion_mode_steps=torch.full((4,), 9999, dtype=torch.long),
        recovery_mode_steps=torch.full((4,), 9999, dtype=torch.long),
        device="cpu",
    )

    randomize_progress(env)

    assert torch.equal(
        env.episode_length_buf[env.natural_env_ids],
        env.locomotion_mode_steps[env.natural_env_ids],
    )
    assert torch.all(env.episode_length_buf[env.natural_env_ids] < 1000)
    assert torch.count_nonzero(env.recovery_mode_steps[env.natural_env_ids]) == 0
    assert torch.count_nonzero(env.episode_length_buf[env.dedicated_recovery_env_ids]) == 0
    assert torch.count_nonzero(env.locomotion_mode_steps[env.dedicated_recovery_env_ids]) == 0
    assert torch.count_nonzero(env.recovery_mode_steps[env.dedicated_recovery_env_ids]) == 0


def test_terminal_recovery_amp_transition_uses_pre_reset_state_not_teleport_state():
    runner = RENetAmpOnPolicyRunner.__new__(RENetAmpOnPolicyRunner)
    runner.device = "cpu"
    # Y is the new post-reset fallen state; X is the true pre-reset terminal state.
    next_amp_obs = {
        "loco": torch.tensor([[10.0], [20.0]]),
        "recovery": torch.tensor([[100.0], [200.0]]),
    }
    infos = {
        "terminal_locomotion_amp_states": torch.tensor([[2.0]]),
        "terminal_recovery_amp_states": torch.tensor([[7.0]]),
    }

    routed = runner._replace_reset_rows_with_terminal_amp_states(
        next_amp_obs,
        infos,
        torch.tensor([1]),
    )

    torch.testing.assert_close(routed["loco"], torch.tensor([[10.0], [2.0]]))
    torch.testing.assert_close(routed["recovery"], torch.tensor([[100.0], [7.0]]))
    # The original post-reset bundle remains Y for the next new episode.
    torch.testing.assert_close(next_amp_obs["recovery"], torch.tensor([[100.0], [200.0]]))


def test_missing_recovery_amp_expert_has_dedicated_53d_error(tmp_path):
    missing = tmp_path / "missing_recovery_amp_53d.txt"

    with pytest.raises(
        FileNotFoundError,
        match="59D recovery reset files cannot replace 53D D_REC expert files",
    ):
        RENetAmpOnPolicyRunner._validate_recovery_amp_expert_files([missing])


class _CaptureReplay:
    def __init__(self):
        self.rows = []

    def insert(self, state, next_state):
        self.rows.append((state.clone(), next_state.clone()))


def test_first_dedicated_transition_routes_to_recovery_ppo_and_drec_only():
    route_rewards, activate = _load_env_methods(
        "_route_action_time_rewards",
        "_activate_dedicated_recovery_mode",
    )
    env = SimpleNamespace(
        recovery_mask=torch.zeros(1, dtype=torch.bool),
        recovery_mask_t=torch.zeros(1, dtype=torch.bool),
        recovery_timer=torch.ones(1, dtype=torch.long),
        recovery_attempt_active=torch.zeros(1, dtype=torch.bool),
        recovery_trigger_armed=torch.ones(1, dtype=torch.bool),
        enter_recovery_buf=torch.ones(1, dtype=torch.bool),
        exit_recovery_buf=torch.ones(1, dtype=torch.bool),
        recovery_failed_buf=torch.ones(1, dtype=torch.bool),
    )
    activate(env, torch.tensor([0]))
    recovery_mask_t = env.recovery_mask.clone()
    loco_reward, task_reward, reg_reward = route_rewards(
        torch.tensor([9.0]),
        torch.tensor([2.0]),
        torch.tensor([-0.5]),
        recovery_mask_t,
    )
    torch.testing.assert_close(loco_reward, torch.tensor([0.0]))
    torch.testing.assert_close(task_reward, torch.tensor([2.0]))
    torch.testing.assert_close(reg_reward, torch.tensor([-0.5]))

    algorithm = RENetAMPPPO.__new__(RENetAMPPPO)
    algorithm.device = "cpu"
    algorithm.discriminator_loco = SimpleNamespace(input_dim=100)
    algorithm.discriminator_recovery = SimpleNamespace(input_dim=106)
    algorithm.amp_storage_loco = _CaptureReplay()
    algorithm.amp_storage_recovery = _CaptureReplay()
    amp_obs = {"loco": torch.zeros(1, 50), "recovery": torch.zeros(1, 53)}
    next_amp_obs = {"loco": torch.ones(1, 50), "recovery": torch.ones(1, 53)}
    algorithm._store_amp_transition(amp_obs, next_amp_obs, recovery_mask_t)

    assert algorithm.amp_storage_loco.rows == []
    assert len(algorithm.amp_storage_recovery.rows) == 1
    auxiliary_mask = algorithm._get_auxiliary_sample_mask(
        {"recovery_mask_t": recovery_mask_t.unsqueeze(1)},
        torch.zeros(1, 1),
    )
    assert auxiliary_mask.item() is False
