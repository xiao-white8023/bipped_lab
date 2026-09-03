"""Deployment-contract tests for the G1 RENet MuJoCo bridge.

The four-scene, ten-second integration test is opt-in because it creates an
OpenGL depth renderer.  Run it with ``RUN_G1_MUJOCO_INTEGRATION=1``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


mujoco = pytest.importorskip("mujoco")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "legged_lab/scripts/sim2sim_renet.py"


def _load_sim2sim_module():
    spec = importlib.util.spec_from_file_location("g1_sim2sim_renet", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sim2sim = _load_sim2sim_module()


class _ContractPolicy(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lidar_mean = observations[:, 780:967].mean(dim=1, keepdim=True)
        depth_mean = observations[:, 967:2119].mean(dim=1, keepdim=True)
        mask = observations[:, 2119:2120]
        active = mask * lidar_mean + (1.0 - mask) * depth_mean
        return active.repeat(1, 23)


def _write_contract_policy(path: Path) -> Path:
    traced = torch.jit.trace(_ContractPolicy().eval(), torch.zeros(1, 2120))
    traced.save(str(path))
    return path


def _make_lidar_runner():
    cfg = sim2sim.G1RENetSim2SimCfg()
    runner = sim2sim.G1RENetMujocoRunner.__new__(sim2sim.G1RENetMujocoRunner)
    runner.cfg = cfg
    runner.seed = 7
    runner.dt = cfg.sim.dt * cfg.sim.decimation
    runner.pose_source_name = "truth"
    runner.show_lidar_map = False
    runner.debug_lidar = False
    runner.lidar_fig = None
    model_path = REPO_ROOT / "legged_lab/assets/g1/g1_23dof/g1_23dof_rev_1_0.xml"
    runner.model = mujoco.MjModel.from_xml_path(str(model_path))
    runner.data = mujoco.MjData(runner.model)
    runner.configure_sensor_geom_groups()
    runner.data.qpos[:3] = [0.0, 0.0, cfg.robot.init_height]
    runner.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(runner.model, runner.data)
    runner.init_lidar()
    return runner


def test_mid360_packet_size_and_temporal_cursor():
    cfg = sim2sim.G1RENetSim2SimCfg()
    pattern = sim2sim.Mid360ScanPattern(cfg, seed=3)
    start = pattern.cursor
    directions = pattern.next_directions()

    assert directions.shape == (4000, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert (pattern.cursor - start) % pattern.sequence.shape[0] == 20_000


def test_mujoco_lidar_mount_groups_and_rolling_map_dropout():
    runner = _make_lidar_runner()
    runner.refresh_pose_estimate()
    sensor_position = (
        runner.current_truth_pose.position
        + runner.current_truth_pose.rotation @ runner.lidar_offset_pos
    )
    assert sensor_position[2] == pytest.approx(1.265434, abs=0.03)

    assert runner.update_lidar_scan(dropout_active=False)
    assert 0 < runner.last_lidar_hit_count < runner.lidar_pattern.rays_per_scan
    height = runner.get_lidar_observation()
    assert tuple(height.shape) == (11, 17)
    assert torch.isfinite(height).all()
    assert torch.all(height[runner.latest_lidar_confidence == 0.0] == -1.0)

    for geom_id in range(runner.model.ngeom):
        if int(runner.model.geom_group[geom_id]) != 1:
            continue
        body_id = int(runner.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(runner.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        assert body_name in sim2sim.G1_23DOF_LINKS

    scan_count = runner.scan_count
    cursor = runner.lidar_pattern.cursor
    assert not runner.update_lidar_scan(dropout_active=True)
    assert runner.scan_count == scan_count
    assert runner.lidar_pattern.cursor != cursor

    runner.data.time += runner.cfg.elevation_map.history_seconds
    expired = runner.get_lidar_observation()
    assert torch.all(expired == runner.cfg.elevation_map.invalid_height)
    assert torch.count_nonzero(runner.latest_lidar_confidence) == 0


def test_policy_contract_and_pose_switch_clear_map(tmp_path: Path):
    runner = _make_lidar_runner()
    runner.num_actions = 23
    runner.depth_height = 18
    runner.depth_width = 32
    runner.policy = torch.jit.load(str(_write_contract_policy(tmp_path / "policy.pt")))
    runner.validate_policy_contract()

    runner.refresh_pose_estimate()
    runner.update_lidar_scan()
    assert int(runner.elevation_map.frame_count[0]) == 1
    runner.set_pose_source("noisy_odom", announce=False)
    assert runner.pose_source_name == "noisy_odom"
    assert int(runner.elevation_map.frame_count[0]) == 0
    assert runner.scan_count == 0


@pytest.mark.skipif(
    os.environ.get("RUN_G1_MUJOCO_INTEGRATION") != "1",
    reason="set RUN_G1_MUJOCO_INTEGRATION=1 to run OpenGL MuJoCo integration",
)
@pytest.mark.parametrize("scene", [None, "gaps", "slope", "mix"])
def test_headless_ten_second_scene_integration(tmp_path: Path, scene: str | None):
    policy_path = _write_contract_policy(tmp_path / "policy.pt")
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--policy",
        str(policy_path),
        "--duration",
        "10",
        "--headless",
        "--no-depth-view",
        "--estimator",
        "lp",
    ]
    if scene is not None:
        command.extend(["--scene", scene])
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "policy obs shape: (1, 2120)" in result.stdout
    assert "Simulation finished." in result.stdout
