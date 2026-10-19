"""训练联调检查器的判定逻辑；测试产物来自真实 Torch runner，环境是协议替身。"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from rsl_rl.runners.selector_on_policy_runner import SelectorOnPolicyRunner


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_selector_training_check_tests", ROOT / "legged_lab/scripts/check_selector_ppo_training.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ProtocolEnv:
    device, num_actions, num_envs, step_dt, low_level_dt = "cpu", 1, 2, .2, .02

    def __init__(self):
        self.actor = torch.linspace(0., .3, 3466).repeat(2, 1)
        self.actor[1] += .1
        self.critic = torch.cat((self.actor.clone(), torch.zeros(2, 194)), dim=-1)
        self.sim_step_counter = 0
        self.frozen_policy_metadata = {"source_checkpoint_sha256": "a" * 64, "reference_policy_sha256": "b" * 64}

    def get_observations(self):
        return self.actor, {"observations": {"critic": self.critic}}

    def step(self, actions):
        self.sim_step_counter += 40
        self.actor += .01
        self.critic[:, :3466].copy_(self.actor)
        self.critic[:, 3466:] += .02
        return self.actor, torch.tensor([.2, .7]) + .1 * actions[:, 0], torch.zeros(2, dtype=torch.bool), {
            "observations": {"critic": self.critic}, "time_outs": torch.zeros(2, dtype=torch.bool),
            "low_level_steps": torch.full((2,), 10, dtype=torch.long), "log": {},
        }


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def _config():
    return {
        "seed": 42, "num_steps_per_env": 2, "save_interval": 1, "logger": "jsonl",
        "runner_class_name": "SelectorOnPolicyRunner",
        "policy": {
            "class_name": "SelectorActorCritic", "proprio_hidden_dims": [4], "height_hidden_dims": [4],
            "state_hidden_dims": [4], "actor_hidden_dims": [4], "critic_hidden_dims": [4],
            "depth_encoder": {"input_dim": (18, 32), "input_channels": 4, "output_channels": [2],
                              "kernel_size": [1], "stride": [2], "padding": [0], "mlp_output_dim": 4},
        },
        "algorithm": {"class_name": "DiscretePPO", "num_learning_epochs": 1, "num_mini_batches": 1,
                      "schedule": "fixed", "learning_rate": .001},
    }


def _write_entry_artifacts(directory, runner, start):
    check.write_json(directory / "report.json", {
        "status": "completed", "start_iteration": start, "completed_iteration": runner.current_learning_iteration,
        "num_envs": 2, "seed": 42, "frozen_policy_remained_frozen": True, "resume_environment_state": False,
        "physics_steps": runner.env.sim_step_counter, "total_timesteps": runner.total_timesteps,
        "total_low_level_steps": runner.total_low_level_steps, "observation_contract": runner.signature,
        "frozen_policy": runner.env.frozen_policy_metadata, "checkpoint": runner.last_checkpoint,
    })
    check.write_json(directory / "run.json", {"seed": 42})
    check.write_json(directory / "fault_audit.json", {"seed": 42})
    (directory / "params").mkdir()
    check.write_json(directory / "params/env_cfg.json", {"scene": {"num_envs": 2}})
    check.write_json(directory / "params/agent_cfg.json", runner.cfg)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("selector_checker_real_torch_artifacts")
    train = SelectorOnPolicyRunner(ProtocolEnv(), _config(), directory / "train")
    train.learn(2)
    _write_entry_artifacts(directory / "train", train, 0)
    resume = SelectorOnPolicyRunner(ProtocolEnv(), _config(), directory / "resume")
    resume.load(directory / "train/model_2.pt")
    resume.learn(1)
    _write_entry_artifacts(directory / "resume", resume, 2)
    return directory


@pytest.fixture
def artifacts(tmp_path, baseline):
    for name, end in (("train", 2), ("resume", 3)):
        destination = tmp_path / name
        shutil.copytree(baseline / name, destination)
        report = check.read_json(destination / "report.json")
        report["checkpoint"] = str(destination / f"model_{end}.pt")
        check.write_json(destination / "report.json", report)
    return tmp_path


def _verify(directory):
    return check.verify_training_runs(directory / "train", directory / "resume", num_envs=2, rollout_steps=2, seed=42)


def test_checker_accepts_real_network_adam_and_resume_artifacts(artifacts):
    before = torch.get_rng_state().clone()
    result = _verify(artifacts)
    assert set(result["checks"].values()) == {"passed"} and len(result["checkpoints"]) == 3
    assert result["signature"]["actor_obs_dim"] == 3466 and result["signature"]["critic_obs_dim"] == 3660
    assert result["coverage"]["failure_episodes"] == result["coverage"]["timeout_episodes"] == 0
    assert len(result["inference_modes"]) == 2 and all(mode in (0, 1) for mode in result["inference_modes"])
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("change,match", [
    ({"physics_steps": 999}, "physics step"),
    ({"start_iteration": 1}, "iteration range"),
    ({"frozen_policy_remained_frozen": False}, "verified frozen"),
    ({"resume_environment_state": True}, "newly initialized"),
    ({"checkpoint": "/wrong/model_3.pt"}, "checkpoint path"),
    ({"total_low_level_steps": 123}, "low_level_steps disagree"),
])
def test_incorrect_native_report_fails(artifacts, change, match):
    path = artifacts / "resume/report.json"
    report = check.read_json(path)
    report.update(change)
    check.write_json(path, report)
    with pytest.raises(check.TrainingCheckError, match=match):
        _verify(artifacts)


@pytest.mark.parametrize("key,value,match", [
    ("Loss/value_loss", float("nan"), "Non-finite"),
    ("iteration", 2, "Repeated, missing"),
    ("Train/total_timesteps", 0, "sample count"),
    ("Train/total_low_level_steps", 0, "counters disagree"),
    ("Train/true_failures", 1, "termination statistics"),
    ("Policy/entropy", 1., "entropy"),
    ("Policy/lp_fraction", 2., "fractions"),
])
def test_invalid_training_metrics_fail(artifacts, key, value, match):
    path = artifacts / "resume/metrics.jsonl"
    row = json.loads(path.read_text().strip())
    row[key] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(check.TrainingCheckError, match=match):
        _verify(artifacts)


@pytest.mark.parametrize("change,match", [
    ("adam", "Adam steps"), ("actor", "No actor parameter"),
    ("signature", "contract changed"), ("extra_network", "unexpected network"),
    ("learning_rate", "learning rate"),
])
def test_checkpoint_regression_fails(artifacts, change, match):
    path = artifacts / "resume/model_3.pt"
    data = torch.load(path, weights_only=True)
    if change == "adam":
        for state in data["optimizer_state_dict"]["state"].values():
            state["step"].zero_()
    elif change == "actor":
        previous = torch.load(artifacts / "train/model_2.pt", weights_only=True)
        for name in data["model_state_dict"]:
            if name.startswith("actor"):
                data["model_state_dict"][name] = previous["model_state_dict"][name]
    elif change == "signature":
        data["signature"]["frozen_policy"]["source_checkpoint_sha256"] = "c" * 64
    elif change == "extra_network":
        data["model_state_dict"]["frozen_policy.weight"] = torch.ones(1)
    else:
        data["learning_rate"] *= 2.
    torch.save(data, path)
    with pytest.raises(check.TrainingCheckError, match=match):
        _verify(artifacts)


def test_python_bootstrap_passes_paths_as_argv_without_shell_interpolation(tmp_path):
    setup = tmp_path / "setup with spaces and $special.sh"
    setup.write_text("export SELECTOR_CHECK_TEST_VALUE=setup_loaded\n")
    args = ["-c", "import os,sys;print(os.environ['SELECTOR_CHECK_TEST_VALUE']);print(sys.argv[1])", "literal $(false)"]
    result = check.run_process(check.python_command(args, setup), tmp_path / "process.log", 10.)
    assert result["returncode"] == 0 and not result["timed_out"]
    assert (tmp_path / "process.log").read_text().splitlines() == ["setup_loaded", "literal $(false)"]


def test_timed_out_child_is_terminated_and_reported(tmp_path):
    command = [sys.executable, "-c", "import time;time.sleep(60)"]
    result = check.run_process(command, tmp_path / "timeout.log", .05)
    assert result["timed_out"] and result["returncode"] is not None and result["returncode"] != 0


def test_main_blocks_before_training_if_cuda_is_unavailable(tmp_path, monkeypatch):
    calls = []

    def run(command, log_path, timeout):
        calls.append(command)
        check.write_json(tmp_path / "preflight.json", {"status": "blocked", "reasons": ["CUDA unavailable"]})
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(check, "run_process", run)
    assert check.main(["--output", str(tmp_path)]) == 2
    report = check.read_json(tmp_path / "report.json")
    assert report["status"] == "blocked" and report["reasons"] == ["CUDA unavailable"]
    assert len(calls) == 1
    assert all(report["phases"][phase]["status"] == "not_run" for phase in ("train", "resume", "artifact_validation"))


def test_main_runs_training_and_resume_in_separate_processes_then_checks_artifacts(tmp_path, baseline, monkeypatch):
    commands = []

    def run(command, log_path, timeout):
        commands.append(command)
        if len(commands) == 1:
            check.write_json(tmp_path / "preflight.json", {"status": "ready", "reasons": []})
        else:
            name, end = ("train", 2) if len(commands) == 2 else ("resume", 3)
            destination = tmp_path / name
            shutil.copytree(baseline / name, destination)
            report = check.read_json(destination / "report.json")
            report["checkpoint"] = str(destination / f"model_{end}.pt")
            check.write_json(destination / "report.json", report)
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(check, "run_process", run)
    assert check.main(["--output", str(tmp_path), "--num-envs", "2", "--rollout-steps", "2"]) == 0
    assert len(commands) == 3
    assert "--resume" not in commands[1] and "--resume" in commands[2]
    assert commands[1][commands[1].index("--max-iterations") + 1] == "2"
    assert commands[2][commands[2].index("--max-iterations") + 1] == "1"
    report = check.read_json(tmp_path / "report.json")
    assert report["status"] == "passed" and report["phases"]["artifact_validation"]["status"] == "passed"
    assert (tmp_path / "checks.json").is_file()


def test_failed_training_does_not_start_resume(tmp_path, monkeypatch):
    calls = []

    def run(command, log_path, timeout):
        calls.append(command)
        if len(calls) == 1:
            check.write_json(tmp_path / "preflight.json", {"status": "ready", "reasons": []})
        return {"returncode": 0 if len(calls) == 1 else 1, "timed_out": False}

    monkeypatch.setattr(check, "run_process", run)
    assert check.main(["--output", str(tmp_path)]) == 1
    report = check.read_json(tmp_path / "report.json")
    assert len(calls) == 2 and report["status"] == "failed"
    assert report["phases"]["resume"]["status"] == "not_run"


@pytest.mark.parametrize("args", [
    ["--device", "cpu"], ["--num-envs", "0"], ["--rollout-steps", "-1"],
    ["--timeout-s", "nan"], ["--seed", "-1"], ["--isaac-setup", "/missing/setup.sh"],
])
def test_invalid_check_options_are_rejected(args):
    with pytest.raises(SystemExit) as error:
        check.parse_args(args)
    assert error.value.code == 2
