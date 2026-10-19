"""独立训练入口的参数、配置覆盖、文件输出及启动时序；使用 SDK 替身，不启动 Kit。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_selector_ppo_entry_checks", ROOT / "legged_lab/scripts/train_selector_ppo.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Config(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def _args(*argv):
    args = entry.build_parser().parse_args(argv)
    args.device, args.headless = "cpu", True
    return args


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _fake_env_packages(monkeypatch):
    for name in ("legged_lab.envs", "legged_lab.envs.g1"):
        module = _module(monkeypatch, name)
        module.__path__ = []


@pytest.mark.parametrize("flag,value", [
    ("--num-envs", "0"), ("--num-envs", "-2"), ("--max-iterations", "0"),
    ("--num-steps-per-env", "0"), ("--save-interval", "-1"),
    ("--seed", "-1"), ("--seed", str(2**63)), ("--resume", "/missing/selector.pt"),
])
def test_invalid_arguments_fail_before_any_sdk_start(flag, value):
    args = _args(flag, value)
    with pytest.raises(SystemExit) as error:
        entry.validate_args(args, entry.build_parser())
    assert error.value.code == 2


def test_parse_registers_app_flags_without_constructing_the_application(monkeypatch):
    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--headless", action="store_true")

        def __init__(self, args):
            raise AssertionError("Parsing arguments must not launch Kit.")

    _module(monkeypatch, "isaaclab.app", AppLauncher=Launcher)
    args = entry.parse_args(["--num-envs", "8", "--seed", "7", "--headless"])
    assert args.num_envs == 8 and args.seed == 7 and args.headless and args.enable_cameras


def test_make_configs_applies_overrides_and_syncs_before_environment_creation(tmp_path, monkeypatch):
    checkpoint, reference, resume = (tmp_path / name for name in ("frozen.pt", "reference.pt", "model.pt"))
    for path in (checkpoint, reference, resume):
        path.write_bytes(b"configuration-check")
    generator = Config(num_rows=10, num_cols=20, seed=None, use_cache=True, curriculum=False)
    original_generator = generator.to_dict()
    synchronized = []

    def env_cfg():
        return Config(device="cuda:0", seed=42,
                      scene=Config(num_envs=4096, seed=42, terrain_generator=generator, max_init_terrain_level=5),
                      low_level=Config(checkpoint_path="unset", reference_policy_path="unset"),
                      faults=Config(enable=True))

    def agent_cfg():
        return Config(device="cuda:0", seed=42, max_iterations=5000, num_steps_per_env=24,
                      save_interval=100, logger="tensorboard", run_name="", resume=False)

    def synchronize(env, agent):
        assert env.scene.num_envs == 8 and agent.num_steps_per_env == 4
        env.seed = agent.seed
        synchronized.append((env, agent))

    _fake_env_packages(monkeypatch)
    _module(monkeypatch, "legged_lab.envs.g1.g1_selector_cfg", G1SelectorEnvCfg=env_cfg,
            G1SelectorAgentCfg=agent_cfg, configure_selector_agent=synchronize)
    args = _args("--num-envs", "8", "--max-iterations", "2", "--num-steps-per-env", "4", "--seed", "7",
                 "--save-interval", "1", "--logger", "jsonl", "--run-name", "check", "--small-terrain",
                 "--disable-faults", "--low-level-checkpoint", str(checkpoint), "--reference-policy", str(reference),
                 "--resume", str(resume))
    env, agent = entry.make_configs(args)
    assert len(synchronized) == 1 and env.seed == env.scene.seed == agent.seed == 7
    assert env.device == agent.device == "cpu" and not env.faults.enable and agent.resume
    assert agent.max_iterations == 2 and agent.save_interval == 1 and agent.logger == "jsonl"
    assert agent.run_name == "check" and env.scene.max_init_terrain_level == 1
    assert env.scene.terrain_generator.num_rows == 2 and env.scene.terrain_generator.num_cols == 3
    assert env.scene.terrain_generator.seed == 7 and not env.scene.terrain_generator.use_cache
    assert generator.to_dict() == original_generator
    assert env.low_level.checkpoint_path == str(checkpoint) and env.low_level.reference_policy_path == str(reference)


def test_log_directory_must_be_new_or_empty(tmp_path):
    args = _args("--log-dir", str(tmp_path / "run"))
    assert entry.make_log_dir(args) == tmp_path / "run"
    (tmp_path / "run/metrics.jsonl").write_text("{}\n")
    with pytest.raises(FileExistsError, match="empty"):
        entry.make_log_dir(args)


def test_json_serialization_keeps_configs_tensor_support_and_nonfinite_audit_readable(tmp_path):
    payload = Config(seed=42, support=torch.tensor([[0., 1.]]), source=Path("source.pt"),
                     diagnostic=torch.tensor([-float("inf")]), callback=entry.build_parser)
    path = tmp_path / "params/config.json"
    entry.write_json(path, payload)
    result = json.loads(path.read_text())
    assert result["seed"] == 42 and result["support"] == [[0., 1.]]
    assert result["source"] == "source.pt" and result["diagnostic"] == ["-inf"]
    assert result["callback"].endswith(":build_parser")
    hashes = entry.source_hashes()
    assert "rsl_rl/rsl_rl/runners/selector_on_policy_runner.py" in hashes
    assert "legged_lab/high_level/high_level_faults.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_cuda_preflight_failure_is_reported_without_launching_kit(tmp_path, monkeypatch):
    args = _args("--log-dir", str(tmp_path / "no-gpu"))
    args.device = "cuda:0"
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Launcher:
        def __init__(self, args):
            raise AssertionError("GPU preflight must run before Kit construction.")

    _module(monkeypatch, "isaaclab.app", AppLauncher=Launcher)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        entry.main([])
    report = json.loads((tmp_path / "no-gpu/report.json").read_text())
    assert report["status"] == "failed" and "CUDA is unavailable" in report["error"]
    assert not report["resume_environment_state"]


def test_entry_constructs_environment_resumes_learns_and_closes_with_complete_artifacts(tmp_path, monkeypatch):
    resume = tmp_path / "previous.pt"
    resume.write_bytes(b"resume-source")
    args = _args("--resume", str(resume), "--log-dir", str(tmp_path / "training"))
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    env_cfg, agent_cfg = Config(seed=7, scene=Config(num_envs=4), faults=Config(enable=True)), Config(
        seed=7, max_iterations=2, device="cpu", policy={}, algorithm={},
    )
    monkeypatch.setattr(entry, "make_configs", lambda args: (env_cfg, agent_cfg))
    app = Config(closed=False)
    app.is_running = lambda: True
    app.close = lambda: setattr(app, "closed", True)
    calls = []

    class Launcher:
        def __init__(self, args):
            calls.append("launch")
            args.sdk_internal_flag = "added-after-parsing"
            self.app = app

    class Env:
        def __init__(self, cfg, headless):
            assert calls == ["launch"] and cfg is env_cfg and headless
            calls.append("environment")
            self.frozen_policy = torch.nn.Linear(2, 1).eval().requires_grad_(False)
            self.frozen_policy_metadata = {"source_checkpoint_sha256": "frozen-identity"}
            self.sim_step_counter = 320
            self.faults = Config(audit=lambda: {"seed": 7, "plans": torch.tensor([[1, 0, 0, 0]])})

    class Runner:
        def __init__(self, env, cfg, log_dir, device):
            calls.append("runner")
            assert cfg == agent_cfg.to_dict() and device == "cpu"
            self.current_learning_iteration = self.total_timesteps = self.total_low_level_steps = 0
            self.signature, self.stopped, self.last_checkpoint = {"num_actions": 1}, False, None
            self.log_dir = log_dir

        def load(self, path):
            assert path == resume
            self.current_learning_iteration = 4
            calls.append("load")

        def learn(self, iterations, *, should_continue):
            assert iterations == 2 and should_continue()
            calls.append("learn")
            self.current_learning_iteration += iterations
            self.total_timesteps, self.total_low_level_steps = 32, 320
            self.last_checkpoint = str(self.log_dir / "model_6.pt")
            Path(self.last_checkpoint).write_bytes(b"saved")
            return {"Loss/value_loss": .5}

        def close(self):
            calls.append("close-runner")

    _module(monkeypatch, "isaaclab.app", AppLauncher=Launcher)
    _fake_env_packages(monkeypatch)
    _module(monkeypatch, "legged_lab.envs.g1.g1_selector_env", G1SelectorEnv=Env)
    _module(monkeypatch, "rsl_rl.runners.selector_on_policy_runner", SelectorOnPolicyRunner=Runner)
    handlers = []
    monkeypatch.setattr(entry.signal, "signal", lambda *args: handlers.append(args) or "previous-handler")
    entry.main([])
    assert calls == ["launch", "environment", "runner", "load", "learn", "close-runner"]
    assert app.closed and len(handlers) == 2
    report = json.loads((tmp_path / "training/report.json").read_text())
    assert report["status"] == "completed" and report["start_iteration"] == 4 and report["completed_iteration"] == 6
    assert report["seed"] == 7 and report["frozen_policy_remained_frozen"] and report["physics_steps"] == 320
    assert "sdk_internal_flag" not in report["arguments"]
    assert report["total_timesteps"] == 32 and Path(report["checkpoint"]).is_file()
    for name in ("params/env_cfg.json", "params/agent_cfg.json", "run.json", "fault_audit.json"):
        assert (tmp_path / "training" / name).is_file()
    audit = json.loads((tmp_path / "training/fault_audit.json").read_text())
    assert audit["seed"] == 7 and audit["plans"] == [[1, 0, 0, 0]]
