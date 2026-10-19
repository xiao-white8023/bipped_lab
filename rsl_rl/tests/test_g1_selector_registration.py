"""任务注册及通用 train.py 路由检查；执行真实入口逻辑，替换仿真和训练副作用。"""

import ast
import builtins
from datetime import datetime
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENV_INIT = ROOT / "legged_lab/envs/__init__.py"
TRAIN = ROOT / "legged_lab/scripts/train.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TaskRegistry = _load(ROOT / "legged_lab/utils/task_registry.py", "_selector_registry_class_checks").TaskRegistry
update_rsl_rl_cfg = _load(ROOT / "legged_lab/utils/cli_args.py", "_selector_registered_cli_checks").update_rsl_rl_cfg


def test_environment_package_registers_selector_and_preserves_existing_tasks(monkeypatch):
    registry = TaskRegistry()
    tree = ast.parse(ENV_INIT.read_text())
    modules = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = modules.setdefault(node.module, ModuleType(node.module))
        for alias in node.names:
            value = registry if alias.name == "task_registry" else type(alias.name, (), {})
            setattr(module, alias.name, value)
    original_import = builtins.__import__

    def import_without_simulation(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_without_simulation)
        exec(compile(tree, str(ENV_INIT), "exec"), {})
    assert set(registry.task_classes) == {
        "g1_rough", "g1_film", "g1_renet", "g1_depth", "g1_atten", "g1_recovery", "g1_selector",
    }
    assert registry.get_task_class("g1_selector") is modules["legged_lab.envs.g1.g1_selector_env"].G1SelectorEnv
    env_cfg, agent_cfg = registry.get_cfgs("g1_selector")
    assert isinstance(env_cfg, modules["legged_lab.envs.g1.g1_selector_cfg"].G1SelectorEnvCfg)
    assert isinstance(agent_cfg, modules["legged_lab.envs.g1.g1_selector_cfg"].G1SelectorAgentCfg)


def _train_harness(task="g1_selector", *, distributed=False, checkpoint_path=None):
    calls = []
    env_cfg = SimpleNamespace(
        device="cuda:0", seed=42, scene=SimpleNamespace(num_envs=4096, seed=42),
        sim=SimpleNamespace(device="cuda:0"), renet=SimpleNamespace(lp_height_source="pgtt_full_atten"),
    )
    if task in ("g1_selector", "g1_renet"):
        env_cfg.lp_height = object()
    agent_cfg = SimpleNamespace(
        seed=42, device="cuda:0", max_iterations=5000, experiment_name="registered_selector_test",
        run_name="", runner_class_name="SelectorOnPolicyRunner" if task == "g1_selector" else "OnPolicyRunner",
        logger="tensorboard", resume=False, load_run=".*", load_checkpoint="model_.*.pt",
    )
    agent_cfg.to_dict = lambda: {"seed": agent_cfg.seed, "runner_class_name": agent_cfg.runner_class_name}
    args = SimpleNamespace(
        task=task, num_envs=8, seed=7, device="cpu", headless=True, distributed=distributed,
        max_iterations=2, lp_height_source=None, checkpoint_path=checkpoint_path,
        experiment_name=None, run_name=None, resume=None, load_run=None, checkpoint=None,
        logger=None, log_project_name=None,
    )

    class Env:
        def __init__(self, cfg, headless):
            assert cfg is env_cfg and headless
            calls.append(("environment", cfg.device, cfg.seed, cfg.scene.seed, cfg.scene.num_envs))

    class Runner:
        def __init__(self, env, cfg, log_dir, device):
            calls.append(("runner", cfg, device, log_dir))

        def learn(self, *, num_learning_iterations, init_at_random_ep_len):
            calls.append(("learn", num_learning_iterations, init_at_random_ep_len))

        def load(self, path):
            calls.append(("load", path))

    def configure_selector(cfg, agent):
        assert cfg.scene.num_envs == 8 and cfg.device == agent.device == "cpu"
        assert agent.seed == 7 and agent.max_iterations == 2
        cfg.seed = agent.seed
        calls.append(("selector_configuration",))

    def configure_renet(cfg, agent):
        if task == "g1_selector":
            raise AssertionError("Selector was incorrectly routed through RENet configuration.")
        calls.append(("renet_configuration",))

    registry = TaskRegistry()
    registry.register(task, Env, env_cfg, agent_cfg)
    scope = {
        "os": os, "datetime": datetime, "args_cli": args, "task_registry": registry,
        "update_rsl_rl_cfg": update_rsl_rl_cfg, "app_launcher": SimpleNamespace(local_rank=1),
        "dump_yaml": lambda path, cfg: calls.append(("yaml", path)),
    }
    tree = ast.parse(TRAIN.read_text())
    # 仅将入口实际导入的符号加入全局变量；漏掉 runner/config 导入会真实触发 NameError。
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if node.module.startswith("rsl_rl.runners"):
                scope[name] = Runner
            elif alias.name == "configure_selector_agent":
                scope[name] = configure_selector
            elif alias.name == "configure_agent_for_lp_source":
                scope[name] = configure_renet
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    exec(compile(ast.Module(body=[train], type_ignores=[]), str(TRAIN), "exec"), scope)
    return scope["train"], env_cfg, agent_cfg, calls


def test_registered_selector_uses_own_sync_runner_device_seed_and_episode_clocks():
    train, env_cfg, agent_cfg, calls = _train_harness()
    train()
    assert ("selector_configuration",) in calls and ("renet_configuration",) not in calls
    assert ("environment", "cpu", 7, 7, 8) in calls
    runners = [call for call in calls if call[0] == "runner"]
    assert len(runners) == 1 and runners[0][1]["runner_class_name"] == "SelectorOnPolicyRunner"
    assert runners[0][2] == "cpu" and agent_cfg.max_iterations == 2
    assert calls[-1] == ("learn", 2, False)
    assert len([call for call in calls if call[0] == "yaml"]) == 2


@pytest.mark.parametrize("task", ["g1_renet", "g1_rough", "g1_depth"])
def test_existing_task_configuration_and_random_episode_behavior_are_preserved(task):
    train, env_cfg, agent_cfg, calls = _train_harness(task)
    train()
    assert ("selector_configuration",) not in calls
    assert (("renet_configuration",) in calls) == (task == "g1_renet")
    assert calls[-1] == ("learn", 2, True)
    assert env_cfg.scene.seed == agent_cfg.seed == 7 and env_cfg.scene.num_envs == 8
    assert env_cfg.device == ("cpu" if task == "g1_depth" else "cuda:0")


def test_selector_distributed_flag_fails_before_constructing_environment():
    train, _, _, calls = _train_harness(distributed=True)
    with pytest.raises(ValueError, match="single-process"):
        train()
    assert not calls


def test_existing_distributed_route_is_preserved_for_renet():
    train, env_cfg, agent_cfg, calls = _train_harness("g1_renet", distributed=True)
    train()
    assert env_cfg.sim.device == agent_cfg.device == "cuda:1"
    assert env_cfg.scene.seed == agent_cfg.seed == 8
    assert calls[-1] == ("learn", 2, True)


def test_selector_direct_checkpoint_is_loaded_before_learning(tmp_path):
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"entry-routing-test")
    train, _, _, calls = _train_harness(checkpoint_path=str(checkpoint))
    train()
    assert ("load", str(checkpoint)) in calls
    assert calls.index(("load", str(checkpoint))) < calls.index(("learn", 2, False))


@pytest.mark.parametrize("task,expected", [("g1_selector", True), ("g1_depth", True), ("g1_sensor_test", True),
                                           ("g1_rough", False), ("g1_renet", False), (None, False)])
def test_camera_extensions_are_enabled_before_app_launch(task, expected):
    nodes = ast.parse(TRAIN.read_text()).body
    camera_index, camera_node = next(
        (index, node) for index, node in enumerate(nodes) if isinstance(node, ast.If)
        and any(isinstance(item, ast.Attribute) and item.attr == "enable_cameras" for item in ast.walk(node))
    )
    app_index = next(index for index, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "app_launcher" for target in node.targets))
    assert camera_index < app_index
    args = SimpleNamespace(task=task, enable_cameras=False)
    exec(compile(ast.Module(body=[camera_node], type_ignores=[]), str(TRAIN), "exec"), {"args_cli": args})
    assert args.enable_cameras is expected
