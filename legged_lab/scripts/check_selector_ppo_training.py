"""一键检查真实高层 PPO：短训练两次更新，再在新进程恢复并更新一次。

仅新增输出目录，不改配置和训练代码。GPU 不可用时明确记录 blocked，
不会以 CPU 协议测试代替真实 Isaac Sim 训练检查。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ENTRY = ROOT / "legged_lab/scripts/train_selector_ppo.py"

PREFLIGHT_CODE = r'''
import importlib
import json
import os
from pathlib import Path
import sys

output, root, device = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "rsl_rl"))
result = {"status": "blocked", "python": sys.executable, "device": device,
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "checks": {}, "reasons": []}
torch = None
try:
    import torch
    result["torch_version"] = str(torch.__version__)
    result["checks"]["torch"] = "passed"
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    index = int(device.split(":")[1]) if ":" in device else 0
    result.update(cuda_available=available, gpu_count=count)
    if available and index < count:
        result["gpu_name"] = torch.cuda.get_device_name(index)
        result["checks"]["cuda"] = "passed"
    else:
        result["checks"]["cuda"] = "blocked"
        result["reasons"].append(f"CUDA device {device} is unavailable (visible GPU count: {count}).")
except Exception as error:
    result["checks"]["torch"] = "blocked"
    result["reasons"].append(f"PyTorch/CUDA: {type(error).__name__}: {error}")
try:
    from isaaclab.app import AppLauncher
    result["checks"]["isaaclab_launcher_import"] = "passed"
except Exception as error:
    result["checks"]["isaaclab_launcher_import"] = "blocked"
    result["reasons"].append(f"IsaacLab/Isaac Sim environment: {type(error).__name__}: {error}")
if torch is not None:
    try:
        importlib.import_module("rsl_rl.runners.selector_on_policy_runner")
        result["checks"]["selector_training_import"] = "passed"
    except Exception as error:
        result["checks"]["selector_training_import"] = "blocked"
        result["reasons"].append(f"Selector training import: {type(error).__name__}: {error}")
if not result["reasons"]:
    result["status"] = "ready"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
'''


class TrainingCheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise TrainingCheckError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    require(Path(path).is_file(), f"Missing artifact: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=4, help="每次 PPO 更新的高层 rollout 长度。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="新的空检查目录；默认 outputs 下的时间戳目录。")
    parser.add_argument("--isaac-setup", type=Path, default=None, help="可选的二进制 Isaac Sim setup_conda_env.sh。")
    parser.add_argument("--timeout-s", type=float, default=900., help="每次真实训练进程的超时秒数。")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        parser.error("真实仿真检查的 device 应为 cuda 或 cuda:N。")
    if args.num_envs < 1 or args.rollout_steps < 1:
        parser.error("num-envs 和 rollout-steps 必须为正整数。")
    if not 0 <= args.seed < 2**63:
        parser.error("seed 必须在 [0,2**63) 内。")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s 必须为正的有限数。")
    if args.isaac_setup is not None:
        args.isaac_setup = args.isaac_setup.expanduser().resolve()
        if not args.isaac_setup.is_file():
            parser.error(f"Isaac Sim setup 文件不存在：{args.isaac_setup}")
    return args


def python_command(arguments, setup=None):
    command = [sys.executable, *map(str, arguments)]
    if setup is not None:
        # 路径和 Python 参数通过 argv 传入，不能插值到 shell 代码中。
        command = ["bash", "-c", 'set -e\nsource "$1"\nshift\nexec "$@"',
                   "selector-ppo-check", str(setup), *command]
    return command


def terminate_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5.)
        return
    try:
        process.wait(timeout=5.)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5.)


def run_process(command, log_path, timeout):
    start = time.monotonic()
    timed_out = False
    with Path(log_path).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process)
        except BaseException:
            terminate_process(process)
            raise
    return {"command": command, "returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - start, "log": str(log_path)}


def train_arguments(args, log_dir, *, resume=None):
    arguments = [TRAIN_ENTRY, "--headless", "--device", args.device, "--num-envs", str(args.num_envs),
                 "--num-steps-per-env", str(args.rollout_steps), "--max-iterations", "2" if resume is None else "1",
                 "--seed", str(args.seed), "--save-interval", "1", "--logger", "jsonl", "--small-terrain",
                 "--log-dir", log_dir]
    if resume is not None:
        arguments.extend(("--resume", resume))
    return arguments


def verify_training_runs(train_dir, resume_dir, *, num_envs, rollout_steps, seed):
    """只检查真实训练入口已写出的产物；不生成替身数据或补写 checkpoint。"""
    import torch

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "rsl_rl"))
    from rsl_rl.modules.selector_actor_critic import SelectorActorCritic

    train_dir, resume_dir = Path(train_dir), Path(resume_dir)
    reports = [read_json(directory / "report.json") for directory in (train_dir, resume_dir)]
    expected_signature = {"actor_obs_dim": 3466, "critic_obs_dim": 3660, "num_actions": 1,
                          "mode_names": ["vp", "lp"], "high_level_dt": .2, "low_level_dt": .02}
    logs = []
    for directory, report, start, end in zip((train_dir, resume_dir), reports, (0, 2), (2, 3)):
        require(report["status"] == "completed", f"Training did not complete in {directory}.")
        require(report["start_iteration"] == start and report["completed_iteration"] == end,
                f"Wrong completed iteration range in {directory}.")
        require(report["num_envs"] == num_envs and report["seed"] == seed, "Environment count or seed mismatch.")
        require(report["frozen_policy_remained_frozen"] is True, "Frozen RENet was not verified frozen by the entry point.")
        require(report["resume_environment_state"] is False, "The check expects a newly initialized resume environment.")
        # sim_step_counter 是单个全局仿真时钟，不再乘并行环境数。
        require(report["physics_steps"] == (end - start) * rollout_steps * 40,
                f"Wrong physics step count in {directory}.")
        require(Path(report["checkpoint"]).resolve() == (directory / f"model_{end}.pt").resolve(),
                "Entry point checkpoint path does not identify its final saved update.")
        for name in ("run.json", "fault_audit.json", "params/env_cfg.json", "params/agent_cfg.json"):
            require((directory / name).is_file(), f"Missing training artifact: {directory / name}")
        metrics_path = directory / "metrics.jsonl"
        require(metrics_path.is_file(), f"Missing metrics: {metrics_path}")
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        require([row["iteration"] for row in rows] == list(range(start + 1, end + 1)),
                f"Repeated, missing or extra PPO iterations in {metrics_path}.")
        for row in rows:
            require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in row.values()),
                    f"Non-finite or nonnumeric training metric in {metrics_path}.")
            iteration = row["iteration"]
            require(row["Train/total_timesteps"] == num_envs * rollout_steps * iteration, "Wrong high-level sample count.")
            require(row["Train/total_vector_steps"] == rollout_steps * iteration, "Wrong global vector-step count.")
            require(0 <= row["Policy/clip_fraction"] <= 1 and row["Policy/categorical_kl"] >= 0, "Invalid PPO KL/clipping metric.")
            require(0 <= row["Policy/entropy"] <= math.log(2.) + 1e-6, "Invalid categorical entropy.")
            require(0 <= row["Policy/lp_fraction"] <= 1 and
                    math.isclose(row["Policy/lp_fraction"] + row["Policy/vp_fraction"], 1., abs_tol=1e-6),
                    "Invalid mode selection fractions.")
            require(all(0 <= row[key] <= num_envs * rollout_steps and int(row[key]) == row[key]
                        for key in ("Train/true_failures", "Train/time_outs", "Train/completed_episodes")) and
                    row["Train/completed_episodes"] == row["Train/true_failures"] + row["Train/time_outs"],
                    "Invalid termination statistics.")
        logs.extend(rows)

    checkpoints = [train_dir / "model_1.pt", train_dir / "model_2.pt", resume_dir / "model_3.pt"]
    loaded = []
    signature = None
    for iteration, path in enumerate(checkpoints, 1):
        require(path.is_file(), f"Missing checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint.get("format") == "g1_selector_discrete_ppo" and checkpoint.get("version") == 1,
                f"Wrong checkpoint format: {path}")
        require(checkpoint["iter"] == iteration, f"Wrong checkpoint iteration: {path}")
        current_signature = checkpoint["signature"]
        require(all(current_signature.get(key) == value for key, value in expected_signature.items()),
                "High-level observation, mode or timing contract mismatch.")
        if signature is None:
            signature = current_signature
        require(current_signature == signature, "Frozen policy or environment contract changed across resume.")
        require(checkpoint["total_timesteps"] == num_envs * rollout_steps * iteration and
                checkpoint["total_vector_steps"] == rollout_steps * iteration, "Checkpoint sample counters do not continue.")
        require(checkpoint["total_timesteps"] <= checkpoint["total_low_level_steps"] <= 10 * checkpoint["total_timesteps"],
                "Effective low-level steps are outside the valid termination bounds.")
        require(logs[iteration - 1]["Train/total_low_level_steps"] == checkpoint["total_low_level_steps"],
                "Metric and checkpoint effective low-level counters disagree.")
        cfg = checkpoint["train_cfg"]
        require(cfg["seed"] == seed and cfg["num_steps_per_env"] == rollout_steps, "Checkpoint seed or rollout config mismatch.")
        require(cfg["policy"]["class_name"] == "SelectorActorCritic" and cfg["algorithm"]["class_name"] == "DiscretePPO",
                "The checkpoint does not use the independent categorical training chain.")
        expected_updates = iteration * cfg["algorithm"]["num_learning_epochs"] * cfg["algorithm"]["num_mini_batches"]
        states = checkpoint["optimizer_state_dict"]["state"]
        require(bool(states), "Adam state is empty after training.")
        require(all(float(state["step"]) == expected_updates for state in states.values()),
                "Adam steps did not continue across PPO updates/resume.")
        require(math.isfinite(checkpoint["learning_rate"]) and checkpoint["learning_rate"] > 0, "Invalid checkpoint learning rate.")
        require(all(group["lr"] == checkpoint["learning_rate"] for group in checkpoint["optimizer_state_dict"]["param_groups"]),
                "Optimizer learning rate and adaptive PPO learning rate disagree.")
        require(logs[iteration - 1]["Loss/learning_rate"] == checkpoint["learning_rate"],
                "Metric and checkpoint learning rates disagree.")
        weights = checkpoint["model_state_dict"]
        require(bool(weights) and all(name.split(".")[0] in ("actor", "critic", "actor_encoder", "critic_encoder") for name in weights),
                "Checkpoint contains unexpected network parameters, such as a frozen policy or AMP discriminator.")
        require(all(torch.isfinite(value).all() for value in weights.values()), "Non-finite trained network weights.")
        if loaded:
            require(cfg["policy"] == loaded[0]["train_cfg"]["policy"], "Network config changed across updates.")
            require(cfg["algorithm"] == loaded[0]["train_cfg"]["algorithm"], "PPO config changed across updates.")
            previous = loaded[-1]["model_state_dict"]
            require(weights.keys() == previous.keys(), "Network parameter keys changed.")
            for branch in ("actor", "critic"):
                require(any(not torch.equal(value, previous[name]) for name, value in weights.items() if name.startswith(branch)),
                        f"No {branch} parameter changed during update {iteration}.")
        loaded.append(checkpoint)

    for report, checkpoint, directory in zip(reports, (loaded[1], loaded[2]), (train_dir, resume_dir)):
        require(report["observation_contract"] == signature, "Entry point and checkpoint signatures disagree.")
        for key in ("total_timesteps", "total_low_level_steps"):
            require(report[key] == checkpoint[key], f"Entry point and checkpoint {key} disagree.")
        metadata = report["frozen_policy"]
        require(all(metadata.get(key) == value for key, value in signature["frozen_policy"].items()), "Frozen RENet identity changed.")
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in signature["frozen_policy"].values()),
                "Missing validated frozen-policy SHA256 identity.")
        audit = read_json(directory / "fault_audit.json")
        require(audit["seed"] == seed, "Fault pipeline seed differs from the training seed.")

    # 用实际训练出的权重验证 CPU 确定性推理；不启动仿真，不评估安全性能。
    with torch.random.fork_rng(devices=[]):
        model = SelectorActorCritic(num_actor_obs=3466, num_critic_obs=3660, num_actions=1, **loaded[2]["train_cfg"]["policy"])
    model.load_state_dict(loaded[2]["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    actor = torch.zeros(2, 3466)
    with torch.no_grad():
        first, second = model.act_inference(actor), model.act_inference(actor)
        critic = model.evaluate(torch.zeros(2, 3660))
    require(torch.equal(first, second) and first.shape == (2, 1) and first.dtype == torch.long and
            ((first == 0) | (first == 1)).all(), "Trained policy inference does not produce deterministic VP/LP modes.")
    require(torch.isfinite(critic).all(), "Trained critic inference is non-finite.")
    return {
        "checks": dict.fromkeys(("training_reports", "control_periods", "metrics_and_counts", "frozen_policy_identity",
                                 "actor_and_critic_updates", "adam_resume", "checkpoint_format", "cpu_argmax_inference"), "passed"),
        "checkpoints": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
        "signature": signature,
        "coverage": {"failure_episodes": sum(row["Train/true_failures"] for row in logs),
                     "timeout_episodes": sum(row["Train/time_outs"] for row in logs),
                     "note": "短检查允许没有失败、超时或故障计划；不能据此判断收敛或摔倒率改善。"},
        "inference_modes": first[:, 0].tolist(),
    }


def main(argv=None):
    args = parse_args(argv)
    output = (ROOT / "outputs/g1_selector_ppo_training_check" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
              if args.output is None else args.output.expanduser().resolve())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "scope": "真实 Isaac Sim 短训练两次更新及新进程恢复一次更新。",
              "output": str(output), "seed": args.seed, "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "phases": {name: {"status": "not_run"} for name in ("preflight", "train", "resume", "artifact_validation")}}
    phase = "preflight"
    try:
        preflight_path = output / "preflight.json"
        command = python_command(["-c", PREFLIGHT_CODE, preflight_path, ROOT, args.device], args.isaac_setup)
        result = run_process(command, output / "preflight.log", min(args.timeout_s, 60.))
        report["phases"][phase] = result
        require(result["returncode"] == 0 and not result["timed_out"], "Preflight process failed; see preflight.log.")
        preflight = read_json(preflight_path)
        report["phases"][phase]["status"] = preflight["status"]
        if preflight["status"] != "ready":
            report.update(status="blocked", reasons=preflight["reasons"])
            return 2
        for phase, directory, resume in (("train", output / "train", None),
                                         ("resume", output / "resume", output / "train/model_2.pt")):
            print(f"[Selector check] {phase}: {output / (phase + '.log')}", flush=True)
            command = python_command(train_arguments(args, directory, resume=resume), args.isaac_setup)
            result = run_process(command, output / (phase + ".log"), args.timeout_s)
            result["status"] = "passed" if result["returncode"] == 0 and not result["timed_out"] else "failed"
            report["phases"][phase] = result
            require(result["status"] == "passed", f"{phase} process failed or timed out; see {phase}.log.")
        phase = "artifact_validation"
        checks = verify_training_runs(output / "train", output / "resume", num_envs=args.num_envs,
                                      rollout_steps=args.rollout_steps, seed=args.seed)
        write_json(output / "checks.json", checks)
        report["phases"][phase] = {"status": "passed", "checks": str(output / "checks.json")}
        report["status"] = "passed"
        return 0
    except KeyboardInterrupt:
        report["phases"][phase]["status"] = "interrupted"
        report["status"] = "interrupted"
        return 130
    except Exception as error:
        report["phases"][phase]["status"] = "failed"
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        return 1
    finally:
        write_json(output / "report.json", report)
        print(json.dumps({"status": report["status"], "report": str(output / "report.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
