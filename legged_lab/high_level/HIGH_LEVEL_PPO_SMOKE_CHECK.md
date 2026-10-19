# 高层 PPO 真实训练联调检查

`legged_lab/scripts/check_selector_ppo_training.py` 自动执行独立训练入口，
完成短训练和新进程恢复检查，不改写现有 cfg、环境、网络或 PPO 文件。

## 一条命令运行

在项目根目录，使用 IsaacLab 的 Python：

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python \
  legged_lab/scripts/check_selector_ppo_training.py \
  --isaac-setup /home/cjh/IsaacLab/_isaac_sim/setup_conda_env.sh
```

已在当前 shell 加载好 Isaac Sim 环境时可省略 `--isaac-setup`。
该参数只在子进程中加载指定的环境脚本，不修改系统或 Conda 配置。

默认使用 `cuda:0`、4 个环境、每次更新每环境 4 个高层步、种子 42、2×3 地形块。
两个训练进程分别执行 2 次和 1 次 PPO 更新，故障设置沿用当前 cfg。

可指定新的空输出目录，并按需延长 rollout：

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python \
  legged_lab/scripts/check_selector_ppo_training.py \
  --isaac-setup /home/cjh/IsaacLab/_isaac_sim/setup_conda_env.sh \
  --device cuda:0 --num-envs 4 --rollout-steps 24 \
  --timeout-s 900 --output outputs/selector_ppo_check_longer
```

rollout 总样本数仍需能被 cfg 中的 mini-batch 数整除，由训练入口原有校验负责检查。

## 自动检查的内容

1. 在独立子进程中检查 PyTorch、可见 CUDA 设备、IsaacLab 启动器和高层训练模块导入。
   此时不创建 SimulationApp。无法使用 GPU/SDK 时记录 `blocked`，不启动训练。
2. 调用 `train_selector_ppo.py` 完成第 1、2 次更新，每次保存 checkpoint。
3. 新建仿真进程，从 `model_2.pt` 恢复并完成第 3 次更新。
4. 逐项核对训练报告、日志、checkpoint 和实际训练权重：
   - 更新编号、累计高层样本数、全局步数及有效底层步数连续。
   - 每个高层步有 40 个物理步；恢复进程的物理时钟重新从零开始。
   - Actor/Critic 为 3466/3660 维，模式为 VP=0、LP=1，周期为 0.2/0.02 秒。
   - 三个 checkpoint 的 Actor 和 Critic 均发生更新，所有权重和训练指标有限。
   - Adam 更新计数连续，自适应学习率与优化器及日志保持一致。
   - 冻结 RENet 身份一致，训练入口确认其仍为冻结状态；高层 checkpoint 不含底层或 AMP 网络。
   - 最后一个 checkpoint 可在 CPU 加载，并输出确定性的整数 argmax 模式和有限的 Critic 价值。

短检查不要求一定出现故障计划、失败或超时。`checks.json` 会记录实际失败/超时覆盖情况，
该检查用于验证工程接入，不用于评价 PPO 收敛、策略切换质量或摔倒率改善。

## 输出与退出码

默认目录为 `outputs/g1_selector_ppo_training_check/时间戳`：

- `preflight.json`、`preflight.log`：硬件、SDK 和模块导入结果。
- `train.log`、`train/`：前两次更新的进程日志和完整训练输出。
- `resume.log`、`resume/`：恢复进程日志和第 3 次更新的输出。
- `checks.json`：产物校验结果、checkpoint SHA256 和观测/时间契约。
- `report.json`：总体状态，以及每个阶段是否真正执行。

退出码 `0=全部通过`、`1=训练或产物检查失败`、`2=运行环境阻塞`、`130=人工中断`。
子进程超时或被中断时会终止其进程组，避免留下检查进程继续占用 GPU。
已有的非空输出目录不会被覆盖。

检查器的无仿真测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD" python -m pytest -q \
  rsl_rl/tests/test_selector_ppo_training_check.py
```

这些测试使用真实网络、PPO、runner 和 Adam 生成产物，环境是协议替身；
同时故意构造错误计数、无更新权重和损坏的恢复状态，验证检查器会拒绝它们。
它们不会将真实仿真阶段标为通过。
