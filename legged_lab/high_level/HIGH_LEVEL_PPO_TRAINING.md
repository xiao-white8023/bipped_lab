# 高层 LP/VP 离散 PPO 训练

高层任务已在 `legged_lab/envs/__init__.py` 中以 `g1_selector` 注册，支持通用训练入口：

```bash
python legged_lab/scripts/train.py --task g1_selector --headless --device cuda:0 \
  --num_envs 4 --max_iterations 2
```

通用入口会启用相机扩展，使用高层自己的配置同步和 `SelectorOnPolicyRunner`，
不随机改写 episode 长度。目前该 runner 支持单进程训练；不使用 `--distributed`。

也可使用独立入口 `legged_lab/scripts/train_selector_ppo.py`，直接构造 `G1SelectorEnv`、
`SelectorActorCritic`、`DiscretePPO` 和 `SelectorOnPolicyRunner`。
两种入口使用同一套环境和 PPO，既有 RENet、Film 和 selector 采集任务的路由保持不变。

## 启动

在项目根目录，使用已有 IsaacLab Python 环境：

```bash
conda activate env_isaaclab
source /home/cjh/IsaacLab/_isaac_sim/setup_conda_env.sh
python legged_lab/scripts/train_selector_ppo.py --headless --device cuda:0
```

上面的 `source` 适用于本机的二进制 Isaac Sim 安装，用来加载其 Python 和动态库路径；
使用已配置好的 pip 版 Isaac Sim 时沿用对应环境即可。

未覆盖的参数沿用 `g1_selector_cfg.py`：高层 5 Hz、底层 50 Hz、物理 200 Hz、
新感知帧 10 Hz；Actor/Critic 为 3466/3660 维；rollout 为每环境 24 条高层
transition。模式 `0=VP、1=LP`，立即切换，不使用确认、驻留、动作混合或地形升降级。
环境已累计六项独立奖励，runner/PPO 不再乘时间，也不添加 AMP 奖励。

先用小规模检查采样、更新、日志和保存：

```bash
python legged_lab/scripts/train_selector_ppo.py \
  --headless --device cuda:0 --num-envs 4 --max-iterations 2 \
  --num-steps-per-env 4 --small-terrain --save-interval 1 --logger jsonl \
  --log-dir outputs/g1_selector_ppo_smoke
```

`--small-terrain` 只在该次运行配置中将地形缩小为 2×3，并限制初始等级；
`--disable-faults` 是显式关闭统一感知故障的实验选项。正常训练默认保留故障。
很短的检查不保证出现故障计划、失败或超时，也不用于判断策略效果。

`--low-level-checkpoint` 和 `--reference-policy` 覆盖冻结底层及其参考策略路径；
两者仍经过环境已有的严格加载和一致性检查。`--resume` 则专用于高层 PPO checkpoint。

## 恢复

```bash
python legged_lab/scripts/train_selector_ppo.py \
  --headless --device cuda:0 --num-envs 4 --max-iterations 1 \
  --num-steps-per-env 4 --small-terrain --logger jsonl \
  --resume outputs/g1_selector_ppo_smoke/model_2.pt \
  --log-dir outputs/g1_selector_ppo_resume
```

`max-iterations` 是本次新增更新次数，上例从已完成的第 2 次更新继续到第 3 次。
输出目录必须为新的空目录。checkpoint 保存高层网络、Adam 状态、自适应学习率、
已完成迭代数、累计采样计数，以及 Torch/Python 随机状态；校验网络配置、观测尺寸、
控制周期、模式定义及冻结策略文件身份。允许更换并行环境数。

恢复时仿真场景和故障管线重新初始化，不恢复机器人的物理状态、传感器历史、
逐环境故障随机流或未完成 episode，因而不是原物理轨迹的逐步重放。
runner 不调用环境 reset，也不随机改写 episode 长度。

Ctrl+C 请求在当前完整 rollout 和 PPO 更新结束后停止并保存，避免产生半次更新的
checkpoint。定期保存及正常退出保存均在 rollout 已消费的边界执行。

## 输出与统计

默认目录为 `logs/g1_selector_ppo/时间戳`：

- `params/env_cfg.json`、`params/agent_cfg.json`：应用命令行覆盖后的完整配置。
- `run.json`：启动参数、种子、源文件 SHA256、冻结策略身份和恢复起点。
- `metrics.jsonl`：逐更新的本地日志；`tensorboard` 模式还写入事件文件。
- `model_<已完成更新次数>.pt`：高层训练 checkpoint，不包含冻结底层权重。
- `fault_audit.json`：退出时的故障计划、实际退化帧及残留审计，只用于日志。
- `report.json`：运行状态、最后指标、累计步数和 checkpoint 路径；启动失败也会记录原因。

`Train/total_timesteps` 是高层 transition 数，`Train/total_vector_steps` 是全局高层
步数。`Train/total_low_level_steps` 汇总各环境首次终止前的有效底层步数。
`Train/mean_episode_seconds` 也使用有效底层步数计算；
`Train/total_simulated_seconds` 则使用全局高层步数，包含提前终止后仍执行的物理时间。

episode 平均值基于最近 100 个完整 episode。失败与纯超时分开计数；
`Train/failures_per_effective_minute` 的分母是所有环境有效底层时间之和。
环境提供的其他标量日志按出现次数取均值。
分类熵、KL、裁剪比例和 LP/VP 选择比例用于检查训练；模式概率不具有“一秒摔倒概率”含义。

## 无仿真检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD" python -m pytest -q \
  rsl_rl/tests/test_selector_actor_critic.py \
  rsl_rl/tests/test_selector_rollout_storage.py \
  rsl_rl/tests/test_discrete_ppo.py \
  rsl_rl/tests/test_selector_on_policy_runner.py \
  rsl_rl/tests/test_train_selector_ppo.py
```

这些检查使用真实网络/PPO 和环境协议替身，覆盖原地观测刷新、提前失败、超时末端
估值、奖励尺度、恢复训练和入口时序，不代替上面的真实 Isaac Sim 训练检查。
