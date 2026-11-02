# g1_film：同时消融 actor 和 critic 的 MoE

任务名仍为 `g1_film`。在 `legged_lab/envs/g1/g1_film_cfg.py` 的
`G129MOE_FILMAGENTENV` 中修改唯一开关：

```python
use_moe: bool = False
```

| 配置 | actor | critic | 默认日志根目录 |
| --- | --- | --- | --- |
| `False`，默认 | MLP，228→512→256→128→29 | MLP，1427→512→256→128→1 | `logs/g1_noamp_wo_moe` |
| `True` | 五专家 MoE | 独立五专家 MoE | `logs/g1_noamp` |

`__post_init__` 自动设置 `policy.class_name`、`experiment_name` 和 `run_name`。
保留的 `num_experts=5`、`gate_hidden_dim` 字段只用于 MoE 模式；MLP 模式下 runner
不向普通 ActorCritic 传递这两个参数。此消融同时改变 actor 和 critic 的结构与参数量。

两种模式共用 FiLM CNN、四帧深度、单层 64 维 GRU、十步本体历史、速度估计和
前方 99 点地形重建。算法仍为 `MoePPO`，保留两项辅助损失；MoE 均衡和门控熵
损失在两种模式下都关闭。`MoePPO` 名称并不表示策略必须含有 MoE。

已按保存的六月配置恢复：100 维单步本体观测（含四维左右腿相位）、124 维单步
critic 观测、速度监督偏移 102、七类混合地形、多方向命令，以及旧版
`gait_phase_contact` 奖励。移除了六月没有的两项腰部单独偏差惩罚。

参考配置：提交 `e61eba6734fe100a52c32af35bf1a2f726dc648b` 中
`logs/g1_noamp/2026-06-16_15-53-47/params/{agent,env}.yaml`。
这恢复了可核对的配置和网络结构，未声称恢复全部六月运行时代码。
例如，当前深度故障持续时间和地形辅助目标预处理仍沿用当前实现，两种模式相同。
严格比较时应从同一代码版本训练两种模式；六月已有数据与精确评测 checkpoint 的
绑定问题仍需保留。

在已经配置好的 Isaac Lab 训练环境、仓库根目录执行：

```bash
python legged_lab/scripts/train.py --task g1_film --headless --num_envs 4096 --seed 42 --max_iterations 50000
```

先保留 `use_moe=False` 训练消融；训练完整对照前将其改为 `True` 并重新启动进程。
默认 `resume=False`。两种结构应分别从头训练；如需续训，仅加载同一模式的 checkpoint。
当前 `--resume` 参数使用 `type=bool`，不要传入字符串形式的 `--resume False`；
省略该命令行参数即可沿用 cfg 中的 False。训练总预算和 checkpoint 选择规则应匹配。

启动日志应显示 actor 原始观测 3304、critic 观测 1427、融合特征 228；默认模式
两个头均为 `Sequential`，完整模式两个头均为 `MoeLayer`。
记录实际保存的配置、代码提交、checkpoint SHA256、评测命令与种子及 CSV 哈希。

CPU 检查（需 PyTorch、pytest 和仓库原有 Python 依赖）：

```bash
PYTHONPATH=rsl_rl:. python -m pytest -q tests/test_g1_film_moe_ablation.py
```

配置测试使用惰性的配置类型以避免启动 Isaac Sim；网络、梯度和 PPO 更新使用真实
PyTorch 实现。这些测试不替代 Isaac Sim 中的启动检查与实际训练。

本次验证：Python 3.10 / PyTorch 2.5.1 CPU，6 项检查通过；完整模式还通过了
六月 `2026-06-16_15-53-47/model_49000.pt` 的严格参数加载检查，输出动作维度为 29 且有限。
