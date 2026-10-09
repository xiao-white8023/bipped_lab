# G1 AMP 50／58 维切换

2026-10-08 更新。`g1_renet` 和 `g1_rough` 可以分别在对应任务的 cfg 中选择一种 AMP 格式，默认均为 50 维，使用由原始四段 PKL 转换的 `motion_amp_expert_no_ankle`。

| 任务 | 配置文件 | 修改的开关 |
| --- | --- | --- |
| `g1_renet` | `legged_lab/envs/g1/RENet_cfg.py` | `G1_RENET_AMP_DIM = 50` 或 `58` |
| `g1_rough` | `legged_lab/envs/g1/g1_rough_cfg.py` | `G1_ROUGH_AMP_DIM = 50` 或 `58` |

修改对应开关后重新启动训练进程。环境 cfg 和 agent cfg 自动使用相同格式，无需手动修改算法文件。

| 任务／开关值 | 关节角／关节速度 | 末端位置 | 数据集 | 加载器 | 判别器输入 |
| --- | --- | --- | --- | --- | --- |
| 两项任务／`50` | 19＋19，排除四个脚踝关节 | 12 维，pelvis 坐标系 | `motion_amp_expert_no_ankle` | `AMPLoader` | 100 维 |
| `g1_renet`／`58` | 23＋23，包含脚踝 pitch／roll | 同上 | `g1_23dof_locomotion_v3/amp` | `G1AnkleAMPLoader` | 116 维 |
| `g1_rough`／`58` | 23＋23，包含脚踝 pitch／roll | 同上 | `motion_amp_expert` | `G1AnkleAMPLoader` | 116 维 |

默认 50 维仅加载 `stand_to_walk.txt`、`walk_turn_around.txt`、`walk_turn_left.txt`、`walk_turn_right.txt`，不混入 `*_leg_v1` 或目录中的其他 TXT。每段 `MotionWeight=0.5`，归一化采样概率各 25%。完整机器人预览来自 `motion_visualization/original_pkl`。来源、帧数和关节顺序见 [数据说明](../legged_lab/envs/g1/datasets/motion_amp_expert_no_ankle/README.md)。RENet 的 58 维切换仍保留原 v3 四方向参考；rough 的 58 维使用同四个原始 PKL 的含脚踝版本。

选择 50 维会排除 AMP 的脚踝角/角速度；机器人仍控制全部 23 个关节，脚踝仍参与末端运动学。归一化统计、AMP 回放缓冲、判别器按选定维度创建，LP／VP 的策略输入和辅助学习接口保持原有尺寸。

## RENet 的 VP 本体历史

`RENet_cfg.py` 顶部的 `G1_RENET_VP_PROPRIO_HISTORY_FRAMES = 5` 是唯一的默认窗口开关；改为 `10` 并重新启动训练即可使用十帧。VP attention 为 5+2 或 10+2 个 token，GRU 序列为 5 或 10。原始 actor 缓冲仍为十帧、总输入 2120 维；LP 查询保持五帧，critic 保持十帧/1037 维。

改窗口不改变权重形状，但会改变 VP 动作。恢复原十帧检查点后，runner 按 cfg 的窗口续训并提示时间窗口变化；新检查点和普通策略导出记录 `vp_encoder_contract`。针对 2026-09-04 归档模型的冻结双分支导出/selector 仍保持原十帧模型。

一次训练使用一种格式。若自行改写 `amp_motion_files`，全部文件须匹配所选维度及关节顺序；58 维文件还须带有 `FrameLayout`、`JointCount`、`AMPJointNames` 元数据。配置、机器人 AMP 输出或数据格式不一致时，训练会报错，不能将缺失脚踝特征填零后当作原始数据使用。

在已配置的 IsaacLab 环境中，修改 cfg 后使用原命令：

```bash
cd /home/cjh/TienKung-Lab
python legged_lab/scripts/train.py --task g1_renet --headless
# g1_rough 使用 --task g1_rough
```

预览 TXT 始终使用完整机器人预览的 58 维格式，与 AMP 训练的 58 维字段含义不同。预览命令不需更改；`play_amp_animation.py --save_path ...` 导出的 AMP 文件会按所选 50／58 维生成，并记录关节顺序。

## 检查点恢复

- **同格式恢复**：50→50 和 58→58 可恢复策略、AMP 判别器、归一化统计及优化器状态。RENet 原有的辅助头／预处理迁移规则仍适用。
- **RENet 跨格式恢复**：50→58 或 58→50 保留策略及对应 Adam 状态，重新初始化 AMP 判别器、AMP 统计和判别器 Adam 状态；日志显示 `AMP reset:`。其他网络结构和输入约定仍需兼容。
- **普通 `g1_rough` 跨格式恢复**：当前 `AmpOnPolicyRunner` 要求检查点与 cfg 格式相同。不同格式会在加载策略之前报错；切换到匹配格式后恢复，或用 `resume=False` 开始新训练。

## 验证

`rsl_rl/tests/test_g1_amp_format_switch.py` 用实际 cfg 定义、两种真实数据文件、实际环境 AMP 关节选择／特征函数和两种真实 runner，在 CPU 上验证格式切换、一次 PPO／AMP 更新、保存与恢复、跨格式恢复规则及回放导出。环境状态使用测试代理，未运行 IsaacLab 物理仿真。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/cjh/TienKung-Lab/rsl_rl \
/home/cjh/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  rsl_rl/tests/test_g1_amp_format_switch.py \
  rsl_rl/tests/test_renet_ankle_amp.py -q
```

真实 IsaacLab 特征核对和 RENet 短时训练检查也跟随 cfg；可在有可用 GPU 的训练机执行：

```bash
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py --task g1_renet --headless --device cuda:0
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py --task g1_rough --headless --device cuda:0
python legged_lab/scripts/check_g1_renet_aux_training.py --headless \
  --report work/g1_amp_format_switch_20261004/isaac_validation.json
```

2026-10-08 已使用两个真实 CUDA 环境，分别验证当前 50 维 no_ankle 数据下 VP 五帧和十帧的 LP/VP 训练更新、检查点保存恢复与 TorchScript 导出，见 [本次报告](../work/g1_renet_no_ankle_vp5_20261008/REPORT.md)。此前 50/58 维切换的记录见 [历史验证报告](../work/g1_amp_format_switch_20261004/README.md)。需要分别验证两种格式时，修改对应 AMP 开关后再次执行。数据包内 2026-10-03 的 IsaacLab 记录属于此前 58 维默认格式的验证。
