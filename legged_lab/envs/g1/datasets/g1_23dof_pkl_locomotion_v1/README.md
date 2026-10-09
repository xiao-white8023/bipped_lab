# G1 新 PKL 的 50 维 AMP 数据集

2026-10-04。由 `datasets/pkl/*_leg_v1.pkl` 转换，只使用用户指定的新四段，不包含旧后退和侧向行走。`g1_renet` 在 `RENet_cfg.py` 中使用 `G1_RENET_AMP_DIM = 50`。2026-10-08 起，`g1_rough` 默认改用原始四份 PKL 转换出的 `motion_amp_expert_no_ankle/*.txt`，不再使用本目录。

| 片段 | 帧数 | 原始 FPS | 首末采样点时长 |
| --- | ---: | ---: | ---: |
| `stand_to_walk_leg_v1` | 593 | 39.974277 | 14.810 秒 |
| `walk_turn_around_leg_v1` | 530 | 39.941478 | 13.244 秒 |
| `walk_turn_left_leg_v1` | 412 | 40 | 10.275 秒 |
| `walk_turn_right_leg_v1` | 447 | 40 | 11.150 秒 |

合计 1,982 帧，首末采样点时长合计约 49.479 秒。没有重采样、裁剪、延长、再次 IK 或调整根部高度；左腿减幅及并排站立结尾完整保留。各片段保留约 1.5 秒末尾站立，并使用 `LoopMode="Clamp"`，训练采样不会将结尾与开头拼接。

## 文件和特征

- `sources/*.pkl`：最新四段的完整源文件快照，SHA-256 与上游一致；对应 JSON 保留修改记录。
- `amp/*.txt`：实际用于训练的 JSON 动作文件，每帧 **50 维**。
- `visualization/*.txt`：IsaacLab 完整机器人回放，每帧 **58 维**，字段含义与 AMP 不同。
- `manifest.json`：源哈希、关节顺序、帧率、模型、转换方式和各片段指标。
- `validation.json`：实际加载器、独立 URDF FK、环境特征方法和判别器检查结果。

| AMP 索引（Python 切片） | 特征 |
| --- | --- |
| `[0:19]` | 按 `AMPJointNames` 顺序的 19 个非脚踝关节角，rad |
| `[19:38]` | 同序关节速度，rad/s |
| `[38:41]` | 左手 link 原点在 pelvis 坐标系的位置，m |
| `[41:44]` | 右手 link 原点在 pelvis 坐标系的位置，m |
| `[44:47]` | 左脚踝 link 原点在 pelvis 坐标系的位置，m |
| `[47:50]` | 右脚踝 link 原点在 pelvis 坐标系的位置，m |

四个脚踝的角度和速度不作为 AMP 关节特征。末端位置仍使用完整 23 关节模型做 FK；机器人策略仍控制全部 23 个关节。速度按每段实际 FPS 求中心差分，端点用二阶单边差分。`AMPLoader` 在线性插值后形成相隔 20 ms 的状态对，判别器输入为 **100 维**。每段 `MotionWeight=1`，采样概率各 25%。

## 训练和预览

在已配置的 IsaacLab Python 环境中：

```bash
cd /home/cjh/TienKung-Lab
python legged_lab/scripts/train.py --task g1_renet --headless
```

逐段查看转换后的完整机器人回放：

```bash
python legged_lab/scripts/play_amp_animation.py --task g1_renet --motion_file \
  legged_lab/envs/g1/datasets/g1_23dof_pkl_locomotion_v1/visualization/walk_turn_left_leg_v1.txt
```

将最后的文件名换成表中其他片段即可。此命令需桌面显示，不加 `--headless`。

两个任务的 AMP 开关相互独立。`G1_RENET_AMP_DIM` 改回 58 会切回原来的 v3 四方向数据；当前这套修改版 PKL 只导出 50 维。`g1_rough` 的 50/58 开关现在分别选择原始四段在 `motion_amp_expert_no_ankle` / `motion_amp_expert` 的对应格式。原 v2/v3 数据均保留。训练速度命令范围没有随本次转换更改；这套专家参考只包含前走和转向。普通 `g1_rough` 不能直接加载旧 58 维 AMP 检查点到 50 维配置，当前 `resume=False` 使用新训练；同维度检查点可按原规则恢复。

## 复现和验证

```bash
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/build_g1_pkl_amp_dataset.py
# 从数据包内快照重建，不依赖上游 pkl 目录：
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/build_g1_pkl_amp_dataset.py --from-snapshots
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/validate_g1_renet_final_dataset.py \
  --dataset-dir legged_lab/envs/g1/datasets/g1_23dof_pkl_locomotion_v1
```

验证覆盖全部源帧的姿态、关节速度、URDF FK 与实际环境 AMP 特征方法，以及 200,000 个 20 ms 状态对预加载、回放加载器和判别器前向/梯度惩罚/反向传播。当前验证使用 CPU 和环境状态代理，未运行 IsaacLab 物理训练；生成参考数据不等于已验证策略的跟踪效果。

本次检查通过；独立 URDF FK 与环境特征的最大绝对误差约 `4.13e-7`。`test_g1_amp_format_switch.py` 和 `test_renet_ankle_amp.py` 共 44 项测试通过，包含真实配置提取、50/58 维切换、PPO/AMP 更新和检查点恢复。当前机器 CUDA 不可用，尚未执行下面的真实 IsaacLab 检查。

有可用 CUDA 的训练机可进一步核对真实 IsaacLab 资产的特征：

```bash
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py \
  --task g1_renet --headless --device cuda:0
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py \
  --task g1_rough --headless --device cuda:0
```
