# G1 23DoF 最终动作 AMP 数据集 v3（包含脚踝）

`g1_renet` 或 `g1_rough` 设置对应的 AMP 开关为 58 时使用本目录。2026-10-03 根据用户要求，将 v2 中已验收的四段动作导出为包含全部 23 个关节的 AMP 特征，新增左右脚踝 pitch/roll 的角度与速度。原始 PKL 和预览文件与 v2 字节一致。2026-10-04，两项任务默认均已切为 50 维的 [新四段 PKL 数据](../g1_23dof_pkl_locomotion_v1/README.md)。详见 [切换说明](../../../../../docs/g1_amp_format_switch.md)。

| 文件 | 内容 | 帧数 | 时长 | 采样概率 |
| --- | --- | ---: | ---: | ---: |
| `walk_forward_turns` | 前走 v8，左转 90°、右转 90°、转身 180°及继续行走 | 1800 | 60 秒 | 25% |
| `walk_backward` | 后退 v2，髋部、躯干及摆臂协调最终版 | 1200 | 40 秒 | 25% |
| `walk_left` | 左侧走最终版 | 1200 | 40 秒 | 25% |
| `walk_right` | 右侧走最终版 | 1200 | 40 秒 | 25% |

共 5400 帧、180 秒，30 FPS，首尾各 3 秒自然站立。保留完整起步、移动、转向和收步。每段 `MotionWeight=1`，按片段等概率、片段内有效时间均匀采样；状态对相隔 0.02 秒，在线插值，`LoopMode=Clamp`。

## 训练格式

`amp/*.txt` 的 `Frames` 为 **T×58**：

| 切片 | 特征 | 单位 |
| --- | --- | --- |
| `[0:23]` | 全部 23 个关节角，包含脚踝 | rad |
| `[23:46]` | 同顺序的 23 个关节速度，包含脚踝 | rad/s |
| `[46:49]` | 左手连杆原点在 pelvis 坐标系的位置 | m |
| `[49:52]` | 右手连杆原点在 pelvis 坐标系的位置 | m |
| `[52:55]` | 左脚踝连杆原点在 pelvis 坐标系的位置 | m |
| `[55:58]` | 右脚踝连杆原点在 pelvis 坐标系的位置 | m |

关节顺序：左髋 pitch/roll/yaw、左膝、左脚踝 pitch/roll；右髋 pitch/roll/yaw、右膝、右脚踝 pitch/roll；腰 yaw；左肩 pitch/roll/yaw、左肘、左腕 roll；右肩 pitch/roll/yaw、右肘、右腕 roll。

JSON 中包含 `FrameLayout="g1_amp_joint_pos_vel_end_effectors_v1"`、`JointCount=23` 和 `AMPJointNames`。`G1AnkleAMPLoader` 检查格式和片段间关节顺序，runner 再与机器人实际 AMP 关节顺序核对。判别器输入是当前、下一状态拼接后的 **116 维**。

`visualization/*.txt` 也为 T×58，但顺序是 root XYZ(3)、root Euler(3)、关节角(23)、root 世界线速度(3)、root 局部角速度(3)、关节速度(23)。它不能用于 AMP 训练；加载器会拒绝缺少训练格式标记的预览文件。

`sources/*.pkl` 保存最终原始动作。导出保留关节角和根部姿态，速度使用每段内有限差分，末端位置通过 G1 MJCF FK 计算；没有新增 IK、调整高度、裁剪或改变速度。

## 使用和重建

选择 `G1_RENET_AMP_DIM=58` 时，自动设置 `amp_joint_count=23` 并接入四个 `amp/*.txt`：

```bash
conda activate env_isaaclab
source /home/cjh/isaac-sim/setup_conda_env.sh
cd /home/cjh/TienKung-Lab
python legged_lab/scripts/train.py --task g1_renet --headless
```

预览：

```bash
python legged_lab/scripts/play_amp_animation.py --task g1_renet --num_envs 1 \
  --motion_file legged_lab/envs/g1/datasets/g1_23dof_locomotion_v3/visualization/walk_forward_turns.txt \
  --root_height_offset 0.0 --repeat 1 --realtime
```

仅使用数据包内原始副本重建及验证：

```bash
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/build_g1_renet_final_dataset.py --from-snapshots
/home/cjh/miniconda3/envs/gmr/bin/python legged_lab/scripts/validate_g1_renet_final_dataset.py --device cuda:0
python legged_lab/scripts/test_g1_renet_amp_feature_parity.py --headless --device cuda:0
python legged_lab/scripts/check_g1_renet_aux_training.py --headless
```

默认加载器 `AMPLoader` 仍供历史 50 维任务使用，v2 数据文件保留。若重建旧格式，显式指定 `--amp-joint-count 19 --output-dir legged_lab/envs/g1/datasets/g1_23dof_locomotion_v2`。

### g1_rough

`g1_rough` 选择 `G1_ROUGH_AMP_DIM=58` 时，共享本目录的四个最终片段。其 AMP 观测包含全部 23 个关节及脚踝，使用 `G1AnkleAMPLoader`，状态 58 维、判别器输入 116 维；加载时检查专家和机器人关节顺序。当前默认为 `50`，使用新的 PKL 四段、`AMPLoader` 和 100 维判别器。

```bash
python legged_lab/scripts/train.py --task g1_rough --headless
```

预览时将上面的 `--task g1_renet` 改为 `--task g1_rough`。实际模型特征检查支持 `python legged_lab/scripts/test_g1_renet_amp_feature_parity.py --task g1_rough --headless`。

当前普通 `AmpOnPolicyRunner` 要求检查点与 cfg 的 AMP 格式相同。旧 50 维检查点可在 `G1_ROUGH_AMP_DIM=50` 时恢复；从 50 维检查点切换到本目录的 58 维格式，需要新训练（`resume=False`）。不同格式会在加载策略权重之前报错。

## 旧检查点

通过 `RENetAmpOnPolicyRunner.load()` 从 50 维 AMP 续训时，保留策略权重及对应 Adam 状态，重新初始化 116 维输入的 AMP 判别器、58 维 AMP 归一化统计量和判别器 Adam 状态。旧判别器不能直接当作含脚踝的新判别器使用。

反向切换到 50 维时也保留策略和对应 Adam 状态，重新初始化 100 维输入的判别器、50 维统计及判别器 Adam 状态。两种格式同格式再次恢复均正常加载 AMP；LP／VP 结构仍需与检查点兼容。

检查点记录 AMP 维度、格式和关节顺序。新格式再次加载时正常恢复判别器和归一化统计，不重复初始化。LP/VP 辅助学习的独立迁移逻辑仍适用。

## 来源与验证范围

前走参考 ACCAD 捕获的起停及行走风格，转向经过路线规划；后退及左侧走参考 Kimodo SOMA 人体风格并经过机器人规划、IK 和姿态修正；右侧走是左侧机器人动作的镜像。长片段是编排的机器人参考，不是等时长独立真人动捕。

- `manifest.json`：全部来源、动作阶段、关节/连杆顺序、模型及数据哈希。
- `validation.json`：全部 5400 帧、源姿态和速度、独立 URDF FK、实际环境特征函数、20 万条 GPU 状态对、20 ms 插值及 116 维判别器前向/梯度/奖励检查。
- `isaac_feature_parity.json`：实际 IsaacLab G1 模型的跨阶段姿态与世界坐标变换检查。
- `rebuild_validation.json`：原始副本独立重建，12 个数据文件字节一致。
- `isaac_training_smoke.json`：真实环境下 LP/VP 各 4 步采样与训练更新、58 维 AMP/116 维判别器接口和检查点保存/加载验证；75 项相关测试通过，1 项 sim2sim 测试模块因环境缺少 MuJoCo 跳过。

这些验证确认数据和训练接口正确；动力学跟踪与最终步态需正式训练后评估。
