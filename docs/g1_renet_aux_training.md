# G1 RENet 辅助训练与 VP 不移动惩罚

2026-10-08 更新。适用于 `g1_renet`；LP/VP 仍共用 actor。`RENet_cfg.py` 的 `G1_RENET_AMP_DIM` 默认 50，仅使用 `motion_amp_expert_no_ankle` 中由原始 PKL 转换的起步、掉头、左转、右转四段；选择 58 切回原 v3 四方向参考（包含脚踝）。VP 本体窗口通过 `G1_RENET_VP_PROPRIO_HISTORY_FRAMES` 配置，默认 5、可改为 10。当前 actor=2120、critic=1037；LP 脚高头关闭、VP 脚高头开启。环境和 agent 的 AMP 格式同步切换，详见 [配置说明](g1_amp_format_switch.md)。下文带历史日期的验证结果只代表当时配置。

## VP 与物理随机化从训练开始，LP 噪声保留第 30000 次启用

LP 噪声仍以 runner 的全局 PPO 训练迭代数判断：0～29999 关闭，达到 30000 时启用；不是仿真步数，也不是 LP 单独累计的迭代数。续训使用检查点恢复的迭代数，在第一轮采样前同步 LP 阶段。VP 和四项物理随机化不再等待此阈值。

- LP PGTT 建图噪声：`lp_height.noise.start_iteration=30000`；高度噪声、偏置、XY/yaw 偏置、dropout、块缺失、量化和数据包延迟均受该开关控制。显式 `force_active` 仍可用于噪声回放评估。
- VP 深度图噪声：`depth_noise_start_iteration=0`；从第一张新采集图像开始应用距离相关高斯噪声和 50% 概率模糊，已存历史帧不会重复加工。噪声参数和模糊概率保持原配置；`add_camera_noise=False` 仍关闭两种噪声。
- `physics_material`、`add_base_mass`：恢复 `startup` 事件，创建环境时各执行一次，之后保持；摩擦、恢复系数、质量范围不变。`renet.domain_rand_start_iteration=0`。
- `push_robot`：从第一个控制步开始计时，按原有 10～15 秒间隔触发，XY 推力速度范围仍为 ±1 m/s。
- `action_delay`：当前 `enable=False`，首次及后续 reset 均设为 0，执行当前动作。保留 `min_delay=0, max_delay=5` 参数；显式重新开启后才会按 episode 采样 0～5 个控制步（0～100 ms）。

材质、质量和推力随机化由 LP/VP 共用，动作延迟目前独立关闭。本体感知噪声和其他启动/reset 事件维持各自原有配置。阶段日志为 `LP/noise_enabled`、`VP/noise_enabled`、`VP/noise_stage_start_iteration`、`DomainRand/enabled`、`DomainRand/start_iteration` 和 `ActionDelay/mean_steps`。

2026-10-04 本次恢复配置的 56 项 CPU 测试通过，覆盖 VP 从第 0 次迭代启用、概率模糊、启动随机化、推力计时、原生动作延迟 buffer、LP 的 29999/30000 边界及原有奖励计算。真实 IsaacLab 检查脚本已同步到新配置，本次未运行 GPU 物理集成检查；下方此前的 GPU 结果属于当时的配置。

## VP 收到命令但不移动

`legged_lab/mdp/rewards.py` 原来已有两个相关函数，但 RENet 没有启用：

- `idle_when_commanded`：根据 XY 命令和实际速度的模长判断不移动，覆盖前进、后退、侧移。
- `dont_wait`：只针对前向命令，对负向、侧向和纯转向不适用。

新增 `vp_idle_when_commanded`，在 `RENet_cfg.py` 的 `Reward` 中启用，当前权重 `-0.1`：

| 条件 | 是否触发原始惩罚 |
| --- | --- |
| VP，XY 命令模长 > 0.2 m/s，实际 XY 速度模长 < 0.1 m/s | 1 |
| VP，XY 命令模长 ≤ 0.2 m/s，转向命令绝对值 > 0.2 rad/s，实际偏航角速度绝对值 < 0.1 rad/s | 1 |
| LP 或零命令自然站立 | 0 |
| 对应实际速度达到阈值 | 0 |

使用动作执行前缓存的命令与模式，避免步末命令重采样造成归因错误。惩罚不判断移动方向；已有速度跟踪奖励继续约束方向和速度大小。起步加速到阈值前也可能短暂受到此惩罚；它不是带等待计时器的长期卡住检测。

奖励管理器仍按仿真控制步长缩放奖励；`-0.1` 是配置权重，不是每 20 ms 直接扣 0.1 分。

## 足底支撑不足惩罚

RENet 还启用了左右脚的 `stair_edge_penalty_L/R`，均调用现有 `single_foot_contact_area_penalty`，每项权重 `-0.05`、高度阈值 `0.04 m`，在 LP/VP 两种模式下共用。只有对应脚当前接触力超过 1 N 时，才按 `1 - 支撑射线比例` 处罚；正常完整支撑返回 0，半数支撑返回 0.5，摆动脚返回 0。日志为 `Episode_Reward/stair_edge_penalty_L/R`。奖励管理器仍按控制步长缩放。

2026-10-03 将每只脚的射线从 14 根扩大到 125 根。旧 `GridPatternCfg` 虽配置为 12×3.5 cm，实际生成的横向点只有 `[-1.75, 0.25] cm`，集中在足底中间且不对称。新 `FootSolePatternCfg` 从 G1 23 自由度的左右 `ankle_roll_link.STL` 最底部 1 mm 内的顶点求凸轮廓，以 1 cm 间距、2 mm 内缩筛选点，局部 X 为 `[-6, 14] cm`、Y 为 `[-3, 3] cm`，覆盖脚跟、脚尖和两侧，并排除圆角外的网格点。射线安装 Z 仍为 `-0.025 m`，跟随足部完整旋转（`ray_alignment="base"`），更新周期 20 ms。

这是鞋底支撑不足的采样估计；当前 URDF 每只脚的碰撞几何是四个小球，并非连续鞋底碰撞面。扩大射线也扩大了 LP/VP 脚高标签的地面平均范围，标签维数和无效点掩码规则不变。覆盖图和数值记录在 `work/g1_renet_vp5_foot_20261003/sole_coverage.png`、`sole_coverage.json`。

## VP 最近五帧本体感知

只需修改 `RENet_cfg.py` 顶部的 `G1_RENET_VP_PROPRIO_HISTORY_FRAMES = 5`，即可统一设置 policy 的 VP 窗口；改为 `10` 恢复十帧，重启训练生效。默认五帧时，VP 在融合前截取最近 5×78 维本体感知。attention 接收 5 个本体 token 与原有 2 个深度 token，GRU 的序列长度为 5；速度估计、地形重建、左右脚高预测均读取这个 VP latent，较早五帧不会影响这些输出或其梯度。20 ms 控制步长下，最早到最新样本相隔 80 ms；十帧时为 180 ms。

当前 LP/VP critic 均保留十帧：`robot.critic_obs_history_length=10`、`policy.critic_history_frames=10`、`policy.vp_critic_history_frames=10`、`algorithm.critic_history_len=10`。critic 输入为 10×85＋187＝1037 维。actor 缓冲保持十帧/2120 维接口，AME LP 查询保持五帧，VP 按自己的窗口读取。速度监督取当前状态 `[843:846]`，脚高取 `[848:850]`，地形为最后 187 维。

新检查点和 `play_new_renet.py` 导出的元数据记录 `vp_encoder_contract`。旧检查点无此记录时按原先完整十帧解释，允许保留网络权重和 Adam 状态继续训练，并提示 VP 从十帧切换到五帧后的动作可能变化。参数形状不变不代表 VP 行为不变；同权重 LP 输出保持一致。针对 2026-09-04 原检查点的冻结双分支导出与 selector 仍重建归档十帧模型。

### 历史五帧 critic 迁移（非当前默认）

2026-10-04 最终将 LP、VP critic 都改为五帧：`robot.critic_obs_history_length=5`、`policy.critic_history_frames=5`、`policy.vp_critic_history_frames=5`、`algorithm.critic_history_len=5`。单帧特权观测仍为 85 维，两种模式的 critic 输入都为 5×85＋187＝612 维，不再保留旧五帧的零占位。外部 actor 缓冲保持 10 帧/2120 维接口，当前 AME LP 和 VP actor 实际均只编码最近五帧；VP 深度历史仍为两帧。

动作采样、最后状态的回报 bootstrap 和 PPO mini-batch 更新继续支持与 critic 对应的 actor 模式标记，以兼容此前 LP 十帧/VP 五帧配置。当前两种模式窗口相同，直接使用五帧 critic。监督标签仍取当前真实状态：速度索引为 `[418:421]`，脚高索引为 `[423:425]`，地形为最后 187 维；有效性标记随样本同步洗牌。

新检查点记录 `critic_observation_contract`。缺少此字段的旧模型根据 critic 首层输入宽度推断历史长度。1037→612 维迁移仅裁剪 `critic.0.weight` 的前 425 列，保留最近五帧及地形权重，并同样裁剪该参数的 Adam 动量；如存在 critic 观测归一化统计，也裁剪对应列。actor 权重、其他兼容参数和原检查点文件保留。迁移后 LP/VP 价值估计可能改变，需要继续训练适应；五帧检查点正常完整恢复，不会重复裁剪。

此次五帧 critic 修改的 CPU 检查覆盖五帧 PPO 及全部辅助任务更新、最新速度/脚高标签索引、旧 critic 权重和 Adam/观测归一化统计裁剪、50/58 维数据加载和再次恢复，以及 actor TorchScript 导出；也保留了此前混合十帧/五帧模式的兼容检查。相关测试 104 项通过。

实际读取 `/home/cjh/桌面/logs/g1_renet/2026-09-04_14-11-25/model_60000.pt`，使用当前完整网络尺寸完成五帧迁移和一次 CPU PPO/AMP 更新，损失均有限；首层 critic 变为 `(512, 612)`，actor 权重在恢复时保留。该旧模型同时按已有规则初始化 LP 脚高头、固定尺度 VP 地形头和旧版 AMP 预处理状态。原文件未修改；本次未运行真实 IsaacLab 物理训练。

## 固定 VP 地形重建尺度

RENet 使用 `terrain_recon_target_normalization="fixed"`。目标仍取无噪声 critic 的真实高度扫描前方 99 点，使用现有高度偏置与比例，裁剪到 `[-1, 1]`，不再按 minibatch 均值、标准差转换。

这使同一地形状态在不同 minibatch 中具有相同监督目标，也保持与 LP 地形目标一致的数值含义。当前高度比例为 1，高度偏置为 0.5 m，目标不是绝对地面 Z 坐标。LP 重建完整 187 点，VP 仍重建前方 99 点。

VP 重建权重保留 `0.5`，warmup 保留 `500`。固定尺度会改变损失和梯度量级，应结合高度误差和实际行走评估后再调权重。消除变化的标签不等于已经证明闭环步态或通过率改善。

其他 AMP 任务默认仍使用原来的 `"batch"` 目标模式。

VP 深度预处理的高斯模糊已改为概率触发：`RENet_cfg.py` 中 `depth_blur_probability=0.5`，每个环境每次新图像采集独立选择，触发时使用 3×3、sigma=1.0 的模糊。0 关闭，1 始终启用；`scene.camera.add_camera_noise=False` 同时关闭模糊与深度高斯噪声。采集周期仍为 100 ms，两帧历史保留各自采集时的模糊结果，不在控制步间重复处理。距离相关的深度高斯噪声参数保留。

## LP 脚高辅助学习

此前版本曾新增 LP 脚高头；当前 AME 配置保持 `use_lp_feet_height_prediction=False`、`lp_feet_height_coef=0.0`，没有实例化 LP 脚高网络。VP 原脚高预测头和系数 0.2 保留。下面的 LP 脚高说明及历史验证记录属于此前开启该头的版本。

- 输入：当前本体感知与融合后的 LP latent，预测左、右脚高度。
- 损失：独立 MSE，`lp_feet_height_coef=0.2`，`lp_feet_height_warmup_iters=500`。
- VP 原来的预测头与系数 `0.2` 保留。
- 与速度、地形辅助任务一样，两个分支每次更新都接受监督，不受当前执行 LP/VP 的 actor mask 限制。
- 新训练前 500 次更新中，脚高系数从 0 线性增至 0.2。

脚高定义仍是脚部射线传感器安装点相对于有效命中地面的平均高度，不能直接当作鞋底离地高度。

修正无效射线的标签处理：只平均有效命中；某只脚全部无效或传感器位置无效时，该脚标签填 0 保持 critic 有限，并通过独立有效性标记跳过监督。输入中的模拟地图丢点不会因此跳过地形学习。

有效性标记随当前状态存入 rollout，并使用与观测相同的索引洗牌。当前 actor 为 2120 维，critic 为 1037 维（10×85 本体/特权历史＋187 高度点）。AMP 当前选择原始 PKL 的 no_ankle 四段 50 维数据，判别器状态对输入为 100 维；切换至 v3 58 维时输入为 116 维。

地形 GT 同样使用独立的 `terrain_height_valid`（187 个布尔值）。有效点要求命中坐标、参考高度、计算结果和射线距离均有限，距离不超过传感器上限；无效点在 critic 中填 0。掩码在动作采样前复制，随 rollout 和 mini-batch 的同一索引传递。LP 全图/前方目标与 VP 前方目标使用各自相同的空间索引，只对有效点求平均 MSE；全无效时损失和梯度为 0。LP/VP 固定目标尺度不变，输入中的模拟地图丢点也不会自动使真实地形 GT 无效。

AMP 统计现在只积累原始当前帧 policy/expert 状态；分类和梯度惩罚共享同一组标准化专家输入，并在标准化空间求输入梯度。训练循环复用同一个 AMP 损失实现。

TensorBoard 新增：

- `Loss/feet_height/lp`
- `Loss/feet_height/vp`
- `Loss/feet_height/lp_coef_current`
- `Loss/feet_height/valid_fraction`
- `Loss/terrain_height/valid_fraction`

保留原来的 `Loss/feet_height`（VP）以及速度和地形损失日志。

## 检查点

新检查点记录辅助目标约定和策略参数名。通过 `RENetAmpOnPolicyRunner.load()` 加载旧检查点时：

1. 保留已有 actor、critic、LP/VP 编码器；判别器是否保留取决于 AMP 特征和预处理约定。
2. 当前配置关闭 LP 脚高头，不创建新的 LP 脚高参数；此前开启该头的历史版本另有初始化规则。
3. 若 VP 地形目标由 batch 尺度改为 fixed 尺度，初始化该辅助解码头，避免直接将旧标准化输出视为物理高度编码。
4. 按参数名和优化器分组迁移旧 Adam 状态，保留未改动参数的动量；新增或重置的辅助头使用新 Adam 状态。

旧 LP 输入和 AME 编码器约定的检查仍保留。形状不兼容、未知键、部分缺失预测头都会报错。迁移不会覆盖原检查点文件。新格式检查点再次加载时不会重复重置地形头。

从旧 50 维 AMP 检查点迁移到含脚踝的 58 维 AMP 时，保留策略权重和对应 Adam 状态，初始化 116 维输入的判别器、58 维 AMP 归一化统计及判别器 Adam 状态。数据格式说明见 `legged_lab/envs/g1/datasets/g1_23dof_locomotion_v3/README.md`。

反向切换 58→50 时同样保留策略及其 Adam 状态，初始化 100 维输入的 AMP 判别器、50 维统计和判别器 Adam 状态。同格式恢复按现有特征与预处理约定加载，不因 cfg 新增开关而重复重置 AMP。

旧 58 维检查点如果没有 AMP 预处理标记，或仍是 v1 约定，归一化统计可能受旧混合尺度逻辑污染。首次恢复时也重建 AMP 判别器、统计量及判别器 Adam 状态，保留策略、已有辅助头和对应 Adam 状态。日志以 `AMP reset:` 明确提示。新检查点采用 v2 约定，记录原始统计来源和标准化梯度惩罚空间，再次加载完整恢复 AMP。未知预处理、关节顺序和格式不兼容仍报错。迁移应在新建 runner 上执行。

现有 warmup 按训练迭代计数。若从超过 500 次迭代的旧检查点续训，脚高系数已处于完整值 0.2；不会自动重新计时。

## 验证

数值与梯度测试覆盖 VP 惩罚、固定目标、无效脚高标签、LP 编码器梯度、真实 AMP PPO 更新、样本与有效性标记的对齐、旧检查点动作一致性和 Adam 状态迁移。

```bash
cd /home/cjh/TienKung-Lab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/cjh/TienKung-Lab/rsl_rl \
/home/cjh/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  rsl_rl/tests/test_renet_auxiliary_updates.py \
  rsl_rl/tests/test_renet_lp_vp_training.py \
  rsl_rl/tests/test_lp_height.py -q
```

这验证实现和训练接口。长期训练收益需通过相同地形、命令和随机种子的 LP/VP 对照评估确认。

加入脚踝后的相关测试共 75 项通过、1 项跳过。IsaacLab CUDA 集成检查也通过：2 个真实 G1 环境、缩小后的粗糙地形生成器，LP/VP 各采样 4 步并更新参数，验证两路脚高损失、有效标签、58 维 AMP/116 维判别器及检查点保存/加载。结果保存在 `work/g1_renet_aux_20261003/isaac_validation.json`。

重新运行短时集成检查：

```bash
conda activate env_isaaclab
source /home/cjh/isaac-sim/setup_conda_env.sh
cd /home/cjh/TienKung-Lab
python legged_lab/scripts/check_g1_renet_aux_training.py --headless
```

该脚本使用临时日志与检查点，不会开始长时间训练。

足底轮廓射线与 VP 五帧历史修改后，相关测试 126 项通过，IsaacLab CUDA 集成检查通过；实际每脚 125 根射线，VP GRU 序列长度为 5。报告见 `work/g1_renet_vp5_foot_20261003/README.md`。
