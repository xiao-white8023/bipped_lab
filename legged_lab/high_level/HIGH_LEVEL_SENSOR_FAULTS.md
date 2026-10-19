# 高层选择器感知故障接口

高层故障与缓存独立放在 `legged_lab/high_level/`。LP 地图扩展和几何变换由本目录维护，
原 `legged_lab/selector/` 的故障实现及包入口保持修改前状态；只读复用其中的故障编号、
随机数工具及冻结策略加载接口，不再向原文件增加高层方法或导出。

`HighLevelSensorFaultPipeline` 只处理传感器故障和缓存，不创建仿真环境、策略或 PPO。
高层环境在两路 **10 Hz 新帧** 到达时调用一次 `update`，在 50 Hz 底层和
5 Hz 高层使用 `read`。读取不会重新加噪、推进故障或消耗随机数。

```python
from legged_lab.high_level import HighLevelSensorFaultPipeline, SensorFrame

faults = HighLevelSensorFaultPipeline(num_envs, env_cfg.faults, seed=42, device=device)

# 所有输入只包含 env_ids 对应的 B 行；时间戳用秒，sequence 为 int64 采集序号。
lp = SensorFrame(clean_lp_height, raw_lp_support, lp_source_stamp, delivery_stamp, lp_sequence)
vp = SensorFrame(cropped_depth_metres, raw_vp_support, vp_source_stamp, delivery_stamp, vp_sequence)
observations = faults.update(lp, vp, torso_world_xyzyaw, env_ids=env_ids)

# 同一份数据同时组装高层 Actor 观测和冻结 RENet 输入。
# LP: (N,11,17)，VP: (N,2,18,32)，历史顺序为旧→新。
lp_height = observations.lp
depth_history = observations.vp

# 仅 episode reset 时清理对应行；切换 LP/VP 时不要调用 reset。
faults.reset(reset_env_ids)
```

LP 输入已经按冻结策略约定变换为相对高程：行 y、列 x，间距 0.1 m，未知值 -1.1，
有效值 [-1,1]。VP 输入为裁剪后的 18×32 **米制单帧**；有效支持必须在填补、裁剪前取得。
不要传入归一化历史。管线注入 Film 故障后，使用 RENet 的 [0.1,2.5] 裁剪并除以 2.5，
因此缺测零的数值输出是 0.04，而有效支持保持 0。LP/VP 各自的 support 必须与数据配对。

两路旧 PGTT/高斯/模糊和旧 selector 故障管线必须关闭；尘雾内置扰动和地图随机游走保留。
未选中的模式也要持续更新，切换不得刷新故障。当前接口要求两路新帧在共同的 10 Hz tick
交付；重复采集序号忽略，乱序或跳过 tick 明确报错。对未来真实异步传感器，需要在调用方
提供同步／交付层，不能把 5 Hz 决策当作感知时钟。

## 时间与状态

- LP 冻结／延迟同步读取数据、支持度和原测量来源时间；`delivery_stamp` 仍是当前交付时间。
- VP 冻结模拟新包重复旧图，图像和支持度固定，新包时间正常更新。内容年龄只出现在
  `audit()`，不能拼入 Actor。
- 线性／累积漂移停止增加后按 τ=0.5 s 连续恢复；1 mm 和 0.01° 以下归零。
- 地图残影在新计划之间保留，需要有效地面确认或移出保留区域才能清除。
- `read()` 返回副本；`initialized=False` 的行尚未收到首帧，调用方不能将其当成有效测量。
- 部分行 reset 不改变其他行的随机流。统计与种子记录覆盖管线整个生命周期，reset 不清统计。

## 计划与统计

正常／仅 LP／仅 VP／双路按 30%／25%／25%／20% 抽取。LP 11 类、VP 8 类各自等概率；
幅度范围均匀采样，与时长独立。正常计划 1–5 s，计划间恢复间隔 0.5–3 s。
全部详细范围由 `high_level_fault_config.py` 定义；`g1_selector_cfg.py` 中的子配置可覆盖参数。

`audit()` 单独提供配置、种子、计划／类型计数、计划注入帧、当前交付帧退化、历史可见退化、
无当前 LP 注入时的地图残留帧，以及内容时间等隐藏诊断。故障标签、剩余时间和内部内容年龄
都不能输入 Actor。计划占比不是实际时间占比，平坦场景的几何故障也不一定产生可见影响。

复现实例（在项目根目录，用安装了 Torch 的 Python）：

```bash
PYTHONPATH=. python legged_lab/scripts/check_selector_sensor_faults.py \
  --num-envs 32 --frames 1200 --seed 42 \
  --output outputs/selector_sensor_faults/replay_seed42.json
```

该回放只验证模块行为和统计，不代表机器人摔倒率或 PPO 训练效果。

`selector.terrain_curriculum=False` 要求高层环境保持每个环境的初始地形等级，episode
reset 不调用 `update_terrain_levels()`。地形生成器仍按难度分层生成现有地形，初始最高等级
仍为 5；这与训练过程中按行走表现升降级分开控制，不修改共享 `ROUGH_TERRAINS_CFG`。

## 高层环境接入

`legged_lab.envs.g1.g1_selector_env.G1SelectorEnv` 独立创建仿真场景，并通过
`load_frozen_adapter()` 加载和校验冻结 RENet。环境不依赖任务注册、训练 runner 或 AMP
判别器。它接收整数模式 `0=VP、1=LP`，直接执行所选分支，不做确认、驻留或动作混合。

一次高层 `step()` 对应 0.2 s：十个 50 Hz 底层步，每步四个 200 Hz 物理子步。两路感知
在每五个底层步共同采集一次；本体历史每个有效底层步更新。环境用整数控制节拍生成
float64 时间，不根据读取次数或高层决策次数推动故障。初始化和 episode reset 从新位姿
取得真实首帧，再填充初始历史；模式切换不会重置历史、故障或残影。

高层和冻结底层共享相同的本体及故障缓存。

高层相机使用 `depth_clipping_behavior="none"` 保留未命中的 NaN/Inf，先取得原始支持度，
再由统一故障管线完成填补、裁剪和归一化，避免把填补后的最大距离当作有效测量。
归档 checkpoint 的 `pgtt_full` 来源名称兼容当前 `pgtt_full_atten`；其余观测契约仍逐项校验。

默认输入分别为：

| 使用方 | 内容 | 维度 |
| --- | --- | ---: |
| 高层 Actor | 本体历史、两帧 VP 及支持度、LP 及支持度、8 维可观测状态 | 3466 |
| 高层 Critic | Actor 内容加当前真实线速度、双脚接触／高度和干净高程 | 3660 |
| 冻结 RENet | 本体历史 780、LP 187、VP 1152、当前模式 1 | 2120 |

`get_observations()` 返回 `(actor_obs, extras)`，只读取已经准备的缓存；Critic 输入位于
`extras["observations"]["critic"]`。不要在读取时滚动历史、重新施加故障或更新本体噪声。
Actor 状态中的 VP 来源年龄使用最新帧的可观测时间，不使用 `audit()` 中的冻结内容时间。

六项独立奖励在每个有效底层步以 `dt=0.02` 计算，高层直接求和。实际模式变化只在首次
执行的一步设置 `selector_switch_event`，一次切换贡献 -0.02；真实失败一次贡献 -4.0。
动作缓存保存实际执行的 23 维归一化动作，不保存高层模式编号。新 episode 的首个动作
填满对应行的动作历史，动作变化不跨 episode 计算。

若某行在第 4 个底层步终止，本条高层 transition 只累计到第 4 步，并立即复制末端
Actor/Critic 观测。后续物理子步仍会全局推进，但该行保持最后关节目标，不再推进其有效
命令、历史、故障、计数或奖励；高层周期末才统一重置。返回观测中已终止行属于新 episode，
`extras["terminal_*_observations"]` 则属于终止前的 episode，两者不能混用。

`terminal_env_ids` 对应末端观测行；`low_level_steps` 记录每条 transition 的实际有效
底层步数；`time_outs` 仅标记纯超时。躯干接触或倾斜失败不 bootstrap，且与超时同时发生
时优先按失败处理。环境不自行添加 Critic 价值补偿，后续 PPO 应使用末端 Critic 观测处理
超时。

## 真实仿真检查

在已经安装 IsaacLab 的 Python 环境中，从项目根目录执行：

```bash
python legged_lab/scripts/check_g1_selector_env.py \
  --headless --device cuda:0 --num-envs 4 --steps 12 --seed 42 \
  --output outputs/g1_selector_env
```

脚本直接构造环境，依次执行固定 VP、固定 LP 和交替选择；不注册任务，不创建 PPO 或
runner。可加 `--disable-faults` 检查无故障基线，这只关闭统一感知故障，保留本体噪声、
命令、重置范围及六项奖励。

检查使用独立的地形配置副本：2 行、3 列、初始最高等级 1、固定种子并关闭地形缓存，
保留按难度分层的地形生成方式。它核验跨 step 和部分 reset 的地形等级、原点不变，
每个高层步恰好 40 次物理推进，未终止行恰好两次感知采集，以及重复观测读取不修改缓存。
终止行额外取得 reset 首帧，其采集数与未终止行分别记录。

输出目录包含完整 `env_cfg.yaml` 和 `report.json`。JSON 记录配置覆盖项、种子、冻结策略
身份、逐步奖励／done／有效步数、采集数、故障审计和最终通过／失败状态。异常同样写入
失败报告并关闭仿真应用。默认短序列不保证启动或覆盖全部 19 类故障；完整故障覆盖应结合
上述模块回放和测试核验。脚本通过表示真实环境接口和时序检查通过，不代表切换策略已训练
或机器人安全性能改善；只有实际运行生成的报告才能作为真实仿真结果。
