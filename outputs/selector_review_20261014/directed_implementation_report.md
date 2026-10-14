# 定向补采代码实现记录

日期：2026-10-14。代码已修改，未运行先导或正式采集，未训练新模型。

## 实现内容

| 位置 | 已完成的改动 |
| --- | --- |
| `legged_lab/selector/supplement.py` | directed_lp / directed_vp / directed_dual 三个纯来源 profile；来源、可观测效应、健康侧和实际结局联合验收；按故障组合记录稳定回放尝试、来源合格及写入的四类结局数量 |
| `legged_lab/selector/faults.py` | 两侧独立强度档位、延迟、持续时间及三种启用顺序；历史效应来源证明；新 VP 状态纳入快照；新事件不能用上一事件的旧帧冒充已可观测效应 |
| `legged_lab/selector/vp_faults.py` | 七类新 VP 机制；每个有限时间戳的源帧仅处理一次；两帧投递历史保留原扰动；冻结、延迟、抖动从已保存帧取数据；滤波拖尾使用新鲜时间戳并支持恢复；不分配重复的旧 VP 图像延迟缓存 |
| `legged_lab/selector/fault_kinds.py` | 原编号 0–18 保留，新机制追加为 19–25 |
| `legged_lab/selector/directed_audit.py` | 来源与 VP 参数、效应、时序的审计合同；审计字段不进入 selector 特征 |
| `legged_lab/selector/config.py` | 新机制的参数范围与校验；原 profile 默认不启用新 VP 帧处理 |
| `legged_lab/selector/collector.py` | 健康侧当前有效支持与时间戳检查；将来源及 VP 审计记录传给筛选器和写入器 |
| `legged_lab/selector/dataset.py` | 可选 source_metadata / vp_fault_metadata 字段；保留 V4 二标签和 [5,283] 特征结构 |
| `legged_lab/selector/analysis.py` | 从真实保存的样本验收来源；审计双路机制组合、先后顺序、独立档位、失败提前量、最大事件样本数和真实间隔；核对候选与特征时刻 |
| `legged_lab/selector/collection_audit.py` | 记录实际行为路径上的父事件机制组合数，不将回放分叉当成新事件 |
| `legged_lab/scripts/collect_selector_data.py` | 新 profile 与 --fault-source；定向 profile 自动绑定对应来源，禁止用 any 绕过来源检查；定向筛选仅用于训练分区 |
| 两份采集脚本 | 先导每库 1 万条，正式每库至少 10 万条；分别验收实际标签、来源、初始安全、审计字段、父事件上限与间隔 |

VP 新机制：不规则局部缺测、障碍边缘深度错配、空间相关偏差、运动相关亚像素形变、时延抖动与短时持帧、时间滤波拖尾、移动前景遮挡。几何错误可保留有效支持和新鲜时间戳；不把所有 VP 故障都处理成无效像素。

仅 LP 注入故障的目标库只写 [1,0]；仅 VP 注入故障的目标库只写 [0,1]；双侧注入故障的目标库只写 [1,1]。标签来自原物理回放，不根据故障类型重标。实际 [0,0] 对照可通过单独的双安全配额采集，不占三个失败库配额。

## 验证结果及边界

- 新测试 `rsl_rl/tests/test_selector_directed_supplement.py`：**53 项 CPU 测试通过**。包括反向结局排除、双侧可观测效应、纯来源配置、健康侧可用性、事件间隔与上限、全部 16 个 VP 故障家族的快照重放、逐环境随机独立、旧帧不重写、长期冻结、延迟、滤波恢复、缺失源时间戳、CLI 参数和数据写入/分析。
- 修改前后的 baseline 与 mixed 故障管理器，各使用 32 个环境、推进 150 个传感器更新；所有返回字段与逐环境 RNG 状态逐项完全一致。此项只验证故障输入生成，没有验证 PhysX 轨迹。
- Python 编译、两份 Bash 脚本语法，以及正式脚本内的三个 Python 程序语法检查通过。
- 修改前先检查了现有 `test_selector_safety_labels.py`：6 项通过、18 项失败。失败主要来自仍使用旧 short_horizon_steps / long_horizon_steps 和 fail_short / fail_long 接口；本次未修改该旧测试文件，也未改变物理标签合同。53 项新增测试中另验证了当前 50 步时间窗的提前超时剔除及最后一步超时保留。
- 当前用于验证的 `/home/cjh/miniconda3/envs/env_isaaclab/bin/python` 可以运行 Torch/pytest，但导入 AppLauncher 时缺少 isaacsim。因此尚未验证真实 Isaac 启动、渲染、PhysX 回放和采集产出，不能据此声称 selector 效果已经改善。原有采集启动回放门槛及逐样本四次有效一致标签检查继续执行。

VP 扰动作用于已通过环境处理的源帧，以米计算后重新裁剪归一化，不重复施加环境高斯噪声/模糊。它是输入层替代模型；完整相机物理渲染与故障参数的真实硬件分布仍需后续验证。本地部署深度处理的默认 invalid_depth_mode 为 training：0 在裁剪后成为近端值，非有限值填充为远端值，二者原始有效支持均为零。

## 运行入口

在能够运行 g1_renet 的完整 Isaac Python 环境中执行；沿用现有脚本的 SELECTOR_BASE、SELECTOR_TERRAIN、SELECTOR_CHECKPOINT、SELECTOR_REFERENCE，路径不同时通过这些环境变量指定。SELECTOR_PYTHON 应指向能够导入 torch / isaaclab / isaacsim 的解释器。

```bash
# 三库先导：各 10,000 条，默认使用独立 seed 8101 / 8102 / 8103。
bash legged_lab/selector/run_selector_directed_pilot.sh

# 先导审计通过后进行正式采集：各至少 100,000 条。
bash legged_lab/selector/run_selector_outcome_supplement.sh
```

先导与正式输出目录默认分开，已存在的输出根会被拒绝复用。每库保存 collection_analysis.json 和 quota_verification.json；成功后保存 outcome_supplement_summary.json。优先检查 supplement_coverage.outcome_yield_by_kind_pair、source_acceptance、directed_acceptance，以及 by_group 中的 dual_start_order / both_failure_lead_time_bin。故障组合的稳定回放产出并不等于自然发生概率。

双安全对照可使用 collect_selector_data.py 的 `--fault-profile directed_lp` / `directed_vp` / `directed_dual`，配合 `--supplement-split train --pair-quotas N 0 0 0 --target-samples N` 及原基准/地形/策略参数，使用新的 seed 和单独输出目录；相同来源与观测效应验收仍生效。

## 复核材料

- [代码修改补丁](directed_implementation.patch)
- [修改前源码快照](code_before_directed/legged_lab/selector/)
- [补采方案](directed_supplement_plan.md)

本次完成的是采集代码与 CPU 验证。是否提高单路救援识别、双失败精确率和误阻挡表现，需要先导采集、后续训练及原 validation 分布校准来判断。
