# V3 故障采集与分析

这次实验将轻／中／重故障的**计划抽样概率**从 40%／40%／20% 调整为
10%／30%／60%。故障幅度范围、四个标签、20／50 步的预测窗口、回放过滤规则和
`[5, 283]` 网络输入保持原有定义。新字段仅用于审计，训练仍读取 features 和 labels。

`balanced_scenarios` 对接受样本的**当前活跃故障类别**设置配额：
默认 normal／VP-only／LP-only／dual 为 30%／25%／25%／20%。
它不控制“两路都安全／只有 LP 失败／只有 VP 失败／两路都失败”的数量。
例如，LP-only 故障样本即使很多，机器人依然可能在 1 秒内两路都安全。

重故障计划通常持续更久；重故障也可能更容易因回放不一致、测量无效等原因被拒绝。
因此，接受帧中的重故障占比不必是 60%。报告同时统计计划、候选帧和接受帧。

## 运行

在原来的 Isaac Lab 环境、仓库根目录运行。沿用上次的 checkpoint、reference policy、
seed 和 terrain histogram，仅修改等级比例并使用新输出目录：

```bash
python legged_lab/scripts/collect_selector_data.py \
  --headless --num-envs 4096 --seed 42 \
  --checkpoint-path /path/to/model.pt \
  --reference-policy /path/to/policy.pt \
  --terrain-histogram outputs/g1_renet_selector/terrain_distribution.json \
  --output outputs/g1_renet_selector_v3_severity_103060/dataset \
  --fault-severity-probabilities 0.10 0.30 0.60
```

请将模型路径和地形直方图路径换成上次使用的实际文件。要复现实验前的概率，
显式指定 `--fault-severity-probabilities 0.40 0.40 0.20`。
这是改变等级的抽样比例；本轮没有扩大各等级内部的噪声数值范围。

采集结束后自动生成 `collection_analysis.json`。发生可捕获异常时，也会保存已有分片、
不完整 manifest 和部分分析；中断数据仍会被现有训练完整性检查拒绝。
强制杀进程无法保证最后的待写数据被保存。`collection_run.json` 在采集开始前写入运行信息。

旧 V3 数据也可离线生成报告，无需启动 Isaac Sim：

```bash
python legged_lab/scripts/analyze_selector_data.py \
  --dataset outputs/g1_renet_selector/dataset
python legged_lab/scripts/analyze_selector_data.py \
  --dataset outputs/g1_renet_selector_v3_severity_103060/dataset
```

## 重点查看

| 报告字段 | 分析用途 |
| --- | --- |
| overall、by_split | 四个 head 的失败数／比例，以及两个窗口下的四种 LP/VP 联合结果 |
| by_group | 当前故障场景、活跃故障等级、故障类型×等级、地形、故障阶段、执行分支和切换阶段的失败比例 |
| first_failure_time | 第 1–50 控制步的首次失败计数，区分 0.4 秒前和 0.4–1 秒 |
| first_failure_causes | 首次失败当步的终止、接触、倾角、低离地高度；允许同一步多个条件同时触发 |
| candidate_safety | 候选时刻已经违反阈值的样本，避免把“已经失稳”全部理解成“提前预测” |
| events | 独立 episode／故障计划数量、其中有失败反事实标签的数量、跨 split 情况 |
| fault_parameter_summary | 按实际活跃故障类型汇总幅度、延迟、持续时间等实际抽样值 |
| collection_audit | 各场景×等级的过滤漏斗，以及回放过滤前／后和写入后的联合结果计数 |
| run_info、fault_config | seed、完整命令、配置、源码 SHA256、git commit、运行库版本 |

过滤漏斗按顺序统计：全部候选 → 历史完整 → 无不可观测近期故障起点 →
剩余 episode 足够 1 秒 → 两路标签有效 → 原有离散回放检查通过 → 配额选择后写入。
相邻计数之差表示该阶段排除的数量。

回放前的联合结果仅用于查看过滤倾向，取第二次回放的结果，不作为被拒绝样本的可靠监督。
首次原因的一致性不新增过滤条件；两次回放原因不同的失败单列为 ambiguous，
不混入稳定原因统计。这里的“原因”指标签触发条件，并非证明故障的物理根因。

`event_identity = [env_id, episode_serial, schedule_count]`，在同一 collection_run_id 内唯一。
它标识一个计划，包含尚未生效、恢复期以及无故障计划。计划数量只观察真实行为推进，
不计入反事实分支和回放探针；首次统计从正式采集开始，不回溯 burn-in。
events 中的失败表示该事件至少一个候选的反事实失败标签，不等于真实机器人摔倒次数。
`group_ids` 是地形 seed/type/level 分组，`scenario_seed` 是候选编号，均不能代替事件 ID。

旧 V3 缺失的新审计字段会显示 available_rows=0／null；不会从聚合指标猜测首次失败原因或事件 ID。
1 秒内没有失败只说明该窗口内未观测到失败，不能推出 1 秒之后安全。
后续是否延长窗口或做针对性采集，应结合单路失败样本数、独立事件数和过滤倾向决定。
