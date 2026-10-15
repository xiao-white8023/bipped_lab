# G1 23dof 脚体积点边缘惩罚

本功能只接入 `g1_depth`。训练使用的网络、观测布局、AMP 数据格式、物理地形和机器人碰撞体保持原样。新增一项 `foot_edge_penalty`；足底支撑率仅作诊断，没有支撑奖励。本文列出的训练命令需由使用者手动执行。

## 算法与论文对应

参考 Instinct 论文第 III-C 节、算法 1、公式 (3)，以及 [InstinctLab 官方实现](https://github.com/project-instinct/InstinctLab)：

1. 从实际地形三角网格的副本提取相邻面二面角超过阈值的锐边，不区分前后、水平与竖直、凸边与凹边。
2. 使用官方 Greedyconcat 思路拼接碎段，用局部固定随机数生成器，避免消耗策略和环境的全局随机数。
3. 沿边构建有限圆柱；轴向投影必须位于线段范围内，没有 capsule 的球形端帽。
4. 每脚使用 100 个规则网格点。点进入圆柱后的径向穿入深度为 `radius - radial_distance`。同一点进入多个圆柱时取最大穿入深度，再对两脚全部点求和。

原始代价为：

\[
c_{\mathrm{edge}}=\sum_i d_i(\|v_i\|+10^{-3}),\qquad
r_{\mathrm{edge}}=w_{\mathrm{edge}}\,c_{\mathrm{edge}}\,dt.
\]

奖励函数只返回非负的原始代价，RewardManager 统一乘负权重及控制周期。没有接触门控、逐点平均或速度方向门控，因此摆腿中的擦边也会产生惩罚。静止穿入仍保留论文的微小代价。

**论文与源码不同的地方：**论文公式使用 `1e-3`，官方 [volume_points.py](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/envs/mdp/rewards/volume_points.py) 使用 `1e-6`。本实现默认采用论文的 `1e-3`。贪心拟合最大偏离采用官方源码实际使用的 `0.05 m`，不采用其未生效的另一个配置字段。

原始来源：

- [锐边检测与 Greedyconcat](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder.py)
- [有限圆柱查询](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/utils/warp/cylinder.py)
- [官方 parkour 参数](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)

来源和移植差异保留在源码及输出的 `foot_edge_contract.json` 中。移植的官方几何算法保留其 CC BY-NC 4.0 归属说明；不能以仓库其他文件的许可替代该部分来源许可。上游 commit SHA 无法获得时，元数据明确标记缺失，同时记录本地移植源码哈希，不以 `main` 冒充固定 commit。

## G1 23dof 坐标适配

左右脚依次绑定 `left_ankle_roll_link`、`right_ankle_roll_link`。点坐标定义在各自的 link 局部坐标系中，采用 `10 × 5 × 2 = 100` 个点：

| 局部轴 | 范围（m） |
|---|---|
| x | −0.06～0.14 |
| y | −0.03～0.03 |
| z | −0.03～0 |

这是适配当前脚 STL 的虚拟体积近似；不替换 URDF 中四个球形碰撞体，不使用官方带鞋版本的点坐标。边缘圆柱同样只是奖励几何，不参与物理碰撞。

点的位置使用 link 原点，速度使用 COM 参考，避免参考点混用：

\[
p_i^W=p_{\mathrm{link}}^W+R_{\mathrm{link}}^W p_i^{\mathrm{local}},\qquad
v_i^W=v_{\mathrm{COM}}^W+\omega^W\times(p_i^W-p_{\mathrm{COM}}^W).
\]

同一地形上的全部环境共享静态边缘索引；重置机器人和地形课程调整环境原点不会重建索引。机器人状态和查询结果按当前控制步缓存，重置后失效。

## 配置

在 `legged_lab/envs/g1/g1_depth_cfg.py` 的 `G1DEPTHENVCFG.foot_edge` 集中配置。奖励权重位于 `Reward.foot_edge_penalty.weight`。默认值为：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enabled` | `True` | 边缘奖励总开关 |
| `geometry.angle_threshold_deg` | `70.0` | 锐边二面角阈值，度 |
| `geometry.cylinder_radius` | `0.05` | 边缘有限圆柱半径，m |
| `geometry.adjacent_angle_threshold_deg` | `30.0` | 贪心拼接方向阈值，度 |
| `geometry.fit_distance_threshold` | `0.05` | 折线到拟合直线的最大偏离，m |
| `geometry.min_points` | `2` | 最少拼接点数 |
| `geometry.num_grid_cells` | `64**3` | Warp 共享空间索引参数 |
| `geometry.seed` | `42` | 几何拼接专用随机种子 |
| `x_range` / `y_range` / `z_range` | 见上表 | G1 脚局部点范围，m |
| `grid_shape` | `(10, 5, 2)` | 每脚规则网格点数 |
| `velocity_epsilon` | `1e-3` | 论文速度常数 |
| `tolerance` | `0.0` | 仅深度大于该值的点参与求和；不从深度中扣除 |
| `debug_vis` | `False` | 显示第 0 环境的脚体积点及附近圆柱 |
| `support_diagnostics` | `True` | 记录足底支撑指标，不改变奖励 |
| `support_distance_threshold` | `0.04` | 足底逐射线距离阈值，m |
| `support_contact_threshold` | `1.0` | 负载状态的正竖直接触力阈值，N |
| `Reward.foot_edge_penalty.weight` | `-4.0` | 原始代价由 RewardManager 转成惩罚 |

关闭方式：把 `foot_edge.enabled=False`，同时保持 `debug_vis=False`。这样不构建静态边缘索引。将奖励权重设为 `0.0` 也可以禁用该奖励。支撑诊断由独立的 `support_diagnostics` 控制。

圆柱半径和体积点布局共同决定安全间距。半径、点数或网格范围变化都会改变代价尺度，配置及元数据应随实验保存。默认 `-4.0` 是官方参数的起点，不是已经证明适合本项目所有地形的最优值。

## 日志与可视化

- `FootEdge/`：原始代价、加权奖励、穿入点比例、最大/平均深度和点速度；包括左右脚分别的统计。
- `FootSupport/`：射线有效比例、接触比例、竖直负载状态，以及负载时的支撑和未支撑比例。

支撑诊断使用每根射线的实际世界坐标起点、方向及命中点；无效命中计为未支撑。它是射线得到的支撑代理，不是接触斑面积测量，也不加入奖励。

奖励及训练日志在 reset 前采集。`foot_edge_contract.json` 保存生效几何参数、点布局、来源及哈希；`foot_edge_geometry.npz` 保存实际边和局部点坐标。训练入口在参数目录保存这些额外元数据。

验证脚本还会输出离线 `foot_edge_overlay.html`，展示真实脚 STL、采样点、地形和附近有限圆柱。文件不依赖外部脚本，可拖动旋转、滚轮缩放，并切换各图层。该图用于核对坐标和几何，不是运动性能结果。

## 必要验证（不训练）

请使用安装了当前 IsaacLab、Warp 和 GPU 运行时的 Python：

```bash
python legged_lab/scripts/check_g1_depth_edge_reward.py --headless --device cuda:0
```

另启动一次独立仿真，检查关闭奖励后的兼容性：

```bash
python legged_lab/scripts/check_g1_depth_edge_reward.py --headless --device cuda:0 --disabled
```

有窗口检查时去掉 `--headless`，按需增加 `--debug_vis`。脚本固定 2 个环境、32 步零动作，不读取 checkpoint，不创建训练器或优化器，也不执行策略更新。启用模式输出目录默认是 `work/g1_depth_edge_reward_validation`；关闭模式默认输出到其 `disabled` 子目录，避免覆盖启用结果。`--output_dir` 可以指定输出目录。

检查包括：配置 `to_dict()` / `from_dict()` 往返和校验、保存实际 `env_cfg.yaml`；actor 1932 维、critic 1037 维、动作 23 维不变；每脚 100 点；RewardManager 正确应用一次权重和一次 `dt`；禁用项后的奖励差值；真实地形圆柱查询；COM 偏移下的纯旋转速度；重置缓存和几何不重建；可视化可开关。关闭模式使用原生 TerrainImporter、没有边缘几何和边缘奖励项，32 步中保持原始边缘代价为零，足底支撑诊断仍然可用；报告明确标记几何和叠加图为 `not_constructed`。

为减少启动成本，脚本使用实际训练地形配置的副本，缩成 `2 × 3` 块并禁用课程、随机化和观测噪声。全部候选地形类型仍保留，但六块随机采样地形不保证覆盖每一种类型；独立几何与数学单测负责边界情况。报告 `isaac_validation.json` 会明确这种限制。

脚本仅验证接入和数值，不评价是否更少摔倒，也不将 32 步零动作结果当成训练收益。几何、奖励公式、坐标和支撑诊断的 CPU/Warp 单测另外运行。

## 手动开始训练

确认配置和验证结果后，由你自行执行现有训练命令：

```bash
python legged_lab/scripts/train.py --task g1_depth --num_envs 128 --headless
```

本次实现不自动启动此命令，不安排训练对照任务。原有网络和 checkpoint 参数形状不因边缘奖励而变化；继续旧训练仍需自行决定是否加载其优化器及已有迭代数。

## 相关文件

- `legged_lab/envs/g1/g1_depth_cfg.py`：中文配置与唯一新增奖励项。
- `legged_lab/envs/g1/g1_depth_env.py`：地形扩展、monitor 生命周期及 reset 前诊断。
- `legged_lab/terrains/foot_edge_terrain.py`：任务局部 TerrainImporter，不修改安装的 IsaacLab。
- `legged_lab/perception/foot_edge_geometry.py`、`foot_edge_warp.py`：锐边、贪心拼接及有限圆柱查询。
- `legged_lab/perception/foot_edge.py`：G1 点布局、COM 参考运动学、代价及支撑诊断数学。
- `legged_lab/mdp/foot_edge.py`：RewardManager 接口、缓存、诊断、元数据与可视化。
- `legged_lab/scripts/train.py`：仅增加额外元数据的保存。
- `legged_lab/scripts/check_g1_depth_edge_reward.py`：不学习的集成检查与离线叠加图。
- `legged_lab/envs/g1/G1_DEPTH_FOOT_EDGE.md`：本说明。
- `rsl_rl/tests/test_foot_edge_geometry.py`、`test_foot_edge_reward.py`：只计算几何与代价的单元测试，不包含策略训练。
- `legged_lab/perception/FOOT_EDGE_UPSTREAM_LICENSE.txt`：仅适用于对应派生模块的上游许可。

原有 `g1_renet`、selector、部署代码、物理碰撞模型及被注释的支撑奖励不在本次修改范围内。
