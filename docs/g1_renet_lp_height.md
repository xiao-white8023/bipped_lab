# G1_RENet：LP 高程输入

本实现提供四种互斥的 LP 观测契约。`mid360` 是原始 MID-360 动态射线与
0.8 秒滚动图基线；三个 PGTT 模式代表部署端建图后的规则高程查询，不是物理
MID-360 扫描模型。VP 始终保持两帧深度输入和原有辅助任务。

## 模式与启动

`renet.lp_height_source` 可取：

- `mid360`：旧 11×17 单帧、完整 187 点 LP 重建，兼容原有匹配模型。
- `pgtt_full`（默认）：机器人四周 11×17 单帧、legacy LP 网络、完整 187 点重建。
- `pgtt_front`：前方 11×9 单帧、前方 99 点 LP 重建。
- `pgtt_full_atten`：与 `pgtt_full` 共用 11×17 地图，但使用独立的 AME 式
  16 头点级交叉注意力 LP。

旧名称 `pgtt` 对应已经删除的 18×35×2 梯形契约，现在会给出迁移错误，不能
隐式映射到新模式。

```bash
python legged_lab/scripts/train.py --task g1_renet --headless --num_envs 128 --lp_height_source pgtt_full
python legged_lab/scripts/train.py --task g1_renet --headless --num_envs 128 --lp_height_source pgtt_full_atten
python legged_lab/scripts/train.py --task g1_renet --headless --num_envs 128 --lp_height_source pgtt_front
python legged_lab/scripts/train.py --task g1_renet --headless --num_envs 128 --lp_height_source mid360
```

该开关选择完整传感器、观测、网络和辅助输出契约。四种契约不能交叉加载；无
契约元数据的旧 checkpoint 只允许在 `mid360` 下尝试严格形状加载。

## PGTT 高程图

PGTT 使用普通 IsaacLab `RayCaster + GridPatternCfg`，挂载在 `torso_link`，
`ray_alignment="yaw"`。射线从 torso 上方 20 m 世界竖直向下，仅查询静态
`/World/ground`，10 Hz 更新，最大距离 40 m。它不创建 GroupedRayCaster，
也不模拟自身遮挡、真实 MID-360 扫描顺序或近场光学盲区。

| 模式 | x 范围 | y 范围 | 形状/射线数 | Actor 原始维度 |
|---|---:|---:|---:|---:|
| `pgtt_full` | [-0.8, 0.8] m | [-0.5, 0.5] m | 11×17 / 187 | 2120 |
| `pgtt_full_atten` | [-0.8, 0.8] m | [-0.5, 0.5] m | 11×17 / 187 | 2120 |
| `pgtt_front` | [0.0, 0.8] m | [-0.5, 0.5] m | 11×9 / 99 | 2032 |

分辨率为 0.1 m，外层按 y 递增、内层按 x 递增。前向图严格等于完整图的
`[:, :, 8:17]`，包含 x=0 中心列。有效高度为
`clip(torso_z - terrain_z - 0.5, -1, 1)`；无命中写为 `-1.1`，不使用最大
量程端点。置信度仅写入 `extras`，不进入 Actor。

网络只接收一个可能延迟的最新高程图。为了实现 0–2 个数据包延迟，传感器前端
内部最多保留三个包，但不构成神经网络的多帧输入。内部 Actor 输入仍为
`78 + 64 + 64 = 206`，输出 23 维动作。

`pgtt_full_atten` 的最后 5 帧本体历史经 MLP 压缩为一个 64 维 Query；单帧
高程图经两层卷积形成 187 个包含 `(x,y,height)` 的 64 维 token，再由 16 头
Cross-Attention 得到 LP latent。该路线没有 LP GRU，也没有后注意力 LP
encoder；其它三条路线仍使用原有单 LiDAR token、LP encoder 和 LP GRU。

## 噪声与辅助学习

三个 PGTT 模式共享建图残差噪声：1–3 cm 高度标准差、±2 cm episode 高度
偏置、±2 cm xy 偏置、±1° yaw 偏置、0–5% 栅格 dropout、持续 1–3 包的
局部块缺失、0–2 包延迟，以及 0.1 m 地图量化。相同数据包重复读取不会重新
采样噪声。旧的 0.7–0.95 m 大面积径向盲区随机化已删除。

LP 高程目标固定裁剪到 [-1,1]，不做 minibatch 标准化：`pgtt_full` 重建完整
187 点，`pgtt_front` 重建 critic 图中 x≥0 的 99 点。VP 仍重建前方 99 点并
执行原有 minibatch 标准化，同时保留速度和足端高度监督。LP/VP 每 20 次迭代
轮换控制分支，双方辅助损失在每个 minibatch 都更新。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  rsl_rl/tests/test_lp_height.py rsl_rl/tests/test_renet_lp_vp_training.py
python legged_lab/scripts/check_renet_lp.py --headless --source pgtt_full --terrain flat --num_envs 1 --steps 100 --iterations 1
python legged_lab/scripts/check_renet_lp.py --headless --source pgtt_full_atten --terrain flat --num_envs 1 --steps 100 --iterations 1
python legged_lab/scripts/check_renet_lp.py --headless --source pgtt_front --terrain stairs --num_envs 1 --steps 100
```

诊断脚本支持 flat/stairs/gap/bottomless/mixed、关闭 PGTT 噪声及匹配 checkpoint。

## MuJoCo sim2sim

`sim2sim_renet.py` 提供两个用途不同的 PGTT 后端：

- `mujoco_ray`（默认）：调用 MuJoCo 原生 `mujoco.mj_ray`，按相同坐标逐格发射
  11×17/11×9 世界竖直向下射线，只查询地形。这与 IsaacLab PGTT 训练观测
  等价，但网格范围、分辨率、形状和展平顺序均在 `sim2sim_renet.py` 中独立
  定义；运行时不导入训练侧 `lp_height` 网格工具，也不创建或调用 IsaacLab
  RayCaster。
- `mid360`：使用物理 MID-360 扫描和 0.8 秒滚动建图，再输出完整 11×17 地图
  或其前半部分 11×9。它用于验证部署侧
  `MID-360 -> 滚动地图 -> PGTT 策略输入` 链路，并不会伪造近场盲区。
  MuJoCo 侧的时变 360° 扫描方向由脚本内的确定性准均匀模型生成，保持点频、
  包周期和垂直视场，但不读取 IsaacLab 的记录角度文件；因此它是物理扫描链路
  近似，不是 Livox 官方逐点扫描序列的复刻。

`lp_height_source=mid360` 始终使用物理 MID-360 后端并保留 G1 自身回波；三个
PGTT 模式过滤机器人自身几何体，以匹配训练时只查询 `/World/ground` 的定义。
旧的 `--pgtt-map-backend raycaster` 仍可使用，但只作为 `mujoco_ray` 的弃用别名。

当前 `play_new_renet.py` 导出的 `policy.pt` 内嵌 LP 模式和观测契约，sim2sim
默认自动识别：

```bash
python legged_lab/scripts/sim2sim_renet.py \
  --policy logs/g1_renet/<run>/exported/policy.pt \
  --model legged_lab/assets/g1/g1_23dof/mix_scene.xml \
  --estimator lp
```

要显式测试物理雷达建图链路：

```bash
python legged_lab/scripts/sim2sim_renet.py \
  --policy logs/g1_renet/<run>/exported/policy.pt \
  --model legged_lab/assets/g1/g1_23dof/mix_scene.xml \
  --pgtt-map-backend mid360 \
  --estimator vp
```

物理 MID-360 在机器人静止启动时无法看到 PGTT 的近场区域，有效率可能只有
约 1%–5%；这与训练时接近全有效的竖直图不是同一输入分布。该后端下应先使用
VP 建图，不能把未观测栅格伪造成平地后直接启动 LP。

旧模型没有内嵌元数据时默认按 `mid360` 解释；旧 PGTT 导出必须显式传入
`--lp-height-source pgtt_full` 或 `--lp-height-source pgtt_front`。新注意力路线
必须包含编码器元数据。`mid360` 与
`pgtt_full` 虽然同为 2120 维，但语义不同；新导出的模型会在仿真开始前拒绝
错误的交叉加载。

历史 `/home/cjh/model_25000.pt` 仅可由 `pgtt_full_atten` 触发迁移：加载器会同时
核对旧 `pgtt_full` 观测契约、AME 编码器契约、全部权重形状及恢复训练时的
Adam 状态形状；后续保存会写成原生 `pgtt_full_atten` 契约。

```bash
python legged_lab/scripts/train.py --task g1_renet --headless --num_envs 128 \
  --lp_height_source pgtt_full_atten --checkpoint_path /home/cjh/model_25000.pt
```

本阶段仍不实现实机 Point-LIO/elevation mapping 或自动 LP/VP 故障选择器。
PGTT 训练查询本身不解决实机雷达近场盲区；实机必须由建图、历史观测和运行
安全逻辑提供相同契约。
