# G1 RENet 选择器代码审查（2026-10-08）

结论：LP/VP 分别预测未来失败风险的方向合理，现有实现适合作为继续开发的基础。但不建议按当前默认配置直接开始 1,048,576 样本采集和正式训练。首先需要修正冻结策略身份、标签、时钟、训练数值和数据读取；真机自动切换还需要动作隔离、兜底接口与完整部署接入。

本次只审查并添加验证脚本和报告，没有修改选择器、RENet 或部署实现，没有启动 Isaac Sim、ROS 或机器人控制。当前目录不存在 `outputs/g1_renet_selector`，因此下面没有声称已有训练效果或真实闭环性能。机载接口对照沿用此前训练到部署审计的已确认结果，并检查本地部署源代码。

## 1. 当前实现实际做了什么

链路：冻结 RENet → 同一候选状态的 LP/VP 分支回放 → 四个失败标签 → 监督训练 → 校准 → 10 Hz 状态机 → 50 Hz 动作渐变。

每帧 283 维特征由以下部分组成：

| 特征 | 维度 |
|---|---:|
| 当前 proprio，含命令与上一执行动作 | 78 |
| LP、VP latent | 64 + 64 |
| LP、VP 速度估计 | 3 + 3 |
| LP、VP 候选动作 | 23 + 23 |
| 两路健康统计 | 8 + 8 |
| 两分支差异统计 | 7 |
| 当前 mask、距上次切换时间 | 2 |

网络使用 5 个 10 Hz 特征帧，MLP `283→256→128`、GRU 128、head `128→64→4`，共 213,956 个参数。输出顺序是 `LP_0.4s, LP_1.0s, VP_0.4s, VP_1.0s`。初始重复帧以外，五帧首尾跨度为 0.4 秒。

有价值的实现已经存在：冻结策略与 PPO 路径分离；LP/VP 二值 mask；显式零 latent 的双分支适配器；短期失败属于长期失败的单调输出；故障状态和历史快照；每支路重复两次并拒绝离散标签不一致样本；按生成地形实例分组划分；独立 7–9 级 OOD 测试集；传感器硬门限、驻留与切换确认；动作输出渐变。

这些设计值得保留。优先解决下面的问题，而不是增加网络规模。

## 2. 必须在正式采集／训练前修正

### 2.1 冻结策略不是当前部署的策略

位置：`legged_lab/selector/config.py:143`，`legged_lab/scripts/collect_selector_data.py:29`、`:190`、`:223`，`legged_lab/selector/policy_adapter.py:124`。

默认 checkpoint 指向 `/data/Tienkung_lab/logs/g1_renet/2026-09-10_19-25-08/model_63999.pt`，该路径在当前机器不存在。此前已确认你部署的 `2026-09-04_14-11-25/exported/policy.pt` 对应 `model_60000.pt`。

仅替换路径还不够。采集脚本从当前 `g1_renet` 配置实例化完整 runner，当前 VP 使用 5 帧，冻结部署策略使用 10 帧；当前 critic 输入 612，部署源 checkpoint 为 1037。runner 允许继续训练时迁移历史窗口和辅助头，这与“保持冻结策略行为一致”的要求不同。适配器的 parity 是对加载、迁移后的 runner 做比较，不能证明与机器人当前策略相同。原始 checkpoint 的再次 metadata 校验还可能因新增头和形状变化直接失败。

验证：重建并严格加载部署源权重，只把 VP 历史由 10 改成 5。在四个合成输入上，LP 动作差为 0，VP 动作最大差约 1.712，VP latent 最大差约 1.434。这是语义变化的演示，不是真实轨迹误差估计。

建议：采集入口直接使用精确重建的冻结 actor／现有 `renet_dual_branch.pt`，不要依赖当前 PPO runner 的迁移行为。现有 `legged_lab/scripts/export_renet_dual_branch.py` 已有精确重建和参考权重校验基础。采集、训练、导出、部署都绑定 checkpoint、参考 policy 和双分支 artifact 的 SHA-256，以及 VP/LP 历史窗口、观测布局和归一化契约。parity 应比较部署参考 artifact，包括 latent 和速度估计，不能只比较适配器与同一新 runner。

### 2.2 离地高度标签会把前方障碍当成脚下地面

位置：`legged_lab/selector/labels.py:77`。

`_local_clearance()` 用整张扫描的 `amax(z)` 作为地面，再执行 `root_z - max_ground < 0.45`。这不是机器人当前位置的支撑面高度。

验证：机身 z=0.8、脚下地面 z=0、前方障碍 z=0.4、姿态正常且无终止，计算 clearance≈0.4，并被标成失败。台阶、斜坡等地形可能因此产生错误标签，甚至让 LP/VP 同时被判失败。

建议：定义清楚失败的物理含义，用机身投影附近的有效地面或支撑足附近的参考面；不要直接取整个视野最高点。对扫描缺失明确处理，当前全部无有效命中时 clearance 变为 `inf`，低高度检查会被绕过。倾角和高度宜验证持续时间，避免把单帧噪声当成失败。

### 2.3 标签读取的是自动 reset 后的身体状态

位置：`legged_lab/selector/collector.py:309`、`:330`，`legged_lab/envs/g1/RENet_env.py:1265`，`legged_lab/selector/labels.py:96`。

`env.step()` 内部先 reset，再返回；accumulator 随后读取 `robot.data`。非 timeout 的 torso 终止仍能由 `dones/time_outs` 捕获，所以不能说所有跌倒标签都丢失。但该步的倾角、clearance、速度误差等度量可能已经属于新 episode；timeout 处的真实姿态也不能如此恢复。

建议：在 reset 前记录 selector 所需的终止状态和逐步安全指标，或使用避免自动 reset 的采集封装。应同时规定 timeout 的处理：排除未来不足 1 秒的候选，或明确记录删失与有效 horizon，不能把短观察当成完整长期安全证据。

### 2.4 传感器年龄使用了不同时间基准

位置：`legged_lab/selector/collector.py:217`，`legged_lab/sensors/grouped_ray_caster/grouped_ray_caster.py:60`，本机 IsaacLab `sensor_base.py:169`，`legged_lab/selector/features.py:81`。

collector 的 `now` 是全局模拟时间，camera 的 `current_timestamps` 是每环境传感器时钟，sensor reset 后归零。`frame_age=now-sensor_timestamp` 因此会把正常的新图像当成很旧的数据。terrain prepass 后统一 reset 会放大这个问题：不是只有采集期间第一次终止才可能出现。

验证：全局时间 100.1 秒、reset 后相机时间 0.1 秒，报告 age=100 秒。实时部署的门限为 0.3 秒，因此训练出来的年龄特征与真机正常帧年龄严重不同。

建议：使用同一全局时钟的实际采集时间，或保存每环境 sensor epoch offset。还要区分 sensor 的当前时钟与最近完成采集的时间，不能用前者替代包时间。时钟偏移也必须纳入快照。

### 2.5 实时接口将绝对时间压成 float32

位置：`legged_lab/selector/deployment.py:131`，`legged_lab/selector/features.py:81`，`legged_lab/selector/runtime.py:86`。

若 ROS epoch 秒数直接传入，float32 在约 1.7e9 秒处无法分辨 0.1 秒更新。验证中，相隔 0.2 秒的两次 runtime 调用只执行了一次风险推理；实际陈旧 1 秒的包报告 age=0。

建议：使用 float64 或整数纳秒计算时间差，再转换年龄／间隔等小量特征为 float32；另一种明确方案是统一减去会话时间原点。应在接口契约规定时间基准，验证倒退和重连。对两帧 depth 分别保留真实采集时间，至少健康年龄不能仅由任意新 header 证明两帧都新鲜。

### 2.6 GPU 混合精度损失可产生 NaN

位置：`legged_lab/selector/model.py:155`，`legged_lab/scripts/train_selector.py:167`。

当前在 autocast 中对概率做手工 BCE。float16 的 `1-1e-6` 会舍入成 1；饱和概率仍可能进入 `log1p(-1)`，随后产生 `0×inf`。GradScaler 不会把这个损失表达式变成数值稳定的表达式。

验证：直接 float16 输入和本机 GPU 的实际 autocast 网络都复现 NaN loss。GPU 演示将最后层 bias 设置为 20，以覆盖饱和情况；不意味着每次普通训练必然立即 NaN。

建议：至少将概率转换为 float32 后再 clamp、计算 loss，并检查非有限 loss／梯度。更稳妥的是基于 logits 或稳定 log-probability 计算单调联合概率的损失。单纯调用 `.float()` 可解决这里的 FP16 clamp 问题，但还应验证极端 logits 的梯度行为。

### 2.7 随机采样配合一个 shard 缓存导致反复完整加载

位置：`legged_lab/selector/dataset.py:269`，`legged_lab/scripts/train_selector.py:69`。

reader 每次访问其他 shard 都 `torch.load` 整个文件，而训练 DataLoader 对全局样本执行 shuffle。4 个 worker 各自只有一个 shard 缓存，不能避免跨 shard 随机跳转。

验证：4 个 shard、40 行，顺序访问完整加载 4 次，固定种子的随机访问加载 25 次。默认每 shard 的 features 数据约 22.1 MiB；百万样本下这种访问方式会放大反序列化与内存复制开销。这个量是 tensor payload 大小，不能直接等同实际物理磁盘读流量，因为操作系统会缓存文件。

建议：按 shard 打乱顺序、在 shard 内打乱行并连续形成 batch；使用分块 sampler／IterableDataset、内存映射数组或在 RAM 足够时预加载。先测实际 samples/s 和内存，再设置 worker 与 batch 大小。

## 3. 会影响风险和切换有效性的设计问题

### 3.1 加权 BCE 输出不能直接解释为真实失败概率

位置：`legged_lab/scripts/train_selector.py:103`、`:204`，`legged_lab/selector/model.py:182`。

当前 positive weight 为负正样本数量比，再只拟合温度。对真实失败率 p、正类权重 w，常量预测的加权 BCE 最优解是 `q=wp/(1-p+wp)`，不是 p。温度只有缩放，没有纠正先验偏移的截距。

验证：真实失败率 1%、w=99 时，最优常量输出为 0.5；无论温度怎么调整，`sigmoid(logit(0.5)/T)` 都是 0.5。当前 fitter 在这个反例中保持温度 1、概率 0.5。

建议：选择不加权的概率损失，或明确进行先验／logit bias 修正，再在独立 calibration 集拟合带截距的校准。四个输出分别校准后，代码又 `max(short,long)`，最终概率与拟合目标并不完全一致；应对最终单调输出进行联合校准和评估。early stopping、温度拟合、种子选择共享 validation，不是训练样本泄漏，但应尽量拆出 calibration 或交叉拟合，test/OOD 保持独立。

### 3.2 回放执行方式与运行时切换不同

位置：`legged_lab/selector/collector.py:293`、`:571`，`legged_lab/selector/runtime.py:231`。

标签回放立即执行新分支的完整动作，没有运行时普通 5 帧、紧急 2 帧动作渐变。采集行为又全体环境按 batch 每 1 秒统一交替 LP/VP。切换时间在完整 1 秒 rollout 结束后才更新，而 rollout 中特征已经用新 mask 和旧切换时间计算 dwell。因此样本最后的 dwell 与实际行为切换时刻不一致。

建议：让分支标签对应“从当前执行状态切换到 m 并保持”的实际控制过程，复用 ActionBlender，保存最后实际执行动作、blend 状态、当前模式和真实切换起点。行为采集分开设计，覆盖持续 LP、持续 VP、不同驻留时间和过渡步，而不是只在一秒块边界选候选。需要时补充闭环选择器访问到的状态，减小行为分布偏移。

### 3.3 初始 RNG 一致不保证每环境未来故障一致

位置：`legged_lab/selector/faults.py:188`、随机 dropout apply；`legged_lab/selector/collector.py:311`。

fault manager 对整个 batch 共用一个 generator，某环境 reset 会消耗这个 generator 生成新参数。LP/VP 轨迹的 reset 集合不同，因而其他未 reset 环境的未来随机 dropout 也可能不同。重复同一分支的一致性只能证明该分支能重复，不能证明两个分支的外生扰动相同。

验证脚本包含：同一故障快照重放完全一致；只 reset 环境 1 后，环境 0 下一次随机 dropout 图发生变化。

建议：按环境独立随机流，或按候选／环境／时间预生成故障轨迹。同步审计 push、动作噪声和命令重采样等外生输入，保证 LP/VP 的比较条件符合标签定义。保存按 terrain、fault、正负标签分层的 replay 拒绝率，防止只留下容易重放的状态。

### 3.4 当前标签只监督安全失败，无法评价走得好不好

位置：`legged_lab/selector/labels.py:105`，`legged_lab/selector/collector.py:573`。

tracking error 被记录到 future_metrics，但没有进入失败标签或训练目标。如果两个分支都不跌倒，当前训练没有直接监督哪一个更能跟踪速度／转向命令；同风险区间又偏向 VP。因此不要期待仅凭这个目标解决某一路“能站住但不愿走”的行为。

建议：保留安全风险独立输出，在两路安全且可用时，用额外的跟踪代价／进展预测作选择依据。可以用已经记录的 future_metrics 做初版辅助目标，但最大瞬时误差不是平均表现；应补充有效时间内平均误差、命令方向的进展和持续停滞。避免把所有任务误差混成与跌倒等价的 failure。

### 3.5 当前离线指标没有验证实际状态机

位置：`legged_lab/selector/metrics.py:67`，`legged_lab/scripts/eval_selector.py:78`。

评估直接执行 `LP_risk+0.05 < VP_risk` 的逐样本判定，实际 runtime 使用 0.10 switch margin、3 次确认、1 秒驻留、传感器硬判定和动作渐变。这两套选择行为不同；AUROC、ECE 和离线 dangerous_branch_rate 通过，不能据此宣称闭环切换安全有效。

此外，`auroc/min` 忽略单类输出的 NaN，`one_safe` 为空时 dangerous rate 返回 0。全局门限不保证每一路和每类故障有足够可判定样本。

建议：补充长 episode 的闭环 LP 固定、VP 固定、健康门限、完整 selector 四组对照；按地形／故障／命令分层报告跌倒、跟踪误差、每分钟切换数、切换后短期失败、故障响应延迟及 both_unreliable 持续时间。独立 OOD 使用相同完整 runtime。门限应在 calibration／validation 上确定，并保留每个指标的样本数、失败数和置信区间。

3 次 10 Hz 确认从首次命中到确认通常占 0.2 秒，故障恰在上个 tick 后开始时还可能等待近 0.1 秒；普通 blend 再占 0.1 秒。1 秒驻留还能进一步阻止风险切换。是否能及时应对 0.4 秒风险需要用闭环结果验证。

## 4. 自动接管真机前必须补齐

### 4.1 未选分支 NaN 污染了已选动作

位置：`legged_lab/selector/runtime.py:225`，`legged_lab/selector/deployment.py:202`。

适配器已经用显式零 latent 隔离两路 actor 输入，但 runtime 和 shadow 又执行 `mask*LP+(1-mask)*VP`。`0*NaN` 仍是 NaN，隔离因此再次失效。

验证：LP 的 23 个动作全 NaN、VP 动作为 1、选择 VP，ActionBlender 输出 23 个 NaN。

建议：二值分支选择使用索引／`torch.where`，不能用乘零混合候选。选中动作、proprio、目标关节必须独立检查有限性；有故障时 blend 的起始动作也必须是有效的已执行动作。模型异常应有明确控制器事件和退出路径，不能只有 Python 异常。

### 4.2 both_unreliable 目前只是标记，动作仍继续输出

位置：`legged_lab/selector/runtime.py:96`、`:176`，`legged_lab/selector/deployment.py:215`。

两路风险都高或两路不可用时，状态机通常保留当前 mask 并设置标记，runtime 仍输出该分支动作。验证中两路风险均为 0.9，`both_unreliable=true`，仍执行 VP 候选动作。这是当前语义，不能当作已经实现的 SAFE 状态。

建议：定义与 C++ 控制器的请求／确认接口和明确的停止、退出或独立 SAFE 控制策略。选择器初版可以先只做风险记录和 LP/VP 建议，不自动接管两路均坏场景。当前 `g1_recovery` 是恢复任务，不能因名字或注册存在就当成已经验证的行走停止／站稳策略。是否使用它需要独立训练目标、起始状态分布和控制交接验证。

### 4.3 现有 TensorRT 后端没有选择器需要的输出

位置：`unitree_deploy_ws/src/g1_control/g1_control/tensorrt_runner.py:43`，`policy_node.py:354`。

机载当前配置使用的 TensorRT engine 只返回 23 维 actions，两次推理取得 LP/VP 动作。283 维特征还需要两路 latent 和速度估计。已有 TorchScript 双分支 artifact 可输出六个 tensor，但本地节点即使使用 TorchScript 也只取 actions，且没有接入 selector。

建议：选择并实现一个统一方案：导出六输出的 TensorRT 双分支 engine；或验证机载 TorchScript 全链路时延后接入；或重新设计仅使用机载现有可观测量的选择器并重新采集训练。不能临时把缺少的 latent、velocity 填零来运行一个以真实值训练的模型。

还要绑定 selector 与底层 policy artifact 身份，并验证从最终已执行 action 写回 previous_action 的闭环一致性。当前 OnlineSelectorRuntime 的外部 current_mask、内部状态机 mask、手动 override 之间也需要同步规则；尤其 shadow 模式内部建议模式变化，实际执行模式未变化，之后的驻留和确认逻辑应明确对应哪个状态。

### 4.4 健康特征与机载 preprocessing 不一致

位置：`legged_lab/selector/deployment.py:153`，`legged_lab/selector/collector.py:145`，`unitree_deploy_ws/src/g1_depth/g1_depth/depth_policy.py:61`。

selector 只用归一化 depth 是否在 `[0,1]` 判断 valid。实机原始无效深度经过填充、clip、归一化或滤波后可能仍在这个区间，因此“数值合法比例”不能代替原始有效像素比例。预处理器已有填充前 DepthFrameStats 和 source health，应该接入。

此外，采集侧健康 tracker 每 10 Hz 更新，而 runtime 每 50 Hz 更新。当前 tracker 在重复包时把 temporal_l1 置零、frozen_ratio 置一；若新包到达时间与 selector tick 不对齐，tick 读到的统计可能丢掉前一个控制步刚计算的新包差异。建议按包更新并保留变化统计，另行计算包龄和持续未更新时长；训练和部署调用节奏保持一致。

真实 LP 是持续高程地图再按当前位姿重采样，采集侧冻结整个 LP tensor 模拟的是其中一种故障，不能覆盖“激光／定位停止，但重采样发布仍继续”的情形。应分别保留原始采集、地图更新、输出发布的时间与有效信息，并按机载处理链构造故障。

ShadowLogWriter 当前记录风险、mask、可用性和附加 robot_state，并未自动记录原始 SelectorInput、283 维特征、两路候选动作和 uncertainty。要用于复盘或再训练，需要明确补齐数据内容、版本、时间和来源。

## 5. 建议实施顺序

1. 固定部署策略身份和完整特征契约；精确 actor 直接用于采集，验证与部署参考的动作／latent／velocity 一致。
2. 修复时钟、clearance、reset 前指标、timeout 处理、真实切换时间与动作执行过程；补上每环境独立故障随机流。
3. 修复 AMP 损失和 shard 读取；先做 64–128 环境的小规模采集，检查正负标签、LP-only/VP-only 安全样本、故障覆盖、健康量和 replay 拒绝分布。环境数量本身是吞吐选择，不应硬性等同正确性。当前机器 16 GiB GPU，4096 相机环境的可行性仍需测量。
4. 在标签有效后扩充数据量，分开验证与校准，训练风险头；需要任务性能选择时增加跟踪／进展辅助目标。网络维持当前大小即可。
5. 用完整 runtime 做闭环仿真和独立 OOD，对驻留、确认、风险门限和动作渐变做校准。
6. 补齐机载六输出推理、健康 metadata、异常退出与控制器交接，验证时延；先记录建议与实际执行结果，再逐步验证自动切换。

完整方案不一定第一轮就需要训练第三个策略。LP/VP 风险选择器与 SAFE／recovery 控制模块有不同的训练目标和验证要求，应该先明确接口，再决定是否训练 SAFE。

## 6. 可复现证据与验证边界

脚本：`work/g1_selector_review_20261008/verify_selector.py`。

结果：`work/g1_selector_review_20261008/verification.json`。

运行命令：

```bash
/home/cjh/miniconda3/envs/env_isaaclab/bin/python -u \
  work/g1_selector_review_20261008/verify_selector.py
```

验证涵盖未选分支 NaN、both_unreliable 输出语义、clearance 标签、局部／绝对时钟、加权概率校准反例、实际 GPU autocast 损失、shard 加载次数、网络 shape／单调性、部署 checkpoint 形状、同权重 VP 历史变化、故障快照与跨环境 RNG。

没有执行全量采集、正式训练、Isaac 完整快照回放、真实闭环行走或机载 selector 推理时延测试。此前训练到部署审计见 `docs/g1_renet_training_to_robot_audit_20261008.md`。
