# G1 RENet selector：采集问题 1–6 的替换文件

基于 `xiao-white8023/bipped_lab` 的提交 `fbbcce8b36cf8d3ba2654303318b4a8b98f5d42c`，2026-10-09。
这是可覆盖到仓库的源文件包；不是已训练的 selector 权重。现有冻结 `model_60000.pt` / `policy.pt` 仍作为采集基策略。

## 修改内容

| 编号 | 修改 | 主要文件 |
|---|---|---|
| 1 | 控制步推进后在第 5、10…50 步更新特征，10 Hz 候选的最后特征、快照、标签起点一致；保存两份 float64 时间并验证相等 | `selector/collector.py`, `selector/dataset.py` |
| 2 | 健康统计按新数据包缓存，50 Hz 重复查询只更新包龄；只比较两包共同有效部分；保存填充/模糊/裁剪前的有效支持；源时间与发布包时间分开，时钟用 float64 | `selector/features.py`, `RENet_env.py`, `selector/deployment.py`, 深度预处理及消息 |
| 3 | 行为策略按环境随机选择初始分支、正常驻留 1–5 秒，每 0.1 秒采候选；接受目标模式时立即记录切换时间，跟随真实过渡动作；硬健康切换可提前发生 | `selector/collector.py`, `selector/runtime.py` |
| 4 | LP/VP 标签回放也使用同一个 ActionBlender，正常 5 步、紧急 2 步；快照恢复已执行动作和正在进行的过渡 | `selector/collector.py`, `selector/runtime.py` |
| 5 | 故障持续 0.3–5 秒、初始延迟 0–3 秒、恢复间隔 0.5–3 秒；每个 episode 可多次故障；双路可错开 0–1 秒开始且不同时间恢复；对实际通过筛选的正常/VP/LP/双故障样本按 30/25/25/20 配额采集，并记录覆盖情况 | `selector/faults.py`, `selector/config.py` |
| 6 | 故障、行为、命令、reset/push、动作延迟、本体/相机/PGTT 噪声按环境管理随机流，并在分叉时恢复；邻居 reset 不推进本环境随机状态 | `selector/randomness.py`, `selector/snapshot.py`, `lp_height.py`, `RENet_env.py` |

283 维 × 5 帧结构保留，特征**语义版本改为 2**；数据 manifest 版本为 3。训练、评估、导出拒绝旧特征契约及未完成的数据集。
回放仍保留原来的失败标签、reset 前指标捕获、timeout 截断处理和同分支重复回放过滤，没有增加任务表现奖励头。

重复故障调度引入了新的未来随机事件。若下一次故障起点或下一次随机调度落在 1 秒标签窗口内，而当前特征尚未观察到它，候选会被过滤，避免把不可观察的随机故障当成可预测风险。
这会降低采集速度；短故障和某些恢复片段的接受率尤其可能偏低，应查看 `safety_label_filter.unobservable_future_fault` 和 `coverage`。

## 安装

解压后，`files/` 内目录与仓库目录一一对应。先查看 `SHA256SUMS.json` 和 `selector_fixes.patch`，然后从仓库根目录运行：

```bash
python /解压目录/install_selector_fixes.py --repo "$PWD"
```

安装器会检查每个待替换文件是否仍与基准提交一致，全部检查通过后备份原文件再复制。已有本地修改时会停止并列出冲突文件，便于手动合并；不会覆盖 checkpoint 或其他文件。

也可以手动将 `files/legged_lab`、`files/rsl_rl/tests`、`files/unitree_deploy_ws` 中对应文件覆盖到仓库同一路径。
仅替换一个 `collector.py` 不够：新的健康、质量 metadata、随机流和数据版本是配套接口。

## 启动命令

在已安装 Isaac Lab 的原训练环境里，从仓库根目录执行。以下路径沿用原代码默认值；`REF_POLICY` 必须指向你实际用于部署的原始 `policy.pt`。

```bash
export CKPT="/data/Tienkung_lab/logs/g1_renet/model_60000.pt"
export REF_POLICY="$PWD/logs/g1_renet/2026-09-04_14-11-25/exported/policy.pt"
export SELECTOR_RUN="$PWD/outputs/g1_selector_v2_$(date +%Y%m%d_%H%M%S)"
```

先做小规模仿真采集，检查 replay、故障接受率和显存：

```bash
python legged_lab/scripts/collect_selector_data.py \
  --task g1_renet --headless --device cuda:0 --num-envs 64 \
  --seed 42 --terrain-mode main --smoke-test \
  --checkpoint-path "$CKPT" --reference-policy "$REF_POLICY" \
  --prepass-steps 2000 --prepass-burn-in 1900 \
  --target-samples 512 --skip-dual-branch-export \
  --output "$SELECTOR_RUN/smoke_dataset"
```

你的正式采集命令可以保留 `20000/19000`，加上显式的冻结策略路径及防重复导出参数：

```bash
python legged_lab/scripts/collect_selector_data.py \
  --task g1_renet --headless --device cuda:0 --num-envs 4096 \
  --seed 42 --terrain-mode main \
  --checkpoint-path "$CKPT" --reference-policy "$REF_POLICY" \
  --prepass-steps 20000 --prepass-burn-in 19000 \
  --target-samples 1048576 --skip-dual-branch-export \
  --output "$SELECTOR_RUN/dataset"
```

默认即启用六项修正。可选参数以 10 Hz 的“帧”为单位：
`--fault-min-duration-frames 3 --fault-max-duration-frames 50`、
`--fault-max-start-delay-frames 30`、
`--fault-min-recovery-frames 5 --fault-max-recovery-frames 30`、
`--fault-max-dual-stagger-frames 10`、
`--behavior-min-dwell-frames 10 --behavior-max-dwell-frames 50`。
`--unbalanced-scenarios` 仅用于诊断；正式采集保留默认配额。

采集结束查看 `dataset/manifest.json` 中：`collection_complete=true`、`coverage.accepted`、`coverage.label_positive`、`coverage.dwell_counts`、`snapshot_replay_validation`、`per_candidate_replay_filter` 和 `safety_label_filter`。
若达到候选批次上限仍未满足配额，会保存标为未完成的部分数据并报错；不要直接用部分数据训练。输出目录已存在数据时会拒绝覆盖。

训练三个种子、评估、导出：

```bash
python legged_lab/scripts/train_selector.py \
  --dataset "$SELECTOR_RUN/dataset" --output "$SELECTOR_RUN/models" \
  --device cuda:0 --num-workers 4 --batch-size 8192 --max-epochs 100 --patience 10

python legged_lab/scripts/eval_selector.py \
  --dataset "$SELECTOR_RUN/dataset" \
  --checkpoint "$SELECTOR_RUN/models/lp_vp_selector_checkpoint.pt" \
  --device cuda:0 --split test --output "$SELECTOR_RUN/test_report.json"

python legged_lab/scripts/export_selector.py \
  --checkpoint "$SELECTOR_RUN/models/lp_vp_selector_checkpoint.pt" \
  --output-dir "$SELECTOR_RUN/exported"

python legged_lab/scripts/export_renet_dual_branch.py \
  --checkpoint "$CKPT" --reference-policy "$REF_POLICY" \
  --output "$SELECTOR_RUN/exported/renet_dual_branch.pt"
```

评估未达到原质量门槛时返回退出码 2，并保存报告；导出成功不等于已经满足部署指标。

OOD 采集复用正式地形统计，避免重写同一导出文件：

```bash
python legged_lab/scripts/collect_selector_data.py \
  --task g1_renet --headless --device cuda:0 --num-envs 4096 \
  --seed 43 --terrain-mode ood \
  --checkpoint-path "$CKPT" --reference-policy "$REF_POLICY" \
  --terrain-histogram "$SELECTOR_RUN/terrain_distribution.json" \
  --target-samples 131072 --skip-dual-branch-export \
  --output "$SELECTOR_RUN/dataset_ood_7_9"

python legged_lab/scripts/eval_selector.py \
  --dataset "$SELECTOR_RUN/dataset_ood_7_9" \
  --checkpoint "$SELECTOR_RUN/models/lp_vp_selector_checkpoint.pt" \
  --device cuda:0 --split test --output "$SELECTOR_RUN/ood_report.json"
```

## 面向 unitree_deploy_ws

新增的 `DepthPolicyInput.valid_support` 是填充前有效支持，形状为 1152，顺序与 `history` 相同，必须一起送入 runtime。`valid_ratio` 标量仍表示最新帧，不能复制它来替代两帧空间有效支持。
修改 ROS 消息后，要在原 ROS 环境重新构建消息、深度和控制包，所有发布者/订阅者使用同一份新消息定义：

```bash
cd unitree_deploy_ws
colcon build --symlink-install --packages-select g1_policy_msgs g1_depth g1_control
source install/setup.bash
```

将来的联合部署可用 `OnlineSelectorRuntime.from_artifacts(...)` 加载双分支和 selector，自动校验冻结策略身份、特征契约、标签契约和动作过渡步数。

`SelectorInput` 还需要真实 `lp_valid_mask`、`lp_packet_timestamp`（当前本地地图发布包的时间）与 `lp_timestamp`（底层扫描源时间）。VP 的 `vp_timestamp` 来自深度源帧的 header；没有单独 VP 包时间时默认使用源时间。所有绝对/epoch 时钟先保留 float64，再相减。
LP 源时间与发布包时间不能混用：机器人运动时，同一次扫描可生成新的局部地图发布包；包间变化应更新，而底层扫描包龄仍继续增长。

自动模式内部维护实际目标模式和切换时间；shadow 模式读取外部真实模式/驻留时间。最新 78 维 proprio 的最后 23 维必须是上次真正执行的混合动作；这里沿用冻结策略的 action 观测缩放 1.0。手动 override 应调用 runtime.reset 对齐实际模式与切换时间。

本包补齐这些训练/运行时接口及深度质量消息，没有把 selector 自动接入现有 `policy_node.py` 的机器人输出。现有 TensorRT 后端还需要六输出双分支推理和节点接线，这部分仍需后续联合部署实现与闭环验证。

## 验证范围

验证记录见 `TEST_RESULTS.json`。使用真实 CPU PyTorch；覆盖时间对齐、包缓存、epoch 包龄、独立随机流、PGTT/相机噪声、命令/事件 timer、重复/双路故障、动作过渡与快照、完整模拟环境采集、配额与部分数据拒绝，以及三种子监督训练、评估、TorchScript 导出/加载。
训练烟雾使用合成数据和合成冻结策略身份，验证管线连通，不代表实际 checkpoint 效果。

```bash
PYTHONPATH="$PWD:$PWD/rsl_rl:$PWD/unitree_deploy_ws/src/g1_depth" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  rsl_rl/tests/test_selector_safety_labels.py \
  rsl_rl/tests/test_selector_collection_v2.py \
  rsl_rl/tests/test_selector_training_v2.py \
  unitree_deploy_ws/src/g1_depth/test/test_depth_policy.py -q
```

当前环境没有 Isaac Sim、ROS 和 GPU，没有运行真实 checkpoint 的 Isaac 采集、4096 相机吞吐测试、ROS 构建或真机控制。EventManager 包装核对了 Isaac Lab v2.2.0 的布局；不支持的 timer/global interval 布局会明确报错。
命令/reset/push 兼容包装采用逐环境隔离调用，可能增加采集开销；先以 smoke 检查你安装的 Isaac 版本及吞吐，再扩到 4096。
