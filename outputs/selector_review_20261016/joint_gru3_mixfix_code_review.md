# selector_joint_gru3_mixfix_20261016.zip 复审

日期：2026-10-16。文件：`/home/cjh/桌面/selector_joint_gru3_mixfix_20261016.zip`。

SHA256：`421525dce5a83b1936b1d9aa98d1e4ff49607be6a750f1d39397fcbb4a8b4928`。

结论：本次全局混批、固定类别权重归约和日志口径修复有效，没有发现新的模型数学或训练数据隔离错误。存在一处当前仓库安装兼容阻断，以及一处AMP恢复机制被提前异常打断的长训健壮性问题。未安装到当前仓库，未修改训练源码，未执行完整训练。

## 已修复的核心问题

1. `legged_lab/selector/joint_training.py:64` 对所有训练行执行一次全局 `randperm`，再切批；不再连续按 shard 组成训练批次。各训练来源按原比例参与，无放回，每轮覆盖一次。
2. `legged_lab/selector/joint_model.py:96` 使用 `cross_entropy(..., reduction="none").mean()`。其分母是行数，纯类别批次的类别权重不再被抵消。
3. `joint_training.py:122` 仅从训练计数计算一次权重归一化因子，归一后训练分布的期望权重为1；验证复用相同权重，不使用验证批次或类别占比重新定义分母。
4. `RowMeanLossAccumulator` 按组件均值乘样本数累计；对固定预测，整体损失和梯度在不同批次划分下保持一致。
5. 日志区分在线训练目标与轮末固定模型验证目标；最后的 `train_raw` 和 `validation_raw` 使用同一选定模型、eval模式和损失归约，可以比较泛化差距。
6. 只读 mmap 缓存避免全局打乱后每行重载整份 PT 分片，缓存只包含输入、标签和ID，未来审核字段不进入网络。

因此，上一轮“纯类批次权重抵消”的工程理由已不再成立。等权CE仍值得做实验对照，但不应再把关闭权重作为规避这个已修复错误的必要条件。加权损失对概率目标的影响及后续校准，属于独立的统计取舍。

## 问题一：当前仓库的旧文件无法被安装器识别升级（P2，确定复现）

位置：包内 `install_joint_selector.py:40–42`，`baseline_files_sha256.json`。

安装器只接受与已知旧版本完全相同的文件哈希。这一保护原则合理，未知代码修改应被阻止。但当前仓库六个 joint 文件都与上一压缩包的相应内容仅差一个末尾LF，六项哈希全部不同。因此，对当前仓库执行安装会在预检阶段拒绝，无法应用mixfix；临时复制当前六文件执行安装器已复现。

| 文件 | 当前仓库字节数 | 旧压缩包字节数 |
|---|---:|---:|
| joint_model.py | 8501 | 8502 |
| joint_data.py | 17411 | 17412 |
| joint_training.py | 12708 | 12709 |
| train_selector_joint.py | 16326 | 16327 |
| eval_selector_joint.py | 3645 | 3646 |
| run_selector_joint_training.sh | 2879 | 2880 |

逐项检查均满足 `当前文件字节 + b"\n" == 上一压缩包文件字节`，不是其他逻辑差异。`baseline_files_sha256.json` 的六项哈希确实正确匹配旧压缩包。

建议：将已核实的末尾LF缺失版本纳入已知旧版本识别，或在核实差异后规范末尾换行，再走原有备份升级。保留未知修改保护，不应简单禁用哈希检查或忽略任意空白差异。

## 问题二：AMP临时梯度溢出会在GradScaler恢复前终止训练（P2，机制复现）

位置：包内 `legged_lab/scripts/train_selector_joint.py:95–99`。

当前顺序：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 1., error_if_nonfinite=True
)
scaler.step(optimizer)
scaler.update()
```

FP16动态缩放过程中，有限loss也可能产生临时溢出的梯度。`error_if_nonfinite=True`会提前抛异常，阻止后面的跳步和scale调整。PyTorch的常规AMP流程允许 `scaler.step` 跳过含inf/NaN梯度的更新，再调用 `scaler.update` 调整缩放系数：[官方AMP示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-clipping)。

独立CPU机制复现使用新包实际 `JointOutcomeNetwork`，最后线性层权重及偏置置零，单条双安全标签，等权CE加边缘BCE，CPU float16 autocast及GradScaler：

- loss有限：`2.079441547393799`。
- 默认scale：`65536`。
- unscale后的最后bias梯度：`[-inf, 0.25, 0.25, 0.5]`。
- 当前严格裁剪抛出non-finite norm异常。
- 允许正常scaler流程继续后，scale可降至`32768`，该次优化更新被跳过。

建议：AMP分支允许scaler处理临时溢出，并记录跳过更新及scale变化；非AMP分支仍可严格检查，连续异常另行终止。补充一个实际触发缩放溢出的回归测试。

这项复现验证了控制流问题，没有证明默认4096批次一定溢出，也未进行真实CUDA训练复现。问题在旧版本也存在，不是全局混批引入的新错误。

## 实际验证

验证环境：本地Python 3.11、PyTorch 2.7.0+cu128；测试计算均在CPU，训练源码在隔离目录加载。

| 测试组 | 结果 |
|---|---|
| test_joint_selector.py | 20通过 |
| test_global_training.py | 7通过，1跳过 |
| test_install.py | 5通过 |
| Python编译与bash语法 | 通过 |

共33项，32通过，1项跳过。跳过的是多worker实际读取对照，原因是当前沙箱禁止PyTorch tensor IPC所需的本地Unix socket；不是测试通过，也不是已经证实的数据实现错误。目标服务器应实际运行这一项。

安装器测试在临时目录验证了新装、重复安装、已知旧版备份升级及未知修改拒绝。使用上一压缩包精确文件字节，可成功升级，5个有内容变化的文件均保留精确备份；未变化的eval文件无需备份。

数据验证：

- 使用真实来源计数 `[668369,20000,20000,20000]`、batch4096、seed42，采样得到728,369个唯一行、178批、0个纯来源批，各来源总计数完全一致。
- 从真实基础集和LP库各取一个8192行分片，建立16,384行临时缓存；随机跨分片读取features、labels、sample_id与原数据完全一致，单进程DataLoader的顺序与randperm一致。
- 728,369行5帧的缓存容量约3.8503 GiB。这里的数据组合是原始V4加6万补采，并非文档默认的清洗基础集组合。

## 仍需区分的运行范围

- 本地默认清洗目录 `outputs/g1_renet_selector_cross_clean_20261015_164009/dataset` 仍不存在；服务器是否存在需以服务器文件为准。
- 新结果使用argmax动作与告警；旧规则采用阈值和方向比较。新旧整体系统可以比较，但不能把差异全部归因于网络或BCE。
- `selection/low_risk_hold_count`仍表示预测双安全的保持数量，并不等同旧阈值保持口径；建议命名或表格中明确标注。
- V6旧导出/在线部署接入仍未实现，包内已明确声明。
- 没有发现验证/测试数据进入训练缓存，也没有发现用test指标选轮次、种子或校准参数。数据审计读取test标签用于数量、血缘核验，与使用测试指标调参不同。

建议在处理上述两个工程问题后进行相同数据、相同决策规则下的CE-only与CE+BCE对照，并保留GRU深度对照。当前证据支持训练实现修正到位，不代表真实任务指标已经改善。
