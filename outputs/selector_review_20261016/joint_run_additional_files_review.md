# 同次联合selector训练的补充文件核对

日期：2026-10-16。对应运行：`g1_renet_selector_joint_gru3_20261016_222818`。

## 文件对应

本次9个附件逐字段与此前完整训练报告比较：

| 附件顺序 | 内容 | 对照结果 |
|---|---|---|
| 1 | original_test_calibrated | 与总报告完全一致 |
| 2 | test_calibrated | 与总报告完全一致 |
| 3 | train_calibrated | 与总报告完全一致 |
| 4 | train_raw | 与总报告完全一致 |
| 5 | validation_calibrated | 与总报告完全一致 |
| 6 | validation_raw | 与总报告完全一致 |
| 7 | validation_raw | 与附件6字节相同，重复文件 |
| 8 | 训练行缓存manifest | 与附件9公共字段完全相同 |
| 9 | training_cache | 与总报告完全一致，仅比附件8多directory/access字段 |

这些文件补齐的是独立报告及训练缓存元信息，没有增加完整验证集logits或概率。

训练缓存arrays为：features `[722949,5,283]`、labels `[722949,2]`、sample_id `[722949]`。它保存模型输入和标签，不是网络对validation的逐样本预测。payload_bytes=4103458524，约3.82GiB。

## 来源和权重账目一致

662949行清洗基础训练＋三组各20000行补采＝722949。

| 类别 | 基础训练 | 定向补采 | 合计 | 定向样本占该类 |
|---|---:|---:|---:|---:|
| 双安全 | 615165 | 0 | 615165 | 0% |
| 仅LP失败 | 32510 | 20000 | 52510 | 38.09% |
| 仅VP失败 | 9594 | 20000 | 29594 | 67.58% |
| 双失败 | 5680 | 20000 | 25680 | 77.88% |

统计中的类别权重、训练归一化常数和期望权重1.0000000103相互一致，没有发现补采没被计入或权重归一化失败。上述来源占比说明分析泛化时需要分别评估基础域与补采域；它本身不证明来源捷径或过拟合。

## VP库failed=true：实际是采集后源码保护检查失败

上一轮审查没有覆盖这条采集日志，需要收窄“来源检查通过”的结论：逐行文件/元信息审计通过，不代表采集全过程的每项保护均通过。

实际日志：

`outputs/g1_renet_selector_directed_remaining_20261014_171524/logs/collect_vp_failure.log`

- 5158行：accepted=20000/20000。
- 5161行：finalize_collection_audit，collection_failed=false。
- 5163行：dataset_saved，collection_complete=true。
- 5167–5169行：CPU分析确认20000行，标签类别`[0,0,20000,0]`。
- 5217–5224行：verify_frozen_sources抛出 `protected RENet source files changed during selector collection`，最终failed=true。

独立重读20个真实分片：20000行、features全部有限、scenario_seed无重复，manifest哈希与本次报告一致。

源码保护对象是RENet_env.py、RENet_cfg.py、renet_actor_critic.py三个磁盘文件；不是checkpoint或policy.pt权重哈希。采集前后磁盘源码不同，不直接等于已经加载进程中的模型权重或配置实例发生改变。

VP启动时受保护哈希与原始V4及LP补采一致。次日双失败采集启动时，只有RENet_cfg.py的起始哈希发生变化，是优先追查线索。但没有记录VP结束时各文件哈希，不能确定报错具体由哪个文件引起、何时修改、是否影响已经运行的仿真。

当前clean_sources和joint_data会记录failed状态，并检查数据行、标签、来源契约等，但不会因failed=true自动拒收；也不能通过这些检查补足源码保护证据。

因此：该库文件和行数完整，来源审计仍有未闭合项。目前证据不足以认定标签错误、直接删除该库，或将当前模型性能问题归因于这次源码变动。需核查采集运行期间的源文件差异和修改时点。

## VP复杂故障确实启用了

日志中的drift_mature/person_near/person_depart等是LP故障生命周期统计，不是VP结构化噪声开关。不能根据这些零值断言新VP故障未采到。

实际VP故障元信息包含16个家族，共19922条活动期样本，另有78条恢复期样本。新增7类（编号19–25）共8494行，占全库42.47%。其中：

| 家族 | 活动期样本数 |
|---|---:|
| compound | 5105 |
| local_dropout | 4099 |
| motion_warp | 2565 |
| filter_trail | 508 |
| spatial_bias | 387 |
| edge_mismatch | 234 |
| dynamic_occlusion | 190 |

其余家族也都有实际样本。compound、local_dropout、motion_warp合计占全库58.85%；个别家族较少。已按故障家族过滤读取其VP参数，确认局部丢失、空间偏差、运动扭曲、时间抖动、滤波拖尾与动态遮挡参数及效果记录实际存在。

这些比例是定向筛选后保留的失败样本比例，不是运行中的故障发生率；16类都有样本也不代表其泛化充分。当前补采只进入train，没有独立同故障域的验证集。

## 已准备完整validation预测导出器

新增：`legged_lab/scripts/export_selector_joint_validation_predictions.py`。

- 固定读取checkpoint和validation，复用已有标准化、模型、温度及bias；不训练、不重新校准、不搜索阈值。
- 不调用会扫描train/test的scan_root或dataset_index。
- 校验V6类型、策略身份、特征/标签契约、基础manifest哈希、validation行数、分片数、组摘要、ID唯一性及原始当前分支。
- 保存完整raw_logits、raw_q、calibrated_q、derived_risks、labels、current_mask、sample_id、温度/bias、逐行分片位置。
- 可附带现有故障/地形审计元信息，用于错误分组，不作为模型输入。
- 对checkpoint、manifest、validation分片做前后SHA256核查，原子创建新输出并拒绝覆盖已有文件。

已通过CPU两分片端到端验证：使用torch.load哨兵确认没有打开train/test；与手工推理比较概率及逐行标签/ID/mask/审计信息一致；源文件摘要未改变；重复输出、错误来源manifest和重复ID均被拒绝。Python编译检查通过。真实运行checkpoint尚在服务器，本地没有执行该模型的完整validation推理。

将新增脚本同步到服务器，在原训练环境执行：

```bash
cd ~/TienKung-Lab
selector_joint_run="$PWD/outputs/g1_renet_selector_joint_gru3_20261016_222818"

CUDA_VISIBLE_DEVICES=1 python -u legged_lab/scripts/export_selector_joint_validation_predictions.py \
  --dataset "$PWD/outputs/g1_renet_selector_cross_clean_20261015_164009/dataset" \
  --checkpoint "$selector_joint_run/joint_selector_checkpoint.pt" \
  --output "$selector_joint_run/validation_predictions_full.pt" \
  --device cuda:0 --include-audit-metadata
```

后续动作诊断需要这个新生成的`validation_predictions_full.pt`；本次附带的训练缓存JSON不能替代它。已有评估结论保持：双安全请求达标，漏救援和双失败识别仍需改进。
