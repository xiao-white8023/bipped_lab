"""Reproduce the report comparison without loading or selecting on test predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CLASS_NAMES = ["双安全", "仅LP失败", "仅VP失败", "双失败"]
BINS = ["(0,0.2]秒", "(0.2,0.5]秒", "(0.5,0.7]秒", "(0.7,1]秒"]
METRICS = {
    "outcome/accuracy": "总体分类准确率",
    "outcome/log_loss": "未加权四分类NLL",
    "switch/correct_rescue_count": "正确救援数",
    "switch/correct_rescue_rate": "正确救援率",
    "switch/missed_rescue_count": "漏救数",
    "selection/dangerous_branch_count": "单失败时选坏支路数",
    "selection/dangerous_branch_rate": "单失败时选坏支路率",
    "switch/harmful_switch_count": "从安全支路切坏的次数",
    "switch/both_safe_switch_rate": "双安全切换请求率",
    "switch/both_failure_switch_rate": "双失败切换请求率",
    "selection/both_failure_flag_recall": "双失败识别召回",
    "selection/both_unreliable_flag_precision": "双失败识别精确率",
    "selection/lp_only_failure_wrong_rate": "仅LP失败选坏支路率",
    "selection/vp_only_failure_wrong_rate": "仅VP失败选坏支路率",
    "switch/request_count": "切换请求总数",
}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_result(result):
    matrix = result["confusion_rows_true_columns_predicted"]
    cross = result["true_outcome_x_predicted_outcome_x_request"]["counts"]
    m = result["metrics"]
    assert [sum(r) for r in matrix] == result["class_counts"]
    assert sum(map(sum, matrix)) == result["samples"]
    for c in range(4):
        for p in range(4):
            assert sum(cross[c][p]) == matrix[c][p]
    assert math.isclose(sum(matrix[c][c] for c in range(4))/result["samples"],
                        m["outcome/accuracy"], abs_tol=1e-7)
    assert sum(cross[c][p][1] for c in range(4) for p in range(4)) == m["switch/request_count"]
    for c, prefix in ((0, "both_safe"), (3, "both_failure")):
        assert sum(cross[c][p][1] for p in range(4)) == m[f"switch/{prefix}_switch_count"]
    assert cross[1][1][1]+cross[2][2][1] == m["switch/correct_rescue_count"]
    assert cross[1][2][1]+cross[2][1][1] == m["switch/harmful_switch_count"]
    assert m["switch/missed_rescue_count"]+m["switch/harmful_switch_count"] == m["selection/dangerous_branch_count"]
    assert sum(row[3] for row in matrix) == m["selection/both_unreliable_flag_count"]
    assert matrix[3][3] == m["selection/both_failure_flag_true_positive_count"]
    if "failure_timing" not in result:
        return
    rows = result["failure_timing"]["rows"]
    assert len(rows) == 15
    for c in (1, 2, 3):
        parts = [r for r in rows if r["class"] == c and not r["all_times"]]
        total, = [r for r in rows if r["class"] == c and r["all_times"]]
        assert len(parts) == 4
        assert sum(r["samples"] for r in parts) == total["samples"] == result["class_counts"][c]
        assert [sum(r["predicted_class_counts"][p] for r in parts) for p in range(4)] == matrix[c]
        assert total["predicted_class_counts"] == matrix[c]
        for row in (*parts, total):
            assert row["class_true_positive"] == row["predicted_class_counts"][c]
            assert math.isclose(row["class_recall"], row["class_true_positive"]/row["samples"])
            if c < 3:
                assert row["correct_rescue"]+row["missed_rescue"] == row["rescue_needed"]
                assert row["q0_missed_rescue"]+row["q3_missed_rescue"] <= row["missed_rescue"]
        if c < 3:
            prefix = "lp" if c == 1 else "vp"
            assert total["samples"]-total["safe_branch_selected"] == m[f"selection/{prefix}_only_failure_wrong_count"]
        else:
            assert total["switch_requests"] == m["switch/both_failure_switch_count"]
    totals = [r for r in rows if r["all_times"] and r["class"] < 3]
    for key, metric in (("correct_rescue", "correct_rescue_count"), ("rescue_needed", "correct_rescue_samples"),
                        ("missed_rescue", "missed_rescue_count")):
        assert sum(r[key] for r in totals) == m[f"switch/{metric}"]


def old_validation_timing(predictions_path, metadata_path, diagnostic):
    """Reconstruct old rescue/hold diagnostics from all 204750 aligned rows."""
    with np.load(predictions_path, allow_pickle=False) as p, np.load(metadata_path, allow_pickle=False) as meta:
        y, current = p["labels"], p["current_branch"]
        q, desired = p["probabilities_raw"], p["desired_branch_raw"]
        assert np.array_equal(p["sample_id"], meta["sample_id"])
        assert np.array_equal(y, meta["labels"])
        target, predicted = (y[:, 0]+2*y[:, 1]).astype(int), q.argmax(1)
        first = meta["future_metrics"][:, :, 0]
        assert np.isfinite(first).all() and np.array_equal(first, first.round())
        rows = []
        for c in (1, 2, 3):
            times = first[:, c-1] if c < 3 else first.max(1)
            for i, (lo, hi) in enumerate(((1, 10), (11, 25), (26, 35), (36, 50), (1, 50))):
                mask = (target == c) & (times >= lo) & (times <= hi)
                n = int(mask.sum())
                hist = np.bincount(predicted[mask], minlength=4).tolist()
                row = dict(**{"class": c}, all_times=i == 4, samples=n,
                           interval_s=[(lo-1)*.02, hi*.02], predicted_class_counts=hist,
                           class_true_positive=hist[c], class_recall=hist[c]/n)
                if i < 4:
                    archived, = [r for r in diagnostic["validation"] if r["class"] == c and r["time_bin"] == diagnostic["validation"][i]["time_bin"]]
                    assert archived["predicted_counts_q0_q1_q2_q3"] == hist
                if c < 3:
                    safe = 0 if c == 1 else 1
                    need = mask & (current != safe)
                    needed = int(need.sum())
                    rescued = int((need & (desired == safe)).sum())
                    row.update(rescue_needed=needed, correct_rescue=rescued, rescue_rate=rescued/needed,
                               missed_rescue=needed-rescued,
                               q0_missed_rescue=int((need & (predicted == 0)).sum()),
                               q3_missed_rescue=int((need & (predicted == 3)).sum()))
                rows.append(row)
    return rows


def percent(value):
    return f"{value*100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, default=Path("/home/cjh/.codex/attachments/2104ad84-8466-419b-8bf4-2869d2faa921/已粘贴的文本.txt"))
    parser.add_argument("--old", type=Path, default=Path("/home/cjh/桌面/joint_training_report.json"))
    parser.add_argument("--old-predictions", type=Path, default=Path("/home/cjh/桌面/validation_predictions.npz"))
    parser.add_argument("--old-metadata", type=Path, default=ROOT.parent/"selector_review_20261016/full_validation_20261016_232332/validation_metadata_join.npz")
    parser.add_argument("--old-timing", type=Path, default=ROOT/"time_classification_4bins_diagnostic.json")
    parser.add_argument("--output", type=Path, default=ROOT/"branch_gru_20261017_164954")
    args = parser.parse_args()
    old, new, diagnostic = load(args.old), load(args.new), load(args.old_timing)
    assert new["report_version"] == 4
    assert old["decision_probability_mode"] == new["decision_probability_mode"] == "raw"
    assert old["training_sources"] == new["training_sources"]
    assert old["loss_config"] == new["loss_config"]
    assert old["sampler_config"] == new["sampler_config"]
    assert old["source_sha256"]["legged_lab/selector/metrics.py"] == new["source_sha256"]["legged_lab/selector/metrics.py"]
    for report in (old, new):
        for split in ("train", "validation", "test", "original_test"):
            verify_result(report["results"][split+"_raw"])
    baseline_rows = old_validation_timing(args.old_predictions, args.old_metadata, diagnostic)
    assert sum(r["q0_missed_rescue"] for r in baseline_rows if r["all_times"] and r["class"] < 3) == 800
    aggregate = []
    for split in ("train", "validation", "test", "original_test"):
        a, b = old["results"][split+"_raw"], new["results"][split+"_raw"]
        assert a["class_counts"] == b["class_counts"]
        for key, label in METRICS.items():
            before, after = a["metrics"][key], b["metrics"][key]
            aggregate.append(dict(split=split, metric=key, label=label, old=before, new=after, delta=after-before))
    timing = []
    for split in ("validation", "test"):
        for c in (1, 2, 3):
            previous_tp, previous_n = diagnostic["user_supplied_tables"][split][str(c)]
            rows = [r for r in new["results"][split+"_raw"]["failure_timing"]["rows"] if r["class"] == c and not r["all_times"]]
            assert sum(previous_tp) == old["results"][split+"_raw"]["confusion_rows_true_columns_predicted"][c][c]
            for i, row in enumerate(rows):
                assert previous_n[i] == row["samples"]
                old_rate = previous_tp[i]/previous_n[i]
                item = dict(split=split, **{"class": c}, class_name=CLASS_NAMES[c], time_bin=BINS[i],
                            samples=row["samples"], old_tp=previous_tp[i], new_tp=row["class_true_positive"],
                            old_recall=old_rate, new_recall=row["class_recall"], delta=row["class_recall"]-old_rate)
                if split == "validation" and c < 3:
                    old_row = [r for r in baseline_rows if r["class"] == c and not r["all_times"]][i]
                    assert old_row["rescue_needed"] == row["rescue_needed"]
                    item.update(rescue_needed=row["rescue_needed"], old_rescue=old_row["correct_rescue"],
                                new_rescue=row["correct_rescue"], old_rescue_rate=old_row["rescue_rate"],
                                new_rescue_rate=row["rescue_rate"])
                timing.append(item)
    hold_counts = {}
    for label, rows in (("old", baseline_rows), ("new", new["results"]["validation_raw"]["failure_timing"]["rows"])):
        totals = [r for r in rows if r["all_times"] and r["class"] < 3]
        missed, q0, q3 = (sum(r[key] for r in totals) for key in ("missed_rescue", "q0_missed_rescue", "q3_missed_rescue"))
        hold_counts[label] = dict(missed=missed, q0=q0, q3=q3, wrong_direction=missed-q0-q3)
    generalization = []
    for c in (1, 2, 3):
        item = {"class": c, "class_name": CLASS_NAMES[c]}
        for split in ("train", "validation", "test"):
            row, = [r for r in new["results"][split+"_raw"]["failure_timing"]["rows"] if r["class"] == c and r["interval_s"] == [.7, 1.0]]
            item[split] = row
        item["train_minus_validation_recall"] = item["train"]["class_recall"]-item["validation"]["class_recall"]
        generalization.append(item)
    class_counts = new["training_sources"]["train_class_counts"]
    source_share = {CLASS_NAMES[c]: 20000/class_counts[c] for c in (1, 2, 3)}
    source_checks = {}
    for path, sha in new["source_sha256"].items():
        local = REPO/path
        source_checks[path] = dict(old_server_equals_new_server=old["source_sha256"].get(path) == sha,
                                  new_server_equals_local=local.exists() and digest(local) == sha)
    output = dict(
        provenance={name: dict(path=str(path), sha256=digest(path)) for name, path in
                    (("new_report", args.new), ("old_report", args.old), ("old_validation_predictions", args.old_predictions),
                     ("old_validation_metadata", args.old_metadata), ("old_timing_diagnostic", args.old_timing))},
        checks=dict(same_training_sources=True, same_loss=True, same_sampler=True, same_metrics_code_between_runs=True,
                    raw_modes_compared=True, matrices_request_tables_and_time_totals_verified=True),
        source_code_comparison=source_checks, architecture=new["architecture"], selected=new["selected"],
        all_candidates=new["all_candidates"], aggregate_comparison=aggregate, timing_comparison=timing,
        validation_missed_rescue_decomposition=hold_counts, far_time_train_validation_test=generalization,
        supplement_fraction_within_training_class=source_share, server_prediction_exports=new["prediction_exports"],
        limitations=["New checkpoint and NPZ arrays are recorded on the server, not attached locally.",
                     "Old test timing comes from archived user-supplied tables; class totals match the old report.",
                     "Counts are net changes, not matched per-sample transitions.",
                     "Metrics are offline snapshot action requests; runtime gates and closed-loop switching are not evaluated.",
                     "Mixed-source train/validation gaps do not isolate ordinary overfitting or source shortcuts.",
                     "Three seeds were selected by validation loss; no validation action/time metrics for the other two checkpoints are supplied."])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for filename, rows in (("aggregate_comparison.csv", aggregate), ("timing_comparison.csv", timing)):
        with (args.output/filename).open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    val_a, val_b = old["results"]["validation_raw"], new["results"]["validation_raw"]
    a, b = old["results"]["test_raw"]["metrics"], new["results"]["test_raw"]["metrics"]
    lines = ["# 20261017 branch_gru训练结果复核", "",
        "结论：本轮选出的公共编码＋双分支GRU模型没有改善预期的单路救援和远期识别。"
        "双安全误切减少、部分双失败召回提高，但单失败时选坏支路及漏救增多。当前不能作为旧模型的性能升级。", "",
        "## 比较范围与可比性", "",
        f"- 新报告：`{new['checkpoint']}`，report_version=4，selected seed=42，best_epoch=3，实际训练18轮。",
        f"- 基线：`{old['checkpoint']}`，selected seed={old['selected']['seed']}，best_epoch={old['selected']['best_epoch']}。",
        "- 数据源清单及其哈希、训练722949条、验证204750条、清洗测试172358条、损失配置和全局随机混合采样配置完全一致。",
        "- 对比两轮raw主结果；旧、新服务器metrics.py哈希相同。新报告大部分实现哈希与当前本地代码相同；"
        "本地metrics.py与服务器版本不同，但不影响两份报告之间的同版本比较。",
        "- 已核对所有raw分割的混淆矩阵、请求交叉表、救援/漏救/切坏恒等式及新报告时间分段总数。"
        "旧验证时间表由204750条预测与元数据逐条重算，旧测试分段依据此前用户提供的归档表，合计与旧报告一致。",
        "- 所有数字是离线快照上的请求和选择，未运行驻留门控或真实闭环切换；样本数不等于独立跌倒事件数。", "",
        "## 1. 主结果：误切减少，救援退步", "",
        "| 清洗测试集指标 | 旧模型 | 新模型 |", "|---|---:|---:|"]
    for key in ("outcome/accuracy", "switch/correct_rescue_count", "switch/correct_rescue_rate", "switch/missed_rescue_count",
                "selection/dangerous_branch_rate", "switch/both_safe_switch_rate", "switch/both_failure_switch_rate",
                "selection/both_failure_flag_recall", "selection/both_unreliable_flag_precision"):
        fmt = (lambda v: f"{v:,}") if key.endswith("count") else percent
        lines.append(f"| {METRICS[key]} | {fmt(a[key])} | {fmt(b[key])} |")
    lines += ["", "正确救援净减少149次；这是总量差，缺少新NPZ时不能说成具体149条旧成功样本变失败。"
        "验证集也从5665/6822降到5534/6822，正确救援净减少131次。双失败≤10%切换、≥80%召回仍未达到。",
        "原始压力测试的双失败召回反而从67.16%降到66.14%，双失败切换21.11%升到21.81%。"
        "因此双失败改善只在清洗分布小幅出现。", "",
        "## 2. 准确率上升为什么不等于更会救援", "",
        "验证集四类正确识别数量变化依次为：双安全+697、仅LP失败−204、仅VP失败−60、双失败+20；"
        "合计+453。准确率的提高主要来自占92.58%的双安全类。未加权NLL、Brier也下降，"
        "但这些总体概率指标不保证在当前argmax规则下提高救援。", "",
        "验证集漏救按预测原因：", "",
        "| 原因 | 旧模型 | 新模型 | 净变化 |", "|---|---:|---:|---:|"]
    for key, label in (("q0", "预测双安全保持"), ("q3", "预测双失败保持"), ("wrong_direction", "单失败方向预测错误"), ("missed", "漏救合计")):
        before, after = hold_counts["old"][key], hold_counts["new"][key]
        lines.append(f"| {label} | {before} | {after} | {after-before:+d} |")
    lines += ["", "净增漏救主要落在双安全保持一项。此表描述输出行为，不能直接推断网络内部使用了哪些特征。", "",
        "## 3. 远期识别：没有达到改结构的初衷", "",
        "下表时间统一为从当前快照到失败的(0.7,1]秒；单失败取失败支路首次触发时间，双失败取两条反事实回放中较晚的首次触发时间。", "",
        "| 分割 | 类别 | 旧正确数/样本数 | 新正确数/样本数 | 旧→新召回 |", "|---|---|---:|---:|---:|"]
    for row in timing:
        if row["time_bin"] != BINS[-1]:
            continue
        lines.append(f"| {row['split']} | {row['class_name']} | {row['old_tp']}/{row['samples']} | {row['new_tp']}/{row['samples']} | {percent(row['old_recall'])} → {percent(row['new_recall'])} |")
    lines += ["", "验证集真正需要救援的远期子集中：仅LP从1130/1415（79.86%）降到1030/1415（72.79%）；"
        "仅VP从327/590（55.42%）降到317/590（53.73%）。分类召回与需救援时救援率的分母不同。", "",
        "## 4. 这次新增的训练时间分解揭示了什么", "",
        "本报告已经有完整训练时间分解，不能继续认为缺失。远期各类召回如下：", "",
        "| 类别 | 混合训练集 | 验证集 | 差距（百分点） |", "|---|---:|---:|---:|"]
    for row in generalization:
        lines.append(f"| {row['class_name']} | {percent(row['train']['class_recall'])} | {percent(row['validation']['class_recall'])} | {row['train_minus_validation_recall']*100:.2f} |")
    lines += ["", "LP远期在训练集也只有74.29%，说明当前选中模型在训练分布上也未充分识别此类。"
        "VP与双失败还有较大训练/验证差距，说明不能只用统一的‘预测1秒太难’解释。",
        f"但训练集仅VP失败的{source_share['仅VP失败']*100:.2f}%、双失败的{source_share['双失败']*100:.2f}%来自各自的20k定向补采，"
        "验证集只来自base。这不是同分布条件下的纯训练/验证对照，不能直接把差距全归因于过拟合，"
        "也不能仅凭训练高召回就断言base远期观测信息一定足够。应分base_train和三个补采来源分别计算时间指标。",
        "与旧模型相比，新模型连混合训练表现也降低：训练目标0.25505升到0.30403，"
        "仅LP/仅VP/双失败总体分类召回分别从88.16/85.20/88.53%降到83.83/78.32/86.26%。"
        "这不支持‘训练拟合更好，只是验证过拟合’的简单叙述。", "",
        "## 5. 结构和选模要分别判断", "",
        "- 实际运行的是新结构：公共87维、每路私有98维、双路各1层GRU、最后状态＋注意力池化＋当前帧，"
        "参数615940（旧模型412100），历史仍为5帧0.4秒。所有编码器、GRU、注意力和输出头都有参数更新记录；"
        "这能排除整块未训练的明显问题，不能证明学到了有效表征。",
        f"- 选模目标是验证集加权CE＋两路边缘BCE。旧{val_a['objective']:.9f}，新{val_b['objective']:.9f}，"
        "损失确实更低，但救援更差。说明该标量不充分代表当前行为目标，不代表已证明损失代码错误。",
        "- seed42/43/44各自最佳验证目标为0.250726521、0.255069286、0.250862032。"
        "42与44只差0.000135511，其他seed的行为/时间指标未提供，不能断言44更好。",
        "- best_epoch=3是这次验证目标最优的位置，不等于程序只训练3轮，也不能据此建议单纯延长训练。"
        "缺少逐epoch曲线和其他最佳权重的验证行为。",
        "- 这次同时改变了分支编码、GRU深度、读出和参数规模，单次比较不能把退步单独归因于注意力、公共编码或1层GRU。"
        "A（旧单流GRU）→B（只改读出）→C（分支GRU）的控制实验尚未齐全。", "",
        "## 6. 下一步的明确顺序", "",
        "1. 保留旧模型作为性能基线，这轮C模型先作为实验结果保留。"
        "先比较已有seed最佳权重在同一验证集上的救援、选坏率、双安全/双失败误切、双失败召回/精确率及远期分段，"
        "配合训练曲线判断是否为选模取舍；使用固定训练归一化参数，不按测试挑选seed。",
        "2. 利用本轮已在服务器导出的train_predictions.npz，按base_train、LP补采、VP补采、双失败补采×类别×时间拆解；"
        "识别远期训练分数来自哪个来源，避免混合高分遮住base的困难。",
        "3. 用新validation_predictions.npz与旧预测按sample_id对齐，统计新增/修复漏救以及q0、q3、风险差变化。"
        "这会区分原有边界样本受argmax限制和新模型输出本身退步；当前只有报告，不能完成这一步。",
        "4. 再决定受控结构实验或来源采样调整。优先补齐只改读出的B对照；任何改动以验证集行为与时间指标判断，"
        "另设尚未参与选择的新保留集做最终验收，避免把已经多次查看的测试集继续用于调参。", "",
        "所需数组路径已写在报告prediction_exports中，当前只拿到路径/哈希，没有本轮NPZ或checkpoint文件。"
        "本次未修改训练代码、未重新训练，也未以测试指标选择新配置。", "",
        "[完整统计JSON](comparison.json) · [总体对照CSV](aggregate_comparison.csv) · [四时间段对照CSV](timing_comparison.csv)", ""]
    (args.output/"review.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(args.output), checks=output["checks"], missed=hold_counts,
                          test_correct_rescue_old=a["switch/correct_rescue_count"], test_correct_rescue_new=b["switch/correct_rescue_count"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
