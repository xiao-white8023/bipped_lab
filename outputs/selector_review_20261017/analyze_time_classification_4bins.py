"""Verify supplied time-stratified recalls and inspect validation error destinations."""
from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/cjh/桌面/validation_predictions.npz')
META = ROOT.parent/'selector_review_20261016/full_validation_20261016_232332/validation_metadata_join.npz'
BIN_NAMES = ['<=0.2s', '>0.2s and <=0.5s', '>0.5s and <=0.7s', '>0.7s and <=1.0s']
SUPPLIED = {
    'validation': {1: ([1085, 2788, 2056, 2738], [1220, 3099, 2376, 3597]),
                   2: ([47, 503, 800, 871], [81, 659, 1037, 1495]),
                   3: ([698, 310, 89, 72], [746, 479, 188, 225])},
    'test': {1: ([907, 2279, 1737, 2231], [1008, 2527, 1970, 2800]),
             2: ([37, 421, 634, 748], [51, 539, 784, 1274]),
             3: ([644, 292, 123, 89], [699, 436, 220, 241])},
}


def main():
    p = np.load(SOURCE, allow_pickle=False)
    m = np.load(META, allow_pickle=False)
    y, current, q = p['labels'], p['current_branch'], p['probabilities_raw']
    ids = p['sample_id']
    assert np.array_equal(ids, m['sample_id']) and np.array_equal(y, m['labels'])
    c, predicted, desired = y[:, 0]+2*y[:, 1], q.argmax(1), p['desired_branch_raw']
    first = m['future_metrics'][:, :, 0].astype(int)
    report = json.loads(Path('/home/cjh/桌面/joint_training_report.json').read_text())
    rows = []
    for class_id in [1, 2, 3]:
        times = first[:, class_id-1] if class_id < 3 else first.max(1)
        time_bin = np.searchsorted([10, 25, 35, 50], times, side='left')
        for bin_index, bin_name in enumerate(BIN_NAMES):
            selected = (c == class_id) & (time_bin == bin_index)
            histogram = np.bincount(predicted[selected], minlength=4)
            n, tp = int(selected.sum()), int(histogram[class_id])
            expected_tp, expected_n = SUPPLIED['validation'][class_id]
            assert (tp, n) == (expected_tp[bin_index], expected_n[bin_index])
            row = {'class': class_id, 'time_bin': bin_name, 'rows': n, 'predicted_counts_q0_q1_q2_q3': histogram.tolist(),
                   'class_true_positive': tp, 'class_recall': tp/n}
            if class_id < 3:
                safe_branch = 0 if class_id == 1 else 1
                need = selected & (current != safe_branch)
                correct = int((selected & (desired == safe_branch)).sum())
                rescue = int((need & (desired == safe_branch)).sum())
                row.update(safe_branch_selected=correct, safe_branch_selection_rate=correct/n,
                    rescue_needed=int(need.sum()), correct_rescue=rescue, rescue_rate=rescue/int(need.sum()))
            rows.append(row)
    for split in ['validation', 'test']:
        matrix = report['results'][split+'_raw']['confusion_rows_true_columns_predicted']
        for class_id, (tp, n) in SUPPLIED[split].items():
            assert sum(tp) == matrix[class_id][class_id]
            assert sum(n) == sum(matrix[class_id])
    dual_single = (c == 3) & np.isin(predicted, [1, 2])
    earlier, later = first.min(1), first.max(1)
    same = first[:, 0] == first[:, 1]
    predicts_lp = predicted == 1
    matches_earlier = ~same & np.where(predicts_lp, first[:, 0] < first[:, 1], first[:, 1] < first[:, 0])
    dual_probe = {}
    for name, mask in [('all_dual_predicted_single', dual_single),
                       ('later_gt_0_7_dual_predicted_single', dual_single & (later > 35))]:
        n, e, ties = int(mask.sum()), int((mask & matches_earlier).sum()), int((mask & same).sum())
        dual_probe[name] = {'rows': n, 'predicts_earlier_failing_branch': e,
            'predicts_later_failing_branch': n-e-ties, 'same_time': ties,
            'earlier_failure_median_seconds': float(np.median(earlier[mask])*.02),
            'later_failure_median_seconds': float(np.median(later[mask])*.02),
            'time_gap_median_seconds': float(np.median(later[mask]-earlier[mask])*.02)}
    assert dual_probe['all_dual_predicted_single']['rows'] == 392
    assert dual_probe['later_gt_0_7_dual_predicted_single']['predicts_earlier_failing_branch'] == 97
    output = {'probability_mode': 'raw', 'rows': len(c), 'validation': rows,
        'user_supplied_tables': SUPPLIED, 'validation_all_bins_verified': True,
        'test_bin_counts_verification': 'User supplied; aggregate counts match the saved raw test confusion matrix. No per-row test predictions available.',
        'dual_predicted_single_timing_probe': dual_probe,
        'time_semantics': 'First safety trigger in each forced-policy rollout, dt=.02s; two times are counterfactual trajectories, not sequential real sensor failures.',
        'runtime_gates_evaluated': False}
    (ROOT/'time_classification_4bins_diagnostic.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (ROOT/'dual_predicted_single_validation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'predicted_class', 'lp_first_failure_seconds', 'vp_first_failure_seconds',
            'predicts_earlier_failing_branch', 'same_time', 'q0', 'q1', 'q2', 'q3'])
        for i in np.flatnonzero(dual_single):
            writer.writerow([int(ids[i]), int(predicted[i]), f'{first[i, 0]*.02:.2f}', f'{first[i, 1]*.02:.2f}',
                bool(matches_earlier[i]), bool(same[i]), *q[i].tolist()])
    lines = ['# 四时间段分类召回：指标核对与误判去向', '',
        '验证表12格均由完整raw预测逐条复现。测试表分段依据用户提供数据，其各类总量及正确数与已保存测试混淆矩阵一致；未重新使用测试预测调参。', '',
        '用户表统计各真实类别的识别召回。单失败的分类召回、最终选安全支路比例、当前坏支路需救援时的救援率应分别解释。', '',
        '| 真实类 | 时间 | q0 | q1 | q2 | q3 | 类别召回 | 选安全支路率 | 需救援时救援率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        counts = ' | '.join(str(v) for v in row['predicted_counts_q0_q1_q2_q3'])
        selection = f"{100*row['safe_branch_selection_rate']:.2f}%" if 'safe_branch_selection_rate' in row else '不适用'
        rescue = f"{100*row['rescue_rate']:.2f}%" if 'rescue_rate' in row else '不适用'
        lines.append(f"| {row['class']} | {row['time_bin']} | {counts} | {100*row['class_recall']:.2f}% | {selection} | {rescue} |")
    lines.extend(['', '两个不同的晚失败误判模式：', '',
        '- 仅VP失败、>0.7秒：1495条中624条分类错误，其中555条判双安全，占错误的88.94%。',
        '- 双失败、较晚一路>0.7秒：225条中153条分类错误，其中123条判单失败，占错误的80.39%。',
        '- 上述123条中97条指出了较早失败的支路，25条指出较晚支路，1条同刻。'
        '较早/较晚时间中位数分别0.54/0.84秒，时间差中位数0.30秒。'
        '这是回放结果和预测的对应关系，不能据此宣称已经证明模型内部采用何种特征或推理。', '',
        '两条首次失败时间属于从同一快照分别执行LP、VP的两条回放轨迹，不是一个机器人先后发生两次传感器故障。', '',
        '同checkpoint的训练逐条预测仍缺失，当前不能区分训练阶段没有学会与验证泛化不足。'
        '现有训练总体分类召回为仅LP88.16%、仅VP85.20%、双失败88.53%，但训练来源构成不同，'
        '不能用总体差距替代按来源、时间段的比较。', '',
        '[完整JSON](time_classification_4bins_diagnostic.json) · [392条双失败被判单失败的明细](dual_predicted_single_validation.csv)', ''])
    (ROOT/'time_classification_4bins_analysis.md').write_text('\n'.join(lines))
    print(json.dumps(dual_probe, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
