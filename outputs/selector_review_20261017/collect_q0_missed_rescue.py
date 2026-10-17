"""Export the four raw probabilities for all 800 q0-hold missed rescues."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/cjh/桌面/validation_predictions.npz')
MANIFEST = SOURCE.with_suffix('.json')
METADATA = ROOT.parent/'selector_review_20261016/full_validation_20261016_232332/validation_metadata_join.npz'
DEST = ROOT/'q0_missed_rescue_800'
CLASS_NAMES = ['both_safe', 'lp_only_failure', 'vp_only_failure', 'both_failure']
BRANCH_NAMES = ['VP', 'LP']
EDGES = np.linspace(0, 1, 11)


def counts(values):
    return np.bincount(values, minlength=4).tolist()


def describe(probabilities):
    result = {}
    for column, name in enumerate(CLASS_NAMES):
        values = probabilities[:, column].astype(np.float64)
        quantiles = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
        result[f'q{column}'] = {
            'class': name,
            'mean': float(values.mean()),
            **dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'], quantiles.tolist())),
            'histogram_counts': np.histogram(values, EDGES)[0].tolist(),
        }
    return result


def fmt(value):
    return f'{100*value:.4f}%'


def main():
    manifest = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == manifest['prediction_sha256']
    assert manifest['primary_probability_mode'] == 'raw'
    p = np.load(SOURCE, allow_pickle=False)
    m = np.load(METADATA, allow_pickle=False)
    assert np.array_equal(m['sample_id'], p['sample_id'])
    assert np.array_equal(m['labels'], p['labels'])
    y, current, q = p['labels'], p['current_branch'], p['probabilities_raw']
    target = y[:, 0]+2*y[:, 1]
    predicted = q.argmax(1)
    desired = current.copy()
    desired[predicted == 1] = 0
    desired[predicted == 2] = 1
    current_failure = np.where(current == 1, y[:, 0], y[:, 1]).astype(bool)
    alternative_failure = np.where(current == 1, y[:, 1], y[:, 0]).astype(bool)
    missed = current_failure & ~alternative_failure & (desired == current)
    mask = missed & (predicted == 0)
    assert np.array_equal(predicted, p['predicted_class_raw'])
    assert np.array_equal(missed, p['missed_rescue_raw'])
    assert np.array_equal(mask, p['wrong_safe_hold_raw'])
    assert mask.sum() == 800
    selected = np.flatnonzero(mask)
    risks = np.stack((q[:, 1]+q[:, 3], q[:, 2]+q[:, 3]), 1)
    delta = np.where(current == 1, risks[:, 0]-risks[:, 1], risks[:, 1]-risks[:, 0])
    groups = {'all': mask, 'current_LP_should_switch_VP': mask & (current == 1),
              'current_VP_should_switch_LP': mask & (current == 0)}
    report = {
        'source': str(SOURCE), 'source_sha256': digest, 'split': 'validation', 'probability_mode': 'raw',
        'selection': 'current branch fails, alternative safe, no argmax switch, predicted class 0',
        'rows': int(mask.sum()), 'class_order': CLASS_NAMES,
        'true_class_counts': counts(target[mask]), 'predicted_class_counts': counts(predicted[mask]),
        'histogram_bin_edges': EDGES.tolist(),
        'histogram_interval': '[left, right); final bin includes 1',
        'groups': {},
        'risk_difference_counts': {f'current_minus_alternative_gt_{threshold}': int((mask & (delta > threshold)).sum())
            for threshold in [0, .01, .05, .1, .2]},
        'q0_confidence_counts': {f'q0_ge_{threshold}': int((mask & (q[:, 0] >= threshold)).sum())
            for threshold in [.5, .6, .7, .8, .9, .95, .99]},
        'top_probability_tie_rows': int(((q[mask] == q[mask].max(1, keepdims=True)).sum(1) > 1).sum()),
        'note': 'Conditional error-subgroup statistics; means are not class frequencies or an estimate of population calibration.',
    }
    for name, group in groups.items():
        report['groups'][name] = {'rows': int(group.sum()), 'true_class_counts': counts(target[group]),
            'probability_distribution': describe(q[group]), 'sum_of_probability_means': float(q[group].astype(float).mean(0).sum())}
    DEST.mkdir(exist_ok=True)
    (DEST/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    header = ['sample_id', 'split_row_index', 'shard_index', 'row_in_shard', 'current_branch', 'required_branch',
              'true_class', 'true_class_name', 'lp_failure_label', 'vp_failure_label',
              'q0_both_safe', 'q1_lp_only_failure', 'q2_vp_only_failure', 'q3_both_failure',
              'p_lp_failure', 'p_vp_failure', 'current_minus_alternative_risk',
              'q0_minus_runner_up', 'lp_first_failure_seconds', 'vp_first_failure_seconds']
    identifiers = {key: p[key] for key in ['sample_id', 'split_row_index', 'shard_index', 'row_in_shard']}
    first_failure_steps = m['future_metrics'][:, :, 0]
    with (DEST/'samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i in selected:
            times = [float(v*.02) if v >= 0 else '' for v in first_failure_steps[i]]
            writer.writerow([int(identifiers['sample_id'][i]), int(identifiers['split_row_index'][i]), int(identifiers['shard_index'][i]),
                int(identifiers['row_in_shard'][i]), BRANCH_NAMES[int(current[i])], BRANCH_NAMES[1-int(current[i])],
                int(target[i]), CLASS_NAMES[int(target[i])], int(y[i, 0]), int(y[i, 1]),
                *[float(v) for v in q[i]], *[float(v) for v in risks[i]], float(delta[i]),
                float(q[i, 0]-q[i, 1:].max()), *times])
    lines = ['# 800条q0保持漏救：四分类输出概率分布', '',
        '来源：完整validation，raw概率。筛选条件独立复算，与导出错误mask逐行一致。', '',
        f"真实类别数量：{report['true_class_counts']}；预测类别数量：{report['predicted_class_counts']}。", '',
        '以下概率均值和分位数描述各样本的网络输出，不是800条样本的真实类别占比，也不用于估计总体校准。', '']
    chinese = {'q0': '双安全', 'q1': '仅LP失败', 'q2': '仅VP失败', 'q3': '双失败'}
    for group_name, group in report['groups'].items():
        lines.extend([f"## {group_name}：{group['rows']}条", '',
            '| 输出 | 均值 | P10 | P25 | 中位数 | P75 | P90 | 最小 | 最大 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for key, row in group['probability_distribution'].items():
            values = [fmt(row[field]) for field in ['mean', 'p10', 'p25', 'median', 'p75', 'p90', 'min', 'max']]
            lines.append(f"| {key} {chinese[key]} | "+' | '.join(values)+' |')
        lines.append('')
    lines.extend(['## 全800条的概率分箱计数', '',
        '| 概率区间 | q0 | q1 | q2 | q3 |', '|---|---:|---:|---:|---:|'])
    for bin_index in range(10):
        interval = f'[{10*bin_index}%, {10*(bin_index+1)}%'+(']' if bin_index == 9 else ')')
        values = [str(report['groups']['all']['probability_distribution'][f'q{k}']['histogram_counts'][bin_index]) for k in range(4)]
        lines.append('| '+interval+' | '+' | '.join(values)+' |')
    lines.extend(['', 'q0置信度计数：`'+json.dumps(report['q0_confidence_counts'])+'`。', '',
        '当前支路减备选支路的预测风险差计数：`'+json.dumps(report['risk_difference_counts'])+'`。', '',
        '样本逐条明细见[samples.csv](samples.csv)，全部精度统计见[summary.json](summary.json)。',
        '生成脚本为上一级目录的`collect_q0_missed_rescue.py`。', ''])
    (DEST/'distribution.md').write_text('\n'.join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
