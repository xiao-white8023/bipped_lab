"""Compare the audited training timing table with raw validation predictions."""
from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'selector_review_20261016/full_validation_20261016_232332'
SOURCE = Path('/home/cjh/桌面/validation_predictions.npz')
BINS = ['<=0.2s', '>0.2s and <=0.5s', '>0.5s and <=1.0s']
TRAIN = {
    'lp_all_failure': (78190, [18550, 23324, 36316], .48),
    'vp_all_failure': (55274, [11371, 14963, 28940], .54),
    'lp_only_failure': (52510, [5694, 14804, 32012], .60),
    'vp_only_failure': (29594, [477, 6006, 23111], .72),
    'both_failure_earlier': (25680, [13876, 8635, 3169], .18),
    'both_failure_later': (25680, [9874, 8842, 6964], .28),
}
SOURCES = {
    'lp_only_failure': {'base': [3830, 9467, 19213], 'supplement': [1864, 5337, 12799]},
    'vp_only_failure': {'base': [202, 1928, 7464], 'supplement': [275, 4078, 15647]},
    'both_failure_later': {'base': [2533, 1616, 1531], 'supplement': [7341, 7226, 5433]},
}


def counts(steps):
    assert np.all((steps >= 1) & (steps <= 50))
    return np.array([(steps <= 10).sum(), ((steps > 10) & (steps <= 25)).sum(), (steps > 25).sum()])


def main():
    p = np.load(SOURCE, allow_pickle=False)
    m = np.load(PREVIOUS/'validation_metadata_join.npz', allow_pickle=False)
    y, current = p['labels'], p['current_branch']
    assert np.array_equal(p['sample_id'], m['sample_id']) and np.array_equal(y, m['labels'])
    c = y[:, 0]+2*y[:, 1]
    predicted = p['probabilities_raw'].argmax(1)
    first = m['future_metrics'][:, :, 0].astype(int)
    need = np.where(current == 1, y[:, 0], y[:, 1]).astype(bool) & ~np.where(current == 1, y[:, 1], y[:, 0]).astype(bool)
    validation = {}
    rescue_bins = []
    for name, class_id, branch in [('lp_only_failure', 1, 0), ('vp_only_failure', 2, 1)]:
        all_rows = c == class_id
        steps = first[all_rows, branch]
        validation[name] = {'rows': int(all_rows.sum()), 'counts': counts(steps).tolist(),
            'median_seconds': float(np.median(steps)*.02)}
        masks = [first[:, branch] <= 10, (first[:, branch] > 10) & (first[:, branch] <= 25), first[:, branch] > 25]
        for time_bin, time_mask in zip(BINS, masks):
            selected = all_rows & need & time_mask
            n = int(selected.sum())
            q0 = int((selected & (predicted == 0)).sum())
            q3 = int((selected & (predicted == 3)).sum())
            rescued = int((selected & p['correct_rescue_raw']).sum())
            other = n-q0-q3-rescued
            assert other >= 0
            rescue_bins.append({'class': name, 'time_bin': time_bin,
                'all_single_failure_rows': int((all_rows & time_mask).sum()),
                'rescue_needed': n, 'correct_rescue': rescued,
                'q0_missed': q0, 'q3_missed': q3, 'other_missed': other,
                'q0_miss_rate': q0/n, 'correct_rescue_rate': rescued/n})
    later = first.max(1)
    both = c == 3
    validation['both_failure_later'] = {'rows': int(both.sum()), 'counts': counts(later[both]).tolist(),
        'median_seconds': float(np.median(later[both])*.02)}
    dual_bins = []
    for name, mask in zip(BINS, [later <= 10, (later > 10) & (later <= 25), later > 25]):
        selected = both & mask
        n, tp = int(selected.sum()), int((selected & (predicted == 3)).sum())
        dual_bins.append({'time_bin': name, 'true_both_failure': n, 'q3_argmax_true_positive': tp, 'recall': tp/n})
    training = {name: {'rows': n, 'counts': values, 'median_seconds': median} for name, (n, values, median) in TRAIN.items()}
    for name, group in training.items():
        assert sum(group['counts']) == group['rows']
        if name in SOURCES:
            assert (np.array(SOURCES[name]['base'])+SOURCES[name]['supplement']).tolist() == group['counts']
    early_late = []
    for class_name in ['all', 'lp_only_failure', 'vp_only_failure']:
        relevant = rescue_bins if class_name == 'all' else [row for row in rescue_bins if row['class'] == class_name]
        for period in ['<=0.5s', '>0.5s']:
            selected = [row for row in relevant if (row['time_bin'] == BINS[2]) == (period == '>0.5s')]
            total = {key: sum(row[key] for row in selected) for key in ['rescue_needed', 'correct_rescue', 'q0_missed', 'q3_missed', 'other_missed']}
            early_late.append({'class': class_name, 'period': period, **total,
                'q0_miss_rate': total['q0_missed']/total['rescue_needed'],
                'correct_rescue_rate': total['correct_rescue']/total['rescue_needed']})
    assert sum(row['q0_missed'] for row in rescue_bins) == 800
    report = {'training': training, 'training_sources': SOURCES,
        'training_provenance': 'User-provided table, independently reproduced from all four actual training sources with the exact base cleaning filter.',
        'validation': validation, 'rescue_by_time_bin': rescue_bins, 'rescue_early_late': early_late,
        'dual_argmax_recall_by_time': dual_bins,
        'scope': 'Raw full validation, before runtime gates. Counts are correlated snapshot rows, not independent failure events.',
        'time_origin': 'Candidate snapshot; first safety trigger in the forced branch rollout, dt=.02s.',
        'bin_labels': BINS}
    (ROOT/'training_validation_timing_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (ROOT/'validation_rescue_by_failure_time.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rescue_bins[0]))
        writer.writeheader(); writer.writerows(rescue_bins)
    lines = ['# 训练失败时间分布与验证漏救：分母对齐后的结论', '',
        '用户提供的训练时间表已经通过全部实际训练分片独立复现。以下验证统计来自完整raw预测与原始元数据关联。', '',
        '| 类别 | 训练>0.5秒占比 | 验证>0.5秒占比 | 训练/验证中位数（秒） |',
        '|---|---:|---:|---:|']
    for name in ['lp_only_failure', 'vp_only_failure', 'both_failure_later']:
        t, v = training[name], validation[name]
        lines.append(f"| {name} | {100*t['counts'][2]/t['rows']:.2f}% | {100*v['counts'][2]/v['rows']:.2f}% | {t['median_seconds']:.2f}/{v['median_seconds']:.2f} |")
    lines.extend(['', '训练中晚失败很多，验证与训练的粗时间结构接近。基础与补采同类的时间结构也接近；'
        '不能把当前主要问题归结为晚失败数量不足或显著的总体时间分布迁移。', '',
        '分母进一步限制为当前正在执行坏支路、另一支路安全的需救援样本：', '',
        '| 类别 | 时间段 | 需救援 | q0保持漏救 | q0漏救率 | 正确救援率 |',
        '|---|---|---:|---:|---:|---:|'])
    for row in early_late:
        lines.append(f"| {row['class']} | {row['period']} | {row['rescue_needed']} | {row['q0_missed']} | {100*row['q0_miss_rate']:.2f}% | {100*row['correct_rescue_rate']:.2f}% |")
    lines.extend(['', '需救援6822行中，晚失败3643行，占53.40%；q0漏救800行中，晚失败664行，占83%。'
        '晚段q0漏救率18.23%，早段4.28%，约4.26倍。这个差异在对齐分母后仍存在，VP方向最明显。', '',
        'q0漏救率只计算预测双安全导致的漏救；正确救援率还受q3保持及方向错误影响，两列并非互补。', '',
        '下一项最有辨别力的统计是对同一冻结checkpoint的逐条训练预测进行相同分组，再与验证对照。'
        '若训练也差，优先检查可观测信息、表示与优化；若训练好而验证差，优先检查条件覆盖、来源/事件/地形泛化。'
        '该对照只是定位线索，数据分布差异仍需控制。当前附件缺逐条训练预测，因此不能完成这一步判别。', '',
        '较晚失败样本行数多不保证独立事件、多样性、匹配负例或早期信号充分。'
        '总体时间分布接近也不能排除按故障家族、阶段、地形等条件划分的分布变化。', '',
        '这些单点预测不能证明后续连续采样点也一直漏报；需用冻结模型沿事件时间轴评估首次有效请求与实际执行时刻。', '',
        '[完整统计JSON](training_validation_timing_comparison.json) · [按时间段的验证救援CSV](validation_rescue_by_failure_time.csv)', ''])
    (ROOT/'training_validation_timing_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'validation': validation, 'rescue_early_late': early_late, 'dual_bins': dual_bins}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
