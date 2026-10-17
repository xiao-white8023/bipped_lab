"""First-failure timing for the previously verified 800 raw q0 missed rescues."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
DEST = ROOT/'q0_missed_rescue_800'
SOURCE = DEST/'samples.csv'
META = ROOT.parent/'selector_review_20261016/full_validation_20261016_232332/validation_metadata_join.npz'
DT = .02


def stats(steps):
    seconds = steps*DT
    quantiles = np.quantile(seconds, [0, .1, .25, .5, .75, .9, 1])
    return {
        'rows': len(steps), 'mean_seconds': float(seconds.mean()),
        **dict(zip(['min_seconds', 'p10_seconds', 'p25_seconds', 'median_seconds',
                    'p75_seconds', 'p90_seconds', 'max_seconds'], quantiles.tolist())),
        'histogram_counts': np.bincount((steps-1)//5, minlength=10).tolist(),
        'gt_0_5_seconds': int((steps > 25).sum()),
        'ge_0_8_seconds': int((steps >= 40).sum()),
        'exactly_0_8_seconds': int((steps == 40).sum()),
        'exact_step_counts_1_to_50': np.bincount(steps, minlength=51)[1:].tolist(),
    }


def main():
    samples = list(csv.DictReader(SOURCE.open()))
    assert len(samples) == 800 and len({row['sample_id'] for row in samples}) == 800
    m = np.load(META, allow_pickle=False)
    ids, labels, future = m['sample_id'], m['labels'], m['future_metrics']
    indices = np.array([int(row['split_row_index']) for row in samples])
    assert np.array_equal(ids[indices], [int(row['sample_id']) for row in samples])
    branches = np.array([0 if row['current_branch'] == 'LP' else 1 for row in samples])
    assert np.all(labels[indices, branches] == 1)
    assert np.all(labels[indices, 1-branches] == 0)
    steps = future[indices, branches, 0].astype(int)
    assert np.array_equal(steps, future[indices, branches, 0])
    assert np.all((steps >= 1) & (steps <= 50))
    assert np.all(future[indices, 1-branches, 0] == -1)
    for row, step in zip(samples, steps):
        time_column = row['current_branch'].lower()+'_first_failure_seconds'
        assert np.isclose(float(row[time_column]), step*DT, atol=1e-9, rtol=0)
    groups = {'all': stats(steps), 'LP_failure': stats(steps[branches == 0]),
              'VP_failure': stats(steps[branches == 1])}
    report = {
        'source_csv': str(SOURCE), 'source_csv_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'original_prediction_sha256': json.loads((DEST/'summary.json').read_text())['source_sha256'],
        'rows': 800, 'dt_seconds': DT, 'groups': groups,
        'definition': 'Elapsed time from candidate selector snapshot to first safety-failure trigger in the current failing branch rollout.',
        'failure_conditions': ['non_timeout_termination', 'torso_contact', 'tilt_gt_0_8_rad', 'clearance_lt_0_45_m'],
        'bin_interval': '(left, right], implemented with integer control steps: 1..5, 6..10, ..., 46..50',
        'scope': 'Pointwise counterfactual rollout labels; not evidence that a closed-loop selector never rescues before the trigger.',
        'quantile_method': 'numpy linear interpolation',
    }
    (DEST/'failure_time_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    header = ['sample_id', 'current_failing_branch', 'first_failure_step', 'first_failure_seconds',
              'time_bin_lower_exclusive_seconds', 'time_bin_upper_inclusive_seconds',
              'q0_both_safe', 'q1_lp_only_failure', 'q2_vp_only_failure', 'q3_both_failure']
    with (DEST/'failure_times.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i in np.argsort(steps, kind='stable'):
            row, step = samples[i], int(steps[i])
            bin_index = (step-1)//5
            writer.writerow([row['sample_id'], row['current_branch'], step, f'{step*DT:.2f}',
                f'{bin_index/10:.1f}', f'{(bin_index+1)/10:.1f}',
                *[row[k] for k in header[-4:]]])
    with (DEST/'failure_time_histogram.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['lower_exclusive_seconds', 'upper_inclusive_seconds', 'all_count',
            'all_fraction', 'lp_failure_count', 'vp_failure_count'])
        for i in range(10):
            total = groups['all']['histogram_counts'][i]
            writer.writerow([f'{i/10:.1f}', f'{(i+1)/10:.1f}', total, total/800,
                groups['LP_failure']['histogram_counts'][i], groups['VP_failure']['histogram_counts'][i]])
    lines = ['# 800条q0保持漏救：首次失败时间分布', '',
        '时间原点为selector作出本次判断的候选快照；每条取当前实际失败支路的首次失败时刻。', '',
        '失败指第一次触发非超时终止、躯干接触、倾角或低间隙判据，不必然是摔倒结束时刻。', '',
        '原始元数据与800条明细逐行核对，另一支路在1秒内均未失败。以整数控制步分箱，步长0.02秒。', '',
        '| 首次失败时间（秒，左开右闭） | 全部 | 占800条比例 | LP失败 | VP失败 |',
        '|---|---:|---:|---:|---:|']
    for i in range(10):
        n = groups['all']['histogram_counts'][i]
        lines.append(f"| ({i/10:.1f}, {(i+1)/10:.1f}] | {n} | {n/8:.3f}% | "
            f"{groups['LP_failure']['histogram_counts'][i]} | {groups['VP_failure']['histogram_counts'][i]} |")
    lines.extend(['| 合计 | 800 | 100% | 401 | 399 |', '',
        '| 分组 | 均值（秒） | 中位数（秒） | P10 | P25 | P75 | P90 | 最早 | 最晚 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for name, group in groups.items():
        values = [group[key] for key in ['mean_seconds', 'median_seconds', 'p10_seconds', 'p25_seconds',
            'p75_seconds', 'p90_seconds', 'min_seconds', 'max_seconds']]
        lines.append('| '+name+' | '+' | '.join(f'{value:.4f}' for value in values)+' |')
    lines.extend(['', '- 0.5秒之后首次失败：664/800=83.00%。',
        '- 0.8秒及以后首次失败：379/800=47.375%，包含正好0.8秒的23条。',
        '- 中位首次失败时间：总体0.78秒，LP组0.80秒，VP组0.74秒。', '',
        '这说明这些采样点的漏报大多发生在较长预测提前量上。统计不能证明selector在后续采样点仍不切换，'
        '也不能直接转换为闭环最终摔倒率。', '',
        '[逐条时间明细](failure_times.csv) · [分箱CSV](failure_time_histogram.csv) · [完整统计JSON](failure_time_summary.json)', ''])
    (DEST/'failure_time_distribution.md').write_text('\n'.join(lines))
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
