#!/usr/bin/env python3
"""Audit the exact training roots in a joint selector training report.

Run in the server's existing PyTorch environment; Isaac Sim is not imported.
Only <source>/train/shard_*.pt is opened. No features are used as targets.
Time = first_failure_step * 0.02 s; -1 means no failure within the 1 s window.
Coarse bins: (0,.2], (.2,.5], (.5,1], matching the validation lead-time audit.
Each failure histogram is conditional on failure, not on all training rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

DT = 0.02
STEPS = 50
ORDER = ['both_safe', 'lp_only_failure', 'vp_only_failure', 'both_failure']
ROLES = {
    'lp_all_failures': 'LP全部失败',
    'vp_all_failures': 'VP全部失败',
    'lp_only_failure': '仅LP失败：LP时间',
    'vp_only_failure': '仅VP失败：VP时间',
    'both_failure_lp': '双失败：LP时间',
    'both_failure_vp': '双失败：VP时间',
    'both_failure_earlier': '双失败：较早一路',
    'both_failure_later': '双失败：较晚一路',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def empty_counts():
    return {'rows': 0, 'class_counts': np.zeros(4, dtype=np.int64),
            'histograms': {k: np.zeros(STEPS, dtype=np.int64) for k in ROLES}}


def update_counts(counts, labels, steps):
    y, f = as_numpy(labels), as_numpy(steps)
    n = len(y)
    if y.shape != (n, 2) or f.shape != (n, 2):
        raise ValueError('labels and first failure steps must both be [N,2] in LP,VP order')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(f).all():
        raise ValueError('invalid binary labels or nonfinite failure time')
    if not ((f == np.round(f)) & ((f == -1) | ((f >= 1) & (f <= STEPS)))).all():
        raise ValueError('failure steps must be -1 or integers 1..50')
    y, f = y.astype(bool), f.astype(np.int64)
    if not np.array_equal(f > 0, y):
        raise ValueError('failure times disagree with LP/VP labels')
    target = y[:, 0].astype(np.int64) + 2 * y[:, 1].astype(np.int64)
    both = target == 3
    values = {
        'lp_all_failures': f[y[:, 0], 0],
        'vp_all_failures': f[y[:, 1], 1],
        'lp_only_failure': f[target == 1, 0],
        'vp_only_failure': f[target == 2, 1],
        'both_failure_lp': f[both, 0],
        'both_failure_vp': f[both, 1],
        'both_failure_earlier': f[both].min(axis=1),
        'both_failure_later': f[both].max(axis=1),
    }
    counts['rows'] += n
    counts['class_counts'] += np.bincount(target, minlength=4)
    for key, observed in values.items():
        counts['histograms'][key] += np.bincount(observed, minlength=STEPS + 1)[1:]


def merge_counts(destination, source):
    destination['rows'] += source['rows']
    destination['class_counts'] += source['class_counts']
    for key in ROLES:
        destination['histograms'][key] += source['histograms'][key]


def summarize(counts):
    rows = counts['rows']
    classes = counts['class_counts']
    result = {'rows': rows, 'class_counts': classes.tolist(),
              'no_failure_within_1s': {'lp': int(classes[0] + classes[2]),
                                       'vp': int(classes[0] + classes[1])},
              'failure_time_distributions': {}}
    for key, hist in counts['histograms'].items():
        n = int(hist.sum())
        coarse = [int(hist[:10].sum()), int(hist[10:25].sum()), int(hist[25:].sum())]
        fine = hist.reshape(10, 5).sum(axis=1).astype(int).tolist()
        quantiles = None
        if n:
            positions = np.searchsorted(hist.cumsum(), np.ceil(np.array([.1, .25, .5, .75, .9]) * n)) + 1
            quantiles = dict(zip(['p10', 'p25', 'p50', 'p75', 'p90'], (positions * DT).tolist()))
        result['failure_time_distributions'][key] = {
            'failures': n, 'step_counts_1_to_50': hist.tolist(),
            'coarse_counts': coarse,
            'coarse_percent_of_failures': [100 * x / n for x in coarse] if n else [None] * 3,
            'fine_0_1s_counts': fine,
            'mean_seconds': float(np.dot(np.arange(1, STEPS + 1), hist) * DT / n) if n else None,
            'quantiles_seconds_discrete_cdf': quantiles,
        }
    expected = {'lp_all_failures': int(classes[1] + classes[3]),
                'vp_all_failures': int(classes[2] + classes[3]),
                'lp_only_failure': int(classes[1]), 'vp_only_failure': int(classes[2]),
                **{k: int(classes[3]) for k in ROLES if k.startswith('both_failure_')}}
    for key, n in expected.items():
        if result['failure_time_distributions'][key]['failures'] != n:
            raise ValueError('histogram totals disagree with class counts: ' + key)
    return result


def read_shard(path):
    import torch
    # mmap avoids eagerly reading the large feature tensors, which are unused.
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def audit_report(report_path, loader=read_shard):
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    sampler = report['sampler_config']
    roots, expected_counts = sampler['source_roots'], sampler['source_counts']
    if len(roots) != len(expected_counts) or not roots:
        raise ValueError('invalid report training source configuration')
    if len({str(Path(x).expanduser().resolve()) for x in roots}) != len(roots):
        raise ValueError('duplicate training roots in report')
    combined, supplements, sources = empty_counts(), empty_counts(), []
    for source_index, (root, expected_rows) in enumerate(zip(roots, expected_counts)):
        root = Path(root).expanduser().resolve()
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('label_order') != ['lp_1.0s', 'vp_1.0s']:
            raise ValueError(f'unsupported label order in {manifest_path}')
        fields = manifest['future_metric_order_per_branch']
        metric_index = fields.index('first_failure_step')
        contract = manifest.get('metadata', {}).get('safety_label_contract', manifest.get('safety_label_contract', {}))
        if contract.get('control_dt_s', DT) != DT or contract.get('horizon_steps', [50]) != [50]:
            raise ValueError(f'expected control_dt=.02 and horizon_steps=[50] in {manifest_path}')
        if manifest['counts']['train'] != expected_rows:
            raise ValueError(f'training row count differs from report: {root}')
        paths = sorted((root / 'train').glob('shard_*.pt'))
        shard_count = manifest['shards']['train']
        if paths != [root / 'train' / f'shard_{i:05d}.pt' for i in range(shard_count)]:
            raise ValueError(f'training shards missing or unexpected in {root}')
        counts = empty_counts()
        for i, path in enumerate(paths):
            data = loader(path)
            future = data['future_metrics']
            if future.ndim != 3 or tuple(future.shape[1:]) != (2, len(fields)):
                raise ValueError(f'invalid future_metrics shape in {path}')
            update_counts(counts, data['labels'], future[:, :, metric_index])
            del data, future
            if (i + 1) % 10 == 0 or i + 1 == len(paths):
                print(f'[audit] source={source_index} train shards={i+1}/{len(paths)} rows={counts["rows"]}', flush=True)
        if counts['rows'] != expected_rows:
            raise ValueError(f'actual training shard rows differ from manifest: {root}')
        merge_counts(combined, counts)
        if source_index:
            merge_counts(supplements, counts)
        sources.append({'source_index': source_index, 'dataset': str(root),
                        'manifest_sha256': sha256(manifest_path), 'train_shards': len(paths),
                        **summarize(counts)})
    training = report['training_statistics']
    if combined['rows'] != training['rows'] or combined['class_counts'].tolist() != training['class_counts']:
        raise ValueError('combined training rows/classes disagree with training report')
    return {'version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'training_report': str(report_path), 'training_report_sha256': sha256(report_path),
            'split': 'train', 'validation_and_test_opened': False,
            'control_dt_seconds': DT, 'class_order': ORDER,
            'coarse_bins': ['(0,0.2]', '(0.2,0.5]', '(0.5,1.0]'],
            'fine_bins': [f'({i/10:.1f},{(i+1)/10:.1f}]' for i in range(10)],
            'denominator': 'Each histogram is conditional on that role failing within 1s.',
            'scope': 'Saved training rows; correlated rows are not independent failure events.',
            'source_order': 'As saved in sampler_config.source_roots; first=base, rest=supplements.',
            'verified_against_training_report': True,
            'all_training': summarize(combined), 'base_training': sources[0],
            'supplements_training': summarize(supplements), 'by_source': sources}


def print_table(title, stats):
    print(f'\n{title}: rows={stats["rows"]:,}, classes={stats["class_counts"]}')
    print('| 分组 | 失败数 | ≤0.2秒 | >0.2～0.5秒 | >0.5～1秒 |')
    print('|---|---:|---:|---:|---:|')
    for key, name in ROLES.items():
        item = stats['failure_time_distributions'][key]
        cells = [f'{n:,} ({p:.2f}%)' if p is not None else '—'
                 for n, p in zip(item['coarse_counts'], item['coarse_percent_of_failures'])]
        print('| ' + ' | '.join([name, f'{item["failures"]:,}', *cells]) + ' |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-report', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_report(args.training_report)
    output = args.output or args.training_report.parent / 'train_failure_time_distribution.json'
    output = output.expanduser().resolve()
    if output == args.training_report.expanduser().resolve() or any(output.is_relative_to(Path(s['dataset'])) for s in result['by_source']):
        raise ValueError('output must not replace the training report or write inside source datasets')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print_table('完整训练集', result['all_training'])
    print_table('清洗后的主训练集', result['base_training'])
    print_table('补采训练集合计', result['supplements_training'])
    for source in result['by_source'][1:]:
        print_table(source['dataset'], source)
    print(f'\n[verified] saved {output}')


if __name__ == '__main__':
    main()
