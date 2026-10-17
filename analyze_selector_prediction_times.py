#!/usr/bin/env python3
"""Audit complete V6 prediction bundles by first failure time, without fitting.

Requires NumPy and the server's existing PyTorch only for reading data shards.
Input: NPZ + same-stem JSON from eval_selector_joint.py --export-predictions.
Reads only the prediction's validation/test split and verifies every row ID,
label, current branch, shard position, manifest digest and checkpoint identity.
Bins use integer control steps: <=10, 11..25, 26..35, 36..50 (dt=.02 s).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ORDER = ['both_safe', 'lp_only_failure', 'vp_only_failure', 'both_failure']
BINS = ['(0,0.2]', '(0.2,0.5]', '(0.5,0.7]', '(0.7,1.0]']
NAMES = {'lp_only_failure': '仅LP失败：LP时间',
         'vp_only_failure': '仅VP失败：VP时间',
         'both_failure_later': '双失败：较晚一路',
         'both_failure_earlier': '双失败：较早一路'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def read_shard(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def summarize_arrays(arrays, failure_steps):
    y = np.asarray(arrays['labels'])
    c = np.asarray(arrays['current_branch'])
    q = np.asarray(arrays['probabilities_raw'])
    f = np.asarray(failure_steps)
    n = len(y)
    if y.shape != (n, 2) or c.shape != (n,) or q.shape != (n, 4) or f.shape != (n, 2):
        raise ValueError('invalid label, branch, probability or time shape')
    if not np.isin(y, [0, 1]).all() or not np.isin(c, [0, 1]).all():
        raise ValueError('labels/current branch must be binary')
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError('invalid four-class probabilities')
    if not np.isfinite(f).all() or not ((f == np.round(f)) & ((f == -1) | ((f >= 1) & (f <= 50)))).all():
        raise ValueError('failure steps must be -1 or integers 1..50')
    if not np.array_equal(f > 0, y.astype(bool)):
        raise ValueError('failure times disagree with labels')
    y, f = y.astype(np.int64), f.astype(np.int64)
    t = y[:, 0] + 2 * y[:, 1]
    p = q.argmax(1)
    if 'predicted_class_raw' in arrays and not np.array_equal(p, arrays['predicted_class_raw']):
        raise ValueError('stored classes disagree with raw probabilities')
    desired = np.where(p == 1, 0, np.where(p == 2, 1, c))
    request = desired != c
    selected_failure = np.where(desired == 1, y[:, 0], y[:, 1]).astype(bool)
    rescue = ((c == 1) & (t == 1)) | ((c == 0) & (t == 2))
    roles = {'lp_only_failure': (t == 1, f[:, 0], 1),
             'vp_only_failure': (t == 2, f[:, 1], 2),
             'both_failure_later': (t == 3, f.max(1), 3),
             'both_failure_earlier': (t == 3, f.min(1), 3)}
    confusion = np.bincount(t * 4 + p, minlength=16).reshape(4, 4)
    out = {'samples': n, 'class_counts': np.bincount(t, minlength=4).tolist(),
           'confusion_rows_true_columns_predicted': confusion.tolist(),
           'single_failure_wrong': int((np.isin(t, [1, 2]) & selected_failure).sum()),
           'correct_rescue': int((rescue & request).sum()),
           'rescue_opportunities': int(rescue.sum()), 'groups': {}}
    for name, (mask, time, target) in roles.items():
        denominator = int(mask.sum())
        bucket = np.searchsorted([10, 25, 35], time, side='left')
        rows = []
        for b in range(4):
            m = mask & (bucket == b)
            size = int(m.sum())
            predicted = np.bincount(p[m], minlength=4)
            correct = int(predicted[target])
            opportunities = int((m & rescue).sum())
            rescued = int((m & rescue & request).sum())
            rows.append({'bin': BINS[b], 'rows': size,
                'percent_of_group': size / denominator * 100 if denominator else None,
                'pred_class_counts': predicted.tolist(), 'correct_class': correct,
                'missed_class': size - correct,
                'class_recall_percent': correct / size * 100 if size else None,
                'switch_requests': int((m & request).sum()),
                'rescue_opportunities': opportunities, 'correct_rescue': rescued,
                'missed_rescue': opportunities - rescued,
                'rescue_rate_percent': rescued / opportunities * 100 if opportunities else None,
                'harmful_switch': int((m & ~rescue & request).sum()) if target in [1, 2] else None})
        if sum(r['rows'] for r in rows) != denominator or sum(r['correct_class'] for r in rows) != confusion[target, target]:
            raise ValueError('binned totals disagree with overall confusion matrix')
        out['groups'][name] = {'rows': denominator, 'bins': rows}
    return out


def audit_bundle(path, loader=read_shard):
    path = Path(path).expanduser().resolve()
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta.get('format') != 'selector_joint_predictions' or meta.get('version') != 1:
        raise ValueError('expected a V6 complete prediction export')
    if not meta.get('full_split_export') or meta.get('class_order') != ORDER:
        raise ValueError('partial split or wrong class order')
    if sha256(path) != meta['prediction_sha256']:
        raise ValueError('NPZ checksum differs from sidecar')
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in ['labels', 'current_branch', 'sample_id', 'split_row_index',
             'shard_index', 'row_in_shard', 'probabilities_raw', 'predicted_class_raw']}
    n = len(a['labels'])
    if n != meta['rows'] or not np.array_equal(a['split_row_index'], np.arange(n)) or len(np.unique(a['sample_id'])) != n:
        raise ValueError('prediction rows incomplete, out of order or duplicated')
    source = meta['source']
    split = source['split']
    if split not in ['validation', 'test']:
        raise ValueError('this audit accepts validation/test only')
    root = Path(source['dataset']).expanduser().resolve()
    manifest_path = root / 'manifest.json'
    manifest_sha = sha256(manifest_path)
    if manifest_sha != source['dataset_manifest_sha256']:
        raise ValueError('dataset manifest changed since prediction export')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['label_order'] != ['lp_1.0s', 'vp_1.0s'] or manifest['counts'][split] != n:
        raise ValueError('wrong dataset label order or split size')
    metric = manifest['future_metric_order_per_branch'].index('first_failure_step')
    if source.get('safety_label_contract', {}).get('control_dt_s', .02) != .02:
        raise ValueError('expected control_dt=.02')
    shards = [root / split / f'shard_{i:05d}.pt' for i in range(manifest['shards'][split])]
    if [str(p) for p in shards] != meta['shard_order']:
        raise ValueError('shard order differs from prediction export')
    times = np.full((n, 2), -99, dtype=np.int64)
    offset = 0
    for i, shard in enumerate(shards):
        data = loader(shard)
        labels = numpy(data['labels'])
        size = len(labels)
        rows = slice(offset, offset + size)
        if (not np.array_equal(a['shard_index'][rows], np.full(size, i))
                or not np.array_equal(a['row_in_shard'][rows], np.arange(size))):
            raise ValueError('shard row trace mismatch: ' + str(shard))
        if not np.array_equal(a['sample_id'][rows], numpy(data['scenario_seed']).reshape(-1)):
            raise ValueError('sample IDs do not align: ' + str(shard))
        if not np.array_equal(a['labels'][rows], labels):
            raise ValueError('labels do not align: ' + str(shard))
        if not np.array_equal(a['current_branch'][rows], numpy(data['features'][:, -1, -2])):
            raise ValueError('observed current branch does not align: ' + str(shard))
        future = numpy(data['future_metrics'][:, :, metric])
        if not np.isfinite(future).all() or not (future == np.round(future)).all():
            raise ValueError('invalid noninteger failure step')
        times[rows] = future.astype(np.int64)
        offset += size
        del data, future
        if (i + 1) % 10 == 0 or i + 1 == len(shards):
            print(f'[audit] {split}: shards={i+1}/{len(shards)} rows={offset}', flush=True)
    if offset != n or (times == -99).any():
        raise ValueError('failure times do not cover the full prediction split')
    return {'split': split, 'dataset': str(root), 'dataset_manifest_sha256': manifest_sha,
            'predictions': str(path), 'prediction_sha256': meta['prediction_sha256'],
            'checkpoint_sha256': source['checkpoint_sha256'], **summarize_arrays(a, times)}


def print_tables(result):
    print(f'\n{result["split"]}: samples={result["samples"]:,}; classes={result["class_counts"]}')
    for name, group in result['groups'].items():
        print(f'\n{NAMES[name]}')
        print('| 时间 | 样本数 | 最高分类正确数 | 分类召回率 | 需要救援数 | 正确救援数 |')
        print('|---|---:|---:|---:|---:|---:|')
        for row in group['bins']:
            recall = f'{row["class_recall_percent"]:.2f}%' if row['class_recall_percent'] is not None else '—'
            print(f'| {row["bin"]} | {row["rows"]} | {row["correct_class"]} | {recall} | {row["rescue_opportunities"]} | {row["correct_rescue"]} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.training_report.read_text(encoding='utf-8'))
    results = []
    for path in args.predictions:
        result = audit_bundle(path)
        if result['checkpoint_sha256'] != report['checkpoint_sha256_before_test']:
            raise ValueError('predictions are not from the requested latest checkpoint')
        saved = report['results'][result['split'] + '_raw']
        if (saved['samples'] != result['samples'] or saved['class_counts'] != result['class_counts']
                or saved['confusion_rows_true_columns_predicted'] != result['confusion_rows_true_columns_predicted']):
            raise ValueError('fresh raw predictions differ from the saved training report')
        results.append(result)
    output = args.output.expanduser().resolve()
    if output.exists() or output == args.training_report.expanduser().resolve():
        raise ValueError('choose a new output file')
    if any(output.is_relative_to(Path(result['dataset'])) for result in results):
        raise ValueError('output must be outside source datasets')
    data = {'version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'time_bins': BINS, 'control_dt_seconds': .02, 'probability_mode': 'raw',
            'class_order': ORDER, 'current_branch_codes': {'VP': 0, 'LP': 1},
            'training_report_sha256': sha256(args.training_report),
            'weights_updated': False, 'calibration_refitted': False,
            'thresholds_refitted': False, 'runtime_gates_evaluated': False,
            'notes': ['Class recall is not rescue recall; both are saved separately.',
                      'Both-failure primary grouping uses max(T_LP,T_VP).',
                      'Safe branches have no failure time within the horizon.',
                      'Counts are saved rows, not independent failure events.'],
            'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    for result in results:
        print_tables(result)
    print(f'\n[verified] saved {output}', flush=True)


if __name__ == '__main__':
    main()
