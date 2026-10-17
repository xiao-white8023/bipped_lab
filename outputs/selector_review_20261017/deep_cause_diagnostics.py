"""Read-only causal diagnostics; no optimizer, calibration fit, or test access."""
from pathlib import Path
import argparse
import json
import hashlib

import numpy as np


ROOT = Path('/home/cjh/TienKung-Lab')
DESKTOP = Path('/home/cjh/桌面')
META = ROOT/'outputs/selector_review_20261016/full_validation_20261016_232332/validation_metadata_join.npz'
DEST = Path(__file__).resolve().parent


def save(name, value):
    (DEST/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def quantile(a):
    return np.quantile(a, [.1, .5, .9]).tolist() if len(a) else None


def softmax(a):
    x = np.exp(a-a.max(1, keepdims=True))
    return x/x.sum(1, keepdims=True)


def score_diagnostics():
    p = np.load(DESKTOP/'validation_predictions.npz', allow_pickle=False)
    m = np.load(META, allow_pickle=False)
    r = json.loads((DESKTOP/'joint_training_report.json').read_text())
    y = p['labels'].astype(float)
    c = y[:, 0].astype(int)+2*y[:, 1].astype(int)
    z = p['logits'].astype(float)
    q = softmax(z)
    weights = np.array(r['loss_config']['class_weights'])
    lp, vp = q[:, 1]+q[:, 3], q[:, 2]+q[:, 3]
    ce = -np.log(q[np.arange(len(c)), c])*weights[c]
    bces = -(y*np.log(np.stack((lp, vp), 1))+(1-y)*np.log(np.stack((1-lp, 1-vp), 1)))
    # Per-row derivatives, before the common division by row count.
    gce = q.copy()
    gce[np.arange(len(c)), c] -= 1
    gce *= weights[c, None]
    gbce = np.zeros_like(q)
    for branch, positive, negative in ((0, [1, 3], [0, 2]), (1, [2, 3], [0, 1])):
        risk = lp if branch == 0 else vp
        gbce[:, positive] += .5*(risk-y[:, branch])[:, None]*softmax(z[:, positive])
        gbce[:, negative] -= .5*(risk-y[:, branch])[:, None]*softmax(z[:, negative])
    nc = np.linalg.norm(gce, axis=1)
    nb = np.linalg.norm(gbce, axis=1)
    cosine = (gce*gbce).sum(1)/np.maximum(nc*nb, 1e-30)
    total = ce+bces.mean(1)
    by_class = {}
    for k in range(4):
        mask = c == k
        by_class[str(k)] = {
            'rows': int(mask.sum()), 'weight': float(weights[k]),
            'mean_weighted_ce': float(ce[mask].mean()),
            'mean_lp_bce': float(bces[mask, 0].mean()), 'mean_vp_bce': float(bces[mask, 1].mean()),
            'global_objective_contribution': float(total[mask].sum()/len(c)),
            'share_of_global_objective': float(total[mask].sum()/total.sum()),
            'ce_gradient_norm_p10_p50_p90': quantile(nc[mask]),
            'aux_gradient_norm_p10_p50_p90': quantile(nb[mask]),
            'ce_aux_gradient_cosine_p10_p50_p90': quantile(cosine[mask]),
            'opposing_ce_aux_rows': int((cosine[mask] < -1e-8).sum()),
        }
    train_n = np.array(r['training_statistics']['class_counts'])
    loss = {'joint_ce': float(ce.mean()), 'lp_bce': float(bces[:, 0].mean()),
            'vp_bce': float(bces[:, 1].mean()), 'objective': float(total.mean()),
            'aux_fraction_of_objective': float(bces.mean()/total.mean()), 'by_true_class': by_class,
            'train_class_fraction': (train_n/train_n.sum()).tolist(),
            'train_ce_weight_mass_fraction': (train_n*weights/(train_n*weights).sum()).tolist(),
            'validation_class_fraction': np.bincount(c, minlength=4).__truediv__(len(c)).tolist(),
            'note': 'Gradients are at fixed validation logits, not the unobserved training trajectory.'}
    save('loss_diagnostic.json', loss)
    flags = m['first_failure_flags']
    future = m['future_metrics']
    assert np.array_equal(flags != 0, y.astype(bool))
    assert np.isfinite(future).all()
    pred = q.argmax(1)
    causes = {}
    for name, mask in [('all', np.ones(len(c), bool)), ('dual_detected', (c == 3)&(pred == 3)),
                       ('dual_missed', (c == 3)&(pred != 3))]:
        branches = []
        for b in range(2):
            v = mask&(y[:, b] == 1)
            unique, n = np.unique(flags[v, b], return_counts=True)
            branches.append({'positive': int(v.sum()),
                'first_flag_histogram': {str(int(a)): int(k) for a, k in zip(unique, n)},
                'replay_cause_stable': int(m['first_failure_cause_stable'][v, b].sum()),
                'non_timeout_termination_within_horizon': int((future[v, b, 5] > .5).sum()),
                'failure_without_non_timeout_termination': int((future[v, b, 5] <= .5).sum()),
                'first_failure_step_quantiles': quantile(future[v, b, 0])})
        causes[name] = branches
    safe = m['candidate_safety']
    initial_unsafe = (safe[:, 3] > .5)&((safe[:, 0] > .8)|(safe[:, 1] < .45)|(safe[:, 2] > .5))
    cohorts = {}
    for name, mask in [('initial_safe', ~initial_unsafe), ('initial_unsafe', initial_unsafe)]:
        true = (c == 3)&mask
        scores, target = q[mask, 3], c[mask] == 3
        order = np.argsort(-scores, kind='stable')
        scores, target = scores[order], target[order]
        ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
        tp, n = np.cumsum(target)[ends], ends+1
        recall, precision = tp/target.sum(), tp/n
        feasible = precision >= .5
        cohorts[name] = {'rows': int(mask.sum()), 'class_counts': np.bincount(c[mask], minlength=4).tolist(),
            'dual_argmax_tp': int((true&(pred == 3)).sum()),
            'dual_argmax_recall': float((true&(pred == 3)).sum()/true.sum()),
            'max_q3_recall_at_precision_50': float(recall[feasible].max()) if feasible.any() else None}
    save('target_definition_diagnostic.json', {'first_failure_bits_match_labels': True,
        'first_failure_flags': causes, 'initial_state_cohorts': cohorts})
    current = p['current_branch']
    requests = p['switch_request_raw']
    current_failure = np.where(current == 1, y[:, 0], y[:, 1]).astype(bool)
    alternative_failure = np.where(current == 1, y[:, 1], y[:, 0]).astype(bool)
    need = current_failure & ~alternative_failure
    thresholds = [r['calibration_candidates']['raw']['warning_threshold_diagnostic']['threshold'],
                  .4340755045413971]
    overlap = []
    for threshold in thresholds:
        vetoed = requests & (p['probabilities_raw'][:, 3] >= threshold)
        overlap.append({'q3_threshold': threshold,
            'existing_requests_with_warning': int(vetoed.sum()),
            'true_class_counts': np.bincount(c[vetoed], minlength=4).tolist(),
            'correct_rescues_vetoed': int((vetoed & need).sum())})
    save('warning_action_overlap_diagnostic.json', {
        'raw_warning_veto_counterfactual': overlap,
        'note': 'Validation-only diagnostics at already recorded thresholds; no new threshold search.'})
    print('score/target diagnostics complete', flush=True)


def training_distribution():
    import torch
    torch.set_num_threads(1)
    r = json.loads((DESKTOP/'joint_training_report.json').read_text())
    base = ROOT/'outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset'
    assert hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest() == r['training_sources']['original_manifest_sha256']
    sources = [base,
        ROOT/'outputs/g1_renet_selector_directed_20k_20261014_104733/lp_failure/dataset',
        ROOT/'outputs/g1_renet_selector_directed_remaining_20261014_171524/vp_failure/dataset',
        ROOT/'outputs/g1_renet_selector_directed_dual_20k_20261015_082634/both_failure/dataset']
    output = []
    new_families = {key: np.zeros(4, dtype=np.int64) for key in ('lp_kind_ge_14', 'vp_kind_ge_17')}
    for source_index, source in enumerate(sources):
        labels, faults, futures, events = [], [], [], []
        for path in sorted((source/'train').glob('shard_*.pt')):
            data = torch.load(path, map_location='cpu', weights_only=False)
            y = data['labels'].numpy()
            f = data['fault_metadata'].numpy()
            keep = np.ones(len(y), bool)
            if source_index == 0:
                keep &= ~(((f[:, 6] != 0)&(f[:, 7] == 0)&(y[:, 1] != 0))
                    |((f[:, 7] != 0)&(f[:, 6] == 0)&(y[:, 0] != 0)))
            labels.append(y[keep]); faults.append(f[keep])
            futures.append(data['future_metrics'].numpy()[keep, :, 0])
            events.append(data['event_identity'].numpy()[keep])
            del data
        y, f, future, event = map(np.concatenate, (labels, faults, futures, events))
        c = y[:, 0]+2*y[:, 1]
        active = f[:, 6]+2*f[:, 7]
        planned = (f[:, 0] != 0).astype(int)+2*(f[:, 1] != 0)
        pair = f[:, 0].astype(int)*100+f[:, 1]
        summary = {'root': str(source), 'rows': len(c), 'classes': np.bincount(c, minlength=4).tolist(),
            'by_actual_active_source': np.bincount(active*4+c, minlength=16).reshape(4, 4).tolist(),
            'by_planned_source': np.bincount(planned*4+c, minlength=16).reshape(4, 4).tolist(),
            'unique_events': len(np.unique(event, axis=0)),
            'by_vp_kind': {str(int(k)): np.bincount(c[f[:, 1] == k], minlength=4).tolist() for k in np.unique(f[:, 1])},
            'by_lp_kind': {str(int(k)): np.bincount(c[f[:, 0] == k], minlength=4).tolist() for k in np.unique(f[:, 0])},
            'by_planned_kind_pair': {str(int(k)): np.bincount(c[pair == k], minlength=4).tolist() for k in np.unique(pair)},
            'dual_later_failure_step_p10_p50_p90': quantile(future[c == 3].max(1)),
            'positive_first_failure_steps': [quantile(future[y[:, b] == 1, b]) for b in range(2)]}
        for key, mask in [('lp_kind_ge_14', f[:, 0] >= 14), ('vp_kind_ge_17', f[:, 1] >= 17)]:
            new_families[key] += np.bincount(c[mask], minlength=4)
        output.append(summary)
        print('training source analyzed', source_index, len(c), flush=True)
    total = np.array([s['classes'] for s in output]).sum(0)
    assert total.tolist() == r['training_statistics']['class_counts']
    base_pairs = set(output[0]['by_planned_kind_pair'])
    unmatched = [v for k, v in output[3]['by_planned_kind_pair'].items() if k not in base_pairs]
    save('training_distribution_diagnostic.json', {'sources': output, 'total_class_counts': total.tolist(),
        'new_family_class_counts': {key: value.tolist() for key, value in new_families.items()},
        'dual_bank_rows_with_kind_pair_absent_from_base': int(np.array(unmatched).sum()) if unmatched else 0,
        'source_code_order': ['none', 'lp_only', 'vp_only', 'dual'],
        'note': 'New family counters use planned kinds, not a claim that every row is currently active.'})


def observable_diagnostics():
    import torch
    torch.set_num_threads(1)
    p = np.load(DESKTOP/'validation_predictions.npz', allow_pickle=False)
    m = np.load(META, allow_pickle=False)
    root = ROOT/'outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset'
    columns = list(range(258, 281))
    values = np.zeros((len(p['labels']), len(columns)), dtype=np.float32)
    repeated_prefix = np.zeros(len(values), np.uint8)
    for i, path in enumerate(sorted((root/'validation').glob('shard_*.pt'))):
        data = torch.load(path, map_location='cpu', weights_only=False)
        where = np.flatnonzero(m['original_shard_index'] == i)
        rows = m['original_row_in_shard'][where]
        assert np.array_equal(data['scenario_seed'].numpy()[rows], p['sample_id'][where])
        x = data['features'].numpy()[rows]
        values[where] = x[:, -1, columns]
        equal = np.all(x[:, 1:] == x[:, :1], axis=2)
        repeated_prefix[where] = 1+np.cumprod(equal, axis=1).sum(1)
        del data
    q = p['probabilities_raw']; y = p['labels']; c = y[:, 0]+2*y[:, 1]
    cur, pred = p['current_branch'], q.argmax(1)
    need = (cur == 0)&(c == 2)
    missed = need&~p['switch_request_raw']
    source_kind = m['fault_metadata'][:, 1]
    available = (values[:, 8] <= .30)&(values[:, 10] >= .05)
    output = {'feature_columns': columns,
        'initial_frame_exact_replication_prefix_histogram': np.bincount(repeated_prefix, minlength=6).tolist(),
        'cohorts': {}}
    cohorts = [('current_VP_need_rescue', need), ('current_VP_missed_rescue', missed),
        ('current_VP_rescued', need&p['switch_request_raw']),
        ('VP_collapse_need_rescue', need&(source_kind == 7)),
        ('VP_bias_scale_shift_occlusion_need_rescue', need&np.isin(source_kind, [3, 4, 5, 6])),
        ('true_both_safe', c == 0), ('true_both_failure', c == 3),
        ('q3_false_alert_at_pr50', (c != 3)&(q[:, 3] >= .4340755045413971)),
        ('q3_missed_argmax', (c == 3)&(pred != 3))]
    names = ['lp_age', 'lp_interval', 'lp_valid', 'lp_abnormal', 'lp_temporal_l1', 'lp_frozen', 'lp_mean', 'lp_std',
        'vp_age', 'vp_interval', 'vp_valid', 'vp_abnormal', 'vp_temporal_l1', 'vp_frozen', 'vp_mean', 'vp_std',
        'latent_l2_gap', 'latent_cosine', 'velocity_gap_x', 'velocity_gap_y', 'velocity_gap_z', 'action_l2_gap', 'action_max_gap']
    for name, mask in cohorts:
        output['cohorts'][name] = {'rows': int(mask.sum()), 'vp_runtime_available': int((mask&available).sum()),
            'vp_full_valid_ratio_ge_0_95': int((mask&(values[:, 10] >= .95)).sum()),
            'all_five_frames_exactly_equal': int((mask&(repeated_prefix == 5)).sum()),
            'features_p10_p50_p90': {field: quantile(values[mask, k]) for k, field in enumerate(names)}}
    save('observable_diagnostic.json', output)
    np.savez_compressed(DEST/'validation_observable_summary.npz', sample_id=p['sample_id'],
        columns=np.array(columns), last_frame_features=values, repeated_prefix=repeated_prefix)
    print('observable diagnostics complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=('scores', 'training', 'observables'))
    part = parser.parse_args().part
    {'scores': score_diagnostics, 'training': training_distribution, 'observables': observable_diagnostics}[part]()
