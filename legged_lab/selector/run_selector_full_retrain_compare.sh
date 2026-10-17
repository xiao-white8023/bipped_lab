#!/usr/bin/env bash
# Two independent full selector refits: fresh risks, then fresh four-outcome heads.
set -euo pipefail

selector_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$selector_repo_root"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_device="${SELECTOR_DEVICE:-cuda:0}"
selector_original="${SELECTOR_ORIGINAL_DATASET:-$selector_repo_root/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
selector_clean="${SELECTOR_CLEAN_DATASET:-$selector_repo_root/outputs/g1_renet_selector_cross_clean_20261015_164009/dataset}"
selector_lp="${SELECTOR_LP_DATASET:-$selector_repo_root/outputs/g1_renet_selector_directed_20k_20261014_104733/lp_failure/dataset}"
selector_vp="${SELECTOR_VP_DATASET:-$selector_repo_root/outputs/g1_renet_selector_directed_remaining_20261014_171524/vp_failure/dataset}"
selector_dual="${SELECTOR_DUAL_DATASET:-$selector_repo_root/outputs/g1_renet_selector_directed_dual_20k_20261015_082634/both_failure/dataset}"
selector_run="${SELECTOR_FULL_RETRAIN_RUN:-$selector_repo_root/outputs/g1_renet_selector_full_retrain_compare_$(date +%Y%m%d_%H%M%S)}"
selector_batch="${SELECTOR_BATCH_SIZE:-4096}"
selector_workers="${SELECTOR_NUM_WORKERS:-4}"
selector_risk_epochs="${SELECTOR_RISK_MAX_EPOCHS:-${SELECTOR_MAX_EPOCHS:-100}}"
selector_outcome_epochs="${SELECTOR_OUTCOME_MAX_EPOCHS:-${SELECTOR_MAX_EPOCHS:-100}}"
selector_patience="${SELECTOR_PATIENCE:-15}"

if (( $# != 0 )); then
    printf 'Usage: bash legged_lab/selector/run_selector_full_retrain_compare.sh\n' >&2
    exit 2
fi
for selector_file in "$selector_original/manifest.json" "$selector_clean/manifest.json" \
    "$selector_clean/original_manifest.json" "$selector_clean/filter_report.json" \
    "$selector_lp/manifest.json" "$selector_vp/manifest.json" "$selector_dual/manifest.json" \
    legged_lab/scripts/train_selector_clean_risks.py legged_lab/scripts/train_selector_full_outcomes.py \
    legged_lab/scripts/train_selector_clean_outcomes.py; do
    if [[ ! -f "$selector_file" ]]; then
        printf 'Required file not found: %s\n' "$selector_file" >&2
        exit 1
    fi
done
if [[ -e "$selector_run" ]]; then
    printf 'Use a new full-retraining output directory: %s\n' "$selector_run" >&2
    exit 1
fi
mkdir -p "$selector_run"
printf '[full retrain] output: %s\n' "$selector_run"

"$selector_python" - "$selector_run" "$selector_original" "$selector_clean" \
    "$selector_lp" "$selector_vp" "$selector_dual" "$selector_batch" "$selector_workers" \
    "$selector_risk_epochs" "$selector_outcome_epochs" "$selector_patience" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
run, original, clean, lp, vp, dual = map(Path, sys.argv[1:7])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
files = ('legged_lab/scripts/train_selector_clean_risks.py',
         'legged_lab/scripts/train_selector_full_outcomes.py',
         'legged_lab/scripts/train_selector_clean_outcomes.py',
         'legged_lab/selector/run_selector_full_retrain_compare.sh')
git = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
data = {
    'experiment': 'two independent full selector retrainings: random risk network, then new outcome network',
    'git_commit': git.stdout.strip() if git.returncode == 0 else None,
    'source_sha256': {name: sha(Path(name)) for name in files},
    'risk_seeds': [42, 43, 44], 'outcome_seeds': [42, 43, 44], 'outcome_top_k': 2,
    'risk_initial_checkpoint': None, 'risk_initialization': 'random in each arm',
    'normalizer': 'refitted independently from each arm training rows',
    'risk_calibration': 'refitted from the shared filtered validation split',
    'outcome_stage': 'new outcome encoder/head; its own newly trained risk network is held fixed',
    'validation_reference': 'each arm newly trained calibrated risk-only predictor, before outcome decisions',
    'original_dataset': str(original), 'original_manifest_sha256': sha(original/'manifest.json'),
    'clean_dataset': str(clean), 'clean_manifest_sha256': sha(clean/'manifest.json'),
    'supplements': [{'root': str(p), 'manifest_sha256': sha(p/'manifest.json')} for p in (lp, vp, dual)],
    'batch_size': int(sys.argv[7]), 'num_workers': int(sys.argv[8]),
    'risk_max_epochs': int(sys.argv[9]), 'outcome_max_epochs': int(sys.argv[10]),
    'patience': int(sys.argv[11]),
    'validation_test': 'same filtered validation/test in both arms; original test separately, reporting only',
    'arms': {'clean_only': [], 'clean_plus_60k': [str(lp), str(vp), str(dual)]},
}
(run/'experiment_manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
PY

for selector_arm in clean_only clean_plus_60k; do
    selector_arm_root="$selector_run/$selector_arm"
    mkdir -p "$selector_arm_root/logs" "$selector_arm_root/evaluation"
    selector_supplements=()
    if [[ "$selector_arm" == clean_plus_60k ]]; then
        selector_supplements=(--supplement-datasets "$selector_lp" "$selector_vp" "$selector_dual")
    fi
    printf '\n[full retrain] %s: stage 1, randomly initialized risk network\n' "$selector_arm"
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
        legged_lab/scripts/train_selector_clean_risks.py \
        --original-dataset "$selector_original" --dataset "$selector_clean" \
        "${selector_supplements[@]}" --output "$selector_arm_root/risks" \
        --device "$selector_device" --batch-size "$selector_batch" --num-workers "$selector_workers" \
        --max-epochs "$selector_risk_epochs" --patience "$selector_patience" --seeds 42 43 44 \
        2>&1 | tee "$selector_arm_root/logs/risk_training.log"

    selector_new_risk="$("$selector_python" - "$selector_arm_root/risks/risk_training_report.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
path = Path(report['checkpoint'])
if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != report['checkpoint_sha256']:
    raise SystemExit('New risk checkpoint is missing or changed')
if report['risk_training'].get('risk_network_retrained') is not True:
    raise SystemExit('Stage one did not record risk retraining')
print(path)
PY
)"
    printf '[full retrain] %s: stage 2, new outcome network using %s\n' "$selector_arm" "$selector_new_risk"
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
        legged_lab/scripts/train_selector_full_outcomes.py \
        --original-dataset "$selector_original" --dataset "$selector_clean" \
        "${selector_supplements[@]}" \
        --checkpoint "$selector_new_risk" --baseline-checkpoint "$selector_new_risk" \
        --output "$selector_arm_root/models" --device "$selector_device" \
        --batch-size "$selector_batch" --num-workers "$selector_workers" \
        --max-epochs "$selector_outcome_epochs" --patience "$selector_patience" --top-k 2 --seeds 42 43 44 \
        --max-dangerous-rate 0.03 --max-both-safe-switch-rate 0.05 \
        --max-both-failure-switch-rate 0.10 --min-both-failure-recall 0.80 \
        --min-both-failure-precision 0.50 \
        2>&1 | tee "$selector_arm_root/logs/outcome_training.log"

    selector_selected="$("$selector_python" - "$selector_arm_root/models/outcome_training_report.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
path = Path(report['checkpoint'])
if not path.is_file():
    raise SystemExit('Selected outcome checkpoint is missing')
print(path)
PY
)"
    set +e
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
        legged_lab/scripts/eval_selector.py \
        --dataset "$selector_original" --checkpoint "$selector_selected" --split test \
        --device "$selector_device" --batch-size "$selector_batch" \
        --output "$selector_arm_root/evaluation/original_test_report.json" \
        2>&1 | tee "$selector_arm_root/logs/eval_original_test.log"
    selector_eval_status=("${PIPESTATUS[@]}")
    set -e
    if (( selector_eval_status[1] != 0 )); then
        printf 'Failed to save original-test evaluation log.\n' >&2
        exit 1
    fi
    if (( selector_eval_status[0] != 0 && selector_eval_status[0] != 2 )); then
        exit "${selector_eval_status[0]}"
    fi
    printf '[full retrain] completed %s; original-test evaluator exit=%s\n' \
        "$selector_arm" "${selector_eval_status[0]}"
done

"$selector_python" - "$selector_run" <<'PY'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
experiment = json.loads((run/'experiment_manifest.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for root, expected in [(experiment['clean_dataset'], experiment['clean_manifest_sha256']),
                       (experiment['original_dataset'], experiment['original_manifest_sha256']),
                       *[(s['root'], s['manifest_sha256']) for s in experiment['supplements']]]:
    if sha(Path(root)/'manifest.json') != expected:
        raise SystemExit('A dataset manifest changed during full retraining')
for name, expected in experiment['source_sha256'].items():
    if sha(Path(name)) != expected:
        raise SystemExit('A training entry point changed during full retraining')
results = {}
for arm in ('clean_only', 'clean_plus_60k'):
    root = run/arm
    risk = json.loads((root/'risks/risk_training_report.json').read_text())
    report = json.loads((root/'models/outcome_training_report.json').read_text())
    sources = json.loads((root/'models/training_sources.json').read_text())
    natural = json.loads((root/'evaluation/original_test_report.json').read_text())
    checkpoint = Path(report['checkpoint'])
    risk_checkpoint = Path(risk['checkpoint'])
    if sha(checkpoint) != report['checkpoint_sha256_before_test']:
        raise SystemExit('Outcome checkpoint changed after selection/test reporting')
    if (sha(risk_checkpoint) != risk['checkpoint_sha256']
            or report['risk_pretraining_checkpoint_sha256'] != risk['checkpoint_sha256']
            or Path(report['risk_pretraining_checkpoint']).resolve() != risk_checkpoint.resolve()
            or risk_checkpoint.parent.resolve() != (root/'risks').resolve()):
        raise SystemExit('Outcome stage did not retain its own new risk checkpoint unchanged')
    if (report.get('risk_network_retrained_this_experiment') is not True
            or risk['risk_training']['initial_checkpoint'] is not None
            or risk['risk_training']['initialization'] != 'random'
            or risk.get('test_used_for_selection') is not False
            or report.get('test_used_for_selection') is not False):
        raise SystemExit('Full retraining or validation-only selection proof is missing')
    for candidate in risk['all_candidates']:
        if not all(candidate['updated_components'].get(key, 0) > 0
                   for key in ('frame_encoder', 'temporal', 'head')):
            raise SystemExit('A risk seed did not update the encoder, GRU and output heads')
    if (sources['base_manifest_sha256'] != experiment['clean_manifest_sha256']
            or sources['verified_original_manifest_sha256'] != experiment['original_manifest_sha256']):
        raise SystemExit('Training data differs from the declared experiment')
    expected = sources['base_counts']['train'] + (0 if arm == 'clean_only' else 60000)
    if sources['training_samples'] != expected or risk['risk_training']['training_samples'] != expected:
        raise SystemExit('Risk/outcome stages used unexpected training counts')
    results[arm] = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'risk_checkpoint': str(risk_checkpoint), 'risk_checkpoint_sha256': sha(risk_checkpoint),
        'risk_training': risk, 'accepted_on_validation': report['accepted_on_validation'],
        'training_samples': expected, 'train_class_counts': report['train_class_counts'],
        'validation_guard': report['decision_guard'],
        'filtered_validation': report['validation']['candidate'],
        'filtered_test': report['test']['candidate'], 'original_test': natural,
        'training_sources': str(root/'models/training_sources.json'),
    }
summary = {'completed': True, 'experiment': experiment, 'results': results,
           'automatic_winner': None, 'test_used_for_selection': False,
           'note': 'Both arms refit risks, normalization, calibration and outcome networks. '
                   'Each guard uses its own retrained risk-only reference. Failed guards stay '
                   'diagnostic; no requirement is relaxed and no deployment is performed.'}
(run/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
for arm, result in results.items():
    print(arm, 'training_samples=', result['training_samples'], 'classes=', result['train_class_counts'],
          'risk_seed=', result['risk_training']['selected']['seed'],
          'accepted_on_validation=', result['accepted_on_validation'])
print('[full retrain] both arms completed; comparison:', run/'comparison.json')
PY
