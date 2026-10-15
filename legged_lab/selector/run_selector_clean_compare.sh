#!/usr/bin/env bash
# Two independent V5 outcome-training arms. Run with bash, never source it.
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
selector_checkpoint="${SELECTOR_CALIBRATED_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_recalibrated_20261013_111729/models/lp_vp_selector_checkpoint.pt}"
selector_baseline="${SELECTOR_BASELINE_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_baseline_train_20261013_100718/models/lp_vp_selector_checkpoint.pt}"
selector_run="${SELECTOR_COMPARE_RUN:-$selector_repo_root/outputs/g1_renet_selector_clean_compare_$(date +%Y%m%d_%H%M%S)}"
selector_batch="${SELECTOR_BATCH_SIZE:-4096}"
selector_workers="${SELECTOR_NUM_WORKERS:-4}"
selector_epochs="${SELECTOR_MAX_EPOCHS:-100}"
selector_patience="${SELECTOR_PATIENCE:-15}"

if (( $# != 0 )); then
    printf 'Usage: bash legged_lab/selector/run_selector_clean_compare.sh\n' >&2
    exit 2
fi
for selector_file in "$selector_checkpoint" "$selector_baseline" \
    "$selector_original/manifest.json" "$selector_clean/manifest.json" \
    "$selector_clean/original_manifest.json" "$selector_clean/filter_report.json" \
    "$selector_lp/manifest.json" "$selector_vp/manifest.json" "$selector_dual/manifest.json"; do
    if [[ ! -f "$selector_file" ]]; then
        printf 'Required file not found: %s\n' "$selector_file" >&2
        exit 1
    fi
done
if [[ -e "$selector_run" ]]; then
    printf 'Use a new comparison output directory: %s\n' "$selector_run" >&2
    exit 1
fi
mkdir -p "$selector_run"
printf '[clean compare] output: %s\n' "$selector_run"

"$selector_python" - "$selector_run" "$selector_original" "$selector_clean" \
    "$selector_checkpoint" "$selector_baseline" "$selector_lp" "$selector_vp" "$selector_dual" \
    "$selector_batch" "$selector_workers" "$selector_epochs" "$selector_patience" <<'PY'
import hashlib, json, sys
from pathlib import Path
run, original, clean, checkpoint, baseline, lp, vp, dual = map(Path,sys.argv[1:9])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data={"experiment":"two independent V5 outcome-network refits", "seeds":[42,43,44],
      "risk_network":"same frozen calibrated risk predictor in both arms",
      "clean_dataset":str(clean),"clean_manifest_sha256":sha(clean/'manifest.json'),
      "original_dataset":str(original),"original_manifest_sha256":sha(original/'manifest.json'),
      "risk_checkpoint":str(checkpoint),"risk_checkpoint_sha256":sha(checkpoint),
      "baseline_checkpoint":str(baseline),"baseline_checkpoint_sha256":sha(baseline),
      "supplements":[{"root":str(p),"manifest_sha256":sha(p/'manifest.json')} for p in (lp,vp,dual)],
      "batch_size":int(sys.argv[9]),"num_workers":int(sys.argv[10]),
      "max_epochs":int(sys.argv[11]),"patience":int(sys.argv[12]),"top_k":2,
      "validation_test":"same filtered validation/test in both arms; original test separately, reporting only",
      "arms":{"clean_only":[],"clean_plus_60k":[str(lp),str(vp),str(dual)]}}
(run/'experiment_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
PY

for selector_arm in clean_only clean_plus_60k; do
    selector_arm_root="$selector_run/$selector_arm"
    mkdir -p "$selector_arm_root/logs" "$selector_arm_root/evaluation"
    selector_supplements=()
    if [[ "$selector_arm" == clean_plus_60k ]]; then
        selector_supplements=(--supplement-datasets "$selector_lp" "$selector_vp" "$selector_dual")
    fi
    printf '\n[clean compare] starting %s; independent seeds=42,43,44\n' "$selector_arm"
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
        legged_lab/scripts/train_selector_clean_outcomes.py \
        --original-dataset "$selector_original" --dataset "$selector_clean" \
        "${selector_supplements[@]}" \
        --checkpoint "$selector_checkpoint" --baseline-checkpoint "$selector_baseline" \
        --output "$selector_arm_root/models" --device "$selector_device" \
        --batch-size "$selector_batch" --num-workers "$selector_workers" \
        --max-epochs "$selector_epochs" --patience "$selector_patience" \
        --top-k 2 --seeds 42 43 44 \
        --max-dangerous-rate 0.03 --max-both-safe-switch-rate 0.05 \
        --max-both-failure-switch-rate 0.10 --min-both-failure-recall 0.80 \
        --min-both-failure-precision 0.50 \
        2>&1 | tee "$selector_arm_root/logs/outcome_training.log"

    selector_selected="$("$selector_python" - "$selector_arm_root/models/outcome_training_report.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text()); p=Path(d['checkpoint'])
if not p.is_file():raise SystemExit('Selected checkpoint is missing')
print(p)
PY
)"
    # Exit 2 saves a quality-failure report; it must not prevent the second arm.
    set +e
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
        legged_lab/scripts/eval_selector.py \
        --dataset "$selector_original" --checkpoint "$selector_selected" \
        --split test --device "$selector_device" --batch-size "$selector_batch" \
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
    printf '[clean compare] completed %s; original-test evaluator exit=%s\n' \
        "$selector_arm" "${selector_eval_status[0]}"
done

"$selector_python" - "$selector_run" <<'PY'
import hashlib, json, sys
from pathlib import Path
run=Path(sys.argv[1]); experiment=json.loads((run/'experiment_manifest.json').read_text())
results={}
for arm in ('clean_only','clean_plus_60k'):
    root=run/arm
    report=json.loads((root/'models/outcome_training_report.json').read_text())
    natural=json.loads((root/'evaluation/original_test_report.json').read_text())
    sources=json.loads((root/'models/training_sources.json').read_text())
    chosen=Path(report['checkpoint'])
    digest=hashlib.sha256(chosen.read_bytes()).hexdigest()
    if digest!=report['checkpoint_sha256_before_test']:
        raise SystemExit('Selected checkpoint changed after selection/test reporting')
    if sources['base_manifest_sha256']!=experiment['clean_manifest_sha256']:
        raise SystemExit('Training dataset differs from the declared experiment')
    if sources['verified_original_manifest_sha256']!=experiment['original_manifest_sha256']:
        raise SystemExit('Original lineage differs from the declared experiment')
    expected=0 if arm=='clean_only' else 60000
    if sources['training_samples']!=sources['base_counts']['train']+expected:
        raise SystemExit('Unexpected augmentation sample count')
    results[arm]={"checkpoint":str(chosen),"checkpoint_sha256":digest,
        "accepted_on_validation":report['accepted_on_validation'],
        "training_samples":sources['training_samples'],"train_class_counts":report['train_class_counts'],
        "validation_guard":report['decision_guard'],
        "filtered_validation":report['validation']['candidate'],
        "filtered_test":report['test']['candidate'],"original_test":natural,
        "training_sources":str(root/'models/training_sources.json')}
summary={"completed":True,"experiment":experiment,"results":results,
         "automatic_winner":None,"test_used_for_selection":False,
         "note":"Both runs are retained for comparison. Failed validation guards remain diagnostic; "
                "no requirement is relaxed and no deployment is performed."}
(run/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('[clean compare] both training arms completed')
for arm,r in results.items():
    print(arm, 'training_samples=',r['training_samples'],'classes=',r['train_class_counts'],
          'accepted_on_validation=',r['accepted_on_validation'])
print('[clean compare] comparison:',run/'comparison.json')
PY
