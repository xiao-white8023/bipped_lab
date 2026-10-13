#!/usr/bin/env bash
# Strict options apply to this child process, not the interactive terminal.
set -euo pipefail
selector_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$selector_repo_root"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_dataset="${SELECTOR_DATASET:-$selector_repo_root/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
selector_checkpoint="${SELECTOR_CALIBRATED_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_recalibrated_20261013_111729/models/lp_vp_selector_checkpoint.pt}"
selector_baseline="${SELECTOR_BASELINE_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_baseline_train_20261013_100718/models/lp_vp_selector_checkpoint.pt}"
selector_run="${SELECTOR_DECISION_RUN:-$selector_repo_root/outputs/g1_renet_selector_v4_decision_$(date +%Y%m%d_%H%M%S)}"
for selector_file in "$selector_checkpoint" "$selector_baseline" "$selector_dataset/manifest.json"; do
    if [[ ! -f "$selector_file" ]]; then
        printf 'Required file not found: %s\n' "$selector_file" >&2
        exit 1
    fi
done
if [[ -e "$selector_run/models" ]]; then
    printf 'Use a new output directory: %s\n' "$selector_run" >&2
    exit 1
fi
mkdir -p "$selector_run/logs"
printf '[selector decision] output: %s\n' "$selector_run"
CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
    legged_lab/scripts/tune_selector_decision.py \
    --dataset "$selector_dataset" --checkpoint "$selector_checkpoint" \
    --baseline-checkpoint "$selector_baseline" --output "$selector_run/models" \
    --device "${SELECTOR_DEVICE:-cuda:0}" \
    --batch-size "${SELECTOR_BATCH_SIZE:-8192}" --num-workers "${SELECTOR_NUM_WORKERS:-4}" \
    2>&1 | tee "$selector_run/logs/decision_tuning.log"
printf '[selector decision] comparison report: %s\n' "$selector_run/models/decision_tuning_report.json"
