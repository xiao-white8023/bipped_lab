#!/usr/bin/env bash
# Run as a child process so a failed preflight never exits the user's shell.
set -euo pipefail
selector_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$selector_repo_root"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_dataset="${SELECTOR_DATASET:-$selector_repo_root/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
selector_checkpoint="${SELECTOR_CALIBRATED_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_recalibrated_20261013_111729/models/lp_vp_selector_checkpoint.pt}"
selector_baseline="${SELECTOR_BASELINE_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_baseline_train_20261013_100718/models/lp_vp_selector_checkpoint.pt}"
selector_run="${SELECTOR_OUTCOME_RUN:-$selector_repo_root/outputs/g1_renet_selector_v5_outcomes_$(date +%Y%m%d_%H%M%S)}"
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
printf '[selector outcomes] output: %s\n' "$selector_run"
CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
    legged_lab/scripts/train_selector_outcomes.py \
    --dataset "$selector_dataset" --checkpoint "$selector_checkpoint" \
    --baseline-checkpoint "$selector_baseline" --output "$selector_run/models" \
    --device "${SELECTOR_DEVICE:-cuda:0}" \
    --batch-size "${SELECTOR_BATCH_SIZE:-4096}" --num-workers "${SELECTOR_NUM_WORKERS:-4}" \
    --max-epochs "${SELECTOR_MAX_EPOCHS:-100}" --patience "${SELECTOR_PATIENCE:-15}" \
    --top-k 2 --seeds 42 43 44 \
    --max-dangerous-rate 0.03 \
    --max-both-safe-switch-rate 0.05 --max-both-failure-switch-rate 0.10 \
    --min-both-failure-recall 0.80 --min-both-failure-precision 0.50 \
    2>&1 | tee "$selector_run/logs/outcome_training.log"
printf '[selector outcomes] report: %s\n' "$selector_run/models/outcome_training_report.json"
if [[ -f "$selector_run/models/lp_vp_selector_checkpoint.pt" ]]; then
    printf '[selector outcomes] validation accepted; inspect test report: %s\n' "$selector_run/models/lp_vp_selector_checkpoint.pt"
else
    printf '[selector outcomes] no accepted model; diagnostic checkpoint only: %s\n' "$selector_run/models/candidate_selector_checkpoint.pt"
fi
