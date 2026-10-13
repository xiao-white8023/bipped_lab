#!/usr/bin/env bash
# Run this file with bash. Strict options affect this child process only.
set -euo pipefail

selector_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$selector_repo_root"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_device="${SELECTOR_DEVICE:-cuda:0}"
selector_dataset="${SELECTOR_DATASET:-$selector_repo_root/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
selector_source="${SELECTOR_SOURCE_CHECKPOINT:-$selector_repo_root/outputs/g1_renet_selector_v4_baseline_train_20261013_100718/models/lp_vp_selector_checkpoint.pt}"
selector_run="${SELECTOR_RECAL_RUN:-$selector_repo_root/outputs/g1_renet_selector_v4_recalibrated_$(date +%Y%m%d_%H%M%S)}"
selector_batch="${SELECTOR_BATCH_SIZE:-8192}"
selector_workers="${SELECTOR_NUM_WORKERS:-4}"

if [[ ! -f "$selector_source" ]]; then
    printf 'Selector checkpoint not found: %s\n' "$selector_source" >&2
    exit 1
fi
if [[ ! -f "$selector_dataset/manifest.json" ]]; then
    printf 'Dataset manifest not found: %s\n' "$selector_dataset/manifest.json" >&2
    exit 1
fi
if ! command -v "$selector_python" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$selector_python" >&2
    exit 1
fi
if [[ -e "$selector_run/models" ]]; then
    printf 'Use a new output directory; models already exists: %s\n' "$selector_run/models" >&2
    exit 1
fi
mkdir -p "$selector_run/logs" "$selector_run/evaluation"
printf '[selector] source: %s\n[selector] dataset: %s\n[selector] output: %s\n' \
    "$selector_source" "$selector_dataset" "$selector_run"

CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
    legged_lab/scripts/recalibrate_selector.py \
    --dataset "$selector_dataset" --checkpoint "$selector_source" \
    --output "$selector_run/models" --device "$selector_device" \
    --batch-size "$selector_batch" --num-workers "$selector_workers" \
    2>&1 | tee "$selector_run/logs/recalibrate.log"

# Exit 2 means the report was saved but one or more quality gates failed.
# Retain it for diagnosis. Unexpected failures and log-write failures stop.
set +e
CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u \
    legged_lab/scripts/eval_selector.py \
    --dataset "$selector_dataset" \
    --checkpoint "$selector_run/models/lp_vp_selector_checkpoint.pt" \
    --split test --device "$selector_device" --batch-size "$selector_batch" \
    --output "$selector_run/evaluation/test_report.json" \
    2>&1 | tee "$selector_run/logs/eval_test.log"
selector_eval_status=("${PIPESTATUS[@]}")
set -e
if (( selector_eval_status[1] != 0 )); then
    printf 'Failed to write the evaluation log.\n' >&2
    exit 1
fi
if (( selector_eval_status[0] != 0 && selector_eval_status[0] != 2 )); then
    printf 'Evaluation failed with exit code %s.\n' "${selector_eval_status[0]}" >&2
    exit "${selector_eval_status[0]}"
fi
if (( selector_eval_status[0] == 2 )); then
    printf '[selector] Report saved; some quality gates did not pass.\n'
fi
printf '[selector] Comparison: %s\n[selector] Test report: %s\n' \
    "$selector_run/models/recalibration_report.json" "$selector_run/evaluation/test_report.json"
