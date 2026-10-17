#!/usr/bin/env bash
# Fresh branch-aware joint training; select either single-stream control explicitly.
set -euo pipefail
selector_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$selector_repo"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_clean="${SELECTOR_CLEAN_DATASET:-$selector_repo/outputs/g1_renet_selector_cross_clean_20261015_164009/dataset}"
selector_original="${SELECTOR_ORIGINAL_DATASET:-$selector_repo/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
selector_architecture="${SELECTOR_ARCHITECTURE:-branch_gru}"
selector_output="${SELECTOR_JOINT_OUTPUT:-$selector_repo/outputs/g1_renet_selector_joint_${selector_architecture}_$(date +%Y%m%d_%H%M%S)}"
selector_layers="${SELECTOR_GRU_LAYERS:-auto}"
if [[ "$selector_layers" == auto ]]; then
    case "$selector_architecture" in
        branch_gru) selector_layers=1 ;;
        single_gru|single_gru_readout) selector_layers=3 ;;
        *) printf 'Unknown architecture: %s\n' "$selector_architecture" >&2; exit 1 ;;
    esac
fi
read -r -a selector_seeds <<< "${SELECTOR_SEEDS:-42 43 44}"
selector_supplements=()
selector_extra_args=()
if (( $# != 0 )); then
    printf 'Usage: bash legged_lab/selector/run_selector_joint_training.sh\n' >&2
    exit 2
fi
if [[ "${SELECTOR_USE_SUPPLEMENTS:-1}" == 1 ]]; then
    if [[ -n "${SELECTOR_SUPPLEMENTS_FILE:-}" ]]; then
        while IFS= read -r selector_path || [[ -n "$selector_path" ]]; do
            selector_path="${selector_path%$'\r'}"
            [[ -z "$selector_path" || "$selector_path" == \#* ]] && continue
            selector_supplements+=("$selector_path")
        done < "$SELECTOR_SUPPLEMENTS_FILE"
        if (( ${#selector_supplements[@]} == 0 )); then
            printf 'Supplement list is empty.\n' >&2
            exit 1
        fi
    else
        selector_supplements=(
          "$selector_repo/outputs/g1_renet_selector_directed_20k_20261014_104733/lp_failure/dataset"
          "$selector_repo/outputs/g1_renet_selector_directed_remaining_20261014_171524/vp_failure/dataset"
          "$selector_repo/outputs/g1_renet_selector_directed_dual_20k_20261015_082634/both_failure/dataset"
        )
    fi
fi
if (( ${#selector_supplements[@]} )); then
    selector_extra_args+=(--supplement-datasets "${selector_supplements[@]}")
fi
if [[ "${SELECTOR_HISTORY_FRAMES:-auto}" != auto ]]; then
    selector_extra_args+=(--history-frames "$SELECTOR_HISTORY_FRAMES")
fi
if [[ -n "${SELECTOR_TRAINING_CACHE_DIR:-}" ]]; then
    selector_extra_args+=(--training-cache-dir "$SELECTOR_TRAINING_CACHE_DIR")
fi
if [[ "${SELECTOR_VERIFY_FILTERED_LINEAGE:-1}" == 1 ]]; then
    selector_extra_args+=(--original-dataset "$selector_original")
fi
if [[ -e "$selector_output" ]]; then
    printf 'Choose a new training output directory: %s\n' "$selector_output" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u legged_lab/scripts/train_selector_joint.py \
    --dataset "$selector_clean" --output "$selector_output" \
    --architecture "$selector_architecture" --gru-layers "$selector_layers" \
    --gru-dropout "${SELECTOR_GRU_DROPOUT:-0.1}" \
    --decision-probability-mode "${SELECTOR_PROBABILITY_MODE:-raw}" \
    --marginal-weight "${SELECTOR_MARGINAL_WEIGHT:-1}" \
    --class-weighting "${SELECTOR_CLASS_WEIGHTING:-sqrt_inverse_frequency}" \
    --seeds "${selector_seeds[@]}" --device "${SELECTOR_DEVICE:-cuda:0}" \
    --batch-size "${SELECTOR_BATCH_SIZE:-4096}" --num-workers "${SELECTOR_NUM_WORKERS:-4}" \
    --max-epochs "${SELECTOR_MAX_EPOCHS:-100}" --patience "${SELECTOR_PATIENCE:-15}" \
    "${selector_extra_args[@]}"
