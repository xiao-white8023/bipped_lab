#!/usr/bin/env bash
# Usage: bash legged_lab/selector/run_selector_temporal_pilot.sh [lp_map vp_hard dual recovery]
set -euo pipefail

selector_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$selector_script_dir/../.."

export PYTHONUNBUFFERED=1
export SELECTOR_BASE="${SELECTOR_BASE:-$PWD/outputs/g1_renet_selector_v4_1s_normal/dataset}"
export SELECTOR_TERRAIN="${SELECTOR_TERRAIN:-$PWD/outputs/g1_renet_selector_v4_1s_normal/terrain_distribution.json}"
export SELECTOR_CHECKPOINT="${SELECTOR_CHECKPOINT:-/data/Tienkung_lab/logs/g1_renet/2026-09-10_19-25-08/model_60000.pt}"
export SELECTOR_REFERENCE="${SELECTOR_REFERENCE:-$PWD/Exported_policy/policy.pt}"
export SELECTOR_TEMPORAL_RUN="${SELECTOR_TEMPORAL_RUN:-$PWD/outputs/g1_renet_selector_temporal_pilot_$(date +%Y%m%d_%H%M%S)}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_python="${SELECTOR_PYTHON:-python}"
selector_seed_base="${SELECTOR_SEED_BASE:-4100}"

for selector_path in "$SELECTOR_BASE/manifest.json" "$SELECTOR_TERRAIN" "$SELECTOR_CHECKPOINT" "$SELECTOR_REFERENCE"; do
    if [[ ! -f "$selector_path" ]]; then
        printf 'Missing required file: %s\n' "$selector_path" >&2
        exit 1
    fi
done
if [[ -e "$SELECTOR_TEMPORAL_RUN" ]]; then
    printf 'Output already exists; choose a new SELECTOR_TEMPORAL_RUN: %s\n' "$SELECTOR_TEMPORAL_RUN" >&2
    exit 1
fi

selector_profiles=("$@")
if (( ${#selector_profiles[@]} == 0 )); then
    selector_profiles=(lp_map vp_hard dual recovery)
fi
for selector_profile in "${selector_profiles[@]}"; do
    case "$selector_profile" in
        lp_map|vp_hard|dual|recovery) ;;
        *) printf 'Unknown pilot profile: %s\n' "$selector_profile" >&2; exit 1 ;;
    esac
done
mkdir -p -- "$SELECTOR_TEMPORAL_RUN/logs"
printf 'Pilot output: %s\n' "$SELECTOR_TEMPORAL_RUN"
printf 'GPU: %s; real warmup: 15 s; sample span: at least 30 s\n' "$selector_gpu"

for selector_profile in "${selector_profiles[@]}"; do
    # Seeds depend on bank identity, even when banks are launched separately.
    # The training loader rejects reused fault seeds across supplemental roots.
    case "$selector_profile" in
        lp_map) selector_seed="$((selector_seed_base + 1))" ;;
        vp_hard) selector_seed="$((selector_seed_base + 2))" ;;
        dual) selector_seed="$((selector_seed_base + 3))" ;;
        recovery) selector_seed="$((selector_seed_base + 4))" ;;
    esac
    selector_severity=(0.10 0.30 0.60)
    if [[ "$selector_profile" == recovery ]]; then
        selector_severity=(0.45 0.40 0.15)
    fi
    selector_dataset="$SELECTOR_TEMPORAL_RUN/pilot_${selector_profile}/dataset"
    printf 'Starting pilot: %s\n' "$selector_profile"
    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u legged_lab/scripts/collect_selector_data.py \
        --headless --device cuda:0 --task g1_renet --terrain-mode main \
        --checkpoint-path "$SELECTOR_CHECKPOINT" \
        --reference-policy "$SELECTOR_REFERENCE" \
        --terrain-histogram "$SELECTOR_TERRAIN" \
        --base-dataset "$SELECTOR_BASE" \
        --skip-dual-branch-export \
        --num-envs 4096 --supplement-split train \
        --fault-profile "$selector_profile" \
        --fault-severity-probabilities "${selector_severity[@]}" \
        --no-outcome-quotas --seed "$selector_seed" \
        --target-samples 8192 \
        --warmup-seconds 15 --min-collection-span-seconds 30 \
        --label-interval-frames 10 --randomize-label-phase --max-samples-per-labelled-batch 256 \
        --max-samples-per-event 8 --min-event-spacing-s 0.2 \
        --max-candidate-batches 5000 --max-no-progress-batches 1000 \
        --heartbeat-seconds 30 \
        --output "$selector_dataset" \
        2>&1 | tee "$SELECTOR_TEMPORAL_RUN/logs/pilot_${selector_profile}.log"

    # Check the saved result independently of Kit's exit status. No next bank
    # starts after a partial collection or a missing analysis report.
    "$selector_python" - "$selector_dataset" <<'PY' | tee "$SELECTOR_TEMPORAL_RUN/logs/verify_${selector_profile}.log"
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
report = json.loads((root / "collection_analysis.json").read_text())
assert manifest["metadata"].get("collection_complete"), "Collection incomplete"
assert manifest["counts"]["train"] == 8192, "Wrong training row count"
assert manifest["counts"]["validation"] == manifest["counts"]["test"] == 0, "Unexpected split"
temporal = report["supplement_coverage"]["temporal_coverage"]
assert temporal["collection_elapsed_s"] + 1e-6 >= 30, "Temporal span missing"
assert sum(temporal["written"]) == 8192, "Temporal audit misaligned"
print("Verified:", root)
print("1.0s:", report["overall"]["pair_counts"][0])
print("Real sample span:", temporal["collection_elapsed_s"])
print("Written stages:", dict(zip(temporal["lifecycle_order"], temporal["written"])))
print("Missing observed stages:", temporal["missing_observed_stages"])
PY
done
printf 'Pilot collection finished: %s\n' "$SELECTOR_TEMPORAL_RUN"
printf 'Review outcomes and stage coverage before formal quota collection.\n'
