#!/usr/bin/env bash
# Diagnose the same 4096-env LP source bank; no dataset or policy export is written.
set -euo pipefail
selector_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$selector_script_dir/../.."
export PYTHONUNBUFFERED=1
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"
selector_base="${SELECTOR_BASE:-$PWD/outputs/g1_renet_selector_v3_label_replay_full1048576/dataset}"
selector_terrain="${SELECTOR_TERRAIN:-$PWD/outputs/g1_renet_selector_v3_severity_103060/terrain_distribution.json}"
selector_checkpoint="${SELECTOR_CHECKPOINT:-/data/Tienkung_lab/logs/g1_renet/2026-09-10_19-25-08/model_60000.pt}"
selector_reference="${SELECTOR_REFERENCE:-$PWD/Exported_policy/policy.pt}"
selector_run="${SELECTOR_REPLAY_RUN:-$PWD/outputs/g1_renet_selector_replay_probe_$(date +%Y%m%d_%H%M%S)}"
selector_seed="$(( ${SELECTOR_SEED_BASE:-4100} + 1 ))"
for selector_path in "$selector_base/manifest.json" "$selector_terrain" "$selector_checkpoint" "$selector_reference"; do
    if [[ ! -f "$selector_path" ]]; then
        printf 'Missing required file: %s\n' "$selector_path" >&2
        exit 1
    fi
done
if [[ -e "$selector_run" ]]; then
    printf 'Output already exists; choose a new SELECTOR_REPLAY_RUN: %s\n' "$selector_run" >&2
    exit 1
fi
mkdir -p -- "$selector_run"
printf 'Replay probe output: %s\nGPU: %s; 4096 envs; four restored replay probes\n' "$selector_run" "$selector_gpu"
if CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u legged_lab/scripts/collect_selector_data.py \
    --headless --device cuda:0 --task g1_renet --terrain-mode main \
    --checkpoint-path "$selector_checkpoint" --reference-policy "$selector_reference" \
    --terrain-histogram "$selector_terrain" --base-dataset "$selector_base" \
    --num-envs 4096 --supplement-split train --fault-profile lp_map \
    --fault-severity-probabilities 0.10 0.30 0.60 --no-outcome-quotas \
    --seed "$selector_seed" --target-samples 8192 --replay-check-only \
    --skip-dual-branch-export --heartbeat-seconds 30 --output "$selector_run/dataset" \
    2>&1 | tee "$selector_run/replay.log"; then
    selector_collect_status=0
else
    selector_collect_status=$?
fi
if [[ ! -f "$selector_run/replay_diagnostics.json" ]]; then
    printf 'Replay report was not saved; inspect %s\n' "$selector_run/replay.log" >&2
    exit 1
fi
# Print the saved report even when the simulation correctly exits with failure.
if "$selector_python" -u legged_lab/scripts/inspect_selector_replay.py \
    --report "$selector_run/replay_diagnostics.json" --require-pass \
    2>&1 | tee "$selector_run/replay_summary.log"; then
    selector_summary_status=0
else
    selector_summary_status=$?
fi
if (( selector_collect_status != 0 )); then
    exit "$selector_collect_status"
fi
exit "$selector_summary_status"
