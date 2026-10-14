#!/usr/bin/env bash
# Same source/replay/spacing acceptance as formal collection, 10,000 rows/bank.
set -euo pipefail
selector_pilot_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SELECTOR_COLLECTION_PHASE=pilot
export SELECTOR_SEED_BASE="${SELECTOR_SEED_BASE:-8100}"
exec bash "$selector_pilot_dir/run_selector_outcome_supplement.sh" "$@"
