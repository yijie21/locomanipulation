#!/bin/bash
# Wall diagnostic through the collect stage: bash scripts/run_wall_pilot.sh [rounds] [seed] [extra overrides]
set -uo pipefail
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1
ROUNDS="${1:-10}"; SEED="${2:-0}"; shift 2 2>/dev/null || true
PYTHONPATH=src exec "$VENV_SIM_PY" -m locomani.cli collect task=crate_wall task.expert=wall embodiment=g1_bimanual \
  scene=crate_table30_depth04 collect.batch_rounds="$ROUNDS" collect.max_batches=1 collect.successes="$ROUNDS" \
  collect.seed0="$SEED" collect.record=false collect.instance="${INSTANCE:-1}" gpu="${GPU:-1}" name="wall_pilot_$SEED" "$@"
