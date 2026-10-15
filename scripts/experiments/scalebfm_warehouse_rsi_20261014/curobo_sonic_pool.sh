#!/usr/bin/env bash
# TODO 26.5.6 (plan §21): one SONIC + CuRobo start-state pool for the all-CuRobo redo -- sonic_pool.sh with the redo's
# configs, LANES collection lanes (default 1) on GPUS, then warehouse_states.py turns the rounds into the round files
# ScaleBFM starts from. The route service must already be up (route_service.sh). Nothing else on the machine.
#   Usage: [LANES=2 GPUS=0,1 NAME=run-name] bash curobo_sonic_pool.sh <experiment.yaml> <states-out-dir> [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
CFG=${1:?experiment yaml}
OUT=${2:?states output directory}
L=${3:-$OUT.log}
LANES=${LANES:-1}
GPUS=${GPUS:-0}
NAME=${NAME:-$(basename "$CFG" .yaml)}
RUN=outputs/runs/$NAME
mkdir -p "$(dirname "$L")" "$OUT"
[ -f "$CFG" ] || { echo "no config $CFG" >> "$L"; exit 1; }
echo "=== collect $CFG lanes $LANES gpus $GPUS $(date -u)" >> "$L"
bash scripts/start_stack.sh collect "$CFG" --gpu "$GPUS" --lanes "$LANES" --name "$NAME" >> "$L" 2>&1 &&
  bash scripts/bimanual/run_agent_server.sh "$CFG" --gpu "$GPUS" --name "$NAME" >> "$L" 2>&1
echo "=== collect done rc=$? $(date -u)" >> "$L"
for i in $(seq 0 $((LANES - 1))); do bash scripts/stop_stack.sh --instance "$i" >> "$L" 2>&1; done
"$VENV_SIM_PY" scripts/experiments/controller_compare_20261013/warehouse_states.py "$RUN" "$OUT" --program "$NAME" >> "$L" 2>&1
echo "=== states rc=$? $(ls "$OUT"/*.npz 2>/dev/null | wc -l) rounds $(date -u)" >> "$L"
echo "=== SONIC POOL DONE $(date -u)" >> "$L"
