#!/usr/bin/env bash
# TODO 26.3.2 (user 2026-10-15): SONIC runs the bottle RSI acceptance or sealed start-state pool on instance 0 / GPU 0 with
# nothing else on the machine (recorded, so the pose sidecar exists), then bottle_states.py turns the run into round
# files. A watchdog stops the stack when the collect log has not changed for 12 min (a hung round; L2 lost two that way).
# Usage: bash sonic_pool.sh accept|sealed [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
POOL=${1:?accept or sealed}
O=outputs/scalebfm_bottle_rsi_20261015_hidden
L=${2:-$O/sonic_$POOL.log}
CFG=configs/experiments/scalebfm_bottle_rsi_${POOL}_sonic_20261015.yaml
RUN=outputs/runs/scalebfm_bottle_rsi_${POOL}_sonic_20261015
mkdir -p "$O"
echo "=== collect $CFG $(date -u)" >> "$L"
( while sleep 60; do
    f=$(ls -t $RUN/collect_00.log 2>/dev/null | head -1); [ -n "$f" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ "$age" -gt 720 ]; then echo "=== watchdog: $f silent ${age}s, stopping the stack $(date -u)" >> "$L"; bash scripts/stop_stack.sh --instance 0 >> "$L" 2>&1; break; fi
  done ) &
WD=$!
bash scripts/start_stack.sh collect "$CFG" --gpu 0 >> "$L" 2>&1 && bash scripts/single_arm/run_agent_server.sh "$CFG" --gpu 0 >> "$L" 2>&1
echo "=== collect done rc=$? $(date -u)" >> "$L"
kill $WD 2>/dev/null
bash scripts/stop_stack.sh --instance 0 >> "$L" 2>&1
"$VENV_SIM_PY" scripts/experiments/controller_compare_20261013/bottle_states.py "$RUN" "$O/pool_$POOL/states" --program pool_$POOL >> "$L" 2>&1
echo "=== states rc=$? $(ls $O/pool_$POOL/states/*.npz 2>/dev/null | wc -l) rounds $(date -u)" >> "$L"
echo "=== SONIC POOL DONE $(date -u)" >> "$L"
