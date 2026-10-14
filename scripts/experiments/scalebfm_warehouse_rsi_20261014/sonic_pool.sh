#!/usr/bin/env bash
# TODO 25.9.1 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §4): SONIC runs one start-state pool on instance 0 / GPU 0
# with nothing else on the machine, then its rounds become warehouse_states.py round files -- the start states ScaleBFM
# begins from. Usage: bash sonic_pool.sh accept|sealed [log]
#   accept: seeds 49000.., 60 rounds (phase 0);  sealed: seeds 48000.., 80 rounds (phase 3 only)
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
POOL=${1:?accept or sealed}
O=outputs/scalebfm_warehouse_rsi_20261014_hidden
L=${2:-$O/sonic_$POOL.log}
CFG=configs/experiments/scalebfm_warehouse_rsi_${POOL}_sonic_20261014.yaml
RUN=outputs/runs/scalebfm_warehouse_rsi_${POOL}_sonic_20261014
mkdir -p "$O"
[ -f "$CFG" ] || { echo "no config $CFG" >> "$L"; exit 1; }
echo "=== collect $CFG $(date -u)" >> "$L"
bash scripts/start_stack.sh collect "$CFG" --gpu 0 --lanes 1 >> "$L" 2>&1 && bash scripts/bimanual/run_agent_server.sh "$CFG" --gpu 0 >> "$L" 2>&1
echo "=== collect done rc=$? $(date -u)" >> "$L"
bash scripts/stop_stack.sh --instance 0 >> "$L" 2>&1
"$VENV_SIM_PY" scripts/experiments/controller_compare_20261013/warehouse_states.py "$RUN" "$O/pool_$POOL/states" --program pool_$POOL >> "$L" 2>&1
echo "=== states rc=$? $(ls $O/pool_$POOL/states/*.npz 2>/dev/null | wc -l) rounds $(date -u)" >> "$L"
echo "=== SONIC POOL DONE $(date -u)" >> "$L"
