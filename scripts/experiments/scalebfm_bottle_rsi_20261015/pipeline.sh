#!/usr/bin/env bash
# TODO 26.3 (user 2026-10-15: run until every video is exported): the rest of the bottle instance without a person, each
# step waiting for the one before:
#   1. the acceptance pool: the first batch's sidecar rounds are in place; convert the unrecorded k1 batch
#      (trace_states.py, prefix sonicb) and keep the first 60 usable starts (prep_pools.py accept);
#   2. once the sealed pool's SONIC runs are done: convert its batches (k1 sonic, k2 sonicb, ...), keep 80;
#   3. the sandbox (sandbox.py), a stub iteration in it (plumbing + the incumbent's diagnostic rounds);
#   4. the loop, cold, arm A design, both GPUs, 12 ScaleBFM processes;
#   5. finish.sh: after the loop stops, the sealed set, the comparison and the evolution videos.
# Usage: bash pipeline.sh [sandbox] [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
SB=${1:-/workspace/code/rsi-phase4/bottle-1}
L=${2:-outputs/scalebfm_bottle_rsi_20261015/pipeline.log}
H=scripts/experiments/scalebfm_bottle_rsi_20261015
O=outputs/scalebfm_bottle_rsi_20261015_hidden
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
mkdir -p "$(dirname "$L")"
say() { echo "=== $* $(date -u)" >> "$L"; }

say "1. acceptance pool: waiting for the accept runs to stop"
until grep -q 'accept stopped by switch_to_sealed\|SONIC POOL accept DONE' $O/sonic_accept.log 2>/dev/null; do sleep 30; done
$PY $H/trace_states.py outputs/runs/scalebfm_bottle_rsi_accept_k1_sonic_20261015 $O/pool_accept/states \
  --program pool_accept --label-prefix sonicb >> "$L" 2>&1
python3 $H/prep_pools.py accept >> "$L" 2>&1
say "acceptance pool: $(ls $O/pool_accept/states/*.npz | wc -l) starts"

say "2. sealed pool: waiting for its SONIC runs"
until grep -q 'SONIC POOL sealed DONE' $O/sonic_sealed.log 2>/dev/null; do sleep 60; done
k=1
for prefix in sonic sonicb sonicc sonicd sonice; do
  R=outputs/runs/scalebfm_bottle_rsi_sealed_k${k}_sonic_20261015
  [ -d "$R" ] || break
  $PY $H/trace_states.py "$R" $O/pool_sealed/states --program pool_sealed --label-prefix $prefix >> "$L" 2>&1
  k=$((k + 1))
done
python3 $H/prep_pools.py sealed >> "$L" 2>&1
say "sealed pool: $(ls $O/pool_sealed/states/*.npz | wc -l) starts"

say "3. sandbox $SB and a stub iteration"
python3 $H/sandbox.py "$SB" >> "$L" 2>&1 || { say "sandbox failed"; exit 1; }
( cd "$SB" && $PY $H/loop.py run --stub --max-iters 1 --state "$SB/outputs/scalebfm_bottle_rsi_20261015/loop_stub" \
    --arm A --gpu 0,1 --procs 12 >> "$L" 2>&1 )
if ! grep -q '"step": "gate"' "$SB/outputs/scalebfm_bottle_rsi_20261015/loop_stub/ledger.jsonl" 2>/dev/null; then
  say "stub iteration did not reach the gate: stopping here"; exit 1
fi
say "stub ok: diagnostic incumbent $(ls $SB/outputs/scalebfm_bottle_rsi_20261015/diag/*/ | grep -c 'bfm_.*json') rounds"

say "4. the loop"
( cd "$SB" && nohup setsid $PY $H/loop.py run --arm A --gpu 0,1 --procs 12 \
    > "$SB/outputs/scalebfm_bottle_rsi_20261015/driver.log" 2>&1 < /dev/null & )
sleep 30
say "5. finish.sh waits for the loop to stop"
bash $H/finish.sh "$SB" outputs/scalebfm_bottle_rsi_20261015/finish.log
say "PIPELINE DONE"
