#!/usr/bin/env bash
# TODO 26.3.4 (docs/handoff/scalebfm-bottle-rsi-20261015.md §10): once the bottle loop in SANDBOX has stopped, export
# everything without a person: the sealed set for the start and the final incumbent (sealed.py), the comparison videos
# against SONIC (compare.py --task bottle) and the evolution videos from the ledger (evolution.py --task bottle).
# Waits for the loop's `stop` ledger entry first. Usage: bash finish.sh /workspace/code/rsi-phase4/bottle-1 [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
SB=${1:?sandbox}
L=${2:-outputs/scalebfm_bottle_rsi_20261015/finish.log}
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
H=scripts/experiments/scalebfm_bottle_rsi_20261015
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1
mkdir -p "$(dirname "$L")"
LED=$SB/outputs/scalebfm_bottle_rsi_20261015/loop/ledger.jsonl
echo "=== waiting for the loop to stop $(date -u)" >> "$L"
until [ -f "$LED" ] && grep -q '"step": "stop"' "$LED"; do sleep 60; done
grep '"step": "stop"' "$LED" | tail -1 >> "$L"
sleep 30  # the driver exits right after writing `stop`
echo "=== sealed $(date -u)" >> "$L"
$PY $H/sealed.py run "$SB" start final >> "$L" 2>&1 && $PY $H/sealed.py judge >> "$L" 2>&1
echo "=== compare $(date -u)" >> "$L"
$PY scripts/experiments/rsi_demo_20261015/compare.py --task bottle >> "$L" 2>&1
echo "=== evolution $(date -u)" >> "$L"
$PY scripts/experiments/rsi_demo_20261015/evolution.py "$SB" --loop-rel $H --task bottle \
  --out outputs/rsi_demo_20261015/bottle/evolution >> "$L" 2>&1
echo "=== FINISH DONE rc=$? $(date -u)" >> "$L"
