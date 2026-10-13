#!/bin/bash
# Bottle L1 batch, ScaleBFM side + measurement (TODO 25.2). Needs the SONIC round files first:
#   $SIM_PY bottle_states.py outputs/runs/controller_compare_bottle_sonic_20261013 $OUT/sonic/states
# Then: walk (B1 B2 B8) and replay in modes 1 / 2 / 6 / 7 (B4 B5 B6), then bottle_measure over everything.
# ScaleBFM runs are CPU-bound torch: never beside a live SONIC stack. Usage: bash bottle_batch.sh [outdir]
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=${1:-outputs/controller_compare_20261013/bottle}
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
BFM_PY=third_party/Isaac-GR00T/.venv/bin/python
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
D=scripts/experiments/controller_compare_20261013
for margin in 0.03 0.06; do  # SONIC's guard margin, and the margin ScaleBFM needs never to cross (hands swing ~3 cm)
  $BFM_PY $D/bfm_bottle.py --program walk --margin $margin --sonic "$OUT/sonic/states" --out "$OUT/bfm_walk_${margin#0.0}cm"
done
for mode in 2 1 6 7; do
  $BFM_PY $D/bfm_bottle.py --program replay --mode $mode --sonic "$OUT/sonic/states" --out "$OUT/bfm_replay_m$mode"
done
$SIM_PY $D/bottle_measure.py "$OUT/sonic/states" "$OUT"/bfm_walk_* "$OUT"/bfm_replay_m* --out "$OUT"
echo "=== BOTTLE L1 DONE"
