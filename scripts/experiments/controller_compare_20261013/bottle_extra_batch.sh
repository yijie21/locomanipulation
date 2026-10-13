#!/bin/bash
# Bottle L1 extrapolation tier + B3, ScaleBFM side and the measurement (TODO 25.2.2 / 25.2.3). Needs the SONIC runs of
# sonic_bottle_extra.sh and sonic_b3_probe.sh first. Converts them, then: walks at 0.7 m/s and from the +-40 deg
# spawns, the four carries after a mode-6 replay of SONIC's production grasps, the B3 probe from one walked-in stance,
# and bottle_measure over production + extrapolation. Never beside a live SONIC stack. Usage: bash bottle_extra_batch.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=outputs/controller_compare_20261013/bottle
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
BFM_PY=third_party/Isaac-GR00T/.venv/bin/python
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
D=scripts/experiments/controller_compare_20261013
# SONIC: production expert variants and carry plans into round files
for v in fast yaw40; do
  $SIM_PY $D/bottle_states.py outputs/runs/controller_compare_bottle_sonic_${v}_20261013 "$OUT/sonic_$v/states" --program "production_$v"
done
for plan in back1 back2 turn90slow turn180slow; do
  $SIM_PY $D/bottle_states.py --plan outputs/runs/controller_compare_bottle_carry_${plan}_20261013 "$OUT/sonic_carry/states" "carry_$plan"
done
# ScaleBFM: B1 extrapolation (the SONIC variants' own spawns), 6 cm margin
$BFM_PY $D/bfm_bottle.py --program walk --margin 0.06 --walk-speed 0.7 --sonic "$OUT/sonic_fast/states" --out "$OUT/bfm_walk_fast"
$BFM_PY $D/bfm_bottle.py --program walk --margin 0.06 --tag _yaw40 --sonic "$OUT/sonic_yaw40/states" --out "$OUT/bfm_walk_yaw40"
# ScaleBFM: B7 extrapolation (after a mode-6 replay of SONIC's completed production grasps)
for plan in back1 back2 turn90slow turn180slow; do
  $BFM_PY $D/bfm_bottle.py --program replay --mode 6 --carry $plan --sonic "$OUT/sonic/states" --out "$OUT/bfm_carry_$plan"
done
# B3: ScaleBFM from its first 6 cm walk-in
$BFM_PY $D/bfm_b3_probe.py --walk "$(ls "$OUT"/bfm_walk_6cm/*.npz | head -1)" --out "$OUT/b3/scalebfm"
$SIM_PY $D/bottle_measure.py "$OUT/sonic_fast/states" "$OUT/sonic_yaw40/states" "$OUT/bfm_walk_fast" "$OUT/bfm_walk_yaw40" \
  "$OUT/sonic_carry/states" "$OUT"/bfm_carry_* --out "$OUT/extra"
python3 $D/b3_summary.py "$OUT/b3"
echo "=== BOTTLE EXTRA DONE"
