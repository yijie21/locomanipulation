#!/bin/bash
# Warehouse L1 (TODO 25.3), everything after the SONIC runs (sonic_warehouse.sh with the production, edges and kg24
# configs; sonic_warehouse_plans.sh, with and without the mars_ prefix): converts SONIC's runs, runs every ScaleBFM program and measures both sides into
# the production and the extrapolation tables. Never beside a live SONIC stack. Usage: bash warehouse_batch.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=outputs/controller_compare_20261013/warehouse
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
BFM_PY=third_party/Isaac-GR00T/.venv/bin/python
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
D=scripts/experiments/controller_compare_20261013
R=outputs/runs
# SONIC into round files
$SIM_PY $D/warehouse_states.py $R/controller_compare_warehouse_sonic_20261013 "$OUT/sonic/states"
$SIM_PY $D/warehouse_states.py $R/controller_compare_warehouse_sonic_edges_20261013 "$OUT/sonic_edges/states" --program production_edges
$SIM_PY $D/warehouse_states.py $R/controller_compare_warehouse_sonic_kg24_20261013 "$OUT/sonic_kg24/states" --program production_kg24
# the plans: the stock skills profile (warehouse_box) first, then the mars one (warehouse_box_mars, sonic_warehouse_plans.sh mars_)
for P in "" mars_; do
  S=$([ -z "$P" ] && echo _stock || echo "")
  T=$([ -z "$P" ] && echo _stockprofile || echo "")
  $SIM_PY $D/warehouse_states.py --plan $R/controller_compare_warehouse_${P}carry_back1_noturn_20261013 "$OUT/sonic_plans$S/states" carry_back1$T
  $SIM_PY $D/warehouse_states.py --plan $R/controller_compare_warehouse_${P}carry_back1_20261013 "$OUT/sonic_plans$S/states" carry_turn180$T
  $SIM_PY $D/warehouse_states.py --plan $R/controller_compare_warehouse_${P}put_down_20261013 "$OUT/sonic_putdown$S/states" put_down$T
done
B="$BFM_PY $D/bfm_warehouse.py"
# W1: walk-in, open loop and with the post-stop pelvis servo; the production median stance on every edge
$B --program walk --sonic "$OUT/sonic/states" --out "$OUT/bfm_walk"
$B --program walk --walk-servo --sonic "$OUT/sonic/states" --out "$OUT/bfm_walk_servo"
STAND=$($SIM_PY -c "import json,glob; print(json.load(open(sorted(glob.glob('$OUT/bfm_walk/*.json'))[0]))['walk_target']['standoff'])")
$B --program walk --standoff "$STAND" --tag _edges --sonic "$OUT/sonic_edges/states" --out "$OUT/bfm_walk_edges"
$B --program walk --standoff "$STAND" --walk-servo --tag _edges --sonic "$OUT/sonic_edges/states" --out "$OUT/bfm_walk_edges_servo"
# W2-W4, W8 and the mode sweep: open loop, wrist servo, pad servo
for m in 2 6; do
  $B --program replay --mode $m --sonic "$OUT/sonic/states" --out "$OUT/bfm_replay_m$m"
  $B --program replay --servo wrist --mode $m --sonic "$OUT/sonic/states" --out "$OUT/bfm_replay_m${m}s"
done
for m in 1 2 6 7; do
  $B --program replay --servo pad --mode $m --sonic "$OUT/sonic/states" --out "$OUT/bfm_replay_m${m}p"
done
# extrapolation: the 2.4 kg box, the carries, the stand-off sweep, the put-down
for m in 2 6; do
  $B --program replay --servo pad --mode $m --box-mass 2.4 --sonic "$OUT/sonic/states" --out "$OUT/bfm_replay_m${m}p_kg24"
  for c in back1 turn180; do
    $B --program replay --servo pad --mode $m --carry $c --sonic "$OUT/sonic/states" --out "$OUT/bfm_carry_${c}_m${m}p"
  done
  for S in "" _stock; do
    $B --program replay --servo pad --mode $m --tag _put_down$S --sonic "$OUT/sonic_putdown$S/states" --out "$OUT/bfm_putdown${S}_m${m}p"
  done
done
for s in -0.05 0.05 0.10; do
  $B --program replay --servo pad --mode 2 --pelvis-shift $s --sonic "$OUT/sonic/states" --out "$OUT/bfm_shift_m2p_$s"
done
# measure
$SIM_PY $D/warehouse_measure.py "$OUT/sonic/states" "$OUT/bfm_walk" "$OUT/bfm_walk_servo" "$OUT"/bfm_replay_m2 "$OUT"/bfm_replay_m6 \
  "$OUT"/bfm_replay_m2s "$OUT"/bfm_replay_m6s "$OUT"/bfm_replay_m1p "$OUT"/bfm_replay_m2p "$OUT"/bfm_replay_m6p "$OUT"/bfm_replay_m7p \
  --out "$OUT" --name warehouse_l1 --title "warehouse L1 生产档（TODO 25.3）"
$SIM_PY $D/warehouse_measure.py "$OUT/sonic_edges/states" "$OUT/bfm_walk_edges" "$OUT/bfm_walk_edges_servo" "$OUT/sonic_kg24/states" \
  "$OUT"/bfm_replay_m2p_kg24 "$OUT"/bfm_replay_m6p_kg24 "$OUT"/bfm_shift_m2p_* "$OUT/sonic_plans/states" "$OUT/sonic_plans_stock/states" \
  "$OUT"/bfm_carry_* "$OUT/sonic_putdown/states" "$OUT/sonic_putdown_stock/states" "$OUT"/bfm_putdown* \
  --out "$OUT/extra" --name warehouse_l1_extra --title "warehouse L1 外推档（TODO 25.3）"
echo "=== WAREHOUSE DONE"
