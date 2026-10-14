#!/usr/bin/env bash
# After sonic_l2_p25_queue.sh: SONIC's runs -> round files (bottle_states.py / warehouse_states.py), then the tables.
#   bash scripts/experiments/controller_compare_20261013/p25_l2_post.sh convert   # SONIC runs -> states (CPU only)
#   bash scripts/experiments/controller_compare_20261013/p25_l2_post.sh tables    # measure both controllers per cell
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
D=scripts/experiments/controller_compare_20261013
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
R=outputs/runs
O=outputs/controller_compare_20261013
case ${1:-} in
convert)
  for tier in "" _combo; do
    t=${tier:-_nominal}; t=${t#_}
    $SIM_PY $D/bottle_states.py $R/controller_compare_l2_bottle_sonic${tier}_20261013 $O/l2/bottle/sonic_$t/states --program l2_$t
    [ -d $R/controller_compare_l2_bottle_sonic${tier}_b_20261013 ] && $SIM_PY $D/bottle_states.py \
      $R/controller_compare_l2_bottle_sonic${tier}_b_20261013 $O/l2/bottle/sonic_$t/states --program l2_$t --label-prefix sonicb
    $SIM_PY $D/warehouse_states.py $R/controller_compare_l2_warehouse_sonic${tier}_20261013 $O/l2/warehouse/sonic_$t/states --program l2_$t
  done
  for cell in b20 b40 c d04 f e050 e150; do
    $SIM_PY $D/bottle_states.py $R/controller_compare_p25_bottle_${cell}_20261013 $O/p25/bottle/sonic_$cell/states --program p25_$cell
    $SIM_PY $D/warehouse_states.py $R/controller_compare_p25_warehouse_${cell}_20261013 $O/p25/warehouse/sonic_$cell/states --program p25_$cell
  done
  for cell in c g e050 e150; do
    $SIM_PY $D/bottle_states.py --plan $R/controller_compare_p25_bottle_carry_${cell}_20261013 $O/p25/bottle/sonic_carry_$cell/states carry_back2_$cell
    $SIM_PY $D/warehouse_states.py --plan $R/controller_compare_p25_warehouse_carry_${cell}_20261013 $O/p25/warehouse/sonic_carry_$cell/states carry_turn180_$cell
  done
  ;;
tables)
  # bottle / warehouse production cells and carries: both controllers in one table per task
  $SIM_PY $D/bottle_measure.py $O/bottle/sonic/states $O/p25/bottle/sonic_*/states $O/p25/bottle/bfm_* --out $O/p25/bottle
  $SIM_PY $D/warehouse_measure.py $O/warehouse/sonic/states $O/p25/warehouse/sonic_*/states $O/p25/warehouse/bfm_* \
    --out $O/p25/warehouse --name warehouse_p25 --title "warehouse 扰动（TODO 25.5）"
  # clips under f / d04 / g, both controllers, one table per cell (clip_measure groups by clip x controller x mode)
  for cell in f d04 g; do
    $SIM_PY $D/clip_measure.py $O/p25/clips/sonic_$cell $O/p25/clips/bfm_$cell --out $O/p25/clips/$cell
  done
  # nominal C1-C7 (both controllers) and S1
  $SIM_PY $D/clip_measure.py $O/complex/bfm $O/complex/sonic --out $O/complex
  python3 $D/switch_summary.py
  # L2
  $SIM_PY $D/l2_measure.py --out $O/l2
  ;;
*) echo "usage: $0 convert|tables"; exit 2 ;;
esac
