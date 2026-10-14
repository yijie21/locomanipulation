#!/usr/bin/env bash
# TODO 25.8 L2, ScaleBFM side: the 25.7 ports on SONIC's L2 rounds (their starts), both tiers, four processes at once.
# Needs the SONIC round files (p25_l2_post.sh convert) and no SONIC stack running.
#   bash scripts/experiments/controller_compare_20261013/bfm_l2_batch.sh
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-1}
D=scripts/experiments/controller_compare_20261013
O=outputs/controller_compare_20261013/l2
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
COMBO='{"delay_s":0.02,"noise":{"gyro":0.1,"dq":0.5},"gain_scale":[0.8,1.2],"seed":25}'  # the SONIC combo configs' stack_env
for tier in nominal combo; do
  p=(); [ $tier = combo ] && p=(--perturb "$COMBO")
  $SIM_PY $D/bfm_expert_bottle.py $O/bottle/sonic_$tier/states --config configs/experiments/controller_compare_l2_bottle_sonic_20261013.yaml \
    --mode 4 --tag _$tier "${p[@]}" --out $O/bottle/bfm_$tier > $O/bottle_bfm_$tier.log 2>&1 &
  $SIM_PY $D/bfm_expert_warehouse.py $O/warehouse/sonic_$tier/states --config configs/experiments/controller_compare_l2_warehouse_sonic_20261013.yaml \
    --mode 2 --set visual.tracking_gain=0.5 --tag _$tier "${p[@]}" --out $O/warehouse/bfm_$tier > $O/warehouse_bfm_$tier.log 2>&1 &
done
wait
echo "BFM L2 DONE $(date -u)"
