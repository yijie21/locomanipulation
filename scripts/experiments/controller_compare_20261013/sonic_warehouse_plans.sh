#!/bin/bash
# Warehouse L1 W5-W7, SONIC side (TODO 25.3): the warehouse_box skills through plan_episode on the mars production config,
# 10 rounds each with the sim's state trace on -- carry back 1 m heading kept (W5's first 0.3 m and W6 straight),
# turn 180 deg then 1 m (W6 turn), put the box back down on the platform (W7). Instance 0, GPU 0, nothing else on the
# machine (SONIC is wall-clock). Usage: bash sonic_warehouse_plans.sh <log> [prefix]; prefix `mars_` runs the
# configs/plans/warehouse_mars_*.json plans (skills profile warehouse_box_mars) instead of the stock-profile ones.
set -uo pipefail
cd "$(dirname "$0")/../../.."
. scripts/paths.sh
export TMPDIR=/workspace/tmp
L=$1
P=${2:-}
echo "=== start $(date)" >> "$L"
for plan in carry_back1_noturn carry_back1 put_down; do
  name=controller_compare_warehouse_${P}${plan}_20261013
  export SIM_STATE_TRACE_DIR=$PWD/outputs/runs/$name/trace
  echo "=== plan $plan $(date)" >> "$L"
  "$VENV_SIM_PY" -u scripts/plan_episode.py configs/experiments/warehouse_mars_200.yaml --gpu 0 --instance 0 --rounds 10 \
    --seed 7000 --plan configs/plans/warehouse_${P}${plan}.json --name $name >> "$L" 2>&1
  echo "=== plan $plan done rc=$? $(date)" >> "$L"
done
echo "=== SONIC WAREHOUSE PLANS DONE $(date)" >> "$L"
