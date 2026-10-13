#!/bin/bash
# Bottle L1 extrapolation tier, SONIC side (TODO 25.2.3): the production expert at 0.7 m/s and at +-40 deg spawn yaw
# (one 10-round recorded batch each), then four carry plans through plan_episode with the sim's state trace on
# (back 1 m, back 2 m, slow turn 90 deg + 1.5 m, slow turn 180 deg + 1.5 m; 10 rounds each). Instance 0, GPU 0,
# nothing else on the machine (SONIC is wall-clock). Usage: bash sonic_bottle_extra.sh [log]
set -uo pipefail
cd "$(dirname "$0")/../../.."
. scripts/paths.sh
export TMPDIR=/workspace/tmp
L=${1:-outputs/controller_compare_20261013/b_sonic_extra.log}
echo "=== start $(date)" > "$L"
for v in fast yaw40; do
  C=configs/experiments/controller_compare_bottle_sonic_${v}_20261013.yaml
  echo "=== collect $v $(date)" >> "$L"
  bash scripts/start_stack.sh collect $C --gpu 0 >> "$L" 2>&1 && bash scripts/single_arm/run_agent_server.sh $C --gpu 0 >> "$L" 2>&1
  echo "=== collect $v done rc=$? $(date)" >> "$L"
done
for plan in back1 back2 turn90slow turn180slow; do
  name=controller_compare_bottle_carry_${plan}_20261013
  export SIM_STATE_TRACE_DIR=$PWD/outputs/runs/$name/trace
  echo "=== plan $plan $(date)" >> "$L"
  "$VENV_SIM_PY" -u scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 0 --instance 0 --rounds 10 \
    --seed 40000 --plan configs/plans/bottle_carry_${plan}.json --name $name >> "$L" 2>&1
  echo "=== plan $plan done rc=$? $(date)" >> "$L"
done
echo "=== SONIC EXTRA DONE $(date)" >> "$L"
