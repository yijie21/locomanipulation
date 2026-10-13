#!/bin/bash
# Warehouse L1, SONIC side (TODO 25.3): the production expert's batches named on the command line (configs/experiments/
# controller_compare_warehouse_sonic*_20261013.yaml), one lane on instance 0, GPU 0, nothing else on the machine
# (SONIC is wall-clock). Usage: bash sonic_warehouse.sh <log> <config>...
set -uo pipefail
cd "$(dirname "$0")/../../.."
. scripts/paths.sh
export TMPDIR=/workspace/tmp
L=$1; shift
echo "=== start $(date)" > "$L"
for C in "$@"; do
  echo "=== collect $C $(date)" >> "$L"
  bash scripts/start_stack.sh collect "$C" --gpu 0 --lanes 1 >> "$L" 2>&1 && bash scripts/bimanual/run_agent_server.sh "$C" --gpu 0 >> "$L" 2>&1
  echo "=== collect $C done rc=$? $(date)" >> "$L"
done
echo "=== SONIC WAREHOUSE DONE $(date)" >> "$L"
