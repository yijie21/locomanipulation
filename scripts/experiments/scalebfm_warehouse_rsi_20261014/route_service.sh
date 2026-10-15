#!/usr/bin/env bash
# The CuRobo route services for visual.reach_route planned (tasks/crate/reach_route.py), detached, with the mars arms +
# waist model of the 10-08 study. N instances (default 1) on ports PORT0.. (default TF_CUROBO_PORT or 5795), spread over
# GPUS (default 0; "0,1" alternates), each with the service flags in FLAGS. The default --no-self-collision-opt planned
# the 50 diagnostic routes bit-identical to the full optimisation in 0.34 s instead of 5.5 s (the self-collision cost
# is zero along them but its kernel was 99.5% of the GPU time; the result is still checked; TODO 26.5.5, 2026-10-15).
# Prints the TF_CUROBO_PORTS line round processes need (bfm_round.py spreads its shards over them).
#   Usage: [N=2 GPUS=0,1 FLAGS="..."] bash route_service.sh [log]
#   Stop: pgrep -f '[c]urobo_service' (never a plain pkill -f: it matches the calling shell)
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp
ROBOT=${ROBOT:-outputs/curobo_warehouse_20261008/mars_curobo.json}
LOG=${1:-outputs/scalebfm_warehouse_rsi_20261014/route/service.log}
N=${N:-1}
PORT0=${PORT0:-${TF_CUROBO_PORT:-5795}}
IFS=, read -ra DEVICES <<< "${GPUS:-0}"
[ -f "$ROBOT" ] || { echo "no robot model $ROBOT (curobo_warehouse_20261008: export_geometry.py, curobo_robot.py)"; exit 1; }
ports=()
for k in $(seq 0 $((N - 1))); do
  port=$((PORT0 + k))
  log=$LOG
  [ "$N" -gt 1 ] && log=${LOG%.log}_$port.log
  # shellcheck disable=SC2086 # FLAGS is a list of service flags
  CUDA_VISIBLE_DEVICES=${DEVICES[k % ${#DEVICES[@]}]} PYTHONPATH=src nohup setsid third_party/SIMPLE/.venv/bin/python \
    -m locomani.primitives.curobo_service "$ROBOT" --port "$port" --batch 2 ${FLAGS---no-self-collision-opt} \
    >> "$log" 2>&1 < /dev/null &
  echo "route service pid $! port $port gpu ${DEVICES[k % ${#DEVICES[@]}]} log $log"
  ports+=("$port")
done
echo "export TF_CUROBO_PORTS=$(IFS=,; echo "${ports[*]}")"
