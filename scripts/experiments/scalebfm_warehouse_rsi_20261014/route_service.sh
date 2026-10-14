#!/usr/bin/env bash
# The CuRobo route service for visual.reach_route planned (tasks/crate/reach_route.py), detached, on GPU ${GPU:-0} and
# TF_CUROBO_PORT (default 5795), with the mars arms + waist model of the 10-08 study. Usage: bash route_service.sh [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp
ROBOT=${ROBOT:-outputs/curobo_warehouse_20261008/mars_curobo.json}
LOG=${1:-outputs/scalebfm_warehouse_rsi_20261014/route/service.log}
[ -f "$ROBOT" ] || { echo "no robot model $ROBOT (curobo_warehouse_20261008: export_geometry.py, curobo_robot.py)"; exit 1; }
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=src nohup setsid third_party/SIMPLE/.venv/bin/python -m locomani.primitives.curobo_service \
  "$ROBOT" --port "${TF_CUROBO_PORT:-5795}" --batch 2 >> "$LOG" 2>&1 < /dev/null &
echo "route service pid $! log $LOG"
