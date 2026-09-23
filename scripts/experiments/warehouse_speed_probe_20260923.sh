#!/usr/bin/env bash
# Warehouse sim speed with and without camera reading (warehouse_speed_probe_20260923.py). Extra args go there.
set -uo pipefail
cd /workspace/code/locomanipulation
O="${PROBE_OUT:-$PWD/outputs/diagnostics/warehouse_speed_20260923}"
mkdir -p "$O/trace"
GPU="${PROBE_GPU:-0}"

export INSTANCE=0 CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export SCENE_RANDOM=0 SIM_VIS=none SIM_STATE_TRACE_DIR="$O/trace"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export LOCOMANI_SIM_DIR="$PWD/outputs/box_rim_execution_20260920/scene_reverse_camera"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python

clean() { "$PY" -m locomani.stack.cli down --instance 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

echo "===== $(date -u +%H:%M:%S) scene $LOCOMANI_SIM_DIR"
"$PY" -m locomani.stack.cli up --instance 0 --gpu "$GPU" --no-record --scene "$LOCOMANI_SIM_DIR" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held; backing off"; tail -3 "$O/stack.log"; exit 3; }
[ "$rc" = 0 ] || { echo "stack failed rc=$rc, see $O/stack.log"; clean; exit 1; }
timeout 3600 "$PY" -u scripts/experiments/warehouse_speed_probe_20260923.py --out "$O" "$@" > "$O/probe.log" 2>&1
echo "probe exit $? $(date -u +%H:%M:%S)"
clean
