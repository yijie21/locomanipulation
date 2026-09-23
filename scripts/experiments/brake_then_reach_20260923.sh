#!/usr/bin/env bash
# Guard brake to a line, then reach to the bimanual envelope extreme (brake_then_reach_20260923.py).
# Extra args go to that script (--targets/--repeats/--marker).
set -uo pipefail
cd /workspace/code/locomanipulation
O="${BRAKE_OUT:-$PWD/outputs/diagnostics/brake_reach_20260923}"
mkdir -p "$O/trace"
GPU="${BRAKE_GPU:-0}"

export INSTANCE=0 CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export SCENE_RANDOM=0 SIM_VIS=none SIM_STATE_TRACE_DIR="$O/trace"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export LOCOMANI_SIM_DIR="$PWD/outputs/scene/walk"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python

clean() { "$PY" -m locomani.stack.cli down --instance 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

echo "===== $(date -u +%H:%M:%S) scene $LOCOMANI_SIM_DIR"
"$PY" -m locomani.stack.cli up --instance 0 --gpu "$GPU" --no-record --scene "$LOCOMANI_SIM_DIR" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held; backing off"; tail -3 "$O/stack.log"; exit 3; }
[ "$rc" = 0 ] || { echo "stack failed rc=$rc, see $O/stack.log"; clean; exit 1; }
timeout 5400 "$PY" -u scripts/experiments/brake_then_reach_20260923.py --out "$O" "$@" > "$O/reach.log" 2>&1
echo "probe exit $? $(date -u +%H:%M:%S)"
clean
