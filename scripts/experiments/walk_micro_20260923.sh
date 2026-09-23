#!/usr/bin/env bash
# Can SONIC v1.1 make small moves, and is the limit the controller or our closed loop?
# diag_micro in the walk scene (G1 alone on the floor), one stack on instance 0, video on.
# Stop early: touch "$O/STOP".
set -uo pipefail
cd /workspace/code/locomanipulation
O="${MICRO_OUT:-$PWD/outputs/diagnostics/walk_micro_v11_20260923}"
mkdir -p "$O/video"
GPU="${MICRO_GPU:-0}"

export INSTANCE=0 CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export SCENE_RANDOM=0 SIM_VIS=none SIM_VIDEO_DIR="$O/video"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export LOCOMANI_SIM_DIR="$PWD/outputs/scene/walk"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python

clean() { "$PY" -m locomani.stack.cli down --instance 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM
rm -f "$O/STOP"

echo "===== $(date -u +%H:%M:%S) scene $LOCOMANI_SIM_DIR"
"$PY" -m locomani.stack.cli up --instance 0 --gpu "$GPU" --no-record --scene "$LOCOMANI_SIM_DIR" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held; backing off"; tail -3 "$O/stack.log"; exit 3; }
[ "$rc" = 0 ] || { echo "stack failed rc=$rc, see $O/stack.log"; clean; exit 1; }
timeout 3600 "$PY" -u -m locomani.backends.sonic.diag_micro --json "$O/micro.json" "$@" > "$O/micro.log" 2>&1
echo "probe exit $? $(date -u +%H:%M:%S)"
clean
touch "$O/done"
