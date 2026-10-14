#!/usr/bin/env bash
# C1-C7 clips, SONIC side (TODO 25.4): the stack on one mars floor scene, sonic_clip.py over that scene's clips, down.
#   bash scripts/experiments/controller_compare_20261013/sonic_clip.sh <scene> <instance> <gpu> [sonic_clip.py flags]
#   scenes: walk_mars (most clips), walk_mars_load12 (the loaded C4 clips). Nothing else may run beside it.
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
SCENE=$1; INST=$2; GPU=$3; shift 3
O=${CLIP_OUT:-outputs/controller_compare_20261013/complex/sonic}  # CLIP_OUT: a perturbation cell's own dir (25.5)
mkdir -p "$O/trace_$SCENE"
export INSTANCE="$INST" CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export SCENE_RANDOM=0 SIM_VIS=none SIM_STATE_TRACE_DIR="$PWD/$O/trace_$SCENE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 TMPDIR=/workspace/tmp
export PYTHONPATH="$PWD/src" LOCOMANI_SIM_DIR="$PWD/outputs/scene/$SCENE"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
clean() { "$PY" -m locomani.stack.cli down --instance "$INST" >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM
echo "===== $(date -u +%H:%M:%S) scene $SCENE instance $INST gpu $GPU"
"$PY" -m locomani.stack.cli up --instance "$INST" --gpu "$GPU" --no-record --scene "$LOCOMANI_SIM_DIR" > "$O/stack_$SCENE.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance lock held; backing off"; tail -3 "$O/stack_$SCENE.log"; exit 3; }
[ "$rc" = 0 ] || { echo "stack failed rc=$rc, see $O/stack_$SCENE.log"; clean; exit 1; }
timeout 7200 "$PY" -u scripts/experiments/controller_compare_20261013/sonic_clip.py --scene "$SCENE" --out "$O" "$@" > "$O/sonic_$SCENE.log" 2>&1
echo "clips exit $? $(date -u +%H:%M:%S)"
clean
echo "===== done"
