#!/bin/bash
# B3, SONIC side (TODO 25.2.2): approach + the 108-cell probe on the stock bottle scene, one instance.
#   bash sonic_b3_probe.sh <instance> <gpu> [sonic_b3_probe.py flags]
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
export PYTHONPATH="$PWD/src" TMPDIR=/workspace/tmp
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${EGL_GPU:-1}
export NUMPY_MADVISE_HUGEPAGE=0
exec third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/experiments/controller_compare_20261013/sonic_b3_probe.py \
    --instance "$1" --gpu "$2" "${@:3}"
