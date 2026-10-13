#!/bin/bash
# wbc_compare step 2 (TODO 24.2): the open-loop hand probe on the SONIC lockstep stack, one object scene, one instance
# (sonic_probe.py; the environment is run_one.sh's).
#   bash sonic_probe.sh <object id> <instance> <gpu> [sonic_probe.py flags, e.g. --hold 2.0 --tag _b]
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
export PYTHONPATH="$PWD/src" TMPDIR=/workspace/tmp
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${EGL_GPU:-1}
export NUMPY_MADVISE_HUGEPAGE=0
exec third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/experiments/wbc_compare_20261012/sonic_probe.py \
    --object "$1" --instance "$2" --gpu "$3" "${@:4}"
