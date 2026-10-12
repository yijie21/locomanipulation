#!/bin/bash
# TODO 21.3.4 / 23.2: one object on one instance (run_grasp.py; its defaults: cached grasp, IK planner, one round).
#   bash run_one.sh <object id> <instance> <gpu> [tag] [run_grasp.py flags, e.g. --grasp-source graspnet]
set -u
cd "$(dirname "$0")/../../.." || exit 1
# the deploy scripts/paths.sh picked (setup step 3's patched one); without it the original author's patched copy
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$PWD/outputs/bottle_continuous_20260914/deploy}"
export PYTHONPATH="$PWD/src" TMPDIR=/workspace/tmp
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${EGL_GPU:-1}
export NUMPY_MADVISE_HUGEPAGE=0  # a fragmented host stalls numpy's huge-page faults in compaction (ghost_overlay.py)
exec third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/experiments/graspnet_bottle_20261008/run_grasp.py \
    "$1" --instance "$2" --gpu "$3" ${4:+--tag "$4"} "${@:5}"
