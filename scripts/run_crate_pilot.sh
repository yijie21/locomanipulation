#!/bin/bash
# Crate (bimanual) pilot: a thin shell over src/locomani/pipeline/run_crate_pilot.py.
#
#   bash scripts/run_crate_pilot.sh [rounds] [seed] [extra args to run_crate_pilot.py]
#   CRATE_EXPERT=carry bash scripts/run_crate_pilot.sh 10 0 --config configs/tasks/crate_wall.json
#
# This script does THREE things and no more: fix INSTANCE before paths.sh is
# sourced, pick the venv, and forward. The stack lifecycle, the run directory,
# the manifests and the stop live in the Python driver -- the same split the
# single-arm line has (run_learn_pilot.py owns the stack, grasp_gym.py is an
# expert process that assumes it is up). Until 2026-09-11 this file and its
# twin run_bimanual_pilot.sh each carried all of that inline, in ~80 lines of
# near-identical text, including a hand-rolled `pgrep -f` copy of
# the stop script and a third copy of the port arithmetic.
#
#   OUT            run directory (default outputs/videos/wall_pilot/<UTC stamp>;
#                  one run per directory, so two pilots never append to one manifest)
#   INSTANCE       stack instance (default 1; 0 is the collection/eval stack)
#   CRATE_EXPERT   wall (default) or carry
#   GPU            CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID (default 1)
set -uo pipefail
# INSTANCE must be fixed BEFORE paths.sh is sourced: paths.sh derives the whole
# port block from it and defaults it to 0, which is the collection/eval stack.
# The first version of this script sourced paths.sh first, inherited instance 0
# and tried to start a second stack on the live one (2026-09-07 06:00).
export INSTANCE="${INSTANCE:-1}"
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

# WALL_WALK / WALL_FEATURE stopped being read on 2026-09-09 (refactor step 3:
# CrateTaskConfig); a run that still sets them silently gets the ready-stand
# default (2026-09-10: 10 rendered episodes without a walk). Refuse, don't warn.
for dead in WALL_WALK WALL_FEATURE; do
    [ -z "${!dead:-}" ] || { echo "refusing: $dead is no longer read; pass --config configs/tasks/crate_wall.json or --set wall_walk=true / wall_feature=..."; exit 2; }
done
# Same rule for the two the driver now owns: it writes videos to <OUT>/videos_NN
# and traces to <OUT>/trace (where scripts/render_wall_pilot.sh looks for them),
# so an inherited value would be overridden rather than honoured.
for owned in SIM_VIDEO_DIR SIM_STATE_TRACE_DIR; do
    [ -z "${!owned:-}" ] || { echo "refusing: $owned is set by the driver from --out; use OUT=<dir> instead"; exit 2; }
done

ROUNDS="${1:-3}"
SEED="${2:-0}"
shift $(($# < 2 ? $# : 2))  # drop [rounds] [seed]; "$@" now holds only the driver's extra args
OUT="${OUT:-$LOCOMANI_OUTPUTS/videos/wall_pilot/$(date -u +%Y%m%d_%H%M%S)}"
echo "[crate] run dir $OUT"

exec "$VENV_SIM_PY" -u src/locomani/pipeline/run_crate_pilot.py \
    --expert "${CRATE_EXPERT:-wall}" --rounds "$ROUNDS" --seed "$SEED" \
    --out "$OUT" --instance "$INSTANCE" --gpu "${GPU:-1}" "$@"
