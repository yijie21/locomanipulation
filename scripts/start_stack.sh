#!/bin/bash
# Bring up the evaluation stack for a run, then return and leave it running.
#
#   bash scripts/start_stack.sh outputs/runs/bottle_simple_200 --gpu 1
#
# Terminal 1 of three. The other two:
#   bash scripts/run_vla_server.sh <same run> --gpu 1                  # terminal 2
#   bash scripts/single_arm/run_vla_inference.sh <same run> --gpu 1    # terminal 3
#   bash scripts/bimanual/run_vla_inference.sh  <same run> --gpu 1
#
# This is NOT the collection stack: the deploy runs WITHOUT its encoder here or the token stream
# will not hold, and WITH it during collection or the planner's arm and crouch references are
# dropped. Which block is used is decided by --mode, in one place (locomani.pipeline.run_stack).
#
# Stop it with scripts/stop_stack.sh; the round driver also takes it down when it finishes.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir|experiment.yaml> [--gpu N] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_stack "$@" --mode eval
