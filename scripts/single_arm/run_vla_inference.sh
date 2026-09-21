#!/bin/bash
# Drive single-arm closed-loop rounds against a running stack and a running policy server.
#
#   bash scripts/single_arm/run_vla_inference.sh outputs/runs/bottle_simple_200 --gpu 1
#
# Terminal 3 of three:
#   bash scripts/start_stack.sh    <same run> --gpu 1   # terminal 1
#   bash scripts/run_vla_server.sh <same run> --gpu 1   # terminal 2
#
# Refuses before the first round if either of the other two is missing, naming the command that
# fixes it. Takes the stack down when the rounds end; --keep-stack leaves it up.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/../paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--keep-stack] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_eval_single_arm "$@"
