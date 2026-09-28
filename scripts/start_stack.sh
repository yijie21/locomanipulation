#!/bin/bash
# Bring up the stack a run asks for, then return and leave it running.
#
#   bash scripts/start_stack.sh collect configs/experiments/bottle_simple_200.yaml --gpu 1
#   bash scripts/start_stack.sh collect configs/experiments/crate_table30_record.yaml --gpu 1 --dry-run
#   bash scripts/start_stack.sh eval    outputs/runs/bottle_simple_200 --gpu 1
#
# The mode is the whole difference between the two stacks and is spelled out rather than inferred
# from the argument: collection runs the deploy WITH its encoder (without it the planner's arm and
# crouch references are dropped) and evaluation runs it WITHOUT (the token stream will not hold
# otherwise). Until 2026-09-22 these were two scripts, collect_data.sh and start_stack.sh, whose
# only executable difference was this one word -- a second file per mode is a second place to
# forget (docs/handoff/flat-functional-scripts-20260922.md, decision 15).
#
# This is terminal 1. The others take the SAME target:
#
#   collect                                    eval
#   scripts/<line>/run_agent_server.sh         scripts/run_vla_server.sh            # terminal 2
#                                              scripts/<line>/run_vla_inference.sh  # terminal 3
#
# The stack is sim + deploy + exporter. It outlives this script (the children are started in their
# own sessions and their state is in <session_root>/stack.json), so the instance is guarded by the
# state file rather than by this process holding a lock. Stop it with scripts/stop_stack.sh; the
# round driver also takes it down when it finishes.
#
# Nothing from the experiment YAML is read or translated here: the file is handed down whole, and
# what the children are told is decided in exactly one place (locomani.stack.StackSpec.settings).
# Copying fields across this boundary by hand is how a hundred rounds once spawned at 0.7 m while
# the config said 1.1-1.3 m (2026-09-17).
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

usage() {
    echo "usage: $(basename "$0") collect|eval <experiment.yaml|run-dir> [--gpu N|N,M|all] [--lanes N|auto] [--name NAME] [--dry-run]" >&2
    exit 1
}
case "${1:-}" in
    collect|eval) MODE="$1"; shift ;;
    *) usage ;;
esac
[ $# -ge 1 ] || usage

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_stack "$@" --mode "$MODE"
