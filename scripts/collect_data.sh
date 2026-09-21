#!/bin/bash
# Bring up the recording stack for a collection run, then return and leave it running.
#
#   bash scripts/collect_data.sh configs/experiments/bottle_simple_200.yaml --gpu 1
#   bash scripts/collect_data.sh configs/experiments/crate_table30_record.yaml --gpu 1 --dry-run
#
# Terminal 1 of two. Terminal 2 drives the rounds:
#   bash scripts/single_arm/run_agent_server.sh <same yaml> --gpu 1
#   bash scripts/bimanual/run_agent_server.sh   <same yaml> --gpu 1
#
# The stack is sim + deploy + exporter. It outlives this script (the children are started in their
# own sessions and their state is in <session_root>/stack.json), so the instance is guarded by the
# state file rather than by this process holding a lock. Stop it with scripts/stop_stack.sh.
#
# Nothing from the experiment YAML is read or translated here: the file is handed down whole, and
# what the children are told is decided in exactly one place (locomani.stack.StackSpec.settings).
# Copying fields across this boundary by hand is how a hundred rounds once spawned at 0.7 m while
# the config said 1.1-1.3 m (2026-09-17).
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <experiment.yaml> [--gpu N] [--name NAME] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_stack "$@" --mode collect
