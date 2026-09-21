#!/bin/bash
# Stop the stack on one instance: kill sim, deploy and exporter by process group, wait for the
# ports, drop the state file.
#
#   bash scripts/stop_stack.sh --instance 0
#   bash scripts/stop_stack.sh --instance 1
#
# The round drivers take their own stack down when they finish, so this is for the cases they
# cannot cover: a driver killed mid-run, a stack started for a look, a terminal closed. It stops
# ONLY the named instance -- an unscoped pkill takes down other sessions' sims, and the tell is a
# frozen-teleport failure repeated to the end of their run.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.stack.cli down "$@"
