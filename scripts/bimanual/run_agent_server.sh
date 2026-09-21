#!/bin/bash
# Drive bimanual collection: attach to the stack terminal 1 started, run batch after batch.
#
#   bash scripts/bimanual/run_agent_server.sh configs/experiments/crate_table30_record.yaml --gpu 1
#
# Terminal 2 of two; terminal 1 is:
#   bash scripts/collect_data.sh <same yaml> --gpu 1
#
# "Agent server" is what this is: the scripted stand-in for the Pico teleop server
# (src/locomani/backends/sonic/agent_server.py), driving the robot from privileged simulator state
# instead of from a human in a headset. task.expert picks which crate expert plays that part.
#
# The stack is taken down when the run ends; --keep-stack leaves it up.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/../paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <experiment.yaml> [--gpu N] [--name NAME] [--keep-stack] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_collect_bimanual "$@"
