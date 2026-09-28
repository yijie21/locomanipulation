#!/bin/bash
# Drive bimanual collection: attach to the stacks terminal 1 started, run batch after batch in every lane.
#
#   bash scripts/bimanual/run_agent_server.sh configs/experiments/crate_table30_record.yaml
#
# Terminal 2 of two; terminal 1 is:
#   bash scripts/start_stack.sh collect <same yaml>
#
# Terminal 1 decided how many lanes (one stack + one driver process each) fit on this machine and wrote them to
# <run>/collect/lanes.json (src/locomani/pipeline/lanes.py); this runs one driver process per lane and prints the
# count before the first round.
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
exec "$VENV_SIM_PY" -m locomani.tasks.crate.collect "$@"
