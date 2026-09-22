#!/bin/bash
# Drive single-arm collection: attach to the stack terminal 1 started, run batch after batch.
#
#   bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_simple_200.yaml --gpu 1
#   bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_simple_200.yaml --dry-run
#
# Terminal 2 of two; terminal 1 is:
#   bash scripts/start_stack.sh collect <same yaml> --gpu 1
#
# "Agent server" is what this is: the scripted stand-in for the Pico teleop server
# (src/locomani/backends/sonic/agent_server.py), driving the robot from privileged simulator state
# instead of from a human in a headset. The expert process this launches is the agent; the stack is
# the robot and the world.
#
# The stack is taken down when the run ends (a recording stack holds ~1800 of the container's 3840
# pids). Pass --keep-stack to leave it up for a second run.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/../paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <experiment.yaml> [--gpu N] [--name NAME] [--keep-stack] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.tasks.bottle.collect "$@"
