#!/bin/bash
# Serve a run's checkpoint for closed-loop evaluation. Foreground: Ctrl-C frees the card.
#
#   bash scripts/run_vla_server.sh outputs/runs/bottle_simple_200 --gpu 1
#   bash scripts/run_vla_server.sh outputs/runs/my_box --gpu 0 --backbone-path ~/Checkpoints/Cosmos-Reason2-2B
#
# --backbone-path: a checkpoint trained on another machine (the cluster) records its VLM backbone at a
# path that does not exist here; this points serve_policy.py at the local copy.
#
# Terminal 2 of three (terminal 1: scripts/start_stack.sh eval; terminal 3: the line's
# run_vla_inference.sh). Serves both lines -- the checkpoint, the modality config and the
# embodiment tag all come from the run itself.
#
# Output is shown here and appended to <run>/eval/serve.log. The round driver waits for the port,
# so this can be started first or second; it just has to be up before the rounds begin.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--backbone-path DIR] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_vla_server "$@"
