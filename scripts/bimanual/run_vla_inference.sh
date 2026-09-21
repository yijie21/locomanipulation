#!/bin/bash
# Drive bimanual closed-loop rounds against a running stack and a running policy server.
#
#   bash scripts/bimanual/run_vla_inference.sh outputs/runs/crate_table30_record --gpu 1
#
# Terminal 3 of three:
#   bash scripts/start_stack.sh    <same run> --gpu 1   # terminal 1
#   bash scripts/run_vla_server.sh <same run> --gpu 1   # terminal 2
#
# This line's evaluator restarts the DEPLOY between batches rather than the whole stack: an
# evaluation runs no exporter, so there are no exporter threads to reclaim, and the deploy is what
# the per-batch restart was ever for.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/../paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_eval_bimanual "$@"
