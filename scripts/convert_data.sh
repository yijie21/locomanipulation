#!/bin/bash
# Turn a run's exporter sessions into a LeRobot dataset, compute its statistics, audit it.
#
#   bash scripts/convert_data.sh outputs/runs/bottle_simple_200
#
# No card and no stack. Serves both lines unchanged: single-arm and bimanual runs differ in how
# they were collected, not in what the converter is handed. The session/manifest pairing comes
# from the run's own summary.json, positionally, in order.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_convert "$@"
