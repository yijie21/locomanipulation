#!/bin/bash
# Fill a backend.render offline run's dataset cameras from its state traces; run it before convert.
#
#   bash scripts/render_dataset_cameras.sh outputs/runs/box_room_grasp_baseline_20260924 --gpu 0
#   bash scripts/render_dataset_cameras.sh outputs/runs/box_room_grasp_baseline_20260924 --dry-run
#
# No stack, one card for EGL. Offline, the exporter records black placeholder frames; this renders each
# parquet row's cameras from the trace at the seq the row used and writes them over the placeholders.
# A live run is a no-op, and convert refuses an offline run until this has run.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.render_dataset_cameras "$@"
