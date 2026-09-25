#!/bin/bash
# Fill a backend.render offline run's dataset cameras with Isaac pictures; run it before convert.
#
#   bash scripts/render_dataset_cameras_isaac.sh outputs/runs/box_room_grasp_200_20260925 --gpu 1
#   bash scripts/render_dataset_cameras_isaac.sh outputs/runs/box_room_grasp_200_20260925 --dry-run
#
# For a run whose experiment sets `eval.image_source: isaac` (the others use render_dataset_cameras.sh):
# the images it trains on come from the same Isaac view the eval mirror serves. Kept episodes of finished
# batches only, so it can run on another card beside a collection still recording; rerun until done.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras "$@"
