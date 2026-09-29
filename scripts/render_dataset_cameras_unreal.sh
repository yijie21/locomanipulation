#!/bin/bash
# Fill a backend.render offline run's dataset cameras with Unreal pictures; run it before convert.
#
#   bash scripts/render_dataset_cameras_unreal.sh outputs/runs/<run>
#   bash scripts/render_dataset_cameras_unreal.sh outputs/runs/<run> --dry-run
#
# For a run whose experiment sets `images.source: unreal` (the others use render_dataset_cameras.sh or
# render_dataset_cameras_isaac.sh): the images it trains on come from the same Unreal view the eval mirror
# serves. Needs the Unreal editor (scripts/start_unreal.sh --gpu N chooses the card). Kept episodes of finished
# batches only, so it can run beside a collection still recording; rerun until done.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--dry-run]" >&2; exit 1; }
[ -x "$BRIDGE_PY" ] || { echo "BRIDGE_PY not executable: $BRIDGE_PY (scripts/setup_unreal.sh)" >&2; exit 1; }

exec "$BRIDGE_PY" -m locomani.sim.unreal.render_dataset_cameras "$@"
