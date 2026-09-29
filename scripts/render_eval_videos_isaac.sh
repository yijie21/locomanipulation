#!/bin/bash
# Film a closed-loop evaluation's rounds with Isaac, for a run whose `images.source` is isaac.
#
#   bash scripts/render_eval_videos_isaac.sh outputs/runs/warehouse_smoke --gpu 1 [--all]
#   bash scripts/render_eval_videos_isaac.sh outputs/runs/warehouse_smoke --dry-run
#
# The collection's third-person video (same shot, ghost grippers, ego/wrist insets), drawn by the renderer the
# policy saw: <run>/eval/videos_isaac/<round>.mp4, previews <= 500 KB in <run>/eval/previews_isaac/. Successful
# rounds only unless --all; a rerun films only what is missing. Unreal runs: render_eval_videos_unreal.sh.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--all] [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RENDER_PY" -m locomani.sim.isaac.render_eval_videos "$@"
