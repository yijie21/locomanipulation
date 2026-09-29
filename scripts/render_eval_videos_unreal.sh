#!/bin/bash
# Film a closed-loop evaluation's rounds with Unreal, for a run whose `images.source` is unreal.
#
#   bash scripts/render_eval_videos_unreal.sh outputs/runs/warehouse_smoke_unreal [--all]
#   bash scripts/render_eval_videos_unreal.sh outputs/runs/warehouse_smoke_unreal --dry-run
#
# The collection's third-person video (same shot, ghost grippers, ego/wrist insets), drawn by the renderer the
# policy saw: <run>/eval/videos_unreal/<round>.mp4, previews <= 500 KB in <run>/eval/previews_unreal/. Needs the
# Unreal editor (scripts/start_unreal.sh --gpu N chooses the card). Successful rounds only unless --all.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--all] [--dry-run]" >&2; exit 1; }
[ -x "$BRIDGE_PY" ] || { echo "BRIDGE_PY not executable: $BRIDGE_PY (scripts/setup_unreal.sh)" >&2; exit 1; }

exec "$BRIDGE_PY" -m locomani.sim.unreal.render_eval_videos "$@"
