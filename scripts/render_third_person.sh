#!/bin/bash
# A run's third-person videos, drawn by the renderer of its images.source (mujoco | isaac | unreal); optional, run
# whenever wanted -- no other step needs them.
#
#   bash scripts/render_third_person.sh outputs/runs/warehouse_smoke --gpu 1           # collection: kept rounds
#   bash scripts/render_third_person.sh outputs/runs/warehouse_smoke --gpu 1 --eval    # closed-loop rounds
#   bash scripts/render_third_person.sh outputs/runs/warehouse_smoke --all --dry-run   # every round; plan only
#
# Writes <run>/videos_<renderer>/<round>.mp4 (<run>/eval/... with --eval; MuJoCo: videos/) and previews <= 500 KB in
# previews_<renderer>/; a filmed round is skipped on a rerun. Run it before convert_data.sh to ship the collection's
# videos in the dataset's third_person/. Unreal renders in the editor start_unreal.sh started, on its card.
# This file only picks the renderer's venv; locomani.sim.render_third_person does the rest.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--eval] [--all] [--dry-run] [rounds...]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE=$("$VENV_SIM_PY" -c '
import sys
from pathlib import Path
from locomani.pipeline import run_context
print(run_context.load_run_config(Path(sys.argv[1]))["images"]["source"])' "$1") || exit 1
case "$SOURCE" in
  mujoco) PY="$VENV_SIM_PY" ;;
  isaac) PY="$RENDER_PY" ;;
  unreal) PY="$BRIDGE_PY" ;;
  *) echo "images.source $SOURCE: no renderer for it" >&2; exit 1 ;;
esac
[ -x "$PY" ] || { echo "$PY is not executable: the $SOURCE renderer is not installed (docs/setup.md)" >&2; exit 1; }
exec "$PY" -m locomani.sim.render_third_person "$@"
