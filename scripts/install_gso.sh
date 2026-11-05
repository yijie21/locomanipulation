#!/bin/bash
# Install the Google Scanned Objects MuJoCo models (kevinzakka/mujoco_scanned_objects: 1030 objects, 2 GB of OBJ
# meshes, textures and V-HACD hulls) into assets/mujoco_scanned_objects/models/, which git ignores; the tracked
# files next to it are catalog.json, categories.json (scripts/gso_categories.py), LICENSE, README.md and VERSION
# (docs/rsi_logistic_room_1030.md, 2026-11-03).
#
#   bash scripts/install_gso.sh                  # git clone from GitHub (shallow)
#   bash scripts/install_gso.sh --from <dir>     # copy from a local checkout or extracted archive holding models/
#
# Already installed (models/ holds every object the catalog names) means nothing is done. After a fresh install,
# rebuild the catalog only when the model set changed: $VENV_SIM_PY scripts/gso_catalog.py (the tracked catalog is
# the one the scenes were drawn from; a rebuilt one with the same models gives the same file).
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
ROOT="$LOCOMANI_ROOT"
ASSETS="$ROOT/assets/mujoco_scanned_objects"
MODELS="$ASSETS/models"
REPO_URL="https://github.com/kevinzakka/mujoco_scanned_objects.git"
FROM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift ;;
        *) echo "usage: $(basename "$0") [--from <dir>]" >&2; exit 1 ;;
    esac
    shift
done

expected() {  # how many objects the tracked catalog names (0 when there is no catalog yet)
    [ -f "$ASSETS/catalog.json" ] || { echo 0; return; }
    python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["objects"]))' "$ASSETS/catalog.json"
}
present() { [ -d "$MODELS" ] && find "$MODELS" -mindepth 2 -maxdepth 2 -name model.xml | wc -l || echo 0; }

WANT=$(expected)
HAVE=$(present)
if [ "$HAVE" -gt 0 ] && { [ "$WANT" -eq 0 ] || [ "$HAVE" -ge "$WANT" ]; }; then
    echo "already installed: $HAVE objects under $MODELS (catalog names $WANT)"
    exit 0
fi

mkdir -p "$ASSETS"
if [ -n "$FROM" ]; then
    SRC="$FROM"
    [ -d "$SRC/models" ] && SRC="$SRC/models"
    [ -f "$(find "$SRC" -mindepth 2 -maxdepth 2 -name model.xml | head -1)" ] || { echo "no models/<name>/model.xml under $FROM" >&2; exit 1; }
    echo "copying $SRC -> $MODELS"
    mkdir -p "$MODELS"
    cp -r "$SRC"/. "$MODELS"/
else
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/gso.XXXXXX")
    echo "cloning $REPO_URL (2 GB) into $TMP"
    git clone --depth 1 "$REPO_URL" "$TMP/repo"
    mkdir -p "$MODELS"
    cp -r "$TMP/repo/models"/. "$MODELS"/
    for f in LICENSE README.md VERSION; do [ -f "$ASSETS/$f" ] || cp "$TMP/repo/$f" "$ASSETS/$f"; done
    rm -rf "$TMP"
fi
HAVE=$(present)
echo "installed $HAVE objects under $MODELS"
if [ "$WANT" -gt 0 ] && [ "$HAVE" -lt "$WANT" ]; then
    echo "the catalog names $WANT objects but $HAVE are present" >&2
    exit 1
fi
[ "$WANT" -gt 0 ] || echo "no catalog yet: \$VENV_SIM_PY scripts/gso_catalog.py"
