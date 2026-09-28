#!/bin/bash
# Build a scene by the name experiment files use (configs/scenes.yaml): runs that entry's `build`, or with
# --isaac its Isaac stage's `isaac.build`, from the repository root. Once per machine; the result lands in the
# entry's `dir` (outputs/scene/...) or under assets/isaac/scene/.
#
#   bash scripts/build_scene.sh bottle
#   bash scripts/build_scene.sh warehouse_textured --isaac
#   bash scripts/build_scene.sh crate --dry-run
#
# The commands live in the registry, next to the directory they make, so a runbook, an error message and this
# script cannot name different recipes for the same scene.
set -eu
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

NAME="" PART=build DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --isaac) PART=isaac ;;
        --dry-run) DRY=1 ;;
        -*) echo "usage: $(basename "$0") <scene name> [--isaac] [--dry-run]" >&2; exit 1 ;;
        *) NAME="$1" ;;
    esac
    shift
done
[ -n "$NAME" ] || { echo "usage: $(basename "$0") <scene name> [--isaac] [--dry-run]" >&2; exit 1; }

CMD=$("$VENV_SIM_PY" - "$NAME" "$PART" <<'PY'
import sys, yaml
name, part = sys.argv[1:]
scenes = yaml.safe_load(open("configs/scenes.yaml"))
if name not in scenes:
    sys.exit(f"no scene called {name!r}; configs/scenes.yaml names {', '.join(sorted(scenes))}")
entry = scenes[name] if part == "build" else (scenes[name].get("isaac") or {})
if not entry.get("build"):
    what = "Isaac stage" if part == "isaac" else "build command"
    sys.exit(f"scene {name!r} has no {what} in configs/scenes.yaml")
print(entry["build"])
PY
)
echo "+ $CMD"
[ "$DRY" = 1 ] && exit 0
case "$CMD" in
    *'$USD_PY'*)  # the Isaac stage builders need a python with pxr; python3 has none unless usd-core is installed
        USD_PY=$(locomani_usd_python) || {
            echo "no python with the pxr module: pip install usd-core into python3, or export USD_PY=<such a python>" \
                "(Blender's bundled python has one) in \$LOCOMANI_WORKSPACE/locomani.local.sh" >&2
            exit 1
        }
        export USD_PY
        echo "  USD_PY=$USD_PY" ;;
esac
exec bash -c "set -eu; $CMD"
