#!/bin/bash
# Unpack the pre-built simulation scene into outputs/scene and rewrite the
# absolute asset paths inside the MJCF to match THIS checkout's location.
#
# The scene was generated once with EmbodiedGen (slow: text-to-3D + layout +
# physics injection). Shipping the finished scene lets a fresh clone run the
# data-collection sim immediately.
#
# Usage:  bash scripts/install_scene.sh
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
ROOT="$LOCOMANI_ROOT"
TAR="$ROOT/assets/scene_locomani_v1.tar.gz"
[ -f "$TAR" ] || { echo "missing $TAR"; exit 1; }

mkdir -p "$ROOT/outputs"
tar -xzf "$TAR" -C "$ROOT/outputs"

# The MJCF references meshes by absolute path (MuJoCo resolves relative paths
# against the XML, and the scene mixes robot meshes from third_party with
# object meshes from outputs). Retarget them to this checkout.
#
# The prefix is DETECTED, not assumed: the tarball carries whatever repo path
# it was packed on, and hard-coding that path here means the next person to
# re-pack the scene from a different checkout silently ships a tarball this
# script cannot fix.
BAKED=$(grep -rhoE '/[^"'"'"' <>]+/outputs/scene/' "$ROOT/outputs/scene" \
        --include="*.xml" --include="*.json" 2>/dev/null \
        | head -1 | sed 's#/outputs/scene/$##')
if [ -z "$BAKED" ]; then
    echo "no absolute paths found in the scene — nothing to retarget"
elif [ "$BAKED" = "$ROOT" ]; then
    echo "scene already targets $ROOT"
else
    echo "retargeting $BAKED -> $ROOT"
    grep -rl "$BAKED" "$ROOT/outputs/scene" \
        --include="*.xml" --include="*.json" 2>/dev/null \
      | xargs -r sed -i "s#$BAKED#$ROOT#g"
fi

# Rebuild the simulation scene from the unpacked source assets rather than
# shipping a built one. The tarball carries mjcf/ and layout/ -- the parts
# EmbodiedGen is slow to produce -- while the arrangement, the surface
# heights and the primitive target objects all live in scene_lib.py and
# change often. A prebuilt scene_locomani.xml in the tarball goes stale the
# first time anyone edits that file, and the failure is quiet: the sim runs,
# the new target objects are simply absent. The rebuild costs ~1 s.
PY="$SONIC_ROOT/.venv_sim/bin/python"
if [ -x "$PY" ]; then
    "$PY" "$LOCOMANI_SRC/build_sim_scene.py" \
        "$ROOT/outputs/scene" "$ROOT/outputs/scene/sim" \
        | tail -n 3
else
    echo
    echo "NOTE: $PY not found, so the scene was NOT rebuilt and still holds"
    echo "      whatever scene_lib.py looked like when the tarball was packed."
    echo "      After creating the sim venv, run:"
    echo "        \$PY src/teleop_free/build_sim_scene.py \\"
    echo "            outputs/scene outputs/scene/sim"
fi

echo "scene installed at $ROOT/outputs/scene"
