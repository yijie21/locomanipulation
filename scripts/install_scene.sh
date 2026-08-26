#!/bin/bash
# Unpack the pre-built simulation scene into outputs/scene and rewrite the
# absolute asset paths inside the MJCF to match THIS checkout's location.
#
# The scene was generated once with EmbodiedGen (slow: text-to-3D + layout +
# physics injection). Shipping the finished scene lets a fresh clone run the
# data-collection sim immediately.
#
# Usage:  bash scripts/install_scene.sh
#         SCENE_PRESET=bottle_only bash scripts/install_scene.sh
#
# SCENE_PRESET is read at REBUILD time (scene_lib.py, build_sim_scene.py), not
# when the stack starts, so it has to be on THIS command. Getting it wrong is
# silent by nature -- the sim runs fine, it just holds a different table -- so
# this script now prints what it actually built.
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

# Rebuild the simulation scene from the unpacked source assets, unconditionally.
#
# The tarball DOES carry a built scene/sim/ (scene_locomani.xml +
# scene_meta.json) alongside the mjcf/ and layout/ sources -- but it is a
# snapshot of whatever scene_lib.py looked like when the tarball was packed,
# and the arrangement, surface heights and primitive targets all live in
# scene_lib.py and change often. So the shipped one is treated as disposable
# and overwritten every time. The rebuild costs ~1 s.
#
# Right now the two happen to agree: a default rebuild produces exactly the
# baked scene (target_object=can, target_pool=[can, bottle, cup, jar], and the
# 8 objects bottle/can/cup/fork/jar/napkin/orange/plate). That coincidence is
# why the skip branch below has never bitten anyone -- and why it will, quietly,
# the first time scene_lib.py changes. Do not read it as "the baked scene is
# fine".
PY="$SONIC_ROOT/.venv_sim/bin/python"
if [ -x "$PY" ]; then
    "$PY" "$LOCOMANI_SRC/build_sim_scene.py" \
        "$ROOT/outputs/scene" "$ROOT/outputs/scene/sim" \
        | tail -n 3
    # Echo the result. A scene built with the wrong SCENE_PRESET is otherwise
    # indistinguishable from a right one until grasp_gym says "targets not in
    # the scene" -- or worse, does not, and quietly collects a whole dataset
    # against four rotating targets and a fork.
    "$PY" - "$ROOT/outputs/scene/sim/scene_meta.json" <<'EOF'
import json, sys, os
m = json.load(open(sys.argv[1]))
print(f"  preset      {os.environ.get('SCENE_PRESET') or '<none> (default multi-object table)'}")
print(f"  target      {m.get('target_object')}")
print(f"  target_pool {m.get('target_pool') or [m.get('target_object')]}")
print(f"  on the table {sorted(m.get('objects', {}))}")
EOF
else
    echo
    echo "=========================================================="
    echo "WARNING: THE SCENE WAS NOT REBUILT."
    echo "=========================================================="
    echo "  $PY does not exist, so the rebuild step was skipped and"
    echo "  outputs/scene/sim still holds whatever scene_lib.py looked like"
    echo "  when the tarball was PACKED -- the old multi-object table, and"
    echo "  NOT whatever SCENE_PRESET you just passed."
    echo
    echo "  Collecting against it produces a dataset for a different task"
    echo "  than you think, and nothing downstream will tell you."
    echo
    echo "  Create the sim venv (docs/setup.md step 2), then re-run THIS"
    echo "  script with the same preset:"
    echo
    echo "    ${SCENE_PRESET:+SCENE_PRESET=$SCENE_PRESET }bash scripts/install_scene.sh"
    echo
    # Non-zero on purpose. The unpack and retarget above did succeed, but the
    # scene on disk is not the one that was asked for, and the whole point of
    # this branch is that nobody notices. Nothing calls this script
    # programmatically, so failing loudly costs nothing.
    exit 1
fi

echo "scene installed at $ROOT/outputs/scene"
