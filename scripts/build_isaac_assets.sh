#!/bin/bash
# Build the USD assets the Isaac re-render needs, into $ISAAC_ASSETS
# (assets/isaac/, gitignored -- ~700 MB, rebuildable, never committed).
#
#   bash scripts/build_isaac_assets.sh                     # robot + the working scene
#   bash scripts/build_isaac_assets.sh --snapshot <dir>    # scene from a batch snapshot
#   bash scripts/build_isaac_assets.sh --robot-only
#   bash scripts/build_isaac_assets.sh --force             # ignore the caches
#
# TWO LAYERS, because they change at different rates:
#
#   robot  : imported from the robot-only MJCF ($ROBOT_MJCF). Scene-independent
#            -- it is the same articulation whatever the table height is -- so
#            it is built ONCE and cached on the MJCF's hash.
#   scene  : room + surfaces + objects. collect.sh rebuilds outputs/scene/sim
#            per batch from SCENE_SEED and the surface tops move with it
#            (0.620-0.646 m for surface A across the cover400 snapshots, 0.647
#            to 0.855 for surface B), so this is cached per scene_meta.json
#            hash and keyed by a label, normally the batch's seed.
#
# Both steps boot Kit, which is minutes each. A cache hit costs nothing, so
# re-running this per episode in a collection loop is fine.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

SNAPSHOT=""; ROBOT_ONLY=0; FORCE=0; LABEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --snapshot)   SNAPSHOT="$2"; shift 2 ;;
        --label)      LABEL="$2"; shift 2 ;;
        --robot-only) ROBOT_ONLY=1; shift ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

fail() { echo "FAIL: $*" >&2; exit 1; }
[ -x "$RENDER_PY" ] || fail "RENDER_PY not executable: $RENDER_PY
  run scripts/setup_render_runtime.sh first"

sha() { sha256sum "$1" 2>/dev/null | cut -c1-16; }
secs() { date +%s; }

# --- robot -------------------------------------------------------------------
ROBOT_DIR="$ISAAC_ASSETS/robot"
ROBOT_USD="$ROBOT_DIR/g1_dex1.usda"
[ -f "$ROBOT_MJCF" ] || fail "robot MJCF not found: $ROBOT_MJCF"
# The key covers the post-processor too: a change to split_visual_refs.py
# changes the artifact without touching the MJCF.
SPLITTER="$LOCOMANI_ROOT/src/locomani/sim/isaac/split_visual_refs.py"
want="$(sha "$ROBOT_MJCF")-$(sha "$SPLITTER")"
if [ "$FORCE" = 0 ] && [ -f "$ROBOT_USD" ] && [ "$(cat "$ROBOT_DIR/.stamp" 2>/dev/null)" = "$want" ]; then
    echo "robot  = cached  $ROBOT_USD  (mjcf $want)"
else
    mkdir -p "$ROBOT_DIR"
    echo "robot  + building from $(basename "$ROBOT_MJCF") ..."
    # Remove the old artifact first. Without this a failed convert leaves the
    # PREVIOUS usd in place, the -s check passes, and the build reports success
    # while nothing was rebuilt -- which is how a crash in the importer stage
    # went unnoticed on 2026-09-06.
    rm -f "$ROBOT_USD" "$ROBOT_DIR/.stamp"
    t0=$(secs)
    # NOTE: locomani.sim.isaac.mjcf_split writes a render-only variant that would
    # avoid the importer's mesh-merge problem (see isaac_convert). It is NOT
    # used: Isaac 4.5's MJCF importer aborts (SIGABRT) on the jointless child
    # bodies it introduces, with and without an explicit inertial. Kept because
    # the variant itself is correct -- 41 bodies, same 34 joints, same 77 geoms,
    # kinematics identical to 0.0000 mm -- and a newer importer may accept it.
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$LOCOMANI_ROOT/src" \
        "$RENDER_PY" -m locomani.sim.isaac.isaac_convert \
        "$ROBOT_MJCF" "$ROBOT_USD" >/dev/null 2>&1 \
        || fail "robot import failed; see $ROBOT_DIR/convert.log"
    [ -s "$ROBOT_USD" ] || fail "robot import produced no USD; see $ROBOT_DIR/convert.log"
    grep -q "^FAILED" "$ROBOT_DIR/convert.log" 2>/dev/null && \
        fail "robot import logged a failure; see $ROBOT_DIR/convert.log"
    # The importer composes every visual mesh of a multi-mesh body onto ONE
    # prim, and USD keeps only one of the same-named `mesh` children -- the
    # wrist cylinders, the pelvis shell and three torso meshes were simply
    # absent. Rewrite the layer so each mesh has its own child prim. Plain
    # python on the .usda text; no Kit, no pxr.
    python3 "$SPLITTER" "$ROBOT_USD" "$ROBOT_USD" | tee -a "$ROBOT_DIR/convert.log" \
        | grep -E '^split [0-9]+ multi' \
        || fail "split_visual_refs failed; see $ROBOT_DIR/convert.log"
    echo "$want" > "$ROBOT_DIR/.stamp"
    echo "robot  + done in $(( $(secs) - t0 ))s  $(du -h "$ROBOT_USD" | cut -f1)"
fi
[ "$ROBOT_ONLY" = 1 ] && { echo "robot only, done"; exit 0; }

# --- scene -------------------------------------------------------------------
# The meta is the per-seed truth; the asset library it draws OBJs from
# (outputs/scene) is seed-independent.
if [ -n "$SNAPSHOT" ]; then
    [ -d "$SNAPSHOT" ] || fail "no such snapshot dir: $SNAPSHOT"
    META="$SNAPSHOT/scene_meta.json"
    [ -n "$LABEL" ] || LABEL="$(basename "$SNAPSHOT" | sed 's/^scene_//')"
else
    META="$LOCOMANI_OUTPUTS/scene/sim/scene_meta.json"
    [ -n "$LABEL" ] || LABEL="work"
fi
[ -f "$META" ] || fail "no scene_meta.json at $META"
SCENE_SRC_XML="$(dirname "$META")/scene_locomani.xml"
[ -f "$SCENE_SRC_XML" ] || fail "no scene_locomani.xml next to $META"

SCENE_DIR="$ISAAC_ASSETS/scene/$LABEL"
want=$(sha "$META")
if [ "$FORCE" = 0 ] && [ -f "$SCENE_DIR/scene.usda" ] && \
   [ "$(cat "$SCENE_DIR/.stamp" 2>/dev/null)" = "$want" ]; then
    echo "scene  = cached  $SCENE_DIR/scene.usda  (meta $want)"
    exit 0
fi
mkdir -p "$SCENE_DIR"
echo "scene  + building '$LABEL' from $META ..."
t0=$(secs)
SCENE_META="$META" SCENE_XML="$SCENE_SRC_XML" OMNI_KIT_ACCEPT_EULA=YES \
    "$RENDER_PY" src/locomani/sim/isaac_scene.py \
    "$LOCOMANI_OUTPUTS/scene" "$SCENE_DIR" > "$SCENE_DIR/kit.log" 2>&1 \
    || fail "scene build failed; see $SCENE_DIR/kit.log and build.log"
[ -s "$SCENE_DIR/scene.usda" ] || fail "scene build produced no scene.usda; see $SCENE_DIR/kit.log"
echo "$want" > "$SCENE_DIR/.stamp"
echo "scene  + done in $(( $(secs) - t0 ))s  $(du -sh "$SCENE_DIR" | cut -f1)"
# isaac_scene.py writes its own build.log (Isaac swallows print into its
# logger), so Kit's stream goes to kit.log to keep the two from clobbering
# each other -- they used to be the same path and the summary was lost.
grep -E "scene_meta:|^  surface [AB]:" "$SCENE_DIR/build.log" | sed 's/^/  /' 
