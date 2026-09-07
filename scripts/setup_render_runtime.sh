#!/bin/bash
# Bring up the Isaac rendering runtime: check the SIMPLE pin, apply our
# patches, verify the render python can import isaacsim.
#
#   bash scripts/setup_render_runtime.sh [--with-replay] [--with-alignment] [--deep]
#
# Idempotent: run it twice and the second run reports "already applied" for
# every patch and changes nothing. Applies ONLY the required patch by default;
# the optional ones belong to SIMPLE's own replay CLI and to the MuJoCo/Isaac
# alignment harness, neither of which the collection or eval path needs.
#
# Overrides: SIMPLE_ROOT (checkout location), SIMPLE_PIN (expected commit),
# RENDER_PY (the python carrying isaacsim), SIMPLE_ALLOW_DRIFT=1 (proceed on a
# commit mismatch instead of refusing).
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

PATCH_DIR="$LOCOMANI_ROOT/patches/simple"
WANT_REPLAY=0; WANT_ALIGN=0; WANT_DEEP=0
for a in "$@"; do
    case "$a" in
        --with-replay)    WANT_REPLAY=1 ;;
        --with-alignment) WANT_ALIGN=1 ;;
        --deep)           WANT_DEEP=1 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 1. the checkout ---------------------------------------------------------
[ -d "$SIMPLE_ROOT/.git" ] || fail "no SIMPLE checkout at $SIMPLE_ROOT
  clone it, or set SIMPLE_ROOT:
    git clone https://github.com/physical-superintelligence-lab/SIMPLE.git $SIMPLE_ROOT
    git -C $SIMPLE_ROOT checkout $SIMPLE_PIN"

have=$(git -C "$SIMPLE_ROOT" rev-parse HEAD)
if [ "$have" != "$SIMPLE_PIN" ]; then
    if [ "${SIMPLE_ALLOW_DRIFT:-0}" = "1" ]; then
        echo "WARN: SIMPLE at $have, pin is $SIMPLE_PIN (SIMPLE_ALLOW_DRIFT=1, continuing)"
    else
        fail "SIMPLE is at $have but the pin is $SIMPLE_PIN
  git -C $SIMPLE_ROOT checkout $SIMPLE_PIN
  or re-pin in scripts/paths.sh, or set SIMPLE_ALLOW_DRIFT=1"
    fi
fi
echo "checkout  $SIMPLE_ROOT @ ${have:0:12}"

# --- 2. the patches ----------------------------------------------------------
# Three states per patch, decided by git apply's own checks, never by grepping
# for a marker string: reverse-check clean means it is already in, forward-check
# clean means it can go in, neither means the tree has drifted and a silent
# re-apply would corrupt it.
apply_patch() {
    p="$PATCH_DIR/$1"; label="$2"
    [ -f "$p" ] || fail "missing patch $p"
    if git -C "$SIMPLE_ROOT" apply --reverse --check "$p" 2>/dev/null; then
        echo "  = $1  already applied ($label)"
    elif git -C "$SIMPLE_ROOT" apply --check "$p" 2>/dev/null; then
        git -C "$SIMPLE_ROOT" apply "$p" || fail "apply failed: $1"
        echo "  + $1  applied ($label)"
    else
        fail "$1 neither applies nor is applied -- $SIMPLE_ROOT has drifted.
  inspect with: git -C $SIMPLE_ROOT diff"
    fi
}

echo "patches"
apply_patch 0001-mujoco-renderer-get.patch "required"
[ "$WANT_REPLAY" = 1 ] && apply_patch 0002-replay-video-recorder.patch "replay CLI"
if [ "$WANT_ALIGN" = 1 ]; then
    apply_patch 0004-dex1-new-files.patch    "alignment harness files"
    apply_patch 0003-dex1-registration.patch "alignment harness registration"
fi

# --- 3. the render python ----------------------------------------------------
[ -x "$RENDER_PY" ] || fail "RENDER_PY is not executable: $RENDER_PY"
# Resolve the module WITHOUT executing it. A real `import isaacsim` boots Kit
# and takes minutes, which is not what this check is for -- it would make the
# setup script unusable as a fast precondition. --deep does the real import.
ver=$("$RENDER_PY" -c '
try:
    import importlib.metadata as md, importlib.util as iu
    v = md.version("isaacsim")
    print(v if iu.find_spec("isaacsim") is not None else "ERR spec not found")
except Exception as e:
    print("ERR", e)
' 2>&1)
case "$ver" in
    ERR*) fail "$RENDER_PY cannot resolve isaacsim: $ver" ;;
esac
if [ "$WANT_DEEP" = 1 ]; then
    echo "  deep import (boots Kit, minutes)..."
    OMNI_KIT_ACCEPT_EULA=YES timeout 900 "$RENDER_PY" -c "import isaacsim" >/dev/null 2>&1 \
        || fail "deep import of isaacsim failed or timed out"
    echo "  deep import OK"
fi
echo "render py $RENDER_PY"
echo "isaacsim  $ver"

# 3DGS / NuRec needs 5.0+; 4.5 is expected today, this is a note not an error.
case "$ver" in
    4.*) echo "note      isaacsim 4.x has no NuRec/3DGS (needs 5.0+); see ticket 12" ;;
esac
echo "render runtime OK"
