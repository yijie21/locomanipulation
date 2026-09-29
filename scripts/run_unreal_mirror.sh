#!/bin/bash
# Render the policy's camera images in Unreal, from the eval stack's live MuJoCo state. Foreground: Ctrl-C stops it.
#
#   bash scripts/run_unreal_mirror.sh outputs/runs/<run>
#
# Only for a run whose experiment sets `images.source: unreal`, and only after the Unreal editor
# (scripts/start_unreal.sh --gpu N, which chooses the card) and terminal 1 (scripts/start_stack.sh eval) are up:
# the mirror opens the scene's level in the editor and renders the stack's scene from its privileged state. Start
# it before the rounds; the policy client asks it for images on every observation (docs/rendering.md).
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--dry-run]" >&2; exit 1; }
[ -x "$BRIDGE_PY" ] || { echo "BRIDGE_PY not executable: $BRIDGE_PY (scripts/setup_unreal.sh)" >&2; exit 1; }

exec "$BRIDGE_PY" -m locomani.sim.unreal.mirror "$@"
