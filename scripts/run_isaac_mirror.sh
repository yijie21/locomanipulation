#!/bin/bash
# Render the policy's camera images in Isaac, from the eval stack's live MuJoCo state. Foreground: Ctrl-C frees the card.
#
#   bash scripts/run_isaac_mirror.sh outputs/runs/<run> --gpu 1
#
# Only for a run whose experiment sets `images.source: isaac`, and only after terminal 1
# (scripts/start_stack.sh eval) is up: the mirror renders that stack's scene from its privileged
# state. Start it before the rounds (terminal 3); the policy client asks it for images on every
# observation. Its own GPU: the render is ~90 ms of RTX per request (docs/rendering.md).
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> --gpu N [--dry-run]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RENDER_PY" -m locomani.sim.isaac.mirror "$@"
