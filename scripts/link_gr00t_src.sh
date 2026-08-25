#!/bin/bash
# Point the Isaac-GR00T venv's `import gr00t` at src/gr00t (the vendored,
# committed copy) instead of the gitignored third_party clone.
#
#   bash scripts/link_gr00t_src.sh
#
# Mechanism: the venv was created with `pip install -e` from the upstream
# clone, which drops an __editable__*.pth + finder module into site-packages
# that hardcode the clone's path. Rather than fight the finder, this replaces
# the .pth content with a plain directory line -- the oldest, least magical
# form of .pth -- so `src/` lands on sys.path and `src/gr00t` wins.
set -eu
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"

VENV="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv"
SP=$(ls -d "$VENV"/lib/python3.*/site-packages | head -1)
PTH="$SP/__editable__.gr00t-0.1.0.pth"
[ -f "$PTH" ] || { echo "no editable pth at $PTH -- was the venv made with 'pip install -e'?"; exit 1; }

echo "$LOCOMANI_ROOT/src" > "$PTH"
# The finder module the old .pth imported is now unreferenced; leave it.

GOT=$("$VENV/bin/python" -c "import gr00t; print(gr00t.__file__)")
case "$GOT" in
    "$LOCOMANI_ROOT/src/gr00t/"*) echo "ok: gr00t -> $GOT" ;;
    *) echo "FAILED: gr00t still resolves to $GOT"; exit 1 ;;
esac
