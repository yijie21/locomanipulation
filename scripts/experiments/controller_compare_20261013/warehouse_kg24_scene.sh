#!/bin/bash
# TODO 25.3 W4 extrapolation tier: outputs/scene/warehouse_box_45x32x33_mars with the box at 2.4 kg (the inertial's
# mass doubled, its diagonal inertia doubled with it); meshes and textures are links to the source scene.
set -euo pipefail
cd "$(dirname "$0")/../../.."
S=outputs/scene/warehouse_box_45x32x33_mars; O=${S}_kg24
mkdir -p "$O"
for f in "$S"/*; do
  n=$(basename "$f")
  case $n in scene_locomani.xml) ;; *.json) cp "$f" "$O/$n" ;; *) ln -sfn "$PWD/$f" "$O/$n" ;; esac
done
python3 - "$S/scene_locomani.xml" "$O/scene_locomani.xml" <<'PY'
import sys
s = open(sys.argv[1]).read()
old = '<inertial pos="0 0 0.165" mass="1.2" diaginertia="0.02113 0.03114 0.03049" />'
assert s.count(old) == 1, "box inertial not found"
open(sys.argv[2], "w").write(s.replace(old, '<inertial pos="0 0 0.165" mass="2.4" diaginertia="0.04226 0.06228 0.06098" />'))
PY
echo "$O"
