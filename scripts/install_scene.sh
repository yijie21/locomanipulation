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
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAR="$ROOT/assets/scene_locomani_v1.tar.gz"
[ -f "$TAR" ] || { echo "missing $TAR"; exit 1; }

mkdir -p "$ROOT/outputs"
tar -xzf "$TAR" -C "$ROOT/outputs"

# The MJCF references meshes by absolute path (MuJoCo resolves relative paths
# against the XML, and the scene mixes robot meshes from third_party with
# object meshes from outputs). Retarget them to this checkout.
grep -rl "/workspace/code/locomanipulation" "$ROOT/outputs/scene" \
    --include="*.xml" --include="*.json" 2>/dev/null \
  | xargs -r sed -i "s#/workspace/code/locomanipulation#$ROOT#g"

echo "scene installed at $ROOT/outputs/scene"
