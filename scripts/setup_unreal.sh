#!/bin/bash
# Install the Unreal side of `images.source: unreal` (optional; docs/setup.md): the host project with the patched
# URLab plugin, built, and the bridge venv. Idempotent: rerun after a failure, each step skips what is done.
#
#   bash scripts/setup_unreal.sh                  # into $UNREAL_ROOT (default third_party/unreal)
#
# Before: Unreal Engine 5.7.4 for Linux unzipped at $UE_ENGINE (default $UNREAL_ROOT/UE_5.7.4). Epic hands the
# Linux build out only to a logged-in account (https://www.unrealengine.com/linux, "Linux_Unreal_Engine_5.7.4.zip",
# ~26 GB zipped, 69 GB unzipped), so this script cannot fetch it. Not 5.8: its Vulkan RHI regressed. Also git,
# uv (https://docs.astral.sh/uv/), clang from the engine's toolchain (bundled), ~15 GB more disk.
# After: bash scripts/start_unreal.sh --gpu N, then the room level once and a level per scene:
#   "$BRIDGE_PY" -m locomani.sim.unreal.build_level --room <bake dir>      # bake: scripts/unreal/warehouse_bake.py
#   bash scripts/build_scene.sh warehouse_textured --unreal
#
# What it builds, and why each pin:
#   $UE_PROJECT            the host project, from scripts/unreal/host/ (URLabHost renamed to the .uproject's name):
#                          render defaults in Config/ (no Nanite shadows, 8 captures a request, 2 cm near clip, no
#                          crash reports to Epic, no editor CPU throttling)
#   Plugins/UnrealRoboticsLab   URLab at URLAB_PIN, branch loco-fresh-capture, patches/urlab_fresh_capture.patch
#                          applied (the "fresh" / "stream" captures, fov and near-clip fixes; patches/*.md), then
#                          its third_party (MuJoCo 3.10.0, CoACD, libzmq; ~10 min) and the editor target
#   $UNREAL_ROOT/bridge    urlab_bridge at BRIDGE_PIN with its venv ($BRIDGE_PY): Python 3.12, mujoco==3.10.0 (URLab's
#                          MuJoCo; the handshake refuses another), pyyaml for this repo's modules, the mesh
#                          packages URLab's importer runs (trimesh, scipy, networkx, Pillow), and OpenCV for the
#                          dataset fill and third-person video (view_render, sim_video): the plugin's
#                          Config/LocalUnrealRoboticsLab.ini points it at this venv. Without it the import asks, in a
#                          dialog an unattended editor cancels, to pip-install them into UE's own Python.
# Builds run on UE_CPUS (default the last 4 cores): the container's CPU quota is shared with the SONIC stacks.
set -eu
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
URLAB_REPO=https://github.com/URLab-Sim/UnrealRoboticsLab.git
URLAB_PIN=efbc5e2
BRIDGE_REPO=https://github.com/URLab-Sim/urlab_bridge.git
BRIDGE_PIN=c1fb34e
N=$(nproc)
UE_CPUS=${UE_CPUS:-$((N > 4 ? N - 4 : 0))-$((N - 1))}
HOST=$(dirname "$UE_PROJECT"); NAME=$(basename "$UE_PROJECT" .uproject)
PLUGIN="$HOST/Plugins/UnrealRoboticsLab"
BRIDGE_DIR="$(dirname "$(dirname "$(dirname "$BRIDGE_PY")")")"  # .../bridge/.venv/bin/python -> .../bridge
step() { echo "=== $*"; }

step "engine $UE_ENGINE"
[ -x "$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor" ] || {
    echo "no UnrealEditor under $UE_ENGINE: download Linux_Unreal_Engine_5.7.4.zip from https://www.unrealengine.com/linux (Epic login) and unzip it there" >&2
    exit 1
}

step "host project $UE_PROJECT"
if [ ! -f "$UE_PROJECT" ]; then
    mkdir -p "$HOST"
    (cd "$LOCOMANI_ROOT/scripts/unreal/host" && find . -type f) | while read -r f; do
        dst="$HOST/${f//URLabHost/$NAME}"
        mkdir -p "$(dirname "$dst")"
        sed "s/URLabHost/$NAME/g" "$LOCOMANI_ROOT/scripts/unreal/host/$f" > "$dst"
    done
fi

step "URLab $URLAB_PIN + patches/urlab_fresh_capture.patch"
[ -d "$PLUGIN/.git" ] || git clone "$URLAB_REPO" "$PLUGIN"
if ! git -C "$PLUGIN" rev-parse -q --verify loco-fresh-capture > /dev/null; then
    git -C "$PLUGIN" checkout -q -b loco-fresh-capture "$URLAB_PIN"
    git -C "$PLUGIN" apply "$LOCOMANI_ROOT/patches/urlab_fresh_capture.patch"
    git -C "$PLUGIN" -c user.name=locomani -c user.email=locomani@localhost commit -q -am \
        "loco fresh capture (locomanipulation patches/urlab_fresh_capture.patch)"
fi
[ "$(git -C "$PLUGIN" diff "$URLAB_PIN" loco-fresh-capture | git patch-id --stable | cut -d' ' -f1)" = \
  "$(git patch-id --stable < "$LOCOMANI_ROOT/patches/urlab_fresh_capture.patch" | cut -d' ' -f1)" ] ||
    { echo "$PLUGIN branch loco-fresh-capture is not $URLAB_PIN + the repo's patch; fix it by hand" >&2; exit 1; }

step "URLab third_party (MuJoCo, CoACD, libzmq)"
if [ ! -d "$PLUGIN/third_party/install" ]; then
    taskset -c "$UE_CPUS" bash "$PLUGIN/third_party/build_all.sh" --engine "$UE_ENGINE"
fi

step "editor target ${NAME}Editor"
taskset -c "$UE_CPUS" bash "$UE_ENGINE/Engine/Build/BatchFiles/Linux/Build.sh" "${NAME}Editor" Linux Development \
    "-Project=$UE_PROJECT" -WaitMutex

step "bridge $BRIDGE_PIN at $BRIDGE_DIR"
[ -d "$BRIDGE_DIR/.git" ] || git clone "$BRIDGE_REPO" "$BRIDGE_DIR"
git -C "$BRIDGE_DIR" checkout -q "$BRIDGE_PIN"
[ -x "$BRIDGE_PY" ] || uv venv --python 3.12 "$BRIDGE_DIR/.venv"
# --no-deps for the package itself: its pyproject pins mujoco 3.8.1, and URLab's handshake wants its own 3.10.0
uv pip install --python "$BRIDGE_PY" --no-deps -e "$BRIDGE_DIR"
uv pip install --python "$BRIDGE_PY" "mujoco==3.10.0" "pyzmq>=25" "numpy>=1.26" "msgpack>=1.0" "gymnasium>=1.0" pyyaml \
    "trimesh==5.1.0" "scipy==1.18.1" "networkx==3.7" "pillow==12.3.0" "opencv-python-headless==4.11.0.86"
"$BRIDGE_PY" -c "import mujoco, urlab_client, yaml, trimesh, scipy, networkx, PIL, cv2; print('bridge ok: mujoco', mujoco.__version__)"
# PortBase: every plugin port derives from it (the patch; state = base ... step RPC = base + 4); 5800 is clear of
# the MuJoCo stacks' instance blocks. locomani.sim.unreal.editor reads the same file.
printf '[PythonSettings]\nPythonPath=%s\n\n[BridgeServer]\nPortBase=5800\n' "$BRIDGE_PY" > "$PLUGIN/Config/LocalUnrealRoboticsLab.ini"
echo "done: bash scripts/start_unreal.sh --gpu N"
