# Shell half of src/locomani/paths.py — source this, never hard-code a path.
#
#   . "$(dirname "$0")/../scripts/paths.sh"
#
# Same split as the Python module: the repo root is DERIVED (a fresh clone at
# any path works with zero configuration), everything outside the repo comes
# from $LOCOMANI_WORKSPACE (default /workspace). A collaborator whose home is
# /home/workspace sets that one variable:
#
#   export LOCOMANI_WORKSPACE=/home/workspace
#
# The three things that commonly live somewhere else entirely — conda, the
# TensorRT/ONNX Runtime deps, and the cache root — each keep their own
# override, because "under the workspace" is only the default, not a rule.
#
# Safe under `set -u` and sourceable from bash or zsh.

if [ -n "${BASH_SOURCE:-}" ]; then _lm_self="${BASH_SOURCE}"; else _lm_self="$0"; fi
export LOCOMANI_ROOT="${LOCOMANI_ROOT:-$(cd "$(dirname "$_lm_self")/.." && pwd)}"
unset _lm_self

export LOCOMANI_WORKSPACE="${LOCOMANI_WORKSPACE:-/workspace}"

# This machine's own settings, kept out of Git (e.g. a SONIC_DEPLOY_ROOT that points at a deploy
# built elsewhere). Read before the defaults below so what it sets wins; a fresh clone has none.
if [ -f "$LOCOMANI_WORKSPACE/locomani.local.sh" ]; then
    # shellcheck disable=SC1091
    . "$LOCOMANI_WORKSPACE/locomani.local.sh"
fi

# Inside the repo — derived, not configurable.
export SONIC_ROOT="$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl"
export VENV_SIM_PY="$SONIC_ROOT/.venv_sim/bin/python"
export VENV_DATA_PY="$SONIC_ROOT/.venv_data_collection/bin/python"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$SONIC_ROOT/gear_sonic_deploy}"
# Which SONIC checkpoint the deploy runs, relative to $SONIC_DEPLOY_ROOT. The
# DEFAULT IS v1.1 since 2026-09-11 (user ruling); policy/release is the older
# NVIDIA release checkpoint and is now only a comparison arm. One definition for
# the whole repo: locomani.stack.manager and locomani.paths all read
# it, so the offline encoder/decoder can never disagree with the running deploy.
# A token is a checkpoint-private latent -- see docs/reference/silent_failures.md.
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
export LOCOMANI_SRC="$LOCOMANI_ROOT/src/locomani"
# locomani is not pip-installed: every python that imports it finds it through src/ (added once).
case ":${PYTHONPATH:-}:" in
    *":$LOCOMANI_ROOT/src:"*) ;;
    *) export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export LOCOMANI_STACK="$LOCOMANI_ROOT/scripts/stack"
export LOCOMANI_OUTPUTS="$LOCOMANI_ROOT/outputs"

# Outside the repo — per-machine, each independently overridable.
export DEPS_DIR="${DEPS_DIR:-$LOCOMANI_WORKSPACE/deps}"
export CONDA_BASE="${CONDA_BASE:-$LOCOMANI_WORKSPACE/miniconda3}"
export LOCOMANI_CACHE="${LOCOMANI_CACHE:-$LOCOMANI_WORKSPACE/.cache}"
export LOCOMANI_TMP="${LOCOMANI_TMP:-$LOCOMANI_WORKSPACE/tmp}"
export LOCOMANI_SECRETS="${LOCOMANI_SECRETS:-$LOCOMANI_WORKSPACE/.secrets}"

# --- Isaac rendering runtime -------------------------------------------------
#
# SIMPLE is a checkout we pin, not vendor: it lives under the gitignored
# third_party/ like the other upstream clones (docs/setup.md step 7). See
# patches/simple.md for the pin and the patches. It is NOT a code dependency of the render path -- the offline
# re-renderer never imports `simple`; it converts our own scene_locomani.xml
# through Isaac's MJCF importer and drives the camera from MuJoCo FK.
export SIMPLE_ROOT="${SIMPLE_ROOT:-$LOCOMANI_ROOT/third_party/SIMPLE}"
export SIMPLE_PIN="${SIMPLE_PIN:-6d10628794d9c7de4596b4f2afb2c054a637c2bc}"
#
# The python that carries isaacsim. Today that is SIMPLE's venv, which pins
# isaacsim[all,extscache]==4.5.0 in its pyproject.toml. Deliberately a separate
# variable from SIMPLE_ROOT: 3DGS/NuRec needs Isaac Sim 5.0+, so the render
# runtime must be able to move to a newer isaacsim WITHOUT waiting for SIMPLE
# to lift its pin. Point RENDER_PY at another venv and nothing else changes.
export RENDER_PY="${RENDER_PY:-$SIMPLE_ROOT/.venv/bin/python}"
export ISAAC_ASSETS="${ISAAC_ASSETS:-$LOCOMANI_ROOT/assets/isaac}"
# A python with `pxr` for the Isaac stage builders in configs/scenes.yaml. They only write USD text, so neither
# Isaac nor the render venv is needed (and neither has a bare pxr: Kit loads it). Empty means "let
# locomani_usd_python probe"; set it (locomani.local.sh) only when the probe picks wrong or finds nothing.
export USD_PY="${USD_PY:-}"

# The python `pxr` imports in: the first of $USD_PY, python3 (pip install usd-core) and the python bundled with the
# `blender` on PATH (Blender ships pxr) that imports it. $USD_PY is checked too, not trusted: a shell that sourced an
# older paths.sh still carries its `python3` default. Prints nothing (and fails) when none imports it.
locomani_usd_python() {
    _lm_blender="$(command -v blender 2>/dev/null || true)"
    _lm_seen=""
    for _lm_py in "${USD_PY:-}" python3 "${_lm_blender:+$(dirname "$(readlink -f "$_lm_blender")")}"/*/python/bin/python3*; do
        [ -n "$_lm_py" ] && [ "$_lm_py" != "$_lm_seen" ] || continue
        if command -v "$_lm_py" >/dev/null 2>&1 && "$_lm_py" -c "import pxr" >/dev/null 2>&1; then
            echo "$_lm_py"; unset _lm_blender _lm_py _lm_seen; return 0
        fi
        [ -n "$_lm_seen" ] || [ -z "${USD_PY:-}" ] || echo "USD_PY=$USD_PY has no pxr module; trying the others" >&2
        _lm_seen="$_lm_py"
    done
    unset _lm_blender _lm_py _lm_seen
    return 1
}
#
# --- Unreal rendering (only for `images.source: unreal`; docs/setup.md, the optional Unreal step) ---
# One directory holds the Unreal install: the engine (UE 5.7.4), the host project with the URLab plugin,
# the bridge venv (python + urlab_client + mujoco 3.10.0, URLab's pin) and the editor's HOME, caches and logs.
# scripts/setup_unreal.sh builds it; a machine with an existing install points these at it
# ($LOCOMANI_WORKSPACE/locomani.local.sh).
export UNREAL_ROOT="${UNREAL_ROOT:-$LOCOMANI_ROOT/third_party/unreal}"
export UE_ENGINE="${UE_ENGINE:-$UNREAL_ROOT/UE_5.7.4}"
export UE_PROJECT="${UE_PROJECT:-$UNREAL_ROOT/host/URLabHost.uproject}"
export BRIDGE_PY="${BRIDGE_PY:-$UNREAL_ROOT/bridge/.venv/bin/python}"
# Per level: the MJCF export it was built from and its source.json (the scene XML's hash). Beside the host project
# that holds the levels, not in a checkout: the levels are the machine's, and every checkout must find their records
# (under $LOCOMANI_ROOT a level built from a worktree was unknown to the main checkout, 2026-09-29).
export UNREAL_ASSETS="${UNREAL_ASSETS:-$UNREAL_ROOT/levels}"
export ROBOT_MJCF="${ROBOT_MJCF:-$SONIC_ROOT/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_dex1.xml}"

# Logs + exporter sessions for instance $1 (default $INSTANCE, default 0).
# Mirrors paths.session_root() so multi-instance runs never collide.
locomani_session_root() {
    _lm_i="${1:-${INSTANCE:-0}}"
    if [ "$_lm_i" = "0" ]; then echo "$LOCOMANI_TMP/p03"
    else echo "$LOCOMANI_TMP/p03_i$_lm_i"; fi
}

# --- ZMQ ports and the DDS domain, derived from INSTANCE, in ONE place -------
#
# Every one of these used to be re-derived by whoever needed it: the old start_stack.sh
# had all six, the old restart_deploy.sh two, par_batch.sh three, and the Python side
# read them from the environment with instance-0 defaults baked in. Which meant
# a launcher that did not export them -- collect_pilot.sh and collect.sh
# mention INSTANCE nowhere -- silently handed grasp_gym port 5556, i.e. THE
# STACK BELONGING TO INSTANCE 0, however the stack itself had been started.
# Same shape as the DDS-domain trap, one layer up: everything connects and
# nothing complains, it is just the wrong stack.
#
# Derived here so `INSTANCE=k` means one thing everywhere. Assignments are
# unconditional on purpose: an inherited TF_SONIC_PORT from a different
# instance is exactly the bug being closed, so it must not win.
export INSTANCE="${INSTANCE:-0}"
export SONIC_DDS_DOMAIN="$INSTANCE"
export TF_CAMERA_PORT=$((5555 + 10 * INSTANCE))      # sim  -> exporter/agent
export TF_SONIC_PORT=$((5556 + 10 * INSTANCE))       # agent -> deploy
export TF_OUT_PORT=$((5557 + 10 * INSTANCE))         # deploy -> exporter/agent
export TF_KEYBOARD_PORT=$((5580 + 10 * INSTANCE))    # agent -> exporter
export TF_PRIVILEGED_PORT=$((5581 + 10 * INSTANCE))  # sim  -> agent
export TF_EPISODE_PORT=$((5582 + 10 * INSTANCE))     # agent -> sim
# NOTE: TF_KEYBOARD_PORT for INSTANCE=1 is 5590, which is serve_policy.py's
# default port. Parallel collection and closed-loop evaluation cannot run at
# the same time without moving one of them (docs/reference.md#端口).
