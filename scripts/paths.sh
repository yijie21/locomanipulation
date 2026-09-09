# Shell half of src/teleop_free/paths.py — source this, never hard-code a path.
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

# Inside the repo — derived, not configurable.
export SONIC_ROOT="$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl"
export VENV_SIM_PY="$SONIC_ROOT/.venv_sim/bin/python"
export VENV_DATA_PY="$SONIC_ROOT/.venv_data_collection/bin/python"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$SONIC_ROOT/gear_sonic_deploy}"
export LOCOMANI_SRC="$LOCOMANI_ROOT/src/teleop_free"
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
# Every one of these used to be re-derived by whoever needed it: start_stack.sh
# had all six, restart_deploy.sh two, par_batch.sh three, and the Python side
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
