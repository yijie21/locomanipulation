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

# Logs + exporter sessions for instance $1 (default $INSTANCE, default 0).
# Mirrors paths.session_root() so multi-instance runs never collide.
locomani_session_root() {
    _lm_i="${1:-${INSTANCE:-0}}"
    if [ "$_lm_i" = "0" ]; then echo "$LOCOMANI_TMP/p03"
    else echo "$LOCOMANI_TMP/p03_i$_lm_i"; fi
}
