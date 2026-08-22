#!/bin/bash
# Install EmbodiedGen into its own conda env, isolated from the SONIC envs.
#
# Every cache is forced onto $LOCOMANI_WORKSPACE: / is a 32 GB overlay with ~1.5 GB
# and the default pip/HF/conda temp locations all live there.
#
# Usage:  bash src/teleop_free/install_embodiedgen.sh <stage>
#         stages, in order: cu128 -> basic -> affordance

# No `set -u`: conda's cuda-nvcc activation hook dereferences NVCC_PREPEND_FLAGS
# before setting it, which aborts the whole stage under nounset.
STAGE="${1:?usage: install_embodiedgen.sh <cu128|basic|affordance|room|scene3d>}"

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE:-$0}")/../../scripts/paths.sh"
export TMPDIR="$LOCOMANI_TMP"
export TMP="$LOCOMANI_TMP"
export PIP_CACHE_DIR="$LOCOMANI_CACHE/pip"
export HF_HOME="$LOCOMANI_WORKSPACE/.hf_home"
export TORCH_HOME="$LOCOMANI_CACHE/torch"
export XDG_CACHE_HOME="$LOCOMANI_CACHE"
# The affordance stage compiles flash-attn and torch-scatter from source.
# Keep MAX_JOBS low: flash-attn's nvcc jobs peak around 6-10 GB EACH, so 24 jobs
# OOM-kill themselves on this 125 GB box (the failure shows up only as dozens of
# bare "Killed" lines followed by "Error compiling objects for extension").
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate embodiedgen

cd "$LOCOMANI_ROOT"/third_party/EmbodiedGen || exit 1

echo "=== EmbodiedGen install stage: $STAGE ==="
echo "python: $(which python) $(python -V 2>&1)"
bash install.sh "$STAGE"
echo "=== stage $STAGE exit: $? ==="
