#!/bin/bash
# EmbodiedGen affordance stage, minus flash-attn.
#
# Why skip flash-attn: EmbodiedGen's installer builds it defensively, but the
# grasp path never uses it. GraspGen's model builder pins the transformer to
# non-flash attention (thirdparty/GraspGen/grasp_gen/models/generator.py:144
# passes enable_flash=False); flash_attn is imported behind try/except in
# ptv3.py:29-31 and only asserted inside the enable_flash=True branch.
#
# Building it here also fails: flash-attn compiles for sm_80/90/100/120 at once
# (it ignores TORCH_CUDA_ARCH_LIST) with nvcc --threads 4 per job, which OOM-kills
# itself on this 125 GB box regardless of MAX_JOBS.
#
# torch-scatter is already installed by the earlier runs, so it is skipped too.
#
# Usage:  bash src/teleop_free/install_affordance_noflash.sh

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE:-$0}")/../../scripts/paths.sh"
export TMPDIR="$LOCOMANI_TMP"
export PIP_CACHE_DIR="$LOCOMANI_CACHE/pip"
export HF_HOME="$LOCOMANI_WORKSPACE/.hf_home"
export XDG_CACHE_HOME="$LOCOMANI_CACHE"
export MAX_JOBS=4
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate embodiedgen

REPO="$LOCOMANI_ROOT"/third_party/EmbodiedGen
cd "$REPO" || exit 1

echo "=== python: $(which python) ==="

echo "=== [1/3] python packages ==="
pip install \
    "fpsample==1.0.2" \
    "viser==1.0.30" \
    "pickle5==0.0.11" \
    "webdataset==1.0.2" \
    "yourdfpy==0.0.56" \
    "sharedarray==3.2.4" \
    "meshcat==0.3.2" || echo "WARN: some packages failed"

echo "=== [2/3] GraspGen pointnet2_ops (CUDA extension) ==="
pip install --no-build-isolation "$REPO/thirdparty/GraspGen/pointnet2_ops" \
    && rm -rf "$REPO/thirdparty/GraspGen/pointnet2_ops/build"

echo "=== [3/3] opencv pins ==="
pip install opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80

echo "=== affordance-noflash exit: $? ==="
