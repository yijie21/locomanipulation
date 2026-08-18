#!/bin/bash
# Setup script for GR00T-WholeBodyControl (SONIC) training/eval env
set -euo pipefail

ROOT=/workspace/code/locomanipulation
REPO=$ROOT/third_party/GR00T-WholeBodyControl
ENV_NAME=sonic

# keep caches off the small (4GB) root overlay
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source /workspace/miniconda3/etc/profile.d/conda.sh

echo "=== [1/5] conda env ==="
conda env list | grep -qE "^${ENV_NAME}\s" || conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
python -V

echo "=== [2/5] Isaac Sim 5.1.0 ==="
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "=== [3/5] torch cu128 (RTX 5090 / sm_120) ==="
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/5] Isaac Lab ==="
cd "$ROOT/third_party"
if [ ! -d IsaacLab ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab
./isaaclab.sh --install

echo "=== [5/5] gear_sonic[training] ==="
cd "$REPO"
pip install -e "gear_sonic/[training]"

echo "=== DONE ==="
