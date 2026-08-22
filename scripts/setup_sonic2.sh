#!/bin/bash
# Stage 2: Isaac Lab v2.3.2 (pairs with Isaac Sim 5.1.0 + torch 2.7.0/cu128) + gear_sonic
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE:-$0}")/paths.sh"
ROOT="$LOCOMANI_ROOT"
REPO=$ROOT/third_party/GR00T-WholeBodyControl
ENV_NAME=sonic

export PIP_CACHE_DIR="$LOCOMANI_CACHE/pip"
export TMPDIR="$LOCOMANI_TMP"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source "$CONDA_BASE"/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "=== [1/4] checkout Isaac Lab v2.3.2 ==="
cd "$ROOT/third_party/IsaacLab"
git checkout -q v2.3.2
git status -sb | head -2

echo "=== [2/4] isaaclab install ==="
./isaaclab.sh --install

echo "=== [3/4] pin torch 2.7.0 cu128 (sm_120 / RTX 5090) ==="
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/4] gear_sonic[training] ==="
cd "$REPO"
pip install -e "gear_sonic/[training]"

echo "=== DONE ==="
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
