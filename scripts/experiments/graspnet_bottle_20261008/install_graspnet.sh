#!/bin/bash
# Install graspnet-baseline for `run_grasp.py --grasp-source graspnet`. The default grasp source (the offline cache,
# grasp_cache.py) needs none of this. Every step is skipped when its result is already there, so a rerun after a
# failure resumes; the last step loads the network and runs graspnet-baseline's own example scene.
#   bash scripts/experiments/graspnet_bottle_20261008/install_graspnet.sh [checkout dir] [graspnetAPI dir]
# Defaults third_party/graspnet-baseline and third_party/graspnetAPI. Needs uv, git, a CUDA 12.8 toolkit (nvcc) and
# one GPU. graspnet-baseline and its weights are licensed for non-commercial research; nothing here goes into git.
# Pinned to the commits verified on 2026-10-08 (graspnet-baseline 280c215, graspnetAPI eb57dd2).
set -eu
cd "$(dirname "$0")/../../.." || exit 1
REPO=$PWD
GN=$(realpath -m "${1:-third_party/graspnet-baseline}")
API=$(realpath -m "${2:-third_party/graspnetAPI}")
export TMPDIR=${TMPDIR:-/workspace/tmp}
mkdir -p "$TMPDIR"
# the image points uv's cache and interpreters at root-owned /.uv; fall back to TMPDIR when they are not writable
writable() { [ -n "$1" ] && mkdir -p "$1" 2>/dev/null && [ -w "$1" ]; }
writable "${UV_CACHE_DIR:-}" || export UV_CACHE_DIR=$TMPDIR/uv_cache
writable "${UV_PYTHON_INSTALL_DIR:-}" || export UV_PYTHON_INSTALL_DIR=$TMPDIR/uv_python
# torch 2.7 cu128 refuses to build extensions with any other toolkit (the image's CUDA_HOME is 13.0)
CUDA=${CUDA_HOME:-}
"$CUDA/bin/nvcc" --version 2>/dev/null | grep -q "release 12.8" || CUDA=/usr/local/cuda-12.8
"$CUDA/bin/nvcc" --version 2>/dev/null | grep -q "release 12.8" \
    || { echo "[install_graspnet] needs the CUDA 12.8 toolkit: set CUDA_HOME to it"; exit 1; }
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
PY=$GN/.venv/bin/python
step() { echo "[install_graspnet] $*"; }

step "1/7 checkouts"
[ -d "$GN/.git" ] || { git clone -q https://github.com/graspnet/graspnet-baseline.git "$GN" && git -C "$GN" checkout -q 280c215; }
[ -d "$API/.git" ] || { git clone -q https://github.com/graspnet/graspnetAPI.git "$API" && git -C "$API" checkout -q eb57dd2; }

step "2/7 patch the CUDA extensions for torch 2.7 (Tensor.type().is_cuda() and Tensor.data<T>() are gone)"
grep -rl --include=*.h --include=*.cpp -e '\.type()\.is_cuda()' -e '\.data<' "$GN/knn" "$GN/pointnet2" \
    | xargs -r sed -i -E 's/\.type\(\)\.is_cuda\(\)/.is_cuda()/g; s/\.data<(float|int|long)>\(\)/.data_ptr<\1>()/g'

step "3/7 venv (python 3.10, torch 2.7.0 cu128, the versions of the 2026-10-08 install)"
[ -x "$PY" ] || uv venv -q -p 3.10 "$GN/.venv"
uv pip install -q -p "$PY" torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# the second half is what graspnetAPI imports (its own requirement list pins versions that no longer install)
uv pip install -q -p "$PY" numpy==1.26.4 scipy==1.15.3 pillow==12.3.0 tqdm==4.70.1 open3d==0.20.0 trimesh==5.1.1 \
    transforms3d==0.4.2 opencv-python-headless==4.11.0.86 grasp-nms==1.0.2 gdown==6.4.1 setuptools==78.1.0 \
    wheel==0.48.0 ninja==1.13.2 pyzmq==27.2.0 \
    matplotlib==3.10.9 scikit-image==0.25.2 scikit-learn==1.7.2 autolab-core==1.1.1 cvxopt==1.3.3 dill==0.4.1 \
    h5py==3.16.0

step "4/7 build pointnet2 and knn for compute capability $ARCH"
if ! (cd "$GN/pointnet2" && "$PY" -c "import torch, pointnet2._ext" 2>/dev/null) \
    || ! (cd "$GN/knn" && "$PY" -c "import torch, knn_pytorch" 2>/dev/null); then
    NV=$(ls -d "$GN"/.venv/lib/python3.10/site-packages/nvidia/*/include | tr '\n' ':')
    for ext in pointnet2 knn; do
        (cd "$GN/$ext" && rm -rf build && CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH TORCH_CUDA_ARCH_LIST=$ARCH \
            MAX_JOBS=8 CPATH=$NV "$PY" setup.py install > "$TMPDIR/graspnet_build_$ext.log" 2>&1) \
            || { echo "[install_graspnet] $ext build failed, see $TMPDIR/graspnet_build_$ext.log"; exit 1; }
    done
fi

step "5/7 graspnetAPI (without its pinned old dependencies)"
"$PY" -c "import graspnetAPI" 2>/dev/null || uv pip install -q -p "$PY" --no-deps "$API"

step "6/7 weights logs/checkpoint-rs.tar (RealSense, 12 MB)"
mkdir -p "$GN/logs"
[ -s "$GN/logs/checkpoint-rs.tar" ] || "$GN/.venv/bin/gdown" -q 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O "$GN/logs/checkpoint-rs.tar"

step "7/7 selftest on graspnet-baseline's example scene"
"$PY" -u "$REPO/scripts/experiments/graspnet_bottle_20261008/graspnet_server.py" --selftest --gpu "${GPU:-0}" \
    --graspnet-dir "$GN"
step "done: $GN"
