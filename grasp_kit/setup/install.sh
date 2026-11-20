#!/usr/bin/env bash
# Installs the kit's two Python environments:
#   envs/perception  SAM 3 (segmentation) and LingBot-Depth (depth completion); PyTorch 2.11.0, xformers 0.0.35
#   envs/grasp       GraspGen-X (grasp generation); PyTorch 2.7.1
# They are two because GraspGen-X pins older numpy / timm / huggingface-hub than SAM 3 accepts.
# Needs: Linux x86_64, an NVIDIA GPU and driver, git, uv (https://docs.astral.sh/uv/). About 15 GB of disk.
#   bash setup/install.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
SAM3=(https://github.com/facebookresearch/sam3.git f92a5efc4001ebfe9d050d43e61f6fb10e02d73c)
LINGBOT=(https://github.com/robbyant/lingbot-depth.git f3a237e434ae987bc38281476d6cfb5df3e4d739)
GRASPGENX=(https://github.com/NVlabs/GraspGenX.git b9429097728cb1c430dd78b92edf17ba318aad03)
TORCH=(torch==2.7.1 torchvision==0.22.1)
# LingBot-Depth needs xformers (its depth completion runs nested tensors). xformers 0.0.31 (the build for torch 2.7)
# picks a Hopper-only attention kernel on RTX 50xx and fails there; the kit was tested with 0.0.35 and torch 2.11.
PTORCH=(torch==2.11.0 torchvision==0.26.0 xformers==0.0.35)
TORCH_INDEX=https://download.pytorch.org/whl/$TORCH_CUDA

clone() {  # url commit dir: the repository at the commit the kit was tested with
  [ -d "$3/.git" ] || git clone -q "$1" "$3"
  git -C "$3" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$3" fetch -q origin
  git -C "$3" -c advice.detachedHead=false checkout -q "$2"
}
mkdir -p "$SRC" "$ENVS"
echo "== sources -> $SRC"
clone "${SAM3[@]}" "$SRC/sam3"
clone "${LINGBOT[@]}" "$SRC/lingbot-depth"
clone "${GRASPGENX[@]}" "$SRC/GraspGenX"

echo "== perception env -> $ENVS/perception"
P=$ENVS/perception/bin/python
[ -x "$P" ] || uv venv -q -p 3.11 "$ENVS/perception"
uv pip install -q -p "$P" "${PTORCH[@]}" --index-url "$TORCH_INDEX"
uv pip install -q -p "$P" -e "$SRC/sam3"
# LingBot-Depth pins torch 2.6.0 and xformers 0.0.29; it runs on the versions above
uv pip install -q -p "$P" --no-deps -e "$SRC/lingbot-depth"
uv pip install -q -p "$P" "numpy<2" scipy pillow matplotlib "opencv-python-headless<4.12" trimesh click omegaconf einops \
  "huggingface_hub[cli]" pycocotools psutil

echo "== grasp env -> $ENVS/grasp"
G=$ENVS/grasp/bin/python
[ -x "$G" ] || uv venv -q -p 3.11 "$ENVS/grasp"
uv pip install -q -p "$G" "${TORCH[@]}" --index-url "$TORCH_INDEX"
# GraspGen-X caps torch below 2.7 (its wheels for 2.7 are CUDA 12.6+ only); 2.7.1 runs it, and RTX 50xx needs it
printf '%s\n' "${TORCH[@]}" > "$ENVS/torch_override.txt"
# sharedarray builds against numpy with a setuptools that still has pkg_resources
uv pip install -q -p "$G" "setuptools>=45,<78" wheel numpy==1.26.4
uv pip install -q -p "$G" --prerelease=allow --override "$ENVS/torch_override.txt" --no-build-isolation-package sharedarray \
  -e "$SRC/GraspGenX"

echo "== check"
"$P" -c "import torch, xformers.ops, sam3, mdm, cv2, scipy; print('perception: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$G" -c "import torch, graspgenx.samplers, matplotlib; print('grasp: torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || echo "grasp env: run setup/download_weights.sh first if the import stops at the gripper descriptions"
echo "done. Next: bash setup/download_weights.sh"
