#!/usr/bin/env bash
# Downloads the models' weights (about 6.5 GB) into $WEIGHTS (setup/env.sh):
#   SAM 3                 facebook/sam3 (3.3 GB). Gated: accept its licence at https://huggingface.co/facebook/sam3 with
#                         your Hugging Face account, then log in here once:  envs/perception/bin/hf auth login
#   LingBot-Depth         robbyant/lingbot-depth-pretrain-vitl-14-v0.5 (1.2 GB)
#   GraspGen-X            adithyamurali/GraspGenXModel (1.6 GB) and its gripper descriptions (0.3 GB)
# In mainland China, set a mirror first:  export HF_ENDPOINT=https://hf-mirror.com
#   bash setup/download_weights.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
HF=$ENVS/perception/bin/hf
[ -x "$HF" ] || { echo "no $HF: run setup/install.sh first"; exit 1; }
mkdir -p "$HF_HOME"
"$HF" download robbyant/lingbot-depth-pretrain-vitl-14-v0.5 model.pt
"$HF" download adithyamurali/GraspGenXModel --local-dir "$GRASPGENX_CHECKPOINT_DIR"
"$HF" download adithyamurali/gripper_descriptions --repo-type dataset --local-dir "$GRASPGENX_GRIPPER_CFG_DIR"
"$HF" download facebook/sam3 config.json sam3.pt || { echo "SAM 3 failed: accept its licence on huggingface.co/facebook/sam3, then '$HF auth login', and run this again"; exit 1; }
ls "$GRASPGENX_CHECKPOINT_DIR/release/gen" "$GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets/x_grippers/unitree_g1" > /dev/null
echo "done. Next: bash run_examples.sh"
