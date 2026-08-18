#!/bin/bash
# Common environment for running EmbodiedGen CLIs.
# Source this, then run text3d-cli / img3d-cli / layout-cli / affordance-cli
# FROM THE EmbodiedGen REPO ROOT — gen_grasp.py does a relative
# sys.path.append("thirdparty/GraspGen"), so a different cwd breaks grasping.

# Text-to-image backend (textto3d.py:76 reads TEXT_MODEL, default "sd35").
# Options: sd35 | cosmos | kolors | flux | chroma  (image_comm_model.py:332)
#
# Using sd35: EmbodiedGen's defaults (text_guidance_scale=7.0,
# img_denoise_step=25, gen_layout.py:50-51) are tuned for it, so no CLI
# overrides are needed.
#
# flux (FLUX.1-schnell, 12B) is the bigger model but is distilled — it needs
# `--text_guidance_scale 0.0 --img_denoise_step 4`, otherwise the sd35 defaults
# over-saturate it. Its loader also needed the OOM fix in image_comm_model.py.
#
# sd35 and flux are gated on Hugging Face; the licence has been accepted for
# this account, so both resolve. kolors and chroma need no licence.
export TEXT_MODEL="${TEXT_MODEL:-sd35}"

export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.hf_home
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"

# OpenAI credentials for the quality checkers (kept out of the repo).
# shellcheck disable=SC1091
source /workspace/.secrets/embodiedgen.env

# shellcheck disable=SC1091
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate embodiedgen

cd /workspace/code/locomanipulation/third_party/EmbodiedGen || return 1
