#!/bin/bash
set -euo pipefail
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate sonic
export OMNI_KIT_ACCEPT_EULA=YES
cd /workspace/code/locomanipulation/GR00T-WholeBodyControl

python gear_sonic/eval_agent_trl.py \
    +checkpoint=sonic_release/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=8 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/tmp/sonic_renders" \
    ++manager_env.config.env_spacing=10.0 \
    "~manager_env/recorders=empty" "+manager_env/recorders=render" \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
