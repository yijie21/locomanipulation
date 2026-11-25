#!/usr/bin/env bash
# notebook: skip development launcher of the HumanoidArena harness (no experiment by itself)
# Runs one of this directory's Isaac-side scripts in HumanoidArena's runtime, launched as eval_vla_suite.py launches
# sim_eval_vla.py: env.sh sourced, cwd = isaaclab_twist2_g1, CUDA_DEVICE_ORDER=PCI_BUS_ID, --device cuda:N (sim_eval_vla
# puts the renderer and PhysX on N), depth on.
#   GPU=3 bash run_ha.sh ha_bench.py --ha_task ppbox ...      (the script's own arguments follow)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
source $R/env.sh
cd $ISAACLAB_ROOT
export PYTHONPATH=$ISAACLAB_ROOT:$HERE PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID ENABLE_DEPTH=1
export SONIC_VLA_USE_HEADING_ALIGN=${SONIC_VLA_USE_HEADING_ALIGN:-1} SONIC_OUTPUT_DELAY_STEPS=${SONIC_OUTPUT_DELAY_STEPS:-0}
GPU=${GPU:-3}
SCRIPT=$1; shift
exec python $HERE/$SCRIPT \
  --seed ${SEED_ARG:-100} \
  --sonic_encoder_path "$SONIC_POLICY_ROOT/model_encoder.onnx" \
  --sonic_decoder_path "$SONIC_POLICY_ROOT/model_decoder.onnx" \
  --sonic_vla_root_rot6d_layout row --sonic_vla_root_max_delta_deg 26.0 \
  --lerobot_server_url http://127.0.0.1:9 \
  --headless --device cuda:$GPU "$@"
