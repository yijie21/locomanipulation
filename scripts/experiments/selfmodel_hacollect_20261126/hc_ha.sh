#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# One hc_ha.py process on one physical card (as selfmodel_xfer_20261125/sx_ha.sh: env.sh, cwd = isaaclab_twist2_g1,
# only that card visible, the renderer pinned to it by sx_ha.kit_single_gpu, ppbox's SONIC arguments).
#   hc_ha.sh <gpu> --phase site|prof|drive --controller C --out DIR [...]
set -euo pipefail
R=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
HERE=$(cd "$(dirname "$0")" && pwd)
source $R/env.sh
cd $ISAACLAB_ROOT
GPU=$1; shift
export PYTHONPATH=$ISAACLAB_ROOT PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID ENABLE_DEPTH=${SX_DEPTH:-0}
export CUDA_VISIBLE_DEVICES=$GPU HA_PHYS_GPU=$GPU SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=2
export SONIC_VLA_USE_HEADING_ALIGN=1 SONIC_OUTPUT_DELAY_STEPS=0
exec nice -n 10 python $HERE/${HC_SCRIPT:-hc_ha.py} --ha_task ${SX_HA_TASK:-ppbox} --seed 100 \
  --sonic_encoder_path "$SONIC_POLICY_ROOT/model_encoder.onnx" --sonic_decoder_path "$SONIC_POLICY_ROOT/model_decoder.onnx" \
  --sonic_vla_root_rot6d_layout row --sonic_vla_root_max_delta_deg 26.0 \
  --lerobot_server_url http://127.0.0.1:9 --headless --device cuda:0 "$@"
