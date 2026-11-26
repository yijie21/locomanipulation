#!/usr/bin/env bash
# heft_walkstop_20261126 revision 2: one hw_ha.py process in HA's runtime on one physical card (as hacollect's hc_ha.sh:
# env.sh, cwd = isaaclab_twist2_g1, only that card visible, the renderer pinned to it, SONIC's arguments), the code from
# the round's copy (SX_REPO = LC_CODE_ROOT; hw_run's code check), cameras off.
#   hw_ha.sh <gpu> --controller heft|sonic --out DIR [--arms ... --cells ... --reps a:b --traces | --probe]
set -euo pipefail
R=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
HERE=$(cd "$(dirname "$0")" && pwd)
source $R/env.sh
cd $ISAACLAB_ROOT
GPU=$1; shift
W=${LC_CODE_ROOT:?set LC_CODE_ROOT to the private copy}
export SX_REPO=$W SX_HA_EXP=$W/scripts/experiments/humanoidarena_20261124 SX_HA_CTRL=$W/ha_ctrl HC_XFER=$W/scripts/experiments/selfmodel_xfer_20261125
export PYTHONPATH=$ISAACLAB_ROOT PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID ENABLE_DEPTH=0
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU HA_PHYS_GPU=$GPU SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=2
export SONIC_VLA_USE_HEADING_ALIGN=1 SONIC_OUTPUT_DELAY_STEPS=0
exec nice -n 10 python $HERE/hw_ha.py --ha_task ppbox --seed 100 \
  --sonic_encoder_path "$SONIC_POLICY_ROOT/model_encoder.onnx" --sonic_decoder_path "$SONIC_POLICY_ROOT/model_decoder.onnx" \
  --sonic_vla_root_rot6d_layout row --sonic_vla_root_max_delta_deg 26.0 \
  --lerobot_server_url http://127.0.0.1:9 --headless --device cuda:0 --no-camera "$@"
