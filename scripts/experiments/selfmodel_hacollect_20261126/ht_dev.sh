#!/usr/bin/env bash
# notebook: skip development run of ht_task.py (selfmodel_hatask_20261126 revision 2: the R code path on dev seeds, no result read)
# One ht_task.py process from the private development copy $OT/dev_code (asrun_real2 copied, ht_*.py added), with
# asrun_real2's environment (hr_formal.sh's variables), on one 4090 card:
#   ht_dev.sh <gpu> <controller> <task> <seeds> <repeats> <arms> <out> [extra ht_task args...]
# R's model for the dev runs: LocoManiLab's D (train3b, the controller, seed 0; REACHWM_FDVEL off, as it was trained),
# the template lmlreal's records (lc3_run's).
set -uo pipefail
GPU=$1 C=$2 T=$3 SEEDS=$4 REPS=$5 ARMS=$6 OUT=$7; shift 7
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
OT=$L/selfmodel_hatask_20261126
A=$OT/dev_code
X=(); [ $C = sonic ] && X=(--no-camera)
[ $C = heft ] && X=(--heft-gait)
export HT_REACHWM_REPO=$L/selfmodel_simpledata_20261126/asrun/repo_b
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=${HT_FDVEL:-0}
MR=${HT_MODEL_ROOT:-$L/selfmodel_train3b_20261125/$C/models_D_s0}
MT=${HT_MODEL_TEMPLATE:-$L/selfmodel_lmlreal_20261125/$C/data/all.npz}
mkdir -p $OUT
SX_REPO=$A/repo SX_HA_EXP=$A/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$A/selfmodel_xfer_20261125 \
  SX_DEPTH=1 SX_HA_TASK=$T HC_SCRIPT=ht_task.py \
  bash $A/selfmodel_hacollect_20261126/hc_ha.sh $GPU --controller $C --out $OUT --seeds $SEEDS --repeats $REPS --arms $ARMS \
  --via 1 --via-rule lift --model-root $MR --model-template $MT "${X[@]}" "$@"
