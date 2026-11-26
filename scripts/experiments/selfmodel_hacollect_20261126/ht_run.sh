#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# One ht_task.py process from the code copy this script lies in (its parent: dev_code, or the frozen asrun), with
# asrun_real2's environment (hr_formal.sh's variables), on one 4090 card:
#   ht_run.sh <gpu> <controller> <task> <seeds> <repeats> <arms> <out> [extra ht_task args...]
# R's model: HT_MODEL_ROOT (models_<arm>_s<seed>) and HT_MODEL_TEMPLATE (a data file of its layout); default (the dev
# runs): LocoManiLab's D (train3b, the controller, seed 0) and lmlreal's records. REACHWM_FDVEL = HT_FDVEL (default 0,
# as LocoManiLab's D was trained; the HA models: 1). The flags the model reader saw are in every episode log.
set -uo pipefail
GPU=$1 C=$2 T=$3 SEEDS=$4 REPS=$5 ARMS=$6 OUT=$7; shift 7
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
A=$(cd "$(dirname "$0")/.." && pwd)
X=(); [ $C = sonic ] && X=(--no-camera)
[ $C = heft ] && X=(--heft-gait)
export HT_REACHWM_REPO=$L/selfmodel_simpledata_20261126/asrun/repo_b
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=${HT_FDVEL:-0}
MR=${HT_MODEL_ROOT:-$L/selfmodel_train3b_20261125/$C/models_D_s0}
MT=${HT_MODEL_TEMPLATE:-$L/selfmodel_lmlreal_20261125/$C/data/all.npz}
mkdir -p $OUT
echo "ht_run $(date -u +%FT%TZ): code $A; model $MR; template $MT; FDVEL $REACHWM_FDVEL; $C $T seeds $SEEDS repeats $REPS arms $ARMS card $GPU; $*"
SX_REPO=$A/repo SX_HA_EXP=$A/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$A/selfmodel_xfer_20261125 \
  SX_DEPTH=1 SX_HA_TASK=$T HC_SCRIPT=ht_task.py \
  bash $A/selfmodel_hacollect_20261126/hc_ha.sh $GPU --controller $C --out $OUT --seeds $SEEDS --repeats $REPS --arms $ARMS \
  --via 1 --via-rule lift --model-root $MR --model-template $MT "${X[@]}" "$@"
