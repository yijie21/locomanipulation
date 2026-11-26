#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# One formal (or pre-formal code check) ht_task process from the frozen asrun2 (revision 2 section 7), on one 4090 card:
#   ht_formal.sh <card> <controller> <arm S|T> <task> <seeds> <out> [budget-x, default 3]
# asrun2 checked against its MD5 first; PYTHONDONTWRITEBYTECODE=1 (coordinator 11-26: nothing written into the frozen
# copy); R's model = the registration's choice (hadata 2b: evaluation 3's arm, data 100 %, seed 0, the run's best
# checkpoint): <collection>/train/<controller>/models_<arm>_s0, sized by that controller's HA d3 file; REACHWM_FDVEL=1
# (as the HA models were trained). Pairs A then R in this one process, repeat 0; an interrupted run is resumed by
# running the same command again (asrun2's rule: an unfinished pair's records moved to stale/, both arms rerun).
set -uo pipefail
G=$1 C=$2 X=$3 T=$4 SEEDS=$5 OUT=$6 BX=${7:-3}
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
F2=$L/selfmodel_hatask_20261126/asrun2
OH=$L/selfmodel_hadata_20261126; [ $C = sonic ] && OH=$L/selfmodel_hadata_sonic_20261126
case $X in S|T) ;; *) echo "arm $X: S or T"; exit 2;; esac
(cd $F2 && md5sum -c --quiet MD5) || { echo "asrun2 differs from its MD5: stop"; exit 1; }
MR=$OH/train/$C/models_${X}_s0
MT=$OH/train/$C/data/d3.npz
[ -e $MR/E3_ensemble_f1/model.pt ] && [ -e $MT ] || { echo "no $MR/E3_ensemble_f1/model.pt or $MT"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 HT_FDVEL=1 HT_MODEL_ROOT=$MR HT_MODEL_TEMPLATE=$MT
echo "ht_formal $(date -u +%FT%TZ): asrun2 (FROZEN $(cat $F2/FROZEN_UTC)) = MD5; $C R model $X: $MR (model.pt $(md5sum < $MR/E3_ensemble_f1/model.pt | cut -c1-8)); template $MT; $T seeds $SEEDS card $G budget ${BX}x"
exec bash $F2/selfmodel_hacollect_20261126/ht_run.sh $G $C $T $SEEDS 0 A,R $OUT --budget-x $BX
