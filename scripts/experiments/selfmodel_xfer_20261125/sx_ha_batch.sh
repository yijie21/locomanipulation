#!/usr/bin/env bash
# notebook: selfmodel_xfer_20261125
# One HumanoidArena combination's record (revision 2), then merge / check / summary / eval, from the frozen asrun_ha/.
#   sx_ha_batch.sh <name> <controller sonic|scalebfm|heft> <gpu0> <gpu1> [resets=400]
# Two sx_ha.py processes, one per physical card (CUDA_VISIBLE_DEVICES = that card only), shards 0 / 1 of the resets.
# SONIC runs without the provider's per-step camera render (SX_NORENDER=1; revision 2: three runs bit-identical with and
# without it). Out: $O/<name>/{data,data_both,run.json,logs/w*.log,merge.log,check.log,summary.txt,eval*.txt,DONE}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125
D3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125
A=${SX_ASRUN:-$O/asrun_ha}
NAME=$1 CTRL=$2 G0=$3 G1=$4 N=${5:-400}
[ -e $A/MD5 ] || { echo "not frozen"; exit 1; }
(cd $A && md5sum -c --quiet MD5.check) || { echo "the frozen copy or a pinned install changed"; exit 1; }
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
H=$A/selfmodel_xfer_20261125
X=$O/$NAME
mkdir -p $X/logs
echo "$(date -u +%FT%TZ) start $NAME: $CTRL on HumanoidArena (ppbox), $N resets on cards $G0 and $G1" >> $X/run.log
unset SX_D3
export SX_REPO=$A/repo SX_HA_EXP=$A/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl
[ "$CTRL" = sonic ] && export SX_NORENDER=1
PIDS=()
k=0
for g in $G0 $G1; do
  bash $H/sx_ha.sh $g --phase drive --controller $CTRL --out $X --resets $N --shards 2 --shard $k > $X/logs/w$k.log 2>&1 &
  PIDS+=($!)
  k=$((k + 1))
done
echo "${PIDS[@]}" > $X/pids.txt
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) driven" >> $X/run.log
cd $A/repo || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
$VPY $H/sx_collect.py merge --out $X 2>&1 | grep -v Warn > $X/merge.log
$VPY $H/sx_collect.py check --out $X --resets $N --id0 40000000 2>&1 | grep -v Warn > $X/check.log
$VPY $H/sx_collect.py summary --out $X 2>&1 | grep -v Warn > $X/summary.txt
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 OMP_NUM_THREADS=4
nice -n 10 $VPY $H/sx_eval.py --lml $D3/$CTRL --xfer $X --json $X/eval.json 2>&1 | grep -v Warn > $X/eval.txt
nice -n 10 $VPY $H/sx_eval.py --lml $D3/$CTRL --xfer $X --keep ended --json $X/eval_ended.json 2>&1 | grep -v Warn > $X/eval_ended.txt
echo "$(date -u +%FT%TZ) done" >> $X/run.log
touch $X/DONE
