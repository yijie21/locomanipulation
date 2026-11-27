#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp_20261126
# Formal run of selfmodel_lmlcmp_20261126 on 4090 (tmux lc_formal): lc_formal.sh <shards per task> <backend>...
# Per backend, in order: code and inputs checked against asrun/MD5, the 7 tasks x episodes -1..29 x both methods
# (lc_batch.sh), merge, lc_check, lc_eval (frozen with the code), code checked again.
NS=$1; shift
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
CODE=$E/selfmodel_lmlcmp_20261126
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
VPY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
TASKS="IThorBathroomLiftCandleFromFloor IThorKitchenApproachTableAndLiftApple IThorKitchenApproachTableAndLiftBowl IThorKitchenApproachTableAndLiftSaltShaker IThorKitchenTakeStatueFromCabinet IThorLivingRoomApproachTableAndLiftDumbbell IThorLivingRoomApproachTableAndLiftWateringCan"
echo "$@" > $O/FORMAL_BACKENDS  # for the progress poll
md5ok() { (cd $E && grep -v '^[0-9a-f]*  /' $O/asrun/MD5 | md5sum -c --quiet); }  # the code (repo copy = asrun)
inok() { grep '^[0-9a-f]*  /' $O/asrun/MD5 | md5sum -c --quiet && (cd $O/asrun && md5sum -c --quiet LOOKS_MD5); }
if ! inok; then echo "aborted: inputs differ from asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
echo "inputs = asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log
for BE in "$@"; do
  mkdir -p $O/$BE
  if ! md5ok; then echo "$BE aborted: code differs from asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
  echo "$BE formal start $(date -u +%FT%TZ): 7 tasks x episodes -1..29 x alone/selfmodel, $NS shards per task, code $E (md5 = asrun)" | tee -a $O/formal.log
  bash $CODE/lc_batch.sh $BE -1:30 $NS $TASKS >> $O/formal.log 2>&1
  n=$(ls $O/$BE/episodes/*/*/ep*.json 2>/dev/null | wc -l)
  echo "$BE runs done $(date -u +%FT%TZ): $n episode records" | tee -a $O/formal.log
  cd $G
  export PYTHONPATH=$G:$W/src OMP_NUM_THREADS=4 LC_OUT=$O CUDA_VISIBLE_DEVICES=
  nice -n 19 $PY $CODE/lc_run.py merge --backend $BE > $O/$BE/merge.log 2>&1
  nice -n 19 $PY $CODE/lc_check.py --backend $BE 2>&1 | grep -v Warn > $O/$BE/check.txt
  echo "$BE merged and checked $(date -u +%FT%TZ): $(tail -1 $O/$BE/check.txt)" | tee -a $O/formal.log
  cd $W
  REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= LC_OUT=$O \
    nice -n 19 $VPY $CODE/lc_eval.py --backend $BE --json $O/$BE/eval.json 2>&1 | grep -v Warn > $O/$BE/eval.txt
  echo "$BE evaluated $(date -u +%FT%TZ)" | tee -a $O/formal.log
  if md5ok; then echo "$BE md5 after: same as asrun" | tee -a $O/formal.log; else echo "$BE md5 after: CHANGED" | tee -a $O/formal.log; fi
done
echo "formal done $(date -u +%FT%TZ)" | tee -a $O/formal.log
