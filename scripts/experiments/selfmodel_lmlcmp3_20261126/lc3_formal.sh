#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp3_20261126
# Formal run of selfmodel_lmlcmp3_20261126 on 4090 (tmux lc3_formal): lc3_formal.sh <shards per task> <backend>...
# lmlcmp2's lc2_formal.sh with this round's groups and seeds: ScaleBFM A/S/R/E, SONIC A/S/R, HEFT A/R, the 7 tasks x the
# new seeds 30..119. Per backend, in order: the private code copy and the inputs checked against asrun/, the runs
# (lc3_batch.sh), merge, lc3_check, lc3_eval (frozen with the code), the copy checked again.
# Revision 1 addendum: everything runs from the round's private copy C0 (code/, made by lc3_freeze.sh): scripts,
# PYTHONPATH, the imports of every run process (lc3_run's code check, LC_CODE_ROOT) -- the shared repo tree is never
# imported from. Only VPY (the eval's interpreter: numpy + scipy) stays the shared tree's.
NS=$1; shift
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126
C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code
A=$O/asrun
E=$C0/scripts/experiments
CODE=$E/selfmodel_lmlcmp3_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
VPY=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
TASKS="IThorBathroomLiftCandleFromFloor IThorKitchenApproachTableAndLiftApple IThorKitchenApproachTableAndLiftBowl IThorKitchenApproachTableAndLiftSaltShaker IThorKitchenTakeStatueFromCabinet IThorLivingRoomApproachTableAndLiftDumbbell IThorLivingRoomApproachTableAndLiftWateringCan"
# each controller's groups (not GROUPS: bash's own read-only array of the user's group ids; the first launch, 02:25Z,
# ran with LC_METHODS=0 and was stopped before any episode ran)
declare -A GRP=([scalebfm]=alone,selfmodel,reachable,entryinv [sonic]=alone,selfmodel,reachable [heft]=alone,reachable)
echo "$@" > $O/FORMAL_BACKENDS  # for the progress poll
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5); }  # the private copy, every file
inok() { md5sum -c --quiet $A/MD5 && (cd $A && md5sum -c --quiet LOOKS3_MD5); }  # models, data, joint band, looks
if ! inok; then echo "aborted: inputs differ from asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
echo "inputs = asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log
for BE in "$@"; do
  case ${GRP[$BE]} in alone,*) ;; *) echo "$BE aborted: no groups for this controller $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1;; esac
  mkdir -p $O/$BE
  if ! md5ok; then echo "$BE aborted: code differs from asrun/CODE_MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
  echo "$BE formal start $(date -u +%FT%TZ): 7 tasks x episodes 30..119 x ${GRP[$BE]}, $NS shards per task, code $C0 (md5 = asrun)" | tee -a $O/formal.log
  LC_CODE_ROOT=$C0 LC_OUT=$O LC_METHODS=${GRP[$BE]} bash $CODE/lc3_batch.sh $BE 30:120 $NS $TASKS >> $O/formal.log 2>&1
  n=$(ls $O/$BE/episodes/*/*/ep*.json 2>/dev/null | wc -l)
  echo "$BE runs done $(date -u +%FT%TZ): $n episode records" | tee -a $O/formal.log
  nst=$(cat $O/$BE/logs/run_*.log | grep -c "^lc code check start: env.py $C0/")
  nf=$(cat $O/$BE/logs/run_*.log | grep -c "^lc code check .*FAIL")
  echo "$BE code check: $nst processes started with env.py from the copy; $(cat $O/$BE/logs/run_*.log | grep -h "^lc code check start" | sed 's/.*env.py \([^;]*\);.*/\1/' | sort | uniq -c | tr -s ' ' | tr '\n' ';') failures $nf" | tee -a $O/formal.log
  [ "$nf" -gt 0 ] && echo "$BE CODE CHECK FAILED $(date -u +%FT%TZ)" | tee -a $O/formal.log
  cd $G
  export PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=4 LC_OUT=$O CUDA_VISIBLE_DEVICES=
  nice -n 19 $PY $CODE/lc3_run.py merge --backend $BE > $O/$BE/merge.log 2>&1
  nice -n 19 $PY $CODE/lc3_check.py --backend $BE 2>&1 | grep -v Warn > $O/$BE/check.txt
  echo "$BE merged and checked $(date -u +%FT%TZ): $(tail -1 $O/$BE/check.txt)" | tee -a $O/formal.log
  cd $C0
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= LC_OUT=$O nice -n 19 $VPY $CODE/lc3_eval.py --backend $BE --json $O/$BE/eval.json 2>&1 | grep -v Warn > $O/$BE/eval.txt
  echo "$BE evaluated $(date -u +%FT%TZ)" | tee -a $O/formal.log
  if md5ok; then echo "$BE md5 after: same as asrun" | tee -a $O/formal.log; else echo "$BE md5 after: CHANGED" | tee -a $O/formal.log; fi
done
echo "formal done $(date -u +%FT%TZ)" | tee -a $O/formal.log
