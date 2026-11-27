#!/usr/bin/env bash
# notebook: selfmodel_xfer_20261125
# One combination's record in the SIMPLE body, then merge / check / summary / eval. Runs the asrun copy (sx_freeze.sh).
#   sx_batch.sh <name> <controller> <planner> <stance> <delay_f> <procs> [resets=400] [lml backend=<controller>]
# Out: $O/<name>/{data,data_both,run.json,logs/w*.log,merge.log,check.log,summary.txt,eval.txt,eval.json,eval_ended.*,DONE}
# SX_ASRUN (default asrun): the frozen copy to run. When it holds a whole repo copy (repo/FULL: asrun_heft, 2026-11-25),
# the SIMPLE body and the evaluation run from that copy (SX_REPO = it); otherwise from the live repo, checked file by file.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125
D3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125
A=${SX_ASRUN:-$O/asrun}
NAME=$1 CTRL=$2 PLAN=$3 STANCE=$4 DELAY=$5 P=$6 N=${7:-400} BE=${8:-$2}
VPY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
S=$A/selfmodel_xfer_20261125/sx_collect.py
E=$A/selfmodel_xfer_20261125/sx_eval.py
[ -e $A/MD5 ] || { echo "not frozen"; exit 1; }
unset SX_D3
if [ -e $A/repo/FULL ]; then
  W=$A/repo
  (cd $A && md5sum -c --quiet MD5.check) || { echo "the frozen copy changed"; exit 1; }
  cd $W || exit 1
else
  cd $W || exit 1
  for r in $(cat $A/repo/FILES); do cmp -s $r $A/repo/$r || { echo "$r differs from its frozen copy"; exit 1; }; done
fi
X=$O/$NAME
mkdir -p $X/logs
echo "$(date -u +%FT%TZ) start $NAME: $CTRL planner $PLAN stance $STANCE delay $DELAY, $N resets on $P processes" >> $X/run.log
export SX_REPO=$W OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=
PIDS=()
for k in $(seq 0 $((P - 1))); do
  nice -n 10 $VPY $S drive --controller $CTRL --planner $PLAN --stance $STANCE --delay-f $DELAY --out $X --resets $N \
    --shards $P --shard $k > $X/logs/w$k.log 2>&1 &
  PIDS+=($!)
done
echo "${PIDS[@]}" > $X/pids.txt
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) driven" >> $X/run.log
$VPY $S merge --out $X 2>&1 | grep -v Warn > $X/merge.log
$VPY $S check --out $X --resets $N 2>&1 | grep -v Warn > $X/check.log
$VPY $S summary --out $X 2>&1 | grep -v Warn > $X/summary.txt
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 OMP_NUM_THREADS=4
nice -n 10 $VPY $E --lml $D3/$BE --xfer $X --json $X/eval.json 2>&1 | grep -v Warn > $X/eval.txt
nice -n 10 $VPY $E --lml $D3/$BE --xfer $X --keep ended --json $X/eval_ended.json 2>&1 | grep -v Warn > $X/eval_ended.txt
echo "$(date -u +%FT%TZ) done" >> $X/run.log
touch $X/DONE
