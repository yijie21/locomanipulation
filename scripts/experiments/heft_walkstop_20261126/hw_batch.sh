#!/usr/bin/env bash
# heft_walkstop_20261126: hw_batch.sh <arms> <cells a:b | c1,c2> <reps a:b> <processes> [hw_run.py args ...]
# hw_run.py run in <processes> shards in the background (nohup, nice ${HW_NICE:-10}, single-threaded, no GPU); the arms
# must share one backend; logs in $HW_OUT/logs/<backend>_<arms>_n<P>_<k>.log, pids appended to $HW_OUT/pids.txt (the
# python processes); waits for its own processes (by pid). Code: $LC_CODE_ROOT (the round's private copy; lc3_run's code
# check on every process) else the copy under $HW_OUT/code.
ARMS=$1; CELLS=$2; REPS=$3; NP=$4; shift 4
export HW_OUT=${HW_OUT:-/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126}
W=${LC_CODE_ROOT:-$HW_OUT/code}
CODE=$W/scripts/experiments/heft_walkstop_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
LOG=$HW_OUT/logs
mkdir -p $LOG
TAG=$(echo $ARMS | tr , -)
PIDS=()
for ((k=0; k<NP; k++)); do
  nohup nice -n ${HW_NICE:-10} $PY $CODE/hw_run.py run --arms $ARMS --cells $CELLS --reps $REPS --shard $k --shards $NP --out $HW_OUT "$@" \
    > >(grep -a --line-buffered -v "RL step took\|Control mode set to policy\|motion layout updated" > $LOG/${TAG}_n${NP}_$k.log) 2>&1 &
  PIDS+=($!)
  echo "$! $ARMS $k/$NP $(date -u +%FT%TZ)" >> $HW_OUT/pids.txt
done
echo "$(date -u +%FT%TZ) launched $ARMS cells $CELLS reps $REPS: ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $ARMS finished"
