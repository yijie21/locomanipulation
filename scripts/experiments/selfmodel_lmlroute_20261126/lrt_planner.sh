#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# lrt_planner.sh <gpu> <routes dir> <log>: one lrt_plan.py on 4090 card <gpu> (never 0) when the card stays <= 20 GB with it
# (needs <= 18500 MiB used before; a planner reserves ~0.5 GB in the smoke); a guard stops it (by pid) above 20000 MiB
G=$1; RT=$2; LOG=$3
[ "$G" = 0 ] && { echo "card 0 is not ours" >> $LOG; exit 1; }
LR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
CC=${LRT_CODE:-$LR/code}
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
until M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' '); [ "$M" -le 18500 ]; do
  echo "$(date -u +%T) card $G at $M MiB: waiting" >> $LOG; sleep 30
done
echo "$(date -u +%T) card $G at $M MiB: start" >> $LOG
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 nice -n 10 $S/.venv/bin/python -I \
  $CC/scripts/experiments/selfmodel_lmlroute_20261126/lrt_plan.py $RT >> $LOG 2>&1 &
P=$!
echo "$(date -u +%T) planner pid $P card $G" >> $LOG
echo "$P $G" >> $RT/planner_pids.txt
bash $CC/scripts/experiments/selfmodel_lmlroute_20261126/lrt_guard.sh $G $P $LOG.guard &
wait $P
echo "$(date -u +%T) planner $P ended rc $?" >> $LOG
