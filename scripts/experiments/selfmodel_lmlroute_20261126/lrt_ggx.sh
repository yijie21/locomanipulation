#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# lrt_ggx.sh <gpu> <port> <log>: hardgrasp's GraspGen-X service (hg_ggx_server.py, the round's copy) on 4090 card <gpu>
# (never 0) once the card is <= 18500 MiB (the service ~1.0 GB: the card stays <= 20 GB); a guard stops it (by pid) above
# 20000 MiB. The pid goes to <log>.pid.
G=$1; PORT=$2; LOG=$3
[ "$G" = 0 ] && { echo "card 0 is not ours" >> $LOG; exit 1; }
LR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
W=${LRT_CODE:-$LR/code}
P=/wuji-vefps-D/wuji-il/yijie/tools/perception
until M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' '); [ "$M" -le 18500 ]; do
  echo "$(date -u +%T) card $G at $M MiB: waiting" >> $LOG; sleep 30
done
echo "$(date -u +%T) card $G at $M MiB: start GraspGen-X on port $PORT" >> $LOG
cd $W/scripts/experiments/hardgrasp_20261125
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYOPENGL_PLATFORM=egl nice -n 10 $P/GraspGenX/.venv/bin/python hg_ggx_server.py --port $PORT \
  --assets-dir $P/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets >> $LOG 2>&1 &
PID=$!
echo $PID > $LOG.pid
echo "$(date -u +%T) GraspGen-X pid $PID card $G" >> $LOG
bash $LR/dev/guard.sh $G $PID $LOG.guard &
wait $PID
echo "$(date -u +%T) GraspGen-X $PID ended rc $?" >> $LOG
