#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# lrt_guard.sh <gpu> <pid> <log>: every 5 s log the card's memory and utilisation; above 20000 MiB stop <pid> (ours) by pid
G=$1; P=$2; LOG=$3
while kill -0 $P 2>/dev/null; do
  L=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i $G)
  echo "$(date -u +%T) $L" >> $LOG
  M=$(echo $L | cut -d, -f2 | tr -d ' ')
  if [ "$M" -gt 20000 ]; then echo "$(date -u +%T) over 20000 MiB: kill $P" >> $LOG; kill $P; fi
  sleep 5
done
echo "$(date -u +%T) $P ended" >> $LOG
