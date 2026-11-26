#!/usr/bin/env bash
# notebook: heft_walkstop_20261126
# heft_walkstop_20261126 revisions 2 + 3, the formal HA free-space part on 4090b card 0 (b0b slot, one process at a time):
# HEFT P0, P2, then P4 (one process; jobs run arm by arm, so P0+P2 before P4), then SONIC Q0 / Q2 unless $O/ha/NO_SONIC
# exists; cells 0.05 / 0.20 / 0.50 / 1.00 m (0, 2, 4, 5), reps 0-4. Card 0 must have <= 12000 MiB used before each process.
# Output $O/ha (not $O). DONE marker with the exit codes. hold_b0b is removed by hand after the run.
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126
S=$O/ha GPU=0
mkdir -p $S
echo 100 > $S/TOTAL
: > $S/steps.log
step() {
  local n=$1; shift
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  echo "$(date -u +%FT%TZ) $n: card $GPU used ${used} MiB before start" >> $S/steps.log
  if [ "$used" -gt 12000 ]; then echo "$(date -u +%FT%TZ) $n: SKIPPED (card $GPU over 12000 MiB)" >> $S/steps.log; return 3; fi
  bash $O/hw_ha_b.sh $GPU --out $S "$@" > $S/$n.log 2>&1 &
  local p=$!
  echo "$p $n $(date -u +%FT%TZ)" >> $S/pids.txt
  (sleep 120; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $S/$n.gpu 2>&1; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $S/$n.gpu) &
  wait $p; local rc=$?
  echo "$(date -u +%FT%TZ) $n: exit $rc" >> $S/steps.log
  return $rc
}
step heft --controller heft --arms p0,p2,p4 --cells 0,2,4,5 --reps 0:5 --traces; r1=$?
if [ -e $S/NO_SONIC ]; then echo "$(date -u +%FT%TZ) sonic: not run (NO_SONIC)" >> $S/steps.log; r2=skip; echo 60 > $S/TOTAL
else step sonic --controller sonic --arms q0,q2 --cells 0,2,4,5 --reps 0:5 --traces; r2=$?; fi
echo "$r1 $r2" > $S/DONE
