#!/usr/bin/env bash
# notebook: skip heft_walkstop revision 2 HA smoke on 4090b
# heft_walkstop_20261126 revisions 2 + 3, HA smoke on 4090b card 0 (the b0b slot, one process at a time): probe; the gait
# check (revision 3: P0 at 0.20 m rep 100 alone, its first walk also built by hadata's hr_real_gait.heft_walk_frames from
# $O/ref/hr_real_gait.py, md5 7dfaa816); HEFT P0 / P2 / P4 at 0.20 / 0.50 m (rep 100; P0 0.20 done by the check); p0hg at
# 0.20 m (reps 100, 101); SONIC Q0 at 0.20 m (rep 100).
# Before each process: card 0 must have <= 12000 MiB used (b0a's ~7.8 GB + ours ~7.8 GB <= 20 GB). Output $O/ha_smoke
# (not $O: the LML summaries glob $O/*/runs). DONE marker with the exit codes.
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126
S=$O/ha_smoke GPU=0
mkdir -p $S
: > $S/steps.log
step() {  # step <name> <hw_ha.py args...>
  local n=$1; shift
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  echo "$(date -u +%FT%TZ) $n: card $GPU used ${used} MiB before start" >> $S/steps.log
  if [ "$used" -gt 12000 ]; then echo "$(date -u +%FT%TZ) $n: SKIPPED (card $GPU over 12000 MiB)" >> $S/steps.log; return 3; fi
  bash $O/hw_ha_b.sh $GPU --out $S "$@" > $S/$n.log 2>&1 &
  local p=$!
  echo "$p $n $(date -u +%FT%TZ)" >> $S/pids.txt
  (sleep 90; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $S/$n.gpu 2>&1; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $S/$n.gpu) &
  wait $p; local rc=$?
  echo "$(date -u +%FT%TZ) $n: exit $rc" >> $S/steps.log
  return $rc
}
step probe --controller heft --probe; r1=$?
step gait --controller heft --arms p0 --cells 2 --reps 100:101 --traces --gait-check $O/ref/hr_real_gait.py; r2=$?
step heft --controller heft --arms p0,p2,p4 --cells 2,4 --reps 100:101 --traces; r2b=$?
step p0hg --controller heft --arms p0hg --cells 2 --reps 100:102 --traces; r3=$?
step sonic --controller sonic --arms q0 --cells 2 --reps 100:101 --traces; r4=$?
echo "$r1 $r2 $r2b $r3 $r4" > $S/DONE
