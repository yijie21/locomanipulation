#!/usr/bin/env bash
# notebook: heft_p4robust_20261126
# heft_p4robust_20261126 formal run on H20 CPU: pr_run.sh <NA> <NB> <NC> <ND> <NE> (processes per batch, sum <= 24 and
# measured + sum <= 75 cores). Batches (hw_batch.sh, one backend each, all concurrently):
#   A p4j4,p4j    cells 0-10, 13-16  reps 300-319   B p2tj,p2tj1  cells 5,6,13-16  reps 300-319
#   C lhn,lhs,lhw cells 17-22 reps 300-309          D lqn,lqs,lqw (SONIC) / E lbn,lbs,lbw (ScaleBFM) cells 17-22 reps 300-302
# Code: $P/code (asrun/CODE_MD5 verified first). Output $P. FORMAL_DONE marker.
set -uo pipefail
P=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_p4robust_20261126
C=$P/code
(cd $C && md5sum -c --quiet $P/asrun/CODE_MD5) || { echo "code changed since CODE_MD5"; exit 2; }
[ -e $P/FORMAL_START ] && { echo "already started"; exit 3; }
export HW_OUT=$P LC_CODE_ROOT=$C LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws SONIC_POLICY=/wuji-vefps-D/wuji-il/yijie/ws/sonic_models/sonic_v1_1
B=$C/scripts/experiments/heft_walkstop_20261126/hw_batch.sh
{ md5sum $P/asrun/CODE_MD5; date -u +%FT%TZ; echo "processes A $1 B $2 C $3 D $4 E $5"; } > $P/FORMAL_START
bash $B p4j4,p4j 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16 300:320 $1 >> $P/formal.log 2>&1 &
bash $B p2tj,p2tj1 5,6,13,14,15,16 300:320 $2 >> $P/formal.log 2>&1 &
bash $B lhn,lhs,lhw 17:23 300:310 $3 >> $P/formal.log 2>&1 &
bash $B lqn,lqs,lqw 17:23 300:303 $4 >> $P/formal.log 2>&1 &
bash $B lbn,lbs,lbw 17:23 300:303 $5 >> $P/formal.log 2>&1 &
wait
date -u +%FT%TZ > $P/FORMAL_DONE
