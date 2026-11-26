#!/usr/bin/env bash
# notebook: skip heft_p4robust pipeline smoke (pipeline only; the rep-300 trials re-run identically in the formal)
# heft_p4robust_20261126 smoke on H20 CPU, output $P/smoke (4 batches, 1 process each): the default-path check (p4, p2 at
# cells 1, 5, 6, rep 0 -- compared with the 2x2's records by pr_sum --x2), p4j4 / p4j / p2tj1 at 1.50 m rep 300, HEFT
# lateral +-2 cm from the three stances, SONIC and ScaleBFM lateral -2 cm from the stand.
set -uo pipefail
P=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_p4robust_20261126
C=$P/code
(cd $C && md5sum -c --quiet $P/asrun/CODE_MD5) || { echo "code changed since CODE_MD5"; exit 2; }
export HW_OUT=$P/smoke LC_CODE_ROOT=$C LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws SONIC_POLICY=/wuji-vefps-D/wuji-il/yijie/ws/sonic_models/sonic_v1_1
B=$C/scripts/experiments/heft_walkstop_20261126/hw_batch.sh
mkdir -p $HW_OUT
bash $B p4,p2 1,5,6 0:1 1 >> $HW_OUT/smoke.log 2>&1 &
bash $B p4j4,p4j,p2tj1 6 300:301 1 >> $HW_OUT/smoke.log 2>&1 &
bash $B lhn,lhs,lhw 17,18 300:301 1 >> $HW_OUT/smoke.log 2>&1 &
( bash $B lqn 18 300:301 1; bash $B lbn 18 300:301 1 ) >> $HW_OUT/smoke.log 2>&1 &
wait
date -u +%FT%TZ > $HW_OUT/DONE
