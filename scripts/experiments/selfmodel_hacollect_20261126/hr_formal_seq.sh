#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Revision 2a addendum: the rest of hr_formal.sh's jobs with SONIC before HEFT and HEFT on gait_heft. hr_formal.sh's bash
# wrapper is stopped by pid while ScaleBFM DoubleDesk's hr_real process runs (that process is left alone); this script
# waits for it, resumes ScaleBFM DoubleDesk only if it ended short of 20 logs (hr_real skips logged episodes), then
# runs hr_formal.sh's command lines unchanged: SONIC P&PBox / DoubleDesk 200-209 (--no-camera) from asrun_real; then,
# once asrun_real2 is frozen and $A2/GO exists (written after the addendum's commit), HEFT P&PBox / DoubleDesk 200-239
# from asrun_real2 with --heft-gait. Each frozen copy is verified before each process. Then hr_smoke_stats over every
# directory -> <out>/real/stats.txt, hold_1b removed, DONE (as hr_formal.sh). Run from a copy that is never edited:
#   cp hr_formal_seq.sh $OH/real/ && nohup bash $OH/real/hr_formal_seq.sh <ScaleBFM DoubleDesk pid> >> $OH/real/formal.log 2>&1 &
set -uo pipefail
OH=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126
OS=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_sonic_20261126
A=$OH/asrun_real    # ScaleBFM, SONIC: revision 2a's freeze
A2=$OH/asrun_real2  # HEFT: the addendum's freeze (hr_real with --heft-gait)
L=$OH/logs
VIA_RULE=lift
PREV=$1
touch $L/hold_1b
echo "$(date -u +%FT%TZ) hr_formal_seq start pid $$ (hold_1b kept; waiting for ScaleBFM DoubleDesk's hr_real pid $PREV)"
while kill -0 $PREV 2>/dev/null; do sleep 30; done
n() { ls $1/log/*.json 2>/dev/null | wc -l; }
echo "$(date -u +%FT%TZ) end scalebfm doubledesk (pid $PREV gone) episodes $(n $OH/real/scalebfm_doubledesk)"
run() {  # <frozen copy> <controller> <task> <out root> <seeds> [extra hr_real arguments]
  local F=$1 C=$2 T=$3 R=$4 SEEDS=$5; shift 5
  (cd $F && md5sum -c --quiet MD5.check) || { echo "the frozen copy $F or a pinned install changed"; exit 2; }
  local X=(); [ $C = sonic ] && X=(--no-camera)
  echo "$(date -u +%FT%TZ) start $C $T from $(basename $F) $*"
  SX_REPO=$F/repo SX_HA_EXP=$F/humanoidarena_20261124 SX_HA_CTRL=$F/ha_ctrl HC_XFER=$F/selfmodel_xfer_20261125 \
    SX_DEPTH=1 SX_HA_TASK=$T HC_SCRIPT=hr_real.py \
    bash $F/selfmodel_hacollect_20261126/hc_ha.sh 1 --controller $C --out $R/real/${C}_$T --seeds $SEEDS --repeats 0 \
    --via 1 --via-rule $VIA_RULE "${X[@]}" "$@" >> $R/real/${C}_$T.log 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $C $T exit $rc episodes $(n $R/real/${C}_$T)"
}
[ $(n $OH/real/scalebfm_doubledesk) -lt 20 ] && run $A scalebfm doubledesk $OH 200-219
run $A sonic ppbox $OS 200-209
run $A sonic doubledesk $OS 200-209
[ -e $A2/GO ] || echo "$(date -u +%FT%TZ) waiting for $A2/GO (the addendum's commit and freeze)"
while [ ! -e $A2/GO ] || [ ! -e $A2/MD5 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) GO: $(cat $A2/GO)"
run $A2 heft ppbox $OH 200-239 --heft-gait
run $A2 heft doubledesk $OH 200-239 --heft-gait
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
for R in $OH $OS; do
  (cd /tmp && $VPY -I $A/selfmodel_hacollect_20261126/hr_smoke_stats.py $R/real/*_*/ > $R/real/stats.txt 2>&1)
done
rm -f $L/hold_1b
date -u +%FT%TZ | tee $OH/real/DONE
