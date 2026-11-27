#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# The formal real HA episodes (selfmodel_hadata_20261126 revision 2a; SONIC: selfmodel_hadata_sonic_20261126's revision),
# one hr_real.py process at a time on card 1 in the collection scheduler's slot 1b (held by $L/hold_1b while this runs;
# removed at the end so the scheduler takes the slot back), from the frozen copy asrun_real (hr_freeze.sh), verified
# before each process. Dev seeds, repeat 0, per controller and task (JOBS below: ScaleBFM 200-219, HEFT 200-239, SONIC
# 200-209; revision 2a, from the smoke's share of clean reaches); the lift rule (revision 2a's choice). hr_real skips
# episodes whose log exists, so a rerun resumes. Then hr_smoke_stats over every directory -> <out>/real/stats.txt, DONE.
#   nohup bash hr_formal.sh > $OH/real/formal.log 2>&1 &
set -uo pipefail
OH=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126
OS=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_sonic_20261126
A=$OH/asrun_real
L=$OH/logs
VIA_RULE=lift
# controller task out-root seeds
JOBS=("scalebfm ppbox $OH 200-219" "scalebfm doubledesk $OH 200-219" "heft ppbox $OH 200-239" "heft doubledesk $OH 200-239"
      "sonic ppbox $OS 200-209" "sonic doubledesk $OS 200-209" ${SONIC_DOOR:+"sonic opendoor $OS 200-209"})
[ -e $A/MD5 ] || { echo "not frozen: $A"; exit 2; }
mkdir -p $OH/real $OS/real
touch $L/hold_1b
echo "$(date -u +%FT%TZ) hr_formal start pid $$ (hold_1b set; via rule $VIA_RULE)"
for _ in $(seq 1 120); do  # the slot's current shard (if any) finishes first: card 1 keeps two processes
  r=$(grep -l "^1b " $L/jobs/*.run 2>/dev/null | while read f; do t=$(basename $f .run); [ -e $L/jobs/$t.done ] || echo $t; done)
  [ -z "$r" ] && break
  echo "$(date -u +%FT%TZ) waiting for slot 1b's shard $r"; sleep 300
done
for j in "${JOBS[@]}"; do
  set -- $j; C=$1 T=$2 R=$3 SEEDS=$4
  (cd $A && md5sum -c --quiet MD5.check) || { echo "the frozen copy or a pinned install changed"; exit 2; }
  X=(); [ $C = sonic ] && X=(--no-camera)
  echo "$(date -u +%FT%TZ) start $C $T"
  SX_REPO=$A/repo SX_HA_EXP=$A/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$A/selfmodel_xfer_20261125 \
    SX_DEPTH=1 SX_HA_TASK=$T HC_SCRIPT=hr_real.py \
    bash $A/selfmodel_hacollect_20261126/hc_ha.sh 1 --controller $C --out $R/real/${C}_$T --seeds $SEEDS --repeats 0 \
    --via 1 --via-rule $VIA_RULE "${X[@]}" >> $R/real/${C}_$T.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end $C $T exit $rc episodes $(ls $R/real/${C}_$T/log/*.json 2>/dev/null | wc -l)"
done
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
for R in $OH $OS; do
  (cd /tmp && $VPY -I $A/selfmodel_hacollect_20261126/hr_smoke_stats.py $R/real/*_*/ > $R/real/stats.txt 2>&1)
done
rm -f $L/hold_1b
date -u +%FT%TZ | tee $OH/real/DONE
