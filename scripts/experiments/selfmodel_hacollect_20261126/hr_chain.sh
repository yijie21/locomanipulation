#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Revision 2a's checks before the freeze, in the collection's slot 1b once its shard is done (hold_1b set by hand):
# (1) the contact record does not change an episode: ScaleBFM PPBox seed 296, lift, with the new hr_real against
#     dev/real_smoke6 (hr_same.py); on a difference, the pre-record hr_real (hr_real_old.py) once more against
#     real_smoke6, to tell the change from run-to-run noise; (2) SONIC's smoke: PPBox and DoubleDesk seeds 296-299,
#     OpenDoor 296-297, lift, its CLEAR, no per-step camera. Dev copy; writes dev/real_check7/ and dev/real_smoke7/.
set -uo pipefail
E=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments/selfmodel_hacollect_20261126
DV=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
PY=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
C=$DV/real_check7 S=$DV/real_smoke7
mkdir -p $C $S
[ -e $L/hold_1b ] || { echo "no hold_1b"; exit 2; }
while :; do
  r=$(grep -l "^1b " $L/jobs/*.run 2>/dev/null | while read f; do t=$(basename $f .run); [ -e $L/jobs/$t.done ] || echo $t; done)
  [ -z "$r" ] && break; sleep 60
done
echo "$(date -u +%FT%TZ) slot 1b free; hr_real $(md5sum < $E/hr_real.py | cut -c1-8)"
cd $E
run() {  # script controller task seeds out log [extra]
  local s=$1 c=$2 t=$3 sd=$4 o=$5 lg=$6; shift 6
  SX_DEPTH=1 SX_HA_TASK=$t HC_SCRIPT=$s bash hc_ha.sh 1 --controller $c --out $o --seeds $sd --repeats 0 --via 1 --via-rule lift "$@" > $lg 2>&1
  echo "$(date -u +%FT%TZ) $s $c $t $sd: exit $?"
}
run hr_real.py scalebfm ppbox 296 $C/new/scalebfm_ppbox $C/new.log
(cd /tmp && $PY -I $E/hr_same.py $C/new/scalebfm_ppbox $DV/real_smoke6/lift/scalebfm_ppbox ppbox_s296_r0) | tee $C/same_new.txt
if ! grep -q "^IDENTICAL" $C/same_new.txt; then
  run hr_real_old.py scalebfm ppbox 296 $C/old/scalebfm_ppbox $C/old.log
  (cd /tmp && $PY -I $E/hr_same.py $C/old/scalebfm_ppbox $DV/real_smoke6/lift/scalebfm_ppbox ppbox_s296_r0) | tee $C/same_old.txt
fi
run hr_real.py sonic ppbox 296-299 $S/sonic_ppbox $S/sonic_ppbox.log --no-camera
run hr_real.py sonic doubledesk 296-299 $S/sonic_doubledesk $S/sonic_doubledesk.log --no-camera
run hr_real.py sonic opendoor 296-297 $S/sonic_opendoor $S/sonic_opendoor.log --no-camera
(cd /tmp && $PY -I $E/hr_smoke_stats.py $S/sonic_*/ $C/new/scalebfm_ppbox/) > $S/stats.txt 2>&1
cat $S/stats.txt
echo "$(date -u +%FT%TZ) CHAIN DONE"
