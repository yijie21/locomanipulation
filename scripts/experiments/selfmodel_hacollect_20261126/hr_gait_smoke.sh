#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Revision 2a addendum's checks before HEFT's formal episodes go to gait_heft, on 4090b card 0 in the collection slot
# b0b (held by $L/hold_b0b, lent for this; the hold stays: the coordinator hands b0b on afterwards), the dev copy with
# the 4090b environment (hr_video.sh's), SAM 3 through the Mac relay. hr_real_gait.py = the addendum's hr_real.
#   1. switch on: HEFT P&PBox and DoubleDesk, dev seeds 296-298, lift rule, --heft-gait, with hr_video's recorder
#      (--hrvid: foot loads, lift-offs) -> $G/on/heft_<task>, $G/vid_<task>
#   2. switch off, bit for bit: HEFT P&PBox 296 with smoke6's colour images (--replay-looks), hr_real.py (e92180f9,
#      asrun_real's) and hr_real_gait.py without the switch -> $G/same_{old,new}; hr_same new vs old
#   nohup bash hr_gait_smoke.sh > $G/smoke.log 2>&1 &
set -uo pipefail
E=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments/selfmodel_hacollect_20261126
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments
A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/asrun
DV=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev
D=$DV/b4090b
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
G=$DV/real_gait
S6=$DV/real_smoke6/lift/heft_ppbox
PY=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
mkdir -p $G
[ -e $L/hold_b0b ] || { echo "no hold_b0b"; exit 2; }
echo "$(date -u +%FT%TZ) start on $(hostname) card 0; hr_real $(md5sum < $E/hr_real.py | cut -c1-8), hr_real_gait $(md5sum < $E/hr_real_gait.py | cut -c1-8)"
unset SX_D3 SX_NORENDER
export SX_REPO=$D/repo_b SX_HA_EXP=$W/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$W/selfmodel_xfer_20261125
export LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws
cd $E
run() {  # script task seeds out log [extra]
  local S=$1 T=$2 SEEDS=$3 O=$4 LG=$5; shift 5
  SX_DEPTH=1 SX_HA_TASK=$T HC_SCRIPT=$S bash hc_ha.sh 0 --controller heft --out $O --seeds $SEEDS --repeats 0 \
    --via 1 --via-rule lift "$@" > $LG 2>&1
  echo "$(date -u +%FT%TZ) $S $T $SEEDS $*: exit $?; episodes $(ls $O/log/*.json 2>/dev/null | wc -l)"
}
for T in ppbox doubledesk; do
  run hr_real_gait.py $T 296-298 $G/on/heft_$T $G/on_$T.log --heft-gait --hrvid $G/vid_$T
  grep -a "heft-gait:\|hr_real heft $T ${T}_s" $G/on_$T.log | cut -c1-260
done
echo "SWITCH-ON DONE"
run hr_real_gait.py ppbox 296 $G/same_new/heft_ppbox $G/same_new.log --replay-looks $S6/looks
run hr_real.py ppbox 296 $G/same_old/heft_ppbox $G/same_old.log --replay-looks $S6/looks
for f in new old; do echo "$f: $(grep -c "replay-looks: rgb from" $G/same_$f.log) looks replayed, $(grep -c "missing; live rgb" $G/same_$f.log) missing, $(grep -c "heft-gait:" $G/same_$f.log) heft-gait lines"; done
for pr in "new old" "new smoke6"; do
  set -- $pr; a=$G/same_$1/heft_ppbox; b=$G/same_$2/heft_ppbox; [ $2 = smoke6 ] && b=$S6
  echo "== $1 vs $2"; (cd /tmp && $PY -I $E/hr_same.py $a $b ppbox_s296_r0) | tee $G/same_${1}_$2.txt | grep -E "^(IDENTICAL|DIFFERENT)|arrays compared"
done
echo "$(date -u +%FT%TZ) SMOKE DONE"
