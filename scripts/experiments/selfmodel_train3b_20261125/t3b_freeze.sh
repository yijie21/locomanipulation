#!/usr/bin/env bash
# selfmodel_train3b_20261125: copy this directory's code and the reachwm / selfwm files it runs into $T/asrun, write
# $T/asrun/MD5 and $T/FROZEN_UTC (t3b_run.sh checks both before it starts).
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
T=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125
A=$T/asrun
mkdir -p $A/selfmodel_train3b_20261125 $A/reachwm_20261030 $A/selfwm_20261029 $A/selfmodel_data3_20261125
cp $E/selfmodel_train3b_20261125/{t3b_mid.py,t3b_split_check.py,t3b_cref.py,t3b_train.py,t3b_equiv.py,t3b_eval.py,t3b_run.sh,t3b_freeze.sh} $A/selfmodel_train3b_20261125/
cp $E/reachwm_20261030/{data,train,models}.py $A/reachwm_20261030/
cp $E/selfwm_20261029/{env,wm,fk}.py $A/selfwm_20261029/
cp $E/selfmodel_data3_20261125/d3_eval.py $A/selfmodel_data3_20261125/
(cd $A && md5sum */*.py */*.sh) > $A/MD5
(cd $W && git rev-parse HEAD && git status --short scripts/experiments/reachwm_20261030 scripts/experiments/selfwm_20261029) > $A/GIT
date -u +%FT%TZ > $T/FROZEN_UTC
cat $A/MD5 $A/GIT $T/FROZEN_UTC
echo "against selfmodel_train3_20261125's frozen training code (heft/asrun/train/MD5):"
F=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/asrun/train/MD5
for f in reachwm_20261030/data.py reachwm_20261030/train.py reachwm_20261030/models.py selfwm_20261029/env.py selfmodel_data3_20261125/d3_eval.py; do
  m=$(md5sum $A/$f | cut -c1-32); t=$(grep " $(basename $f)\$" $F | cut -c1-32)
  [ "$m" = "$t" ] && echo "  $f = train3's ($m)" || { echo "  $f DIFFERS from train3's ($m vs $t)"; exit 1; }
done
