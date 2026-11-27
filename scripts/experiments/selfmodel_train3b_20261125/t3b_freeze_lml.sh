#!/usr/bin/env bash
# selfmodel_train3b_20261125, lmlreal revision 1: freeze the lmlreal MX builder, evaluation and runner into $T/asrun_lml
# (MD5, GIT, FROZEN_UTC there); t3b_lml.sh checks them before it starts. asrun / asrun_mx stay as they were.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
T=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125
A=$T/asrun_lml
mkdir -p $A/selfmodel_train3b_20261125 $A/reachwm_20261030 $A/selfwm_20261029
cp $E/selfmodel_train3b_20261125/{t3b_lmlmx.py,t3b_lmleval.py,t3b_cref.py,t3b_lml.sh,t3b_freeze_lml.sh} $A/selfmodel_train3b_20261125/
cp $E/reachwm_20261030/{data,train,models}.py $A/reachwm_20261030/
cp $E/selfwm_20261029/{env,wm,fk}.py $A/selfwm_20261029/
(cd $A && md5sum */*.py */*.sh) > $A/MD5
(cd $W && git rev-parse HEAD && git status --short scripts/experiments/reachwm_20261030 scripts/experiments/selfwm_20261029) > $A/GIT
date -u +%FT%TZ > $A/FROZEN_UTC
cat $A/MD5 $A/FROZEN_UTC
for f in reachwm_20261030/data.py reachwm_20261030/train.py reachwm_20261030/models.py selfwm_20261029/env.py \
         selfwm_20261029/wm.py selfwm_20261029/fk.py selfmodel_train3b_20261125/t3b_cref.py; do
  cmp -s $A/$f $T/asrun/$f && echo "  $f = asrun's (train3b freeze)" || { echo "  $f DIFFERS from asrun's"; exit 1; }
done
