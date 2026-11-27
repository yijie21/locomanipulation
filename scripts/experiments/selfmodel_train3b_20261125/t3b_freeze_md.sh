#!/usr/bin/env bash
# selfmodel_train3b_20261125: freeze t3b_mdirect.py and its runner into $T/asrun_md (MD5, GIT, FROZEN_UTC there).
# t3b_mdirect.py imports t3b_eval.py from asrun_mx (md5 checked at run time); asrun, asrun_mx, asrun_lml stay as they were.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
T=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125
A=$T/asrun_md
mkdir -p $A/selfmodel_train3b_20261125
cp $E/selfmodel_train3b_20261125/{t3b_mdirect.py,t3b_md.sh,t3b_freeze_md.sh} $A/selfmodel_train3b_20261125/
(cd $A && md5sum */*.py */*.sh) > $A/MD5
(cd $W && git rev-parse HEAD) > $A/GIT
date -u +%FT%TZ > $A/FROZEN_UTC
cat $A/MD5 $A/FROZEN_UTC
md5sum $T/asrun_mx/selfmodel_train3b_20261125/t3b_eval.py
