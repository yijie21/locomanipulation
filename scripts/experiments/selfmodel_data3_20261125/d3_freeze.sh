#!/usr/bin/env bash
# selfmodel_data3_20261125: copy the collector and everything it imports into <out>/<backend>/asrun, print md5s.
#   d3_freeze.sh <backend>...        (D3_NB=selfmodel_data3b_20261125: data3b's output directory, with d3b_formal.sh)
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments
NB=${D3_NB:-selfmodel_data3_20261125}
for BE in "$@"; do
  A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/$NB/$BE/asrun
  mkdir -p $A/selfmodel_data3_20261125 $A/locomanilab_20261117 $A/hardgrasp_20261125
  cp $W/selfmodel_data3_20261125/{d3_collect.py,d3_batch.sh,d3_formal.sh,d3_reach_table.json} $A/selfmodel_data3_20261125/
  [ "$NB" = selfmodel_data3b_20261125 ] && cp $W/selfmodel_data3_20261125/d3b_formal.sh $A/selfmodel_data3_20261125/
  [ "$BE" = sonic ] && cp $W/selfmodel_data3_20261125/d3_sonic.py $A/selfmodel_data3_20261125/
  cp $W/locomanilab_20261117/{lml_host.py,lml_selfmodel.py} $A/locomanilab_20261117/
  cp $W/hardgrasp_20261125/hg_walk.py $A/hardgrasp_20261125/
  (cd $A && md5sum */*.py */*.sh */*.json) | tee $A/MD5
done
