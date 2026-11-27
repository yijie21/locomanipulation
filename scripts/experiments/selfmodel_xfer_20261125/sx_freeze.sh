#!/usr/bin/env bash
# selfmodel_xfer_20261125: freeze the code that runs. Copies this directory's scripts, data3's frozen collector
# (selfmodel_data3_20261125/scalebfm/asrun: d3_collect.py 581fadf5, its reach table, lml_selfmodel / lml_host, hg_walk) in
# the same layout (so sx_collect.py finds d3_collect next to it), and, for the record, the repo modules the SIMPLE body
# and the evaluation import (they run from the repo; sx_batch.sh refuses to start when one differs from its copy here).
# Writes asrun/MD5 (+ the repo's git rev and status of those paths) and asrun/FROZEN_UTC.
#   sx_freeze.sh            (once; refuses when asrun/MD5 exists)
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125
D3A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/asrun
A=$O/asrun
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
mkdir -p $A/selfmodel_xfer_20261125 $A/selfmodel_data3_20261125 $A/locomanilab_20261117 $A/hardgrasp_20261125 $A/repo
cp $W/scripts/experiments/selfmodel_xfer_20261125/{sx_collect.py,sx_eval.py,sx_batch.sh,sx_freeze.sh} $A/selfmodel_xfer_20261125/
cp $D3A/selfmodel_data3_20261125/{d3_collect.py,d3_reach_table.json} $A/selfmodel_data3_20261125/
cp $D3A/locomanilab_20261117/{lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $D3A/hardgrasp_20261125/hg_walk.py $A/hardgrasp_20261125/
cd $W || exit 1
REL=$(ls scripts/experiments/selfwm_20261029/env.py scripts/rsi/capability_probe.py \
  scripts/experiments/reachwm_20261030/{leg_odo,data,train,models}.py \
  src/locomani/backends/scalebfm/*.py src/locomani/backends/sonic_onnx/*.py src/locomani/primitives/*.py src/locomani/paths.py \
  outputs/scene/simple_g1_sonic/scene_locomani.xml outputs/scene/walk_mars/scene_locomani.xml)
for r in $REL; do mkdir -p $A/repo/$(dirname $r); cp $r $A/repo/$r; done
echo "$REL" > $A/repo/FILES
(cd $A && find . -type f ! -path '*/__pycache__/*' ! -name MD5 ! -name FROZEN_UTC | sort | xargs md5sum) > $A/MD5
{ echo "# repo $W"; git rev-parse HEAD; git status --short $REL scripts/experiments/selfmodel_xfer_20261125; } >> $A/MD5
date -u +%FT%TZ | tee $A/FROZEN_UTC
md5sum $A/selfmodel_xfer_20261125/* $A/selfmodel_data3_20261125/d3_collect.py
