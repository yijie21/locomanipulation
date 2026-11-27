#!/usr/bin/env bash
# selfmodel_xfer_20261125, combination 6 (HEFT -> SIMPLE, revision 1): freeze everything that runs into asrun_heft/.
# HEFT's SIMPLE interface lives in a separate checkout (loco_heft) that another agent keeps editing, so unlike asrun/
# (which checks the live repo file by file) the whole code the record and the evaluation import is copied here and run
# from here (sx_batch.sh with SX_ASRUN=asrun_heft: repo/FULL -> SX_REPO = asrun_heft/repo):
#   this experiment's scripts; data3's frozen collector (as sx_freeze.sh); from loco_heft: src/ (locomani, backends.heft
#   included), scripts/rsi, scripts/experiments/{selfwm_20261029,reachwm_20261030,locomanilab_20261117}, the two scenes.
#   assets/ and third_party/ (ScaleBFM's metadata, the venv) stay links to the live repo, as in loco_heft itself.
# HEFT's weights stay where HeftConfig reads them ($LOCOMANI_WORKSPACE/models/heft/pmg); their md5 is in MD5.check, which
# sx_batch.sh verifies (with every copied file) before it starts.
#   sx_freeze_heft.sh            (once; refuses when asrun_heft/MD5 exists)
H=/wuji-vefps-D/wuji-il/yijie/ws/code/loco_heft
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125
D3A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/asrun
PMG=/workspace/models/heft/pmg
A=$O/asrun_heft
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
set -e
mkdir -p $A/selfmodel_xfer_20261125 $A/selfmodel_data3_20261125 $A/locomanilab_20261117 $A/hardgrasp_20261125
mkdir -p $A/repo/scripts/experiments $A/repo/outputs/scene
cp $W/scripts/experiments/selfmodel_xfer_20261125/{sx_collect.py,sx_eval.py,sx_batch.sh,sx_freeze.sh,sx_freeze_heft.sh,sx_speed.py} \
  $A/selfmodel_xfer_20261125/
cp $D3A/selfmodel_data3_20261125/{d3_collect.py,d3_reach_table.json} $A/selfmodel_data3_20261125/
cp $D3A/locomanilab_20261117/{lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $D3A/hardgrasp_20261125/hg_walk.py $A/hardgrasp_20261125/
X=(--exclude __pycache__ --exclude .pytest_cache --exclude '*.pyc')
rsync -a "${X[@]}" $H/src $A/repo/
mkdir -p $A/repo/scripts
rsync -a "${X[@]}" $H/scripts/rsi $A/repo/scripts/
for e in selfwm_20261029 reachwm_20261030 locomanilab_20261117; do rsync -a "${X[@]}" $H/scripts/experiments/$e $A/repo/scripts/experiments/; done
for s in simple_g1_sonic walk_mars; do rsync -aL $H/outputs/scene/$s $A/repo/outputs/scene/; done
ln -s $W/assets $A/repo/assets
ln -s $W/third_party $A/repo/third_party
touch $A/repo/FULL
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -path './repo/assets/*' ! -path './repo/third_party/*' \
  | sort | xargs md5sum > MD5.check
md5sum $PMG/policy.onnx $PMG/policy.yaml $PMG/policy.json >> MD5.check
{ cat MD5.check; echo "# repo copy from $H (not a git checkout); assets -> $W/assets, third_party -> $W/third_party";
  echo "# live repo $W at $(cd $W && git rev-parse HEAD)"; } > MD5
date -u +%FT%TZ | tee FROZEN_UTC
md5sum $A/selfmodel_xfer_20261125/* $A/selfmodel_data3_20261125/d3_collect.py $A/repo/src/locomani/backends/heft/*.py
