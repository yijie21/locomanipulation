#!/usr/bin/env bash
# Freeze the LocoManiLab code of selfmodel_hardgrasp_20261125 into $O/asrun (same relative layout) and record md5s
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; O=$W/outputs/hardgrasp_20261125/locomanilab; E=$W/scripts/experiments
mkdir -p $O/asrun/hardgrasp_20261125 $O/asrun/locomanilab_20261117 $O/asrun/reachwm_20261030 $O/asrun/selfwm_20261029
cp $E/hardgrasp_20261125/hg_{run,look,walk,ggx_server}.py $E/hardgrasp_20261125/hg_{batch,formal}.sh $O/asrun/hardgrasp_20261125/
cp $E/locomanilab_20261117/*.py $O/asrun/locomanilab_20261117/
cp $E/reachwm_20261030/*.py $O/asrun/reachwm_20261030/
cp $E/selfwm_20261029/*.py $O/asrun/selfwm_20261029/
cd $O/asrun && md5sum */*.py */*.sh > MD5SUMS && md5sum ../wm_s0.pt ../train_q_range.npz >> MD5SUMS
date -u +%FT%TZ > FROZEN_AT
cat FROZEN_AT; grep hardgrasp MD5SUMS; grep -c . MD5SUMS
