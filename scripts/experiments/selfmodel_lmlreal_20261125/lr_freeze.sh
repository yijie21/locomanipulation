#!/usr/bin/env bash
# notebook: selfmodel_lmlreal_20261125
# Freeze the code selfmodel_lmlreal_20261125 runs into $O/asrun (same relative layout under scripts/experiments) and
# write $O/asrun/MD5 (the files, then the inputs read: hardgrasp's hostdata model and joint band, train3's frac-1.0
# models, data3's training data) and $O/asrun/FROZEN_UTC. The runs execute the repo copy in $W/scripts/experiments (selfwm
# env.py takes the repo root from its own location, so a copy under outputs cannot run: hardgrasp's launch 1), checked
# against this list before and after (lr_formal.sh).
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125
A=$O/asrun
HGO=$W/outputs/hardgrasp_20261125/locomanilab
D3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125
mkdir -p $A/selfmodel_lmlreal_20261125 $A/hardgrasp_20261125 $A/selfmodel_data3_20261125 $A/locomanilab_20261117 \
  $A/reachwm_20261030 $A/selfwm_20261029
cp $E/selfmodel_lmlreal_20261125/{lr_run.py,lr_check.py,lr_eval.py,lr_batch.sh,lr_formal.sh,lr_freeze.sh} $A/selfmodel_lmlreal_20261125/
cp $E/hardgrasp_20261125/{hg_run.py,hg_look.py,hg_walk.py} $A/hardgrasp_20261125/
cp $E/selfmodel_data3_20261125/{d3_collect.py,d3_sonic.py} $A/selfmodel_data3_20261125/
cp $E/locomanilab_20261117/{lml_tasks.py,lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $E/reachwm_20261030/{data.py,models.py,train.py} $A/reachwm_20261030/
cp $E/selfwm_20261029/{env.py,fk.py,wm.py} $A/selfwm_20261029/
cd $A
md5sum */*.py */*.sh > MD5
md5sum $HGO/wm_s0.pt $HGO/train_q_range.npz >> MD5
for BE in scalebfm heft sonic; do
  md5sum $D3/$BE/models_s0/E3_ensemble_f1/model.pt $D3/$BE/data/all.npz >> MD5
done
date -u +%FT%TZ > FROZEN_UTC
cat FROZEN_UTC; cat MD5
