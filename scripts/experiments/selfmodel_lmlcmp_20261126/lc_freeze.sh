#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp_20261126
# Freeze the code selfmodel_lmlcmp_20261126 runs into $O/asrun (same relative layout under scripts/experiments) and
# write $O/asrun/MD5 (the files, then the inputs read: the T3 / M / D model.pt of every controller, lmlreal's records
# (the models' rebuild template), hardgrasp's joint band, data3's data (lc_eval's band and val check)),
# $O/asrun/LOOKS_MD5 (hardgrasp's cached looks of the 7 tasks) and $O/asrun/FROZEN_UTC. The runs execute the repo copy
# in $W/scripts/experiments (selfwm env.py takes the repo root from its own location), checked against this list before
# and after (lc_formal.sh).
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp_20261126
A=$O/asrun
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
HGO=$W/outputs/hardgrasp_20261125/locomanilab
mkdir -p $A/selfmodel_lmlcmp_20261126/heft_gait $A/selfmodel_lmlreal_20261125 $A/hardgrasp_20261125 $A/selfmodel_data3_20261125 \
  $A/locomanilab_20261117 $A/reachwm_20261030 $A/selfwm_20261029
cp $E/selfmodel_lmlcmp_20261126/{lc_run.py,lc_check.py,lc_eval.py,lc_batch.sh,lc_formal.sh,lc_freeze.sh} $A/selfmodel_lmlcmp_20261126/
cp $E/selfmodel_lmlcmp_20261126/heft_gait/{gait.py,gait_heft.py} $A/selfmodel_lmlcmp_20261126/heft_gait/
cp $E/selfmodel_lmlreal_20261125/lr_run.py $A/selfmodel_lmlreal_20261125/
cp $E/hardgrasp_20261125/{hg_run.py,hg_look.py,hg_walk.py} $A/hardgrasp_20261125/
cp $E/selfmodel_data3_20261125/{d3_collect.py,d3_sonic.py} $A/selfmodel_data3_20261125/
cp $E/locomanilab_20261117/{lml_tasks.py,lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $E/reachwm_20261030/{data.py,models.py,train.py} $A/reachwm_20261030/
cp $E/selfwm_20261029/{env.py,fk.py,wm.py} $A/selfwm_20261029/
cd $A
md5sum */*.py */*.sh */heft_gait/*.py > MD5
md5sum $HGO/train_q_range.npz >> MD5
for BE in scalebfm heft sonic; do
  md5sum $R/selfmodel_data3_20261125/$BE/models_s0/E3_ensemble_f1/model.pt $R/selfmodel_train3b_20261125/$BE/models_M_s0/E3_ensemble_f1/model.pt \
    $R/selfmodel_train3b_20261125/$BE/models_D_s0/E3_ensemble_f1/model.pt $R/selfmodel_lmlreal_20261125/$BE/data/all.npz \
    $R/selfmodel_data3_20261125/$BE/data/all.npz >> MD5
done
(cd $HGO/looks && md5sum $(ls -d */ | tr -d / | sed 's|$|/*.npz|') | sed "s|^\([0-9a-f]*\)  |\1  $HGO/looks/|") > LOOKS_MD5
date -u +%FT%TZ > FROZEN_UTC
cat FROZEN_UTC; cat MD5; echo "looks: $(wc -l < LOOKS_MD5) files"
