#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# Freeze selfmodel_freespace_20261126's code as the round's PRIVATE COPY C0 = $O/code (lc3_freeze.sh's way: the shared
# repo tree on 4090 gets synced into by other rounds, so the formal run imports only from this copy):
#   copied (rsync -a, no __pycache__): src/, configs/, scripts/rsi, and the scripts/experiments directories of the chain:
#     freespace, lmlcmp3 (lc3_run), lmlcmp2, lmlcmp (heft_gait), lmlreal (lr_run), hardgrasp (hg_run, hg_walk), data3
#     (d3_collect, d3_sonic, the reach table), locomanilab (lml_*), reachwm (data / train / models), selfwm;
#   linked to the shared tree (data and third-party trees reached through repo-relative paths; lc3_run's code check
#     stops any run process that imports a module through them): assets, outputs, third_party, grasp_kit, rsi_kit.
# Writes $A = $O/asrun: CODE_MD5 (every file of the copy, relative to C0), MD5 (the inputs: grid/ labels, the T3 / M / D
# model.pt of every controller, lmlreal's data (the models' template), D's training files (fs_eval's start coverage)),
# FROZEN_UTC. One shot: refuses an existing copy.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_freespace_20261126
A=$O/asrun
C0=$O/code
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
DIRS="selfmodel_freespace_20261126 selfmodel_lmlcmp3_20261126 selfmodel_lmlcmp2_20261126 selfmodel_lmlcmp_20261126 selfmodel_lmlreal_20261125 hardgrasp_20261125 selfmodel_data3_20261125 locomanilab_20261117 reachwm_20261030 selfwm_20261029"
[ -e $C0 ] && { echo "$C0 exists: refusing"; exit 1; }
[ -e $A ] && { echo "$A exists: refusing"; exit 1; }
mkdir -p $C0/scripts/experiments $A
rsync -a --exclude __pycache__ $W/src $W/configs $C0/
rsync -a --exclude __pycache__ $W/scripts/rsi $C0/scripts/
for d in $DIRS; do rsync -a --exclude __pycache__ $W/scripts/experiments/$d $C0/scripts/experiments/; done
for d in assets outputs third_party grasp_kit rsi_kit; do ln -s $W/$d $C0/$d; done
date -u +%FT%TZ > $A/FROZEN_UTC
(cd $C0 && find src configs scripts -type f -not -path "*/__pycache__/*" | sort | xargs md5sum) > $A/CODE_MD5
md5sum $O/grid/grid.json $O/grid/ik.json $O/grid/cover_*.json > $A/MD5
for BE in scalebfm heft sonic; do
  md5sum $R/selfmodel_data3_20261125/$BE/models_s0/E3_ensemble_f1/model.pt $R/selfmodel_train3b_20261125/$BE/models_M_s0/E3_ensemble_f1/model.pt \
    $R/selfmodel_train3b_20261125/$BE/models_D_s0/E3_ensemble_f1/model.pt $R/selfmodel_lmlreal_20261125/$BE/data/all.npz \
    $R/selfmodel_data3_20261125/$BE/data/all.npz $R/selfmodel_data3b_20261125/$BE/data/all.npz \
    $R/selfmodel_train3b_20261125/$BE/data/mid_data3.npz $R/selfmodel_train3b_20261125/$BE/data/mid_data3b.npz >> $A/MD5
done
cat $A/FROZEN_UTC; echo "copy: $(wc -l < $A/CODE_MD5) files; inputs: $(wc -l < $A/MD5) lines"
(cd $C0 && md5sum scripts/experiments/selfmodel_freespace_20261126/*.py scripts/experiments/selfmodel_freespace_20261126/*.sh scripts/experiments/selfmodel_lmlcmp3_20261126/lc3_run.py)
