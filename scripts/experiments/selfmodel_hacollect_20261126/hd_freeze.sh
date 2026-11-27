#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Freeze everything the formal HA collections run into <asrun>/ and run from there (hd_job.sh), as
# selfmodel_xfer_20261125/sx_freeze_ha.sh did for xfer: this directory's collector (hc_ha.py / hc_ha.sh) and tools; xfer's
# HA host and record (sx_ha.py, sx_collect.py, sx_eval.py); data3's collector with data3b's functions (d3_collect.py,
# d3_reach_table.json) and its LocoManiLab / hardgrasp imports; humanoidarena_20261124's ha_sim.py; ha_ctrl's
# ha_g1ctrl.py; data3b's b_config.json per controller; from the live repo src/, scripts/experiments/{selfwm_20261029,
# reachwm_20261030,locomanilab_20261117} and the two scenes. assets/ and third_party/ stay links to the live repo.
# HumanoidArena, SONIC's ONNX and g1_backend are installs outside the repo: the files the runs read are pinned by md5 in
# MD5.check, which hd_job.sh verifies (with every copied file) before each process starts; so are fdvel's frozen FDVEL
# data.py and its env.py, the reader hd_finish.sh checks the data with.
#   hd_freeze.sh <asrun dir>      (once; refuses when <asrun>/MD5 exists)
set -euo pipefail
A=$1
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
O3B=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3b_20261125
HR=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
IL=$HR/HumanoidArena/isaaclab_twist2_g1
SP=$HR/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
G1B=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
FDR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_fdvel_20261126/asrun/repo/scripts/experiments  # the check's reader (hd_finish.sh)
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
mkdir -p $A/selfmodel_hacollect_20261126 $A/selfmodel_xfer_20261125 $A/selfmodel_data3_20261125 $A/locomanilab_20261117 \
  $A/hardgrasp_20261125 $A/humanoidarena_20261124 $A/ha_ctrl $A/b_config $A/repo/scripts/experiments $A/repo/outputs/scene
cp $E/selfmodel_hacollect_20261126/{hc_ha.py,hc_ha.sh,hc_rates.py,hc_start.py,hd_freeze.sh,hd_job.sh,hd_formal.sh,hd_finish.sh,hd_same.py} \
  $A/selfmodel_hacollect_20261126/
cp $E/selfmodel_xfer_20261125/{sx_ha.py,sx_collect.py,sx_eval.py} $A/selfmodel_xfer_20261125/
cp $E/selfmodel_data3_20261125/{d3_collect.py,d3_reach_table.json} $A/selfmodel_data3_20261125/
cp $E/locomanilab_20261117/{lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $E/hardgrasp_20261125/hg_walk.py $A/hardgrasp_20261125/
cp $E/humanoidarena_20261124/ha_sim.py $A/humanoidarena_20261124/
cp $HR/scripts/ha_ctrl/ha_g1ctrl.py $A/ha_ctrl/
for c in scalebfm heft sonic; do cp $O3B/$c/b_config.json $A/b_config/$c.json; done
X=(--exclude __pycache__ --exclude .pytest_cache --exclude '*.pyc')
rsync -a "${X[@]}" $W/src $A/repo/
for e in selfwm_20261029 reachwm_20261030 locomanilab_20261117; do rsync -a "${X[@]}" $E/$e $A/repo/scripts/experiments/; done
for s in simple_g1_sonic walk_mars; do rsync -aL $W/outputs/scene/$s $A/repo/outputs/scene/; done
ln -s $W/assets $A/repo/assets
ln -s $W/third_party $A/repo/third_party
touch $A/repo/FULL
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -path './repo/assets/*' ! -path './repo/third_party/*' \
  | sort | xargs md5sum > MD5.check
md5sum $HR/env.sh $IL/action_provider/{action_provider_sonic.py,vla_robot_current_local_runtime_v3.py,vla_smpl_runtime.py} \
  $IL/script/eval_scripts/sonic/sim_eval_vla.py $IL/tasks/common_test_config/base_test/pp_box_sonic_test.yaml \
  $IL/tasks/g1_tasks/move_pickplace_box_g1_29dof_dex3_wholedoby/move_pickplace_box_g1_29dof_dex3_hw_env_cfg.py \
  $IL/tasks/common_config/robot_configs.py $IL/tasks/common_scene/base_scene_pickplace_box.py \
  $SP/model_encoder.onnx $SP/model_decoder.onnx \
  $G1B/checkpoints/scalebfm/humanoid_transformer_m/{policy.onnx,policy.yaml} $G1B/checkpoints/heft/pmg/{policy.onnx,policy.yaml} \
  $FDR/reachwm_20261030/data.py $FDR/selfwm_20261029/env.py \
  >> MD5.check
{ cat MD5.check; echo "# repo copy from $W at $(cd $W && git rev-parse HEAD) (status of the copied paths below)";
  (cd $W && git status --short src scripts/experiments/selfwm_20261029 scripts/experiments/reachwm_20261030 \
     scripts/experiments/locomanilab_20261117 scripts/experiments/humanoidarena_20261124 scripts/experiments/selfmodel_xfer_20261125 \
     scripts/experiments/selfmodel_data3_20261125 scripts/experiments/selfmodel_hacollect_20261126 scripts/experiments/hardgrasp_20261125);
  echo "# g1_backend $(cd $G1B && git rev-parse HEAD)"; } > MD5
date -u +%FT%TZ | tee FROZEN_UTC
md5sum $A/selfmodel_hacollect_20261126/* $A/selfmodel_xfer_20261125/* $A/selfmodel_data3_20261125/* $A/humanoidarena_20261124/ha_sim.py \
  $A/ha_ctrl/ha_g1ctrl.py $A/b_config/* $A/locomanilab_20261117/* $A/hardgrasp_20261125/hg_walk.py
