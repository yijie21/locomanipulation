#!/usr/bin/env bash
# selfmodel_xfer_20261125, HumanoidArena combinations (revision 2: 3 SONIC, 4 ScaleBFM, 5 HEFT): freeze everything that
# runs into asrun_ha/, run from there (sx_ha_batch.sh): this experiment's scripts; data3's frozen collector (as
# sx_freeze.sh); humanoidarena_20261124's ha_sim.py (HA's sim_eval_vla start-up); ha_controllers_20261125's ha_g1ctrl.py
# (the ScaleBFM / HEFT provider); from the live repo: src/, scripts/experiments/{selfwm_20261029,reachwm_20261030,
# locomanilab_20261117}, the two scenes (SIMPLE's G1 + Dex3 is the arm record's mirror). assets/ and third_party/ stay
# links to the live repo. HumanoidArena itself, SONIC's ONNX and g1_backend are installs outside the repo: the files the
# runs read are listed with their md5 in MD5.check, which sx_ha_batch.sh verifies (with every copied file) before it starts.
#   sx_freeze_ha.sh            (once; refuses when asrun_ha/MD5 exists)
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125
D3A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/asrun
HR=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
IL=$HR/HumanoidArena/isaaclab_twist2_g1
SP=$HR/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
G1B=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
A=${SX_ASRUN:-$O/asrun_ha}  # (a dev freeze elsewhere for a pipeline test)
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
set -e
mkdir -p $A/selfmodel_xfer_20261125 $A/selfmodel_data3_20261125 $A/locomanilab_20261117 $A/hardgrasp_20261125 \
  $A/humanoidarena_20261124 $A/ha_ctrl $A/repo/scripts/experiments $A/repo/outputs/scene
cp $W/scripts/experiments/selfmodel_xfer_20261125/{sx_collect.py,sx_eval.py,sx_speed.py,sx_ha.py,sx_ha.sh,sx_ha_batch.sh,sx_freeze_ha.sh} \
  $A/selfmodel_xfer_20261125/
cp $D3A/selfmodel_data3_20261125/{d3_collect.py,d3_reach_table.json} $A/selfmodel_data3_20261125/
cp $D3A/locomanilab_20261117/{lml_selfmodel.py,lml_host.py} $A/locomanilab_20261117/
cp $D3A/hardgrasp_20261125/hg_walk.py $A/hardgrasp_20261125/
cp $W/scripts/experiments/humanoidarena_20261124/ha_sim.py $A/humanoidarena_20261124/
cp $HR/scripts/ha_ctrl/ha_g1ctrl.py $A/ha_ctrl/
X=(--exclude __pycache__ --exclude .pytest_cache --exclude '*.pyc')
rsync -a "${X[@]}" $W/src $A/repo/
for e in selfwm_20261029 reachwm_20261030 locomanilab_20261117; do rsync -a "${X[@]}" $W/scripts/experiments/$e $A/repo/scripts/experiments/; done
for s in simple_g1_sonic walk_mars; do rsync -aL $W/outputs/scene/$s $A/repo/outputs/scene/; done
ln -s $W/assets $A/repo/assets
ln -s $W/third_party $A/repo/third_party
touch $A/repo/FULL
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -path './repo/assets/*' ! -path './repo/third_party/*' \
  | sort | xargs md5sum > MD5.check
md5sum $IL/action_provider/{action_provider_sonic.py,vla_robot_current_local_runtime_v3.py,vla_smpl_runtime.py} \
  $IL/script/eval_scripts/sonic/sim_eval_vla.py $IL/tasks/common_test_config/base_test/pp_box_sonic_test.yaml \
  $IL/tasks/g1_tasks/move_pickplace_box_g1_29dof_dex3_wholedoby/move_pickplace_box_g1_29dof_dex3_hw_env_cfg.py \
  $IL/tasks/common_config/robot_configs.py $IL/tasks/common_scene/base_scene_pickplace_box.py \
  $SP/model_encoder.onnx $SP/model_decoder.onnx \
  $G1B/checkpoints/scalebfm/humanoid_transformer_m/{policy.onnx,policy.yaml} $G1B/checkpoints/heft/pmg/{policy.onnx,policy.yaml} \
  >> MD5.check
{ cat MD5.check; echo "# repo copy from $W at $(cd $W && git rev-parse HEAD) (status of the copied paths below)";
  (cd $W && git status --short src scripts/experiments/selfwm_20261029 scripts/experiments/reachwm_20261030 \
     scripts/experiments/locomanilab_20261117 scripts/experiments/humanoidarena_20261124 scripts/experiments/selfmodel_xfer_20261125);
  echo "# g1_backend $(cd $G1B && git rev-parse HEAD)"; } > MD5
date -u +%FT%TZ | tee FROZEN_UTC
md5sum $A/selfmodel_xfer_20261125/* $A/selfmodel_data3_20261125/d3_collect.py $A/humanoidarena_20261124/ha_sim.py $A/ha_ctrl/ha_g1ctrl.py
