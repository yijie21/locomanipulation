#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Freeze for the real HA episodes (revision 2a): the collection's frozen copy (asrun, hd_freeze.sh; verified first)
# copied as it is (assets/ and third_party/ stay links), plus this directory's real-episode files (hr_*: hr_video.py only
# because hr_real imports it under --hrvid; hr_same.py, hr_chain.sh, hr_replay.sh are the checks before the freeze), and
# humanoidarena_20261124's ha_bench.py / perceive.py / g1arm.py with g1arm's URDF. MD5.check covers every copied file,
# the collection's pinned installs (its MD5.check lines outside the copy) and the doubledesk / opendoor HA task files;
# hr_formal.sh verifies it before each process. The SAM 3 service (tmux sam3 since 2026-11-24, perceive.Sam's address)
# is a running process: its launcher, server script and sam3 checkout are recorded in MD5, not checked.
#   hr_freeze.sh <collection asrun> <new asrun_real>      (once; refuses when <asrun_real>/MD5 exists)
set -euo pipefail
S=$1 A=$2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
HR=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
IL=$HR/HumanoidArena/isaaclab_twist2_g1
PC=/wuji-vefps-D/wuji-il/yijie/tools/perception
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
(cd $S && md5sum -c --quiet MD5.check) || { echo "the collection's frozen copy does not verify"; exit 2; }
mkdir -p $A
rsync -a --exclude MD5 --exclude MD5.check --exclude FROZEN_UTC $S/ $A/
cp $E/selfmodel_hacollect_20261126/{hr_real.py,hr_video.py,hr_same.py,hr_smoke_stats.py,hr_train_range.py,hr_yaw.py,hr_chain.sh,hr_replay.sh,hr_freeze.sh,hr_formal.sh} $A/selfmodel_hacollect_20261126/
cp $E/humanoidarena_20261124/{ha_bench.py,perceive.py,g1arm.py} $A/humanoidarena_20261124/
mkdir -p $A/humanoidarena_20261124/curobo
cp $E/humanoidarena_20261124/curobo/g1_29dof_with_dex3_ha.urdf $A/humanoidarena_20261124/curobo/
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -path './repo/assets/*' ! -path './repo/third_party/*' \
  | sort | xargs md5sum > MD5.check
grep -E '  /' $S/MD5.check >> MD5.check  # the collection's pinned installs (absolute paths)
md5sum $IL/tasks/common_test_config/base_test/{doubledesk_sonic_test.yaml,open_door_sonic_test.yaml} \
  $IL/tasks/g1_tasks/move_pickplace_doubledesk_g1_29dof_dex3_wholebody/*.py \
  $IL/tasks/g1_tasks/move_open_door_g1_29dof_dex3_wholebody/move_open_door_g1_29dof_dex3_hw_env_cfg.py >> MD5.check
{ cat MD5.check; echo "# copied from the collection freeze $S (FROZEN_UTC $(cat $S/FROZEN_UTC)), real-episode files from $W at $(cd $W && git rev-parse HEAD)";
  (cd $W && git status --short scripts/experiments/selfmodel_hacollect_20261126 scripts/experiments/humanoidarena_20261124);
  echo "# SAM 3 service (not checked): $(md5sum $PC/start_sam3.sh) / $(md5sum $W/scripts/rsi/sam3_server.py) / sam3 $(cd $PC/sam3 && git rev-parse HEAD)";
  echo "# $(tmux ls 2>/dev/null | grep '^sam3:' || echo 'no sam3 tmux')"; } > MD5
date -u +%FT%TZ | tee FROZEN_UTC
md5sum $A/selfmodel_hacollect_20261126/hr_* $A/humanoidarena_20261124/{ha_bench.py,perceive.py,g1arm.py,curobo/g1_29dof_with_dex3_ha.urdf}
