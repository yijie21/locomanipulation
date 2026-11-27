#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Revision 2a addendum's freeze for HEFT's formal episodes on gait_heft: asrun_real (revision 2a's freeze; verified
# first) copied as it is (repo/assets and repo/third_party stay links; no __pycache__), hr_real.py replaced by the
# addendum's (--heft-gait), plus hr_formal_seq.sh / hr_gait_smoke.sh / hr_gait_cmp.py / hr_freeze2.sh, and lmlcmp's heft_gait copies
# (gait.py, gait_heft.py, byte-identical to src/locomani/backends/heft/) where hr_real's HEFT_GAIT_DIR points
# (<asrun_real2>/selfmodel_lmlcmp_20261126/heft_gait). MD5.check covers every copied file plus asrun_real's MD5.check
# lines outside the copy (the collection's pinned installs, the HA task files); hr_formal_seq.sh verifies it before each
# HEFT process and starts HEFT only once <asrun_real2>/GO exists (written by hand after the addendum's commit).
#   hr_freeze2.sh <asrun_real> <new asrun_real2> <hr_real md5>      (once; refuses when <asrun_real2>/MD5 exists)
set -euo pipefail
S=$1 A=$2 WANT=$3
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
PC=/wuji-vefps-D/wuji-il/yijie/tools/perception
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
[ "$(md5sum < $E/selfmodel_hacollect_20261126/hr_real.py | cut -d' ' -f1)" = "$WANT" ] || { echo "hr_real.py is not $WANT"; exit 2; }
(cd $S && md5sum -c --quiet MD5.check) || { echo "asrun_real does not verify"; exit 2; }
mkdir -p $A
rsync -a --exclude MD5 --exclude MD5.check --exclude FROZEN_UTC --exclude __pycache__ $S/ $A/
cp $E/selfmodel_hacollect_20261126/{hr_real.py,hr_formal_seq.sh,hr_gait_smoke.sh,hr_gait_cmp.py,hr_freeze2.sh} $A/selfmodel_hacollect_20261126/
mkdir -p $A/selfmodel_lmlcmp_20261126/heft_gait
cp $E/selfmodel_lmlcmp_20261126/heft_gait/{gait.py,gait_heft.py} $A/selfmodel_lmlcmp_20261126/heft_gait/
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -name GO ! -path '*/__pycache__/*' \
  ! -path './repo/assets/*' ! -path './repo/third_party/*' | sort | xargs md5sum > MD5.check
grep -E '  /' $S/MD5.check >> MD5.check
{ cat MD5.check; echo "# copied from asrun_real $S (FROZEN_UTC $(cat $S/FROZEN_UTC)), addendum files from $W at $(cd $W && git rev-parse HEAD)";
  (cd $W && git status --short scripts/experiments/selfmodel_hacollect_20261126 scripts/experiments/selfmodel_lmlcmp_20261126/heft_gait);
  echo "# SAM 3 service (not checked): $(md5sum $PC/start_sam3.sh) / $(md5sum $W/scripts/rsi/sam3_server.py) / sam3 $(cd $PC/sam3 && git rev-parse HEAD)";
  echo "# $(tmux ls 2>/dev/null | grep '^sam3:' || echo 'no sam3 tmux')"; } > MD5
date -u +%FT%TZ | tee FROZEN_UTC
md5sum $A/MD5 $A/MD5.check $A/selfmodel_hacollect_20261126/{hr_real.py,hr_formal_seq.sh} $A/selfmodel_lmlcmp_20261126/heft_gait/*.py
