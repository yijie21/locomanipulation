#!/usr/bin/env bash
# selfmodel_fdvel_20261126: freeze everything that runs into $A (default $O/asrun) and run from $A/repo (fd_run.sh), as
# selfmodel_xfer_20261125's sx_freeze_ha.sh does: from the live 4090 repo src/ and scripts/experiments/{selfwm_20261029,
# reachwm_20261030, locomanilab_20261117}; data.py replaced by the Mac's REACHWM_FDVEL version (staged in $ST); train3's
# frozen d3_eval.py; xfer's frozen sx_eval.py; this experiment's scripts (staged). assets/ and third_party/ stay links
# to the live repo. train.py, models.py and env.py must equal train3's frozen copies, wm.py and fk.py train3b's.
# Writes $A/MD5.check (every file; fd_run.sh verifies it before it starts), $A/MD5 (+ git revs and status, the
# interpreter), $A/data_py.diff (against train3's data.py) and $A/FROZEN_UTC.
#   fd_freeze.sh [dest]        (refuses when dest/MD5 exists)
set -euo pipefail
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
O=$R/selfmodel_fdvel_20261126
ST=${FD_STAGE:-$O/stage}
A=${1:-$O/asrun}
T3A=$R/selfmodel_data3_20261125/scalebfm/asrun/train
T3B=$R/selfmodel_train3b_20261125/asrun
XA=$R/selfmodel_xfer_20261125/asrun_ha
[ -e $A/MD5 ] && { echo "already frozen: $A/MD5"; exit 1; }
E=$A/repo/scripts/experiments
mkdir -p $E/selfmodel_data3_20261125 $E/selfmodel_xfer_20261125 $E/selfmodel_fdvel_20261126
X=(--exclude __pycache__ --exclude .pytest_cache --exclude '*.pyc')
rsync -a "${X[@]}" $W/src $A/repo/
for e in selfwm_20261029 reachwm_20261030 locomanilab_20261117; do rsync -a "${X[@]}" $W/scripts/experiments/$e $E/; done
cp $ST/reachwm_20261030/data.py $E/reachwm_20261030/data.py
cp $T3A/d3_eval.py $E/selfmodel_data3_20261125/
cp $XA/selfmodel_xfer_20261125/sx_eval.py $E/selfmodel_xfer_20261125/
cp $ST/selfmodel_fdvel_20261126/*.py $ST/selfmodel_fdvel_20261126/*.sh $E/selfmodel_fdvel_20261126/
ln -s $W/assets $A/repo/assets
ln -s $W/third_party $A/repo/third_party
for f in train.py models.py; do cmp $E/reachwm_20261030/$f $T3A/$f; done
cmp $E/selfwm_20261029/env.py $T3A/env.py
for f in wm.py fk.py; do cmp $E/selfwm_20261029/$f $T3B/selfwm_20261029/$f; done
cmp $E/selfmodel_data3_20261125/d3_eval.py $T3A/d3_eval.py
cmp $E/selfmodel_xfer_20261125/sx_eval.py $R/selfmodel_xfer_20261125/asrun/selfmodel_xfer_20261125/sx_eval.py
diff $T3A/data.py $E/reachwm_20261030/data.py > $A/data_py.diff || true
cd $A
find . -type f ! -name MD5 ! -name MD5.check ! -name FROZEN_UTC ! -path './repo/assets/*' ! -path './repo/third_party/*' \
  | sort | xargs md5sum > MD5.check
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
{ cat MD5.check
  echo "# repo copy from $W at $(cd $W && git rev-parse HEAD) (status of the copied paths below)"
  (cd $W && git status --short src scripts/experiments/selfwm_20261029 scripts/experiments/reachwm_20261030 \
     scripts/experiments/locomanilab_20261117)
  echo "# staged from the Mac:"; cat $ST/MAC_GIT
  echo "# interpreter $VPY -> $(readlink $VPY): $($VPY -c 'import sys, torch, numpy; print(sys.prefix, sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__)')"
} > MD5
date -u +%FT%TZ | tee FROZEN_UTC
grep -E 'reachwm_20261030/(data|train|models)\.py|selfwm_20261029/(env|wm|fk)\.py|d3_eval\.py|sx_eval\.py|selfmodel_fdvel_20261126/' MD5.check
