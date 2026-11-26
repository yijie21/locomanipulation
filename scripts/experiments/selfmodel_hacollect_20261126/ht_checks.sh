#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# Revision 2 section 1's two checks on finished ht_task runs, CPU only, read-only on the runs:
#   ht_checks.sh <out dir>[,<out dir>...] <json prefix>
# (1) ht_equiv.py in LocoManiLab's environment (lc3_formal.sh's interpreter, PYTHONPATH and lmlcmp3's frozen code copy:
#     lc3_run.py eca94321), the HA run's WIDE taken from its start line; (2) ht_feat_check.py (repo_b's data.py, the
#     runs' flags). Writes <prefix>_equiv.{json,txt} and <prefix>_feat.txt.
set -uo pipefail
RUNS=$1 P=$2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code
LC3=$C0/scripts/experiments/selfmodel_lmlcmp3_20261126
RB=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/repo_b
r0=${RUNS%%,*}
W=$(grep -a -h -o "WIDE \[\[[^]]*\], \[[^]]*\]\]" $r0.log | head -1 | cut -c6-)
FD=$(grep -a -h -o "'FDVEL': [A-Za-z]*" $r0.log | head -1 | grep -c True)
echo "lc3_run $(md5sum < $LC3/lc3_run.py | cut -c1-8); HA WIDE $W; runs' FDVEL $FD"
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1
(cd $G && PYTHONPATH=$G:$C0/src REACHWM_FDVEL=0 nice -n 19 $PY $HERE/ht_equiv.py --lc3 $LC3 --runs $RUNS --wide "$W" --json ${P}_equiv.json) \
  2>&1 | grep -v Warn > ${P}_equiv.txt
tail -2 ${P}_equiv.txt
(cd $RB && REACHWM_FDVEL=$FD LOCO_REPO=$RB nice -n 19 $PY $HERE/ht_feat_check.py $RUNS) 2>&1 | grep -v Warn > ${P}_feat.txt
tail -2 ${P}_feat.txt
