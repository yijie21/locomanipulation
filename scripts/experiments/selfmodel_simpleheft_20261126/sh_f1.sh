#!/usr/bin/env bash
# notebook: selfmodel_simpleheft_20261126 -- F1 on R_D's commands after phase R, read only, CPU, no simulation:
#   sh_f1.sh [run dir, default $O/run] [out dir, default $O/f1]
# D_heft's prediction error on AL's and RD's records (sh_err.py, one arm a file: err_AL.json, err_RD.json) and F1-0 on
# their starts (ss_f1pre.py reads <dir>/A_<task>.jsonl: links to AL_ / RD_ rows; f1pre_AL.json, f1pre_RD.json). sh_eval.py
# --f1 <out dir> reads them. Output: <out>/f1.txt.
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
RUN=${1:-$O/run} F=${2:-$O/f1}
R=${SH_CODE:-$O/asrun/code}
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
X=$R/scripts/experiments
mkdir -p $F/rows_AL $F/rows_RD
cd $R || exit 2
[ -n "${SH_CODE:-}" ] || md5sum -c --quiet CODE_MD5 || { echo "CODE_MD5 differs"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= LOCO_REPO=$R REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=0
T3=$L/selfmodel_train3b_20261125/heft
DSPEC=$L/selfmodel_data3_20261125/heft/data/all.npz,$L/selfmodel_data3b_20261125/heft/data/all.npz,$T3/data/mid_data3.npz,$T3/data/mid_data3b.npz
for a in AL RD; do
  for T in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do ln -sfn $RUN/rows/${a}_$T.jsonl $F/rows_$a/A_$T.jsonl; done
done
{
  echo "F1 $(date -u +%FT%TZ): rows $RUN/rows, records $RUN/corr_AL, $RUN/corr_RD"
  for a in AL RD; do
    nice -n 10 $PY $X/selfmodel_simplesonic_20261126/ss_f1pre.py --label D_heft_$a --n-train 499077 --rows $F/rows_$a \
      --json $F/f1pre_$a.json --train $DSPEC 2>&1 | grep -v Warn
    nice -n 10 $PY $X/selfmodel_simpleheft_20261126/sh_err.py $a=$RUN/corr_$a --models $T3/models_D_s0 --val 3.204 \
      --label D_heft_$a --rows $RUN/rows --json $F/err_$a.json 2>&1 | grep -v Warn
  done
} 2>&1 | tee $F/f1.txt
