#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- F1 on R_D's commands after phase R, read only, no simulation:
# REVISION 2 (2026-11-27, after the first GATE FAIL, before any R_D episode): the frozen ssd_f1.sh (6902ad00) run from the frozen
# copy $O/asrun/code2 (= asrun/code with one line of reachwm_20261030/data.py changed: x0 column 204, walked, set to 0).
# Nothing else changed.
#   ssd_f1.sh [run dir, default $O/run] [out dir, default $O/f1]
# For AL and RD: F1-0 on their starts (ss_f1pre.py reads <dir>/A_<task>.jsonl: links to the arm's rows; f1pre_<arm>.json),
# D_S's error (sh_err.py; err_<arm>.json) and sd_range's three checks per reach (scmp_range.py; range.json, both arms).
# ssd_eval.py --f1 <out dir> reads them. Output: <out>/f1.txt.
set -uo pipefail
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
O=$L/selfmodel_simplesonicdata_20261126
RUN=${1:-$O/run} F=${2:-$O/f1}
R=${SSD_CODE:-$O/asrun/code2}
MD=${SSD_MODELS:-$O/train/models_D_s0}
DS=${SSD_DSPEC:-$(cat $O/train/DSPEC)}
X=$R/scripts/experiments
PY="/usr/bin/python3.10 -S $X/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $F/rows_AL $F/rows_RD
cd $R || exit 2
[ -n "${SSD_CODE:-}" ] || md5sum -c --quiet CODE_MD5 || { echo "CODE_MD5 differs"; exit 2; }
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  CUDA_VISIBLE_DEVICES= LOCO_REPO=$R REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1
unset REACHWM_X0_REF
read -r NTR VAL <<< "$(python3 -c "import json; d=json.load(open('$MD/E3_ensemble_f1/train.json')); print(d['n_train'], round(d['best_val_median_cm'], 4))")"
for a in AL RD; do
  for T in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do ln -sfn $RUN/rows/${a}_$T.jsonl $F/rows_$a/A_$T.jsonl; done
done
{
  echo "F1 $(date -u +%FT%TZ): rows $RUN/rows, records $RUN/corr_AL, $RUN/corr_RD; D_S val $VAL cm, n_train $NTR"
  for a in AL RD; do
    nice -n 10 $PY $X/selfmodel_simplesonic_20261126/ss_f1pre.py --label D_S_$a --n-train $NTR --rows $F/rows_$a \
      --json $F/f1pre_$a.json --train $DS 2>&1 | grep -v Warn
    nice -n 10 $PY $X/selfmodel_simpleheft_20261126/sh_err.py $a=$RUN/corr_$a --models $MD --val $VAL \
      --label D_S_$a --rows $RUN/rows --json $F/err_$a.json 2>&1 | grep -v Warn
  done
  nice -n 10 $PY $X/selfmodel_simplecmp_20261126/scmp_range.py --train $DS --models $MD --out $F/range.json AL=$RUN/corr_AL RD=$RUN/corr_RD 2>&1 | grep -v Warn
} 2>&1 | tee $F/f1.txt
