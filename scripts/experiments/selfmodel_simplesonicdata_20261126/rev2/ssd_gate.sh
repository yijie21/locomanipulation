#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- the gate after phase A (the same-day check of D_S on the real SIMPLE
# REVISION 2 (2026-11-27, after the first GATE FAIL, before any R_D episode): the frozen ssd_gate.sh (55ca61a9) run from the frozen
# copy $O/asrun/code2 (= asrun/code with one line of reachwm_20261030/data.py changed: x0 column 204, walked, set to 0); its default out dir is $O/gate_rev2 (gate/ keeps the first verdict).
# Nothing else changed.
# episodes), read only, no simulation:  ssd_gate.sh [run dir, default $O/run] [out dir, default $O/gate]
# F1-0 (selfmodel_simplesonic_20261126's ss_f1pre.py, unchanged): A's pregrasp and entry starts (S0, the 29 joints at the
# command) against D_S's train split, per joint 0.5 / 99.5 %. F1(a) (selfmodel_simpleheft_20261126's sh_err.py,
# unchanged): D_S's prediction error on AL's recorded reaches at the task label over D_S's own val median (its
# train.json best_val_median_cm). Beside them, not in the verdict: F1(b) on AL's reaches (selfmodel_simplecmp_20261126's
# scmp_range.py: sd_range's three checks, 39 quantities, nearest neighbour, table; D_S has FDVEL, so it runs) and the
# F1(a) lines without a table contact (sh_err.py --rows). Verdict per task: PASS when F1-0 (both stages) >= 80 % and F1(a)
# (both stages) <= 1.3; "GATE PASS" when at least one task passes, else "GATE FAIL". Output: <out>/gate.txt (+ json).
set -uo pipefail
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
O=$L/selfmodel_simplesonicdata_20261126
RUN=${1:-$O/run} G=${2:-$O/gate_rev2}
R=${SSD_CODE:-$O/asrun/code2}
MD=${SSD_MODELS:-$O/train/models_D_s0}
DS=${SSD_DSPEC:-$(cat $O/train/DSPEC)}
X=$R/scripts/experiments
PY="/usr/bin/python3.10 -S $X/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $G
cd $R || exit 2
[ -n "${SSD_CODE:-}" ] || md5sum -c --quiet CODE_MD5 || { echo "CODE_MD5 differs"; exit 2; }
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  CUDA_VISIBLE_DEVICES= LOCO_REPO=$R REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1
unset REACHWM_X0_REF
read -r NTR VAL <<< "$(python3 -c "import json; d=json.load(open('$MD/E3_ensemble_f1/train.json')); print(d['n_train'], round(d['best_val_median_cm'], 4))")"
{
  echo "gate $(date -u +%FT%TZ): rows $RUN/rows, AL record $RUN/corr_AL; D_S $MD/E3_ensemble_f1 (n_train $NTR, val median $VAL cm); train $DS"
  nice -n 10 $PY $X/selfmodel_simplesonic_20261126/ss_f1pre.py --label D_S --n-train $NTR --rows $RUN/rows \
    --json $G/f1pre_D_S.json --train $DS 2>&1 | grep -v Warn
  nice -n 10 $PY $X/selfmodel_simpleheft_20261126/sh_err.py AL=$RUN/corr_AL --models $MD --val $VAL \
    --label D_S --rows $RUN/rows --json $G/err_D_S.json 2>&1 | grep -v Warn
  echo "== F1(b) beside the gate (scmp_range.py, AL's reaches)"
  nice -n 10 $PY $X/selfmodel_simplecmp_20261126/scmp_range.py --train $DS --models $MD --out $G/range_AL.json AL=$RUN/corr_AL 2>&1 | grep -v Warn
  python3 - $G <<'EOF'
import json, sys
G = sys.argv[1]
f0 = json.load(open(f"{G}/f1pre_D_S.json"))["cells"]
fa = json.load(open(f"{G}/err_D_S.json"))["sets"]["AL"]["rows"]
rg = [x for x in json.load(open(f"{G}/range_AL.json"))["sets"]["AL"] if x["kept"]]
ok = []
print("verdict (D_S; F1-0 share >= 80 % and F1(a) ratio <= 1.3, both stages; F1(b) beside, not in the verdict):")
for t in ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"):
    a, b = f0.get(f"{t}/both"), fa.get(f"{t}/both")
    s = a["share"] if a else float("nan")
    r = b["ratio"] if b else float("nan")
    S = [x["in_range"] for x in rg if x["task"] == t]
    fb = sum(S) / len(S) if S else float("nan")
    p = bool(a and b and s >= 0.80 and r <= 1.3)
    ok += [t] if p else []
    print(f"  {t:20s} F1-0 {100 * s:5.1f} %  F1(a) {r:.3f}x  -> {'PASS' if p else 'FAIL'}   (F1(b) {100 * fb:5.1f} % of {len(S)})")
print(f"GATE PASS ({', '.join(ok)})" if ok else "GATE FAIL")
EOF
} 2>&1 | tee $G/gate.txt
