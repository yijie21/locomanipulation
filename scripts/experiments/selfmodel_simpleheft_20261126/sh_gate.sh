#!/usr/bin/env bash
# notebook: selfmodel_simpleheft_20261126 -- the gate after phase A, read only, CPU, no simulation:
#   sh_gate.sh [run dir, default $O/run] [out dir, default $O/gate]
# F1-0 (selfmodel_simplesonic_20261126's ss_f1pre.py, its rule unchanged): the A rows' pregrasp and entry starts (S0, the
# 29 joints at the command) against the model's train split, per joint 0.5 / 99.5 %; F1(a) (sh_err.py): the model's
# prediction error on AL's recorded reaches at the task label over its own val median. D_heft decides (the registered
# model); F_heft (selfmodel_fdvel_20261126's HEFT model, FDVEL on, data3 only) is printed beside it as a reference only
# (its "online = offline" line compares against D_heft's online record: not a check). D_heft's F1(a) lines are repeated
# over the reaches without a table contact (sh_err.py --rows; a description, not part of the gate).
# Then the verdict per task: PASS when D_heft's F1-0 share (both stages) >= 80 % and its F1(a) ratio (both stages) <= 1.3;
# "GATE PASS" when at least one task passes, else "GATE FAIL". Output: <out>/gate.txt (+ the json files).
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
RUN=${1:-$O/run} G=${2:-$O/gate}
R=${SH_CODE:-$O/asrun/code}
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
X=$R/scripts/experiments
mkdir -p $G
cd $R || exit 2
[ -n "${SH_CODE:-}" ] || md5sum -c --quiet CODE_MD5 || { echo "CODE_MD5 differs"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= LOCO_REPO=$R REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1
T3=$L/selfmodel_train3b_20261125/heft
DSPEC=$L/selfmodel_data3_20261125/heft/data/all.npz,$L/selfmodel_data3b_20261125/heft/data/all.npz,$T3/data/mid_data3.npz,$T3/data/mid_data3b.npz
FSPEC=$L/selfmodel_data3_20261125/heft/data/all.npz
{
  echo "gate $(date -u +%FT%TZ): rows $RUN/rows, AL record $RUN/corr_AL"
  REACHWM_FDVEL=0 nice -n 10 $PY $X/selfmodel_simplesonic_20261126/ss_f1pre.py --label D_heft --n-train 499077 --rows $RUN/rows \
    --json $G/f1pre_D_heft.json --train $DSPEC 2>&1 | grep -v Warn
  REACHWM_FDVEL=1 nice -n 10 $PY $X/selfmodel_simplesonic_20261126/ss_f1pre.py --label F_heft_reference --n-train 20518 --rows $RUN/rows \
    --json $G/f1pre_F_heft.json --train $FSPEC 2>&1 | grep -v Warn
  REACHWM_FDVEL=0 nice -n 10 $PY $X/selfmodel_simpleheft_20261126/sh_err.py AL=$RUN/corr_AL --models $T3/models_D_s0 --val 3.204 \
    --label D_heft --rows $RUN/rows --json $G/err_D_heft.json 2>&1 | grep -v Warn
  REACHWM_FDVEL=1 nice -n 10 $PY $X/selfmodel_simpleheft_20261126/sh_err.py AL=$RUN/corr_AL --models $L/selfmodel_fdvel_20261126/heft/models_s0 \
    --val 4.598 --label F_heft_reference --json $G/err_F_heft.json 2>&1 | grep -v Warn
  python3 - $G <<'EOF'
import json, sys
G = sys.argv[1]
f0 = json.load(open(f"{G}/f1pre_D_heft.json"))["cells"]
fa = json.load(open(f"{G}/err_D_heft.json"))["sets"]["AL"]["rows"]
ok = []
print("verdict (D_heft; F1-0 share >= 80 % and F1(a) ratio <= 1.3, both stages):")
for t in ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"):
    a, b = f0.get(f"{t}/both"), fa.get(f"{t}/both")
    s = a["share"] if a else float("nan")
    r = b["ratio"] if b else float("nan")
    p = bool(a and b and s >= 0.80 and r <= 1.3)
    ok += [t] if p else []
    print(f"  {t:20s} F1-0 {100 * s:5.1f} %  F1(a) {r:.3f}x  -> {'PASS' if p else 'FAIL'}")
print(f"GATE PASS ({', '.join(ok)})" if ok else "GATE FAIL")
EOF
} 2>&1 | tee $G/gate.txt
