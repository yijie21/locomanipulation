#!/usr/bin/env bash
# notebook: selfmodel_simplecmp2_20261126 -- the registered evaluation after the formal run (read only, CPU):
#   scmp2_post.sh
# Writes $RUN/eval/: offline.txt (selfmodel_simplecmp's scmp_offline.py: online = offline, D's error at the task label per
# task), range.json / range.txt (scmp_range.py: sd_range's checks per reach), cmp_AJ<k>_ALJ<k>.txt (scmp_cmp.py, episode by
# episode), eval.txt / eval.json (scmp2_eval.py).
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126
O1=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
R=$O/asrun/code RUN=${RUN:-$O/run}
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
DS=$SD/data3/data/all.npz,$SD/data3b/data/all.npz,$SD/train/data/mid_data3.npz,$SD/train/data/mid_data3b.npz
X=scripts/experiments/selfmodel_simplecmp_20261126 X2=scripts/experiments/selfmodel_simplecmp2_20261126
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
cd $R && { [ -n "${SCMP2_SMOKE:-}" ] || md5sum -c --quiet CODE_MD5; } || { echo "CODE_MD5 differs"; exit 2; }
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= LOCO_REPO=$R
mkdir -p $RUN/eval
C=(); S=()
for a in ALJ1 ALJ2 RDJ1 RDJ2 RE RP J0S JDF; do [ -d $RUN/corr_$a ] && { C+=($RUN/corr_$a); S+=($a=$RUN/corr_$a); }; done
nice -n 10 $PY $X/scmp_offline.py "${C[@]}" --models $SD/train/models_D_s0 --ref $SD/real/task/data/all.npz 2>&1 | grep -v Warn > $RUN/eval/offline.txt
nice -n 10 $PY $X/scmp_range.py --train $DS --models $SD/train/models_D_s0 --out $RUN/eval/range.json "${S[@]}" 2>&1 | grep -v Warn > $RUN/eval/range.txt
for k in 1 2; do nice -n 10 $PY $X/scmp_cmp.py $RUN/rows $RUN/rows --files AJ${k}_,ALJ${k}_ > $RUN/eval/cmp_AJ${k}_ALJ${k}.txt 2>&1; done
nice -n 10 $PY $X2/scmp2_eval.py --run $RUN --cmp $O1/run --main $MAIN --range $RUN/eval/range.json --json $RUN/eval/eval.json > $RUN/eval/eval.txt 2>&1
echo "post done: $(ls $RUN/eval | tr '\n' ' ')"
