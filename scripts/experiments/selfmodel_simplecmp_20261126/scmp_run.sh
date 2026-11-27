#!/usr/bin/env bash
# notebook: selfmodel_simplecmp_20261126 -- the formal run (one step or all), from the frozen code copy:
#   scmp_run.sh <step: JL|route|AL|RD|SD|JD|all>
# ScaleBFM, perception column (sensor=rgbd), SIMPLE's four pick tasks, test episodes 0-29 (levels 0-2), simple_bench.py's A
# and J options as bench_via.sh runs them (no --film). Arms (rows -> $RUN/rows_job/ a job, merged into
# $RUN/rows/<arm>_<task>.jsonl; --correct's record -> $RUN/corr_<arm>/):
#   JL  J (--avoid <main metric via> --joint) with --correct-rule dryR, on TabletopGrasp's routed episodes (a via file):
#       D's prediction of J's joint path recorded; the rows must equal J's
#   route  scmp_route.py shifts (from JL) -> plan2.py for JD (scales 1, 2/3, 1/3) and J0 (scale 1) -> scmp_route.py pick
#   AL  A with --correct-rule dryR, all 120: D's prediction of A's commands; the rows must equal A's
#   RD  --correct-rule R (R_D), all 120;  SD  --correct-rule S (S_D), all 120
#   JD  --avoid $RUN/P/JD/via --joint, J0 --avoid $RUN/P/J0/via --joint, both with dryR (the entry sent as it is, recorded);
#       JDR (J_DR, the coordinator's addition of 11-26) JD's plans with --correct-rule R (the entry through R_D's inversion)
# Processes: SLOTS (default "2 2": two processes on GPU 2), each a (arm, task, episodes) job after another; nice 10, one
# thread, both EGL variables. The planner: PLAN_PAR (default 2) at a time on PLAN_GPU (default the first slot's GPU).
# Before every step the frozen copy is checked against asrun/code/CODE_MD5 (scmp_copy.sh asrun/code). Progress: $RUN/run.log.
# SCMP_SMOKE=<dir> (notebook: skip, the route's smoke): the development tree instead (code_dev, the dev bench's via / step1 /
# step2, SIMPLE_LEVELS 0-4, devout), SCMP_ROUTED / SCMP_EPS the episodes, rows into <dir>; no md5 check.
set -uo pipefail
STEP=$1
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
R=$O/asrun/code RUN=$O/run TREE=$O/evalout LEVELS=0,1,2
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
if [ -n "${SCMP_SMOKE:-}" ]; then
  R=$O/code_dev RUN=$SCMP_SMOKE TREE=$O/devout LEVELS=0,1,2,3,4
  MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/bench/scalebfm
fi
MODEL=$SD/train/models_D_s0/E3_ensemble_f1 TEMPLATE=$SD/real/task/data/all.npz
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE; SP=$S/.venv/bin/python
read -ra SL <<< "${SLOTS:-2 2}"
PLAN_PAR=${PLAN_PAR:-2} PLAN_GPU=${PLAN_GPU:-${SL[0]}}
cd $R || exit 2
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112 C=scripts/experiments/curobo_avoid_20261120 X=scripts/experiments/selfmodel_simplecmp_20261126
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src OMP_NUM_THREADS=1
export MUJOCO_GL=egl SIMPLE_ROOT=$S SIMPLE_EVAL_OUT=$TREE SIMPLE_LEVELS=$LEVELS
SETS="--bench-set controller=scalebfm --bench-set sensor=rgbd"
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"; CL=$SIMPLE_EVAL_OUT/$BED_CLOUD
mkdir -p $RUN/rows $RUN/rows_job $RUN/logs
log() { echo "[$(date -u +%T)] $*" >> $RUN/run.log; }
[ -n "${SCMP_SMOKE:-}" ] || md5sum -c --quiet $R/CODE_MD5 > /dev/null 2>&1 || { log "FAIL $STEP: the frozen copy differs from its CODE_MD5"; exit 2; }
TASKS="XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP"
ARM_ID_AL=1 ARM_ID_RD=2 ARM_ID_SD=3 ARM_ID_JL=4 ARM_ID_JD=5 ARM_ID_J0=6 ARM_ID_JDR=7
# one job: <arm> <task> <episodes> <gpu> <rule> [extra...]
job() {
  local arm=$1 T=$2 eps=$3 g=$4 rule=$5; shift 5
  local id=ARM_ID_$arm
  MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/simple_bench.py $T --method direct $SETS --trace \
    --episodes $eps "$@" --correct $MODEL --correct-rule $rule --correct-template $TEMPLATE --correct-out $RUN/corr_$arm \
    --correct-arm-id ${!id} --out $RUN/rows_job/${arm}_${T}__$(echo $eps | tr ',' '_' | cut -c1-40).jsonl >> $RUN/logs/${arm}_$T.log 2>&1
  log "$arm $T $eps (GPU $g): exit $?"
}
# jobs over the slots: lines "<arm> <task> <episodes> <rule> [extra...]" on stdin, round robin, each slot in turn
spread() {
  local lines=() i
  mapfile -t lines
  for i in "${!SL[@]}"; do
    ( for ((k = i; k < ${#lines[@]}; k += ${#SL[@]})); do job $(echo ${lines[$k]} | awk '{print $1, $2, $3}') ${SL[$i]} $(echo ${lines[$k]} | cut -d' ' -f4-); done ) &
  done
  wait
}
merge() {  # merge <arm>: the job files into <arm>_<task>.jsonl
  for T in $TASKS; do ls $RUN/rows_job/${1}_${T}__*.jsonl > /dev/null 2>&1 && cat $RUN/rows_job/${1}_${T}__*.jsonl > $RUN/rows/${1}_$T.jsonl; done
  log "$1: $(cat $RUN/rows_job/${1}_*__*.jsonl 2>/dev/null | wc -l) rows"
}
ROUTED=${SCMP_ROUTED:-$(ls $MAIN/via | sed -n 's/^via_TabletopGraspMP_L\([0-9]\)_e\([0-9]\)\.npz$/\1\2/p' | sed 's/^0//' | sort -n | tr '\n' ',' | sed 's/,$//')}
EPS=${SCMP_EPS:-$(seq -s, 0 29)}
halves() {  # halves <list>: the list in two (one job each)
  python3 -c "import sys; v=sys.argv[1].split(','); h=(len(v)+1)//2; print(','.join(v[:h])); print(','.join(v[h:]))" "$1"
}
if [ $STEP = JL ] || [ $STEP = all ]; then
  log "JL start: routed TabletopGrasp episodes $ROUTED"
  halves $ROUTED | sed "s|^|JL TabletopGraspMP |; s|$| dryR --avoid $MAIN/via --joint|" | spread
  merge JL
fi
if [ $STEP = route ] || [ $STEP = all ]; then
  P=$RUN/P; mkdir -p $P
  $PY -I $X/scmp_route.py shifts $P $RUN/corr_JL $MAIN > $P/logs_shifts.txt 2>&1 || { log "FAIL shifts"; exit 2; }
  log "route: $(wc -l < $P/shifts.jsonl) routed episodes with D's shift"
  plan() {  # plan <scale dir> < lines "arm task ep act"
    CUDA_VISIBLE_DEVICES=$PLAN_GPU xargs -r -P $PLAN_PAR -L1 sh -c "cd $S && nice -n 10 $SP -I $R/$C/plan2.py $S $P/\$0/step1 $P/\$0/$1 \$2 --task \$1 --cloud-dir $CL --act \$3 --ik-seed 1531 2>/dev/null | grep '^{' > $P/\$0/$1/plan_row_\$1_\$2.json"
  }
  failed() {  # failed <scale dir> <arms>: the arms' episodes with no plan there
    python3 - "$P" "$1" "$2" <<'PY'
import json, os, sys
P, d = sys.argv[1], sys.argv[2]
for l in open(os.path.join(P, "shifts.jsonl")):
    x = json.loads(l)
    for arm in sys.argv[3].split():
        f = os.path.join(P, arm, d, "plan_row_%s_%d.json" % (x["task"], x["ep"]))
        ok = os.path.exists(f) and os.path.getsize(f) > 0 and bool(json.loads(open(f).read().splitlines()[0]).get("ok"))
        if not ok:
            print(arm, x["task"], x["ep"], x["act"])
PY
  }
  python3 -c "
import json
for arm in ('JD', 'J0'):
    for l in open('$P/shifts.jsonl'):
        x = json.loads(l); print(arm, x['task'], x['ep'], x['act'])" | plan step2
  failed step2 JD > $P/retry_s2of3.txt; plan step2_s2of3 < $P/retry_s2of3.txt
  failed step2_s2of3 JD | grep -Fxf $P/retry_s2of3.txt > $P/retry_s1of3.txt; plan step2_s1of3 < $P/retry_s1of3.txt
  log "route plans: JD planned $(cat $P/JD/step2/plan_row_*.json 2>/dev/null | grep -c '"ok": true') at the full shift, $(cat $P/JD/step2_s2of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(wc -l < $P/retry_s2of3.txt) at 2/3, $(cat $P/JD/step2_s1of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(wc -l < $P/retry_s1of3.txt) at 1/3; J0 $(cat $P/J0/step2/plan_row_*.json 2>/dev/null | grep -c '"ok": true')"
  $PY -I $X/scmp_route.py pick $P $MAIN > $P/logs_pick.txt 2>&1 || { log "FAIL pick"; exit 2; }
fi
if [ $STEP = AL ] || [ $STEP = all ]; then
  for T in $TASKS; do halves $EPS | sed "s|^|AL $T |; s|$| dryR|"; done | spread
  merge AL
fi
for ARM in RD SD; do
  if [ $STEP = $ARM ] || [ $STEP = all ]; then
    rule=${ARM:0:1}
    for T in $TASKS; do halves $EPS | sed "s|^|$ARM $T |; s|$| $rule|"; done | spread
    merge $ARM
  fi
done
if [ $STEP = JD ] || [ $STEP = all ]; then
  { for ARM in JD J0; do halves $ROUTED | sed "s|^|$ARM TabletopGraspMP |; s|$| dryR --avoid $RUN/P/$ARM/via --joint|"; done
    halves $ROUTED | sed "s|^|JDR TabletopGraspMP |; s|$| R --avoid $RUN/P/JD/via --joint|"; } | spread
  merge JD; merge J0; merge JDR
fi
log "DONE $STEP"
