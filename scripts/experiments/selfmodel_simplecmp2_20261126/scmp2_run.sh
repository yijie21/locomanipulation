#!/usr/bin/env bash
# notebook: selfmodel_simplecmp2_20261126 -- the formal run from the frozen copy (one part, or all in one queue):
#   scmp2_run.sh <part: E1|E2|V1|all>
# ScaleBFM, perception column (sensor=rgbd), simple_bench.py's A options as bench_via.sh runs them (no --film), the main
# metric's test episodes 0-29 (levels 0-2, the same episode numbers and seeds). Arms (rows -> $RUN/rows_job/ a job, merged
# into $RUN/rows/<arm>_<task>.jsonl; --correct's record -> $RUN/corr_<arm>/):
#   E1  BendPick with --stance-jitter 0.06 --jitter-seed 0, --jitter-rep 1 and 2 (the draw keyed by task, level, index and
#       rep, the same for every arm of an episode): AJ<k> A (no --correct), ALJ<k> A with --correct-rule dryR (D's
#       prediction of A's commands recorded; the rows must equal AJ<k>'s), RDJ<k> --correct-rule R (R_D)
#   E2  BendPick, no jitter: RE --correct-rule R --correct-stages entry (only the entry corrected), RP --correct-stages
#       pregrasp (only the pregrasp)
#   V1  TabletopGrasp's 28 routed episodes with selfmodel_simplecmp_20261126's plans, --correct-rule dryR as there:
#       J0S J0's plans (--avoid <simplecmp>/run/P/J0/via --joint) retimed to J_D's pregrasp durations (--joint-dur
#       jdur_J0S.json), JDF J_D's plans (P/JD/via) retimed to J0's (jdur_JDF.json)
# Processes: a queue of (arm, task, episodes) jobs of 5 episodes each; PER_GPU (default 2) workers on each of GPUS
# (default "2 3"); before each job a worker waits until its card's used memory + NEED (default 1024 MiB) <= CAP (default
# 19800 MiB), under one lock ($O/.gpu_lock, shared by every queue of this entry) held 45 s after a start so the next check
# sees the new process. nice 10, one thread, both EGL variables. Before every part the frozen copy is checked against
# asrun/code/CODE_MD5. Progress: $RUN/run.log.
# SCMP2_SMOKE=<dir> (notebook: skip, the pre-registration checks): the dev copy $O/code_dev, rows into <dir>, the
# development tree (devout, SIMPLE_LEVELS 0-4) unless SCMP2_TEST=1 (the test tree, for the bit-identity checks against
# earlier rows); SCMP2_JOBS=<file> the job lines ("arm task episodes rule|none extra...") instead of a part's; no md5 check.
set -uo pipefail
PART=$1
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126
O1=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
R=$O/asrun/code RUN=$O/run TREE=$O/evalout LEVELS=0,1,2
if [ -n "${SCMP2_SMOKE:-}" ]; then
  R=$O/code_dev RUN=$SCMP2_SMOKE
  [ "${SCMP2_TEST:-0}" = 1 ] || { TREE=$O/devout LEVELS=0,1,2,3,4; }
fi
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
MODEL=$SD/train/models_D_s0/E3_ensemble_f1 TEMPLATE=$SD/real/task/data/all.npz
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
read -ra GP <<< "${GPUS:-2 3}"
PER_GPU=${PER_GPU:-2} NEED=${NEED:-1024} CAP=${CAP:-19800}
cd $R || exit 2
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src OMP_NUM_THREADS=1
export MUJOCO_GL=egl SIMPLE_ROOT=$S SIMPLE_EVAL_OUT=$TREE SIMPLE_LEVELS=$LEVELS
SETS="--bench-set controller=scalebfm --bench-set sensor=rgbd"
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"
mkdir -p $RUN/rows $RUN/rows_job $RUN/logs
log() { echo "[$(date -u +%T)] $*" >> $RUN/run.log; }
md5ok() { [ -n "${SCMP2_SMOKE:-}" ] || md5sum -c --quiet $R/CODE_MD5 > /dev/null 2>&1 || { log "FAIL $PART: the frozen copy differs from its CODE_MD5"; exit 2; }; }
md5ok
declare -A ARM_ID=([ALJ1]=11 [ALJ2]=12 [RDJ1]=13 [RDJ2]=14 [RE]=15 [RP]=16 [J0S]=17 [JDF]=18
                   [Rk]=19 [REk]=20 [RPk]=21 [ALJk]=22 [RDJk]=23 [J0k]=24 [J0Ik]=25 [J0Sk]=26 [Rt]=27 [JDFk]=28)  # the last row: checks
JIT="--stance-jitter 0.06 --jitter-seed 0"
# one job: <arm> <task> <episodes> <gpu> <rule|none> [extra...]
job() {
  local arm=$1 T=$2 eps=$3 g=$4 rule=$5; shift 5
  local cor=()
  [ "$rule" = none ] || cor=(--correct $MODEL --correct-rule $rule --correct-template $TEMPLATE --correct-out $RUN/corr_$arm --correct-arm-id ${ARM_ID[$arm]})
  MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/simple_bench.py $T --method direct $SETS --trace \
    --episodes $eps "$@" "${cor[@]}" --out $RUN/rows_job/${arm}_${T}__$(echo $eps | tr ',' '_' | cut -c1-40).jsonl >> $RUN/logs/${arm}_$T.log 2>&1
  log "$arm $T $eps (GPU $g): exit $?"
}
room() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ $((u + NEED)) -le $CAP ]; }
worker() {  # worker <gpu> <queue file>
  local g=$1 Q=$2 line
  while true; do
    exec 9> $O/.gpu_lock
    flock 9
    line=$(head -1 $Q)
    if [ -z "$line" ]; then flock -u 9; exec 9>&-; return; fi
    if ! room $g; then flock -u 9; exec 9>&-; sleep 30; continue; fi
    sed -i '1d' $Q
    log "start $line (GPU $g, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $g) used)"
    job $(echo $line | awk '{print $1, $2, $3}') $g $(echo $line | cut -d' ' -f4-) &
    local pid=$!
    sleep 45
    flock -u 9; exec 9>&-
    wait $pid
  done
}
run_queue() {  # run_queue < job lines
  local Q=$RUN/queue_$PART.txt w
  cat > $Q
  log "queue $PART: $(grep -c . $Q) jobs, GPUs ${GP[*]} x $PER_GPU, need $NEED of cap $CAP MiB"
  for g in "${GP[@]}"; do for ((w = 0; w < PER_GPU; w++)); do worker $g $Q & done; done
  wait
}
merge() {  # merge <arm>...: the job files into <arm>_<task>.jsonl
  local a T
  for a in "$@"; do
    for T in BendPickMP TabletopGraspMP XMovePickTeleop XMoveBendPickTeleop; do
      ls $RUN/rows_job/${a}_${T}__*.jsonl > /dev/null 2>&1 && cat $RUN/rows_job/${a}_${T}__*.jsonl > $RUN/rows/${a}_$T.jsonl
    done
    log "$a: $(cat $RUN/rows_job/${a}_*__*.jsonl 2>/dev/null | wc -l) rows"
  done
}
chunks() {  # chunks <list>: the list in pieces of 5 (one job each)
  python3 -c "import sys; v=sys.argv[1].split(','); [print(','.join(v[i:i + 5])) for i in range(0, len(v), 5)]" "$1"
}
EPS=$(seq -s, 0 29)
ROUTED=$(ls $MAIN/via | sed -n 's/^via_TabletopGraspMP_L\([0-9]\)_e\([0-9]\)\.npz$/\1\2/p' | sed 's/^0//' | sort -n | tr '\n' ',' | sed 's/,$//')
if [ -n "${SCMP2_JOBS:-}" ]; then
  log "smoke jobs from $SCMP2_JOBS"
  run_queue < $SCMP2_JOBS
  merge $(awk '{print $1}' $SCMP2_JOBS | sort -u)
  log "DONE smoke"
  exit 0
fi
jobs_E1() {
  for k in 1 2; do
    chunks $EPS | sed "s|^|RDJ$k BendPickMP |; s|$| R $JIT --jitter-rep $k|"
    chunks $EPS | sed "s|^|ALJ$k BendPickMP |; s|$| dryR $JIT --jitter-rep $k|"
    chunks $EPS | sed "s|^|AJ$k BendPickMP |; s|$| none $JIT --jitter-rep $k|"
  done
}
jobs_E2() {
  chunks $EPS | sed "s|^|RE BendPickMP |; s|$| R --correct-stages entry|"
  chunks $EPS | sed "s|^|RP BendPickMP |; s|$| R --correct-stages pregrasp|"
}
jobs_V1() {
  chunks $ROUTED | sed "s|^|J0S TabletopGraspMP |; s|$| dryR --avoid $O1/run/P/J0/via --joint --joint-dur $O/asrun/jdur_J0S.json|"
  chunks $ROUTED | sed "s|^|JDF TabletopGraspMP |; s|$| dryR --avoid $O1/run/P/JD/via --joint --joint-dur $O/asrun/jdur_JDF.json|"
}
declare -A ARMS=([E1]="AJ1 AJ2 ALJ1 ALJ2 RDJ1 RDJ2" [E2]="RE RP" [V1]="J0S JDF")
case $PART in
  E1 | E2 | V1) PARTS=$PART ;;
  all) PARTS="E1 E2 V1" ;;  # one queue, E1's jobs first
  *) log "FAIL: no part $PART"; exit 2 ;;
esac
log "parts $PARTS; routed TabletopGrasp episodes $ROUTED"
for p in $PARTS; do jobs_$p; done | run_queue
md5ok
for p in $PARTS; do merge ${ARMS[$p]}; done
log "DONE $PART"
