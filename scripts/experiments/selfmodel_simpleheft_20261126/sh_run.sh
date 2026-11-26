#!/usr/bin/env bash
# notebook: selfmodel_simpleheft_20261126 -- the formal run from the frozen copy, one phase:
#   sh_run.sh <phase: A|R>
# HEFT (--bench-set controller=heft, in process from its ONNX file, gait_heft), perception column (sensor=rgbd: the
# cached RGB-D look every controller of the main metric gets), simple_bench.py's A options as bench_via.sh runs them (no
# --film), the main metric's test episodes 0-29 of the four pick tasks (levels 0-2, the same episode numbers and seeds).
# Arms (rows -> $RUN/rows_job/ a job, merged into $RUN/rows/<arm>_<task>.jsonl; --correct's record -> $RUN/corr_<arm>/):
#   phase A  A   direct execution, no --correct
#            AL  A with --correct <D_heft> --correct-rule dryR (D_heft's prediction of A's commands recorded with the
#                episode; the commands sent are A's, so the rows must equal A's)
#   phase R  RD  --correct <D_heft> --correct-rule R (R_D: the pregrasp's and the entry's commands inverted on D_heft)
# D_heft: selfmodel_train3b_20261125's HEFT model D (heft/models_D_s0/E3_ensemble_f1; LocoManiLab data3 + data3b + the
# two mid files; REACHWM_GOAL_REF / APPROACH / QFIX = 1, FDVEL off), the model LocoManiLab's HEFT R ran with.
# Processes: a queue of (arm, task, episodes) jobs of 5 episodes each; PER_GPU (default 2) workers on each of GPUS
# (default "1 2 3"); before each job a worker waits until its card's used memory + NEED (default 1024 MiB) <= CAP (default
# 19800 MiB), under one lock ($O/.gpu_lock) held 45 s after a start so the next check sees the new process. nice 10, one
# thread, both EGL variables. Before and after the phase the frozen copy is checked against asrun/code/CODE_MD5.
# Progress: $RUN/run.log.
# The episodes' tree (SIMPLE_EVAL_OUT): the copy's own outputs/simple_eval_20261112 (sh_copy.sh: a real directory, scene/
# real, the cached looks linked read only); checked before anything runs.
# SH_SMOKE=<dir> (notebook: skip, the pre-registration checks): the dev copy $O/code_dev (or SH_CODE=<code dir>: another
# copy, e.g. ws/code/loco_heft for the bit-identity check; its own simple_bench, never --correct, on code_dev's trees),
# rows into <dir>, the development tree outputs/simple_dev_20261124 (SIMPLE_LEVELS 0-4) unless SH_TEST=1 (the test tree);
# SH_JOBS=<file> the job lines ("arm task episodes rule|none extra...") instead of a phase's; no md5 check.
set -uo pipefail
PHASE=$1
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
R=$O/asrun/code RUN=$O/run LEVELS=0,1,2
[ -z "${SH_SMOKE:-}" ] || { R=${SH_CODE:-$O/code_dev} RUN=$SH_SMOKE; }
TREE=$R/outputs/simple_eval_20261112
if [ -n "${SH_SMOKE:-}" ]; then
  [ -z "${SH_CODE:-}" ] || TREE=$O/code_dev/outputs/simple_eval_20261112  # another copy: still code_dev's trees
  [ "${SH_TEST:-0}" = 1 ] || { TREE=$(dirname $TREE)/simple_dev_20261124 LEVELS=0,1,2,3,4; }
fi
# the tree's scene/ (written every episode) must be a real directory of this entry's own (coordinator 11-26)
case "$(realpath $TREE/scene)" in $O/*) ;; *) echo "FAIL: $TREE/scene is $(realpath $TREE/scene), not under $O"; exit 2 ;; esac
[ ! -L $TREE ] && [ ! -L $TREE/scene ] || { echo "FAIL: $TREE or its scene/ is a link"; exit 2; }
MODEL=$L/selfmodel_train3b_20261125/heft/models_D_s0/E3_ensemble_f1
TEMPLATE=$L/selfmodel_simpledata_20261126/real/task/data/all.npz  # shapes only (scmp_correct.ModelD)
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
read -ra GP <<< "${GPUS:-1 2 3}"
PER_GPU=${PER_GPU:-2} NEED=${NEED:-1024} CAP=${CAP:-19800}
cd $R || exit 2
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=0 PYTHONPATH=src OMP_NUM_THREADS=1
export REACH_HEFT_DELAY=0 MUJOCO_GL=egl SIMPLE_ROOT=$S SIMPLE_EVAL_OUT=$TREE SIMPLE_LEVELS=$LEVELS PYTHONDONTWRITEBYTECODE=1
SETS="--bench-set controller=heft --bench-set sensor=rgbd"
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"
mkdir -p $RUN/rows $RUN/rows_job $RUN/logs
log() { echo "[$(date -u +%T)] $*" >> $RUN/run.log; }
md5ok() { [ -n "${SH_SMOKE:-}" ] || md5sum -c --quiet $R/CODE_MD5 > /dev/null 2>&1 || { log "FAIL $PHASE: the frozen copy differs from its CODE_MD5"; exit 2; }; }
md5ok
declare -A ARM_ID=([AL]=31 [RD]=32 [ALk]=33 [RDk]=34)  # the reset ids of the --correct records (95M + 1M x id + ...); k: checks
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
  local Q=$RUN/queue_$PHASE.txt w
  cat > $Q
  log "queue $PHASE: $(grep -c . $Q) jobs, GPUs ${GP[*]} x $PER_GPU, need $NEED of cap $CAP MiB"
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
TASKS="XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP"
if [ -n "${SH_JOBS:-}" ]; then
  log "smoke jobs from $SH_JOBS (code $R, tree $TREE)"
  run_queue < $SH_JOBS
  merge $(awk '{print $1}' $SH_JOBS | sort -u)
  log "DONE smoke"
  exit 0
fi
jobs_A() {  # A and AL of a task one after the other, so that the two arms of an episode run close in time
  for T in $TASKS; do
    chunks $EPS | sed "s|^|A $T |; s|$| none|"
    chunks $EPS | sed "s|^|AL $T |; s|$| dryR|"
  done
}
jobs_R() {
  for T in $TASKS; do chunks $EPS | sed "s|^|RD $T |; s|$| R|"; done
}
declare -A ARMS=([A]="A AL" [R]="RD")
case $PHASE in
  A | R) ;;
  *) log "FAIL: no phase $PHASE"; exit 2 ;;
esac
log "phase $PHASE (code $R, tree $TREE, levels $LEVELS, model $MODEL)"
jobs_$PHASE | run_queue
md5ok
merge ${ARMS[$PHASE]}
log "DONE $PHASE"
