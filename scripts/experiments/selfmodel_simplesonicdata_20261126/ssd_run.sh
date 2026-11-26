#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- the SIMPLE task episodes on H20b from the frozen copy, one phase:
#   ssd_run.sh <phase: A|R>
# SONIC (--bench-set controller=sonic, vr_frame "robot" by default, the three nets on the CPU: REACH_SONIC_NETS=allcpu,
# rendering only on the card), perception column (sensor=rgbd: the cached RGB-D look), simple_bench.py's A options as
# bench_via.sh runs them (--trace, no --film), the main metric's test episodes 0-29 of the four pick tasks (levels 0-2).
# Arms (rows -> $RUN/rows_job/ a job, merged into $RUN/rows/<arm>_<task>.jsonl; --correct's record -> $RUN/corr_<arm>/):
#   phase A  A   direct execution, no --correct
#            AL  A with --correct <D_S> --correct-rule dryR (D_S's prediction of A's commands recorded; the commands are
#                A's, so the rows must equal A's)
#   phase R  RD  --correct <D_S> --correct-rule R (R_D)
# D_S: this entry's model, $O/train/models_D_s0/E3_ensemble_f1 (REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1).
# Processes: a queue of (arm, task, episodes) jobs of 5 episodes; PER_GPU (default 5) workers on each of GPUS (default
# "0 1 2 3 4 5 6 7"); before each job a worker waits until its card's used memory + NEED (1024 MiB) <= CAP (80 000 MiB,
# the coordinator's per-card cap, keepalive included), under one lock held HOLD s (default 3) after a start. nice 10, one
# thread, both EGL variables. The frozen copy is checked against its CODE_MD5 before and after. Progress: $RUN/run.log.
# The episodes' tree (SIMPLE_EVAL_OUT): the copy's own outputs/simple_eval_20261112 (ssd_copy.sh: a real directory,
# scene/ real, the cached looks linked read only), checked before anything runs; after the phase, the shared ws tree's
# files newer than the phase's start are listed (must be none).
# SSD_SMOKE=<dir> (notebook: skip, the pre-registration checks): the dev copy $O/code_dev (or SSD_CODE), rows into <dir>,
# SSD_JOBS=<file> the job lines ("arm task episodes rule|none"), SSD_MODEL another model dir; no md5 check.
set -uo pipefail
PHASE=$1
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
O=$L/selfmodel_simplesonicdata_20261126
R=$O/asrun/code RUN=$O/run LEVELS=0,1,2
[ -z "${SSD_SMOKE:-}" ] || { R=${SSD_CODE:-$O/code_dev} RUN=$SSD_SMOKE; }
TREE=$R/outputs/simple_eval_20261112
case "$(realpath $TREE/scene)" in $O/*) ;; *) echo "FAIL: $TREE/scene is $(realpath $TREE/scene), not under $O"; exit 2 ;; esac
[ ! -L $TREE ] && [ ! -L $TREE/scene ] || { echo "FAIL: $TREE or its scene/ is a link"; exit 2; }
MODEL=${SSD_MODEL:-$O/train/models_D_s0/E3_ensemble_f1}
TEMPLATE=$L/selfmodel_simpledata_20261126/real/task/data/all.npz  # shapes only (scmp_correct.ModelD)
S=$Y/tools/simple_isaac/SIMPLE
read -ra GP <<< "${GPUS:-0 1 2 3 4 5 6 7}"
PER_GPU=${PER_GPU:-5} NEED=${NEED:-1024} CAP=${CAP:-80000} HOLD=${HOLD:-3}
cd $R || exit 2
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 REACH_SONIC_NETS=allcpu
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src OMP_NUM_THREADS=1
export MUJOCO_GL=egl SIMPLE_ROOT=$S SIMPLE_EVAL_OUT=$TREE SIMPLE_LEVELS=$LEVELS PYTHONDONTWRITEBYTECODE=1
unset REACHWM_X0_REF
SETS="--bench-set controller=sonic --bench-set sensor=rgbd"
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"
mkdir -p $RUN/rows $RUN/rows_job $RUN/logs
log() { echo "[$(date -u +%T)] $*" >> $RUN/run.log; }
md5ok() { [ -n "${SSD_SMOKE:-}" ] || md5sum -c --quiet $R/CODE_MD5 > /dev/null 2>&1 || { log "FAIL $PHASE: the frozen copy differs from its CODE_MD5"; exit 2; }; }
md5ok
declare -A ARM_ID=([AL]=41 [RD]=42 [ALk]=43 [RDk]=44)  # the reset ids of the --correct records (95M + 1M x id + ...); k: checks
job() {
  local arm=$1 T=$2 eps=$3 g=$4 rule=$5; shift 5
  local cor=()
  [ "$rule" = none ] || cor=(--correct $MODEL --correct-rule $rule --correct-template $TEMPLATE --correct-out $RUN/corr_$arm --correct-arm-id ${ARM_ID[$arm]})
  MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/simple_bench.py $T --method direct $SETS --trace \
    --episodes $eps "$@" "${cor[@]}" --out $RUN/rows_job/${arm}_${T}__$(echo $eps | tr ',' '_' | cut -c1-40).jsonl >> $RUN/logs/${arm}_$T.log 2>&1
  log "$arm $T $eps (GPU $g): exit $?"
}
room() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ $((u + NEED)) -le $CAP ]; }
worker() {
  local g=$1 Q=$2 line
  while true; do
    exec 9> $RUN/.gpu_lock
    flock 9
    line=$(head -1 $Q)
    if [ -z "$line" ]; then flock -u 9; exec 9>&-; return; fi
    if ! room $g; then flock -u 9; exec 9>&-; sleep 30; continue; fi
    sed -i '1d' $Q
    log "start $line (GPU $g, $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $g) used)"
    job $(echo $line | awk '{print $1, $2, $3}') $g $(echo $line | cut -d' ' -f4-) &
    local pid=$!
    sleep $HOLD
    flock -u 9; exec 9>&-
    wait $pid
  done
}
run_queue() {
  local Q=$RUN/queue_$PHASE.txt
  cat > $Q
  log "queue $PHASE: $(grep -c . $Q) jobs, GPUs ${GP[*]} x $PER_GPU, need $NEED of cap $CAP MiB"
  for g in "${GP[@]}"; do for ((w = 0; w < PER_GPU; w++)); do worker $g $Q & done; done
  wait
}
merge() {
  local a T
  for a in "$@"; do
    for T in BendPickMP TabletopGraspMP XMovePickTeleop XMoveBendPickTeleop; do
      ls $RUN/rows_job/${a}_${T}__*.jsonl > /dev/null 2>&1 && cat $RUN/rows_job/${a}_${T}__*.jsonl > $RUN/rows/${a}_$T.jsonl
    done
    log "$a: $(cat $RUN/rows_job/${a}_*__*.jsonl 2>/dev/null | wc -l) rows"
  done
}
chunks() { python3 -c "import sys; v=sys.argv[1].split(','); [print(','.join(v[i:i + 5])) for i in range(0, len(v), 5)]" "$1"; }
shared_written() {  # files of the shared ws trees newer than the marker
  local n
  n=$(find $WS/code/loco/outputs/simple_eval_20261112/ $WS/code/loco/outputs/simple_dev_20261124/ -newer $1 -type f 2>/dev/null | wc -l)
  log "shared ws trees: $n files newer than the phase start$([ $n = 0 ] || echo ' (listed in shared_written.txt)')"
  [ $n = 0 ] || find $WS/code/loco/outputs/simple_eval_20261112/ $WS/code/loco/outputs/simple_dev_20261124/ -newer $1 -type f > $RUN/shared_written_$PHASE.txt
}
EPS=$(seq -s, 0 29)
TASKS="XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP"
touch $RUN/.start_$PHASE
if [ -n "${SSD_JOBS:-}" ]; then
  log "smoke jobs from $SSD_JOBS (code $R, tree $TREE, model $MODEL)"
  run_queue < $SSD_JOBS
  merge $(awk '{print $1}' $SSD_JOBS | sort -u)
  shared_written $RUN/.start_$PHASE
  log "DONE smoke"
  exit 0
fi
jobs_A() { for T in $TASKS; do chunks $EPS | sed "s|^|A $T |; s|$| none|"; chunks $EPS | sed "s|^|AL $T |; s|$| dryR|"; done; }
jobs_R() { for T in $TASKS; do chunks $EPS | sed "s|^|RD $T |; s|$| R|"; done; }
declare -A ARMS=([A]="A AL" [R]="RD")
case $PHASE in A | R) ;; *) log "FAIL: no phase $PHASE"; exit 2 ;; esac
[ -f $MODEL/model.pt ] || { log "FAIL: no model at $MODEL"; exit 2; }
log "phase $PHASE (code $R, tree $TREE, levels $LEVELS, model $MODEL)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $RUN/nvidia_smi_$PHASE.txt
jobs_$PHASE | run_queue
md5ok
merge ${ARMS[$PHASE]}
shared_written $RUN/.start_$PHASE
log "DONE $PHASE"
