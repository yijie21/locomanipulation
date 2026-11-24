#!/usr/bin/env bash
# notebook: selfmodel_s_dev_20261124 -- the S arm's development episodes (SIMPLE's unused TabletopGrasp "hard-old" and
# "hard" episodes, positions 30-49 of a tree of their own) for both controllers: their look's grasps and table edge, A,
# plan and JR (J filmed), the error radius, the three checks (end, clear, both) and their runs, the tables.
#   s_dev.sh <dev tree> ; env: BENCH_REPO, SIMPLE_ROOT, GG_PY (GraspGen-X's python), GG_ASSETS (its gripper assets),
#   GPU_B (ScaleBFM's card, default 1), GPUS_S (SONIC's cards, default "2 3"). Progress: <dev tree>/logs/dev.log
set -uo pipefail
D=$1
W=${BENCH_REPO:?}; cd $W
E=scripts/experiments/simple_eval_20261112; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
GB=${GPU_B:-1}; GS=${GPUS_S:-"2 3"}; GS0=${GS%% *}
export SIMPLE_EVAL_OUT=$D SIMPLE_LEVELS=0,1,2,3,4 BENCH_TASKS=TabletopGraspMP BENCH_SET=sensor=rgbd REACH_SONIC_NETS=gpu
export PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 SIMPLE_ROOT=${SIMPLE_ROOT:?}
M=$W/outputs/simple_eval_20261112/jrefwm_20261124
ONLY=$(seq 30 49 | tr "\n" " ")
mkdir -p $D/logs $D/bench; L=$D/logs/dev.log
log() { echo "[$(date -u +%T)] $*" >> $L; }
cp $E/s_dev.sh $E/s_check.py $E/s_run.sh $E/bench_via.sh $E/episodes.py $D/logs/ 2>/dev/null
log "start $(git log -1 --format=%h) + uncommitted"
if [ ! -f $D/blind_f1_TabletopGraspMP.json ]; then
  CUDA_VISIBLE_DEVICES=$GB ${GG_PY:?} $E/grasps_graspgen.py $D/cloud_blind --assets-dir ${GG_ASSETS:?} > $D/logs/grasps.log 2>&1
  log "grasps (exit $?): $(ls $D/cloud_blind/grasp_*.npz 2>/dev/null | wc -l) files"
  BFM_ROOT=true CUDA_VISIBLE_DEVICES=$GB MUJOCO_EGL_DEVICE_ID=$GB $PY $E/blind.py run TabletopGraspMP --only $ONLY \
    --out $D/blind_f1_TabletopGraspMP.json > $D/logs/blind_run.log 2>&1
  log "table edge (exit $?): $(python3 -c "import json; r=json.load(open('$D/blind_f1_TabletopGraspMP.json')); print(len(r), 'rows,', sum(x.get('edge') is not None for x in r), 'with an edge,', sum(bool(x.get('success')) for x in r), 'blind.py successes')")"
fi
one() {  # one <controller> <card> <cards>: A, plan, JR; radius; three checks; their runs; tables
  local c=$1 g=$2 gs=$3 B=$D/bench/$1
  for s in A plan JR; do BENCH_GPU=$g BENCH_GPUS="$gs" bash $E/bench_via.sh $B $c none $s 30 49; done
  log "$c: A, plan, JR done; $(tail -1 $B/pipeline.log)"
  CUDA_VISIBLE_DEVICES=$g $PY -I $E/jref_data.py build $B/jref_dev outputs/scene/simple_g1_sonic $B/films_JR $B/rows/JR_TabletopGraspMP.jsonl > $B/logs/jref_dev.txt 2>&1
  $PY -I $E/s_check.py calib $M/model_$c $B/jref_dev > $B/calib.json 2> $B/logs/calib.txt
  local r=$(python3 -c "import json; print(json.load(open('$B/calib.json'))['direction']['q90'])")
  log "$c: radius (90 % of the development paths' errors) $r cm; $(cat $B/calib.json)"
  for v in end clear both; do
    case $v in end) o="--tau-end 2 --delta 1e9";; clear) o="--tau-end 1e9 --delta 0 --r $r";; both) o="--tau-end 2 --delta 0 --r $r";; esac
    BENCH_GPU=$g bash $E/s_run.sh $B $c $M/model_$c $B/S_$v 2 $o
    log "$c S_$v: $(tail -1 $B/S_$v/s_pipeline.log)"
  done
  local pids=() i=0 gg
  read -ra G <<< "$gs"
  for v in end clear both; do
    gg=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
    S_DIR=$B/S_$v S_ARM=S_$v BENCH_GPU=$gg BENCH_GPUS=$gg bash $E/bench_via.sh $B $c none S 30 49 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  for v in end clear both; do $PY -I $E/s_check.py table $B/S_$v $B --arm S_$v --j-arm JR > $B/S_$v/table.txt 2>&1; done
  log "$c: S runs and tables done"
}
one scalebfm $GB "$GB" &
one sonic $GS0 "$GS" &
wait
log "all done"
