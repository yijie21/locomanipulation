#!/usr/bin/env bash
# notebook: selfmodel_simpledata_20261126
# selfmodel_simpledata_20261126 (revision 2), evaluation 3's episodes on the 4090 from the asrun copy (tmux sd_real):
#   sd_real.sh <GPUs, comma-separated>
# The main metric's A arm episodes (four pick tasks, episodes 0-29) under sd_real.py's recorder, the main label and the
# task label: 8 processes (label x task), the GPUs in turn (ScaleBFM's policy on the card, as the A arm ran it:
# REACH_BFM_DEVICE unset; CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID the same card), nice 10. nvidia-smi logged
# first. Then sd_real_merge.py per label -> $REAL/<label>/merge.txt ("REAL MERGE <label> PASS"; the task label's rows
# must equal the A arm's). Out $REAL/<label>/{data,rows,log,scene}, logs $REAL/logs/<label>_<task>.log, $REAL/run.log.
# Test only: SD_TEST=1 with SD_A (a repo copy's parent, no md5 check), SD_REAL (out), SD_EPISODES (default 0-29), SD_TASKS.
set -uo pipefail
IFS=, read -ra GPUS <<< "$1"
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
A=${SD_A:-$O/asrun}
R=$A/repo
E=$R/scripts/experiments/selfmodel_simpledata_20261126
REAL=${SD_REAL:-$O/real}
EPS=${SD_EPISODES:-0-29}
TASKS=(${SD_TASKS:-XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP})
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
cd $R || exit 1
[ -n "${SD_TEST:-}" ] || (cd $A && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || { echo "asrun md5 failed"; exit 1; }
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE SIMPLE_EVAL_OUT=/workspace/code/loco/outputs/simple_eval_20261112 SX_REPO=$R
unset REACH_BFM_DEVICE
mkdir -p $REAL/logs
log() { echo "$(date -u +%FT%TZ) $1" | tee -a $REAL/run.log; }
log "start: episodes $EPS, tasks ${TASKS[*]}, GPUs ${GPUS[*]}; sd_real $(md5sum < $E/sd_real.py | cut -c1-8), simple_bench $(md5sum < scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $REAL/run.log
PIDS=(); n=0
for lab in main task; do
  for t in "${TASKS[@]}"; do
    [ -e $REAL/$lab/rows/$t.jsonl ] && { log "refuse: $REAL/$lab/rows/$t.jsonl exists"; exit 1; }
    g=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n + 1))
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g nice -n 10 $PY $E/sd_real.py $t --label $lab --episodes $EPS --out $REAL/$lab \
      > $REAL/logs/${lab}_$t.log 2>&1 &
    PIDS+=($!)
    log "$lab $t on GPU $g: pid $!"
  done
done
rc=0
for p in "${PIDS[@]}"; do wait $p; e=$?; [ $e = 0 ] || { rc=1; log "pid $p exit $e"; }; done
log "episodes done (any nonzero exit: $rc)"
for lab in main task; do
  $PY $E/sd_real_merge.py $REAL/$lab --label $lab --episodes $EPS --tasks "$(IFS=,; echo "${TASKS[*]}")" > $REAL/$lab/merge.txt 2>&1
  tail -n 1 $REAL/$lab/merge.txt | sed "s/^/  /" | tee -a $REAL/run.log
done
log "done"
