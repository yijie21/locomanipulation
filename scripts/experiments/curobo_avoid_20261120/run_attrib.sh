#!/usr/bin/env bash
# curobo_attrib_20261120 (4090, CPU): TabletopGrasp direct, the 28 episodes with a cuRobo path, 7 path groups x 3 lateral
# stances (--side-off -0.20 / -0.19 / -0.21), --trace. A0 straight, A1 cuRobo (via_tabletop), B-F attrib/via_<G> (vias_e1.py).
# Progress: attrib/logs/pipeline.log; "FAIL ..." on failed processes, "DONE" at the end.
set -uo pipefail
C=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo
R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
D=$C/attrib
mkdir -p $D/rows $D/logs
log() { echo "[$(date +%T)] $*" >> $D/logs/pipeline.log; }
log "start, commit $(git -C $R log -1 --format=%h) + uncommitted motor.py via_rot, simple_bench.py --avoid (speed)"
cd $R
export REACH_HAND_OFF=0.0415,0,0 BFM_ROOT=odo REACHWM_GOAL_REF=1 PYTHONPATH=src SIMPLE_ROOT=$R/third_party/SIMPLE
export OMP_NUM_THREADS=1 TMPDIR=/workspace/tmp REACH_BFM_DEVICE=cpu CUDA_VISIBLE_DEVICES=
export PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python E=$R/scripts/experiments/simple_eval_20261112 C D
EPS=$(seq 0 29 | grep -vx 12 | grep -vx 24)
for G in A0 A1 B C D E F; do for so in -0.20 -0.19 -0.21; do for e in $EPS; do echo "$G $so $e"; done; done; done |
  xargs -P 150 -L1 sh -c '
    G=$0; so=$1; e=$2
    case $G in A0) av="";; A1) av="--avoid $C/via_tabletop";; *) av="--avoid $D/via_$G";; esac
    $PY $E/simple_bench.py TabletopGraspMP --method direct --column perception --cloud cloud_eyes_final --trace --side-off=$so $av \
      --episodes $e --out $D/rows/${G}_${so}_$e.jsonl > $D/logs/${G}_${so}_$e.log 2>&1 || echo "FAIL $G $so $e" >> $D/logs/pipeline.log'
log "rows $(ls $D/rows | wc -l) of 588"
log "DONE"
