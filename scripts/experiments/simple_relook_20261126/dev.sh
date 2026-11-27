#!/usr/bin/env bash
# notebook: skip relook dev -- simple_relook_20261126 development: simple_bench.py on the development episodes
# (TabletopGrasp positions 30-49 of outputs/simple_dev_20261124, SIMPLE's unused "hard" layouts), one process.
#   dev.sh <tag> <scalebfm|sonic> <gpu> <episodes> [simple_bench args...]  -> $X/dev/<tag>.jsonl, $X/logs/<tag>.log
# SONIC_NETS (default gpu) is REACH_SONIC_NETS. Runs from the private copy $X/repo.
set -uo pipefail
X=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126
TAG=$1 CTRL=$2 G=$3 EPS=$4; shift 4
cd ${DEV_REPO:-$X/repo}  # DEV_REPO: another private copy (development while a smoke runs from repo)
export SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE SIMPLE_EVAL_OUT=${DEV_OUT:-$X/repo/outputs/simple_dev_20261124}
export SIMPLE_LEVELS=${DEV_LEVELS:-0,1,2,3,4} PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1
export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G REACH_SONIC_NETS=${SONIC_NETS:-gpu}
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
rm -f $X/dev/$TAG.jsonl
echo "[$(date -u +%FT%TZ)] start $TAG $CTRL gpu $G eps $EPS nets $REACH_SONIC_NETS args $* md5 $(md5sum scripts/experiments/simple_eval_20261112/simple_bench.py scripts/experiments/simple_eval_20261112/relook.py | cut -c1-8 | tr '\n' ' ')" >> $X/logs/dev_phase.log
nice -n 10 $PY scripts/experiments/simple_eval_20261112/simple_bench.py ${DEV_TASK:-TabletopGraspMP} --method direct \
  --bench-set controller=$CTRL --bench-set sensor=rgbd --trace --episodes $EPS "$@" --out $X/dev/$TAG.jsonl > $X/logs/$TAG.log 2>&1
echo "[$(date -u +%FT%TZ)] done $TAG exit $? rows $(wc -l < $X/dev/$TAG.jsonl)" >> $X/logs/dev_phase.log
