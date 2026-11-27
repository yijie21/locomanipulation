#!/usr/bin/env bash
# notebook: skip relook dev -- simple_relook_20261126: with --relook off, simple_bench.py (and sfull_run.sh's arms) give
# the main metric's rows and films bit for bit. One controller: A and J on episodes 3 and 24 of each task (J through
# bench_rgbd_20261124's own via files), SF and kin_rr on two routed TabletopGrasp episodes (sfull_20261125's via files,
# --film-ref as there); compared with idcmp.py against bench_rgbd_20261124 and sfull_20261125.
#   idcheck.sh <scalebfm|sonic> "<GPUs, one process each at a time>"   -> $X/idcheck/<ctrl>/, $X/idcheck/<ctrl>.txt
# IDC_REPO (default $X/repo): the copy to run from; IDC_OUT (default $X/idcheck): where to write.
set -uo pipefail
X=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126
C=$1; read -ra GS <<< "$2"
B=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112
I=${IDC_OUT:-$X/idcheck}; D=$I/$C; mkdir -p $D/rows; cd ${IDC_REPO:-$X/repo}
export SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACH_SONIC_NETS=gpu
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python E=scripts/experiments/simple_eval_20261112
SETS="--bench-set controller=$C --bench-set sensor=rgbd"
echo "[$(date -u +%FT%TZ)] idcheck $C start $(pwd) md5 $(md5sum $E/simple_bench.py $E/relook.py | cut -c1-8 | tr '\n' ' ')" >> $X/logs/dev_phase.log
jobs_=()
run() {  # run <arm> <task> <eps> <extra...>: one process on the next GPU, at most ${#GS[@]} at a time
  local arm=$1 t=$2 eps=$3; shift 3
  local g=${GS[$((${#jobs_[@]} % ${#GS[@]}))]}
  if [ ${#jobs_[@]} -ge ${#GS[@]} ]; then wait ${jobs_[0]}; jobs_=("${jobs_[@]:1}"); fi
  rm -f $D/rows/${arm}_$t.jsonl
  CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g nice -n 10 $PY $E/simple_bench.py $t --method direct $SETS --trace \
    --episodes $eps --film $D/films_$arm "$@" --out $D/rows/${arm}_$t.jsonl > $D/${arm}_$t.log 2>&1 &
  jobs_+=($!)
}
for t in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do run A $t 3,24; done
for t in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do run J $t 3,24 --avoid $B/bench_rgbd_20261124/$C/via --joint; done
for arm in SF kin_rr; do run $arm TabletopGraspMP 4,24 --avoid $B/sfull_20261125/$C/P/$arm/via --joint --film-ref; done
wait
{ for arm in A J; do $PY -I $X/idcmp.py $D $B/bench_rgbd_20261124/$C $arm 3,24; done
  for arm in SF kin_rr; do $PY -I $X/idcmp.py $D $B/sfull_20261125/$C $arm 4,24; done; } > $I/$C.txt 2>&1
echo "[$(date -u +%FT%TZ)] idcheck $C done: $(grep -h 'rows the same' $I/$C.txt | tr '\n' ' ')" >> $X/logs/dev_phase.log
