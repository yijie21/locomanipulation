#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# The round's looks on 4090 (revision 1: the simulator's mask): lrt_look.sh <eps a:b> <shards per task> <task...>
# lmlcmp3's lc3_look.sh from the round's private copy (code/): hardgrasp's hg_run.py look (ScaleBFM's driver, the episode up
# to the look, hg_look.look: head RGB-D, the mask, GraspGen-X, cached) with HG_MASK = $LRT_MASK (gt: the simulator's mask;
# sam3: SAM 3 on $LRT_SAM3_PORT, only to show the switch off changes nothing), HG_OUT = $LRT_HG_OUT (looks/ is written
# there; train_q_range.npz must be there), GraspGen-X on $LRT_GGX_PORT, EGL on card $LRT_EGL_DEVICE (never 0), no CUDA,
# nice 19, single-threaded; pids in $LRT_HG_OUT/look_pids.txt; waits for its own processes (by pid).
EPS=$1; NS=$2; shift 2
LR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
W=${LRT_CODE:-$LR/code}
CODE=$W/scripts/experiments/hardgrasp_20261125
export HG_OUT=${LRT_HG_OUT:?LRT_HG_OUT required} HG_MASK=${LRT_MASK:?LRT_MASK required (gt or sam3)}
export HG_GGX_PORT=${LRT_GGX_PORT:?LRT_GGX_PORT required} HG_SAM3_PORT=${LRT_SAM3_PORT:-18832}
[ "${LRT_EGL_DEVICE:?LRT_EGL_DEVICE required}" = 0 ] && { echo "card 0 is not ours"; exit 1; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$LRT_EGL_DEVICE
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
[ -e $HG_OUT/train_q_range.npz ] || { echo "no $HG_OUT/train_q_range.npz"; exit 1; }
mkdir -p $HG_OUT/logs
PIDS=()
for T in "$@"; do
  for ((k=0; k<NS; k++)); do
    nohup nice -n 19 $PY $CODE/hg_run.py look --task $T --eps=$EPS --shard $k --shards $NS > $HG_OUT/logs/look_${EPS/:/-}_${T}_$k.log 2>&1 &
    PIDS+=($!)
    echo "$! $T $k" >> $HG_OUT/look_pids.txt
  done
done
echo "$(date -u +%FT%TZ) launched looks $EPS mask $HG_MASK: $# tasks x $NS shards = ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) looks finished"
