#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp3_20261126
# The new seeds' looks for selfmodel_lmlcmp3_20261126 on 4090: lc3_look.sh <eps a:b> <shards per task> <task...>
# hardgrasp's hg_batch.sh look phase unchanged in substance -- hg_run.py look (ScaleBFM's driver, the episode up to the
# look, hg_look.look: head RGB-D, SAM 3, GraspGen-X, cached) from the repo copy of hardgrasp_20261125 -- with this
# machine's settings: HG_OUT = lmlcmp3's hg/ (looks/ is written there; train_q_range.npz there is hardgrasp's copy),
# EGL on the GPU of LC3_EGL_DEVICE (MUJOCO_EGL_DEVICE_ID; GPU 0 never), the services on LC3_SAM3_PORT / LC3_GGX_PORT,
# CUDA_VISIBLE_DEVICES empty (the look processes need no CUDA), nice 19, single-threaded; pids in hg/look_pids.txt;
# waits for its own processes (by pid).
EPS=$1; NS=$2; shift 2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
CODE=$W/scripts/experiments/hardgrasp_20261125
export HG_OUT=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/hg
export HG_SAM3_PORT=${LC3_SAM3_PORT:?LC3_SAM3_PORT required} HG_GGX_PORT=${LC3_GGX_PORT:?LC3_GGX_PORT required}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${LC3_EGL_DEVICE:?LC3_EGL_DEVICE required}
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
echo "$(date -u +%FT%TZ) launched looks $EPS: $# tasks x $NS shards = ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) looks finished"
