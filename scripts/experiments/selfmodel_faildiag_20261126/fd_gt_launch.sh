#!/usr/bin/env bash
# notebook: selfmodel_faildiag_20261126
# faildiag question 2: GT grasps (fd_gt.py, the GraspGen-X service on GPU 3, port 18873) for every episode q2 uses,
# both controllers' lists merged per task; 2 processes (dumbbell + candle, watering can + apple), CPU only.
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
mkdir -p $O/logs/gt
cd $O/code_q2 && /workspace/venvs/sim/bin/python - > $O/logs/gt/eps.txt <<'PY'
import fd_jobs as J
for t, k in ((J.DB, None), (J.WC, None), (J.CANDLE, 20), (J.APPLE, 20)):
    e = sorted(set(J.post_split("scalebfm", t)[:k]) | set(J.post_split("sonic", t)[:k]))
    print(t, ",".join(map(str, e)))
PY
cd $G
run_pair() {
  for t in "$@"; do
    eps=$(awk -v t=$t '$1 == t {print $2}' $O/logs/gt/eps.txt)
    FD_OUT=$O LC_OUT=$O LC_CODE_ROOT=$C0 PYTHONPATH=$G:$C0/src HG_GGX_PORT=18873 CUDA_VISIBLE_DEVICES= SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      nice -n 19 $PY $O/code_q2/fd_gt.py --task $t --ep-list $eps >> $O/logs/gt/$1.log 2>&1
    echo "$t $?" >> $O/logs/gt/$1.done
  done
}
export -f run_pair
setsid nohup bash -c "$(declare -f run_pair); O=$O G=$G C0=$C0 PY=$PY; run_pair IThorLivingRoomApproachTableAndLiftDumbbell IThorBathroomLiftCandleFromFloor" > /dev/null 2>&1 < /dev/null &
setsid nohup bash -c "$(declare -f run_pair); O=$O G=$G C0=$C0 PY=$PY; run_pair IThorLivingRoomApproachTableAndLiftWateringCan IThorKitchenApproachTableAndLiftApple" > /dev/null 2>&1 < /dev/null &
awk '{print $1, gsub(/,/, ",") + 1}' $O/logs/gt/eps.txt
