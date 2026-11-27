#!/usr/bin/env bash
# faildiag revision 1: the three HA reach commands in LML, waist pitch unlocked / locked, 3 repetitions each (2 processes)
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126
mkdir -p $O/q3/logs $O/q3/waist
cd /wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend; G=$PWD
C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code; PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
HA=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev/heft_video/run/data
export SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= D3_REACHWM=$C0/scripts/experiments/reachwm_20261030 PYTHONPATH=$G:$C0/src
for LOCK in "" "--lock-waist"; do
  setsid nohup nice -n 19 bash -c "for k in 0 1 2; do $PY $O/code/fd_heft.py ha --code $C0/scripts/experiments --ha-npz $HA/ppbox_s297_r0.npz --chains 0,2 --rep \$k $LOCK --out $O/q3/waist; $PY $O/code/fd_heft.py ha --code $C0/scripts/experiments --ha-npz $HA/ppbox_s296_r0.npz --chains 0 --rep \$k $LOCK --out $O/q3/waist; done" > $O/q3/logs/waist${LOCK:+_lock}.log 2>&1 < /dev/null &
  echo "waist ${LOCK:-unlocked} pid $!"
done
