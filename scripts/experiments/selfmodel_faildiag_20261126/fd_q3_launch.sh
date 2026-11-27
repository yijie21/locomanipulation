#!/usr/bin/env bash
# faildiag Q3: HEFT in-place reruns of data3 (shards 0, 1: first 60 resets each) and the HA reach commands, in LML
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126
mkdir -p $O/q3/void $O/q3/logs
for d in d3 ha d3_smoke_c0src d3_smoke_wsrc; do [ -e $O/q3/$d ] && mv $O/q3/$d $O/q3/void/${d}_hostbody_pelvis; done
cd /wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend; G=$PWD
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code
A3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/asrun; PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
export SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
for s in 0 1; do
  D3_REACHWM=$W/scripts/experiments/reachwm_20261030 PYTHONPATH=$G:$W/src setsid nohup nice -n 19 $PY $O/code/fd_heft.py d3 --code $A3 --shard $s --n 60 --qpos --out $O/q3/d3 > $O/q3/logs/d3_w$s.log 2>&1 < /dev/null &
  echo "d3 shard $s pid $!"
done
HA=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev/heft_video/run/data
export D3_REACHWM=$C0/scripts/experiments/reachwm_20261030 PYTHONPATH=$G:$C0/src
setsid nohup nice -n 19 bash -c "$PY $O/code/fd_heft.py ha --code $C0/scripts/experiments --ha-npz $HA/ppbox_s297_r0.npz --chains 0,2 --qpos --out $O/q3/ha; $PY $O/code/fd_heft.py ha --code $C0/scripts/experiments --ha-npz $HA/ppbox_s296_r0.npz --chains 0 --qpos --out $O/q3/ha" > $O/q3/logs/ha.log 2>&1 < /dev/null &
echo "ha pid $!"
