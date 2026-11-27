#!/usr/bin/env bash
# faildiag revision 3: run jobs_rev3.txt (D3 ... = fd_heft.py d3 on data3's frozen code and src; HA ... = fd_heft.py ha on
# lmlcmp3's copy), at most $1 at once (default 4), nice 19, CPU only. Logs q3/logs/rev3/job<NNN>.log, .done = exit code.
SLOTS=${1:-4}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126
L=$O/q3/logs/rev3; mkdir -p $L $O/q3/dur $O/q3/lock3
cd /wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
export G=$PWD W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code \
  A3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/asrun PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python \
  HADIR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev/heft_video/run/data O L \
  SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
nl -nrz -w3 $O/q3/jobs_rev3.txt | xargs -P "$SLOTS" -L 1 bash -c '
  n=$0; kind=$1; shift
  [ -f "$L/job$n.done" ] && exit 0
  args=$(echo "$@" | sed -e "s#DUR#$O/q3/dur#" -e "s#LOCK3#$O/q3/lock3#" -e "s#HADIR#$HADIR#")
  if [ "$kind" = D3 ]; then
    D3_REACHWM=$W/scripts/experiments/reachwm_20261030 PYTHONPATH=$G:$W/src nice -n 19 $PY $O/code/fd_heft.py $args --code $A3 > $L/job$n.log 2>&1
  else
    D3_REACHWM=$C0/scripts/experiments/reachwm_20261030 PYTHONPATH=$G:$C0/src nice -n 19 $PY $O/code/fd_heft.py $args --code $C0/scripts/experiments > $L/job$n.log 2>&1
  fi
  echo $? > $L/job$n.done'
echo "rev3 pool finished $(date -u +%FT%TZ)" >> $L/pool.log
