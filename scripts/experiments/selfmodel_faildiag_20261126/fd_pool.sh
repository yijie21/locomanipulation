#!/usr/bin/env bash
# fd_pool.sh <jobs file> <slots> <tag>: run every line of <jobs file> (fd_run.py arguments) with at most <slots> at once,
# CPU only (CUDA_VISIBLE_DEVICES empty), nice 19, one thread each, from lmlcmp3's frozen code (LC_CODE_ROOT).
# Logs: $FD_OUT/logs/<tag>/job<NNN>.log; job<NNN>.done holds its exit code. Progress: ls logs/<tag>/*.done | wc -l.
set -u
JOBS=$1; SLOTS=$2; TAG=$3
FD_OUT=${FD_OUT:-/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126}
C0=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
FD_CODE=${FD_CODE:-$FD_OUT/code}
L=$FD_OUT/logs/$TAG
mkdir -p "$L"
export FD_OUT LC_OUT=$FD_OUT LC_CODE_ROOT=$C0 PYTHONPATH=$G:$C0/src CUDA_VISIBLE_DEVICES= FD_CODE \
  SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 L PY
cd "$G"
nl -nrz -w3 "$JOBS" | xargs -P "$SLOTS" -L 1 bash -c '
  n=$0
  [ -f "$L/job$n.done" ] && exit 0
  nice -n 19 "$PY" "${FD_CODE:-$FD_OUT/code}/fd_run.py" "$@" > "$L/job$n.log" 2>&1
  echo $? > "$L/job$n.done"'
echo "pool $TAG finished $(date -u +%FT%TZ)" >> "$L/pool.log"
