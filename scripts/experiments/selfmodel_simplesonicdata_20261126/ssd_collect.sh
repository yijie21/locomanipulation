#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- the SONIC collection on H20b (CPU only, nice 10), from a copy:
#   ssd_collect.sh idcheck              (notebook: skip) the copy's sc_collect with no settle option on h20b_bringup's
#                                       asrun_test resets (ScaleBFM, d3 ids 1,5,9,13,30 and d3b ids 1,5,12,29,57, shard 1
#                                       of 4) against h20b_bringup's sc/asrun_test_h20b (simpledata's frozen code): the
#                                       settle options' default must change nothing -> $O/checks/idcheck/same.txt
#   ssd_collect.sh dev <n3> <n3b>       (notebook: skip) the dev ids: data3 style resets 0..19 (ids 144M, seeds 146M) and
#                                       data3b style 0..19 (ids 145M, seeds 147M), <n3> / <n3b> shards each -> $O/dev/
#   ssd_collect.sh formal <n3> <n3b>    the formal collection: data3 style 30 000 resets (ids 140M, seeds 142M) in <n3>
#                                       shards and data3b style 15 000 resets (ids 141M, seeds 143M) in <n3b> shards ->
#                                       $O/data3, $O/data3b; then, when every shard exited 0, merge, check and summary of
#                                       both (-> <m>/check.txt, <m>/summary.txt) and $O/logs/finished
# Every style: --controller sonic --no-walk --settle-v 0.05 --settle-max-s 6 --hold-s 7.5 (the user's decision 11-26:
# SONIC in place only, 5 cm/s, as HumanoidArena's SONIC), SONIC's defaults (perturb 0.005 rad, delay 4 frames; sx_collect's
# SONIC env: REACH_SONIC_NETS allcpu, REACH_SONIC_THREADS 1). One process a shard, one thread, CUDA_VISIBLE_DEVICES empty,
# SX_BFM_DEVICE cpu, 4090b's launcher (py4090b.py; LOCOMANI_WORKSPACE, SONIC_POLICY, SX_REPO = the copy). Code: $O/asrun/code
# (formal; its CODE_MD5 checked first) or SSD_CODE (default $O/code_dev for idcheck / dev).
set -uo pipefail
MODE=$1
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
O=$L/selfmodel_simplesonicdata_20261126
if [ $MODE = formal ]; then R=$O/asrun/code; else R=${SSD_CODE:-$O/code_dev}; fi
cd $R || exit 2
[ $MODE != formal ] || md5sum -c --quiet CODE_MD5 || { echo "CODE_MD5 differs"; exit 2; }
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 SX_REPO=$R OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= SX_BFM_DEVICE=cpu D3_REACHWM=$R/scripts/experiments/reachwm_20261030 PYTHONDONTWRITEBYTECODE=1
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
C=$R/scripts/experiments/selfmodel_simplecollect_20261126/sc_collect.py
export PY C  # the finisher's child shell runs finish()
RULE="--controller sonic --no-walk --settle-v 0.05 --settle-max-s 6 --hold-s 7.5"
launch() {  # launch <out dir> <log name> <args...>: one detached process, exit code and times beside its log
  local d=$1 n=$2; shift 2
  mkdir -p $d/logs
  setsid nohup bash -c "date -u +%FT%TZ > $d/logs/$n.start; nice -n 10 $PY $C drive --out $d $* > $d/logs/$n.log 2>&1; echo \$? > $d/logs/$n.exit; date -u +%FT%TZ > $d/logs/$n.end" > /dev/null 2>&1 < /dev/null &
}
finish() {  # finish <dir> <args...>: merge, check, summary
  local d=$1; shift
  nice -n 10 $PY $C merge --out $d "$@" > $d/merge.txt 2>&1 &&
    nice -n 10 $PY $C check --out $d "$@" > $d/check.txt 2>&1
  local rc=$?
  nice -n 10 $PY $C summary --out $d "$@" > $d/summary.txt 2>&1
  echo "$(date -u +%FT%TZ) finish $d: merge+check rc $rc; $(grep -c '^CHECK PASSED' $d/check.txt) CHECK PASSED, $(grep -c '^SIMPLE CHECK PASSED' $d/check.txt) SIMPLE CHECK PASSED"
  return $rc
}
case $MODE in
  idcheck)
    T=$O/checks/idcheck; B=$L/h20b_bringup_20261126/sc/asrun_test_h20b
    [ -e $T ] && { echo "exists: $T"; exit 2; }
    mkdir -p $T
    cd $R/scripts/experiments/selfmodel_simplecollect_20261126 || exit 2
    nice -n 10 $PY sc_collect.py drive --controller scalebfm --out $T/d3 --ids 1,5,9,13,30 --shards 4 --shard 1 > $T/d3.log 2>&1 &
    nice -n 10 $PY sc_collect.py drive --controller scalebfm --b --out $T/d3b --ids 1,5,12,29,57 --shards 4 --shard 1 > $T/d3b.log 2>&1 &
    wait
    for j in d3 d3b; do echo "== $j: this copy vs $B/$j"; $Y/envs/locomanilab_py311/bin/python $O/../selfmodel_simpledata_20261126/tools/same.py $B/$j $T/$j; done > $T/same.txt 2>&1
    cat $T/same.txt
    ;;
  dev)
    N3=$2 N3B=$3
    for k in $(seq 0 $((N3 - 1))); do launch $O/dev/data3 s$k $RULE --resets 20 --seed 146000000 --id0 144000000 --shards $N3 --shard $k; done
    for k in $(seq 0 $((N3B - 1))); do launch $O/dev/data3b s$k $RULE --b --resets 20 --seed 147000000 --id0 145000000 --shards $N3B --shard $k; done
    echo "dev: $N3 + $N3B processes launched ($(date -u +%T))"
    ;;
  formal)
    N3=$2 N3B=$3
    mkdir -p $O/logs
    [ -e $O/data3 ] || [ -e $O/data3b ] && { echo "a formal output exists"; exit 2; }
    for k in $(seq 0 $((N3 - 1))); do launch $O/data3 s$k $RULE --resets 30000 --seed 142000000 --id0 140000000 --shards $N3 --shard $k; done
    for k in $(seq 0 $((N3B - 1))); do launch $O/data3b s$k $RULE --b --resets 15000 --seed 143000000 --id0 141000000 --shards $N3B --shard $k; done
    echo "$(date -u +%FT%TZ) formal: data3 $N3 shards, data3b $N3B shards launched (code $R)" | tee -a $O/logs/collect.log
    setsid nohup bash -c "
      until [ \$(ls $O/data3/logs/*.exit 2>/dev/null | wc -l) -eq $N3 ] && [ \$(ls $O/data3b/logs/*.exit 2>/dev/null | wc -l) -eq $N3B ]; do sleep 60; done
      bad=\$(cat $O/data3/logs/*.exit $O/data3b/logs/*.exit | grep -vc '^0\$')
      echo \"\$(date -u +%FT%TZ) all shards exited: \$bad nonzero\" >> $O/logs/collect.log
      [ \$bad = 0 ] || { echo FAIL > $O/logs/failed; exit 1; }
      $(declare -f finish)
      finish $O/data3 >> $O/logs/collect.log 2>&1; r1=\$?
      finish $O/data3b --b >> $O/logs/collect.log 2>&1; r2=\$?
      [ \$r1 = 0 ] && [ \$r2 = 0 ] && date -u +%FT%TZ > $O/logs/finished || echo FAIL > $O/logs/failed
    " > $O/logs/finisher.out 2>&1 < /dev/null &
    echo "finisher pid $!"
    ;;
  finish_dev)  # (notebook: skip) ssd_collect.sh finish_dev [dev dir name, default dev]
    finish $O/${2:-dev}/data3; finish $O/${2:-dev}/data3b --b
    ;;
  *) echo "no mode $MODE"; exit 2 ;;
esac
