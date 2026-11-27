#!/usr/bin/env bash
# notebook: selfmodel_fdvel_20261126
# selfmodel_fdvel_20261126: train F and evaluate it, from the frozen copy $A/repo (fd_freeze.sh). tmux fdvel.
#   fd_run.sh <GPU for seed 0> <GPU for seed 1> <GPU for seed 2>
# 1. train: F = reachwm_20261030 train.py E3_ensemble --frac 1 on selfmodel_data3_20261125/<backend>/data/all.npz (the
#    one-hand segments; data.py's splits), train3's T3 recipe unchanged (30k iterations, batch 1024, 5 members,
#    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1) plus REACHWM_FDVEL=1; scalebfm, heft, sonic x seeds 0, 1, 2,
#    the three of a seed on that seed's GPU; nice 10, 4 CPU threads each; each started once the previous one has loaded
#    its data. Models: $O/<backend>/models_s<seed>/E3_ensemble_f1; logs: $O/<backend>/logs/train_s<seed>.log.
# 2. LocoManiLab (the backends side by side, one GPU each): train3's d3_eval.py on every F model (val, test_iid;
#    REACHWM_FDVEL=1): $O/<backend>/d3eval_s<k>_<split>.txt; fd_eval_lml.py, F against T3: $O/<backend>/eval_lml_<split>.{txt,json}.
# 3. xfer: the frozen sx_eval.py (fd_sx1.sh) on the nine records for T3 and F_s0..2, settled (primary) and ended
#    (supplement), six at a time on the CPU: $O/xfer/<record>/<tag>_<keep>.{txt,json}; T3's output compared with the
#    record's own eval.txt / eval_ended.txt; fd_xfer_table.py: $O/xfer_table.{txt,json}.
# Every phase in $O/run.log and $O/STATUS; a failure stops the run (STATUS "FAIL ...").
# Test only: FD_OUT (output dir), FD_ASRUN (frozen copy), FD_ITERS (iterations), FD_BACKENDS, FD_RECORDS.
set -uo pipefail
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
O=${FD_OUT:-$R/selfmodel_fdvel_20261126}
A=${FD_ASRUN:-$O/asrun}
D3=$R/selfmodel_data3_20261125
XF=$R/selfmodel_xfer_20261125
IT=${FD_ITERS:+--iters $FD_ITERS}
BES=${FD_BACKENDS:-scalebfm heft sonic}
RECS=${FD_RECORDS:-scalebfm_simple sonic_simple heft_simple scalebfm_ha heft_ha sonic_ha scalebfm_simple_lmlref sonic_simple_lmlref heft_simple_lmlref}
GPUS=("$@")
[ ${#GPUS[@]} = 3 ] || { echo "three GPUs, one per seed"; exit 1; }
mkdir -p $O
cd $A/repo || exit 1
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
H=scripts/experiments/selfmodel_fdvel_20261126
phase() { echo "$1" > $O/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $O/run.log; }
die() { phase "FAIL $1"; exit 1; }
py() {  # <log> <command...>: unfiltered output in <log>.raw, Warn lines dropped in <log>
  local log=$1; shift
  "$@" > $log.raw 2>&1
  local rc=$?
  grep -v Warn $log.raw > $log
  return $rc
}
(cd $A && md5sum -c --quiet MD5.check) || die "asrun md5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1
unset REACHWM_FDVEL REACHWM_X0_REF REACHWM_SUBSET REACHWM_TRAIN_HELD REACHWM_ROOT
phase "code: asrun md5 ok ($A, frozen $(cat $A/FROZEN_UTC)); GPUs ${GPUS[*]} for seeds 0 1 2 on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $O/run.log
phase "train"
PIDS=()
for s in 0 1 2; do
  for be in $BES; do
    M=$O/$be/models_s$s
    [ -f $M/E3_ensemble_f1/model.pt ] && continue
    mkdir -p $O/$be/logs
    log=$O/$be/logs/train_s$s.log
    env REACHWM_FDVEL=1 REACHWM_ROOT=$O/$be CUDA_VISIBLE_DEVICES=${GPUS[$s]} nice -n 10 $VPY scripts/experiments/reachwm_20261030/train.py \
      E3_ensemble --seed $s --frac 1 --data $D3/$be/data/all.npz --out-root $M $IT > $log 2>&1 &
    p=$!
    PIDS+=($p)
    echo "$(date -u +%FT%TZ) train $be seed $s on GPU ${GPUS[$s]}: pid $p" >> $O/run.log
    until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
    echo "$(date -u +%FT%TZ)   loaded: $(grep " segments, splits" $log | cut -c1-70); free $(free -g | awk '/Mem/ {print $7}') GB" >> $O/run.log
  done
done
echo "${PIDS[@]}" > $O/train_pids.txt
for p in "${PIDS[@]}"; do wait $p || die "training pid $p"; done
grep -H "best val" $O/*/logs/train_s*.log | tee -a $O/run.log
phase "eval lml"
lml_eval() {  # <backend> <GPU>: d3_eval on each F seed, then fd_eval_lml, for val and test_iid
  local be=$1 g=$2 sp s
  for sp in val test_iid; do
    for s in 0 1 2; do
      py $O/$be/d3eval_s${s}_$sp.txt env REACHWM_FDVEL=1 REACHWM_ROOT=$O/$be CUDA_VISIBLE_DEVICES=$g $VPY \
        scripts/experiments/selfmodel_data3_20261125/d3_eval.py $D3/$be $O/$be/models_s$s --split $sp \
        --json $O/$be/d3eval_s${s}_$sp.json || { echo "d3_eval $be s$s $sp failed"; return 1; }
    done
    py $O/$be/eval_lml_$sp.txt env CUDA_VISIBLE_DEVICES=$g $VPY $H/fd_eval_lml.py --data $D3/$be --t3 $D3/$be/models_s0 \
      --f $O/$be/models_s0 $O/$be/models_s1 $O/$be/models_s2 --split $sp --json $O/$be/eval_lml_$sp.json \
      || { echo "fd_eval_lml $be $sp failed"; return 1; }
  done
  echo "$(date -u +%FT%TZ) eval lml $be done" >> $O/run.log
}
LP=()
i=0
for be in $BES; do  # the backends side by side, each on one of the three GPUs
  lml_eval $be ${GPUS[$i]} > $O/$be/eval_lml.status 2>&1 &
  LP+=($!)
  i=$((i + 1))
done
for p in "${LP[@]}"; do wait $p || die "eval lml: $(cat $O/*/eval_lml.status | tr '\n' ' ')"; done
phase "eval xfer"
mkdir -p $O/xfer
export FD_O=$O FD_A=$A FD_D3=$D3 FD_XF=$XF
for r in $RECS; do for t in T3 F_s0 F_s1 F_s2; do for k in settled ended; do echo "$r $t $k"; done; done; done > $O/xfer/jobs.txt
xargs -P 6 -L 1 bash $H/fd_sx1.sh < $O/xfer/jobs.txt || die "sx_eval (see $O/xfer/done.log)"
for r in $RECS; do
  for k in settled ended; do
    ref=$XF/$r/eval.txt; [ $k = ended ] && ref=$XF/$r/eval_ended.txt
    if cmp -s $O/xfer/$r/T3_$k.txt $ref; then echo "  T3 $r $k: identical to $ref"; else echo "  T3 $r $k: DIFFERS from $ref"; fi
  done
done | tee $O/xfer/t3_reproduce.txt | tee -a $O/run.log
phase "table"
py $O/xfer_table.txt env SX_REPO=$A/repo $VPY $H/fd_xfer_table.py --out $O --data3 $D3 --xfer $XF --records $RECS \
  --json $O/xfer_table.json || die "fd_xfer_table"
phase "done"
