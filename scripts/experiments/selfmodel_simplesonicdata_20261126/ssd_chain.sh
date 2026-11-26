#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- the same-day chain after the collection, on H20b from the frozen copy:
#   ssd_chain.sh <GPU for frac 1.0>,<GPU for frac 0.3>
# 1 ssd_train.sh (waits for the collection; D_S and its held-out errors); 2 phase A (ssd_run.sh A: A and AL, 240
# episodes); 3 the consistency (every job exit 0; A = AL on all 120 episodes; plan_same on every AL stage; online =
# offline < 1e-4 m, sh_err.py's): any failure stops the chain before the gate ("stop and find out"); 4 the gate
# (ssd_gate.sh); GATE FAIL stops the chain; GATE PASS: 5 phase R (ssd_run.sh R: R_D, 120 episodes), 6 F1 (ssd_f1.sh),
# 7 the evaluation (ssd_eval.py -> $O/eval.txt, eval.json). Log: $O/chain.log.
set -uo pipefail
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs
O=$L/selfmodel_simplesonicdata_20261126
R=$O/asrun/code S=$R/scripts/experiments/selfmodel_simplesonicdata_20261126
MAIN=$Y/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/sonic
PYE="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
C=$O/chain.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $C; }
log "chain start (GPUs $1)"
bash $S/ssd_train.sh $1 || { log "FAIL train ($(cat $O/train/STATUS))"; exit 1; }
log "train done: $(grep -h 'best val' $O/train/logs/train_*.log | tr '\n' ';')"
log "phase A start"
bash $S/ssd_run.sh A
grep -q "DONE A" $O/run/run.log || { log "FAIL phase A: no DONE A"; exit 1; }
n=$(grep -c "): exit" $O/run/run.log); bad=$(grep "): exit" $O/run/run.log | grep -vc "exit 0$")
log "phase A done: $n jobs, $bad nonzero exits"
[ $bad = 0 ] || { log "STOP: nonzero exits in phase A"; exit 1; }
$PYE $S/ssd_eval.py --run $O/run --main $MAIN --json $O/run/eval_A.json > $O/run/eval_A.txt 2>&1 || { log "FAIL eval_A"; exit 1; }
log "phase A: $(grep -m1 'A = AL' $O/run/eval_A.txt | cut -c1-120)"
bash $S/ssd_gate.sh > $O/gate.out 2>&1
grep -q "^GATE" $O/gate/gate.txt || { log "FAIL gate: no verdict"; exit 1; }
python3 - $O <<'EOF' || { log "STOP: the consistency failed (see consistency.txt)"; exit 1; }
import glob, json, sys
O = sys.argv[1]
same = json.load(open(f"{O}/run/eval_A.json"))["same_A_AL"]
S = [s for f in glob.glob(f"{O}/run/corr_AL/log/*.jsonl") for line in open(f) for s in json.loads(line)["stages_correct"]]
ps = sum(s.get("plan_same") is True for s in S)
oo = json.load(open(f"{O}/gate/err_D_S.json"))["sets"]["AL"]["online_offline_max_m"]
ok = same[0] == same[1] == 120 and ps == len(S) > 0 and oo is not None and oo < 1e-4
msg = f"A = AL {same[0]} / {same[1]}; plan_same {ps} / {len(S)}; online = offline max {oo} m -> {'OK' if ok else 'FAILED'}"
open(f"{O}/run/consistency.txt", "w").write(msg + "\n")
print(msg)
sys.exit(0 if ok else 1)
EOF
log "consistency: $(cat $O/run/consistency.txt)"
log "gate: $(grep '^GATE' $O/gate/gate.txt)"
grep -q "^GATE PASS" $O/gate/gate.txt || { log "GATE FAIL: no R_D (stop)"; exit 0; }
log "phase R start"
bash $S/ssd_run.sh R
grep -q "DONE R" $O/run/run.log || { log "FAIL phase R: no DONE R"; exit 1; }
bad=$(grep "RD .*): exit" $O/run/run.log | grep -vc "exit 0$")
log "phase R done: $bad nonzero exits"
bash $S/ssd_f1.sh > $O/f1.out 2>&1
$PYE $S/ssd_eval.py --run $O/run --main $MAIN --gate $O/gate --f1 $O/f1 --json $O/eval.json > $O/eval.txt 2>&1 || { log "FAIL eval"; exit 1; }
log "DONE chain"
