#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- REVISION 2 (2026-11-27, after the first GATE FAIL, before any R_D
# episode): steps 4-7 of the frozen ssd_chain.sh (da033561), run from the frozen copy $O/asrun/code2 (= asrun/code with
# one line of reachwm_20261030/data.py changed: x0 column 204, walked, set to 0) through asrun/rev2's ssd_gate.sh,
# ssd_run.sh and ssd_f1.sh (the frozen scripts with their code dir changed). Steps 1-3 (training, phase A, the
# consistency) are not rerun: D_S is not retrained and the AL records are reused (A = AL 120 / 120 and plan_same 240 / 240
# hold as before; AL's online = offline cannot hold any more, its online predictions were made with walked = 1: logged,
# not a stop). The online = offline check (sh_err.py's, < 1e-4 m) is done on R_D instead, after phase R; a failure stops
# the chain before the evaluation. Log: $O/chain_rev2.log; the gate's out dir $O/gate_rev2 (gate/ keeps the first one).
#   ssd_chain.sh
# 4 the gate (rev2/ssd_gate.sh); GATE FAIL stops the chain; GATE PASS: 5 phase R (rev2/ssd_run.sh R: R_D, 120
# episodes), 6 F1 (rev2/ssd_f1.sh) and R_D's online = offline, 7 the evaluation (ssd_eval.py -> $O/eval.txt, eval.json).
set -uo pipefail
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs
O=$L/selfmodel_simplesonicdata_20261126
R=$O/asrun/code2 S=$R/scripts/experiments/selfmodel_simplesonicdata_20261126 V=$O/asrun/rev2
MAIN=$Y/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/sonic
PYE="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
C=$O/chain_rev2.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $C; }
log "chain rev2 start (code $R)"
(cd $R && md5sum -c --quiet CODE_MD5) || { log "FAIL: code2 differs from its CODE_MD5"; exit 1; }
bash $V/ssd_gate.sh > $O/gate_rev2.out 2>&1
grep -q "^GATE" $O/gate_rev2/gate.txt || { log "FAIL gate: no verdict"; exit 1; }
oo=$(python3 -c "import json; print(json.load(open('$O/gate_rev2/err_D_S.json'))['sets']['AL']['online_offline_max_m'])")
log "gate: $(grep '^GATE' $O/gate_rev2/gate.txt); AL online = offline max $oo m (AL's online predictions: walked = 1; not a stop)"
grep -q "^GATE PASS" $O/gate_rev2/gate.txt || { log "GATE FAIL: no R_D (stop)"; exit 0; }
log "phase R start"
bash $V/ssd_run.sh R
grep -q "DONE R" $O/run/run.log || { log "FAIL phase R: no DONE R"; exit 1; }
bad=$(grep "RD .*): exit" $O/run/run.log | grep -vc "exit 0$")
log "phase R done: $bad nonzero exits"
[ $bad = 0 ] || { log "STOP: nonzero exits in phase R"; exit 1; }
bash $V/ssd_f1.sh > $O/f1.out 2>&1
python3 - $O <<'PY' || { log "STOP: R_D's consistency failed (see run/consistency_rev2.txt)"; exit 1; }
import glob, json, sys
O = sys.argv[1]
S = [s for f in glob.glob(f"{O}/run/corr_RD/log/*.jsonl") for line in open(f) for s in json.loads(line)["stages_correct"]]
ps = sum(s.get("plan_same") is True for s in S)
oo = json.load(open(f"{O}/f1/err_RD.json"))["sets"]["RD"]["online_offline_max_m"]
ok = len(S) > 0 and oo is not None and oo < 1e-4
msg = f"R_D: {len(S)} stages, plan_same {ps}; online = offline max {oo} m -> {'OK' if ok else 'FAILED'}"
open(f"{O}/run/consistency_rev2.txt", "w").write(msg + "\n")
print(msg)
sys.exit(0 if ok else 1)
PY
log "consistency: $(cat $O/run/consistency_rev2.txt)"
$PYE $S/ssd_eval.py --run $O/run --main $MAIN --gate $O/gate_rev2 --f1 $O/f1 --json $O/eval.json > $O/eval.txt 2>&1 || { log "FAIL eval"; exit 1; }
log "DONE chain"
