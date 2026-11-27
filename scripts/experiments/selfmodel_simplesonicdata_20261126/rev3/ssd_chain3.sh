#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- REVISION 3 (2026-11-27, before either repeat run): the two repeat runs
# on H20b from asrun/code2 (walked set to 0; D_S, R rule, episodes, seeds, levels and every criterion unchanged), one
# after the other, then the three runs merged:
#   G  SSD_NETS=gpu SSD_THREADS=1      (SONIC's three nets on the job's card, CUDA)  -> $O/run_g, gate_g, f1_g, eval_g.*
#   T  SSD_NETS=allcpu SSD_THREADS=2   (all on the CPU, 2 onnxruntime threads)       -> $O/run_t, gate_t, f1_t, eval_t.*
# Each run, as the frozen chain's steps 2-7 (ssd_chain.sh da033561) with rev3/ssd_run.sh, rev2/ssd_gate.sh, rev2/ssd_f1.sh:
# phase A (A, AL) -> every job exit 0 -> ssd_eval.py (eval_A) -> the gate (reported; not a stop: revision 3 runs R_D in
# every run) -> the consistency (A = AL 120 / 120, plan_same on every AL stage, AL online = offline < 1e-4 m; and A not
# identical to run C's A on all 120 episodes: the stages compared, else the float path did not change) -> phase R (R_D)
# -> every job exit 0 -> F1 -> R_D online = offline < 1e-4 m -> ssd_eval.py. Any failure stops the chain there.
# Then rev3/ssd_merge.py C=run G=run_g T=run_t -> $O/merge.txt, merge.json. Log: $O/chain_rev3.log.
set -uo pipefail
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs
O=$L/selfmodel_simplesonicdata_20261126
R=$O/asrun/code2 S=$R/scripts/experiments/selfmodel_simplesonicdata_20261126 V2=$O/asrun/rev2 V3=$O/asrun/rev3
MAIN=$Y/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/sonic
PYE="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
PYV=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python  # for the inline checks (numpy only)
C=$O/chain_rev3.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $C; }
(cd $R && md5sum -c --quiet CODE_MD5) || { log "FAIL: code2 differs from its CODE_MD5"; exit 1; }
one() {
  local tag=$1 nets=$2 thr=$3
  local RUN=$O/run_$tag G=$O/gate_$tag F=$O/f1_$tag
  [ ! -e $RUN ] || { log "FAIL $tag: $RUN exists"; return 1; }
  log "run $tag start (nets $nets, threads $thr)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/nvidia_smi_$tag.txt
  SSD_RUN=$RUN SSD_NETS=$nets SSD_THREADS=$thr bash $V3/ssd_run.sh A
  grep -q "DONE A" $RUN/run.log || { log "FAIL $tag phase A: no DONE A"; return 1; }
  local bad; bad=$(grep "): exit" $RUN/run.log | grep -vc "exit 0$")
  log "$tag phase A done: $(grep -c '): exit' $RUN/run.log) jobs, $bad nonzero exits"
  [ $bad = 0 ] || { log "STOP $tag: nonzero exits in phase A"; return 1; }
  $PYE $S/ssd_eval.py --run $RUN --main $MAIN --json $RUN/eval_A.json > $RUN/eval_A.txt 2>&1 || { log "FAIL $tag eval_A"; return 1; }
  bash $V2/ssd_gate.sh $RUN $G > $O/gate_$tag.out 2>&1
  grep -q "^GATE" $G/gate.txt || { log "FAIL $tag gate: no verdict"; return 1; }
  $PYV - $O $tag $S <<'PY' || { log "STOP $tag: the consistency failed (see run_$tag/consistency.txt)"; return 1; }
import glob, json, sys
O, tag, S = sys.argv[1:4]
sys.path.insert(0, f"{S}/../selfmodel_simplecmp_20261126")
import scmp_eval as SE
RUN = f"{O}/run_{tag}"
same = json.load(open(f"{RUN}/eval_A.json"))["same_A_AL"]
st = [s for f in glob.glob(f"{RUN}/corr_AL/log/*.jsonl") for line in open(f) for s in json.loads(line)["stages_correct"]]
ps = sum(s.get("plan_same") is True for s in st)
oo = json.load(open(f"{O}/gate_{tag}/err_D_S.json"))["sets"]["AL"]["online_offline_max_m"]
from pathlib import Path
a, c = SE.rows_of(Path(RUN) / "rows", "A"), SE.rows_of(Path(O) / "run" / "rows", "A")
ident = sum(SE.clean(a[k])["stages"] == SE.clean(c[k])["stages"] for k in a if k in c)
ok = same[0] == same[1] == 120 and ps == len(st) > 0 and oo is not None and oo < 1e-4 and ident < 120
msg = (f"{tag}: A = AL {same[0]} / {same[1]}; plan_same {ps} / {len(st)}; AL online = offline max {oo} m; "
       f"A's stages identical to run C's on {ident} / {len(a)} episodes -> {'OK' if ok else 'FAILED'}")
open(f"{RUN}/consistency.txt", "w").write(msg + "\n")
print(msg)
sys.exit(0 if ok else 1)
PY
  log "consistency: $(cat $RUN/consistency.txt)"
  log "$tag gate (reported, not a stop): $(grep '^GATE' $G/gate.txt)"
  log "$tag phase R start"
  SSD_RUN=$RUN SSD_NETS=$nets SSD_THREADS=$thr bash $V3/ssd_run.sh R
  grep -q "DONE R" $RUN/run.log || { log "FAIL $tag phase R: no DONE R"; return 1; }
  bad=$(grep "RD .*): exit" $RUN/run.log | grep -vc "exit 0$")
  log "$tag phase R done: $bad nonzero exits"
  [ $bad = 0 ] || { log "STOP $tag: nonzero exits in phase R"; return 1; }
  bash $V2/ssd_f1.sh $RUN $F > $O/f1_$tag.out 2>&1
  python3 - $RUN $F <<'PY' || { log "STOP $tag: R_D's consistency failed (see run_$tag/consistency_rd.txt)"; return 1; }
import glob, json, sys
RUN, F = sys.argv[1:3]
st = [s for f in glob.glob(f"{RUN}/corr_RD/log/*.jsonl") for line in open(f) for s in json.loads(line)["stages_correct"]]
ps = sum(s.get("plan_same") is True for s in st)
oo = json.load(open(f"{F}/err_RD.json"))["sets"]["RD"]["online_offline_max_m"]
ok = len(st) > 0 and oo is not None and oo < 1e-4
msg = f"R_D: {len(st)} stages, plan_same {ps}; online = offline max {oo} m -> {'OK' if ok else 'FAILED'}"
open(f"{RUN}/consistency_rd.txt", "w").write(msg + "\n")
print(msg)
sys.exit(0 if ok else 1)
PY
  log "$tag consistency: $(cat $RUN/consistency_rd.txt)"
  $PYE $S/ssd_eval.py --run $RUN --main $MAIN --gate $G --f1 $F --json $O/eval_$tag.json > $O/eval_$tag.txt 2>&1 || { log "FAIL $tag eval"; return 1; }
  log "run $tag done"
}
log "chain rev3 start (code $R)"
one g gpu 1 || exit 1
one t allcpu 2 || exit 1
SSD_MERGE_JSON=$O/merge.json $PYE $V3/ssd_merge.py C=$O/run:$O/gate_rev2 G=$O/run_g:$O/gate_g T=$O/run_t:$O/gate_t > $O/merge.txt 2>&1 || { log "FAIL merge"; exit 1; }
log "merge: $(grep -h 'REPRODUCED\|NOT reproduced' $O/merge.txt | tr '\n' ';')"
log "DONE chain3"
