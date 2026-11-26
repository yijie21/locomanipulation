#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# Revision 4's pre-formal code check on one controller's seed-380 pairs (both tasks), read-only on the runs; prints no
# success, lift or landing number:  ht_codecheck.sh <controller> <arm S|T>
# Runs: $OT/check/<controller>_<task> (logs <dir>.log), from ht_formal.sh at 3x. Checks, each PASS / FAIL:
#   (1) each process printed DONE; no Traceback; no episode ended in error
#   (2) the pair's "prefix A == R: True"
#   (3) A's and R's episode logs exist; their model md5 = the chosen model.pt's; the reader's FDVEL True
#   (4) asrun2's ht_checks.sh on both runs (R's code = lc3_run's; features = data.load's), PYTHONDONTWRITEBYTECODE=1
#   (5) asrun2 = its MD5, no __pycache__ in it
set -uo pipefail
C=$1 X=$2
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
OT=$L/selfmodel_hatask_20261126
F2=$OT/asrun2
OH=$L/selfmodel_hadata_20261126; [ $C = sonic ] && OH=$L/selfmodel_hadata_sonic_20261126
MP=$OH/train/$C/models_${X}_s0/E3_ensemble_f1/model.pt
M8=$(md5sum < $MP | cut -c1-8)
export PYTHONDONTWRITEBYTECODE=1
ok=1
pf() { if [ "$2" = 1 ]; then echo "PASS $1"; else echo "FAIL $1"; ok=0; fi; }
echo "code check $C, R model $X ($MP, md5 $M8), $(date -u +%FT%TZ)"
for T in ppbox doubledesk; do
  D=$OT/check/${C}_$T; G=$D.log; n=${T}_s380_r0
  pf "$T: DONE printed" $(grep -a -c ": DONE" $G)
  pf "$T: no Traceback ($(grep -a -c "^Traceback" $G))" $([ "$(grep -a -c "^Traceback" $G)" = 0 ] && echo 1 || echo 0)
  pf "$T: no error end ($(grep -a -c "end='error" $G))" $([ "$(grep -a -c "end='error" $G)" = 0 ] && echo 1 || echo 0)
  pf "$T: $(grep -a -o "$n: prefix A == R: [A-Za-z]*" $G | tail -1)" $(grep -a -c "$n: prefix A == R: True" $G)
  for arm in A R; do
    r=$(python3 - "$D/$arm/log/$n.json" "$M8" <<'EOF'
import json, sys
p, m8 = sys.argv[1], sys.argv[2]
try:
    md = json.load(open(p))["model"] or {}
except FileNotFoundError:
    print(0, "no log"); sys.exit()
ok = md.get("model_md5") == m8 and md.get("flags", {}).get("FDVEL") is True
print(int(ok), f"model_md5 {md.get('model_md5')}, FDVEL {md.get('flags', {}).get('FDVEL')}")
EOF
)
    pf "$T $arm: log, ${r#* }" ${r%% *}
  done
done
echo "-- ht_checks.sh (asrun2):"
bash $F2/selfmodel_hacollect_20261126/ht_checks.sh $OT/check/${C}_ppbox,$OT/check/${C}_doubledesk $OT/check/${C}_checks
# ht_equiv's overall verdict fails by construction here (revision 4): it was written for the dev runs and requires the
# runs' reader FDVEL off (LocoManiLab's D) and >= 20 right-hand groups. Neither bears on R's arithmetic (the replay
# answers from the recorded predictions), so its parts are read: the text check, every group bit-identical, the
# synthetic shrink, and no FAIL line other than those two kinds.
r=$(python3 - $OT/check/${C}_checks_equiv.txt <<'EOF'
import json, re, sys
L = open(sys.argv[1]).read().splitlines()
t = json.loads(next(l for l in L if l.startswith("text:"))[5:])
m = re.search(r"right-hand groups: (\d+)/(\d+) .*left-hand (\d+)/(\d+); two-hand (\d+)/(\d+)", "\n".join(L))
g = [int(x) for x in m.groups()]
sy = json.loads(next(l for l in L if l.startswith("synthetic shrink:"))[17:])
other = [l for l in L if l.startswith("FAIL") and "had FDVEL on" not in l and "right-hand groups (the coordinator asked" not in l]
ok = all(t[k] for k in ("shrink", "clip_t", "block", "constants")) and g[0] == g[1] >= 1 and g[2] == g[3] and g[4] == g[5] \
    and sy["one_ok"] == sy["n"] and sy["two_ok"] == sy["n"] and sy["shrunk"] >= sy["n"] // 2 and not other
print(int(ok), f"text {t}; right {g[0]}/{g[1]}, left {g[2]}/{g[3]}, two-hand {g[4]}/{g[5]} bit-identical; synthetic {sy}; other FAIL lines {len(other)}")
EOF
)
pf "equivalence parts: ${r#* }" ${r%% *}
echo "   (ht_equiv's own verdict: $(tail -1 $OT/check/${C}_checks_equiv.txt); its FAIL lines: $(grep -c "^FAIL" $OT/check/${C}_checks_equiv.txt))"
f=$(tail -1 $OT/check/${C}_checks_feat.txt)
pf "features: $f" $(echo "$f" | grep -c "PASS")
pc=$(find $F2 -name __pycache__ | wc -l)
pf "asrun2 = MD5, __pycache__ dirs $pc" $( (cd $F2 && md5sum -c --quiet MD5) && [ $pc = 0 ] && echo 1 || echo 0)
[ $ok = 1 ] && echo "CODE CHECK PASS" || echo "CODE CHECK FAIL"
