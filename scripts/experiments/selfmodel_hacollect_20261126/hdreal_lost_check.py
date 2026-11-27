"""hadata revision 2c, after eval 3 (eval_code2): the coordinator's two checks on HEFT P&PBox's 7 lost episodes.
  1 their last segments (the reach each was lost in) are not in the merged file eval 3 read;
  2 their earlier segments are: per episode the merged chains and, per segment, how eval 3's filters treated it
    (kept = t3b_eval.kept: not fallen / lost and the end label settled; touching = revision 1's rule; clean = kept and
    not touching); totals: merged, clean, and left out by each filter.
Read-only; hd_real's own DT_kept and the frozen hd_touch rule, from eval_code2.
    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo_b> \
        $PY hdreal_lost_check.py <collection root>
"""

import json
import sys
from pathlib import Path

import numpy as np

OH = Path(sys.argv[1])
sys.path.insert(0, str(OH / "eval_code2" / "selfmodel_hacollect_20261126"))
import hd_real as H  # noqa: E402

TOUCH = H.load_by_path("hd_touch", OH / "asrun_train/selfmodel_hacollect_20261126/hd_touch.py", "5526f05b")
EV = OH / "train/heft/eval"
f = EV / "real/heft_ppbox/data/all.npz"
z = dict(np.load(f))
D = H.DT.load(str(f))
N = len(D["n_steps"])
rs, ch = z["reset"].astype(int), z["chain"].astype(int)
kept = np.zeros(N, bool)
kept[H.DT_kept(D, np.arange(N))] = True
touch = TOUCH.touching(z)
fell = np.asarray(D["fell"]) != 0
se = np.asarray(D["settled_end"]) > 0
print(f"merged P&PBox file {f}: {N} segments")
for ln in (EV / "eval3.txt").read_text().splitlines():
    if ln.startswith("segments ") or ln.startswith("REAL MERGE") or "segments left out" in ln:
        print("eval3.txt: " + ln)
tot = dict(merged=0, kept=0, clean=0, fallen_or_lost=0, not_settled=0, touching_kept=0)
absent = 0
for s in (208, 210, 211, 217, 225, 229, 231):
    lg = json.loads((OH / "real/heft_ppbox/log" / f"ppbox_s{s}_r0.json").read_text())
    r, cut = int(lg["reset"]), len(lg["reaches"])
    m = rs == r
    gone = not ((rs == r) & (ch == cut)).any()
    absent += gone
    rows = []
    for i in np.flatnonzero(m):
        why = "clean" if kept[i] and not touch[i] else "kept, touching" if kept[i] else "fallen / lost" if fell[i] else "not settled" if not se[i] else "?"
        rows.append(f"chain {ch[i]} {why}")
        tot["merged"] += 1
        tot["kept"] += int(kept[i])
        tot["clean"] += int(kept[i] and not touch[i])
        tot["fallen_or_lost"] += int(fell[i])
        tot["not_settled"] += int(not fell[i] and not se[i])
        tot["touching_kept"] += int(kept[i] and touch[i])
    print(f"ppbox_s{s}_r0 (reset {r}, end {lg['end']}, {cut} logged reaches): lost reach chain {cut} in the merged file: {not gone}; merged: " + "; ".join(rows))
print(f"1 the 7 lost reaches absent from the merged file: {absent}/7")
print(f"2 the 7 episodes' earlier segments merged {tot['merged']}: clean {tot['clean']}; left out by the filters: fallen / lost "
      f"{tot['fallen_or_lost']}, not settled {tot['not_settled']}, kept but touching {tot['touching_kept']} (kept {tot['kept']})")
print("LOST CHECK", "PASS" if absent == 7 and tot["merged"] == 17 else "FAIL")
