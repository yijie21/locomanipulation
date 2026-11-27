"""selfmodel_faildiag_20261126 question 1: fd_q1_an.py's per-episode table crossed with fd_q1_moved.py's (object left its
look pose before the hand arrived): per controller / method / task, outcome and missed mechanism split by moved; the
close distance and orientation error among the clean closes whose object did not move.

    python fd_q1_cross.py <q1 dir with q1_episodes.json and q1_moved.json>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
E = {(x["backend"], x["method"], x["task"], x["ep"]): x for x in json.loads((d / "q1_episodes.json").read_text())}
M = {(x["backend"], x["method"], x["task"], x["ep"]): x for x in json.loads((d / "q1_moved.json").read_text())}


def q(v, p=50):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return "nan" if not v else f"{np.percentile(v, p):.1f}"


for be, m in (("sonic", "alone"), ("sonic", "selfmodel"), ("sonic", "reachable"), ("scalebfm", "alone"), ("scalebfm", "reachable")):
    for tn in ("dumbbell", "wcan"):
        K = [k for k in E if k[:3] == (be, m, tn)]
        if not K:
            continue
        mv = [k for k in K if M.get(k, {}).get("moved")]
        st = [k for k in K if k not in mv]
        print(f"== {be} {m} {tn}: {len(K)} episodes; object moved before the hand {len(mv)}")
        for nm, S in (("moved", mv), ("not moved", st)):
            ok = sum(E[k]["success"] for k in S)
            cl = [k for k in S if E[k]["clean"]]
            print(f"   {nm:9s} {len(S):3d}: success {ok}; reasons {dict(Counter(E[k]['reason'] for k in S).most_common())}; "
                  f"clean closes {len(cl)} (ok {sum(E[k]['reason'] == 'ok' for k in cl)}), missed mechanism {dict(Counter(E[k].get('mech') for k in cl if E[k]['reason'] == 'missed').most_common())}")  # fmt: skip
            if nm == "not moved" and cl:
                for r in ("ok", "missed"):
                    C = [E[k] for k in cl if E[k]["reason"] == r]
                    if C:
                        print(f"      {r:6s} n {len(C)}: close dist p50 {q([c.get('close_dist_cm') for c in C])} cm (p90 {q([c.get('close_dist_cm') for c in C], 90)}), rot err p50 {q([c.get('rot_err_deg') for c in C])} deg, "
                              f"object from the look center p50 {q([c.get('obj_look_err_cm') for c in C])} cm, touched {sum(bool(c.get('touched')) for c in C)}, pushed p50 {q([c.get('obj_push_cm') for c in C])} cm")  # fmt: skip
print("== success rate if the moved episodes are left out (not moved only):")
for be, m in (("sonic", "alone"), ("sonic", "selfmodel"), ("sonic", "reachable"), ("scalebfm", "alone"), ("scalebfm", "reachable")):
    for tn in ("dumbbell", "wcan"):
        K = [k for k in E if k[:3] == (be, m, tn)]
        st = [k for k in K if not M.get(k, {}).get("moved")]
        print(f"   {be:8s} {m:9s} {tn:8s}: all {sum(E[k]['success'] for k in K)}/{len(K)}; not moved {sum(E[k]['success'] for k in st)}/{len(st)}")
