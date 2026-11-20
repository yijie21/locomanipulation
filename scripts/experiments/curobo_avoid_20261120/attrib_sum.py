"""curobo_attrib_20261120: the groups' measures from attrib/rows/<G>_<side-off>_<e>.jsonl.

    python3 attrib_sum.py <rows dir>

Per group (84 = 28 episodes x 3 stances) and per stance: stuck (the pregrasp settled > 20 cm from its target), 45 s and 16 s
success, table (a right-side link on a table at a pregrasp tick after the jaw is >= 3 cm from its start: episodes, ticks), the
pregrasp's jaw path length and duration (medians). Paired against A0 and A1 (same episode and stance): stuck in one only.
"""

import glob
import json
import os
import statistics as st
import sys
from math import comb

D = sys.argv[1]
rows = {}
for f in glob.glob(os.path.join(D, "*.jsonl")):
    G, so, e = os.path.basename(f)[:-6].split("_")
    for line in open(f):
        rows[(G, so, int(e))] = json.loads(line)


def measures(r):
    s = {x["stage"]: x for x in r.get("stages", [])}
    pre = s.get("pregrasp", {})
    tk = [t for t in r.get("trace", []) if t[1] == "pregrasp"]
    J = [t[2] for t in tk]
    dist = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 / 1000  # noqa: E731
    hits = sum(1 for t in tk if dist(t[2], J[0]) >= 0.03 and any(n.startswith("right_") for n in t[5])) if tk else 0
    return dict(stuck=pre.get("err_cm") is not None and pre["err_cm"] > 20, s45=bool(r["success_45"]), s16=bool(r["success_official"]),
                table=hits > 0, hits=hits, path=sum(dist(a, b) for a, b in zip(J[:-1], J[1:])) if J else None,
                dur=tk[-1][0] - tk[0][0] if tk else None, pre=pre.get("err_cm"), n_via=(r.get("avoid") or {}).get("n_via"))  # fmt: skip


M = {k: measures(r) for k, r in rows.items()}
G_ = sorted({k[0] for k in M})
S_ = sorted({k[1] for k in M})


def binom_p(a, b):  # two-sided exact sign test on discordant pairs
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


for G in G_:
    ks = [k for k in M if k[0] == G]
    m = [M[k] for k in ks]
    per = {so: sum(M[k]["stuck"] for k in ks if k[1] == so) for so in S_}
    line = dict(n=len(m), stuck=sum(x["stuck"] for x in m), stuck_by_stance=per, s45=sum(x["s45"] for x in m), s16=sum(x["s16"] for x in m),
                table=sum(x["table"] for x in m), hits_median=st.median(x["hits"] for x in m),
                path_cm_median=round(100 * st.median(x["path"] for x in m if x["path"] is not None), 1),
                dur_s_median=round(st.median(x["dur"] for x in m if x["dur"] is not None), 2))  # fmt: skip
    for ref in ("A0", "A1"):
        if ref == G:
            continue
        a = sum(1 for k in ks if M[k]["stuck"] and not M.get((ref, k[1], k[2]), {}).get("stuck", True))
        b = sum(1 for k in ks if not M[k]["stuck"] and M.get((ref, k[1], k[2]), {}).get("stuck", False))
        line[f"vs_{ref}"] = f"only {G} stuck {a}, only {ref} stuck {b}, p {binom_p(a, b):.2g}"
    print(G, json.dumps(line, ensure_ascii=False))
if "--per" in sys.argv:
    for e in sorted({k[2] for k in M}):
        print(
            e, " ".join(f"{G}:" + "".join("X" if M.get((G, so, e), {}).get("stuck") else "." for so in S_) for G in G_)
        )
