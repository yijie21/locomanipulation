"""curobo_avoid_20261120 step 3: the bench's rows with and without the planned path, per method, the pre-registered measures.

    python3 step3.py <rows dir> <straight direct rows dir> <via dir> TASK [--paired]

Rows: <rows dir>/{direct,model}_{avoid,straight}_<e>.jsonl (--trace); direct straight from <straight direct rows dir>/direct_<e>.jsonl
when the rows dir has none.
- stuck: the pregrasp settled > 20 cm from its target
- table: a right arm or hand link on a table at a pregrasp tick after the jaw is >= 3 cm from where the reach started (the hand
  starts on the table's edge in TabletopGrasp: contact at the start is not the reach's)
- dev: the jaw's largest distance from the planned path (the plan's jaw polyline, then on to the reach's own last command)
  over the pregrasp's ticks, avoid rows only
- paired: model - direct entry error per episode (both reached the entry), median, per condition
"""

import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

D, D0, VIA, TASK = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]


def load(method, cond):
    out = {}
    for f in glob.glob(str(D / f"{method}_{cond}_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["task"] == TASK:
                out[(str(r["level"]), int(r["index"]))] = r
    if not out and method == "direct" and cond == "straight":
        for f in glob.glob(str(D0 / "direct_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["task"] == TASK:
                    out[(str(r["level"]), int(r["index"]))] = r
    return out


def seg_dist(X, a, b):
    ab = b - a
    u = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(X - (a + u[:, None] * ab), axis=1)


def poly_dist(X, L):
    return np.min([seg_dist(X, a, b) for a, b in zip(L[:-1], L[1:])], axis=0)


def stats(xs):
    xs = [x for x in xs if x is not None]
    return (
        dict(n=len(xs), median=round(st.median(xs), 2), min=round(min(xs), 2), max=round(max(xs), 2))
        if xs
        else dict(n=0)
    )


R = {(m, c): load(m, c) for m in ("direct", "model") for c in ("straight", "avoid")}
summary, per = {}, {}
for (m, c), rows in R.items():
    if not rows:
        continue
    s = dict(n=len(rows), success_45=0, success_official=0, stuck=0, table=0, no_grasp=0, fell=0, why={}, knocked={}, dev=[], t_success=[],
             pre_err=[], entry_err=[], n_via=[])  # fmt: skip
    for k, r in sorted(rows.items()):
        st_ = {x["stage"]: x for x in r.get("stages", [])}
        pre = st_.get("pregrasp", {})
        s["success_45"] += bool(r["success_45"])
        s["success_official"] += bool(r["success_official"])
        if r.get("why"):
            s["no_grasp"] += 1
            continue
        s["fell"] += r.get("fell") is not None
        stuck = pre.get("err_cm") is not None and pre["err_cm"] > 20
        s["stuck"] += stuck
        s["pre_err"].append(pre.get("err_cm"))
        s["entry_err"].append(st_.get("entry", {}).get("err_cm"))
        if r["t_success"] is not None:
            s["t_success"].append(r["t_success"])
        kn = (r.get("knocked") or {}).get("phase")
        s["knocked"][str(kn)] = s["knocked"].get(str(kn), 0) + 1
        ticks = [t for t in r.get("trace", []) if t[1] == "pregrasp"]
        J = np.array([t[2] for t in ticks]) / 1000 if ticks else np.zeros((0, 3))
        hit = False
        if len(J):
            far = np.linalg.norm(J - J[0], axis=1) >= 0.03
            hit = any(f and any(n.startswith("right_") for n in t[5]) for f, t in zip(far, ticks))
        s["table"] += hit
        dev = None
        if c == "avoid" and len(J) and r.get("avoid", {}).get("n_via"):
            name = f"{TASK}_L{k[0]}_e{k[1]}"
            v = np.load(VIA / f"via_{name}.npz")
            P = np.load(Path(str(v["src"])) / f"plan_{name}.npz")
            last = np.asarray(pre["goal_mm"]) / 1000 if "goal_mm" in pre else P["jaw"][-1]
            L = np.vstack([P["jaw"], last])  # the plan, then the wanted pose (model's last command is shift_cm from it)
            dev = float(poly_dist(J, L).max())
            s["dev"].append(100 * dev)
            s["n_via"].append(r["avoid"]["n_via"])
        per[(m, c, k)] = dict(stuck=stuck, table=hit, s45=r["success_45"], pre=pre.get("err_cm"), entry=st_.get("entry", {}).get("err_cm"),
                              dev=None if dev is None else round(100 * dev, 1), kn=kn, t=r["t_success"])  # fmt: skip
    for key in ("dev", "t_success", "pre_err", "entry_err", "n_via"):
        s[key] = stats(s[key])
    summary[f"{m}_{c}"] = s
for k, v in summary.items():
    print(k, json.dumps(v))
# paired model - direct entry error, and the per-episode table
for c in ("straight", "avoid"):
    d_ = [per[("model", c, k)]["entry"] - per[("direct", c, k)]["entry"] for k in R[("direct", c)]
          if ("model", c, k) in per and ("direct", c, k) in per and per[("model", c, k)]["entry"] is not None
          and per[("direct", c, k)]["entry"] is not None]  # fmt: skip
    if d_:
        print(
            f"paired model - direct entry cm ({c}): n {len(d_)} median {round(st.median(d_), 2)} mean {round(float(np.mean(d_)), 2)}"
        )
if "--per" in sys.argv:
    for k in sorted({k for (_, _, k) in per}):
        print(k, {f"{m}_{c}": per.get((m, c, k)) for m in ("direct", "model") for c in ("straight", "avoid")})
