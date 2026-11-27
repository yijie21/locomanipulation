"""fig_eval.py <runs dir>: Figure 1 v3's counts and the pick (README section 3's rules). Read only. notebook: skip figure
render, not an experiment.

Per placement and arm: the close distance = |palm - grasp point| when simple_bench sends the close (the entry's end,
after its POST_S hold: the snapshot "entry_end"; equal to the row's palm_cm, checked), the close-end distance (fingers
closed + 0.5 s, before the lift: "close_end", reference only), the walk (none), and per reach (pregrasp, entry) D's
predicted palm landing for the command sent and the measured landing (world). Class: A > 2 cm and R <= 2 cm. Pick:
percentiles inside the class per arm (kind='mean'), score max(|pA-50|, |pR-50|), the least, ties the lower index.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1])
THR = 2.0


def rows(arm):
    out = {}
    for f in sorted(glob.glob(str(RUNS / "rows" / f"{arm}_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[int(r["index"])] = r
    return out


def pct(vals, v):
    vals = np.asarray(vals, float)
    return 100.0 * ((vals < v).sum() + 0.5 * (vals == v).sum()) / len(vals)


def main():
    R = {a: rows(a) for a in ("A", "R")}
    idx = sorted(set(R["A"]) & set(R["R"]))
    print(f"rows: A {len(R['A'])}, R {len(R['R'])}, paired {len(idx)}")
    per = {}
    bad = []
    for i in idx:
        e = {}
        for a in ("A", "R"):
            r = R[a][i]
            z = np.load(RUNS / "snap" / f"{a}_p{i:02d}.npz")
            g = z["goal__entry"]
            d_close = 100 * float(np.linalg.norm(z["entry_end__palm"] - g))
            d_end = 100 * float(np.linalg.norm(z["close_end__palm"] - g))
            if abs(d_close - r["palm_cm"]) > 6e-4:
                bad.append((a, i, "palm_cm", d_close, r["palm_cm"]))
            reaches = []
            for s in r["stages"]:
                c = s.get("correct", {})
                st = s["stage"]
                land_w = z[f"{st}_end__palm"]
                reaches.append(dict(stage=st, goal_w=z[f"goal__{st}"].tolist(), cmd_w=c.get("cmd_w"), corr_cm=c.get("corr_cm"),
                                    pred_palm_w=c.get("pred_wp"), land_palm_w=land_w.tolist(), err_cm=s["err_cm"],
                                    pred_err_cm=c.get("pred_err_palm_cm"), pred_miss_cm=c.get("pred_miss_palm_cm"),
                                    plan_same=c.get("plan_same"), in_table=c.get("in_table"), ik_cm=s.get("ik_cm"),
                                    think_s=c.get("think_s")))
                if c.get("plan_same") is not True:
                    bad.append((a, i, st, "plan_same", c.get("plan_same")))
            walk = float(np.linalg.norm(r["walk"]))
            if walk > 0.02:
                bad.append((a, i, "walk", walk))
            e[a] = dict(d_close=d_close, d_close_end=d_end, success_lift=r["success_end"], fell=r["fell"], walk_m=walk,
                        hits=r["hits"], touched=r["touched"], knocked=r["knocked"], reaches=reaches, goal_entry=g.tolist(),
                        palm_close=z["entry_end__palm"].tolist())
        per[i] = e
    print("checks:", "all pass" if not bad else bad)
    print(f"\n{'p':>3} {'A cm':>7} {'R cm':>7} | {'A end':>6} {'R end':>6} | A lift R lift | class | R corr pre/entry cm | "
          f"D pred err A pre/entry | R pre/entry")
    for i in idx:
        a, r = per[i]["A"], per[i]["R"]
        cls = "A miss R hit" if a["d_close"] > THR and r["d_close"] <= THR else (
            "both hit" if a["d_close"] <= THR and r["d_close"] <= THR else ("both miss" if a["d_close"] > THR else "A hit R miss"))
        per[i]["class"] = cls
        pe = lambda x: "/".join(f"{w['pred_err_cm']:.2f}" for w in x["reaches"])  # noqa: E731
        print(f"{i:3d} {a['d_close']:7.3f} {r['d_close']:7.3f} | {a['d_close_end']:6.2f} {r['d_close_end']:6.2f} | "
              f"{str(a['success_lift']):6s} {str(r['success_lift']):6s} | {cls:12s} | "
              f"{'/'.join(str(w['corr_cm']) for w in r['reaches'])} | {pe(a)} | {pe(r)}")
    dA = np.array([per[i]["A"]["d_close"] for i in idx])
    dR = np.array([per[i]["R"]["d_close"] for i in idx])
    print(f"\nall {len(idx)}: A median {np.median(dA):.3f} cm, > {THR} cm {int((dA > THR).sum())}; R median {np.median(dR):.3f} cm, "
          f"<= {THR} cm {int((dR <= THR).sum())}")
    for c in ("A miss R hit", "both hit", "both miss", "A hit R miss"):
        print(f"  {c}: {sum(per[i]['class'] == c for i in idx)}")
    K = [i for i in idx if per[i]["class"] == "A miss R hit"]
    out = dict(threshold_cm=THR, placements=per, checks=bad, cls_counts={c: sum(per[i]["class"] == c for i in idx) for c in
                                                                         ("A miss R hit", "both hit", "both miss", "A hit R miss")})
    if not K:
        print("\nNO placement with A > 2 cm and R <= 2 cm: stop, report.")
        out["pick"] = None
    else:
        a_ = [per[i]["A"]["d_close"] for i in K]
        r_ = [per[i]["R"]["d_close"] for i in K]
        rank = []
        for i in K:
            pA, pR = pct(a_, per[i]["A"]["d_close"]), pct(r_, per[i]["R"]["d_close"])
            rank.append(dict(index=i, A=per[i]["A"]["d_close"], R=per[i]["R"]["d_close"], pA=pA, pR=pR,
                             score=max(abs(pA - 50), abs(pR - 50))))
        rank.sort(key=lambda x: (x["score"], x["index"]))
        print(f"\nclass 'A miss R hit': {len(K)} placements {K}; class medians A {np.median(a_):.3f} R {np.median(r_):.3f}")
        print(f"{'rank':>4} {'p':>3} {'A cm':>7} {'R cm':>7} {'pA':>6} {'pR':>6} {'score':>6}")
        for k, x in enumerate(rank):
            print(f"{k + 1:4d} {x['index']:3d} {x['A']:7.3f} {x['R']:7.3f} {x['pA']:6.2f} {x['pR']:6.2f} {x['score']:6.2f}")
        out["rank"] = rank
        out["pick"] = rank[0]["index"]
        print(f"\npick: placement {rank[0]['index']}")
    json.dump(out, open(RUNS / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
