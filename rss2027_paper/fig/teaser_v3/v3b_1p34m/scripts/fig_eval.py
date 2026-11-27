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
import math
import sys
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1])
THR = 2.0
PANEL = "after3500ms"  # v3b: the panels' moment, 3.50 s after the close command
TILT = 30.0  # deg: knocked over
MIN_CLASS = 10  # v3b: fewer "A miss R hit" -> stop, no final


def tilt_deg(q):
    w, x, y, _ = (float(v) for v in q)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


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
            v0 = z["pregrasp_end__vase_pos"]  # the vase at the pregrasp's end, before the entry (its tilt there: tilt@pre, printed)
            pan = {}
            if f"{PANEL}__t" in z.files:
                pan = dict(dt=float(z[f"{PANEL}__t"] - z["entry_end__t"]), tilt_deg=tilt_deg(z[f"{PANEL}__vase_quat"]),
                           vase_rise_cm=100 * float(z[f"{PANEL}__vase_pos"][2] - v0[2]),
                           palm_to_vase_cm=100 * float(np.linalg.norm(z[f"{PANEL}__palm"] - z[f"{PANEL}__vase_pos"])))
            else:
                bad.append((a, i, "no panel snapshot"))
            e[a] = dict(tilt_close_deg=tilt_deg(z["entry_end__vase_quat"]), tilt_pre_deg=tilt_deg(z["pregrasp_end__vase_quat"]),
                        knocked_over=bool(pan.get("tilt_deg", 0.0) > TILT), panel=pan,
                        d_close=d_close, d_close_end=d_end, success_lift=r["success_end"], fell=r["fell"], walk_m=walk,
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
        print(f"  {c}: {sum(per[i]['class'] == c for i in idx)} {[i for i in idx if per[i]['class'] == c]}")
    print(f"\nvase at the panel moment ({PANEL}; dt from the close command per arm), knocked = tilt > {TILT} deg; lifted = "
          f"simple_bench success_end")
    print(f"{'p':>3} | A tilt@pre close panel  rise cm | R tilt@pre close panel  rise cm | A lifted R lifted")
    for i in idx:
        a, r = per[i]["A"], per[i]["R"]
        print(f"{i:3d} | {a['tilt_pre_deg']:5.1f} {a['tilt_close_deg']:5.1f} {a['panel'].get('tilt_deg', -1):5.1f} "
              f"{a['panel'].get('vase_rise_cm', -99):6.1f} | {r['tilt_pre_deg']:5.1f} {r['tilt_close_deg']:5.1f} "
              f"{r['panel'].get('tilt_deg', -1):5.1f} {r['panel'].get('vase_rise_cm', -99):6.1f} | {str(a['success_lift']):6s} {r['success_lift']}")
    cnt = {}
    for a in ("A", "R"):
        cnt[a] = dict(knocked_panel=[i for i in idx if per[i][a]["knocked_over"]],
                      knocked_at_close=[i for i in idx if per[i][a]["tilt_close_deg"] > TILT],
                      lifted=[i for i in idx if per[i][a]["success_lift"]],
                      panel_dt=sorted({round(per[i][a]["panel"].get("dt", -1), 6) for i in idx}))
        print(f"  {a}: knocked over at the panel {len(cnt[a]['knocked_panel'])} {cnt[a]['knocked_panel']}; already at the close "
              f"command {len(cnt[a]['knocked_at_close'])}; lifted {len(cnt[a]['lifted'])} {cnt[a]['lifted']}; panel dt {cnt[a]['panel_dt']}")
    K = [i for i in idx if per[i]["class"] == "A miss R hit"]
    out = dict(threshold_cm=THR, panel=PANEL, tilt_deg=TILT, vase_counts=cnt, placements=per, checks=bad, cls_counts={c: sum(per[i]["class"] == c for i in idx) for c in
                                                                         ("A miss R hit", "both hit", "both miss", "A hit R miss")})
    if len(K) < MIN_CLASS:
        print(f"\nonly {len(K)} placements with A > 2 cm and R <= 2 cm (< {MIN_CLASS}): stop, report, no final.")
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
