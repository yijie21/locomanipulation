"""selfmodel_faildiag_20261126 revision 6: SONIC's arms tucked for the correction walk (T1a tuck1, T1b tuck1b), for
every walk (T2b tuck2b), the correction walk without the turn in place (N noturn, HG_SHORT_M=0.40) and N with the tucks
(noturn_tuck1, noturn_tuck1b), against the original episodes, paired by episode (question 1's 139 moved SONIC episode
runs; measurement only).

Original: question 1's hit-logged rerun (q1hit/, bit-identical to lmlcmp3). The groups: <root>/<group> (FD_HITLOG=2).
Per episode and arm, from fd_q1_moved.episode (the corrected definition): the object moved before the hand's arrival,
in place at the arrival (<= 2 cm and <= 15 deg off its look pose), its displacement at the arrival, the onset's phase
and hitter (1 s window); the outcome (hit_furniture among the reasons); the stance error (lc3's `arrive`: the simulator
pelvis at the first reach's schedule against the plan's stance), the first walk's end error (the pelvis at the first
command's wait against the stance) and the pelvis's travel over the correction walk; legs on the coffee table (any leg
body in contact with the object's support, sup_hits, from the look to the arrival); the first row where the run leaves
the original and the row its change starts (tuck1 / tuck1b / noturn*: the correction walk's wait; tuck2b: the look):
identical through that row is required; where the left wrist rode during the correction walk.
Then revision 6's verdicts, computed as registered (main measure: in place at the arrival).

    python fd_tuck_an.py --root <dir with the group dirs> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import fd_q1_moved as M
import fd_q2_an as Q

FD = M.FD
GROUPS = ("tuck1", "tuck1b", "tuck2b", "noturn", "noturn_tuck1", "noturn_tuck1b")
ARMS = ("orig",) + GROUPS
START_LOOK = ("tuck2b", "tuck2")  # changed from the look on; every other group from the correction walk's wait


def grp(b: str) -> str:
    if b == "none":
        return "no robot contact"
    if b.startswith("robot0_left_hand"):
        return "left hand"
    if b.startswith("robot0_right_hand"):
        return "right hand"
    if any(t in b for t in ("hip", "knee", "ankle")):
        return "left leg" if "left" in b else "right leg"
    if any(t in b for t in ("wrist", "elbow", "shoulder")):
        return "left arm" if "left" in b else "right arm"
    return "torso / pelvis" if b.startswith("robot0_") else b


def mcnemar(b: int, c: int) -> float:
    return Q.mcnemar(b, c)


def load(root: Path, task: str, method: str, ep: int):
    pj = root / "sonic" / "episodes" / task / method / f"ep{ep:+05d}.json"
    pz = root / "sonic" / "steplog" / task / method / f"ep{ep:+05d}.npz"
    if not pj.exists() or not pz.exists():
        return None, None
    return json.loads(pj.read_text()), np.load(pz)


def one(r: dict, z) -> dict:
    e = M.episode(r, z, True)
    rows = z["rows"]
    C = {str(c): i for i, c in enumerate(z["cols"])}
    st = (r.get("plan") or {}).get("stance") or {}
    out = dict(success=bool(r.get("success")), reason=r.get("reason"), why=r.get("why"), fell=r.get("fell"), n_walks=r.get("n_walks"),
               steps=len(rows), moved=e["moved"], at_arr_cm=e["at_arr_cm"], at_arr_deg=e["at_arr_deg"], f_arr=e["f_arr"],
               in_place=bool(e["at_arr_cm"] <= M.MOVE_CM and e["at_arr_deg"] <= M.MOVE_DEG), phase=e.get("phase"), onset=e.get("onset"),
               hitter=grp(next(iter(e.get("hit_bodies_1s") or {}), "none")) if e["moved"] else None,
               arrive_cm=(r.get("arrive") or {}).get("pos_err_cm"), arrive_deg=(r.get("arrive") or {}).get("yaw_err_deg"),
               tuck=r.get("fd_tuck"))  # fmt: skip
    out["hit_furniture"] = r.get("reason") == "hit_furniture"
    f_look, f_arr = M.F_LOOK0 + max(int(r.get("shift_f") or 0), 0), e["f_arr"]
    if "sup_hits" in z.files:
        names = json.loads(str(z["names"]))["bodies"]
        SH = np.asarray(z["sup_hits"], float).reshape(-1, 4)
        leg = [i for i, n in enumerate(names) if n.startswith("robot0_") and any(t in n for t in ("hip", "knee", "ankle"))]
        w_ = SH[(SH[:, 0] >= f_look) & (SH[:, 0] < (f_arr if f_arr is not None else len(rows))) & np.isin(SH[:, 1].astype(int), leg)]
        out["leg_table_rows"] = int(len(np.unique(w_[:, 0])))
    ws = [w for w in r.get("waits", []) if w.get("fire") is not None]
    if ws and st.get("xy") is not None:
        f1 = int(ws[0]["fire"])  # the first command's wait: where the first walk ended
        out["walk1_end_cm"] = float(100 * np.linalg.norm(rows[f1, [C["pel_px"], C["pel_py"]]] - np.asarray(st["xy"], float)))
    fixes = [w for w in r.get("waits", []) if w.get("then") == "correction walk" and w.get("fire") is not None]
    out["f_fix"] = int(fixes[0]["fire"]) if fixes else None
    # the left wrist during the correction walk (fix .. the first reach command), pelvis heading frame: forward, side, z
    c_first = [int(w["fire"]) for w in r.get("waits", []) if w.get("then") != "correction walk" and w.get("fire") is not None]
    if fixes and c_first:
        a_, b_ = out["f_fix"], c_first[0]
        out["fix_travel_cm"] = float(100 * np.linalg.norm(rows[b_, [C["pel_px"], C["pel_py"]]] - rows[a_, [C["pel_px"], C["pel_py"]]]))
        if b_ > a_ + 10:
            pel = rows[a_:b_][:, [C["pel_px"], C["pel_py"], C["pel_pz"]]]
            q = rows[a_:b_][:, [C["pel_qw"], C["pel_qx"], C["pel_qy"], C["pel_qz"]]] if "pel_qw" in C else None
            wl = rows[a_:b_][:, [C["wl_px"], C["wl_py"], C["wl_pz"]]]
            if q is not None:
                yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                d = wl - pel
                fwd = np.cos(yaw) * d[:, 0] + np.sin(yaw) * d[:, 1]
                side = -np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1]
                out["lw_walk"] = [float(np.median(fwd)), float(np.median(side)), float(np.median(wl[:, 2]))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    eps = [x for x in json.load(open(FD / "q1hit/an/q1_moved_hits.json")) if x.get("moved") and x["backend"] == "sonic" and "hit_bodies_1s" in x]
    per = []
    for x in sorted(eps, key=lambda x: (x["task"], x["method"], x["ep"])):
        task = Q.TN[x["task"]]
        row = dict(task=x["task"], method=x["method"], ep=x["ep"])
        ro, zo = load(FD / "q1hit" / "rerun", task, x["method"], x["ep"])
        row["orig"] = one(ro, zo)
        # the original's support log: q1hit/ has none (FD_HITLOG=1); q1hit2/ (FD_HITLOG=2, bit-identical) has 75 of them
        r2, z2 = load(FD / "q1hit2" / "rerun", task, x["method"], x["ep"])
        if r2 is not None:
            row["orig"]["leg_table_rows"] = one(r2, z2).get("leg_table_rows")
        for g in GROUPS:
            r, z = load(root / g, task, x["method"], x["ep"])
            if r is None:
                continue
            v = one(r, z)
            k = min(len(zo["rows"]), len(z["rows"]))
            dif = np.nonzero(np.any(zo["rows"][:k] != z["rows"][:k], axis=1))[0]
            v["first_diff"] = int(dif[0]) if len(dif) else (None if len(zo["rows"]) == len(z["rows"]) else k)
            # the row the change starts at: the correction walk's wait (tuck1*, noturn*), the look (tuck2b)
            v["change_row"] = M.F_LOOK0 + max(int(r.get("shift_f") or 0), 0) if g in START_LOOK else row["orig"]["f_fix"]
            v["same_before"] = v["first_diff"] is None or (v["change_row"] is not None and v["first_diff"] > v["change_row"])
            row[g] = v
        per.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuck_pairs.json").write_text(json.dumps(per, indent=0, default=float))
    pc = lambda v: np.percentile(v, [25, 50, 75]).round(1).tolist() if len(v) else None  # noqa: E731
    bad = {g: [(p["task"], p["method"], p["ep"], p[g]["first_diff"], p[g]["change_row"]) for p in per if g in p and not p[g]["same_before"]] for g in GROUPS}
    print("== identity before the change (required; a differing episode is left out and reported)")
    for g in GROUPS:
        n = sum(g in p for p in per)
        print(f"  {g:14s} {n - len(bad[g])}/{n} identical through the change row; differing: {bad[g][:10]}")
    ok = lambda p, g: g in p and p[g]["same_before"]  # noqa: E731
    for lab, S in (("dumbbell", [p for p in per if p["task"] == "dumbbell"]), ("wcan", [p for p in per if p["task"] == "wcan"]), ("both", per)):
        print(f"\n== {lab}: {len(S)} episode runs (methods {dict(Counter(p['method'] for p in S))})")
        for g in ARMS:
            G = [p[g] for p in S if g == "orig" or ok(p, g)]
            if not G:
                continue
            ar = [v["arrive_cm"] for v in G if v["arrive_cm"] is not None]
            w1 = [v["walk1_end_cm"] for v in G if v.get("walk1_end_cm") is not None]
            ft = [v["fix_travel_cm"] for v in G if v.get("fix_travel_cm") is not None]
            lw = [v["lw_walk"] for v in G if v.get("lw_walk")]
            print(f"  {g:14s} n {len(G)}: in place at the arrival {sum(v['in_place'] for v in G)} ({100 * sum(v['in_place'] for v in G) / len(G):.0f} %); moved before it {sum(v['moved'] for v in G)}; "
                  f"off at the arrival cm p25/50/75 {pc([v['at_arr_cm'] for v in G])}; success {sum(v['success'] for v in G)}; hit_furniture {sum(v['hit_furniture'] for v in G)}; reasons {dict(Counter(v['reason'] for v in G).most_common())}")  # fmt: skip
            print(f"                 moved: phase x hitter {dict(Counter((v['phase'], v['hitter']) for v in G if v['moved']).most_common(8))}")
            print(f"                 stance error at the first reach cm p25/50/75 {pc(ar)} (n {len(ar)}); first walk's end cm {pc(w1)}; pelvis travel over the correction walk cm {pc(ft)}; "
                  f"correction walks {dict(Counter(v['n_walks'] for v in G))}; fell {sum(v['fell'] is not None for v in G)}, lost {sum('lost' in str(v['why']) for v in G)}, "
                  f"timeout {sum(v['why'] == 'timeout' for v in G)}; legs on the coffee table (look .. arrival) {sum((v.get('leg_table_rows') or 0) > 0 for v in G)} of the {sum(v.get('leg_table_rows') is not None for v in G)} with the support log")  # fmt: skip
            if lw:
                L_ = np.array(lw)
                print(f"                 left wrist during the correction walk (from the root body, heading frame, median of medians): forward {np.median(L_[:, 0]) * 100:.1f} cm, side {np.median(L_[:, 1]) * 100:.1f} cm, z {np.median(L_[:, 2]):.3f} m (n {len(lw)})")
            if g != "orig":
                P = [p for p in S if ok(p, g)]
                b = sum(p[g]["in_place"] and not p["orig"]["in_place"] for p in P)
                c = sum(p["orig"]["in_place"] and not p[g]["in_place"] for p in P)
                bs = sum(p[g]["success"] and not p["orig"]["success"] for p in P)
                cs = sum(p["orig"]["success"] and not p[g]["success"] for p in P)
                da = [p[g]["arrive_cm"] - p["orig"]["arrive_cm"] for p in P if p[g]["arrive_cm"] is not None and p["orig"]["arrive_cm"] is not None]
                td = [p[g]["tuck"]["extra_f"] / 50 for p in P if p[g].get("tuck")]
                med = float(np.median(da)) if da else float("nan")
                worse, better = sum(d > 5 for d in da), sum(d < -5 for d in da)
                verdict = "stands further off" if (med >= 3 and worse > better) else "stands closer" if med <= -3 else "no difference seen"
                print(f"                 paired with orig: in place gained {b} / lost {c}, McNemar p {mcnemar(b, c):.3g}; success gained {bs} / lost {cs}, p {mcnemar(bs, cs):.3g}; inserted tuck s {dict(Counter(td))}")
                print(f"                 secondary (stance error at the first reach, {g} - orig, n {len(da)}): median {med:+.1f} cm, worse > 5 cm {worse}, better > 5 cm {better} -> {verdict}")
    # -- revision 6's verdicts (both tasks together, the registered denominators: the episode runs identical before the change)
    print("\n== revision 6 verdicts (main measure: in place at the arrival, 2 cm / 15 deg; both tasks)")

    def rate(g):
        G = [p for p in per if ok(p, g)]
        return (sum(p[g]["in_place"] for p in G) / len(G) if G else float("nan")), len(G)

    def pair(g1, g2):
        P = [p for p in per if ok(p, g1) and ok(p, g2)]
        b = sum(p[g1]["in_place"] and not p[g2]["in_place"] for p in P)
        c = sum(p[g2]["in_place"] and not p[g1]["in_place"] for p in P)
        d = (sum(p[g1]["in_place"] for p in P) - sum(p[g2]["in_place"] for p in P)) / len(P) if P else float("nan")
        return d, mcnemar(b, c), b, c, len(P)

    for t1 in ("tuck1", "tuck1b"):
        r1, n1 = rate(t1)
        lab = "effective" if r1 >= 0.5 else "partly effective" if r1 >= 0.2 else "not effective"
        print(f"  {t1}: in place {100 * r1:.1f} % of {n1} -> the tuck itself: {lab}")
    for t1, nt1 in (("tuck1", "noturn_tuck1"), ("tuck1b", "noturn_tuck1b")):
        rn, _ = rate("noturn")
        r1, _ = rate(t1)
        rnt, _ = rate(nt1)
        d_n1, p_n1, b1, c1, n_n1 = pair("noturn", t1)
        d_ntn, p_ntn, b2, c2, n_ntn = pair(nt1, "noturn")
        over = (rn >= 0.5 and r1 < 0.2) or (d_n1 >= 0.30 and p_n1 < 0.05)
        supp = (-d_n1 >= 0.30 and p_n1 < 0.05) and (d_ntn >= 0.30 and p_ntn < 0.05)
        both = rnt - max(rn, r1) >= 0.15
        v = "overturns the tuck: change the correction walk's turn" if over else "supports the tuck (the arm pose matters beyond the turn)" if supp else "neither alone decides"
        print(f"  T1 = {t1}: N {100 * rn:.1f} %, T1 {100 * r1:.1f} %, N + T1 {100 * rnt:.1f} %; N - T1 {100 * d_n1:+.1f} pp (only N {b1}, only T1 {c1}, n {n_n1}, p {p_n1:.3g}); "
              f"(N + T1) - N {100 * d_ntn:+.1f} pp (only N + T1 {b2}, only N {c2}, n {n_ntn}, p {p_ntn:.3g}) -> {v}" + ("; N + T1 beats the better of N and T1 by >= 15 pp" if both else ""))  # fmt: skip


if __name__ == "__main__":
    main()
