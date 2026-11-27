# Figure 1 teaser: choose the episode by the coordinator's rule (read-only over lmlroute's formal records).
# Run on a machine with vePFS:  python3 - < select.py   (stdlib only)
import json
import os

O = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126"
BE = "scalebfm"
# lrt_eval E1 for scalebfm: decision set = candle, apple, salt, statue_cab, dumbbell, wcan (bowl excluded)
TASKS = ["IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan"]
EPS = range(30, 120)


def rec(t, m, e):
    with open(f"{O}/{BE}/episodes/{t}/{m}/ep{e:+05d}.json") as f:
        return json.load(f)


pairs, n_all, missing = [], 0, []
for ti, t in enumerate(TASKS):
    for e in EPS:
        a, r = rec(t, "alone", e), rec(t, "reachable", e)
        n_all += 1
        if (not a["success"]) and r["success"]:
            da, dr = a.get("close_dist_cm"), r.get("close_dist_cm")
            if da is None or dr is None:
                missing.append((t, e, da, dr, a.get("reason"), r.get("reason")))
            pairs.append(dict(task=t, ti=ti, ep=e, dA=da, dR=dr, reasonA=a.get("reason"), whyA=a.get("why"),
                              hostA=a.get("host"), hostR=r.get("host")))
print(f"decision set pairs {n_all}; A fail & R success: {len(pairs)}; missing close distance: {len(missing)} {missing}")
P = [p for p in pairs if p["dA"] is not None and p["dR"] is not None]
n = len(P)


def median(v):
    v = sorted(v)
    k = len(v)
    return v[k // 2] if k % 2 else 0.5 * (v[k // 2 - 1] + v[k // 2])


def pct_mean(v, x):  # scipy percentileofscore kind='mean': 100 * (#{<x} + 0.5 #{==x}) / n
    lt = sum(1 for u in v if u < x)
    eq = sum(1 for u in v if u == x)
    return 100.0 * (lt + 0.5 * eq) / len(v)


def pct_rank(v, x):  # kind='rank' (scipy's default): average rank / n
    lt = sum(1 for u in v if u < x)
    eq = sum(1 for u in v if u == x)
    return 100.0 * (lt + (eq + 1) / 2.0) / len(v)


VA, VR = [p["dA"] for p in P], [p["dR"] for p in P]
print(f"n {n}; median close_dist_cm A {median(VA):.4f} R {median(VR):.4f}; ties A {n - len(set(VA))} R {n - len(set(VR))}")
for kind, fn in (("mean", pct_mean), ("rank", pct_rank)):
    for p in P:
        p[f"pA_{kind}"], p[f"pR_{kind}"] = fn(VA, p["dA"]), fn(VR, p["dR"])
        p[f"s_{kind}"] = max(abs(p[f"pA_{kind}"] - 50), abs(p[f"pR_{kind}"] - 50))
    order = sorted(P, key=lambda p: (p[f"s_{kind}"], p["ep"], p["ti"]))
    print(f"-- percentile kind={kind}: top 12 (score, task, ep, dA, dR, pA, pR, A reason)")
    for i, p in enumerate(order[:12]):
        print(f"  {i + 1:2d} {p[f's_{kind}']:6.3f} {p['task']:48s} {p['ep']:3d} dA {p['dA']:7.3f} dR {p['dR']:6.3f} "
              f"pA {p[f'pA_{kind}']:6.2f} pR {p[f'pR_{kind}']:6.2f} {p['reasonA']} | {p['whyA']} | {p['hostA'][-5:]}/{p['hostR'][-5:]}")
    if kind == "mean":
        out = [dict(rank=i + 1, **{k: p[k] for k in ("task", "ep", "dA", "dR", "pA_mean", "pR_mean", "s_mean", "reasonA", "hostA", "hostR")}) for i, p in enumerate(order)]
print("JSON", json.dumps(dict(n=n, medA=median(VA), medR=median(VR), order=out)))
