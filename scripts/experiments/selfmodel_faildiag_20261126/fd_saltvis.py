import numpy as np, json
D = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/hg/looks/IThorKitchenApproachTableAndLiftSaltShaker"
O = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126/q4"
per = json.load(open(O + "/salt.json"))
rows = []
for p in per:
    z = np.load(D + "/ep%+05d.npz" % p["ep"]); dep = z["depth"]; mk = z["mask"] > 0
    ui, vi = int(round(p["u"])), int(round(p["v"]))
    w = dep[max(vi - 6, 0):vi + 7, max(ui - 6, 0):ui + 7]
    near = np.abs(w - p["exp_depth"]) < 0.10
    r = dict(ep=p["ep"], cat=p["cat"], near_px=int(near.sum()))
    if mk.any():
        vv, uu = np.nonzero(mk)
        r.update(mask_px=int(mk.sum()), du=float(uu.mean() - p["u"]), dv=float(vv.mean() - p["v"]), mask_h=int(vv.max() - vv.min() + 1),
                 mask_w=int(uu.max() - uu.min() + 1), mask_dep_minus_exp=float(np.median(dep[mk]) - p["exp_depth"]))
    rows.append(r)
rec = [r for r in rows if "mask_px" in r]; nr = [r for r in rows if r["cat"].startswith("no")]
md = lambda k, R: float(np.median([r[k] for r in R]))
print("salt visibility check (13x13 window around the object's expected pixel; mask = the chosen SAM 3 instance)")
print(" recognized n %d: mask pixels p50 %.0f, mask height x width p50 %.0f x %.0f px; |mask centroid - expected pixel| p50 du %.1f dv %.1f px; mask depth - expected distance p50 %.3f m"
      % (len(rec), md("mask_px", rec), md("mask_h", rec), md("mask_w", rec), float(np.median([abs(r["du"]) for r in rec])), float(np.median([abs(r["dv"]) for r in rec])), md("mask_dep_minus_exp", rec)))
print(" pixels in the window within 10 cm of the expected distance: recognized p50 %.0f (min %d); not recognized p50 %.0f (min %d, zero in %d of %d)"
      % (md("near_px", rec), min(r["near_px"] for r in rec), md("near_px", nr), min(r["near_px"] for r in nr), sum(r["near_px"] == 0 for r in nr), len(nr)))
json.dump(rows, open(O + "/salt_vis.json", "w"), indent=0)
