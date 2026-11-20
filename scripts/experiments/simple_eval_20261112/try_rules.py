"""The unnamed-target rule (see.largest_thing) and variants of it on the Isaac looks of TabletopGraspMP, offline from the
dumps (2026-11-19). Truth (the target's true mask) only scores the result.
  A  as it is: the surface is one height (the mode of the heights), things rise 1.2 cm above it
  P  the surface is a plane fitted to its own points (it may tilt), things rise 1.2 cm above the plane
  R  as A, but only within 1.0 m ahead of the pelvis (what the standing robot could reach for)
  PR both"""

import glob
import sys

import numpy as np
from scipy import ndimage

D = sys.argv[1]
OWN = 0.2


def rule(pts, ok, ahead, pbu, plane, reach):
    u = pts[..., :2] @ ahead
    far = ok & (u > pbu + OWN)
    z = pts[far][:, 2]
    lo = np.percentile(z, 1) + 0.2
    h, e = np.histogram(z[z > lo], bins=np.arange(lo, z.max() + 0.02, 0.01))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    top = float(np.median(z[np.abs(z - top) < 0.008]))
    height = pts[..., 2] - top
    if plane:
        w = pts[..., :2] @ np.array([-ahead[1], ahead[0]])
        sel = far & (np.abs(height) < 0.03)
        for tol in (0.03, 0.015, 0.008, 0.008):
            A = np.column_stack([u[sel], w[sel], np.ones(sel.sum())])
            c, *_ = np.linalg.lstsq(A, pts[sel][:, 2], rcond=None)
            height = pts[..., 2] - (c[0] * u + c[1] * w + c[2])
            sel = far & (np.abs(height) < tol)
    cand = far & (height > 0.012) & (height < 0.4)
    if reach:
        cand &= u < pbu + 1.0
    lab, n = ndimage.label(cand)
    best = None
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= 150 and height[m].min() < 0.03 and (best is None or m.sum() > best.sum()):
            best = m
    return best, (np.degrees(np.arctan(c[0])) if plane else 0.0)


res = {k: [] for k in ("A", "P", "R", "PR")}
tilt = []
for fn in sorted(glob.glob(f"{D}/TabletopGraspMP_L*_e*.npz")):
    z = np.load(fn)
    pts, ok, true = z["pts"], z["ok"].astype(bool), z["true"].astype(bool)
    yaw, pb = float(z["yaw"]), z["pb"]
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    for k, (pl, rc) in dict(A=(0, 0), P=(1, 0), R=(0, 1), PR=(1, 1)).items():
        m, t = rule(pts, ok, ahead, float(pb[:2] @ ahead), pl, rc)
        iou = 0.0 if m is None else float((m & true).sum() / max(1, (m | true).sum()))
        res[k].append(iou)
        if k == "P":
            tilt.append(t)
for k, v in res.items():
    v = np.array(v)
    print(
        k,
        "target right (IoU > 0.5):",
        int((v > 0.5).sum()),
        "/",
        len(v),
        "| IoU median of those",
        round(float(np.median(v[v > 0.5])), 3) if (v > 0.5).any() else None,
        "| wrong:",
        [i for i, x in enumerate(v) if x <= 0.5],
    )
print(
    "fitted surface tilt along ahead, deg: median",
    round(float(np.median(tilt)), 2),
    "range",
    round(float(np.min(tilt)), 2),
    round(float(np.max(tilt)), 2),
)
