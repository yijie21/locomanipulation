"""Why the unnamed-target rule (see.largest_thing) fails on the Isaac look of TabletopGraspMP: the height of the table's
own pixels against how far they are, from the look dumps (2026-11-19). Truth (dtrue) is used only to say what is table."""

import glob
import sys

import numpy as np

D = sys.argv[1]
f, B = 0.5 * 640 / np.tan(np.radians(110) / 2), 0.05
bins = [(0.2, 0.6), (0.6, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.4), (2.4, 3.5)]
acc = {b: [] for b in bins}
dep = {b: [] for b in bins}
for fn in sorted(glob.glob(f"{D}/TabletopGraspMP_L*_e*.npz")):
    z = np.load(fn)
    pts, ok, lb, dt = z["pts"], z["ok"].astype(bool), z["lb"], z["dtrue"]
    yaw, pb = float(z["yaw"]), z["pb"]
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    u = pts[..., :2] @ ahead - pb[:2] @ ahead
    # the points as the true depth would put them: same rays, the true distance along the optical axis
    cam_p = z["cam_p"]
    ray = (pts - cam_p) / np.maximum(lb[..., None], 1e-6)  # per unit of depth along the axis
    ptrue = cam_p + ray * dt[..., None]
    far = ok & (u > 0.2) & (dt > 0.2) & (dt < 5)
    zt = ptrue[far][:, 2]
    lo = np.percentile(zt, 1) + 0.2
    h, e = np.histogram(zt[zt > lo], bins=np.arange(lo, zt.max() + 0.02, 0.01))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    table = far & (np.abs(ptrue[..., 2] - top) < 0.008)  # the table top's own pixels, by the truth
    ut = ptrue[..., :2] @ ahead - pb[:2] @ ahead
    for b in bins:
        m = table & (ut >= b[0]) & (ut < b[1])
        if m.sum() > 200:
            acc[b].append(
                (
                    float(np.median(pts[m][:, 2] - top)),
                    float(np.percentile(pts[m][:, 2] - top, 90)),
                    float((pts[m][:, 2] - top > 0.012).mean()),
                )
            )
            dep[b].append(
                (
                    float(np.median(dt[m])),
                    float(np.median(lb[m] - dt[m])),
                    float(np.percentile(np.abs(lb[m] - dt[m]), 90)),
                )
            )
print(
    "table pixels by distance ahead of the pelvis (m): episodes | seen height above the true top, cm: median, 90th pct | share above the rule's 1.2 cm | depth m, depth error cm median, |error| 90th pct | a quarter pixel of disparity in depth cm"
)
for b in bins:
    if not acc[b]:
        continue
    a, d = np.array(acc[b]), np.array(dep[b])
    Z = np.median(d[:, 0])
    print(
        f"{b[0]:.1f}-{b[1]:.1f} | {len(a):2d} | {100 * np.median(a[:, 0]):5.2f} {100 * np.median(a[:, 1]):5.2f} | {100 * np.median(a[:, 2]):4.0f} % | {Z:.2f} {100 * np.median(d[:, 1]):5.2f} {100 * np.median(d[:, 2]):5.2f} | {100 * Z * Z / (f * B) * 0.25:.1f}"
    )
