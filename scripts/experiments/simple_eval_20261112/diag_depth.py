"""The stereo depth on the table top against the replica's own depth, by distance: matched (sg) and filled (lb), per task."""

import glob
import sys

import numpy as np

D = sys.argv[1]
f, B = 0.5 * 640 / np.tan(np.radians(110) / 2), 0.05
bins = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 2.0)]
for task in sys.argv[2:]:
    out = {b: [] for b in bins}
    for fn in sorted(glob.glob(f"{D}/{task}_L*_e*.npz")):
        z = np.load(fn)
        pts, ok, lb, sg, dt = z["pts"], z["ok"].astype(bool), z["lb"], z["sg"], z["dtrue"]
        cam_p = z["cam_p"]
        ray = (pts - cam_p) / np.maximum(lb[..., None], 1e-6)
        ptrue = cam_p + ray * dt[..., None]
        good = ok & (dt > 0.25) & (dt < 5)
        zt = ptrue[good][:, 2]
        lo = np.percentile(zt, 1) + 0.2
        h, e = np.histogram(zt[zt > lo], bins=np.arange(lo, zt.max() + 0.02, 0.01))
        top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
        table = good & (np.abs(ptrue[..., 2] - top) < 0.008)
        for b in bins:
            m = table & (dt >= b[0]) & (dt < b[1])
            ms = m & (sg > 0)
            if m.sum() > 300:
                out[b].append(
                    (
                        np.median(lb[m] - dt[m]),
                        np.median(sg[ms] - dt[ms]) if ms.sum() > 100 else np.nan,
                        ms.sum() / m.sum(),
                        np.median((f * B / np.maximum(sg[ms], 1e-3) - f * B / dt[ms])) if ms.sum() > 100 else np.nan,
                    )
                )
    print(
        task,
        "| depth range m: episodes, filled - true cm, matched - true cm, share matched, matched disparity error px",
    )
    for b in bins:
        if out[b]:
            a = np.array(out[b], float)
            print(
                f"   {b[0]:.1f}-{b[1]:.1f}: {len(a):2d}  {100 * np.nanmedian(a[:, 0]):6.2f}  {100 * np.nanmedian(a[:, 1]):6.2f}  {100 * np.nanmedian(a[:, 2]):3.0f} %  {np.nanmedian(a[:, 3]):5.2f}"
            )
