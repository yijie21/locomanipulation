"""The unnamed target of TabletopGraspMP from the colour image instead of from the depth: SAM 3 with a generic phrase on the
Isaac looks, then the largest instance among those within reach (2026-11-19, offline from the dumps; truth only scores)."""

import glob
import json
import sys

import numpy as np

from locomani.tasks.scene.services import Client, unpack_mask

D, out = sys.argv[1], sys.argv[2]
sam = Client(("127.0.0.1", 18832))
PH = ["object", "item on the table"]
rows = []
for fn in sorted(glob.glob(f"{D}/TabletopGraspMP_L*_e*.npz")):
    z = np.load(fn)
    left, pts, ok, true = z["left"], z["pts"], z["ok"].astype(bool), z["true"].astype(bool)
    yaw, pb = float(z["yaw"]), z["pb"]
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    u = pts[..., :2] @ ahead - pb[:2] @ ahead
    found = sam.call(dict(image=left, prompts=PH))["results"]
    r = dict(name=fn.split("/")[-1][:-4])
    for ph in PH:
        inst = []
        for score, packed in found.get(ph, []):
            m = unpack_mask(packed).astype(bool)
            a = int(m.sum())
            if score < 0.3 or a < 150 or a > 0.08 * m.size or (m & ok).sum() < 100:
                continue
            um = float(np.median(u[m & ok]))
            inst.append((a, float(score), um, float((m & true).sum() / max(1, (m | true).sum()))))
        reach = [i for i in inst if 0.2 < i[2] < 1.0]
        best = max(reach, key=lambda i: i[0]) if reach else None
        nearest = min(reach, key=lambda i: i[2]) if reach else None
        r[ph] = dict(
            n=len(inst),
            n_reach=len(reach),
            largest_iou=None if best is None else round(best[3], 3),
            nearest_iou=None if nearest is None else round(nearest[3], 3),
            best_any_iou=round(max([i[3] for i in inst], default=0.0), 3),
        )
    rows.append(r)
json.dump(rows, open(out, "w"), indent=1)
for ph in PH:
    li = np.array([(r[ph]["largest_iou"] or 0.0) for r in rows])
    ni = np.array([(r[ph]["nearest_iou"] or 0.0) for r in rows])
    ba = np.array([r[ph]["best_any_iou"] for r in rows])
    print(
        repr(ph),
        "| SAM 3 has the target among its instances:",
        int((ba > 0.5).sum()),
        "/ 30 | largest within reach is the target:",
        int((li > 0.5).sum()),
        "IoU median",
        round(float(np.median(li[li > 0.5])), 3) if (li > 0.5).any() else None,
        "| nearest within reach is the target:",
        int((ni > 0.5).sum()),
        "| instances per image median",
        int(np.median([r[ph]["n"] for r in rows])),
    )
    print("   largest wrong:", [r["name"][16:] for r, x in zip(rows, li) if x <= 0.5])
