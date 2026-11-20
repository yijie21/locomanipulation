"""The chips can as the Isaac look saw it: the left image, SAM's mask (green) and the replica's true outline (red), cropped.

$PY -I can_look.py <cloud dir> <out png> NAME [NAME ...]
"""

import sys

import cv2
import numpy as np

src, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
tiles = []
for n in names:
    z = np.load(f"{src}/{n}.npz")
    rgb = z["rgb"].astype(np.uint8).copy()
    m, t = z["mask"].astype(bool), z["true_mask"].astype(bool)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for mask, col in ((m, (0, 255, 0)), (t, (0, 0, 255))):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 1)
    ys, xs = np.where(m | t)
    y0, y1, x0, x1 = max(ys.min() - 40, 0), ys.max() + 40, max(xs.min() - 40, 0), xs.max() + 40
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    iou = (m & t).sum() / max((m | t).sum(), 1)
    cy = (np.where(m)[0].mean() - np.where(t)[0].mean(), np.where(m)[1].mean() - np.where(t)[1].mean())
    cv2.putText(crop, f"{n} IoU {iou:.2f} centroid dy {cy[0]:+.0f} dx {cy[1]:+.0f} px", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # fmt: skip
    tiles.append(crop)
    print(
        n,
        "IoU",
        round(float(iou), 3),
        "mask px",
        int(m.sum()),
        "true px",
        int(t.sum()),
        "centroid shift px (dy, dx)",
        np.round(cy, 1),
    )
h = max(t.shape[0] for t in tiles)
row = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles])
cv2.imwrite(out, row)
