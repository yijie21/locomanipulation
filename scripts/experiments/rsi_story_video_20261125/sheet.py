"""sheet.py <mp4> <out.jpg>: 3x3 contact sheet of evenly spread frames of a video (also prints frame-motion stats)."""

import sys

import cv2
import numpy as np

cap = cv2.VideoCapture(sys.argv[1])
fr = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    fr.append(f)
n = len(fr)
idx = [round(i * (n - 1) / 8) for i in range(9)]
tiles = []
for i in idx:
    t = cv2.resize(fr[i], (560, 315))
    cv2.putText(t, f"f{i} {i / 12.5:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(t)
cv2.imwrite(
    sys.argv[2], np.vstack([np.hstack(tiles[r * 3 : r * 3 + 3]) for r in range(3)]), [cv2.IMWRITE_JPEG_QUALITY, 80]
)
d = [float(np.abs(fr[i].astype(int) - fr[i - 1].astype(int)).mean()) for i in range(1, n, 5)]
print(n, "frames; mean abs diff per 5th frame: min %.2f mean %.2f" % (min(d), sum(d) / len(d)))
