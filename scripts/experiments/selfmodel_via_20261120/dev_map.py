"""Where does the hand stray most from the path the controller is told to follow? Per reach of a self-model dataset: the
largest distance of the measured jaw centre from the reference jaw path (C, the commanded jaw per tick, in world through
the pelvis pose), i.e. the sideways error that matters for collisions (lag along the path does not count).
Grouped by pelvis height asked (stand / crouch), target height and reach length.
    python -I dev_map.py <dataset dir>"""

import sys
from pathlib import Path

import numpy as np

z = np.load(Path(sys.argv[1]) / "all.npz")
S, W, C = z["S"], z["W"], z["C"]
n, k0s = z["n_steps"].astype(int), z["k0"].astype(int)
so = np.r_[0, np.cumsum(n)[:-1]]


def Rp_of(yaw, g):
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def seg_dist(P, A, B):  # each point of P to the polyline A[i] -> B[i]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-12)
    u = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + u[..., None] * AB[None]), axis=2).min(1)


rows = []
for i in np.where(z["fell"] == 0)[0]:
    a, b = so[i] + k0s[i], so[i] + n[i]
    if b - a < 5:
        continue
    col = 4 if z["side"][i] > 0 else 7
    ref = np.array([W[k, :3] + Rp_of(float(W[k, 3]), S[k, 58:61].astype(float)) @ C[k, :3] for k in range(a, b)])
    act = W[a:b, col : col + 3].astype(float)
    length = float(np.linalg.norm(ref[-1] - ref[0]))
    if length < 0.10:
        continue
    dev = 100 * seg_dist(act, ref[:-1], ref[1:]).max()
    pz = float(z["pelvis_z"][i])
    rows.append((dev, pz, float(ref[-1, 2]), length))
R = np.array(rows)
print(
    "reaches",
    len(R),
    "| deviation from the reference path, cm: median %.2f  p90 %.2f  p99 %.2f  max %.2f"
    % tuple(np.percentile(R[:, 0], [50, 90, 99, 100])),
)
crouch = (R[:, 1] > 0) & (R[:, 1] < 0.70)


def show(name, m):
    d = R[m, 0]
    if len(d):
        print(
            f"  {name:34s} n {len(d):6d}  median {np.median(d):5.2f}  p90 {np.percentile(d, 90):5.2f}  p99 {np.percentile(d, 99):5.2f}"
        )


print("pelvis_z values (head):", np.unique(np.round(R[:, 1], 2))[:6], "...")
show("stand (no pelvis height asked)", R[:, 1] <= 0)
show("pelvis asked >= 0.70 m", R[:, 1] >= 0.70)
show("crouch (pelvis asked 0.52-0.70 m)", crouch)
for lo, hi in ((0.0, 0.4), (0.4, 0.7), (0.7, 1.0), (1.0, 1.6)):
    show(f"target height {lo:.1f}-{hi:.1f} m", (R[:, 2] >= lo) & (R[:, 2] < hi))
for lo, hi in ((0.1, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 2.0)):
    show(f"reach length {lo:.2f}-{hi:.2f} m", (R[:, 3] >= lo) & (R[:, 3] < hi))
show("crouch, target below 0.4 m", crouch & (R[:, 2] < 0.4))
show("crouch, target below 0.4 m, length > 0.4", crouch & (R[:, 2] < 0.4) & (R[:, 3] > 0.4))
