"""selfmodel_via_20261120: does the hand stray the same way on every repeat of a replayed path? The whole offset along the
path, not its largest moment.

    python -I via_profile.py <replay dataset dir>      # all.npz and routes.json of a REACH_VIA_REPLAY dataset

Written after the trial (2026-11-20), because the statistic fixed beforehand turned out not to measure this: via_dev.py's
"steadiness" compares the offset at each repeat's largest moment, and on these paths the hand has two moments of nearly the
same size -- one early on the path, one at the end where it stops short -- that point in nearly opposite directions; which
one is the largest changes from repeat to repeat, so two almost identical repeats can score a cosine near -1. It is kept in
via_dev.py's output as registered and is not to be read as "the direction is noise".

Here, per replayed segment: the jaw centre's offset from the reference jaw path (C, the commanded jaw per tick through the
pelvis pose) at every tick from the command on, placed by how far along the reference path its nearest point lies (0 start,
1 goal) and averaged in 20 bins: a 20 x 3 profile in the heading frame of the command, cm. Per path over its repeats:
  agreement   the mean cosine between two repeats' profiles (as 60 numbers)
  left over   each repeat's profile minus the mean profile of the path's OTHER repeats: the largest bin of what is left
              (cm) beside the largest bin of the profile itself, and the left-over's share of the profile's squared size
  common      the same with the mean profile of all the OTHER paths' repeats in place of the path's own: what a correction
              that does not know the path would leave
Printed over the paths: medians, by task, and the first pass (from the reset's stand) against the later ones (after a short
walk), which start from different states.
"""

import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
z, names = dict(np.load(D / "all.npz")), json.load(open(D / "routes.json"))["names"]
S, W, C = z["S"], z["W"], z["C"]
n, k0s = z["n_steps"].astype(int), z["k0"].astype(int)
so = np.r_[0, np.cumsum(n)[:-1]]
NB = 20


def Rp_of(yaw, g):
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


prof, peak_at, route = [], [], []
for i in np.where((z["chain"] == 0) & (z["fell"] == 0) & (z["via_route"] >= 0))[0]:
    a, b = so[i] + k0s[i], so[i] + n[i]
    yaw0 = float(z["start"][i][3])
    Rh = np.array([[np.cos(yaw0), -np.sin(yaw0), 0], [np.sin(yaw0), np.cos(yaw0), 0], [0, 0, 1]])
    ref = np.array([W[k, :3] + Rp_of(float(W[k, 3]), S[k, 58:61].astype(float)) @ C[k, :3] for k in range(a, b)])
    act = W[a:b, 4:7].astype(float) if z["side"][i] > 0 else W[a:b, 7:10].astype(float)
    A_, AB = ref[:-1], ref[1:] - ref[:-1]
    L = np.linalg.norm(AB, axis=1)
    cum = np.r_[0, np.cumsum(L)]
    u = np.clip(((act[:, None] - A_[None]) * AB[None]).sum(2) / np.maximum((AB * AB).sum(1), 1e-12)[None], 0, 1)
    V = act[:, None] - (A_[None] + u[..., None] * AB[None])
    j, t = np.linalg.norm(V, axis=2).argmin(1), np.arange(len(act))
    v = 100 * V[t, j] @ Rh  # heading frame, cm
    s = (cum[j] + u[t, j] * L[j]) / max(cum[-1], 1e-9)
    bins = np.minimum((s * NB).astype(int), NB - 1)
    prof.append(np.array([v[bins == q].mean(0) if (bins == q).any() else np.zeros(3) for q in range(NB)]))
    peak_at.append(s[int(np.linalg.norm(v, axis=1).argmax())])
    route.append(int(z["via_route"][i]))
prof, peak_at, route = np.array(prof), np.array(peak_at), np.array(route)
ids = sorted(set(route.tolist()))
big = lambda P: np.linalg.norm(P, axis=-1).max(-1)  # noqa: E731
rows = []
for r in ids:
    m = route == r
    P, k = prof[m], int(m.sum())
    if k < 2:
        continue
    F = P.reshape(k, -1)
    U = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    own = np.array([P[q] - np.delete(P, q, 0).mean(0) for q in range(k)])
    rest = P - prof[~m].mean(0)
    rows.append(dict(route=r, agree=float((U @ U.T)[np.triu_indices(k, 1)].mean()), size=float(big(P).mean()),
                     own=float(big(own).mean()), own_share=float((own**2).sum() / (P**2).sum()),
                     rest=float(big(rest).mean()), rest_share=float((rest**2).sum() / (P**2).sum()),
                     where=float(peak_at[m].max() - peak_at[m].min())))  # fmt: skip
col = lambda k: np.array([x[k] for x in rows])  # noqa: E731
print(f"paths {len(rows)}, repeats {int(np.bincount(route).min())}-{int(np.bincount(route).max())} each")
print(f"where on the path the largest offset falls differs between a path's repeats by {np.median(col('where')):.2f} of the "
      f"path (median; 90 % {np.percentile(col('where'), 90):.2f})")  # fmt: skip
print(f"agreement of the whole profile between repeats: median {np.median(col('agree')):.2f}, 10 % {np.percentile(col('agree'), 10):.2f}, "
      f"lowest {col('agree').min():.2f}")  # fmt: skip
print(f"largest bin of the profile: {np.median(col('size')):.2f} cm; left over after the path's other repeats' mean: "
      f"{np.median(col('own')):.2f} cm ({np.median(col('own_share')):.2f} of the squared size); after the other paths' mean: "
      f"{np.median(col('rest')):.2f} cm ({np.median(col('rest_share')):.2f})")  # fmt: skip
task = np.array([names[x["route"]].split("_")[0] for x in rows])
for t in dict.fromkeys(task.tolist()):
    m = task == t
    print(f"  {t:20s} n {int(m.sum()):2d}  agreement {np.median(col('agree')[m]):.2f}  left over {np.median(col('own')[m]):.2f} cm of "
          f"{np.median(col('size')[m]):.2f}; knowing only the other paths {np.median(col('rest')[m]):.2f} cm")  # fmt: skip
first = np.zeros(len(route), bool)
for r in ids:
    first[np.where(route == r)[0][0]] = True
end = prof[:, -1]
print(f"offset in the last bin (ahead, left, up cm): first pass {np.round(end[first].mean(0), 2)}, later passes "
      f"{np.round(end[~first].mean(0), 2)}")  # fmt: skip
