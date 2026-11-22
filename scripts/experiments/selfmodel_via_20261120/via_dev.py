"""selfmodel_via_20261120: how far, and in which direction, the body strays from a path of several via points.

    python -I via_dev.py <dataset dir>      # all.npz of a REACH_VIA / REACH_VIA_REPLAY dataset and arm_spheres.py's arm.npz

Per path (chain 0, no fall), from the command to the end of the recording:
  hand   the jaw centre's largest distance from the reference jaw path (C, the commanded jaw per tick through the pelvis
         pose): dev_map.py's number, so the older datasets compare.
  arm    the largest distance of any of the acting arm's 24 collision spheres from the path the reference plan sweeps with
         that sphere (the polyline through the reference poses, arm.npz XR): "the arm goes where the reference says" as
         the baseline, lag along the path not counted; and which link it is.
  peak   the offset at that moment, as a vector in the heading frame of the command (x ahead, y left, z up), for the hand
         and for the arm's worst sphere.
Printed: the medians by posture, bends and speed; the plan's IK miss and falls. For a replay (via_route >= 0), per path
over its repeats: the mean cosine between two repeats' peak offsets, for the hand and for the elbow's sphere -- 1 when
every repeat strays the same way, 0 on average when the direction is noise (whatever the number of repeats). This
"steadiness" is the first check of whether the offset has a direction a model could learn (the plan's criteria,
docs/handoff/notebook-20261120/selfmodel_via_20261120.md).
-> <dataset dir>/via_dev.npz: one row per path (seg, hand_cm, arm_cm, arm_sphere, hand_vec, arm_vec, elbow_vec).
"""

import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
z, A = (
    dict(np.load(D / "all.npz")),
    dict(np.load(D / "arm.npz")),
)  # every array read once: an NpzFile decompresses the whole array again at each z[...]
S, W, C = z["S"], z["W"], z["C"]
n, k0s = z["n_steps"].astype(int), z["k0"].astype(int)
so, ro = np.r_[0, np.cumsum(n)[:-1]], np.r_[0, np.cumsum(z["n_ref"])[:-1]].astype(int)
LINK = [str(x) for x in A["link"]]
ELBOW = LINK.index("elbow_link")


def Rp_of(yaw, g):
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def off_path(P, Q):
    """Each point of P (t, 3) minus the nearest point of the polyline through Q (r, 3): (t, 3)."""
    if len(Q) == 1:
        return P - Q[0]
    A_, AB = Q[:-1], Q[1:] - Q[:-1]
    u = np.clip(((P[:, None] - A_[None]) * AB[None]).sum(2) / np.maximum((AB * AB).sum(1), 1e-12)[None], 0, 1)
    V = P[:, None] - (A_[None] + u[..., None] * AB[None])
    return V[np.arange(len(P)), np.linalg.norm(V, axis=2).argmin(1)]


rows = []
for i in np.where((z["chain"] == 0) & (z["fell"] == 0))[0]:
    a, b = so[i] + k0s[i], so[i] + n[i]
    yaw0 = float(z["start"][i][3])
    Rh = np.array([[np.cos(yaw0), -np.sin(yaw0), 0.0], [np.sin(yaw0), np.cos(yaw0), 0.0], [0.0, 0.0, 1.0]])
    col = 4 if z["side"][i] > 0 else 7
    ref = np.array([W[k, :3] + Rp_of(float(W[k, 3]), S[k, 58:61].astype(float)) @ C[k, :3] for k in range(a, b)])
    v = off_path(W[a:b, col : col + 3].astype(float), ref)
    t = int(np.linalg.norm(v, axis=1).argmax())
    hand_vec = Rh.T @ v[t]
    X, XR = A["X"][a:b].astype(float), A["XR"][ro[i] : ro[i] + int(z["n_ref"][i])].astype(float)
    V = np.stack([off_path(X[:, s], XR[:, s]) for s in range(X.shape[1])], 1)  # (t, 24, 3), heading frame already
    d = np.linalg.norm(V, axis=2)
    ta, sa = np.unravel_index(int(d.argmax()), d.shape)
    te = int(d[:, ELBOW].argmax())
    rows.append(dict(seg=i, hand_cm=100 * float(np.linalg.norm(hand_vec)), arm_cm=100 * float(d[ta, sa]), arm_sphere=int(sa),
                     hand_vec=100 * hand_vec, arm_vec=100 * V[ta, sa], elbow_vec=100 * V[te, ELBOW]))  # fmt: skip
seg = np.array([r["seg"] for r in rows])
hand, arm = np.array([r["hand_cm"] for r in rows]), np.array([r["arm_cm"] for r in rows])
sph = np.array([r["arm_sphere"] for r in rows])
HV, AV, EV = (np.array([r[k] for r in rows]) for k in ("hand_vec", "arm_vec", "elbow_vec"))
np.savez(D / "via_dev.npz", seg=seg, hand_cm=hand, arm_cm=arm, arm_sphere=sph, hand_vec=HV, arm_vec=AV, elbow_vec=EV)

c0 = z["chain"] == 0
print(f"paths {int(c0.sum())}, fell {int((c0 & (z['fell'] > 0)).sum())}, analysed {len(rows)}; "
      f"plan's IK miss > 2 cm: {int((z['ik_err_cm'][c0] > 2).sum())}; skipped resets "
      f"{sum(len(json.load(open(f))['skipped']) for f in D.glob('w*_skipped.json'))}")  # fmt: skip


def show(name, m):
    if m.sum():
        print(f"  {name:30s} n {int(m.sum()):5d}   hand median {np.median(hand[m]):5.2f} p90 {np.percentile(hand[m], 90):5.2f}"
              f"   arm median {np.median(arm[m]):5.2f} p90 {np.percentile(arm[m], 90):5.2f} cm")  # fmt: skip


pz, bends, speed = z["pelvis_z"][seg], z["via_gen"][seg, 0], z["via_speed"][seg]
print("largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):")
show("all", np.ones(len(seg), bool))
show("standing", pz < 0)
show("crouching (pelvis asked)", pz > 0)
if (z["via_route"][seg] < 0).any():
    for k in range(4):
        show(f"{k} bends", bends == k)
    show("speed 0.15 m/s", np.abs(speed - 0.15) < 1e-6)
    show("speed below 0.15", speed < 0.15 - 1e-6)
    show("speed above 0.15", speed > 0.15 + 1e-6)
top = np.bincount(sph, minlength=len(LINK))
print(
    "the arm's worst sphere is on:",
    {l: int(sum(top[k] for k in range(len(LINK)) if LINK[k] == l)) for l in dict.fromkeys(LINK)},
)

route = z["via_route"][seg]
if (route >= 0).any():
    names = json.load(open(D / "routes.json"))["names"]
    steady = {"hand": [], "elbow": []}
    print(
        "replayed paths, per path over its repeats (steadiness: mean cosine between repeats, 1 = the same direction every time):"
    )
    for r in sorted(set(route[route >= 0].tolist())):
        m = route == r
        line = f"  {names[r]:34s} repeats {int(m.sum())}  hand {np.median(hand[m]):5.2f} cm  arm {np.median(arm[m]):5.2f} cm"
        if m.sum() >= 2:
            for k, V in (("hand", HV), ("elbow", EV)):
                U = V[m] / np.maximum(np.linalg.norm(V[m], axis=1, keepdims=True), 1e-9)
                s = float((U @ U.T)[np.triu_indices(int(m.sum()), 1)].mean())
                steady[k].append(s)
                line += f"  {k} steadiness {s:5.2f}"
        print(line)
    for k, v in steady.items():
        if v:
            print(f"  {k} steadiness over {len(v)} paths: median {np.median(v):.2f}, lowest {min(v):.2f}")
