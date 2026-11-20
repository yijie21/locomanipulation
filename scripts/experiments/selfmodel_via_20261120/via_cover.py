"""selfmodel_via_20261120, the coverage check: do the random paths of a REACH_VIA dataset surround the real cuRobo paths?

    python -I via_cover.py <dataset dir> <via dir>:<prep dir>[,<via dir>:<prep dir> ...]

Every path becomes via_gen.shape's numbers (via points, straight distance, length over it, the largest offset from the
straight line ahead / outwards / up, the wrist's turn along the path, goal; a left hand's path mirrored onto the right's),
each divided by its spread over the dataset's random paths (chain 0 of <dataset dir>/all.npz). The distance of two paths
is the Euclidean one in these ten numbers. The start is printed but left out of the distance: a replayed path is run from
wherever the hand is, as a random one is, and the random paths' starts differ by under a centimetre, so any real start a
few cm off would count as many spreads away.

The rule, fixed before the first run (docs/handoff/notebook-20261120/selfmodel_via_20261120.md): every real path's distance
to its nearest random path must be no larger than the median distance of a random path to its own nearest random path.
A real path further out than that is a kind of shape the generator leaves out: widen the generator before collecting.
Printed per real path: that distance over the median (<= 1 passes) and the feature that contributes most to it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import via_gen as VG  # noqa: E402

D, SPEC = Path(sys.argv[1]), sys.argv[2]
START = ("ax", "a_out", "az")  # printed, not part of the distance


def r6m(r):
    a, b = np.asarray(r[:3], float), np.asarray(r[3:6], float)
    return np.c_[a, b, np.cross(a, b)]


z = np.load(D / "all.npz")
so = np.r_[0, np.cumsum(z["n_steps"])[:-1]].astype(int)
gen = []
for i in np.where(z["chain"] == 0)[0]:
    sd = 1.0 if z["side"][i] > 0 else -1.0
    k, n = so[i] + int(z["k0"][i]), int(z["n_via"][i])
    p0, yaw = z["start"][i][:3].astype(float), float(z["start"][i][3])
    Rh = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    jaw = VG.to_h(z["W"][k, 4:7] if sd > 0 else z["W"][k, 7:10], p0, Rh)[0]
    pts = np.vstack([jaw, z["via_p"][i][:n], z["target"][i]])
    Rs = [Rh.T @ r6m(z["J"][k])] + [r6m(r) for r in z["via_r6"][i][:n]] + [r6m(z["rot6"][i])]
    gen.append(VG.shape(pts, Rs, sd))
gen = np.array(gen)
routes = VG.load_routes(SPEC)
real = np.array(
    [VG.shape(np.vstack([r["start"], r["p"], r["goal"]]), [r["start_R"], *r["R"], r["goal_R"]]) for r in routes]
)
use = [i for i, k in enumerate(VG.FEATURES) if k not in START]
scale = gen.std(0) + 1e-9
G, R = (gen / scale)[:, use], (real / scale)[:, use]
dg = np.linalg.norm(G[:, None] - G[None], axis=2)
np.fill_diagonal(dg, np.inf)
med = float(np.median(dg.min(1)))
dr = np.linalg.norm(R[:, None] - G[None], axis=2)
near = dr.argmin(1)
print(f"random paths {len(G)}, real paths {len(R)}; a random path's nearest random path: median distance {med:.3f}")
print("feature         " + "  ".join(f"{k:>8s}" for k in VG.FEATURES))
for name, x in (("random p05", np.percentile(gen, 5, 0)), ("random p95", np.percentile(gen, 95, 0)),
                ("real min", real.min(0)), ("real max", real.max(0))):  # fmt: skip
    print(f"{name:16s}" + "  ".join(f"{v:8.3f}" for v in x))
passed = 0
for i, r in enumerate(routes):
    ratio = float(dr[i, near[i]]) / med
    worst = VG.FEATURES[use[int(np.argmax(np.abs(R[i] - G[near[i]])))]]
    passed += ratio <= 1.0
    print(
        f"{r['name']:34s} nearest random path at {ratio:5.2f} x the median  {'pass' if ratio <= 1 else 'FAIL'}  largest gap in {worst}"
    )
print(f"covered {passed} of {len(routes)} real paths; the rule asks for all of them")
