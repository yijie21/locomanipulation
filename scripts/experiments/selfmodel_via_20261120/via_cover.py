"""selfmodel_via_20261120, the coverage check: are the real cuRobo paths outliers of a REACH_VIA dataset's random paths?

    python -I via_cover.py <dataset dir> <via dir>:<prep dir>[,<via dir>:<prep dir> ...]

Every path becomes via_gen.shape's numbers (via points, straight distance, length over it, the largest offset from the
straight line ahead / outwards / up, the wrist's turn along the path, goal; a left hand's path mirrored onto the right's),
each divided by its spread over the dataset's random paths (chain 0 of <dataset dir>/all.npz). The distance of two paths
is the Euclidean one in these ten numbers. The start is printed but left out of the distance: a replayed path is run from
wherever the hand is, as a random one is, and the random paths' starts differ by under a centimetre, so any real start a
few cm off would count as many spreads away.

A path's isolation is its distance to the nearest random path (for a random path: to the nearest other one; taken over
3000 of them when there are more).

The rule (2026-11-21, the user's choice after the trial; docs/handoff/notebook-20261120/selfmodel_via_20261120.md): a real
path is covered when it is no more isolated than 95 % of the random paths are, i.e. it is not an outlier of what the
generator draws. Real paths beyond that are reported by name, and the generator is not bent towards them.

The rule fixed before the trial asked for every real path to be no more isolated than the MEDIAN random path. That asks the
real paths to lie in the denser half of the random ones -- half of the random paths themselves fail it -- and the answer
does not change with the number of paths collected. Its count is still printed, as "median rule".
Printed per real path: the share of random paths less isolated than it, and the feature with the largest gap to its
nearest random path.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import via_gen as VG  # noqa: E402

D, SPEC = Path(sys.argv[1]), sys.argv[2]
START = ("ax", "a_out", "az")  # printed, not part of the distance
LEVEL, SAMPLE = 95.0, 3000


def r6m(r):
    a, b = np.asarray(r[:3], float), np.asarray(r[3:6], float)
    return np.c_[a, b, np.cross(a, b)]


def nearest(A, B, skip_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per row of A: the distance to the nearest row of B (the nearest other one with skip_self), and its index."""
    d, at = np.zeros(len(A)), np.zeros(len(A), int)
    for i in range(0, len(A), 256):
        x = np.linalg.norm(A[i : i + 256, None] - B[None], axis=2)
        if skip_self:
            x[x < 1e-12] = np.inf
        d[i : i + 256], at[i : i + 256] = x.min(1), x.argmin(1)
    return d, at


z = dict(np.load(D / "all.npz"))  # every array read once: an NpzFile decompresses the whole array again at each z[...]
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
pick = np.random.default_rng(0).choice(len(G), min(SAMPLE, len(G)), replace=False)
iso = nearest(G[pick], G, skip_self=True)[0]
med, lim = float(np.median(iso)), float(np.percentile(iso, LEVEL))
dr, near = nearest(R, G)
print(
    f"random paths {len(G)}, real paths {len(R)}; a random path's isolation: median {med:.3f}, {LEVEL:.0f} % level {lim:.3f}"
)
print("feature         " + "  ".join(f"{k:>8s}" for k in VG.FEATURES))
for name, x in (("random p05", np.percentile(gen, 5, 0)), ("random p95", np.percentile(gen, 95, 0)),
                ("real min", real.min(0)), ("real max", real.max(0))):  # fmt: skip
    print(f"{name:16s}" + "  ".join(f"{v:8.3f}" for v in x))
inside = ((real[:, use] >= gen[:, use].min(0)) & (real[:, use] <= gen[:, use].max(0))).all(1)
passed = 0
for i, r in enumerate(routes):
    share = float((iso < dr[i]).mean())
    worst = VG.FEATURES[use[int(np.argmax(np.abs(R[i] - G[near[i]])))]]
    ok = dr[i] <= lim
    passed += ok
    print(
        f"{r['name']:34s} more isolated than {100 * share:5.1f} % of the random paths  {'pass' if ok else 'FAIL'}"
        f"  nearest at {dr[i] / np.sqrt(len(use)):4.2f} spreads a feature, largest gap in {worst}"
    )
print(
    f"covered {passed} of {len(routes)} real paths ({LEVEL:.0f} % rule); median rule {int((dr <= med).sum())}; inside the random "
    f"paths' range in every feature {int(inside.sum())}"
)
