"""selfmodel_via_20261120, fourth step, part two, the 2026-11-23 addition (the plan entry's "补充：计分用的手换成包住真实碰撞体
的球"): radii for the 24 scoring spheres that cover the right arm's real collision bodies in the simulation.

    cd $REPO && SIMPLE_ROOT=... $PY -I scripts/experiments/selfmodel_via_20261120/gap_cover.py <scene dir> <out json> \
        <run dir> [<run dir> ...] [--fit K] [--check K]

The independent review found cuRobo's spheres (arm_fk.ArmFK, the scoring spheres of gap_real.py) leave parts of the hand out
-- the palm by up to 3.4 cm -- so a sweep the spheres call clear touched the board in MuJoCo. Every sphere keeps its centre
and its link (the model predicts the centres); only radii grow.

Geoms: every geom of the right arm (bodies right_shoulder*, right_elbow*, right_wrist*, right_hand*) that can touch the gap
obstacles (contype / conaffinity against gap_board). Points on a geom: its convex hull's vertices and each hull triangle's
centroid (fit), or the vertices and a 3 x 3 grid on each triangle (check); a box's 8 corners, a round geom's surface
sampled. The wrist and hand's geoms (bodies right_wrist_pitch_link, right_wrist_yaw_link, right_hand_*: what gap_real.py
calls "through") are covered by the 17 through spheres; the shoulder, elbow and wrist-roll geoms by the 7 shoulder and
elbow spheres, so the arm's bodies do not inflate the hand's.

Covering within each body's own frame fails here: the wrist-yaw link carries the palm and also the wrist's housing, which
lies next to the wrist-pitch spheres, not the palm's (the palm's 3 would grow from 1 to 11.5 cm; tried 2026-11-23). So the
cover is fitted over poses: K arm poses (default 2000) drawn from the runs given -- half recorded ticks (all.npz S, the
actual sweeps), half reference poses (RQ, the wanted sweeps) -- with the fingers at each segment's hand_q. In each pose every
point goes to the sphere of its group whose surface is nearest under the radii so far, and that sphere grows to reach it;
two passes over the fit poses. Then a separate draw of K poses (default 1000) checks it: how far any point stands outside
the union, which should be about zero. Only the arm's own joints matter (the pelvis is put at the origin).

-> <out json>: link (24, arm_fk's order), radius0 and radius (m), grow_cm, group, geoms (name, body, points), uncovered (a
geom with no sphere in its group, none expected), fit / check (poses, outside the union before and after: quantiles, cm).
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_fk import ArmFK  # noqa: E402

ARGS = sys.argv[1:]


def opt(k, d):
    if k in ARGS:
        i = ARGS.index(k)
        v = ARGS[i + 1]
        del ARGS[i : i + 2]
        return v
    return d


KFIT, KCHECK = int(opt("--fit", "2000")), int(opt("--check", "1000"))
SCENE, OUT, RUNS = Path(ARGS[0]), Path(ARGS[1]), [Path(x) for x in ARGS[2:]]
fk = ArmFK(SCENE)
m, d = fk.m, fk.d
name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""  # noqa: E731
board = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gap_board")
assert board >= 0, f"{SCENE} has no gap_board (gap_scene.py's scene)"
link = list(fk.links)
r0 = fk.radius.astype(float)
THR = np.array([x == "wrist_pitch_link" or x.startswith("hand_") for x in link])
GROUP = {"through": np.where(THR)[0], "arm": np.where(~THR)[0]}


def hull_points(g, grid):
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        s = m.geom_size[g]
        return np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
    else:
        s = m.geom_size[g]
        u = np.random.default_rng(0).normal(size=(2000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return s[0] * u
        if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            return s[0] * u + np.c_[np.zeros((len(u), 2)), np.sign(u[:, 2]) * s[1]]
        return s[:3] * u
    h = ConvexHull(v)
    T = v[h.simplices]  # (triangles, 3, 3)
    if not grid:
        return np.concatenate([v[h.vertices], T.mean(1)])
    a, b = np.meshgrid(np.arange(4), np.arange(4))
    keep = a + b <= 3
    w = np.c_[a[keep], b[keep]] / 3.0
    P = T[:, :1] + w[None, :, :1] * (T[:, 1:2] - T[:, :1]) + w[None, :, 1:] * (T[:, 2:3] - T[:, :1])
    return np.concatenate([v[h.vertices], P.reshape(-1, 3)])


geoms, uncovered = [], []
for g in range(m.ngeom):
    bn = name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    if not bn.startswith(("right_shoulder", "right_elbow", "right_wrist", "right_hand")):
        continue
    if not ((m.geom_contype[g] & m.geom_conaffinity[board]) or (m.geom_contype[board] & m.geom_conaffinity[g])):
        continue
    grp = (
        "through" if bn in ("right_wrist_pitch_link", "right_wrist_yaw_link") or bn.startswith("right_hand_") else "arm"
    )
    if not len(GROUP[grp]):
        uncovered.append(dict(geom=name(mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}", body=bn))
        continue
    geoms.append(dict(g=g, geom=name(mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}", body=bn, group=grp,
                      fit=hull_points(g, False).astype(np.float32), check=hull_points(g, True).astype(np.float32)))  # fmt: skip

# poses: (order, joints, fingers), half recorded ticks, half reference poses, over the runs given
rng = np.random.default_rng(20261123)
pool = []
for D in RUNS:
    z = np.load(D / "all.npz")
    S, RQ, hq, side = z["S"], z["RQ"], z["hand_q"], z["side"]
    n, nr = z["n_steps"].astype(int), z["n_ref"].astype(int)
    seg_t, seg_r = np.repeat(np.arange(len(n)), n), np.repeat(np.arange(len(nr)), nr)
    right_t, right_r = np.where(side[seg_t] > 0)[0], np.where(side[seg_r] > 0)[0]
    pool.append((S, RQ, hq, seg_t, seg_r, right_t, right_r))


def draw(k):
    out = []
    for _ in range(k):
        S, RQ, hq, seg_t, seg_r, rt, rr = pool[rng.integers(len(pool))]
        if rng.random() < 0.5:
            j = rt[rng.integers(len(rt))]
            out.append(("meta", S[j, :29].astype(float), hq[seg_t[j]]))
        else:
            j = rr[rng.integers(len(rr))]
            out.append(("mj", RQ[j].astype(float), hq[seg_r[j]]))
    return out


QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def pose(p):
    C, _ = fk.spheres("right", np.zeros(3), QUAT, p[1], p[0], p[2])
    W = {}
    for e in geoms:
        R = d.geom_xmat[e["g"]].reshape(3, 3)
        W[e["g"]] = (d.geom_xpos[e["g"]], R)
    return C, W


def outside(C, W, r, which):
    """per group: each point's distance outside the group's union of spheres (negative inside) and its nearest sphere"""
    res = {}
    for grp, idx in GROUP.items():
        P = [W[e["g"]][0] + e[which] @ W[e["g"]][1].T for e in geoms if e["group"] == grp]
        if not P:
            continue
        P = np.concatenate(P)
        dist = np.linalg.norm(P[:, None] - C[idx][None], axis=-1)
        gap = dist - r[idx][None]
        k = gap.argmin(1)
        res[grp] = (gap[np.arange(len(P)), k], idx[k], dist[np.arange(len(P)), k])
    return res


fit = draw(KFIT)
before = []
r = r0.copy()
for npass in range(2):
    for p in fit:
        C, W = pose(p)
        for grp, (gap, j, dist) in outside(C, W, r, "fit").items():
            if npass == 0:
                before.append(float(outside(C, W, r0, "fit")[grp][0].max()))
            np.maximum.at(r, j, np.where(gap > 0, dist, 0.0))
check = draw(KCHECK)
chk = {g: [] for g in GROUP}
chk0 = {g: [] for g in GROUP}
for p in check:
    C, W = pose(p)
    for grp, (gap, _, _) in outside(C, W, r, "check").items():
        chk[grp].append(float(gap.max()))
    for grp, (gap, _, _) in outside(C, W, r0, "check").items():
        chk0[grp].append(float(gap.max()))
q = lambda v: {f"q{int(100 * x)}": round(100 * float(np.quantile(v, x)), 3) for x in (0.5, 0.95, 0.99, 1.0)}  # noqa: E731
summary = dict(scene=str(SCENE), runs=[str(x) for x in RUNS], link=link, group=["through" if t else "arm" for t in THR],
               radius0=r0.tolist(), radius=r.tolist(), grow_cm=(100 * (r - r0)).round(3).tolist(),
               geoms=[dict(geom=e["geom"], body=e["body"], group=e["group"], fit_points=len(e["fit"]),
                           check_points=len(e["check"])) for e in geoms],
               uncovered=uncovered, fit=dict(poses=KFIT, passes=2), check=dict(poses=KCHECK,
               outside_before_cm={g: q(v) for g, v in chk0.items() if v}, outside_after_cm={g: q(v) for g, v in chk.items() if v}))  # fmt: skip
json.dump(summary, open(OUT, "w"), indent=1)
print(
    f"{len(geoms)} geoms ({sum(e['group'] == 'through' for e in geoms)} through), uncovered {uncovered}; "
    f"fit on {KFIT} poses, check on {KCHECK} others, from {len(RUNS)} runs"
)
for g in GROUP:
    if chk[g]:
        print(
            f"  {g:7s} largest distance of a point outside the union per pose, cm: cuRobo's radii {q(chk0[g])}; "
            f"the cover's {q(chk[g])}"
        )
print("sphere  link                 radius before -> after (cm)")
for j in range(len(link)):
    print(f"  {j:2d}    {link[j]:20s} {100 * r0[j]:5.2f} -> {100 * r[j]:5.2f}  (+{100 * (r[j] - r0[j]):.2f})")
print(f"-> {OUT}")
