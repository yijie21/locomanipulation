"""selfmodel_via_20261120: a reach's random path through several via points, the real cuRobo paths to replay, and a path's
shape features. Numpy only, no simulator: gen_reach.py (REACH_VIA, REACH_VIA_REPLAY), via_cover.py and the test import it.

Frames. "Heading frame": x ahead and y left at the pelvis yaw of the command time, origin on the floor under the pelvis, z
above the floor (gen_reach.py's target frame). A bend's offset lies in the plane across the straight line from the hand's
start to the goal, at the angle phi from `out` (away from the body's midline for the acting hand: to the right for the right
hand) towards `up`: 0 out, 90 up, 180 in, 270 deg down.

random_path draws (docs/handoff/notebook-20261120/selfmodel_via_20261120.md):
  bends     0 / 1 / 2 / 3 with probability 0.15 / 0.35 / 0.30 / 0.20 (0: the straight reach of the older datasets)
  a bend    at 15-85 % of the straight line, phi uniform over the circle, 3-35 cm off the line
  lead-in   10 % of the paths: first 2-10 cm back (against the heading) or 2-15 cm up, half each
  wrist     from the start attitude to the goal's along the path; at each bend a further turn of 0-90 deg about a random axis
            (0-60 in the trial of 2026-11-20; widened after its coverage check: the real paths turn the wrist 89-209 deg)
  speed     half 0.15 m/s (Motor's default), half uniform 0.08-0.25 m/s
The knots (start, lead-in, bends, goal) are joined by a centripetal Catmull-Rom curve sampled every cm, then simplified by
Douglas-Peucker to within 1 cm (curobo_avoid_20261120/via2.py does the same to a planned path); more than 16 via points:
the tolerance grows by half until they fit. A path is drawn again (30 tries, then the straight line) when it goes under
5 cm above the floor, into the trunk (16 cm around the pelvis axis above 0.25 m; not within the first 6 cm, where the
hand already is), into the legs of a crouch, more than 25 cm across the midline, out of the box the commands are drawn from,
or more than 10 cm further from the shoulder than the farther of its two ends.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np

BENDS_P = (0.15, 0.35, 0.30, 0.20)
BEND_U, BEND_M, TURN_DEG = (0.15, 0.85), (0.03, 0.35), 90.0
LEAD_P, LEAD_BACK_M, LEAD_UP_M = 0.10, (0.02, 0.10), (0.02, 0.15)
SPEED, SPEED_BOX = 0.15, (0.08, 0.25)
TOL, VIA_MAX, TRIES, DS = 0.01, 16, 30, 0.01
FLOOR_Z, BODY_R, BODY_Z, CROSS_Y, FREE_M, OVER_M = 0.05, 0.16, 0.25, 0.25, 0.06, 0.10
BOX = (np.array([-0.30, -0.85, 0.05]), np.array([0.95, 0.85, 1.55]))
LEGS = (np.array([-0.10, -0.17]), np.array([0.30, 0.17]))  # x, y of a crouch's thighs and knees
REASONS = ("ok", "floor", "trunk", "legs", "midline", "box", "reach", "knots")
GEN_KEYS = ("bends", "lead", "lead_m", "speed") + tuple(
    f"{k}{i}" for i in (1, 2, 3) for k in ("u", "phi", "mag", "turn")
)


def expm(v) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def logm(R) -> np.ndarray:
    """Rotation vector of a rotation matrix (angle in 0..pi)."""
    R = np.asarray(R, float)
    a = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    if a < 1e-9:
        return np.zeros(3)
    if math.pi - a < 1e-5:  # the axis from the symmetric part: the antisymmetric one vanishes at pi
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        k = A[i] / math.sqrt(max(float(A[i, i]), 1e-12))
        return a * k / np.linalg.norm(k)
    return a / (2.0 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def turn_deg(Ra, Rb) -> float:
    return float(np.degrees(np.linalg.norm(logm(np.asarray(Ra).T @ np.asarray(Rb)))))


def to_h(x, p, Rh) -> np.ndarray:
    """World points -> heading frame of a pelvis at p with heading rotation Rh (z stays the height above the floor)."""
    x = np.atleast_2d(np.asarray(x, float))
    return np.c_[(x[:, :2] - np.asarray(p)[:2]) @ np.asarray(Rh)[:2, :2], x[:, 2]]


def from_h(xh, p, Rh) -> np.ndarray:
    xh = np.atleast_2d(np.asarray(xh, float))
    return np.c_[np.asarray(p)[:2] + xh[:, :2] @ np.asarray(Rh)[:2, :2].T, xh[:, 2]]


def seg_dist(X, a, b) -> np.ndarray:
    ab = b - a
    u = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(X - (a + u[:, None] * ab), axis=1)


def rdp(X, tol: float) -> list[int]:
    """Douglas-Peucker: the indices of X kept so that no point is further than tol from the kept polyline."""
    keep, todo = [0, len(X) - 1], [(0, len(X) - 1)]
    while todo:
        i, j = todo.pop()
        if j <= i + 1:
            continue
        d = seg_dist(X[i + 1 : j], X[i], X[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep.append(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(set(keep))


def poly_dev(X, idx) -> float:
    """The points' largest distance from the polyline through X[idx]."""
    return max(float(seg_dist(X[a : b + 1], X[a], X[b]).max()) for a, b in zip(idx[:-1], idx[1:]))


def catmull(K, ds: float = DS):
    """Centripetal Catmull-Rom curve through the knots K (n, 3), a point about every ds; the knots' indices in it."""
    K = np.asarray(K, float)
    P = np.vstack([2 * K[0] - K[1], K, 2 * K[-1] - K[-2]])
    out, at = [K[0]], [0]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1 : i + 3]
        t1 = math.sqrt(float(np.linalg.norm(p1 - p0)))
        t2 = t1 + math.sqrt(float(np.linalg.norm(p2 - p1)))
        t3 = t2 + math.sqrt(float(np.linalg.norm(p3 - p2)))
        n = max(2, int(math.ceil(1.5 * float(np.linalg.norm(p2 - p1)) / ds)))
        for t in np.linspace(t1, t2, n + 1)[1:]:
            a1 = (t1 - t) / t1 * p0 + t / t1 * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / t2 * a1 + t / t2 * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
        at.append(len(out) - 1)
    return np.array(out), at


def across(e, out0, ahead):
    """Unit vectors (out, up) across the direction e: `up` the vertical's part across e (the heading's, for a vertical e),
    `out` at right angles to both, on out0's side."""
    z = np.array([0.0, 0.0, 1.0])
    ref = z if abs(float(e @ z)) < 0.95 else np.asarray(ahead, float)
    up = ref - float(ref @ e) * e
    up /= np.linalg.norm(up)
    out = np.cross(up, e)
    return (out if float(out @ out0) >= 0 else -out), up


def blocked(Xh, s, sd: float, shoulder_h, legs_z: float | None) -> int:
    """Why the sampled path Xh (heading frame; s its arc length) is not run: an index into REASONS, 0 for a path to run."""
    x, y, z = Xh.T
    if (z < FLOOR_Z).any():
        return 1
    if ((np.hypot(x, y) < BODY_R) & (z > BODY_Z) & (s > FREE_M)).any():
        return 2
    if (
        legs_z is not None
        and (
            (x > LEGS[0][0])
            & (x < LEGS[1][0])
            & (y > LEGS[0][1])
            & (y < LEGS[1][1])
            & (z < legs_z + 0.05)
            & (s > FREE_M)
        ).any()
    ):
        return 3
    if (y * sd > CROSS_Y).any():
        return 4
    if ((Xh < BOX[0]) | (Xh > BOX[1])).any():
        return 5
    r = np.linalg.norm(Xh - shoulder_h, axis=1)
    if r.max() > max(r[0], r[-1]) + OVER_M:
        return 6
    return 0


def random_path(rng, a, b, Ra, Rb, p, Rh, sd: float, shoulder, pelvis_z: float | None = None) -> dict:
    """One random path from the hand's start a (attitude Ra) to the goal b (Rb), world. p, Rh: the pelvis position and
    heading rotation at the command; sd 1 for the right hand, -1 for the left; shoulder: the acting shoulder (world);
    pelvis_z: the commanded pelvis height of a crouch. -> p (k, 3) and R (k, 3, 3), the via points and the wrist rotations
    there (world, the ends not included); speed (m/s); gen (GEN_KEYS, angles in deg); tries; why (REASONS counts of the
    paths drawn and dropped); tol; length (m, the curve's)."""
    a, b, Ra, Rb, p, Rh = (np.asarray(v, float) for v in (a, b, Ra, Rb, p, Rh))
    L = float(np.linalg.norm(b - a))
    e = (b - a) / max(L, 1e-9)
    out, up = across(e, -sd * Rh[:, 1], Rh[:, 0])
    sh = to_h(shoulder, p, Rh)[0]
    why = np.zeros(len(REASONS), int)
    for tries in range(1, TRIES + 2):
        straight = tries > TRIES or L < 0.05
        nb = 0 if straight else int(rng.choice(4, p=BENDS_P))
        lead = 0 if straight or rng.random() >= LEAD_P else 1 if rng.random() < 0.5 else 2
        lead_m = float(rng.uniform(*(LEAD_BACK_M if lead == 1 else LEAD_UP_M))) if lead else 0.0
        u = np.sort(rng.uniform(*BEND_U, nb))
        phi, mag, turn = rng.uniform(0, 2 * math.pi, nb), rng.uniform(*BEND_M, nb), rng.uniform(0, TURN_DEG, nb)
        axes = rng.normal(size=(nb, 3))
        speed = SPEED if rng.random() < 0.5 else float(rng.uniform(*SPEED_BOX))
        K, bend = [a], [False]
        if lead:
            K.append(a - lead_m * Rh[:, 0] if lead == 1 else a + np.array([0.0, 0.0, lead_m]))
            bend.append(False)
        for k in range(nb):
            K.append(a + u[k] * L * e + mag[k] * (math.cos(phi[k]) * out + math.sin(phi[k]) * up))
            bend.append(True)
        K.append(b)
        bend.append(False)
        K = np.array(K)
        if not straight and float(np.linalg.norm(np.diff(K, axis=0), axis=1).min()) < 0.01:
            why[7] += 1
            continue
        X, at = (K, [0, 1]) if len(K) == 2 else catmull(K)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
        bad = 0 if straight else blocked(to_h(X, p, Rh), s, sd, sh, pelvis_z)
        if bad:
            why[bad] += 1
            continue
        tol = TOL
        idx = rdp(X, tol)
        while len(idx) - 2 > VIA_MAX:
            tol *= 1.5
            idx = rdp(X, tol)
        D = logm(Ra.T @ Rb)  # the knots' attitudes: along the way to the goal's, a bend's turned further
        Kr = [Ra @ expm(s[i] / max(s[-1], 1e-9) * D) for i in at]
        j = 0
        for k in range(len(K)):
            if bend[k]:
                Kr[k] = Kr[k] @ expm(axes[j] / np.linalg.norm(axes[j]) * math.radians(turn[j]))
                j += 1
        Kr[-1] = Rb
        R = []
        for i in idx[1:-1]:
            k = min(int(np.searchsorted(at, i, side="right")) - 1, len(at) - 2)
            w = (s[i] - s[at[k]]) / max(s[at[k + 1]] - s[at[k]], 1e-9)
            R.append(Kr[k] @ expm(w * logm(Kr[k].T @ Kr[k + 1])))
        gen = np.zeros(len(GEN_KEYS), np.float32)
        gen[:4] = nb, lead, lead_m, speed
        for k in range(nb):
            gen[4 + 4 * k : 8 + 4 * k] = u[k], math.degrees(phi[k]), mag[k], turn[k]
        return dict(
            p=X[idx[1:-1]], R=np.array(R).reshape(-1, 3, 3), speed=speed, gen=gen, tries=tries, why=why, tol=tol,
            length=float(s[-1]),
        )  # fmt: skip
    raise AssertionError("random_path: the straight line is always run")


def load_routes(spec: str) -> list[dict]:
    """REACH_VIA_REPLAY's paths: spec "<via dir>:<prep dir>[,<via dir>:<prep dir> ...]" (via2.py's via_<NAME>.npz, prep2.py's
    prep_<NAME>.npz). Each path in the heading frame of its reach's start, right hand: name, start / start_R (the hand
    before the reach), p / R (the via points), goal / goal_R (the bench's pregrasp command), speed (None: Motor's default),
    pelvis_z (m, the pelvis above the floor at the start)."""
    routes = []
    for pair in spec.split(","):
        vd, pd = pair.split(":")
        for f in sorted(glob.glob(str(Path(vd) / "via_*.npz"))):
            name = Path(f).stem[4:]
            pf = Path(pd) / f"prep_{name}.npz"
            if not pf.exists():
                continue
            v, pr = np.load(f), np.load(pf, allow_pickle=True)
            o, Rp = pr["pelvis_p"], pr["pelvis_R"]
            yaw = math.atan2(Rp[1, 0], Rp[0, 0])
            c, s = math.cos(yaw), math.sin(yaw)
            Rh = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            routes.append(
                dict(
                    name=name,
                    start=to_h(pr["jaw_p"], o, Rh)[0],
                    start_R=Rh.T @ pr["wrist_R"],
                    p=to_h(v["p"], o, Rh) if len(v["p"]) else np.zeros((0, 3)),
                    R=np.array([Rh.T @ r for r in v["R"]]).reshape(-1, 3, 3),
                    goal=to_h(pr["goal_jaw"], o, Rh)[0],
                    goal_R=Rh.T @ pr["goal_R"],
                    speed=float(v["speed"]) if "speed" in v.files else None,
                    pelvis_z=float(o[2]),
                )  # fmt: skip
            )
    return routes


FEATURES = ("n_via", "straight", "ratio", "off_x", "off_out", "off_z", "turn", "ax", "a_out", "az", "bx", "b_out", "bz")


def shape(pts_h, Rs, sd: float = 1.0) -> np.ndarray:
    """A path's shape (FEATURES): pts_h (k + 2, 3) the start, the via points and the goal in the heading frame, Rs the
    attitudes there (any frame); sd -1 mirrors a left hand's path onto the right's. n_via; the straight distance (m); the
    path's length over it; the point furthest off the straight line: its offset ahead, outwards, up (m); the wrist's turn
    summed along the path (deg); the start and the goal (ahead, outwards, up)."""
    P = np.asarray(pts_h, float) * np.array([1.0, -sd, 1.0])  # y -> outwards
    a, b = P[0], P[-1]
    ab = b - a
    u = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    off = P - (a + u[:, None] * ab)
    i = int(np.argmax(np.linalg.norm(off, axis=1)))
    straight = float(np.linalg.norm(ab))
    length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    turn = sum(turn_deg(Rs[k], Rs[k + 1]) for k in range(len(Rs) - 1))
    return np.r_[len(P) - 2, straight, length / max(straight, 1e-9), off[i], turn, a, b]
