"""curobo_attrib_20261120: the via files of the attribution groups B-F, from the cuRobo via files (A1, via2.py) and the
reach starts (prep2.py).

    python3 vias_e1.py <A1 via dir> <prep dir> <out root>

Per episode with an A1 via file (start s = the jaw at the reach's start, g = the wanted pregrasp jaw, R0 / Rg the wrist
rotations there; K = A1's via count):
  B  line_seg     K points evenly on s -> g; attitude slerp(R0, Rg) by path fraction; speed so that the motor's leg
                  durations (max(reach_min_s 0.6, length / speed)) sum to A1's
  C  plan_pos     A1's points; attitude slerp by A1's path fraction (default speed)
  D  line_planrot B's points and speed; A1's rotations
  E  template     s - 8 cm along the robot's heading, then that at g's height; attitude slerp by path fraction
  F  mismatch     the next episode's (cyclic) A1 points less its start, turned by the heading difference, from s; attitude slerp
-> <out root>/via_<G>/via_<NAME>.npz (p, R, goal, [speed]); a line per episode and group (path length, duration)
"""

import sys
from pathlib import Path

import numpy as np

A1, PREP, OUT = (Path(x) for x in sys.argv[1:4])
SPEED, MIN_S = 0.15, 0.6  # motor reach_speed, reach_min_s


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return th * w


def expmap(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def slerp(R0, Rg, u):
    return R0 @ expmap(u * rotvec(R0.T @ Rg))


def fractions(s, P, g):  # each point's path fraction along s -> P -> g
    X = np.vstack([s, P, g])
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    return L[1:-1] / L[-1], L[-1]


def duration(s, P, g, speed=SPEED):
    X = np.vstack([s, P, g])
    return float(sum(max(MIN_S, np.linalg.norm(b - a) / speed) for a, b in zip(X[:-1], X[1:])))


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rz(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


names = sorted(f.name[4:-4] for f in A1.glob("via_*.npz"))
data = {}
for n in names:
    v, pr = np.load(A1 / f"via_{n}.npz"), np.load(PREP / f"prep_{n}.npz")
    data[n] = dict(
        p=v["p"], R=v["R"], s=pr["jaw_p"], g=pr["goal_jaw"], R0=pr["wrist_R"], Rg=pr["goal_R"], Rp=pr["pelvis_R"]
    )
for G in "BCDEF":
    (OUT / f"via_{G}").mkdir(parents=True, exist_ok=True)
for i, n in enumerate(names):
    d = data[n]
    s, g, R0, Rg, P1 = d["s"], d["g"], d["R0"], d["Rg"], d["p"]
    K = len(P1)
    T1 = duration(s, P1, g)
    out = {}
    # B, D: the straight line, K points, timed as A1
    line = s + np.outer(np.arange(1, K + 1) / (K + 1), g - s)
    leg = float(np.linalg.norm(g - s)) / (K + 1)
    sp = leg / max(MIN_S, T1 / (K + 1))
    fr, _ = fractions(s, line, g)
    out["B"] = dict(p=line, R=np.array([slerp(R0, Rg, u) for u in fr]), speed=sp)
    out["D"] = dict(p=line, R=d["R"], speed=sp)
    fr, _ = fractions(s, P1, g)
    out["C"] = dict(p=P1, R=np.array([slerp(R0, Rg, u) for u in fr]))
    h = np.r_[d["Rp"][:2, 0], 0.0]
    h /= np.linalg.norm(h)
    e1 = s - 0.08 * h
    e2 = np.r_[e1[:2], g[2]]
    PE = np.array([e1, e2])
    fr, _ = fractions(s, PE, g)
    out["E"] = dict(p=PE, R=np.array([slerp(R0, Rg, u) for u in fr]))
    m = data[names[(i + 1) % len(names)]]
    Rd = rz(yaw_of(d["Rp"]) - yaw_of(m["Rp"]))
    PF = s + (m["p"] - m["s"]) @ Rd.T
    fr, _ = fractions(s, PF, g)
    out["F"] = dict(p=PF, R=np.array([slerp(R0, Rg, u) for u in fr]))
    line_s = [n, "A1", K, round(100 * fractions(s, P1, g)[1], 1), round(T1, 2)]
    for G, o in out.items():
        np.savez(
            OUT / f"via_{G}" / f"via_{n}.npz",
            p=o["p"],
            R=o["R"],
            goal=g,
            **({"speed": o["speed"]} if "speed" in o else {}),
        )
        L = fractions(s, o["p"], g)[1]
        line_s += [G, len(o["p"]), round(100 * L, 1), round(duration(s, o["p"], g, o.get("speed", SPEED)), 2)]
    print(*line_s)
