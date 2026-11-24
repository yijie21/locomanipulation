"""The real arrival errors the imagined Motor draws from (TODO 45.8): one row per walk and per reach of the real rounds
that were recorded before shadow mode (v7_kin_gf/manifest.json), never of the drafts the result is read on.

    PYTHONPATH=src:scripts/rsi $VENV_SIM_PY arrival_data.py <manifest.json> <out.npz>   # add --check for the held-out test

Reach: the target in the heading frame of the pelvis where the reach began, how far the jaw had to travel, the wrist's
pitch, how far off the same hand's last reach ended, whether the jaw was open; the error is the measured jaw centre minus the target when the reach ended, heading frame,
with the left hand mirrored so both hands share one pool. Walk: how far, whether it ends on a given heading; the error is
the pelvis minus the goal when the walk ended.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imagine as IM  # noqa: E402

FAMILY = "grasp_floor"


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot2(v, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def build(manifest: Path, out: Path) -> int:
    IM.setup_paths()
    import family as F

    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.motor import JAW_OPEN, SIDES, WRIST

    world = IM.make_world(F.load(FAMILY))
    kp = world.motor.kp

    def jaw(robot, joints, side):
        p, q = kp(robot[:3], robot[3:7], joints)[WRIST[side]]
        p, q = np.asarray(p, float).reshape(3), np.asarray(q, float).reshape(4)
        return p + R.mat(q) @ R.OFF, R.mat(q)

    rx, ro, re_, rs, wd, wf, we, wy, ws = [], [], [], [], [], [], [], [], []
    for job in json.loads(manifest.read_text()):
        src = re.search(r"/([^/]+)/outputs/", job["real"]).group(1)
        for label in job["labels"]:
            rec = json.loads((Path(job["real"]) / f"{label}.json").read_text())
            f = Path(job["real"]) / f"{label}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            t = z["t"]
            at = lambda v: int(np.clip(np.searchsorted(t, v, side="right") - 1, 0, len(t) - 1))  # noqa: E731
            last_err = {}  # how far off the last reach of each hand ended
            for c in rec.get("calls") or []:
                i0, i1 = at(c["t0"]), at(c["t1"])
                if c["name"] == "walk_to" and c["t1"] <= t[-1] + 0.05:
                    goal = np.asarray(c["args"][0], float)
                    final = len(c["args"]) > 1 and c["args"][1] is not None
                    end_yaw = yaw_of(z["robot"][i1][3:7])
                    wd.append(float(np.linalg.norm(goal - z["robot"][i0][:2])))
                    wf.append(final)
                    we.append(rot2(z["robot"][i1][:2] - goal, -end_yaw))
                    wy.append(
                        float(np.arctan2(np.sin(end_yaw - c["args"][1]), np.cos(end_yaw - c["args"][1])))
                        if final
                        else 0.0
                    )
                    ws.append(src)
                elif c["name"] == "reach" and c["args"][0] in SIDES and c["t1"] <= t[-1] + 0.05:
                    side, goal = c["args"][0], np.asarray(c["args"][1], float)
                    rot = c["args"][2] if len(c["args"]) > 2 else c["kwargs"].get("rot")
                    p0, yaw0 = z["robot"][i0][:3], yaw_of(z["robot"][i0][3:7])
                    jaw0, Rw0 = jaw(z["robot"][i0], z["joints"][i0], side)
                    jaw1, _ = jaw(z["robot"][i1], z["joints"][i1], side)
                    sign = -1.0 if side == SIDES[0] else 1.0
                    g = rot2(goal[:2] - p0[:2], -yaw0)
                    pitch = float((Rw0 if rot is None else np.asarray(rot, float))[2, 0])
                    rx.append(
                        [
                            g[0],
                            sign * g[1],
                            goal[2] - p0[2],
                            float(np.linalg.norm(goal - jaw0)),
                            pitch,
                            last_err.get(side, 0.0),
                        ]
                    )
                    ro.append(abs(float(z["jaw"][i0][SIDES.index(side)]) - JAW_OPEN) < 1e-3)
                    e = rot2((jaw1 - goal)[:2], -yaw_of(z["robot"][i1][3:7]))
                    re_.append([e[0], sign * e[1], float(jaw1[2] - goal[2])])
                    rs.append(src)
                    last_err[side] = float(np.linalg.norm(jaw1 - goal))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, r_x=np.array(rx, float), r_open=np.array(ro, bool), r_e=np.array(re_, float), r_src=np.array(rs),
             w_dist=np.array(wd, float), w_final=np.array(wf, bool), w_e=np.array(we, float), w_yaw=np.array(wy, float), w_src=np.array(ws))  # fmt: skip
    print(f"reaches {len(rx)}, walks {len(wd)} -> {out}")
    return 0


def check(out: Path) -> int:
    """Does the similar-command draw know where a reach goes wrong? Each reach against reaches of the other sandboxes."""
    from arrival_motor import K_NEAR, REACH_SCALE

    z = np.load(out)
    x, o, e, s = z["r_x"], z["r_open"], np.linalg.norm(z["r_e"], axis=1), z["r_src"]
    pred = np.zeros(len(x))
    for i in range(len(x)):
        idx = np.flatnonzero((o == o[i]) & (s != s[i]))
        d = np.linalg.norm((x[idx] - x[i]) / REACH_SCALE, axis=1)
        near = idx[np.argpartition(d, K_NEAR)[:K_NEAR]]
        pred[i] = float(np.mean(e[near] > 0.05))
    y = e > 0.05
    a, b = pred[y], pred[~y]
    auc = float(np.mean([(a > v).mean() + 0.5 * (a == v).mean() for v in b]))
    print(f"share above 5 cm {y.mean():.3f}; held-out AUC of the draw's share above 5 cm {auc:.3f}")
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (pred >= lo) & (pred < hi)
        print(
            f"  drawn share {lo:.1f}-{hi:.1f}: n {m.sum():5d}, real share above 5 cm {y[m].mean() if m.any() else float('nan'):.2f}"
        )
    w = np.linalg.norm(z["w_e"], axis=1)
    print(
        f"walk error cm: median {100 * np.median(w):.2f}, p90 {100 * np.percentile(w, 90):.2f}; final walks median {100 * np.median(w[z['w_final']]):.2f}"
    )
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--check"]
    rc = build(Path(args[0]), Path(args[1])) if not Path(args[1]).exists() else 0
    sys.exit(rc or (check(Path(args[1])) if "--check" in sys.argv else 0))
