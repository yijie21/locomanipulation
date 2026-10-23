"""TODO 33.1 / 33.1.4 (plan sections 1 and 1.5, D1, D17, D22): the body model's reach predictor -- what the controller
delivers for a reach command, read from a table of measured reaches, not from inverse kinematics.

`reach()` answers, in under a millisecond, "if this hand is sent to this point with this attitude from this posture,
how likely is it to land 3 cm or more off, and by how much": a distance-weighted k-nearest-neighbour lookup in the
dense envelope probe's grid (15,120 controlled reaches on the A800, 2026-10-21; scripts/rsi/body_envelope.py builds the
table). Validation (2026-10-23, scripts/rsi/body_envelope.py): held-out grid cells AUC 0.96, whole forward slices 0.89;
on the task records' first reaches of the current skill line (678 distinct pre-grasps of the run B / C skills, grasp
point mapped into the grid's frame) AUC 0.78 (side 0.76, top pinch 0.79) against 0.60 for the IK residual. What it
does not see: the furniture. Reaches 0.3-0.4 m from the shoulder at a table fail 51 % in the tasks while the free-
standing grid says 15 % -- the stance's clearance (TODO 33.12, 33.16) is a separate check, not this table's job. The
older skill line's records (the `pre` / `insert` skills, 965 reaches) score 0.59: their recorded grasp point is not
the pre-grasp target in the same way, so do not read this table's number as a verdict on them.

The table is runtime data: `outputs/scene_rsi_20261016/body/reach_knn_envelope.npz` (sandbox.py copies it into a
sandbox; `path=` / `--table` name another file). The loop's rules stand: a prediction is for choosing what to try; only a
closed-loop round is evidence (rules.md 7), and the driver, not a role, rebuilds the table (D1).

    from locomani.tasks.scene import body
    body.reach(x=0.45, y_out=0.05, z=0.90, att="level", pelvis=0.78, hand="right")
    -> {"p_miss": 0.08, "err_cm": 1.1, "n": 15, "far": 0.4, "inside": True}
    body.reach_world(grasp_xyz, pelvis_xyz, pelvis_quat_wxyz, hand="right", pinch=False)

    PYTHONPATH=src python -m locomani.tasks.scene.body reach --x 0.45 --y-out 0.05 --z 0.9 --att level --hand right
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
TABLE = REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_knn_envelope.npz"
ATT = {"level": 0.0, "down": 1.0, "tilt45": 2.0}
HANDS = {"left": 0.0, "right": 1.0}
INSIDE = 1.0  # mean standardised distance to the neighbours below this: the query lies within the grid


class Table:
    def __init__(self, path: Path = TABLE):
        d = np.load(path, allow_pickle=True)
        self.X, self.y_miss, self.y_err = d["X"].astype(float), d["y_miss"].astype(float), d["y_err"].astype(float)
        self.mu, self.sd = d["mu"].astype(float), d["sd"].astype(float)
        self.features = [str(f) for f in d["features"]]
        self.k = int(d["k"])
        self.miss_cm = float(d["miss_cm"])
        self.stand_z = float(d["stand_z"]) if "stand_z" in d.files else 0.78
        self.path = str(path)

    def query(self, row: np.ndarray, k: int | None = None) -> dict:
        k = min(k or self.k, len(self.X))
        z = (np.asarray(row, dtype=float) - self.mu) / self.sd
        d = np.sqrt(((self.X - z) ** 2).sum(1))
        idx = np.argpartition(d, k - 1)[:k]
        w = 1.0 / (d[idx] + 0.05)
        order = np.argsort(self.y_err[idx])
        cw = np.cumsum(w[order]) / w.sum()
        err = float(self.y_err[idx][order][min(int(np.searchsorted(cw, 0.5)), k - 1)])
        far = float(d[idx].mean())
        return dict(p_miss=round(float((w * self.y_miss[idx]).sum() / w.sum()), 3), err_cm=round(err, 1), n=int(k),
                    far=round(far, 2), inside=bool(far < INSIDE))  # fmt: skip


@lru_cache(maxsize=4)
def table(path: str | None = None) -> Table:
    return Table(Path(path) if path else TABLE)


def reach(x: float, y_out: float, z: float, att: str = "level", pelvis: float | None = None, hand: str = "right",
          path: str | None = None) -> dict:  # fmt: skip
    """The odds that a reach lands >= 3 cm off and the error to expect. `x` forward and `y_out` outward of the hand's
    own side, metres from the pelvis in its heading frame; `z` the target's height above the floor; `att` level /
    down (a top pinch) / tilt45; `pelvis` the pelvis height the reach starts from (None: standing)."""
    t = table(path)
    row = [x, y_out, z, ATT[att], t.stand_z if pelvis is None else float(pelvis), HANDS[hand]]
    return dict(t.query(np.array(row)), x=round(x, 3), y_out=round(y_out, 3), z=round(z, 3), att=att,
                pelvis=round(t.stand_z if pelvis is None else float(pelvis), 3), hand=hand)  # fmt: skip


def yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def grid_coords(point_xyz, pelvis_xyz, pelvis_quat_wxyz, hand: str) -> tuple[float, float, float]:
    """A world point as (x forward, y outward of `hand`'s side, z above the floor) in the pelvis heading frame."""
    p = np.asarray(point_xyz, dtype=float) - np.asarray(pelvis_xyz, dtype=float)
    yaw = yaw_of(pelvis_quat_wxyz)
    c, s = math.cos(-yaw), math.sin(-yaw)
    fx, fy = c * p[0] - s * p[1], s * p[0] + c * p[1]
    out = -fy if hand == "right" else fy
    return float(fx), float(out), float(point_xyz[2])


def reach_world(point_xyz, pelvis_xyz, pelvis_quat_wxyz, hand: str = "right", pinch: bool = False,
                att: str | None = None, path: str | None = None) -> dict:  # fmt: skip
    """`reach` for a world point and the pelvis pose it is reached from (a scene skill's view)."""
    x, y_out, z = grid_coords(point_xyz, pelvis_xyz, pelvis_quat_wxyz, hand)
    return reach(x, y_out, z, att or ("down" if pinch else "level"), float(pelvis_xyz[2]), hand, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None, help=f"the reach table (default {TABLE})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reach", help="one query in the pelvis heading frame")
    r.add_argument("--x", type=float, required=True, help="forward, m")
    r.add_argument("--y-out", type=float, default=0.05, help="outward of the hand's own side, m")
    r.add_argument("--z", type=float, required=True, help="height above the floor, m")
    r.add_argument("--att", choices=sorted(ATT), default="level")
    r.add_argument("--pelvis", type=float, default=None, help="pelvis height, m (default standing)")
    r.add_argument("--hand", choices=sorted(HANDS), default="right")
    w = sub.add_parser("world", help="a world point from a pelvis pose")
    w.add_argument("--point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    w.add_argument("--pelvis", type=float, nargs=7, required=True, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    w.add_argument("--hand", choices=sorted(HANDS), default="right")
    w.add_argument("--pinch", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "reach":
        out = reach(a.x, a.y_out, a.z, a.att, a.pelvis, a.hand, path=a.table)
    else:
        out = reach_world(a.point, a.pelvis[:3], a.pelvis[3:], a.hand, a.pinch, path=a.table)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
