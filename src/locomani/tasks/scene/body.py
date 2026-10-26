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
    body.best_level("pick.STANCE_MIN_D_M", task="pick_DishSoap_1")   # a measured best level, or None (TODO 33.1.7)
    body.curves(factor="spawn_bearing")                              # the design probes' curves, tagged

    PYTHONPATH=src python -m locomani.tasks.scene.body reach --x 0.45 --y-out 0.05 --z 0.9 --att level --hand right
    PYTHONPATH=src python -m locomani.tasks.scene.body best --name pick.STANCE_MIN_D_M
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


def attitude(rot) -> str:
    """The grid's attitude class of a commanded wrist rotation. The jaw approaches along the rotation's first column
    (the envelope probe commanded level = identity, down = rot_y(90 deg), tilt45 = rot_y(45 deg)); pitched below the
    horizontal by under 22.5 deg it is level, under 67.5 tilt45, else down."""
    a = np.asarray(rot, dtype=float)[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(a[2]) / (float(np.linalg.norm(a)) or 1.0)))))
    return "level" if pitch < 22.5 else "tilt45" if pitch < 67.5 else "down"


def reach_note(path: str | None = None):
    """TODO 33.1.2: the motor's `reach_note` hook -- before every one-hand reach moves, the body model's prediction for
    the command (pred_p_miss, pred_err_cm, pred_att, pred_inside, pred_open: the jaw was open, as on the grid) goes
    into the reach's record. None when the table cannot be read (the reach records then carry no prediction)."""
    try:
        table(path)
    except (OSError, KeyError, ValueError):
        return None

    def note(side, pos, rot, pelvis_pos, pelvis_quat, pelvis_z, jaw_open) -> dict:
        x, y_out, z = grid_coords(pos, pelvis_pos, pelvis_quat, side)
        att = attitude(rot)
        start_z = float(pelvis_z) if pelvis_z is not None else float(pelvis_pos[2])
        r = reach(x, y_out, z, att, start_z, side, path)
        return dict(pred_p_miss=r["p_miss"], pred_err_cm=r["err_cm"], pred_att=att, pred_inside=r["inside"],
                    pred_open=bool(jaw_open))  # fmt: skip

    return note


BIAS = TABLE.parent / "reach_bias_knn.npz"
BIAS_INSIDE = 1.0  # the neighbours' mean standardised distance under this: the reach lies inside the measured grid
BIAS_MAX_M = 0.10  # a predicted miss is capped at this (the grid's misses after the hold: 0-20 cm, median under 2)


@lru_cache(maxsize=4)
def bias_table(path: str | None = None) -> dict:
    d = np.load(Path(path) if path else BIAS, allow_pickle=True)
    return dict(X=d["X"].astype(float), E=d["E"].astype(float), mu=d["mu"].astype(float), sd=d["sd"].astype(float),
                k=int(d["k"]))  # fmt: skip


def bias(x: float, y_out: float, z: float, att: str = "level", pelvis: float | None = None, hand: str = "right",
         path: str | None = None) -> dict:  # fmt: skip
    """TODO 33.1.4 (2): where the controller puts the jaw centre against the target, on open floor -- the miss vector
    (x ahead, y left, z up, metres, the pelvis heading frame) a distance-weighted k-NN reads off the free-space grid of
    2,160 reaches (scripts/rsi/reach_bias_probe.py, 2026-10-26), with `inside` (the query lies within the grid). The
    grid's closed-loop check: moving the target by minus this halved the miss (3.07 -> 1.57 cm; wrist down 6.6 -> 2.6
    cm). It has no furniture in it: a reach beside a table misses for other reasons too."""
    b = bias_table(path)
    row = np.array([x, y_out, z, ATT[att], table().stand_z if pelvis is None else float(pelvis), HANDS[hand]])
    zq = (row - b["mu"]) / b["sd"]
    dist = np.sqrt(((b["X"] - zq) ** 2).sum(1))
    k = min(b["k"], len(dist))
    idx = np.argpartition(dist, k - 1)[:k]
    w = 1.0 / (dist[idx] + 0.05)
    e = (w[:, None] * b["E"][idx]).sum(0) / w.sum()
    n = float(np.linalg.norm(e))
    if n > BIAS_MAX_M:
        e = e * (BIAS_MAX_M / n)
    far = float(dist[idx].mean())
    return dict(miss=[round(float(v), 4) for v in e], miss_cm=round(100 * min(n, BIAS_MAX_M), 2), far=round(far, 2),
                inside=bool(far < BIAS_INSIDE))  # fmt: skip


def reach_offset(path: str | None = None):
    """TODO 33.1.4 (2): the motor's `reach_offset` hook (MotorConfig.reach_compensation, default off) -- before a
    one-hand reach moves, the world vector to take off its jaw target so the controller's free-space miss lands it on
    the target: the k-NN miss for the command's grid coordinates and attitude, turned into the world by the pelvis
    heading. None (no offset) outside the grid or when the table cannot be read."""
    try:
        bias_table(path)
    except (OSError, KeyError, ValueError):
        return None

    def offset(side, pos, rot, pelvis_pos, pelvis_quat, pelvis_z):
        x, y_out, z = grid_coords(pos, pelvis_pos, pelvis_quat, side)
        start_z = float(pelvis_z) if pelvis_z is not None else float(pelvis_pos[2])
        b = bias(x, y_out, z, attitude(rot), start_z, side, path)
        if not b["inside"]:
            return None
        yaw = yaw_of(pelvis_quat)
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey, ez = b["miss"]
        return np.array([c * ex - s * ey, s * ex + c * ey, ez])

    return offset


CURVES = TABLE.parent / "curves.jsonl"
BEST_MIN_N = 6  # rounds a level needs, pooled over the curves, before best_level names it
BEST_MARGIN = 0.15  # ... and the success rate it must have over every other measured level


def curves(name: str | None = None, factor: str | None = None, task: str | None = None, machine: str | None = None,
           controller: str | None = None, path: str | None = None) -> list[dict]:  # fmt: skip
    """TODO 33.1.7 (plan section 1.2, D17): the measured curves -- a design probe's per-level outcome (rules.md 35),
    tagged with where it was measured (machine, controller, the incumbent's skills, scope: `skill` for a parameter,
    `start` for a start factor) -- filtered by parameter `name`, `factor`, a `task` among the probe's, `machine` and
    `controller`. Read from body/curves.jsonl, the snapshot the sandbox was built with: fixed for a whole run, so a
    skill that asks behaves the same in every round of it (a run's own probes reach the roles, not the skills, until
    scripts/rsi/body_curves.py gathers them). Empty when there is no file."""
    p = Path(path) if path else CURVES
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if name is not None and c.get("name") != name:
            continue
        if factor is not None and c.get("factor") != factor:
            continue
        if task is not None and task not in (c.get("tasks") or []):
            continue
        if machine is not None and c.get("machine") not in (None, machine):
            continue
        if controller is not None and c.get("controller") not in (None, controller):
            continue
        out.append(c)
    return out


def best_level(name: str, task: str | None = None, path: str | None = None, **kw) -> dict | None:
    """TODO 33.1.7: the level of parameter `name` the measured curves say works best, pooled over every matching curve
    (`curves`, factor param): its level, rate and rounds, the other levels, and how many curves and incumbents went in.
    None unless it has BEST_MIN_N rounds and beats every other measured level by BEST_MARGIN -- a skill then keeps its
    own default. A skill asks this instead of writing the number down (D17); the gate still decides what stays."""
    pool: dict[float, list[int]] = {}
    cs = curves(name=name, factor="param", task=task, path=path, **kw)
    for c in cs:
        for lv in c.get("levels") or []:
            if lv.get("n"):
                n_ok = pool.setdefault(float(lv["level"]), [0, 0])
                n_ok[0] += int(lv["n"])
                n_ok[1] += int(lv.get("ok") or 0)
    if len(pool) < 2:
        return None
    rates = {lv: ok / n for lv, (n, ok) in pool.items()}
    best = max(rates, key=lambda lv: (rates[lv], pool[lv][0]))
    if pool[best][0] < BEST_MIN_N or any(rates[best] - r < BEST_MARGIN for lv, r in rates.items() if lv != best):
        return None
    return dict(level=best, rate=round(rates[best], 3), n=pool[best][0],
                levels={lv: dict(rate=round(rates[lv], 3), n=pool[lv][0]) for lv in sorted(pool)}, curves=len(cs),
                incumbents=sorted({c.get("incumbent") for c in cs if c.get("incumbent")}))  # fmt: skip


MISS_CM = 3.0  # a reach that ends this far off counts as missed (the grid's label)
RISKY = 0.5  # a reach predicted this likely to miss is one the skill could have known about


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def score(rounds) -> dict:
    """TODO 33.1.2: the body model's predictions against what was delivered, over round records (dicts with `ok` and
    `steps`). Counted: reach steps that carry a prediction and a delivered error with the jaw open (empty-handed, as on
    the grid). AUC of pred_p_miss for a miss (jaw_err_cm >= MISS_CM); mean predicted vs delivered error; and the failed
    rounds with a reach predicted risky (pred_p_miss >= RISKY) -- the misses a skill could have seen coming."""
    p, err, pred_err, missed, risky_failed, failed, scored = [], [], [], [], 0, 0, 0
    for r in rounds:
        steps = [s for s in r.get("steps") or [] if isinstance(s, dict)]
        reaches = [s for s in steps if s.get("pred_p_miss") is not None and s.get("jaw_err_cm") is not None
                   and s.get("pred_open", True)]  # fmt: skip
        if not reaches:
            continue
        scored += 1
        for s in reaches:
            p.append(float(s["pred_p_miss"]))
            err.append(float(s["jaw_err_cm"]))
            pred_err.append(float(s["pred_err_cm"]))
            missed.append(float(s["jaw_err_cm"]) >= MISS_CM)
        if not r.get("ok"):
            failed += 1
            risky_failed += any(float(s["pred_p_miss"]) >= RISKY for s in reaches)
    mean = lambda xs: round(float(np.mean(xs)), 2) if xs else None  # noqa: E731
    return dict(rounds=scored, reaches=len(p), missed=int(sum(missed)), auc=_auc(p, missed),
                pred_err_cm=mean(pred_err), err_cm=mean(err), failed_rounds=failed,
                failed_with_risky_reach=risky_failed)  # fmt: skip


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
    b = sub.add_parser("best", help="the measured best level of a parameter, from the design probes' curves")
    b.add_argument("--name", required=True, help="the parameter, e.g. pick.STANCE_MIN_D_M")
    b.add_argument("--task", default=None)
    b.add_argument("--curves", default=None, help=f"the curves file (default {CURVES})")
    a = ap.parse_args(argv)
    if a.cmd == "best":
        out = dict(best=best_level(a.name, a.task, path=a.curves),
                   curves=[{k: c.get(k) for k in ("family", "incumbent", "machine", "controller", "tasks", "levels")}
                           for c in curves(name=a.name, factor="param", task=a.task, path=a.curves)])  # fmt: skip
    elif a.cmd == "reach":
        out = reach(a.x, a.y_out, a.z, a.att, a.pelvis, a.hand, path=a.table)
    else:
        out = reach_world(a.point, a.pelvis[:3], a.pelvis[3:], a.hand, a.pinch, path=a.table)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
