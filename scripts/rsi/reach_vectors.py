"""TODO 33.1.4 (2): the reach's error vector from the 50 Hz traces -- the data for pre-compensation, and the check
whether a model of it generalises beyond the starts it saw.

`extract`: every `reach` segment of every round trace (diag/*/<label>.npz with its <label>.json) under the given
sandboxes: the hand (the wrist whose reference moved; both-hand reaches are left out), the pelvis at the start (height,
heading), the start and the final commanded wrist and the delivered wrist at the segment's end, all in the start's
pelvis heading frame (x forward, y outward of the hand's own side, z up from the pelvis), the error vector delivered -
commanded, and the round's label, skills, motor signature and sandbox. Exact copies (the replays copied rounds between
sandboxes) are dropped: one row per (label, skills, segment start time).

`check`: five folds over start labels (a model cannot win by knowing the start): how much of the error vector a model
of it removes on rounds of starts it never saw -- the mean |e|, the mean |e - e_hat|, and the same over the reaches that
missed by 3 cm or more. Models: none (e_hat = 0), the mean bias, k nearest neighbours and ridge on the target's
position, the pelvis height and the reach length. Pre-compensation (commanding target - e_hat) only helps if the
residual is clearly under |e| out of start.

    PYTHONPATH=src $PY scripts/rsi/reach_vectors.py extract --out FILE.npz [--procs 6] SANDBOX [...]
    PYTHONPATH=src $PY scripts/rsi/reach_vectors.py check FILE.npz [--motor SIG]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
PAD_OFF = np.array([0.1515, 0.0, 0.0])  # runtime.OFF: wrist frame -> the jaw centre (the pads)
MISS_M = 0.03
FIELDS = ("hand", "pelvis_z", "sx", "sy", "sz", "cx", "cy", "cz", "dx", "dy", "dz", "t0", "dur", "att")


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _windows(z, rec: dict) -> list[tuple[float, float, int, np.ndarray, np.ndarray]]:
    """(t0, t1, hand, commanded jaw centre, commanded wrist rotation) of each one-hand reach in the round's primitive
    call log (TODO 33.20 on; older rounds have none and give nothing: their trace holds the servo-shifted wrist
    reference, not what the skill commanded)."""
    out = []
    for c in rec.get("calls") or []:
        args = c.get("args") or []
        if c.get("name") != "reach" or len(args) < 2 or args[0] not in SIDES:
            continue
        rot = args[2] if len(args) > 2 else (c.get("kwargs") or {}).get("rot")
        if rot is None:
            continue
        out.append(
            (float(c["t0"]), float(c["t1"]), SIDES.index(args[0]), np.asarray(args[1], float), np.asarray(rot, float))
        )
    return out


def segments(z, rec: dict) -> list[dict]:
    """The one-hand reaches of one round (module docstring): the commanded wrist (the jaw target less the pad offset
    along the commanded rotation, runtime.OFF) against the measured wrist at the call's end."""
    from locomani.tasks.scene.body import ATT, attitude

    t = np.asarray(z["t"], float)
    out = []
    for t0, t1, k, jaw, rot in _windows(z, rec):
        s = max(int(np.searchsorted(t, t0)) - 1, 0)
        e = min(int(np.searchsorted(t, t1, side="right")) - 1, len(t) - 1)
        if e <= s or rot.shape != (3, 3):
            continue
        pel = z["robot"][s]
        yaw = _yaw(pel[3:7])
        c, sn = math.cos(-yaw), math.sin(-yaw)
        flip = -1.0 if SIDES[k] == "right" else 1.0  # y outward of the hand's own side

        def frame(v, pel=pel, c=c, sn=sn, flip=flip):
            d = np.asarray(v, float) - pel[:3]
            return np.array([c * d[0] - sn * d[1], flip * (sn * d[0] + c * d[1]), d[2]])

        start, cmd, got = frame(z["wrists"][s, k]), frame(jaw - rot @ PAD_OFF), frame(z["wrists"][e, k])
        out.append(dict(hand=float(k), pelvis_z=float(pel[2]), sx=start[0], sy=start[1], sz=start[2], cx=cmd[0],
                        cy=cmd[1], cz=cmd[2], dx=got[0] - cmd[0], dy=got[1] - cmd[1], dz=got[2] - cmd[2],
                        t0=float(t0), dur=float(t1 - t0), att=float(ATT[attitude(rot)])))  # fmt: skip
    return out


def _one(path: str) -> list[dict]:
    p = Path(path)
    try:
        rec = json.loads(p.with_suffix(".json").read_text())
        z = np.load(p, allow_pickle=True)
        rows = segments(z, rec)
    except Exception:  # noqa: BLE001 -- a truncated or old-format record is skipped
        return []
    meta = dict(label=str(rec.get("label")), skills=str(rec.get("skills")), motor=str(rec.get("motor")),
                sandbox=str(p.parents[5].name if len(p.parents) > 5 else ""), ok=bool(rec.get("ok")))  # fmt: skip
    return [dict(r, **meta) for r in rows]


def extract(sandboxes: list[Path], out: Path, procs: int = 6) -> dict:
    files = sorted(str(f) for sb in sandboxes for f in sb.glob("outputs/scene_rsi_20261016/*/diag/*/*.npz")
                   if not f.name.startswith("_"))  # fmt: skip
    rows, seen = [], set()
    with ProcessPoolExecutor(procs) as ex:
        for rs in ex.map(_one, files, chunksize=32):
            for r in rs:
                key = (r["label"], r["skills"], round(r["t0"], 2))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    cols = {f: np.array([r[f] for r in rows], float) for f in FIELDS}
    for f in ("label", "skills", "motor", "sandbox"):
        cols[f] = np.array([r[f] for r in rows])
    cols["ok"] = np.array([r["ok"] for r in rows])
    np.savez_compressed(out, **cols)
    return dict(files=len(files), reaches=len(rows), out=str(out))


def check(data: Path, motor: str | None = None, k: int = 15) -> dict:
    d = np.load(data, allow_pickle=True)
    keep = np.ones(len(d["dx"]), bool) if motor is None else d["motor"] == motor
    X = np.stack(
        [d["cx"], d["cy"], d["cz"], d["pelvis_z"], np.hypot(d["cx"] - d["sx"], d["cy"] - d["sy"]), d["hand"]], 1
    )[keep]
    E = np.stack([d["dx"], d["dy"], d["dz"]], 1)[keep]
    labels = d["label"][keep]
    uniq = np.unique(labels)
    rng = np.random.default_rng(0)
    fold_of = {lab: i % 5 for i, lab in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[x] for x in labels])
    preds = {m: np.zeros_like(E) for m in ("bias", "knn", "ridge")}
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    for f in range(5):
        tr, te = fold != f, fold == f
        preds["bias"][te] = E[tr].mean(0)
        for i in np.where(te)[0]:
            dist = np.sqrt(((Z[tr] - Z[i]) ** 2).sum(1))
            idx = np.argpartition(dist, k)[:k]
            w = 1.0 / (dist[idx] + 0.05)
            preds["knn"][i] = (w[:, None] * E[tr][idx]).sum(0) / w.sum()
        A = np.c_[np.ones(tr.sum()), Z[tr], Z[tr] ** 2]
        W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ E[tr])
        preds["ridge"][te] = np.c_[np.ones(te.sum()), Z[te], Z[te] ** 2] @ W
    err = np.linalg.norm(E, axis=1)
    miss = err >= MISS_M
    rep = dict(reaches=int(keep.sum()), starts=int(len(uniq)), mean_err_cm=round(100 * float(err.mean()), 2),
               miss_rate=round(float(miss.mean()), 3), mean_err_missed_cm=round(100 * float(err[miss].mean()), 2) if miss.any() else None)  # fmt: skip
    for m, P in preds.items():
        res = np.linalg.norm(E - P, axis=1)
        rep[m] = dict(residual_cm=round(100 * float(res.mean()), 2),
                      residual_missed_cm=round(100 * float(res[miss].mean()), 2) if miss.any() else None,
                      better=round(float((res < err).mean()), 3),
                      still_missed=round(float((res[miss] >= MISS_M).mean()), 3) if miss.any() else None)  # fmt: skip
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("sandboxes", type=Path, nargs="+")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--procs", type=int, default=6)
    c = sub.add_parser("check")
    c.add_argument("data", type=Path)
    c.add_argument("--motor", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        print(json.dumps(extract(a.sandboxes, a.out, a.procs)))
    else:
        print(json.dumps(check(a.data, a.motor), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
