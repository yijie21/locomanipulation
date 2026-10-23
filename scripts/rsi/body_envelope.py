"""TODO 33.1.4 (user 2026-10-23 "先做预测器"): the reach predictor from the dense envelope probe -- 15,120 controlled
reaches on a grid (hand x attitude level/down/45 deg x posture stand/0.68/0.62/0.56/0.52 m x lateral 5 steps x height
0.05-1.45 m x forward 0.2-0.9 m; capability_probe.py envelope, run on the A800 2026-10-21, cells in
outputs/scene_rsi_20261016/capability/envelope_remote/envelope/). The task records alone could not give a predictor:
their 1,737 distinct first reaches sit on only 41 starts, so a model fit on them knows the starts, not the body
(held-out starts AUC 0.49-0.61, body_fit.py). The grid is where the body's shape is sampled evenly.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body_envelope.py [--cells DIR] [--samples FILE] [--out DIR]

Two checks: (1) held-out cells (five folds over cells, and whole forward slices held out) -- does the table
interpolate the grid; (2) the transfer that matters: the table, fit on the grid only, scores the task records' first
reaches (grasp point in the pelvis heading frame -> the grid's coordinates: forward, lateral outward of the hand's own
side, absolute height; top pinch -> "down", side -> "level"; the pelvis height at the reach) against their measured
`jaw_err_cm` -- an AUC on data the table never saw. Ships `reach_knn_envelope.npz` for locomani.tasks.scene.body."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import body_fit as BF  # noqa: E402

REPO = HERE.parents[1]
CELLS = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "envelope_remote" / "envelope"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "body"
FEATURES = ("x", "y_out", "z", "att", "pelvis", "hand")  # att 0 level, 1 down, 2 tilt45; pelvis m (stand = STAND_Z)
STAND_Z = 0.78  # the standing pelvis height the probe starts from (its records: 0.77-0.79 before a squat)
ATT = {"level": 0.0, "down": 1.0, "tilt45": 2.0}


def load_cells(d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Feature matrix, miss label (reach error >= 3 cm, or fell), error cm, and the raw cells."""
    cells = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    X, y, e = [], [], []
    for c in cells:
        k = c["cell"]
        out = abs(k["y"]) if (k["y"] < 0) == (k["side"] == "right") else -abs(k["y"])  # + outward of the hand's side
        X.append([k["x"], out, k["z"], ATT[k["att"]], STAND_Z if k["pelvis"] is None else k["pelvis"],
                  1.0 if k["side"] == "right" else 0.0])  # fmt: skip
        err = c.get("jaw_err_cm")
        err = 40.0 if err is None or c.get("fell") else min(float(err), 40.0)
        e.append(err)
        y.append(1.0 if err >= BF.MISS_CM else 0.0)
    return np.array(X), np.array(y), np.array(e), cells


def task_reaches(samples: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The task records' distinct first reaches in the grid's coordinates, with their measured miss and error."""
    s = np.load(samples, allow_pickle=True)["samples"]
    s = BF.dedupe(s[s["first"]])
    keep = np.isfinite(s["gx"]) & np.isfinite(s["gy"]) & np.isfinite(s["gz"]) & np.isfinite(s["jaw_err_cm"])
    s = s[keep]
    X = np.stack([s["gx"].astype(float), -s["gy"].astype(float), s["pelvis_z0"].astype(float) + s["gz"].astype(float),
                  np.where(s["pinch"], 1.0, 0.0), s["pelvis_z0"].astype(float), s["hand"].astype(float)], axis=1)  # fmt: skip
    y = (s["jaw_err_cm"] >= BF.MISS_CM).astype(float)
    return X, y, np.minimum(s["jaw_err_cm"].astype(float), 40.0), s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, default=CELLS)
    ap.add_argument("--samples", type=Path, default=BF.SAMPLES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args(argv)
    X, y, e, cells = load_cells(a.cells)
    rep = dict(cells=int(len(X)), miss_rate=round(float(y.mean()), 3), k=a.k, features=list(FEATURES), checks={})
    print(f"{len(X)} cells, miss (>= {BF.MISS_CM} cm or fell) {y.mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    folds = np.array_split(perm, 5)
    slices = [np.flatnonzero(X[:, 0] == v) for v in sorted(set(X[:, 0].tolist()))]
    for name, fs in (("cells_5fold", folds), ("forward_slices", slices)):
        p = BF.out_of_fold(lambda: BF.KNN(a.k), X, y, e, fs)
        ok = np.isfinite(p)
        rep["checks"][name] = dict(auc=round(BF.auc(p[ok], y[ok]), 3), calibration=BF.calibration(p[ok], y[ok]))
        print(f"  {name}: AUC {rep['checks'][name]['auc']}", flush=True)
    # the structure the grid shows, for the record
    by_x = {f"{v:.1f}": round(float(y[X[:, 0] == v].mean()), 3) for v in sorted(set(X[:, 0].tolist()))}
    by_att = {k: round(float(y[X[:, 3] == v].mean()), 3) for k, v in ATT.items()}
    rep["miss_by_forward"] = by_x
    rep["miss_by_attitude"] = by_att
    print(f"  miss by forward {by_x}; by attitude {by_att}", flush=True)
    # transfer: the grid table scores the task records' first reaches
    knn = BF.KNN(a.k).fit(X, y, e)
    Xt, yt, et, st = task_reaches(a.samples)
    p, e_pred, far = knn.predict(Xt)
    inside = far < 1.0  # task reaches that lie within the grid's standardised neighbourhood
    ik = np.nan_to_num(st["ik_err_cm"].astype(float), nan=0.0)
    gdist = np.sqrt((Xt[:, 0] ** 2) + (Xt[:, 1] ** 2) + ((Xt[:, 2] - Xt[:, 4]) ** 2))
    rep["transfer"] = dict(
        n=int(len(yt)), miss_rate=round(float(yt.mean()), 3), auc_envelope=round(BF.auc(p, yt), 3),
        auc_envelope_err=round(BF.auc(e_pred, yt), 3), auc_ik_residual=round(BF.auc(ik, yt), 3),
        auc_grasp_distance=round(BF.auc(gdist, yt), 3),
        inside_grid=int(inside.sum()), auc_inside=round(BF.auc(p[inside], yt[inside]), 3) if inside.any() else None,
        calibration=BF.calibration(p, yt),
        by_group={g: dict(n=int(m.sum()), auc=round(BF.auc(p[m], yt[m]), 3))
                  for g in sorted({BF.group_of(x) for x in st["sandbox"]})
                  for m in [np.array([BF.group_of(x) == g for x in st["sandbox"]])]},
    )  # fmt: skip
    print(f"  transfer to {len(yt)} task reaches: envelope AUC {rep['transfer']['auc_envelope']} "
          f"(error-median score {rep['transfer']['auc_envelope_err']}), IK residual {rep['transfer']['auc_ik_residual']}, "
          f"grasp distance {rep['transfer']['auc_grasp_distance']}; inside the grid {int(inside.sum())} "
          f"(AUC {rep['transfer']['auc_inside']})", flush=True)  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "reach_knn_envelope.npz", X=knn.X.astype(np.float32), y_miss=knn.y_miss.astype(np.float32),
                        y_err=knn.y_err.astype(np.float32), mu=knn.mu, sd=knn.sd, features=np.array(FEATURES),
                        k=np.array(a.k), miss_cm=np.array(BF.MISS_CM), stand_z=np.array(STAND_Z))  # fmt: skip
    (a.out / "envelope_fit.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
