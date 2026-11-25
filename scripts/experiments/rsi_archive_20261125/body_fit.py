"""TODO 33.1.4 (user 2026-10-23 "先做预测器"): fit and validate the reach predictor on the samples of body_data.py --
from what is known before the hand moves (the grasp point or the wrist target in the pelvis heading frame, the hand,
top pinch or side, the pelvis height, where the hand starts) to the delivered error (`jaw_err_cm` the skill measured
at the end of the reach) and the odds that the hand lands 3 cm or more off. The gate of TODO 33.1: AUC >= 0.75 on
held-out rounds before any skill uses it at run time; the IK residual (AUC 0.56 on warehouse) is the baseline to beat.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body_fit.py [--samples FILE] [--out DIR] [--k 25]

Two feature sets: `grasp` -- the grasp point the skill chose (what a stance / grasp chooser knows, TODO 33.12) -- and
`ref` -- the wrist reference the motor layer was given at the end of the reach (what a pre-compensation would adjust).
Models: a distance-weighted k-nearest-neighbour table (the one shipped, `reach_knn_<set>.npz`, read by
locomani.tasks.scene.body), an L2 logistic regression with a few quadratic terms, and a small MLP (torch).

Splits, after dropping exact duplicates (the A/B replays copied the same rounds into many sandboxes: 4,665 first
reaches were 1,737 distinct ones): held-out starts (five folds over start labels -- the diagnostic pool's 30 starts
recur in every sandbox, so this is the split that says whether the model knows the body rather than the starts),
held-out skill versions (five folds over skill hashes) and leave-one-sandbox-group-out. Out-of-fold AUC throughout,
calibration bins for the k-NN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_samples.npz"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "body"
SETS = {
    "grasp": ("gx", "gy", "gz", "gdist", "pinch", "pelvis_z0", "hand"),
    "ref": ("tx", "ty", "tz", "dist", "pinch", "pelvis_z0", "wx0", "wy0", "wz0", "hand"),
}
MISS_CM = 3.0
ATT_DEG = 10.0


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan when one class is missing."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_of(sandbox: str) -> str:
    for g in ("ga-cont-c", "ga-cont-b", "grasp-atomic-a", "ga-confirm"):
        if sandbox.startswith(g):
            return g
    return "ab-replays" if sandbox.startswith("ab-") else "other"


def matrix(s: np.ndarray, features) -> np.ndarray:
    cols = []
    for f in features:
        if f == "gdist":
            cols.append(np.sqrt(s["gx"].astype(float) ** 2 + s["gy"].astype(float) ** 2 + s["gz"].astype(float) ** 2))
        else:
            cols.append(s[f].astype(float))
    return np.stack(cols, axis=1)


def dedupe(s: np.ndarray) -> np.ndarray:
    """One row per (start, skills, step, t0): the replays copied rounds between sandboxes."""
    seen, keep = set(), []
    for i, r in enumerate(s):
        k = (str(r["label"]), str(r["skills"]), str(r["step"]), round(float(r["t0"]), 2))
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return s[np.array(keep, dtype=int)]


class KNN:
    """Distance-weighted k nearest neighbours in standardised feature space; the table is the model."""

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, X, y_miss, y_err):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.X = (X - self.mu) / self.sd
        self.y_miss, self.y_err = y_miss.astype(float), y_err.astype(float)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(P(miss), predicted error cm (weighted median), mean distance to the k neighbours)."""
        Z = (X - self.mu) / self.sd
        p, e, far = np.zeros(len(Z)), np.zeros(len(Z)), np.zeros(len(Z))
        k = min(self.k, len(self.X))
        for i, z in enumerate(Z):
            d = np.sqrt(((self.X - z) ** 2).sum(1))
            idx = np.argpartition(d, k - 1)[:k]
            w = 1.0 / (d[idx] + 0.05)
            p[i] = float((w * self.y_miss[idx]).sum() / w.sum())
            order = np.argsort(self.y_err[idx])
            cw = np.cumsum(w[order]) / w.sum()
            e[i] = float(self.y_err[idx][order][min(np.searchsorted(cw, 0.5), k - 1)])
            far[i] = float(d[idx].mean())
        return p, e, far


class Logit:
    def __init__(self, l2: float = 1e-2, steps: int = 3000, lr: float = 0.1):
        self.l2, self.steps, self.lr = l2, steps, lr

    def expand(self, X):
        Z = (X - self.mu) / self.sd
        quad = np.stack([Z[:, 0] ** 2, Z[:, 1] ** 2, Z[:, 2] ** 2, Z[:, 0] * Z[:, 2], Z[:, 3] ** 2], axis=1)
        return np.concatenate([np.ones((len(Z), 1)), Z, quad], axis=1)

    def fit(self, X, y, *_):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        A = self.expand(X)
        w = np.zeros(A.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) / len(y) + self.l2 * np.r_[0, w[1:]]
            w -= self.lr * g
        self.w = w
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-self.expand(X) @ self.w))


class MLP:
    def __init__(self, hidden: int = 64, epochs: int = 300, seed: int = 0):
        self.hidden, self.epochs, self.seed = hidden, epochs, seed

    def fit(self, X, y, *_):
        import torch

        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32)
        self.net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], self.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden, 1))  # fmt: skip
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        loss = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = loss(self.net(Z).squeeze(1), t)
            out.backward()
            opt.step()
        return self

    def predict(self, X):
        import torch

        with torch.no_grad():
            Z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return torch.sigmoid(self.net(Z).squeeze(1)).numpy()


def out_of_fold(make, X, y, y_err, folds) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(te) == 0 or len(tr) == 0:
            continue
        m = make().fit(X[tr], y[tr], y_err[tr])
        out = m.predict(X[te])
        pred[te] = out[0] if isinstance(out, tuple) else out
    return pred


def folds_by(keys: np.ndarray, n: int | None = None, seed: int = 0) -> list[np.ndarray]:
    """One fold per distinct key, or `n` folds over shuffled distinct keys."""
    uniq = np.array(sorted(set(keys.tolist())))
    if n is None:
        return [np.flatnonzero(keys == u) for u in uniq]
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [np.flatnonzero(np.isin(keys, p)) for p in np.array_split(uniq, n)]


def calibration(p: np.ndarray, y: np.ndarray, edges=(0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append(dict(p_from=lo, p_to=round(hi, 2), n=int(m.sum()), predicted=round(float(p[m].mean()), 3),
                             actual=round(float(y[m].mean()), 3)))  # fmt: skip
    return rows


def evaluate(s: np.ndarray, features, models: dict, k: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    X = matrix(s, features)
    keep = np.isfinite(X).all(1) & np.isfinite(s["jaw_err_cm"])
    s, X = s[keep], X[keep]
    y = (s["jaw_err_cm"] >= MISS_CM).astype(float)
    y_err = np.minimum(s["jaw_err_cm"].astype(float), 40.0)
    y_att = np.where(np.isfinite(s["attitude_err_deg"]), s["attitude_err_deg"] >= ATT_DEG, np.nan)
    groups = np.array([group_of(x) for x in s["sandbox"]])
    rep = dict(n=int(len(s)), starts=len(set(s["label"].tolist())), skills=len(set(s["skills"].tolist())),
               miss_rate=round(float(y.mean()), 3), features=list(features), splits={})  # fmt: skip
    baselines = {"ik_residual": np.nan_to_num(s["ik_err_cm"].astype(float), nan=0.0)}
    if "gdist" in features:
        baselines["grasp_distance"] = matrix(s, ("gdist",))[:, 0]
    if "dist" in features:
        baselines["reach_distance"] = s["dist"].astype(float)
    splits = (("starts_5fold", folds_by(s["label"], 5)), ("skills_5fold", folds_by(s["skills"], 5)),
              ("sandbox_group", folds_by(groups)))  # fmt: skip
    for name, folds in splits:
        res = {"baselines": {b: round(auc(v, y), 3) for b, v in baselines.items()}, "models": {}}
        for mname, make in models.items():
            t0 = time.time()
            p = out_of_fold(make, X, y, y_err, folds)
            ok = np.isfinite(p)
            res["models"][mname] = dict(auc=round(auc(p[ok], y[ok]), 3), seconds=round(time.time() - t0, 1))
            if mname == "knn":
                res["calibration_knn"] = calibration(p[ok], y[ok])
                m = ok & np.isfinite(y_att)
                res["models"][mname]["auc_attitude_ge10"] = round(auc(p[m], y_att[m]), 3)
                res["per_group"] = {
                    g: dict(n=int((groups == g).sum()), miss=round(float(y[groups == g].mean()), 3),
                            auc_knn=round(auc(p[(groups == g) & ok], y[(groups == g) & ok]), 3))
                    for g in sorted(set(groups))
                }  # fmt: skip
        rep["splits"][name] = res
        print(f"  {'/'.join(features[:3])} {name}: " + ", ".join(f"{m} {r['auc']}" for m, r in res["models"].items())
              + f"; baselines {res['baselines']}", flush=True)  # fmt: skip
    return rep, X, y, y_err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true")
    a = ap.parse_args(argv)
    s = np.load(a.samples, allow_pickle=True)["samples"]
    s = s[s["first"]]
    n_all = len(s)
    if not a.keep_duplicates:
        s = dedupe(s)
    print(f"{n_all} first reaches, {len(s)} distinct (start, skills, step, t0); {len(set(s['label'].tolist()))} starts, "
          f"{len(set(s['skills'].tolist()))} skill versions; miss rate {float(np.mean(s['jaw_err_cm'] >= MISS_CM)):.3f}",
          flush=True)  # fmt: skip
    models = {"knn": lambda: KNN(a.k), "logit": lambda: Logit()}
    if not a.no_mlp:
        try:
            import torch  # noqa: F401

            models["mlp"] = lambda: MLP()
        except ImportError:
            pass
    report = dict(rows=n_all, distinct=int(len(s)), k=a.k, miss_cm=MISS_CM, sets={})
    a.out.mkdir(parents=True, exist_ok=True)
    for name, features in SETS.items():
        rep, X, y, y_err = evaluate(s, features, models, a.k)
        report["sets"][name] = rep
        knn = KNN(a.k).fit(X, y, y_err)  # the shipped table: every distinct first reach
        np.savez_compressed(a.out / f"reach_knn_{name}.npz", X=knn.X.astype(np.float32),
                            y_miss=knn.y_miss.astype(np.float32), y_err=knn.y_err.astype(np.float32), mu=knn.mu,
                            sd=knn.sd, features=np.array(features), k=np.array(a.k), miss_cm=np.array(MISS_CM))  # fmt: skip
    (a.out / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
