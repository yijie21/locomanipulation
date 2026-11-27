"""selfmodel_relabel_20261125 evaluation, read-only: models A, B, C (E3_ensemble, f1) on the original data's val and
test_iid segments (data.splits of the two all.npz).

Primary: the real slow entries (each reset's third reach) that start still: the acting jaw's speed below 2 cm/s over the
0.5 s before the command, i.e. the 3 rows of 0.1 s the entry records before k0 plus the last 2 intervals of the pre-grasp
that ends where the entry begins (the pre-grasp's last row is the entry's first, same instant). Fallen segments are out (as
train.py's val). Per model the median predicted-landing error (cm, |pred - final jaw|); the direct-execution error
|command - final jaw| on the same segments; ratios A/B, C/B, A/C with 95 % intervals from resampling resets (2000, seed 0).
Reported for val + test_iid pooled (the registered set), and for each split alone.
Secondary: the waypoint and pre-grasp segments of val (and of test_iid), same statistics, A/C.

    cd <repo>; REACHWM_GOAL_REF=1 $PY eval_entry.py --data <B spec> --A <root> --B <root> --C <root> [--Am <root>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as T  # noqa: E402

NB, SEED = 2000, 0
PAIRS = (("A", "B"), ("C", "B"), ("A", "C"))
XPAIRS = (("Am", "B"), ("Am", "C"), ("Am", "A"))  # exploratory model Am (not registered), when given


def place(reset: np.ndarray) -> np.ndarray:
    new = np.r_[True, reset[1:] != reset[:-1]]
    assert new.sum() == len(np.unique(reset))
    i = np.arange(len(reset))
    return i - np.maximum.accumulate(np.where(new, i, 0))


def still_before(D: dict, pl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per segment: max acting-jaw speed (m/s) over the 0.5 s before the command (needs the previous segment of the same
    reset), whether that window exists, and the jump (m) between the previous segment's last row and this one's first."""
    n, so, k0 = D["n_steps"].astype(int), D["so"], D["k0"].astype(int)
    W = D["W"]
    N = len(n)
    vmax, have, jump = np.full(N, np.inf), np.zeros(N, bool), np.full(N, np.nan)
    for i in range(1, N):
        if pl[i] == 0 or D["reset"][i - 1] != D["reset"][i]:
            continue
        c = slice(4, 7) if D["side"][i] > 0 else slice(7, 10)
        cur = W[so[i] : so[i] + k0[i] + 1, c].astype(np.float64)  # rows 0 .. k0
        pl_last = so[i - 1] + n[i - 1] - 1
        prev = W[pl_last - 2 : pl_last + 1, c].astype(np.float64)  # last 3 rows
        v = np.r_[np.linalg.norm(np.diff(prev, axis=0), axis=1), np.linalg.norm(np.diff(cur, axis=0), axis=1)] / 0.1
        vmax[i], have[i] = v.max(), len(v) == 5
        jump[i] = np.linalg.norm(prev[-1] - cur[0])
    return vmax, have, jump


def boot(err: dict, reset: np.ndarray, rng) -> dict:
    """Median per model and the three ratios on reset resamples: 95 % intervals."""
    ur, inv = np.unique(reset, return_inverse=True)
    med = {m: [] for m in err}
    for _ in range(NB):
        w = np.bincount(rng.integers(len(ur), size=len(ur)), minlength=len(ur))[inv]
        for m, e in err.items():
            med[m].append(np.median(np.repeat(e, w)))
    med = {m: np.array(v) for m, v in med.items()}
    out = {}
    for x, y in PAIRS + (XPAIRS if "Am" in err else ()):
        r = med[x] / med[y]
        out[f"{x}/{y}"] = (float(np.median(err[x]) / np.median(err[y])), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
    for m, v in med.items():
        out[f"ci_{m}"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def report(tag: str, idx: np.ndarray, E: dict, D: dict, seed: int) -> dict:
    err = {m: E[m][idx] for m in E}
    direct = D["err_cm"][idx]
    b = boot(dict(err, direct=direct), D["reset"][idx], np.random.default_rng(seed))
    row = dict(n=int(len(idx)), resets=int(len(np.unique(D["reset"][idx]))),
               median_cm={m: float(np.median(v)) for m, v in err.items()}, direct_cm=float(np.median(direct)),
               p90_cm={m: float(np.percentile(v, 90)) for m, v in err.items()}, **b)  # fmt: skip
    md = row["median_cm"]
    print(f"{tag}: n {row['n']} ({row['resets']} resets)  median cm  A {md['A']:.3f} [{b['ci_A'][0]:.3f}, {b['ci_A'][1]:.3f}]"
          f"  B {md['B']:.3f} [{b['ci_B'][0]:.3f}, {b['ci_B'][1]:.3f}]  C {md['C']:.3f} [{b['ci_C'][0]:.3f}, {b['ci_C'][1]:.3f}]"
          f"  direct {row['direct_cm']:.3f} [{b['ci_direct'][0]:.3f}, {b['ci_direct'][1]:.3f}]")  # fmt: skip
    print("    " + "  ".join(f"{k} {b[k][0]:.3f} [{b[k][1]:.3f}, {b[k][2]:.3f}]" for k in ("A/B", "C/B", "A/C"))
          + "  p90 cm " + " ".join(f"{m} {v:.3f}" for m, v in row["p90_cm"].items() if m != "Am"))  # fmt: skip
    if "Am" in md:
        print(f"    [exploratory Am] median {md['Am']:.3f} [{b['ci_Am'][0]:.3f}, {b['ci_Am'][1]:.3f}]  p90 {row['p90_cm']['Am']:.3f}  "
              + "  ".join(f"{k} {b[k][0]:.3f} [{b[k][1]:.3f}, {b[k][2]:.3f}]" for k in ("Am/B", "Am/C", "Am/A")))  # fmt: skip
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the original data (model B's spec)")
    for m in "ABC":
        ap.add_argument(f"--{m}", required=True, help=f"model {m}'s --out-root")
    ap.add_argument("--Am", default=None, help="exploratory model Am's --out-root (C plus the moving mid starts)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(a.data)
    sp = DT.splits(D)
    pl = place(D["reset"])
    GD = M.tensors(D, dev, step=False)
    ev = np.r_[sp["val"], sp["test_iid"]]
    E = {}
    names = ["A", "B", "C"] + (["Am"] if a.Am else [])
    for m in names:
        model = T.load_model("E3_ensemble", 1.0, D, dev, root=Path(getattr(a, m)))
        e = np.full(len(D["n_steps"]), np.nan, np.float32)
        e[ev] = M.pos_error_cm(model, GD, torch.tensor(ev, device=dev)).cpu().numpy()
        E[m] = e
        info = json.load(open(Path(getattr(a, m)) / "E3_ensemble_f1" / "train.json"))
        print(f"model {m}: {getattr(a, m)}  n_train {info['n_train']}  best val median {info['best_val_median_cm']:.4f} cm "
              f"(train.json; own val median here {np.median(e[sp['val'][D['fell'][sp['val']] == 0]]):.4f})")  # fmt: skip
    vmax, have, jump = still_before(D, pl)
    fell = D["fell"].astype(bool)
    out = dict(models={m: getattr(a, m) for m in names}, data=a.data)
    sets = dict(val=sp["val"], test_iid=sp["test_iid"], pooled=ev)
    for k, s in sets.items():
        ent = s[pl[s] == 2]
        st = ent[~fell[ent] & have[ent] & (vmax[ent] < 0.02)]
        print(f"[entries {k}] {len(ent)} entries, fell {int(fell[ent].sum())}, no 0.5 s window {int((~have[ent]).sum())}, "
              f"moving before the command {int((~fell[ent] & have[ent] & (vmax[ent] >= 0.02)).sum())}, "
              f"kept {len(st)}; pre-grasp/entry first-row jump > 1 mm: {int((jump[ent] > 1e-3).sum())}")  # fmt: skip
        out[f"entry_{k}"] = report(f"ENTRY {k}", st, E, D, SEED)
    print("--- secondary: the long reaches")
    for k in ("val", "test_iid"):
        for p, nm in ((0, "waypoint"), (1, "pregrasp")):
            s = sp[k]
            s = s[(pl[s] == p) & ~fell[s]]
            out[f"{nm}_{k}"] = report(f"{nm.upper()} {k}", s, E, D, SEED)
    # by dataset, the pooled entries (the wrist dataset's reset ids start at 1350000)
    print("--- the pooled entries by dataset")
    ent = ev[pl[ev] == 2]
    st = ent[~fell[ent] & have[ent] & (vmax[ent] < 0.02)]
    for nm, msk in (("graspr_odo", D["reset"][st] < 1350000), ("graspr_wrist_odo", D["reset"][st] >= 1350000)):
        out[f"entry_pooled_{nm}"] = report(f"ENTRY pooled {nm}", st[msk], E, D, SEED)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
