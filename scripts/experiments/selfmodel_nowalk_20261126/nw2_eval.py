"""selfmodel_nowalk_20261126 phase 2 (LocoManiLab, one controller): train3b's D (seeds 0-2, the reference) and this
experiment's N and C (seeds 0-2) scored on the same segments; read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY nw2_eval.py --backend <be> --root <phase-2 out>/<be> --splits <root>/splits/splits.npz --out <dir> [--no-check]

Models: E3_ensemble frac 1.0; D<k> = selfmodel_train3b_20261125/<be>/models_D_s<k>, N<k> / C<k> / C2<k> =
<root>/models_<arm>_s<k> (those present). Rebuilt on the scored data and loaded from their state (normalisation
included); arms whose splits.npz entry says so (N, C2) get x0's walk-flag column 204 zeroed, as in their training.
Error: the predicted final hand point to the real one (cm, heading frame); attitude (deg) beside it (t3b_lmleval).
  primary    lmlreal's records of the controller, every segment not fallen / lost and settled at the end label
             (train3b's lmlreal metric). Cells: all (via points included), 第一下 (reach_no 1), 第二下 (reach_no 2),
             walked segments, in-place segments. Per model the median [95 %: 4000 resamples of the episodes (reset
             ids), every statistic on the same resamples]; the seed-paired ratios N<k>/C<k>, C2<k>/C<k>, C<k>/D<k> and
             their mean over the seeds present (the registered statistic: mean_k N<k>/C<k>, "all").
  secondary  the controller's data3 held-out splits (data.py's buckets; val + test_iid, kept settled): first reach in
             place / walk first, chained in place / after a walk; 2000 resamples of the resets.
Reproduction check: D0-D2's medians on the primary "all" cell equal train3b's lml_real.json (computed there on a GPU),
|d| <= 2e-3 cm, else exit 1. Writes <out>/eval.txt (this print), eval.json, per_segment.npz.
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
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3b D's switch"
assert not DT.FDVEL and not DT.X0_REF, "train3b's D: FDVEL / X0_REF off"
M = TR.M
L = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
WALKED_COL = 204
SEEDS = (0, 1, 2)
NB_REAL, NB_HELD = 4000, 2000
PAIRS = (("N", "C"), ("C2", "C"), ("N", "C2"), ("C", "D"), ("N", "D"))


def kept(D: dict) -> np.ndarray:
    return np.flatnonzero((np.asarray(D["fell"]) == 0) & (np.asarray(D["settled_end"]) > 0))


def predict(models: dict, D: dict, idx: np.ndarray) -> tuple[dict, dict]:
    GD = M.tensors(D, "cpu")
    ti = torch.tensor(idx)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos, att = {}, {}
    for name, (root, zero) in models.items():
        ck = torch.load(Path(root) / "E3_ensemble_f1" / "model.pt", map_location="cpu")
        m = M.build("E3_ensemble", D, np.arange(len(D["n_steps"])), "cpu", n_train=ck["n_train"])
        m.load_state_dict(ck["state"])
        m.eval()
        x = GD["x0"].clone()
        if zero:
            for mem in m.members:
                assert float(mem.nx.mu[WALKED_COL]) == 0.0 and abs(float(mem.nx.sd[WALKED_COL]) - 1e-3) < 1e-9, name
            x[:, WALKED_COL] = 0.0
        p = M.predict_batched(m, {**GD, "x0": x}, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).numpy().astype(np.float64)
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).numpy().astype(np.float64)
        del m
    return pos, att


def groups(keys: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(keys, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(keys[o])) + 1)


def table(title, E, A, keys, sel, nb, rng, out, unit) -> None:
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        print(f"  {title}: no segments")
        return
    G = [idx[g] for g in groups(keys[idx])]
    names = list(E)
    pairs = []
    for a, b in PAIRS:
        ks = [k for k in SEEDS if f"{a}{k}" in E and f"{b}{k}" in E]
        if ks:
            pairs.append((a, b, ks))

    def st(s):
        o = {n: np.median(E[n][s]) for n in names}
        o.update({f"p90_{n}": np.percentile(E[n][s], 90) for n in names})
        o.update({f"att_{n}": np.median(A[n][s]) for n in names})
        for a, b, ks in pairs:
            r = [o[f"{a}{k}"] / o[f"{b}{k}"] for k in ks]
            o.update({f"{a}{k}/{b}{k}": v for k, v in zip(ks, r)})
            o[f"{a}/{b} mean"] = float(np.mean(r))
        return o

    pt = st(idx)
    bs = [st(np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])) for _ in range(nb)]
    ci = {k: (float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
    out[title] = dict(n=int(len(idx)), clusters=len(G), unit=unit, point={k: float(v) for k, v in pt.items()}, ci=ci)
    f = lambda k, d=3: f"{pt[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    print(f"  {title}: n {len(idx)} ({len(G)} {unit})")
    print("    median cm: " + " | ".join(f"{n} {f(n)}" for n in names))
    print("    p90 cm:    " + " | ".join(f"{n} {pt['p90_' + n]:.2f}" for n in names))
    print("    attitude deg median: " + " | ".join(f"{n} {pt['att_' + n]:.2f}" for n in names))
    for a, b, ks in pairs:
        print(f"    {a}/{b}: mean over seeds {f(f'{a}/{b} mean')}; per seed " + ", ".join(f"{a}{k}/{b}{k} {f(f'{a}{k}/{b}{k}')}" for k in ks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "sonic", "heft"])
    ap.add_argument("--root", type=Path, required=True, help="this experiment's phase-2 output of the controller")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    be = a.backend
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    Z = np.load(a.splits)
    T = L / "selfmodel_train3b_20261125" / be
    models = {}
    for k in SEEDS:
        if (T / f"models_D_s{k}/E3_ensemble_f1/model.pt").exists():
            models[f"D{k}"] = (str(T / f"models_D_s{k}"), False)
    for arm in ("N", "C", "C2"):
        for k in SEEDS:
            r = a.root / f"models_{arm}_s{k}"
            if (r / "E3_ensemble_f1/model.pt").exists():
                models[f"{arm}{k}"] = (str(r), bool(Z[f"{arm}_zero"]))
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{be}: models {models}; splits {a.splits}; device cpu")
    rng = np.random.default_rng(0)
    res, seg, fails = {}, {}, []
    real = L / f"selfmodel_lmlreal_20261125/{be}/data/all.npz"
    D = DT.load(str(real))
    idx = kept(D)
    E, A = predict(models, D, idx)
    rs = np.asarray(D["reset"]).astype(np.int64)[idx]
    rn, wk = np.asarray(D["reach_no"])[idx], np.asarray(D["walked"])[idx] > 0
    print(f"\n== lmlreal {be} (PRIMARY): {len(idx)} kept segments (not fallen, settled end) of {len(D['n_steps'])}, "
          f"{len(np.unique(rs))} episodes; reach_no 1 {int((rn == 1).sum())}, 2 {int((rn == 2).sum())}, 0 {int((rn == 0).sum())}; walked {int(wk.sum())}")  # fmt: skip
    out = res.setdefault("lmlreal", {})
    table("all (via points included)", E, A, rs, np.ones(len(idx), bool), NB_REAL, rng, out, "episodes")
    table("第一下 (reach_no 1)", E, A, rs, rn == 1, NB_REAL, rng, out, "episodes")
    table("第二下 (reach_no 2)", E, A, rs, rn == 2, NB_REAL, rng, out, "episodes")
    table("walked", E, A, rs, wk, NB_REAL, rng, out, "episodes")
    table("in place", E, A, rs, ~wk, NB_REAL, rng, out, "episodes")
    seg["real"] = dict(idx=idx, reset=rs, reach_no=rn, walked=wk, **{f"err_{n}": E[n] for n in E})
    ref = json.loads((T / "lml_real.json").read_text())
    c0 = next(c for c in ref["cells"] if c["group"] == "all" and c["dim"] == "all")
    for k in SEEDS:
        n = f"D{k}"
        if n in E and n in c0["point"]:
            d = float(np.median(E[n])) - c0["point"][n]
            print(f"  reproduction: {n} median {np.median(E[n]):.5f} vs train3b lml_real.json {c0['point'][n]:.5f} ({d:+.5f}); n {len(idx)} vs {ref['n']}")
            if abs(d) > 2e-3 or len(idx) != ref["n"]:
                fails.append(f"{n} off by {d:+.5f}")
    del D
    D = DT.load(str(L / f"selfmodel_data3_20261125/{be}/data/all.npz"))
    sp = DT.splits(D)
    ok = np.zeros(len(D["n_steps"]), bool)
    ok[kept(D)] = True
    hv = np.zeros(len(ok), bool)
    hv[sp["val"]] = hv[sp["test_iid"]] = True
    idx = np.flatnonzero(ok & hv)
    E, A = predict(models, D, idx)
    rs = np.asarray(D["reset"]).astype(np.int64)[idx]
    ch, wk = np.asarray(D["chain"])[idx], np.asarray(D["walked"])[idx] > 0
    print(f"\n== data3 {be} held out (val + test_iid, kept settled): {len(idx)} segments")
    out = res.setdefault("data3 held out", {})
    for lab, sel in (("first, in place", (ch == 0) & ~wk), ("first, walk first", (ch == 0) & wk),
                     ("chained, in place", (ch >= 1) & ~wk), ("chained, after a walk", (ch >= 1) & wk), ("all", np.ones(len(idx), bool))):
        table(lab, E, A, rs, sel, NB_HELD, rng, out, "resets")
    seg["data3"] = dict(idx=idx, reset=rs, chain=ch, walked=wk, **{f"err_{n}": E[n] for n in E})
    np.savez(a.out / "per_segment.npz", **{f"{k}__{f}": v for k, d in seg.items() for f, v in d.items()})
    (a.out / "eval.json").write_text(json.dumps(dict(backend=be, models={k: list(v) for k, v in models.items()}, results=res, fails=fails), indent=1))
    if a.no_check:
        print("\nreproduction check skipped (--no-check)")
        return
    print(f"\n{'REPRODUCTION PASS' if not fails else 'REPRODUCTION FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
