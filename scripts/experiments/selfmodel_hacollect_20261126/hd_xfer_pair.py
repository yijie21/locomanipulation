"""selfmodel_hadata_20261126 revision 2b, evaluation 2's paired number: "the absolute error on xfer's HA record is lower
than F's" = 1 - model / F (ratio of medians), per seed (S_f1_sK, T_f1_sK against fdvel's F, seed 0), the three seeds'
mean, on the record's every reach (chain >= 0), read-only.

    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 SX_REPO=<repo_b> CUDA_VISIBLE_DEVICES= \
        $PY hd_xfer_pair.py --sx <frozen sx_eval.py> --lml <eval>/lml --xfer <eval>/xfer --root <collection>/train/<ctrl> \
        --F <fdvel>/<ctrl>/models_s0 [--check <eval dir>] [--json F]

Everything is the frozen sx_eval's (81f61381, imported by path, md5 checked): the reader (SX_REPO's data.py), kept
(--keep settled: not fallen / lost and settled at the end label, the primary; ended: not fallen / lost, the supplement),
the model build (train.load_model on --lml's data/all.npz, the HA d3 file: the normalisation comes from the state dict)
and predict. --xfer is the record touch-filtered by hd_touch.py (revision 1's rule), --lml the HA d3 (touch-filtered).
The same kept segments for every model; 95 % intervals from 1000 resamples of the record's resets, every model's median
on the same resamples (paired). --check: every model's median and kept count equal sx_eval's own json
(<dir>/eval2_<model>_<keep>.json; the driver runs sx_eval first). Printed: per keep, each model's median [95 %], per seed
1 - S_f1_sK / F and 1 - T_f1_sK / F, the mean over seeds of each arm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

SX_MD5 = "81f61381"
SEEDS = (0, 1, 2)


def load_by_path(name: str, path: Path, md5: str):
    assert hashlib.md5(path.read_bytes()).hexdigest().startswith(md5), f"{path}: md5 is not {md5}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sx", type=Path, required=True, help="the frozen sx_eval.py")
    ap.add_argument("--lml", type=Path, required=True)
    ap.add_argument("--xfer", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="<collection root>/train/<controller>")
    ap.add_argument("--F", type=Path, required=True, help="fdvel's <controller>/models_s0")
    ap.add_argument("--check", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    assert os.environ.get("REACHWM_FDVEL") == "1", "REACHWM_FDVEL=1 is the HA models' and F's switch"
    SX = load_by_path("sx_eval", a.sx, SX_MD5)
    DT, TR = SX.DT, SX.TR
    assert DT.FDVEL
    dev = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    models = [(f"{arm}_f1_s{s}", a.root / f"models_{arm}_s{s}") for arm in ("S", "T") for s in SEEDS] + [("F", a.F)]
    DL = DT.load(str(a.lml / "data" / "all.npz"))
    DX = DT.load(str(a.xfer / "data" / "all.npz"))
    run = json.loads((a.xfer / "run.json").read_text())
    print(f"xfer {run['controller']} ({run['planner']} planner, {run['stance']} stance): {a.xfer}; models built on {a.lml}/data/all.npz; "
          f"reader {DT.__file__}")  # fmt: skip
    print("models " + ", ".join(f"{n} {TR.model_dir('E3_ensemble', 1.0, r)}" for n, r in models))
    nets = {n: TR.load_model("E3_ensemble", 1.0, DL, dev, root=r) for n, r in models}
    out = dict(lml=str(a.lml), xfer=str(a.xfer), models=[(n, str(r)) for n, r in models], keeps={})
    for keep in ("settled", "ended"):
        xi = SX.kept(DX, ended=keep == "ended")
        err = {n: SX.predict(m, DX, xi, dev)["err"] for n, m in nets.items()}
        rx = np.asarray(DX["reset"])[xi]
        G = SX.groups_of(rx)

        def st(s: np.ndarray) -> dict:
            o = {n: float(np.median(e[s])) for n, e in err.items()}
            for arm in ("S", "T"):
                for j in SEEDS:
                    o[f"1-{arm}_f1_s{j}/F"] = 1 - o[f"{arm}_f1_s{j}"] / o["F"]
                o[f"1-{arm}_f1/F"] = float(np.mean([o[f"1-{arm}_f1_s{j}/F"] for j in SEEDS]))
            return o

        point = st(np.arange(len(xi)))
        boots = [st(np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])) for _ in range(SX.NB)]
        ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
        res = dict(n=int(len(xi)), n_resets=len(G), n_record=int(len(DX["n_steps"])), point=point, ci=ci)
        if a.check is not None:
            same = {}
            for n in err:
                J = json.loads((a.check / f"eval2_{n}_{keep}.json").read_text())
                same[n] = J["n_xfer_kept"] == len(xi) and abs(J["splits"]["all"]["xfer"] - point[n]) < 1e-6
            res["same_as_sx_eval"] = same
            print(f"keep {keep}: medians and kept counts equal sx_eval's json: {all(same.values())} {same}")
        out["keeps"][keep] = res
        f = lambda k, d=3: f"{point[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
        print(f"\n== keep {keep}{' (primary)' if keep == 'settled' else ' (supplement)'}: all reaches, {len(xi)} kept of "
              f"{res['n_record']} record segments, {len(G)} resets")  # fmt: skip
        print("  median cm [95 %]: " + " | ".join(f"{n} " + f(n) for n in err))
        for arm in ("S", "T"):
            print(f"  1 - {arm}/F: mean over seeds {f(f'1-{arm}_f1/F')}; per seed " + ", ".join(f"s{j} " + f(f"1-{arm}_f1_s{j}/F") for j in SEEDS))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
