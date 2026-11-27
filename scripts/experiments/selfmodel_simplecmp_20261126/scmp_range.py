"""selfmodel_simplecmp_20261126: selfmodel_simpledata's evaluation 4 (sd_range.py) per reach, on this entry's records;
read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=<g> \
        $PY scmp_range.py --train <D's data spec> --models <models_D_s0> --out <json> <arm>=<corr dir> ...

The same checks as sd_range.py (imported: quantities, nn_dist; the same training set, limits, nearest-neighbour threshold
and table) at the start row of every recorded reach: range (39), neighbour, table, and "in range" = all three. Written per
reach (reset, chain, task, level, index, the four checks, the nearest-neighbour distance, D's error at the task label in cm,
the 13 cm point, kept = not fallen), for scmp_eval.py to read per task and arm.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", Path(__file__).resolve().parents[3]))
os.environ["LOCO_REPO"] = str(REPO)
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "selfmodel_simpledata_20261126")]
import sd_range as SR  # noqa: E402  (data, train, d3_collect on its path)

DT, TR, M = SR.DT, SR.TR, SR.M
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("sets", nargs="+", help="arm=<corr dir>")
    a = ap.parse_args()
    import d3_collect as D3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)  # sd_range's draw of the internal nearest-neighbour sample
    tpath = a.table or D3.TABLE
    table = json.loads(Path(tpath).read_text())
    DTr = DT.load(a.train)
    tr = DT.splits(DTr)["train"]
    qt = SR.quantities(DTr, tr)
    lim = {g: (np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)) for g, v in qt.items()}
    X = np.asarray(DTr["x0"])[tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep_c = sd > 1e-6
    T = torch.tensor(((X - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
    trr = torch.tensor(np.asarray(DTr["reset"])[tr].astype(np.int64), device=dev)
    pick = np.sort(rng.choice(len(tr), size=min(SR.N_INTERNAL, len(tr)), replace=False))
    thr = float(np.percentile(SR.nn_dist(T[pick], trr[pick], T, trr), 99))
    print(f"training set {len(tr)} segments; nearest-neighbour p99 {thr:.3f} (sd_range's eval4: 21.152)", flush=True)
    del X, DTr
    out = dict(train=a.train, nn_p99=thr, table=str(tpath), sets={})
    for item in a.sets:
        arm, c = item.split("=", 1)
        fs = sorted(glob.glob(f"{c}/data/*.npz"))
        if not fs:
            print(f"{arm}: no records")
            continue
        D = DT.load(",".join(fs))
        idx = np.arange(len(D["n_steps"]))
        q = SR.quantities(D, idx)
        inr = {g: ((q[g] >= lim[g][0]) & (q[g] <= lim[g][1])).all(1) for g in q}
        x = np.asarray(D["x0"]).astype(np.float64)
        Q = torch.tensor(((x - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
        rq = torch.tensor(np.asarray(D["reset"]).astype(np.int64), device=dev)
        d = SR.nn_dist(Q, rq, T, trr)
        side, pz, ct = np.asarray(D["side"]), np.asarray(D["pelvis_z"]), np.asarray(D["cmd_target"])
        tab = np.array([D3.in_table(table["rows"]["right" if s > 0 else "left"][D3.pz_key(None if p < 0 else float(p))], t)
                        for s, p, t in zip(side, pz, ct)])  # fmt: skip
        rng39 = inr["joints"] & inr["pelvis"] & inr["hands"]
        allin = rng39 & (d <= thr) & tab
        m = TR.load_model("E3_ensemble", 1.0, D, dev, root=a.models)
        GD = M.tensors(D, dev)
        p = M.predict_batched(m, GD, torch.tensor(idx, device=dev))
        e = (100 * (p["pos"] - GD["pos_f"]).norm(dim=-1)).cpu().numpy()
        rs = np.asarray(D["reset"]).astype(int) % 1_000_000
        rows = [dict(reset=int(np.asarray(D["reset"])[i]), chain=int(D["chain"][i]), task=TASKS[rs[i] // 1000], level=int((rs[i] % 1000) // 100),
                     index=int(rs[i] % 100), joints=bool(inr["joints"][i]), pelvis=bool(inr["pelvis"][i]), hands=bool(inr["hands"][i]),
                     range39=bool(rng39[i]), neighbour=bool(d[i] <= thr), table=bool(tab[i]), in_range=bool(allin[i]), nn=float(d[i]),
                     err_cm=float(e[i]), kept=bool(np.asarray(D["fell"])[i] == 0)) for i in idx]  # fmt: skip
        out["sets"][arm] = rows
        k = np.asarray(D["fell"]) == 0
        print(f"{arm}: {len(idx)} reaches, in range {100 * allin[k].mean():.1f} % (kept {int(k.sum())}), D median {np.median(e[k]):.3f} cm")
    a.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
