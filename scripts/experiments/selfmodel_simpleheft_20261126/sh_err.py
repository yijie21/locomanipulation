"""selfmodel_simpleheft_20261126: a self model's prediction error on the real SIMPLE episodes (F1(a)); read only, CPU.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=<the model's> PYTHONPATH=src \
        $PY sh_err.py <arm>=<corr dir> [...] --models <models dir (holds E3_ensemble_f1)> --val <cm> --label <name> [--json F]

selfmodel_simplecmp_20261126's scmp_offline.py with the model's own denominator (--val: its train.json best val median)
in place of SIMPLE's 1.886 cm, the stages apart, and intervals. Per arm (the --correct record of simple_bench, data/*.npz
in data3's layout at the task label, sd_real's): the model's prediction of every recorded reach (models.predict_batched,
the 13 cm point) against the landing (pos_f), kept = not fallen; per task and stage (chain 0 the pregrasp, 1 the entry)
and both: n, median error (cm), median / --val (F1(a): > 1.3 is "untrustworthy"), the direct execution's miss (data.py
err_cm: the command's 13 cm target to the landing) and model / direct (medians), 95 % intervals from 1000 resamples of
the layouts (task, index), seed 0. online = offline: the prediction recorded when the stage was sent (pred_h13, sent
stages whose executed plan is the evaluated one, plan_same) against this one, max |difference| (m; the log rounds 1e-5);
meaningful only for the model that ran online (--correct's).
--rows <dir> (a description beside F1(a), not part of it): the arm's rows (<dir>/<arm>_<task>.jsonl) give each stage's
table contacts (the row's hits: [stage, body, "table", ...]); the same lines again over the reaches without one ("free").
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "reachwm_20261030")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
STAGES = ((0, "pregrasp"), (1, "entry"), (None, "both"))
NB = 1000


def predict(D: dict, models: Path) -> np.ndarray:
    m = TR.load_model("E3_ensemble", 1.0, D, "cpu", root=models)
    GD = M.tensors(D, "cpu")
    return M.predict_batched(m, GD, torch.arange(len(D["n_steps"])))["pos"].numpy().astype(np.float64)


def boot(vals: dict, f, rng) -> tuple[float, float]:
    """vals: layout -> list of indices; f(index array) -> number; 95 % interval of f, layouts resampled."""
    L = list(vals)
    bs = []
    for _ in range(NB):
        j = rng.integers(0, len(L), len(L))
        bs.append(f(np.concatenate([vals[L[i]] for i in j])))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="arm=<corr dir>")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--val", type=float, required=True, help="the model's own val median (cm), train.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--rows", type=Path, default=None, help="the arms' rows: the reaches without a table contact apart")
    g = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    sw = {k: os.environ.get(k, "") for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL")}
    print(f"== {g.label}: model {g.models}/E3_ensemble_f1, val {g.val:.3f} cm; switches {sw}")
    out = dict(label=g.label, models=str(g.models), val=g.val, switches=sw, sets={})
    for item in g.sets:
        arm, c = item.split("=", 1)
        fs = sorted(glob.glob(f"{c}/data/*.npz"))
        if not fs:
            print(f"  {arm}: no records in {c}")
            continue
        D = DT.load(",".join(fs))
        pos = predict(D, g.models)
        rs, ch = np.asarray(D["reset"]).astype(int), np.asarray(D["chain"]).astype(int)
        seg = {(int(r), int(k)): i for i, (r, k) in enumerate(zip(rs, ch))}
        diffs, n_ps, n_none = [], 0, 0
        touch = None
        if g.rows is not None:  # (reset, chain) of the reaches with a table contact
            hits = {}
            for t in TASKS:
                f = g.rows / f"{arm}_{t}.jsonl"
                for x in (open(f) if f.exists() else []):
                    r = json.loads(x)
                    hits[(t, int(r["ep"]))] = {h[0] for h in (r.get("hits") or []) if "table" in str(h[2])}
            touch = set()
        for f in sorted(glob.glob(f"{c}/log/*.jsonl")):
            for line in open(f):
                lg = json.loads(line)
                if touch is not None:
                    hs = hits.get((lg["task"], 10 * int(lg["level"]) + int(lg["index"])))
                    assert hs is not None, (arm, lg["task"], lg["level"], lg["index"])
                    touch |= {(int(lg["reset"]), k) for k, st in ((0, "pregrasp"), (1, "entry")) if st in hs}
                for s in lg["stages_correct"]:
                    i = seg.get((int(lg["reset"]), int(s["chain"])))
                    if s.get("plan_same") is not True or i is None or "pred_h13" not in s:
                        n_none += 1
                        continue
                    n_ps += 1
                    diffs.append(float(np.abs(pos[i] - np.asarray(s["pred_h13"])).max()))
        e = 100 * np.linalg.norm(pos - np.asarray(D["pos_f"]), axis=1)
        dr = np.asarray(D["err_cm"]).astype(np.float64)
        keep = np.asarray(D["fell"]) == 0
        task, index = (rs % 1_000_000) // 1000, rs % 100
        print(f"  {arm}: {c}: {len(rs)} segments ({int(keep.sum())} kept, not fallen); online = offline on {n_ps} sent stages "
              f"(plan_same): max |pred diff| {max(diffs) if diffs else float('nan'):.3e} m; {n_none} stages without a pair")  # fmt: skip
        print("    model error at the task label, median cm [95 %, layouts resampled] / val = F1(a) ratio [95 %]; direct miss median; model / direct [95 %]")
        rng = np.random.default_rng(0)
        rows = {}
        free = np.ones(len(rs), bool)
        if touch is not None:
            free = np.array([(int(r), int(k)) not in touch for r, k in zip(rs, ch)])
            print(f"    reaches with a table contact (rows' hits): {int((~free & keep).sum())} of {int(keep.sum())} kept")
        cells = [(t, nm, k, sn, "") for t, nm in [(t, nm) for t, nm in enumerate(TASKS)] + [(None, "all")] for k, sn in STAGES]
        if touch is not None:
            cells += [(t, nm, k, sn, "free") for t, nm in [(t, nm) for t, nm in enumerate(TASKS)] + [(None, "all")] for k, sn in STAGES]
        for t, nm, k, sn, fr in cells:
            s = keep & ((task == t) if t is not None else True) & ((ch == k) if k is not None else True) & (free if fr else True)
            if not s.any():
                continue
            idx = np.flatnonzero(s)
            lay = defaultdict(list)
            for i in idx:
                lay[(int(task[i]), int(index[i]))].append(i)
            lay = {kk: np.array(v) for kk, v in lay.items()}
            md, mdd = float(np.median(e[idx])), float(np.median(dr[idx]))
            ci = boot(lay, lambda j: np.median(e[j]), rng)
            cr = boot(lay, lambda j: np.median(e[j]) / np.median(dr[j]), rng)
            o = dict(n=int(len(idx)), n_layouts=len(lay), median_cm=md, ci_cm=ci, ratio=md / g.val, ratio_ci=(ci[0] / g.val, ci[1] / g.val),
                     direct_cm=mdd, model_over_direct=md / mdd, mod_ci=cr, untrustworthy=bool(md / g.val > 1.3))  # fmt: skip
            rows[f"{nm}/{sn}" + (f"/{fr}" if fr else "")] = o
            print(f"    {nm + (' ' + fr if fr else ''):20s} {sn:8s} n {o['n']:3d}: {md:6.3f} cm [{ci[0]:6.3f}, {ci[1]:6.3f}] = {o['ratio']:.3f}x [{o['ratio_ci'][0]:.3f}, "
                  f"{o['ratio_ci'][1]:.3f}]{'  > 1.3' if o['untrustworthy'] else ''}; direct {mdd:6.3f} cm; model / direct {o['model_over_direct']:.3f} "
                  f"[{cr[0]:.3f}, {cr[1]:.3f}]")  # fmt: skip
        out["sets"][arm] = dict(dir=c, n=int(len(rs)), kept=int(keep.sum()), online_offline_max_m=max(diffs) if diffs else None,
                                n_plan_same=n_ps, n_unpaired=n_none, rows=rows)  # fmt: skip
    if g.json:
        g.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
