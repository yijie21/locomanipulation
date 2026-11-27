"""selfmodel_train3b_20261125: revision 0's "chained ratio" -- model error over direct-execution error (ratio of
position medians) -- with paired intervals, one controller, one split. The frozen final table (t3b_eval.py) has each
median's interval but not this ratio's, and M's per-segment errors were not dumped, so M0-M2 (and T3, descriptive)
are predicted again with t3b_eval.py's own predict / kept / model_list, imported from the frozen asrun_mx copy (its
md5 checked against EVAL_MD5 before anything runs; asrun, asrun_mx, asrun_lml and t3b_eval.py are not changed).
Cells: data3 and data3b, all segments of the split, in the groups all / first (chain 0) / chained (chain >= 1).
Per cell: median(direct), median(M_k), M_k / direct for each seed, M / direct = mean over seeds, T3 / direct; 95 %
intervals from 1000 resamples of the cell's resets (this script's own resamples, rng seed 0 per cell).
Check: every cell's segment count and point medians (direct, T3, M0-M2) must equal final_<split>.json's exactly
(bit for bit as floats), else nothing is written and the exit code is 1, with the differing cell printed. The
criterion reads the point value M / direct (mean of the three seeds' ratios); the intervals are supplementary.
Read-only (data and models).

    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 CUDA_VISIBLE_DEVICES=1 \
        $VPY t3b_mdirect.py --backend heft --split test_iid --json <out.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(os.environ.get("T3B_EVAL_DIR", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/asrun_mx/selfmodel_train3b_20261125"))
EVAL_MD5 = "00e95caf38439c8a4fc6e39995121080"  # t3b_eval.py as frozen in asrun_mx (2026-11-25T15:03:10Z)
assert hashlib.md5((EVAL_DIR / "t3b_eval.py").read_bytes()).hexdigest() == EVAL_MD5, "t3b_eval.py is not asrun_mx's"
sys.path.insert(0, str(EVAL_DIR))
import t3b_eval as E  # noqa: E402  (predict, kept, model_list)

DT = E.DT
NB = 1000


def stats(pos: dict, pos_d: np.ndarray, s: np.ndarray) -> dict:
    o = {"direct": float(np.median(pos_d[s]))}
    for k in pos:
        o[k] = float(np.median(pos[k][s]))
        o[f"{k}/direct"] = o[k] / o["direct"]
    ms = [k for k in ("M0", "M1", "M2") if k in pos]
    o["M/direct"] = float(np.mean([o[f"{k}/direct"] for k in ms]))
    return o


def cell(pos, pos_d, rs, sel) -> dict:
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(sel)
    r = rs[idx]
    o_ = np.argsort(r, kind="stable")
    groups = np.split(idx[o_], np.flatnonzero(np.diff(r[o_])) + 1)
    point = stats(pos, pos_d, idx)
    boots = [stats(pos, pos_d, np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(NB)]
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
    return dict(n=int(len(idx)), n_resets=len(groups), point=point, ci=ci)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--split", default="test_iid", choices=["val", "test_iid", "edge"])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--t3", type=Path, default=None)
    ap.add_argument("--data3", type=Path, default=None)
    ap.add_argument("--data3b", type=Path, default=None)
    ap.add_argument("--final", type=Path, default=None, help="final_<split>.json to check the medians against")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    be = a.backend
    assert not DT.X0_REF, "run without REACHWM_X0_REF (M and T3)"
    root = a.root or E.ROOT / "selfmodel_train3b_20261125" / be
    t3 = a.t3 or E.ROOT / "selfmodel_data3_20261125" / be / "models_s0"
    d3 = a.data3 or E.ROOT / "selfmodel_data3_20261125" / be / "data" / "all.npz"
    d3b = a.data3b or E.ROOT / "selfmodel_data3b_20261125" / be / "data" / "all.npz"
    fin = json.loads((a.final or root / f"final_{a.split}.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m for m in E.model_list(root, t3, None) if m[0] in ("T3", "M0", "M1", "M2")]
    assert [m[0] for m in models] == ["T3", "M0", "M1", "M2"], f"models missing: {[m[0] for m in models]}"
    out = dict(controller=be, split=a.split, models=[(n, str(r), f) for n, r, f in models], sections=[])
    ok, worst = True, 0.0
    print(f"{be} {a.split}: model error / direct-execution error (position medians), paired intervals")
    for title, path in ((f"data3 {a.split}", d3), (f"data3b {a.split}", d3b)):
        D = DT.load(str(path))
        idx = E.kept(D, DT.splits(D)[a.split])
        pos, _, pos_d, _ = E.predict(models, D, idx, dev)
        rs = np.asarray(D["reset"])[idx]
        ch = np.asarray(D["chain"])[idx]
        fs = next(s for s in fin["sections"] if s and s["title"] == title)
        sec = dict(title=title, cells=[])
        for g, sel in (("all", np.ones(len(idx), bool)), ("first", ch == 0), ("chained", ch >= 1)):
            c = cell(pos, pos_d, rs, sel)
            fc = next(x for x in fs["cells"] if x["group"] == g and x["dim"] == "all" and x["cell"] == "all")
            diff = max(abs(c["point"][k] - fc["point"][k]) for k in ("direct", "T3", "M0", "M1", "M2"))
            worst = max(worst, diff)
            good = diff == 0.0 and c["n"] == fc["n"] and c["n_resets"] == fc["n_resets"]
            if not good:
                print(f"  MISMATCH {title} | {g}: n {c['n']} vs {fc['n']}, resets {c['n_resets']} vs {fc['n_resets']}, max |diff| {diff:.3e} cm")
            ok &= good
            c.update(group=g, final_table_max_abs_diff_cm=diff)
            sec["cells"].append(c)
            p, ci = c["point"], c["ci"]
            f = lambda k: f"{p[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"  # noqa: E731
            print(f"  {title} | {g}: n {c['n']} ({c['n_resets']} resets); medians direct {p['direct']:.2f} T3 {p['T3']:.2f} "
                  f"M0 {p['M0']:.2f} M1 {p['M1']:.2f} M2 {p['M2']:.2f} (vs final table: max |diff| {diff:.1e} cm, n equal {c['n'] == fc['n']})")  # fmt: skip
            print(f"     M/direct {f('M/direct')} | M0 {f('M0/direct')}, M1 {f('M1/direct')}, M2 {f('M2/direct')} | T3/direct {f('T3/direct')}")
        out["sections"].append(sec)
        del D
    print(f"MDIRECT medians and counts vs the final table: max |diff| {worst:.1e} cm -> {'PASS' if ok else 'FAIL (nothing written)'}")
    if ok and a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
