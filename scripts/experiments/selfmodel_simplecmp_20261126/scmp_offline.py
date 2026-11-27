"""selfmodel_simplecmp_20261126: the --correct record against itself and against selfmodel_simpledata's; read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 PYTHONPATH=src \
        $PY scmp_offline.py <corr dir> [<corr dir> ...] --models <models_D_s0> [--ref <simpledata real/task/data/all.npz>]

For every --correct record (<corr>/data/*.npz, data3's layout with the task label; <corr>/log/*.jsonl):
  online = offline  D's prediction recorded when the stage was sent (pred_h13, the 13 cm point, the command row's heading
                    frame) against D on the recorded segment of that reach (data.load, as sx_eval / sd_eval read records),
                    for the stages whose executed plan is the evaluated one (plan_same); max |difference| (m).
  --ref             the same episode's segment in selfmodel_simpledata's real record of A (reset 91 000 000 + 1000 task +
                    100 level + index): max |x0 difference|, |u difference| and |prediction difference| (an arm that sent A's
                    commands -- dryR -- must give 0 up to float32).
  per task          D's error at the task label (13 cm point, the final hand point to the real one; kept: not fallen),
                    median, and the ratio to the two registered denominators (SIMPLE val 1.886 cm; data3b test_iid
                    pregrasp + entry, task-style walk 0.993 cm).
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "reachwm_20261030")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
VAL, TIID = 1.886, 0.993
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")


def predict(D: dict, models: Path) -> np.ndarray:
    m = TR.load_model("E3_ensemble", 1.0, D, "cpu", root=models)
    GD = M.tensors(D, "cpu")
    return M.predict_batched(m, GD, torch.arange(len(D["n_steps"])))["pos"].numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corr", nargs="+")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--ref", default="")
    g = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ref = None
    if g.ref:
        R = DT.load(g.ref)
        ref = dict(D=R, pos=predict(R, g.models), at={int(r): i for i, r in enumerate(R["reset"]) if R["chain"][i] == 0},
                   at1={int(r): i for i, r in enumerate(R["reset"]) if R["chain"][i] == 1})  # fmt: skip
    for c in g.corr:
        fs = sorted(glob.glob(f"{c}/data/*.npz"))
        if not fs:
            print(f"{c}: no records")
            continue
        D = DT.load(",".join(fs))
        pos = predict(D, g.models)
        rs, ch = np.asarray(D["reset"]).astype(int), np.asarray(D["chain"]).astype(int)
        seg = {(int(r), int(k)): i for i, (r, k) in enumerate(zip(rs, ch))}
        diffs, n_ps, n_none = [], 0, 0
        xd, ud, pd = [], [], []
        for f in sorted(glob.glob(f"{c}/log/*.jsonl")):
            for line in open(f):
                lg = json.loads(line)
                for s in lg["stages_correct"]:
                    i = seg.get((int(lg["reset"]), int(s["chain"])))
                    if s.get("plan_same") is not True or i is None or "pred_h13" not in s:
                        n_none += 1
                        continue
                    n_ps += 1
                    diffs.append(float(np.abs(pos[i] - np.asarray(s["pred_h13"])).max()))
                    if ref is not None:
                        rid = 91_000_000 + 1000 * TASKS.index(lg["task"]) + 100 * int(lg["level"]) + int(lg["index"])
                        j = (ref["at"] if s["chain"] == 0 else ref["at1"]).get(rid)
                        if j is not None:
                            xd.append(float(np.abs(np.asarray(D["x0"])[i] - np.asarray(ref["D"]["x0"])[j]).max()))
                            ud.append(float(np.abs(np.asarray(D["u"])[i] - np.asarray(ref["D"]["u"])[j]).max()))
                            pd.append(float(np.abs(pos[i] - ref["pos"][j]).max()))
        print(f"{c}: {len(rs)} segments; online = offline on {n_ps} sent stages (plan_same): max |pred diff| "
              f"{max(diffs) if diffs else float('nan'):.3e} m (rounded to 1e-5 in the log); {n_none} stages without a pair")  # fmt: skip
        if ref is not None:
            print(f"  against simpledata's A record ({len(xd)} reaches): max |x0 diff| {max(xd) if xd else float('nan'):.3e}, "
                  f"|u diff| {max(ud) if ud else float('nan'):.3e}, |pred diff| {max(pd) if pd else float('nan'):.3e} m")  # fmt: skip
        e = 100 * np.linalg.norm(pos - np.asarray(D["pos_f"]), axis=1)
        keep = np.asarray(D["fell"]) == 0
        task = (rs % 1_000_000) // 1000
        for t, nm in enumerate(TASKS):
            s = keep & (task == t)
            if s.any():
                md = float(np.median(e[s]))
                print(f"  {nm:20s} n {int(s.sum()):3d}: D error at the task label median {md:.3f} cm = {md / VAL:.3f}x SIMPLE val, "
                      f"{md / TIID:.3f}x test_iid task-walk")  # fmt: skip
        md = float(np.median(e[keep]))
        print(f"  all                  n {int(keep.sum()):3d}: {md:.3f} cm = {md / VAL:.3f}x / {md / TIID:.3f}x")


if __name__ == "__main__":
    main()
