"""Check the vision grasps in the hand-only model before one is chosen (grasp_cache.HandModel: close, lift, hold, shake).

Per placement: the grasps room_grasp would consider (grasp_options, the first tier that has any), best-scored first, up to
--top of them run in the hand-only model; the ones that hold stay, ranked by the model's score. Nothing holds -> the unchecked
set is kept (flag `checked` False).

--shape true   the hand-only model holds the object's own mesh and mass: privileged, the upper bound of any such check.

-> <set>/vision/grasp_<out>_L<k>_s<j>.npz, see_<out>_... (copied pose), validate_<out>.json

    $PY scripts/experiments/reachwm_20261030/vision_validate.py --tag wb100 --out wb100v --procs 6
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HANDS: dict = {}


def one(job):
    import room_grasp as RG

    k, j, tag, out, top = job
    V = RG.ROOM.OUT / "vision"
    f = V / f"grasp_{tag}_L{k}_s{j}.npz"
    row = dict(layout=k, slot=j, n=0)
    if not f.exists():
        return row
    z, pose = np.load(f), np.load(V / f"see_{tag}_L{k}_s{j}.npz")
    slot = json.load(open(RG.ROOM.OUT / f"layout_{k}" / "layout.json"))["slots"][j]
    n = len(z["score"])
    opts = []
    for strict, rmin in RG.tiers():
        opts = sorted(RG.grasp_options(slot, z["tool"], z["R"], z["score"], np.ones(n, bool), pose["p_o"], pose["R_o"], strict, rmin),
                      key=lambda c: -c[1]["s"])  # fmt: skip
        if opts:
            break
    if slot["id"] not in HANDS:
        sys.path.insert(0, str(HERE.parent / "graspnet_bottle_20261008"))
        import grasp_cache as GC

        HANDS[slot["id"]] = GC.HandModel(str(RG.ROOM.GN / "scenes" / f"obj_{slot['id']:03d}" / "scene_locomani.xml"))
    hm = HANDS[slot["id"]]
    held, score = np.zeros(n, bool), z["score"].astype(float).copy()
    for i, _, _, _ in opts[:top]:
        r = hm.run(z["tool"][i], z["R"][i])
        if r.get("held"):
            held[i], score[i] = True, float(r.get("score", 0.0)) + 0.01 * float(z["score"][i])
    row.update(
        n=n,
        options=len(opts),
        tried=min(top, len(opts)),
        held=int(held.sum()),
        first_held=bool(opts and held[opts[0][0]]),
    )
    checked = bool(held.any())
    np.savez_compressed(V / f"grasp_{out}_L{k}_s{j}.npz", tool=z["tool"], R=z["R"], score=score if checked else z["score"],
                        held=held if checked else np.ones(n, bool), checked=checked)  # fmt: skip
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--layouts", default="0,1,2,3,4,5,6")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import grasp_room as ROOM

    jobs = []
    for k in (int(x) for x in a.layouts.split(",")):
        for j in range(len(json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))["slots"])):
            jobs.append((k, j, a.tag, a.out, a.top))
            src = ROOM.OUT / "vision" / f"see_{a.tag}_L{k}_s{j}.npz"
            if src.exists():
                shutil.copy(src, ROOM.OUT / "vision" / f"see_{a.out}_L{k}_s{j}.npz")
    with mp.get_context("spawn").Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    json.dump(rows, open(ROOM.OUT / "vision" / f"validate_{a.out}.json", "w"))
    got = [r for r in rows if r["n"]]
    print(f"{len(got)} placements with grasps, {sum(r['held'] > 0 for r in got)} with a grasp that holds, "
          f"{sum(r['first_held'] for r in got)} whose best-scored grasp holds, tried {sum(r['tried'] for r in got)}")  # fmt: skip
