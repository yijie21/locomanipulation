"""selfmodel_relabel_20261125: the three models' data, loaded as train.py loads it (data.load + data.splits), must agree
segment for segment outside train: val, test_iid, the four bands and the edge set hold the same segments (identity: reset,
chain, k0, n_steps, start, target) with identical x0, u and labels. A's extra segments (k0 = 2) are all in train, from C's
train resets; B's train minus C's is the dropped entries. Read-only; exits 1 on any failure.

    cd <repo>; REACHWM_GOAL_REF=1 $PY split_check.py --A <spec> --B <spec> --C <spec> [--Am <spec>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

HELD_OUT = ("val", "test_iid", "band_x", "band_z", "band_yaw", "band_pel", "edge")
ARRAYS = ("x0", "u", "pos_f", "rot_f", "fell", "settled", "target", "rot6")


def ident(D: dict, idx: np.ndarray) -> list[bytes]:
    return [
        D["reset"][i].tobytes() + D["chain"][i].tobytes() + D["k0"][i].tobytes() + D["n_steps"][i].tobytes()
        + D["start"][i].tobytes() + D["target"][i].tobytes()
        for i in idx
    ]  # fmt: skip


def summary(spec: str) -> dict:
    D = DT.load(spec)
    sp = DT.splits(D)
    out = dict(n=len(D["n_steps"]), sizes={k: len(v) for k, v in sp.items()}, X0_DIM=DT.X0_DIM, x0_dim=D["x0"].shape[1])
    for k in HELD_OUT:
        ids = ident(D, sp[k])
        order = np.argsort(np.array(ids, dtype=object))
        out[k] = dict(ids=[ids[j] for j in order], arrays={a: D[a][sp[k][order]] for a in ARRAYS})
    tr = sp["train"]
    mid = D["k0"] == 2
    out["train_ids"] = set(ident(D, tr[~mid[tr]]))
    out["mid_total"] = int(mid.sum())
    out["mid_in_train"] = int(mid[tr].sum())
    out["mid_in"] = {k: int(mid[sp[k]].sum()) for k in HELD_OUT}
    out["mid_resets"] = set(D["reset"][mid].tolist())
    out["train_resets"] = set(D["reset"][tr[~mid[tr]]].tolist())
    out["mid_bucket_max"] = int(D["bucket"][mid].max()) if mid.any() else -1
    r = D["reset"]
    new = np.r_[True, r[1:] != r[:-1]]
    i = np.arange(len(r))
    place = i - np.maximum.accumulate(np.where(new, i, 0))  # meaningful for B (the originals) only
    out["train_entries"] = int((place[tr] == 2).sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for m in "ABC":
        ap.add_argument(f"--{m}", required=True)
    ap.add_argument("--Am", default=None, help="exploratory model Am's spec (C plus the moving mid starts), checked like A")
    a = ap.parse_args()
    S = {}
    for m in (["A", "B", "C"] + (["Am"] if a.Am else [])):
        S[m] = summary(getattr(a, m))
        print(f"{m}: {S[m]['n']} segments, x0 {S[m]['x0_dim']} (X0_DIM {S[m]['X0_DIM']}), splits {S[m]['sizes']}, "
              f"mid starts {S[m]['mid_total']} (in train {S[m]['mid_in_train']}, elsewhere {S[m]['mid_in']})", flush=True)  # fmt: skip
    ok = True
    for k in HELD_OUT:
        others = [m for m in S if m != "B"]
        same_ids = all(S[m][k]["ids"] == S["B"][k]["ids"] for m in others)
        same_arr = all(
            np.array_equal(S[m][k]["arrays"][x], S["B"][k]["arrays"][x]) for m in others for x in ARRAYS
        )
        dup = len(set(S["B"][k]["ids"])) != len(S["B"][k]["ids"])
        print(f"{k:9s}: {len(S['B'][k]['ids'])} segments; identities equal across {', '.join(S)}: {same_ids}; "
              f"x0/u/labels byte-equal: {same_arr}; duplicate identities: {dup}")  # fmt: skip
        ok &= same_ids and same_arr and not dup
    A, B, C = S["A"], S["B"], S["C"]
    c1 = A["train_ids"] == C["train_ids"]
    c2 = C["train_ids"] <= B["train_ids"]
    c3 = A["mid_in_train"] == A["mid_total"] and all(v == 0 for v in A["mid_in"].values())
    c4 = A["mid_resets"] <= C["train_resets"] and A["mid_bucket_max"] < 8
    c5 = B["mid_total"] == 0 and C["mid_total"] == 0
    c6 = len(B["train_ids"] - C["train_ids"]) == B["train_entries"]
    print(f"A's original train segments == C's train: {c1}; C's train inside B's: {c2} "
          f"(B - C = {len(B['train_ids'] - C['train_ids'])} segments = B's train entries {B['train_entries']}: {c6})")  # fmt: skip
    print(f"A's mid starts all in train, none in val/test_iid/bands/edge: {c3}; their resets inside C's train resets: {c4} "
          f"(max bucket {A['mid_bucket_max']}); no k0 = 2 segments in B or C: {c5}")  # fmt: skip
    ok &= c1 and c2 and c3 and c4 and c5 and c6
    if "Am" in S:
        Am = S["Am"]
        d1 = Am["train_ids"] == C["train_ids"]
        d2 = Am["mid_in_train"] == Am["mid_total"] and all(v == 0 for v in Am["mid_in"].values())
        d3 = Am["mid_resets"] <= C["train_resets"] and Am["mid_bucket_max"] < 8 and Am["mid_total"] <= A["mid_total"]
        print(f"Am (exploratory): original train == C's: {d1}; its {Am['mid_total']} mid starts all in train: {d2}; "
              f"resets inside C's train resets and no more starts than A: {d3}")  # fmt: skip
        ok &= d1 and d2 and d3
    print("SPLIT CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
