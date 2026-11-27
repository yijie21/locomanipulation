"""selfmodel_train3b_20261125: models M and D of one controller, their data loaded as train.py loads it (data.load +
data.splits), must agree segment for segment outside train (selfmodel_relabel_20261125's split_check.py, for M / D):
val, test_iid, the four bands and the edge set hold the same segments (identity: reset, chain, k0, n_steps, start,
target) with byte-equal x0, u and labels. D's extra segments (the mid starts, k0 = 2) are all in train and come from
M's train resets; D's other train segments are M's. Also: M holds no k0 = 2 segment; data3 and data3b share no reset
id; M's val / test_iid restricted to data3's resets are data3's own val / test_iid (the segments T3 is evaluated on,
none in T3's training); M's training fractions 0.3 / 0.6 / 1.0 are nested. Read-only; exits 1 on any failure.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $PY t3b_split_check.py --data3 <all.npz> \
        --data3b <all.npz> --mid <mid_data3.npz>,<mid_data3b.npz>
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
    out = dict(n=len(D["n_steps"]), sizes={k: len(v) for k, v in sp.items()}, X0_DIM=DT.X0_DIM, x0_dim=D["x0"].shape[1], D=D, sp=sp)
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
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data3", required=True)
    ap.add_argument("--data3b", required=True)
    ap.add_argument("--mid", required=True, help="the mid start files, comma-separated")
    a = ap.parse_args()
    specs = dict(M=f"{a.data3},{a.data3b}", D=f"{a.data3},{a.data3b},{a.mid}")
    S = {}
    for m, spec in specs.items():
        S[m] = summary(spec)
        print(f"{m}: {S[m]['n']} segments, x0 {S[m]['x0_dim']} (X0_DIM {S[m]['X0_DIM']}), splits {S[m]['sizes']}, "
              f"mid starts {S[m]['mid_total']} (in train {S[m]['mid_in_train']}, elsewhere {S[m]['mid_in']})", flush=True)  # fmt: skip
    ok = True
    for k in HELD_OUT:
        same_ids = S["D"][k]["ids"] == S["M"][k]["ids"]
        same_arr = all(np.array_equal(S["D"][k]["arrays"][x], S["M"][k]["arrays"][x]) for x in ARRAYS)
        dup = len(set(S["M"][k]["ids"])) != len(S["M"][k]["ids"])
        print(f"{k:9s}: {len(S['M'][k]['ids'])} segments; identities equal in M and D: {same_ids}; x0/u/labels byte-equal: "
              f"{same_arr}; duplicate identities: {dup}")  # fmt: skip
        ok &= same_ids and same_arr and not dup
    M, D = S["M"], S["D"]
    c1 = D["train_ids"] == M["train_ids"]
    c2 = D["mid_in_train"] == D["mid_total"] and all(v == 0 for v in D["mid_in"].values())
    c3 = D["mid_resets"] <= M["train_resets"] and D["mid_bucket_max"] < 8
    c4 = M["mid_total"] == 0 and D["mid_total"] > 0
    print(f"D's original train segments == M's train: {c1}; D's {D['mid_total']} mid starts all in train, none in "
          f"val/test_iid/bands/edge: {c2}; their resets inside M's train resets: {c3} (max bucket {D['mid_bucket_max']}); "
          f"no k0 = 2 segment in M: {c4}")  # fmt: skip
    ok &= c1 and c2 and c3 and c4
    # data3 / data3b: disjoint resets; data3's own val / test_iid = M's restricted to data3's resets
    D3 = DT.load(a.data3)
    sp3 = DT.splits(D3)
    r3 = set(D3["reset"].tolist())
    r3b = set(DT.load(a.data3b)["reset"].tolist())
    c5 = len(r3 & r3b) == 0
    MD, msp = M["D"], M["sp"]
    c6 = True
    for k in ("val", "test_iid"):
        mine = msp[k][np.isin(MD["reset"][msp[k]], list(r3))]
        same = sorted(ident(MD, mine)) == sorted(ident(D3, sp3[k]))
        c6 &= same
        print(f"data3's {k}: {len(sp3[k])} segments; = M's {k} on data3's resets ({len(mine)}): {same}")
    t3_train = set(D3["reset"][sp3["train"]].tolist())
    m_eval = set(MD["reset"][np.r_[msp["val"], msp["test_iid"]]].tolist())
    c7 = len(t3_train & m_eval) == 0
    f = {fr: set(DT.splits(MD, fr)["train"].tolist()) for fr in (0.3, 0.6, 1.0)}
    c8 = f[0.3] <= f[0.6] <= f[1.0]
    print(f"data3 / data3b reset ids disjoint: {c5} ({len(r3)} / {len(r3b)} resets); T3's train resets in M's val / "
          f"test_iid: {len(t3_train & m_eval)}; M's train fractions nested 0.3 <= 0.6 <= 1.0: {c8} "
          f"({len(f[0.3])} / {len(f[0.6])} / {len(f[1.0])} segments)")  # fmt: skip
    ok &= c5 and c6 and c7 and c8
    print("SPLIT CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
