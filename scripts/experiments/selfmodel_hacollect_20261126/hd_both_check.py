"""selfmodel_hadata_20261126 revision 2b: the two-hand segments (data_both, touch-filtered by hd_touch.py) that gate (b)'s
two-hand row is scored on are held out from the training, read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> \
        $PY hd_both_check.py --train <D's data spec> --both <both_d3.npz>,<both_d3b.npz>

data.py splits by the reset id (bucket = crc32("reach" + reset) % 10: train < 8, val 8, test_iid 9; the band and edge
segments are held out per segment by fixed command thresholds), so a reset is in one split whichever file it is in.
Checks, any failure exits 1:
  1. data.load reads the two-hand file (the reader the evaluation uses) and every segment is a two-hand one (hands == 2);
  2. the reset ids of the two-hand val and test_iid segments share none with the training segments' reset ids (D's
     spec, data.splits' train at frac 1.0, which holds the 0.6 and 0.3 subsets), nor with each other;
  3. each of those resets' bucket, recomputed from the id, is 8 (val) / 9 (test_iid).
Prints the counts and "BOTH SPLIT CHECK PASS" / "... FAIL".
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--both", required=True)
    a = ap.parse_args()
    T, B = DT.load(a.train), DT.load(a.both)
    st, sb = DT.splits(T), DT.splits(B)
    ok = True
    hands = np.asarray(B["hands"]) if "hands" in B else None
    two = hands is not None and bool((hands == 2).all())
    print(f"two-hand file: {len(B['n_steps'])} segments, hands == 2 for all: {two}; sides "
          f"{dict(zip(*[x.tolist() for x in np.unique(B['side'], return_counts=True)]))}")  # fmt: skip
    ok &= two
    tr = set(T["reset"][st["train"]].tolist())
    for k in ("val", "test_iid"):
        r = set(B["reset"][sb[k]].tolist())
        inter = r & tr
        bk = DT.hash_bucket(np.array(sorted(r), np.int64), "reach", 10) if r else np.zeros(0, int)
        want = 8 if k == "val" else 9
        good = not inter and bool((bk == want).all())
        print(f"  {k}: {len(sb[k])} segments over {len(r)} resets; resets also in training: {len(inter)}; bucket {want} for all: {bool((bk == want).all())}")
        ok &= good
    vt = set(B["reset"][sb["val"]].tolist()) & set(B["reset"][sb["test_iid"]].tolist())
    print(f"  val and test_iid share {len(vt)} resets; training: {len(st['train'])} segments over {len(tr)} resets")
    ok &= not vt
    print("BOTH SPLIT CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
