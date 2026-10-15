"""TODO 26.3.2 (docs/handoff/scalebfm-bottle-rsi-20261015.md §4): the bottle loop's three start-state pools.

* diag: L1's 10 SONIC rounds (renamed sonic_l1_*, their labels collide with L2's) and L2 nominal's rounds with RTF >= 0.85;
* accept / sealed: the first 60 / 80 rounds of the SONIC pool runs, in label (seed) order, with RTF >= 0.80 (plan §4); the rest
  move to pool_<name>/excluded/ (never deleted).

    python3 scripts/experiments/scalebfm_bottle_rsi_20261015/prep_pools.py diag|accept|sealed
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CC = REPO / "outputs" / "controller_compare_20261013"
LOOP = REPO / "outputs" / "scalebfm_bottle_rsi_20261015"
HIDDEN = REPO / "outputs" / "scalebfm_bottle_rsi_20261015_hidden"
RTF_MIN = 0.85  # the diagnostic pool keeps L2's rule
RTF_POOL = 0.80  # accept / sealed (plan §4, 10-15): this machine runs the bottle stack at 0.80-0.86 recorded or not;
#                  ScaleBFM's start does not depend on it, SONIC's outcome is checked against RTF separately
SIZE = {"accept": 60, "sealed": 80}


def copy_round(src: Path, dst_dir: Path, label: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".npz", ".json"):
        shutil.copy2(src.with_suffix(ext), dst_dir / f"{label}{ext}")


def diag() -> None:
    out = LOOP / "pool_diag" / "states"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists")
    n = 0
    for p in sorted((CC / "bottle" / "sonic" / "states").glob("sonic_*.npz")):
        copy_round(p, out, p.stem.replace("sonic_", "sonic_l1_", 1))
        n += 1
    void = []
    for p in sorted((CC / "l2" / "bottle" / "sonic_nominal" / "states").glob("*.npz")):
        rtf = json.loads(p.with_suffix(".json").read_text()).get("rtf")
        if rtf is not None and rtf < RTF_MIN:
            void.append(p.stem)
            continue
        copy_round(p, out, p.stem)
        n += 1
    print(f"diag: {n} starts; L2 void (RTF < {RTF_MIN}): {void}")


def pool(name: str) -> None:
    d = HIDDEN / f"pool_{name}" / "states"
    keep, drop = [], []
    for p in sorted(d.glob("*.npz")):
        rtf = json.loads(p.with_suffix(".json").read_text()).get("rtf")
        (keep if (rtf is None or rtf >= RTF_POOL) and len(keep) < SIZE[name] else drop).append(p)
    ex = d.parent / "excluded"
    ex.mkdir(exist_ok=True)
    for p in drop:
        for ext in (".npz", ".json"):
            shutil.move(str(p.with_suffix(ext)), ex / p.with_suffix(ext).name)
    ok = sum(bool(json.loads(p.with_suffix(".json").read_text())["ok"]) for p in keep)
    print(f"{name}: kept {len(keep)} (SONIC ok {ok}), moved {len(drop)} to {ex}")
    if len(keep) < SIZE[name]:
        print(f"WARNING: only {len(keep)} usable starts, plan wants {SIZE[name]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    diag() if which == "diag" else pool(which) if which in SIZE else sys.exit(__doc__)
