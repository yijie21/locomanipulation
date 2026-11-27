"""selfmodel_hadata_20261126 revision 2a (2026-11-26): is an hr_real episode the same as an earlier run's? Compares
data/<name>.npz and data/arm/<name>.npz array by array (bytes, dtype, shape) and the log json key by key, leaving out
the wall-clock fields (wall_s, plan_s, wait timings of the host's planner) and the keys only NEW has (e.g. "contacts").

    python hr_same.py NEW_DIR OLD_DIR NAME      # NAME e.g. ppbox_s296_r0; exit 0 when identical
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLOCK = {"wall_s", "plan_s"}


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in CLOCK}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def main() -> int:
    new, old, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    bad, n = [], 0
    for sub in ("data", "data/arm"):
        A, B = np.load(new / sub / f"{name}.npz"), np.load(old / sub / f"{name}.npz")
        if set(A.files) != set(B.files):
            bad.append(f"{sub}: arrays differ {sorted(set(A.files) ^ set(B.files))}")
        for k in sorted(set(A.files) & set(B.files)):
            n += 1
            a, b = A[k], B[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                bad.append(f"{sub}/{k}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
    La = json.loads((new / "log" / f"{name}.json").read_text())
    Lb = json.loads((old / "log" / f"{name}.json").read_text())
    extra = sorted(set(La) - set(Lb))
    for k in sorted(set(Lb)):
        if k not in La:
            bad.append(f"log: {k} missing in new")
        elif strip(La[k]) != strip(Lb[k]):
            bad.append(f"log: {k} differs")
    print(f"{name}: {n} arrays compared; log keys compared {len(Lb)}, only in new {extra}")
    for b in bad:
        print("  DIFF", b)
    print("IDENTICAL" if not bad else f"DIFFERENT ({len(bad)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
