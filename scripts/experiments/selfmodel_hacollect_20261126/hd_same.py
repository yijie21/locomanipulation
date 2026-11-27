"""selfmodel_hadata_20261126: the frozen copy's test -- chunks re-driven from asrun against the dev smoke's chunks of the
same resets: every array both have, exactly. Read only.   python hd_same.py REF_DIR NEW_DIR"""

import sys
from pathlib import Path

import numpy as np

ref, new = Path(sys.argv[1]), Path(sys.argv[2])
bad, n = 0, 0
for sub in ("data", "data_both", "data_detour", "data/arm", "data_both/arm", "data_detour/arm"):
    for p in sorted((ref / sub).glob("w*_*.npz")):
        q = new / sub / p.name
        if not q.exists():
            print(f"{sub}/{p.name}: missing in new"); bad += 1; continue
        a, b = np.load(p), np.load(q)
        only = sorted(set(b.files) - set(a.files))
        for k in a.files:
            n += 1
            if k not in b.files:
                print(f"{sub}/{p.name} {k}: missing in new"); bad += 1; continue
            x, y = a[k], b[k]
            if x.shape != y.shape:
                print(f"{sub}/{p.name} {k}: shape {x.shape} vs {y.shape}"); bad += 1; continue
            if x.dtype.kind in "fc":
                same = np.array_equal(x, y, equal_nan=True)
            else:
                same = np.array_equal(x, y)
            if not same:
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float))) if x.dtype.kind in "fcib" else float("nan")
                print(f"{sub}/{p.name} {k}: differs, max |d| {d:.3g}"); bad += 1
        print(f"{sub}/{p.name}: {len(a.files)} arrays compared; only in new: {only}")
print(f"{'SAME' if bad == 0 else 'DIFFERENT'}: {n} arrays, {bad} differing")
