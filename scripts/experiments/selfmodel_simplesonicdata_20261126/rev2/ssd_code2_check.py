# notebook: selfmodel_simplesonicdata_20261126 -- revision 2, read only, CPU: code2's data.py against code's on the same
# records: x0 and u, column by column (no model, no gate number)
"""    ssd_code2_check.py dump <code dir> <data spec> <out.npz>     (data.load with <code dir>'s reachwm_20261030/data.py)
    ssd_code2_check.py cmp <code.npz> <code2.npz> <label>
cmp prints, per x0 column that differs, its index and the values on each side; u must be byte-equal; expected: only
x0 column 204 (walked) differs, code's value -> 0."""
import sys

import numpy as np

if sys.argv[1] == "dump":
    code, spec, out = sys.argv[2:5]
    sys.path[:0] = [f"{code}/scripts/experiments/reachwm_20261030", f"{code}/src"]
    import data as DT

    assert DT.__file__.startswith(code), DT.__file__
    D = DT.load(spec)
    np.savez(out, x0=D["x0"], u=D["u"], walked=np.asarray(D["walked"]), src=DT.__file__)
    print(f"dump {spec[:120]}: {len(D['x0'])} segments, x0 {D['x0'].shape}, u {D['u'].shape} with {DT.__file__}")
else:
    a, b, label = np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
    xa, xb = a["x0"], b["x0"]
    assert xa.shape == xb.shape and a["u"].shape == b["u"].shape
    diff = [c for c in range(xa.shape[1]) if not np.array_equal(xa[:, c], xb[:, c])]
    print(f"== {label}: {len(xa)} segments; {a['src']} vs {b['src']}")
    print(f"  u byte-equal: {np.array_equal(a['u'], b['u'])}; x0 columns that differ: {diff} of {xa.shape[1]}")
    for c in diff:
        va, vb = sorted(set(np.round(xa[:, c], 6).tolist())), sorted(set(np.round(xb[:, c], 6).tolist()))
        print(f"  x0[{c}]: code values {va[:6]}, code2 values {vb[:6]}; code's equals the walked flag: {np.array_equal(xa[:, c], a['walked'].astype(np.float32))}")
    same = [c for c in range(xa.shape[1]) if c not in diff]
    print(f"  every other x0 column byte-equal: {np.array_equal(xa[:, same], xb[:, same])}; code2's x0[204] all 0: {bool((xb[:, 204] == 0).all())}")
