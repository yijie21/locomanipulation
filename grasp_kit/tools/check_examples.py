"""The examples' results against what the benchmark got on the same looks (examples/expected).

    python tools/check_examples.py EXAMPLES OUT [OUT ...]

Per example and run: the target mask's overlap with the benchmark's (IoU), its centre and the table height
(cm apart), and the chosen grasp. With the look's own depth (--depth raw) the target must come out as the benchmark's;
with the stereo pair the depth is worked out again and may differ a little. GraspGen-X works in this kit's base frame,
the benchmark in the simulator's (turned by the robot's heading), so the grasps are different samples: the check is
that one is found on the object, and how far it is from the benchmark's.
"""

import json
import sys
from pathlib import Path

import numpy as np

ex, outs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
bad = 0
print(
    f"{'example':30s} {'run':8s} {'IoU':>5s} {'centre cm':>9s} {'table cm':>8s} {'depth cm':>8s} {'grasp':>6s} {'d cm':>5s} {'d deg':>6s}  verdict"
)
for fj in sorted((ex / "expected").glob("*.json")):
    name, e = fj.stem, json.load(open(fj))
    en = np.load(ex / "expected" / f"{name}.npz")
    look_depth = np.load(ex / "looks" / f"{name}.npz")["depth"]
    for out in outs:
        t, g = out / f"{name}.target.npz", out / f"{name}.grasp.json"
        if not t.exists() or not g.exists():
            print(f"{name:30s} {out.name:8s} missing output")
            bad += 1
            continue
        z, r = np.load(t), json.load(open(g))
        if "mask" not in z.files:
            verdict = "ok (no target, as expected)" if e.get("expect") == "not seen" else "FAIL: no target seen"
            print(f"{name:30s} {out.name:8s} {'-':>5s}  {verdict}")
            bad += verdict.startswith("FAIL")
            continue
        m, m0 = z["mask"], en["mask"]
        iou = (m & m0).sum() / max(1, (m | m0).sum())
        dc = 100 * np.linalg.norm(np.median(z["cloud"], axis=0) - np.median(en["cloud"], axis=0))
        dt = 100 * abs(float(z["table_top"]) - float(en["table_top"]))
        dd = 100 * float(np.median(np.abs(z["depth"] - look_depth)[m0 & (look_depth > 0)]))
        gs, dp, da = "none", float("nan"), float("nan")
        if r.get("grasp"):
            P, P0 = np.array(r["grasp"]["palm"]), np.array(e["grasp_palm"])
            gs, dp = f"{r['grasp']['score']:.2f}", 100 * np.linalg.norm(P[:3, 3] - P0[:3, 3])
            da = np.degrees(np.arccos(np.clip((np.trace(P[:3, :3].T @ P0[:3, :3]) - 1) / 2, -1, 1)))
        strict = out.name == "raw"
        if e.get("expect") == "wrong target":
            verdict = "ok (known failure: " + e["note"] + ")"
        elif iou < (0.95 if strict else 0.8) or dc > (0.5 if strict else 2) or dt > (0.5 if strict else 2):
            verdict = "FAIL: the target differs from the benchmark's"
        elif not r.get("grasp"):
            verdict = "FAIL: no takeable grasp"
        else:
            verdict = "ok"
        bad += verdict.startswith("FAIL")
        print(
            f"{name:30s} {out.name:8s} {iou:5.3f} {dc:9.2f} {dt:8.2f} {dd:8.2f} {gs:>6s} {dp:5.1f} {da:6.1f}  {verdict}"
        )
print("all ok" if not bad else f"{bad} FAIL")
sys.exit(1 if bad else 0)
