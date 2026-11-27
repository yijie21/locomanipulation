"""cap_check2.py: Figure 1 v3b (the grasp raised to ~1.35 m), read only. notebook: skip figure render, not an experiment.

The same data and filter as cap_check.py (data3 + data3b, right hand, not fell / lost, settled_end): the target z in
1 cm bins over 1.25-1.45 m, and, for the 20 placements' two goals (pregrasp, entry; shelf_scene's x, y rule) at grasp
heights 1.26 (v3a), 1.33, 1.34, 1.35, 1.36 m, how many segments' targets lie within 5 cm (3-D; target = heading xy,
world z; the robot starts at the origin facing +x, so heading xy = world xy here up to the root's ~1 cm offset).
"""

import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import shelf_scene as SS  # noqa: E402

L = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126"
T, ST = [], []
for name in ("data3", "data3b"):
    z = np.load(f"{L}/{name}/data/all.npz")
    keep = (~z["fell"].astype(bool)) & (~z["lost"].astype(bool)) & z["settled_end"].astype(bool) & (z["side"] == 1)
    T.append(z["target"].astype(float)[keep])
    ST.append((z["pelvis_z"] < 0)[keep])
T, ST = np.concatenate(T), np.concatenate(ST)
print(f"right hand, kept: {len(T)} segments (standing {int(ST.sum())})")
print("target z, 1 cm bins (all kept | standing):")
for lo in np.arange(1.25, 1.45, 0.01):
    s = (T[:, 2] >= lo - 1e-9) & (T[:, 2] < lo + 0.01 - 1e-9)
    print(f"  {lo:.2f}-{lo + 0.01:.2f}: {int(s.sum()):4d} | {int((s & ST).sum()):4d}")
PL = SS.placements()
print("segments with the target within 5 cm of the goal (median / min over the 20 placements; all kept | standing):")
for h in (1.26, 1.33, 1.34, 1.35, 1.36):
    out = []
    for stage, dy in (("pregrasp", -0.08), ("entry", 0.0)):
        c = []
        for p in PL:
            g = np.array([p["x"] - SS.ALONG, p["y"] - SS.NORMAL_OFF + dy, h])
            d = np.linalg.norm(T - g, axis=1) < 0.05
            c.append((int(d.sum()), int((d & ST).sum())))
        c = np.array(c)
        out.append(f"{stage} {int(np.median(c[:, 0]))}/{c[:, 0].min()} | {int(np.median(c[:, 1]))}/{c[:, 1].min()}")
    print(f"  grasp z {h:.2f}: " + "; ".join(out))
