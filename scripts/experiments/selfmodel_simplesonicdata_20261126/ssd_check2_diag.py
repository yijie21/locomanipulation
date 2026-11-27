# notebook: selfmodel_simplesonicdata_20261126 -- read-only diagnosis of check item (2) after data3b's SIMPLE CHECK FAILED (1)
"""Read only, CPU, no simulation. Recomputes sc_collect.check's item (2) on the settled_2cm end labels exactly as the check
does (the same 300 segments: default_rng(1)), and prints, per label row and hand: whether the hand is the segment's acting
hand, the FK-vs-W difference, and that hand's mean speed over the last row interval (|W[r] - W[r-1]| / dt, dt = 0.1 s).

    ssd_check2_diag.py --out <data dir> [--sub data_both] [--dt 0.02]
Run with the frozen code's env (ssd_collect.sh's exports) so sc_collect imports as in the check."""
import argparse
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--sub", default="data_both")
ap.add_argument("--dt", type=float, default=0.1, help="W's row interval, s (data.py ROW_S: a row every 5 frames at 50 Hz; "
                "revision 1: the first outputs, check2_diag.txt and check2_diag_all.txt, were made with 0.02, speeds 5x too high)")
ap.add_argument("--all-settled", action="store_true", help="every settled end label of the 300 (the printed, unchecked set), "
                "one summary line: max FK-W, and max of FK-W minus the hand's speed x 5 ms (one substep)")
a = ap.parse_args()
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["SX_REPO"], "scripts", "experiments", "selfmodel_simplecollect_20261126"))
sys.path.insert(0, os.path.join(os.environ["SX_REPO"], "src"))  # as check() does

import mujoco  # noqa: E402
import sc_collect as SC  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SX, L = SC.SX, SC.L
run = json.loads((a.out / "run.json").read_text())
bm = mujoco.MjModel.from_xml_path(str(Path(run["body"]["REACH_SCENE_DIR"]) / "scene_locomani.xml"))
bd = mujoco.MjData(bm)
names = SX._mj_names()
bq = np.array([bm.jnt_qposadr[bm.joint(nm).id] for nm in names])
brq = int(bm.jnt_qposadr[bm.joint("floating_base_joint").id])
bw = {s: bm.body(f"{s}_wrist_yaw_link").id for s in ("right", "left")}
c_off = np.asarray(run["calib"]["c"], float)
to_mj = np.array([L.SCALEBFM_ORDER.index(nm) for nm in names])
z = dict(np.load(a.out / a.sub / "all.npz"))
n = z["n_steps"].astype(int)
so, last = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n) - 1
side = z["side"]
W, S = z["W"].astype(float), z["S"].astype(float)
rng = np.random.default_rng(1)
idx = rng.choice(len(n), size=min(300, len(n)), replace=False)
lab = [i for i in idx if z["settled_end"][i] and not z["fell"][i] and (a.all_settled or z["settled_2cm"][i])]
print(f"{a.out.name}/{a.sub}: {len(n)} segments, {len(idx)} sampled, {len(lab)} settled_2cm end labels (the checked set)")
print(f"  seg   row      side  hand   acting  FK-W mm  hand speed cm/s (|W[r]-W[r-1]| / {a.dt} s)  other hand speed cm/s  "
      "FK-W / (speed x 5 ms)  flag (ratio > 1 and FK-W > 0.2 mm)")
worst = []
for i in lab:
    r = last[i]
    g = S[r, 58:61]
    th, ph = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    R_ = Rotation.from_euler("ZYX", [W[r, 3], th, ph])
    q = R_.as_quat()[[3, 0, 1, 2]]
    bd.qpos[:] = 0
    bd.qpos[brq : brq + 3], bd.qpos[brq + 3 : brq + 7], bd.qpos[bq] = W[r, :3] - R_.as_matrix() @ c_off, q, S[r, :29][to_mj]
    mujoco.mj_kinematics(bm, bd)
    acting = "right" if side[i] > 0 else "left"
    sp = {s: 100 * float(np.linalg.norm(W[r, col] - W[r - 1, col])) / a.dt for s, col in (("right", slice(4, 7)), ("left", slice(7, 10)))}
    for s, col in (("right", slice(4, 7)), ("left", slice(7, 10))):
        hb = bd.xpos[bw[s]] + bd.xmat[bw[s]].reshape(3, 3) @ L.HAND_OFF
        e = 1000 * float(np.linalg.norm(hb - W[r, col]))
        other = "left" if s == "right" else "right"
        if not a.all_settled:
            ratio = e / (10 * sp[s] * 0.005) if sp[s] > 0 else float("inf")
            print(f"  {i:5d} {r:8d}  {acting:5s} {s:5s}  {str(s == acting):6s}  {e:7.4f}  {sp[s]:8.3f}  {sp[other]:8.3f}  {ratio:8.2f}  "
                  f"{'FLAG' if ratio > 1 and e > 0.2 else ''}")
        worst.append((e, s == acting, sp[s]))
w = max(worst)
ex = max(e - 10 * v * 0.005 for e, _, v in worst)  # mm - (cm/s x 5 ms -> mm: x 10 x 0.005)
print(f"  {'every settled' if a.all_settled else 'settled_2cm'} labels {len(lab)}, hands {len(worst)}: max of FK-W minus speed x 5 ms "
      f"{ex:+.4f} mm; hands with FK-W > 0.2 mm {sum(e > 0.2 for e, _, _ in worst)}")
act = [e for e, ac, _ in worst if ac]
print(f"  max FK-W {w[0]:.4f} mm (acting hand {w[1]}, its speed {w[2]:.3f} cm/s); acting hands only: max {max(act, default=0):.4f} mm; "
      f"bound 0.2 mm")
