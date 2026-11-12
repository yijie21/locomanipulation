"""The pick tasks of SIMPLE's evaluation set with ScaleBFM and the Dex3 hand: walk up, reach, close three fingers, lift.

Privileged baseline: the object's pose is read from the simulator and the servo reads the palm's true place. The grasp is
either SIMPLE's own stored Dex3 grasp for the object (--grasp stored) or palm poses from a file written by
grasps_graspgen.py (--grasp <npz>), tried in the file's order of score after the reach filter.

    BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python pick.py XMovePickTeleop --side right --stand 0.27
-> one line per episode and "success k of n"; with --out, a json of the rows
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episodes as EP  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import Motor, MotorConfig  # noqa: E402

GRASPNET = EP.SIMPLE / "data" / "assets" / "graspnet"
PALM = (0.0415, 0.0, 0.0)  # the Dex3 palm link in the wrist-yaw frame (the y offset of 3 mm is left out)
BACK = 0.06  # m, the reach starts this far behind the pregrasp, on the line from the grasp through the pregrasp


def T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def stored_grasp(side: str, obj: str = "0"):
    """SIMPLE's Dex3 grasp for a GraspNet object: palm poses in the object's frame and the 7 finger joints,
    as (grasp, pregrasp, fingers at the grasp, fingers at the pregrasp)."""

    def load(f):
        v = torch.load(GRASPNET / f"dex_grasp/dex3_{side}/{obj}/{f}", map_location="cpu", weights_only=True)
        return v.numpy().ravel().astype(float)

    g, pre = load("0__0.pt"), load("0__0_pregrasp.pt")
    st = np.load(GRASPNET / f"stable/{obj}_stable.npy", allow_pickle=True)[0]
    T_ob = np.linalg.inv(T([0.4, 0.0, st[2]], st[3:]))  # the grasps are stored with the object at this pose
    return T_ob @ T(g[:3], g[3:7]), T_ob @ T(pre[:3], pre[3:7]), g[7:], pre[7:]


def touching(sim, body: int) -> list:
    """The hand links in contact with the body."""
    m, d, out = sim.m, sim.d, set()
    for i in range(d.ncon):
        a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        other = b if a == body else a if b == body else None
        if other is not None and ("_hand_" in m.body(other).name or "wrist" in m.body(other).name):
            out.add(m.body(other).name.split("_hand_")[-1])
    return sorted(out)


def pick(mo, sim, side: str, Tg, Tp, f_close, f_pre, squeeze: float = 0.6, close_s: float = 0.5) -> dict:
    """Reach to the pregrasp, then the grasp, close the fingers past the stored closure by `squeeze`, lift 15 cm."""
    step = Tp[:3, 3] - Tg[:3, 3]
    sim.fingers_cmd[side] = f_pre
    mo.reach(side, Tp[:3, 3] + BACK * step / max(1e-6, np.linalg.norm(step)), Tp[:3, :3])
    mo.reach(side, Tp[:3, 3], Tp[:3, :3], speed=0.1)
    mo.reach(side, Tg[:3, 3], Tg[:3, :3], speed=0.08)
    p, Rm = mo.jaw_centre(side)
    err = 100 * np.linalg.norm(p - Tg[:3, 3])
    ang = np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3].T @ Rm) - 1) / 2, -1, 1)))
    for u in np.linspace(0, 1, int(close_s / 0.02)):
        sim.fingers_cmd[side] = f_pre + u * (f_close + squeeze * (f_close - f_pre) - f_pre)
        mo.hold(0.02)
    mo.hold(0.5)
    p, Rm = mo.jaw_centre(side)
    mo.reach(side, p + [0, 0, 0.15], Rm, speed=0.1)
    mo.hold(1.0)
    return dict(palm_cm=float(err), palm_deg=float(ang))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stand", type=float, default=0.27, help="m, the palm's target ahead of the pelvis at the stance")
    ap.add_argument("--grasp", default="stored")
    ap.add_argument("--out")
    a = ap.parse_args()
    R.OFF[:] = PALM
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    rows = []
    for ep in EP.episodes(a.task)[: a.n]:
        info = EP.build(ep)
        sim = R.Sim(pol.meta, backend=ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""))
        mo = Motor(sim, pol, MotorConfig(), model_xml=info["dir"] + "/scene_locomani.xml")
        mo.reset(info["start"]["xy"], info["start"]["yaw"])
        EP.pull_objects(sim.m, sim.d, info)
        b = sim.m.body(info["target"]).id
        mo.hold(1.0)
        z0 = float(sim.d.xpos[b][2])
        T_wo = T(sim.d.xpos[b], sim.d.xquat[b])
        if a.grasp == "stored":
            T_og, T_op, f_close, f_pre = stored_grasp(a.side)
        else:
            raise SystemExit("--grasp <npz> is not wired yet")
        Tg, Tp = T_wo @ T_og, T_wo @ T_op
        mo.walk_to((Tg[0, 3] - a.stand, sim.d.qpos[1]), info["start"]["yaw"], face="fixed")
        mo.hold(0.5)
        row = dict(level=ep["level"], index=ep["index"], **pick(mo, sim, a.side, Tg, Tp, f_close, f_pre))
        z = float(sim.d.xpos[b][2])
        row.update(touch=touching(sim, b), rise_cm=100 * (z - z0), success=bool(EP.lifted(a.task, z0, z)),
                   fell=mo.fell_at, t=float(mo.t))  # fmt: skip
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    print("success", sum(r["success"] for r in rows), "of", len(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
