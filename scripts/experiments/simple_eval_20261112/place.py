"""SIMPLE's two pick-and-place tasks with ScaleBFM and the Dex3 hand: pick as pick.py does, carry, release over the basket.

  LocomotionPickBetweenTablesTeleop  pick at table 1, turn round, walk to table 2, release over the basket there.
  HandoverTeleop                     pick, step to the left, release over the basket beside the object. SIMPLE's check
                                     for this task is only that the object touches the basket; this script does not
                                     pass the object from the right hand to the left.
The grasp comes from the head camera's cloud (see.py, grasps_graspgen.py). What the robot knows of the object in the
hand is what the look gave: the cloud's footprint, taken as fixed to the palm since the grasp. The object is stood on
the basket's rim and let go there (in collision the basket is closed at the rim, see SEAT). The basket's place and rim height are read from the simulator at the start (privileged). With --odo the pelvis and the palm come from leg odometry and joints.

    BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python place.py LocomotionPickBetweenTablesTeleop --odo
-> one line per episode and "success k of n"; with --out, a json of the rows
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reachwm_20261030"))
import episodes as EP  # noqa: E402
import leg_odo as LO  # noqa: E402
import pick as P  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import WRIST, Motor, MotorConfig  # noqa: E402

SIDE = "right"
OVER_RIM = 0.02  # m, the object's foot above the basket's rim when it arrives over the basket
SEAT = (
    -0.005
)  # m under the rim: where the object's foot is let go. In collision the basket is closed at its rim: MuJoCo
# collides the convex hull of each of its 28 pieces and the hulls span the mouth, so nothing enters it (an upright
# cracker box lowered anywhere over it stops at the rim). Placing in the basket is standing the object on top of it.
LEAN = 0.0  # rad, the object's top leant back toward the robot at the release
FROM_SIDE = (-30.0, 45.0)  # deg, the palm facing left: a hand beside the object leaves its top and foot free
HELD_ABOVE = 0.10  # m, the palm link above the object's foot in the grasp: at the release the fingers, which hang 4 cm
# below the palm link, are then 6 cm above the rim
CARRY = (0.25, -0.22)  # m, the palm ahead and to the left of the pelvis while walking with the object
WIDE = np.array([0.0, 1.0, 0.0, -0.3, 0.0, -0.3, 0.0])  # the Dex3 fingers as far open as their joints go, to let go
HELD = {"LocomotionPickBetweenTablesTeleop": 20, "HandoverTeleop": 9}  # ticks in a row SIMPLE's reward must hold
HANDS_OFF = {"LocomotionPickBetweenTablesTeleop": True, "HandoverTeleop": False}  # whether the hand must have let go


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Belief:
    """Where the robot takes its pelvis to be. Without odometry: the simulator's pose. With it: the start pose, then leg
    odometry over each walk and leg_odo.JawFeed (load-weighted feet, arm joints) while standing and reaching."""

    def __init__(self, sim, mo, odo: bool):
        self.sim, self.mo, self.odo, self.anchored = sim, mo, odo, False
        if odo:  # as grasp_room.make_env with BFM_ROOT=odo: ScaleBFM is told the pelvis position from leg odometry
            self.body = P.Body(sim, mo)
            mo.cfg = dataclasses.replace(mo.cfg, stop_ki=0.0, stop_timeout=mo.cfg.settle_s)
            self.body.odo_root = LO.OdoRoot(self.body)
            self.feed = LO.JawFeed(self.body)

    def pelvis(self):
        """(position, yaw)."""
        if self.odo and self.anchored:
            p, Rp = self.feed.pelvis()
            return p, float(np.arctan2(Rp[1, 0], Rp[0, 0]))
        q = self.sim.d.qpos
        return q[:3].copy(), float(R.yaw_of(q[3:7]))

    def walk_begin(self) -> None:
        if self.odo:
            self.p0, self.yaw0 = self.pelvis()
            self.mo.jaw_source = None
            self.feed.odo = LO.LegOdometry(self.sim.m, self.sim.qadr, self.sim.vadr)  # the lower foot stays put
            self.feed.begin()

    def walk_end(self) -> float:
        """Anchors the standing estimate where the walk's odometry ends; -> its distance from the truth, cm."""
        if not self.odo:
            return 0.0
        p = self.p0 + rot_z(self.yaw0) @ self.feed.odo.p
        self.feed.anchor(p, self.yaw0 + self.feed.odo.yaw, WRIST, R.OFF)
        self.mo.jaw_source, self.anchored = self.feed.jaw, True
        return float(100 * np.linalg.norm((p - self.sim.d.qpos[:3])[:2]))

    def palm(self, side: str):
        return (getattr(self.mo, "jaw_source", None) or self.mo.jaw_centre)(side)


def basket_of(sim, body: int):
    """The basket's centre (xy), its rim height and its half extents along x and y, from the simulator."""
    m, d = sim.m, sim.d
    V = []
    for g in np.flatnonzero(m.geom_bodyid == body):
        k = m.geom_dataid[g]
        V.append(
            d.geom_xpos[g]
            + m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ d.geom_xmat[g].reshape(3, 3).T
        )
    V = np.vstack(V)
    return 0.5 * (V.min(0) + V.max(0))[:2], float(V[:, 2].max()), 0.5 * (V.max(0) - V.min(0))[:2]


def snap(sim, at, prefix: str) -> None:
    import mujoco
    from PIL import Image

    sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = 640, 480
    r = mujoco.Renderer(sim.m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance = at, 0.9
    for k, (az, el) in enumerate(((90, -20), (0, -35))):
        cam.azimuth, cam.elevation = az, el
        r.update_scene(sim.d, camera=cam)
        Image.fromarray(r.render()).save(f"{prefix}_{k}.png")
    r.close()


def contact(sim, body: int, other) -> bool:
    """Whether `body` touches a body whose name `other` accepts."""
    m, d = sim.m, sim.d
    for i in range(d.ncon):
        a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        if (a == body and other(m.body(b).name)) or (b == body and other(m.body(a).name)):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(HELD))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stand", type=float, default=0.27)
    ap.add_argument("--side-off", type=float, default=P.SIDE_OFF)
    ap.add_argument("--table-gap", type=float, default=0.2)
    ap.add_argument("--odo", action="store_true")
    ap.add_argument("--snap", help="path prefix: two pictures of the basket at the moment before the hand opens")
    ap.add_argument("--out")
    a = ap.parse_args()
    R.OFF[:] = P.PALM
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    rows = []
    for ep in EP.episodes(a.task)[: a.n]:
        info = EP.build(ep)
        sim = R.Sim(pol.meta, backend=ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""))
        mo = Motor(sim, pol, MotorConfig(), model_xml=info["dir"] + "/scene_locomani.xml")
        bel = Belief(sim, mo, a.odo)
        mo.reset(info["start"]["xy"], info["start"]["yaw"])
        EP.pull_objects(sim.m, sim.d, info)
        b, bk = sim.m.body(info["target"]).id, sim.m.body(info["objects"]["container"]["body"]).id
        basket = sim.m.body(bk).name
        mo.hold(1.0)
        name = f"{a.task}_L{ep['level']}_e{ep['index']}"
        row = dict(level=ep["level"], index=ep["index"])
        T_wo, yaw = P.T(sim.d.xpos[b], sim.d.xquat[b]), info["start"]["yaw"]
        centre, rim, half = basket_of(sim, bk)
        found = P.graspgen_grasp(
            name, T_wo, yaw, clear=HELD_ABOVE, roll_range=FROM_SIDE
        )  # held from the side, high enough to be lowered to its seat
        if found is None:
            rows.append(dict(row, success=False, none=True))
            print(rows[-1], flush=True)
            continue
        T_og, T_op, f_close, f_pre, row["score"] = found
        Tg, Tp = T_wo @ T_og, T_wo @ T_op
        cloud = np.load(EP.OUT / "cloud" / f"{name}.npz")["pts"]
        lo, hi = np.percentile(
            cloud, [1, 99], axis=0
        )  # the head sees the object's front and top: their span is its footprint
        foot = Tg[:3, :3].T @ (
            np.r_[0.5 * (lo + hi)[:2], lo[2]] - Tg[:3, 3]
        )  # the middle of the object's foot in the palm's
        # frame at the grasp, taken as fixed to the palm from then on
        tables = [t for k, t in info["tables"].items() if k == "table"]
        edge = min(t["pos"][0] - 0.5 * t["size"][0] for t in tables)
        stance = Tg[:2, 3] - rot_z(yaw)[:2, :2] @ [a.stand, a.side_off]
        bel.walk_begin()
        mo.walk_to((min(stance[0], edge - a.table_gap), stance[1]), yaw, face="fixed")
        mo.hold(0.5)
        row["walk_cm"] = bel.walk_end()
        row.update(P.pick(mo, sim, SIDE, Tg, Tp, f_close, f_pre, body=b))
        row["picked_cm"] = float(100 * (sim.d.xpos[b][2] - T_wo[2, 3]))

        # carry: the palm near the body, then to where the release point is within reach
        turn = np.pi if a.task == "LocomotionPickBetweenTablesTeleop" else 0.0
        yaw2 = yaw + turn
        Rr = rot_z(turn) @ Tg[:3, :3]  # the same hand attitude seen from the robot
        Rt = Rr @ rot_y(-LEAN)  # turned about the palm's normal: the object's top leans back toward the robot
        over = np.r_[centre, rim + OVER_RIM + 0.06] - Rr @ foot  # the palm with the object upright over the basket
        release = np.r_[centre, rim - SEAT] - Rt @ foot  # and with its foot between the basket's walls, leaning
        stance2 = release[:2] - rot_z(yaw2)[:2, :2] @ [a.stand, a.side_off]
        if turn:  # table 2 stands behind the start; the robot comes to it facing -x
            stance2[0] = max(stance2[0], centre[0] + half[0] + a.table_gap)
        else:
            stance2[0] = min(stance2[0], edge - a.table_gap)
        pb, yb = bel.pelvis()
        _, Rm = bel.palm(SIDE)
        mo.reach(SIDE, np.r_[pb[:2] + rot_z(yb)[:2, :2] @ CARRY, max(bel.palm(SIDE)[0][2], over[2])], Rm, speed=0.15)
        bel.walk_begin()
        mo.walk_to(tuple(stance2), yaw2, face="travel" if turn else "fixed")
        mo.hold(0.5)
        row["walk2_cm"] = bel.walk_end()
        row["carried"] = bool(P.touching(sim, b))
        mo.reach(SIDE, over, Rr, speed=0.15)
        mo.reach(SIDE, release, Rt, speed=0.05)
        if a.snap:
            snap(sim, np.r_[centre, rim], f"{a.snap}_{name}")
        row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(sim.d.xmat[b].reshape(3, 3)[:, 2] @ T_wo[:3, 2], -1, 1))))
        row["release_cm"] = float(100 * np.linalg.norm(mo.jaw_centre(SIDE)[0] - release))
        row["over_cm"] = [
            float(v) for v in 100 * (sim.d.xpos[b][:2] - sim.d.xpos[bk][:2])
        ]  # the object from the basket
        row["held_at_release"], row["z_cm"] = P.touching(sim, b), float(100 * (sim.d.xpos[b][2] - rim))
        for u in np.linspace(0, 1, 15):
            sim.fingers_cmd[SIDE] = f_close + u * (WIDE - f_close)
            mo.hold(0.02)
        mo.hold(1.0)  # the finger joints follow their targets with a time constant of 0.4 s
        p, Rm = bel.palm(SIDE)
        mo.reach(SIDE, p - 0.12 * Rm[:, 1], Rm, speed=0.1)  # back against the palm's normal
        run = best = 0
        for _ in range(75):  # SIMPLE's reward: the object on the basket (and, for one task, out of the hand) in a row
            mo.hold(0.02)
            on = contact(sim, b, lambda n: n == basket)
            free = not contact(sim, b, lambda n: "hand" in n)
            run = run + 1 if on and (free or not HANDS_OFF[a.task]) else 0
            best = max(best, run)
        row.update(success=bool(best >= HELD[a.task]), in_basket_cm=float(100 * np.linalg.norm(sim.d.xpos[b][:2] - centre)),
                   basket_cm=float(100 * np.linalg.norm(sim.d.xpos[bk][:2] - info["objects"]["container"]["pos"][:2])),
                   fell=mo.fell_at, t=float(mo.t))  # fmt: skip
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    print("success", sum(r["success"] for r in rows), "of", len(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
