"""SIMPLE's two pick-and-place tasks with ScaleBFM and the Dex3 hand: pick as pick.py does, carry, release over the basket.

  LocomotionPickBetweenTablesTeleop  pick at table 1, turn round, walk to table 2, release over the basket there.
  HandoverTeleop                     pick, step to the left, release over the basket beside the object. SIMPLE's check
                                     for this task is only that the object touches the basket; this script does not
                                     pass the object from the right hand to the left.
The grasp comes from the head camera's cloud (see.py, grasps_graspgen.py). What the robot knows of the object in the
hand is what the look gave: the cloud's footprint, taken as fixed to the palm since the grasp. At the place the object is held aside and the head camera looks once at the basket (its pixels from the
simulator's segmentation); with --basket start the basket is taken to be where it stood at the start. The object is stood on
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
import see as SEE  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import WRIST, Motor, MotorConfig  # noqa: E402

SIDE = "right"
OVER_RIM = 0.02  # m, the object's foot above the basket's rim when it arrives over the basket
# m under the rim: where the object's foot is let go (negative: above it). In collision the basket is closed at its rim:
# MuJoCo collides the convex hull of each of its 28 pieces and the hulls span the mouth, so nothing enters it (an upright
# cracker box lowered anywhere over it stops at the rim). Placing in the basket is standing the object on top of it.
SEAT = -0.02
LEAN = 0.0  # rad, the object's top leant back toward the robot at the release
FROM_SIDE = (-30.0, 45.0)  # deg, the palm facing left: a hand beside the object leaves its top and foot free
HELD_ABOVE = 0.10  # m, the palm link above the object's foot in the grasp: at the release the fingers, which hang 4 cm
# below the palm link, are then 6 cm above the rim
PLACE_SIDE = -0.32  # m, the palm to the left of the pelvis at the release: further right than at the pick, so that the
# idle left hand, which hangs 0.2 m to the left of the pelvis, is not over the basket (it knocked the object off it)
LOOK_FROM = 0.5  # m before the place stance: where the robot stops to look at the basket after the long walk
ASIDE = (
    0.20,
    -0.45,
)  # m, the palm ahead and to the left of the pelvis for the look at the basket: the object out of view
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
    """Where the robot takes its pelvis to be, and the bridge between that frame and the simulator's.

    The robot's frame is the world as it was at the start, carried along by leg odometry over each walk; the task's
    goals are worked out in it. The Motor is commanded in the simulator's frame, where a goal means "this far from where
    the pelvis really is". So after each walk the two frames are tied at the pelvis: the place odometry gives (belief)
    and the place the simulator has (sim). A goal goes to the Motor through `to_sim`, which keeps it where it lies from
    the pelvis; the odometry's error then shows as a goal that is off in the world, as on a robot. While standing and
    reaching, leg_odo.JawFeed (load-weighted feet, arm joints) follows the pelvis and the palm from that tie on. Without
    odometry both places are the simulator's and nothing changes."""

    def __init__(self, sim, mo, odo: bool):
        self.sim, self.mo, self.odo, self.anchored = sim, mo, odo, False
        if odo:  # as grasp_room.make_env with BFM_ROOT=odo: ScaleBFM is told the pelvis position from leg odometry
            self.body = P.Body(sim, mo)
            mo.cfg = dataclasses.replace(mo.cfg, stop_ki=0.0, stop_timeout=mo.cfg.settle_s)
            self.body.odo_root = LO.OdoRoot(self.body)
            self.feed = LO.JawFeed(self.body)

    def _truth(self):
        return self.sim.d.qpos[:3].copy(), float(R.yaw_of(self.sim.d.qpos[3:7]))

    def start(self) -> None:
        """The robot's frame is the simulator's now."""
        (self.pb, self.yb), (self.pt, self.yt) = self._truth(), self._truth()

    def to_sim(self, p) -> np.ndarray:
        return self.pt + rot_z(self.yt - self.yb) @ (np.asarray(p, float) - self.pb)

    def to_sim_R(self, Rm) -> np.ndarray:
        return rot_z(self.yt - self.yb) @ Rm

    def from_sim(self, p) -> np.ndarray:
        return self.pb + (np.asarray(p, float) - self.pt) @ rot_z(self.yt - self.yb)

    def _pelvis_sim(self):
        """(position, rotation) of the pelvis in the simulator's frame as the robot's readings give it."""
        if self.odo and self.anchored:
            return self.feed.pelvis()
        return self.sim.d.qpos[:3].copy(), R.mat(self.sim.d.qpos[3:7])

    def pelvis(self):
        """(position, yaw) in the robot's frame."""
        p, Rp = self._pelvis_sim()
        return self.from_sim(p), float(np.arctan2(Rp[1, 0], Rp[0, 0])) - (self.yt - self.yb)

    def palm(self, side: str):
        """(position, rotation) of the palm in the robot's frame."""
        p, Rm = (getattr(self.mo, "jaw_source", None) or self.mo.jaw_centre)(side)
        return self.from_sim(p), rot_z(self.yb - self.yt) @ Rm

    def reach(self, side: str, p, Rm, **kw):
        return self.mo.reach(side, self.to_sim(p), self.to_sim_R(Rm), **P.cont(self.mo, side), **kw)

    def walk(self, xy, yaw: float, **kw):
        """Walk the pelvis to (xy, yaw) of the robot's frame; -> the odometry's error at the end (cm, deg)."""
        if self.odo:
            p0, yaw0 = self.pelvis()
            self.mo.jaw_source = None
            self.feed.odo = LO.LegOdometry(self.sim.m, self.sim.qadr, self.sim.vadr)  # the lower foot stays put
            self.feed.begin()
        self.mo.walk_to(tuple(self.to_sim(np.r_[xy, 0.0])[:2]), yaw + self.yt - self.yb, **kw)
        self.mo.hold(0.5)
        self.pt, self.yt = self._truth()
        if not self.odo:
            self.pb, self.yb = self.pt, self.yt
            return 0.0, 0.0
        self.pb, self.yb = p0 + rot_z(yaw0) @ self.feed.odo.p, yaw0 + self.feed.odo.yaw
        self.feed.anchor(self.pt, self.yt, WRIST, R.OFF)  # the tie: this pose of the simulator is that belief
        self.mo.jaw_source, self.anchored = self.feed.jaw, True
        return float(100 * np.linalg.norm((self.pb - self.pt)[:2])), float(np.degrees(R.wrap(self.yb - self.yt)))

    def scene(self):
        """The head camera's colour image, each pixel's point in the robot's frame and which pixels have a depth. The
        camera sits on the pelvis by the joint readings; the pelvis is the one of the robot's own estimate (before the
        first walk: the place and heading that define the robot's frame, with the IMU's tilt)."""
        rgb, depth = SEE.head_frame(self.sim)
        if not hasattr(self, "kin"):
            self.kin = SEE.HeadKin(self.sim.m, self.sim.qadr)
        rel_p, rel_R = self.kin(self.sim.d.qpos[self.sim.qadr])
        ps, Rs = self._pelvis_sim()
        c = SEE.cam_points(depth).reshape(-1, 3)
        return (
            rgb,
            self.from_sim(ps + (rel_p + c @ rel_R.T) @ Rs.T).reshape(*depth.shape, 3),
            (depth > 0.2) & (depth < 5),
        )

    def look(self, body: int) -> np.ndarray:
        """The body's points from the head camera's depth, in the robot's frame: the camera sits on the pelvis by the
        waist's joint readings (taken here as the simulator's camera pose in its pelvis frame)."""
        c, _, _ = SEE.head_points(self.sim, body)
        cam = self.sim.m.camera(EP.HEAD_CAM).id
        Rp = R.mat(self.sim.d.qpos[3:7])
        rel_p = Rp.T @ (self.sim.d.cam_xpos[cam] - self.sim.d.qpos[:3])
        rel_R = Rp.T @ self.sim.d.cam_xmat[cam].reshape(3, 3)
        ps, Rs = self._pelvis_sim()
        return self.from_sim(ps + (rel_p + c @ rel_R.T) @ Rs.T)


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
    ap.add_argument("--speed", type=float, default=0.5, help="m/s, the long walk between the tables")
    ap.add_argument("--odo", action="store_true")
    ap.add_argument(
        "--basket", default="look", choices=["look", "start"], help="where the basket is taken to be at the release"
    )
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
        bel.start()
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
        row["walk_cm"], _ = bel.walk((min(stance[0], edge - a.table_gap), stance[1]), yaw, face="fixed")
        to_sim = lambda M: np.block([[bel.to_sim_R(M[:3, :3]), bel.to_sim(M[:3, 3])[:, None]], [np.eye(4)[3:]]])  # noqa: E731
        row.update(P.pick(mo, sim, SIDE, to_sim(Tg), to_sim(Tp), f_close, f_pre, body=b))
        row["picked_cm"] = float(100 * (sim.d.xpos[b][2] - T_wo[2, 3]))

        # carry: the palm near the body, then to where the release point is within reach
        turn = np.pi if a.task == "LocomotionPickBetweenTablesTeleop" else 0.0
        yaw2 = yaw + turn
        Rr = rot_z(turn) @ Tg[:3, :3]  # the same hand attitude seen from the robot
        Rt = Rr @ rot_y(-LEAN)
        over = np.r_[centre, rim + OVER_RIM + 0.06] - Rr @ foot  # the palm with the object upright over the basket
        release = np.r_[centre, rim - SEAT] - Rt @ foot  # and with its foot on the rim
        stance2 = release[:2] - rot_z(yaw2)[:2, :2] @ [a.stand, PLACE_SIDE]
        if turn:  # table 2 stands behind the start; the robot comes to it facing -x
            stance2[0] = max(stance2[0], centre[0] + half[0] + a.table_gap)
        else:
            stance2[0] = min(stance2[0], edge - a.table_gap)
        pb, yb = bel.pelvis()
        palm, Rm = bel.palm(SIDE)
        bel.reach(SIDE, np.r_[pb[:2] + rot_z(yb)[:2, :2] @ CARRY, max(palm[2], over[2])], Rm, speed=0.15)
        row["dbg"] = dict(
            stance=np.round(stance, 3).tolist(),
            stance2=stance2.round(3).tolist(),
            edge=edge,
            centre=centre.round(3).tolist(),
            rim=rim,
            release=release.round(3).tolist(),
            over=over.round(3).tolist(),
            foot=foot.round(3).tolist(),
        )
        short = (
            LOOK_FROM if (turn and a.basket == "look") else 0.0
        )  # after a long walk: stop short, look, walk the rest
        row["walk2_cm"], row["walk2_deg"] = bel.walk(stance2 - short * rot_z(yaw2)[:2, 0], yaw2,
                                                     face="travel" if turn else "fixed", speed=a.speed if turn else None)  # fmt: skip
        row["carried"] = bool(P.touching(sim, b))
        if a.basket == "look":  # the object held aside, one look at the basket: where it is in the robot's own frame
            pb, yb = bel.pelvis()
            bel.reach(SIDE, np.r_[pb[:2] + rot_z(yb)[:2, :2] @ ASIDE, bel.palm(SIDE)[0][2]], Rr, speed=0.2)
            seen = bel.look(bk)
            row["basket_px"] = int(len(seen))
            if len(seen) >= 200:
                lo, hi = np.percentile(seen, [1, 99], axis=0)
                shift = np.r_[0.5 * (lo + hi)[:2] - centre, hi[2] - rim]  # against where the basket stood at the start
                row["look_cm"] = [float(v) for v in 100 * shift]
                over, release, stance2 = over + shift, release + shift, stance2 + shift[:2]
        if short:  # the rest of the way, on odometry
            row["walk3_cm"], _ = bel.walk(stance2, yaw2, face="fixed")
        bel.reach(SIDE, over, Rr, speed=0.15)
        bel.reach(SIDE, release, Rt, speed=0.05)
        if a.snap:
            snap(sim, np.r_[centre, rim], f"{a.snap}_{name}")
        row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(sim.d.xmat[b].reshape(3, 3)[:, 2] @ T_wo[:3, 2], -1, 1))))
        row["release_cm"] = float(100 * np.linalg.norm(mo.jaw_centre(SIDE)[0] - bel.to_sim(release)))
        row["over_cm"] = [
            float(v) for v in 100 * (sim.d.xpos[b][:2] - sim.d.xpos[bk][:2])
        ]  # the object from the basket
        row["held_at_release"], row["z_cm"] = P.touching(sim, b), float(100 * (sim.d.xpos[b][2] - rim))
        for u in np.linspace(0, 1, 15):
            sim.fingers_cmd[SIDE] = f_close + u * (WIDE - f_close)
            mo.hold(0.02)
        mo.hold(1.0)  # the finger joints follow their targets with a time constant of 0.4 s
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - 0.04 * Rm[:, 1], Rm, speed=0.05)  # off the object against the palm's normal,
        p, Rm = bel.palm(SIDE)
        bel.reach(
            SIDE, p - 0.08 * Rm[:, 1] + [0, 0, 0.08], Rm, speed=0.1
        )  # then away and up, clear of the basket's rim
        if a.snap:
            snap(sim, np.r_[centre, rim], f"{a.snap}_{name}_after")
        run = best = 0
        row["t_done"] = None
        for _ in range(75):  # SIMPLE's reward: the object on the basket (and, for one task, out of the hand) in a row
            mo.hold(0.02)
            on = contact(sim, b, lambda n: n == basket)
            free = not contact(sim, b, lambda n: "hand" in n)
            run = run + 1 if on and (free or not HANDS_OFF[a.task]) else 0
            best = max(best, run)
            if run == HELD[a.task] and row["t_done"] is None:
                row["t_done"] = float(mo.t)
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
