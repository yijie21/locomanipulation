"""The robot's belief about where it is, and the frame tie to the simulator (TODO 49.2).

Moved from simple_eval_20261112/place.py (Belief, rot_y, rot_z, SIDE, WIDE), simple_eval_20261112/pick.py (Body, cont: what
leg odometry's feeds read of the robot, and the Motor.reach keywords that continue a reach) and escalation_20261116/seed_pkg
skill_simple.py (the Belief subclass with the wait after a walk). Fixed infrastructure, not loop-editable: `Body.pelvis` and
`Belief._truth` read the simulator's true base pose, which is the privileged TIE of this bed (the robot's frame is the world
as it was at the start, tied to the simulator's at each walk's end; odometry gives the belief, the tie keeps goals consistent).
Removing the true pose from here changes behaviour (TODO 49.3, self_pose / controller_root).

`Belief.look` (the target's points from the simulator's segmentation) and place.py's other helpers are privileged baselines the
loop path never calls: not ported.
"""

import dataclasses

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import WRIST
from locomani.perception import head, odometry

SIDE = "right"
WIDE = np.array([0.0, 1.0, 0.0, -0.3, 0.0, -0.3, 0.0])  # the Dex3 fingers as far open as their joints go, to let go
PALM = (0.0415, 0.0, 0.0)  # the Dex3 palm link in the wrist-yaw frame (the y offset of 3 mm is left out)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Body:
    """What leg_odo's feeds read of the robot: joints, gyro and gravity readings. `pelvis` is the simulator's and is read
    once, to say where the episode starts."""

    def __init__(self, sim, mo):
        self.sim, self.m, self.odo_root = sim, mo, None

    def pelvis(self):
        q = self.sim.d.qpos
        return q[:3].copy(), R.mat(q[3:7])

    def self_state(self) -> np.ndarray:
        st = self.sim.state()
        grav = self.pelvis()[1].T @ np.array([0.0, 0.0, -1.0])  # the IMU's gravity reading
        return np.concatenate([st["dof_pos"], st["dof_vel"], grav, st["ang_vel"], [0.0]]).astype(np.float32)


def cont(mo, side: str) -> dict:
    """Motor.reach keywords that start a reach where the last one's command ended, in its posture family. A reach that
    starts at the measured palm instead drops the servo's lead in one step, and the hand recoils (it knocked objects)."""
    p, q = mo.last_fut[WRIST[side]]
    return dict(start=p[0] + R.mat(q[0]) @ R.OFF, start_rot=R.mat(q[0]), ik_from_ref=True)


class PlaceBelief:
    """Where the robot takes its pelvis to be, and the bridge between that frame and the simulator's.

    The robot's frame is the world as it was at the start, carried along by leg odometry over each walk; the task's
    goals are worked out in it. The Motor is commanded in the simulator's frame, where a goal means "this far from where
    the pelvis really is". So after each walk the two frames are tied at the pelvis: the place odometry gives (belief)
    and the place the simulator has (sim). A goal goes to the Motor through `to_sim`, which keeps it where it lies from
    the pelvis; the odometry's error then shows as a goal that is off in the world, as on a robot. While standing and
    reaching, leg_odo.JawFeed (load-weighted feet, arm joints) follows the pelvis and the palm from that tie on. Without
    odometry both places are the simulator's and nothing changes."""

    def __init__(self, sim, mo, odo: bool = True, root_odo: bool | None = None):
        """Two independent switches (TODO 49.3):
        `odo` (self_pose=odometry): the robot's belief of its pelvis and palm is leg odometry (Body + JawFeed), the walk's stop
        servo is off (stop_ki 0, stop_timeout = settle_s) and the frames are tied after each walk; False (self_pose=truth): no
        feed, the Motor's stop correction runs on the true pelvis, the belief is the simulator's.
        `root_odo` (controller_root=odometry; default: as `odo`, the old single flag): OdoRoot is built and sets pol.root_source,
        so ScaleBFM is told the odometry pelvis, re-anchored at each stand; False (controller_root=truth): no OdoRoot, ScaleBFM
        is told the simulator's pelvis."""
        root_odo = odo if root_odo is None else root_odo
        self.sim, self.mo, self.odo, self.root_odo, self.anchored = sim, mo, odo, root_odo, False
        if odo or root_odo:
            self.body = Body(sim, mo)
        if odo:
            mo.cfg = dataclasses.replace(mo.cfg, stop_ki=0.0, stop_timeout=mo.cfg.settle_s)
        if root_odo:  # as grasp_room.make_env with BFM_ROOT=odo: ScaleBFM is told the pelvis position from leg odometry
            self.body.odo_root = odometry.OdoRoot(self.body)
        if odo:
            self.feed = odometry.JawFeed(self.body)

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
        return self.mo.reach(side, self.to_sim(p), self.to_sim_R(Rm), **cont(self.mo, side), **kw)

    def walk(self, xy, yaw: float, **kw):
        """Walk the pelvis to (xy, yaw) of the robot's frame; -> the odometry's error at the end (cm, deg)."""
        if self.odo:
            p0, yaw0 = self.pelvis()
            self.mo.jaw_source = None
            self.feed.odo = odometry.LegOdometry(self.sim.m, self.sim.qadr, self.sim.vadr)  # the lower foot stays put
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

    cam = head.HEAD_CAM  # the camera's name in the robot's model

    def frame(self):
        """The head camera's (colour image, depth image in m, vertical field of view in degrees). A benchmark whose camera
        differs from SIMPLE's (LocoManiLab: its own size, a field of view that comes with each image) overrides this and `cam`."""
        rgb, depth = head.head_frame(self.sim)
        return rgb, depth, head.HEAD_CAM_FOVY

    def scene(self):
        """The head camera's colour image, each pixel's point in the robot's frame and which pixels have a depth. The
        camera sits on the pelvis by the joint readings; the pelvis is the one of the robot's own estimate (before the
        first walk: the place and heading that define the robot's frame, with the IMU's tilt). The camera's own place in the
        robot's frame is left in `cam_b`."""
        rgb, depth, fovy = self.frame()
        if not hasattr(self, "kin"):
            self.kin = head.HeadKin(self.sim.m, self.sim.qadr, self.cam)
        rel_p, rel_R = self.kin(self.sim.d.qpos[self.sim.qadr])
        ps, Rs = self._pelvis_sim()
        c = head.cam_points(depth, fovy).reshape(-1, 3)
        self.cam_b = self.from_sim(ps + Rs @ rel_p)
        return (
            rgb,
            self.from_sim(ps + (rel_p + c @ rel_R.T) @ Rs.T).reshape(*depth.shape, 3),
            (depth > 0.2) & (depth < 5),
        )


class Belief(PlaceBelief):
    """PlaceBelief with the wait after a walk a parameter."""

    walk_hold = 0.5
    odo_stop = False  # with odometry: the walk's stop servo steers on the odometry's pelvis (the bed sets stop_ki back)

    def walk(self, xy, yaw: float, **kw):
        if self.odo:
            p0, yaw0 = self.pelvis()
            self.mo.jaw_source = None
            self.feed.odo = odometry.LegOdometry(self.sim.m, self.sim.qadr, self.sim.vadr)
            self.feed.begin()
            if self.odo_stop:  # where odometry puts the pelvis now, in the simulator's frame through the last tie
                self.mo.stop_xy = lambda: self.to_sim(p0 + rot_z(yaw0) @ self.feed.odo.p)[:2]
        self.mo.walk_to(tuple(self.to_sim(np.r_[xy, 0.0])[:2]), yaw + self.yt - self.yb, **kw)
        self.mo.stop_xy = None
        if self.walk_hold > 0:
            self.mo.hold(self.walk_hold)
        self.pt, self.yt = self._truth()
        if not self.odo:
            self.pb, self.yb = self.pt, self.yt
            return 0.0, 0.0
        self.pb, self.yb = p0 + rot_z(yaw0) @ self.feed.odo.p, yaw0 + self.feed.odo.yaw
        self.feed.anchor(self.pt, self.yt, WRIST, R.OFF)
        self.mo.jaw_source, self.anchored = self.feed.jaw, True
        return float(100 * np.linalg.norm((self.pb - self.pt)[:2])), float(np.degrees(R.wrap(self.yb - self.yt)))
