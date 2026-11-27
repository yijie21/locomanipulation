"""Self world model, minimal proof (2026-10-29): the standing robot of the grip probe (scripts/rsi/capability_probe.py,
walk_mars + a table edge 0.28 m ahead + one 6 cm box, side grasp) driven at 10 Hz through Motor.hand.

Everything a policy or a world model reads is expressed relative to the robot's own pelvis ("self" features); world
poses are kept only for the judge and for the synthetic camera.

  self state s   joints q (29, ScaleBFM order), joint velocities (29), gravity in the pelvis frame (3), pelvis angular
                 velocity (3, pelvis frame), pelvis height (1)                                               -> 65
  command a      right jaw-centre target in the pelvis frame (3) and its velocity lead (3)                  -> 6
  odometry       pelvis displacement in the previous step's heading frame (3) and yaw change (1)
  kinematics     jaw centre in the pelvis frame (3), head camera position (3) and rotation (6, two columns)
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "rsi")]

OUT = REPO / "outputs" / "selfwm_20261029"
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"

SIDE = "right"
TICKS = 5  # 50 Hz motor ticks per 10 Hz step
STEP_S = 0.1
S_DIM, A_DIM, ODO_DIM, KIN_DIM = 65, 6, 4, 12

# the box of the minimal example: capability_probe.GRIP_BASE (6 cm wide, 15 cm tall, top 0.75 m, 0.10 m inboard)
BOX = dict(top=0.75, width=0.06, depth=0.06, height=0.15, mass=0.3, friction=0.6, inboard=0.10, att="side",
           carry=False, place=False, sweep="width", jitter=[0.0, 0.0, 0.0])  # fmt: skip
PARK = np.array([3.0, 3.0, 0.2])  # where the box waits when an episode does not use it

# jaw-centre regions in the stance frame (= world: every episode resets at the origin facing +x, z above the floor):
# where a pre-grasp starts and where commands roam. The probe's grasp point is (0.39, -0.15, 0.825).
START_LO, START_HI = np.array([0.20, -0.20, 0.80]), np.array([0.30, -0.10, 0.88])
ROAM_LO, ROAM_HI = np.array([0.18, -0.23, 0.79]), np.array([0.42, -0.07, 0.92])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Env:
    """One worker's simulator: the grip-probe scene under ScaleBFM, stepped at 10 Hz."""

    def __init__(self, scene: str = "grip", controller: str | None = None, root: str | None = None,
                 sonic_vr_frame: str | None = None):  # fmt: skip
        """scene "grip": the probe's table and box (the minimal proof); "open": walk_mars with nothing on the floor
        (the generalised world model's data, gen_data2.py). `controller` and `root` (2026-11-24, TODO 38.11): an entry
        that resolved the run switches (locomani.bench) passes them; None reads REACH_CONTROLLER / BFM_ROOT as before.
        `sonic_vr_frame` (2026-11-26, simple_sonic_vrref_20261126): SONIC only, SonicMotorConfig.vr_frame ("robot",
        "reference"); None keeps the motor's default ("robot")."""
        import capability_probe as cap

        self.cap = cap
        # REACH_CONTROLLER (2026-11-16): "scalebfm" (default), "sonic" -- SONIC v1.1 in process (backends.sonic_onnx) --
        # "amo" (backends.amo: AMO's legs and waist, our arm solver; 2026-11-23) or "heft" (backends.heft: LocoManiLab's
        # whole-body tracker; 2026-11-25),
        # same scene, robot, joint order and Motor, so the self-model data of the two compare column for column
        self.controller = controller or os.environ.get("REACH_CONTROLLER", "scalebfm")
        sonic = None
        if self.controller == "sonic":
            # REACH_SONIC_NETS: "gpu" (default) all three ONNX sessions on the worker's GPU; "cpu" encoder and decoder on
            # the CPU; "allcpu" the planner too. REACH_SONIC_THREADS: onnxruntime threads per session.
            from locomani.backends.sonic_onnx.runtime import SonicConfig

            where = os.environ.get("REACH_SONIC_NETS", "gpu")
            assert where in ("gpu", "cpu", "allcpu"), where
            sonic = SonicConfig(device_id=-1 if where == "allcpu" else 0, nets_on_cpu=where == "cpu",
                                threads=int(os.environ.get("REACH_SONIC_THREADS", "1")))  # fmt: skip
        heft = None
        if self.controller == "heft":
            # REACH_HEFT_DELAY (2026-11-25): HeftConfig.ref_delay_s, s the reference is read behind the motor's clock;
            # 0 (default) the motor's reference from now on, 0.16 as LocoManiLab's motion buffer
            from locomani.backends.heft.runtime import HeftConfig

            heft = HeftConfig(ref_delay_s=float(os.environ.get("REACH_HEFT_DELAY", "0")))
        # REACH_SCENE_DIR: the sim's scene directory instead of walk_mars (2026-11-16: NVIDIA's own G1 + Dex3);
        # REACH_SONIC_DYN: SonicSim's joint dynamics, "scene" or "training"
        self.probe = cap.Probe(
            "grip" if scene == "grip" else "reach",
            controller=self.controller,
            sonic=sonic,
            scene_dir=os.environ.get("REACH_SCENE_DIR") or None,
            dynamics=os.environ.get("REACH_SONIC_DYN", "scene"),
            heft=heft,
        )
        if self.controller == "amo":
            # REACH_AMO_HEIGHT_MAX (2026-11-23): the cap on AMO's base-height command, m (AmoConfig.height_max). Default
            # 0.75, AMO's own nominal command: sent the reference pelvis height (0.78-0.79 standing) it walked off in
            # about half of the multi-waypoint reaches; "none" restores that.
            # REACH_AMO_STEP_IN: seconds AMO steps in place after a reset before the stand pose is taken
            # (AmoConfig.step_in_s), default 3; 0 restores the reset as it was.
            hmax = os.environ.get("REACH_AMO_HEIGHT_MAX", "0.75")
            self.probe.m.amo = dataclasses.replace(
                self.probe.m.amo,
                height_max=None if hmax == "none" else float(hmax),
                step_in_s=float(os.environ.get("REACH_AMO_STEP_IN", "3")),
            )
        if self.controller == "sonic":  # REACH_SONIC_IFACE: SonicMotorConfig.interface (vr3pt, g1_splice, g1_full)
            self.probe.m.sonic = dataclasses.replace(
                self.probe.m.sonic, interface=os.environ.get("REACH_SONIC_IFACE", self.probe.m.sonic.interface)
            )
        if sonic_vr_frame is not None:
            assert self.controller == "sonic", f"sonic_vr_frame={sonic_vr_frame} with controller {self.controller}"
            assert sonic_vr_frame in ("robot", "reference"), sonic_vr_frame
            self.probe.m.sonic = dataclasses.replace(self.probe.m.sonic, vr_frame=sonic_vr_frame)
        self.m, self.sim, self.R = self.probe.m, self.probe.sim, self.probe.R
        # REACH_HAND_OFF (2026-11-19): the hand point in the wrist-yaw frame, "x,y,z" in m, instead of the Dex1 pad
        # centre (runtime.OFF); SIMPLE's Dex3 palm is "0.0415,0,0" (simple_eval_20261112/pick.py PALM)
        if os.environ.get("REACH_HAND_OFF"):
            self.R.OFF[:] = [float(v) for v in os.environ["REACH_HAND_OFF"].split(",")]
        root = root or os.environ.get(
            "BFM_ROOT", "true"
        )  # what ScaleBFM is told about the pelvis position; "true": the simulator's, as a perfect tracker
        if root != "true":  # the stop correction and the walk's end test read the true pelvis
            self.m.cfg = dataclasses.replace(self.m.cfg, stop_ki=0.0, stop_timeout=self.m.cfg.settle_s)
        if (
            root == "reference"
        ):  # ScaleBridge's local tracking. Below about 0.45 m/s the policy takes no steps (0.25: 1 % of the walk)
            self.m.pol.cfg = dataclasses.replace(self.m.pol.cfg, reference_forcing=True)
            self.m.cfg = dataclasses.replace(self.m.cfg, walk_speed=float(os.environ.get("BFM_LOCAL_WALK", "0.6")))
        elif root == "odo":  # leg odometry: encoders, gyro and gravity only (leg_odo.OdoRoot)
            import leg_odo

            self.odo_root = leg_odo.OdoRoot(self)
        self.box = self.probe._size(BOX)["body"] if scene == "grip" else None
        import mujoco

        self.cam = mujoco.mj_name2id(self.sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        assert self.cam >= 0, "head_camera missing from the probe scene"

    # -- episode set-up ---------------------------------------------------------------------------------------------
    def box_centre(self, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        cap = self.cap
        return np.array([cap.EDGE_M + BOX["inboard"] + dx, -cap.LATERAL_M + dy, BOX["top"] + BOX["height"] / 2 + 0.001])

    def reset(self, box_xy=None) -> None:
        """Standing at the origin facing +x, 1 s settle; the box at `box_xy` offset from its probe place, or parked."""
        m = self.m
        m.reset((0.0, 0.0), 0.0)
        if self.box is None:
            m.hold(1.0)
            return
        if box_xy is None:
            m.place("probe_object", PARK, np.array([1.0, 0.0, 0.0, 0.0]))
        else:
            m.place("probe_object", self.box_centre(*box_xy), np.array([1.0, 0.0, 0.0, 0.0]))
        m.hold(1.0)

    def pregrasp(self, goal: np.ndarray):
        """The probe's own way to a pre-grasp (world `goal`): whole-body IK reach, through a point outside the table
        edge when the goal is past it, servo on. Returns the Outcome, None once fallen."""
        via = []
        x_via = self.cap.EDGE_M - self.cap.VIA_OUT_M
        if goal[0] > x_via:
            via = [np.array([x_via, goal[1], goal[2]])]
        out = self.m.reach(SIDE, np.asarray(goal, float), self.wrist_rot(), via=via)
        return None if self.m.fell_at is not None else out

    # -- measured state ---------------------------------------------------------------------------------------------
    def pelvis(self):
        q = self.sim.d.qpos
        return q[:3].copy(), quat_mat(q[3:7])

    def wrist_rot(self) -> np.ndarray:
        """The side grasp's wrist attitude: level, along the pelvis heading."""
        return rot_z(yaw_of(self.pelvis()[1]))

    def jaw_world(self) -> np.ndarray:
        return self.m.jaw_centre(SIDE)[0]

    def camera_world(self):
        d = self.sim.d
        return d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()

    def self_state(self) -> np.ndarray:
        st = self.sim.state()
        p, Rp = self.pelvis()
        grav = Rp.T @ np.array([0.0, 0.0, -1.0])
        z = (
            p[2]
            if getattr(self, "odo_root", None) is None or self.odo_root.anchor is None
            else self.odo_root.anchor[2] + self.odo_root.odo.p[2]
        )
        return np.concatenate([st["dof_pos"], st["dof_vel"], grav, st["ang_vel"], [z]]).astype(np.float32)

    def kinematics(self) -> np.ndarray:
        """Jaw centre and head camera in the pelvis frame: what the robot's own forward kinematics gives."""
        p, Rp = self.pelvis()
        cp, Rc = self.camera_world()
        jaw = Rp.T @ (self.jaw_world() - p)
        Rcp = Rp.T @ Rc
        return np.concatenate([jaw, Rp.T @ (cp - p), Rcp[:, 0], Rcp[:, 1]]).astype(np.float32)

    def world(self) -> np.ndarray:
        """pelvis xyz + yaw, jaw centre xyz, camera xyz (world; for the judge and for checks only)."""
        p, Rp = self.pelvis()
        return np.concatenate([p, [yaw_of(Rp)], self.jaw_world(), self.camera_world()[0]]).astype(np.float32)

    # -- acting -----------------------------------------------------------------------------------------------------
    def command(self, target_w: np.ndarray, vel_w: np.ndarray) -> np.ndarray:
        """Drive the jaw centre to world `target_w` led along `vel_w`; returns the command as the robot sees it, in
        the pelvis frame now (the action row a)."""
        p, Rp = self.pelvis()
        self.m.hand(SIDE, np.asarray(target_w, float), self.wrist_rot(), np.asarray(vel_w, float))
        return np.r_[Rp.T @ (target_w - p), Rp.T @ vel_w].astype(np.float32)

    def command_pel(self, target_pel: np.ndarray, vel_pel: np.ndarray) -> None:
        """The same from a pelvis-frame command (what a policy outputs)."""
        p, Rp = self.pelvis()
        self.m.hand(SIDE, p + Rp @ target_pel, self.wrist_rot(), Rp @ vel_pel)

    def commanded_world(self) -> np.ndarray:
        """The jaw centre the motor commands now (reference + servo lead): where a hand program must start."""
        return self.m.commanded_jaw(SIDE).copy()

    def step(self) -> bool:
        """TICKS motor ticks on the current command; False once fallen."""
        self.m.primitive = "selfwm"
        for _ in range(TICKS):
            if not self.m.tick():
                return False
        return True

    # -- both hands (gen_data2.py) ------------------------------------------------------------------------------------
    def hand_world(self, side: str) -> np.ndarray:
        return self.m.jaw_centre(side)[0]

    def command_side(self, side: str, target_w, vel_w, rot_w) -> np.ndarray:
        """Drive `side`'s jaw centre with an explicit wrist rotation; returns the action row as the robot sees it:
        side (+1 right, -1 left), target and lead velocity in the pelvis frame, the rotation's first two columns."""
        p, Rp = self.pelvis()
        self.m.hand(side, np.asarray(target_w, float), np.asarray(rot_w, float), np.asarray(vel_w, float))
        Rr = Rp.T @ np.asarray(rot_w, float)
        sgn = 1.0 if side == "right" else -1.0
        return np.r_[sgn, Rp.T @ (target_w - p), Rp.T @ vel_w, Rr[:, 0], Rr[:, 1]].astype(np.float32)

    def world2(self) -> np.ndarray:
        """pelvis xyz + yaw, both jaw centres, head camera position (world; checks only)."""
        p, Rp = self.pelvis()
        return np.concatenate([p, [yaw_of(Rp)], self.hand_world("right"), self.hand_world("left"),
                               self.camera_world()[0]]).astype(np.float32)  # fmt: skip


def odometry(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Pelvis displacement from world row w0 to w1 in w0's heading frame, and the yaw change."""
    d = rot_z(float(w0[3])).T @ (w1[:3] - w0[:3])
    return np.r_[d, wrap(float(w1[3] - w0[3]))].astype(np.float32)
