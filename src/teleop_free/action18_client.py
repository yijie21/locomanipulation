"""Deploy side of the 18-D action space: policy output -> robot, planner mode.

The 65-D token action is published straight through `AgentCommander.token_frame`
(protocol v4, encoder bypassed). The 18-D action is not an intent code, it is a
COMMAND in the pelvis frame, so it needs the same conversion the scripted expert
does at collection time and then goes out on the SAME planner interface the
expert used:

    a18 -> right/left wrist SE(3) targets --IK--> 17 upper-body joint targets
        -> grips                                -> set_hands
        -> vx, vy, dyaw, dz                     -> planner frame (mode, movement,
                                                   facing, speed, height)

Consequences that make this NOT a drop-in swap for the token client, both
deliberate:

  * the deploy must run WITH its encoder (`NO_ENCODER` unset) -- the planner
    message's upper body is only read on the encoder path.
  * the process is in planner mode from the first frame to the last and never
    switches. The 1/3-amplitude degradation measured in docs/lockstep.md is a
    property of switching planner -> streamed inside one process, not of
    planner mode itself, but that has been verified for the expert only, so an
    arm run through this client is compared to a token arm knowing the deploy
    interface differs.

Layout of the 18 numbers (tools/export_action18.py writes it, the dataset
carries it as `action.a18`):

    0:3   right wrist position, pelvis frame, m
    3:6   right wrist orientation, rotation vector, pelvis frame
    6     right grip (jaw command, -0.020 closed .. 0.024 open)
    7:13  left wrist, same layout
    13    left grip
    14    vx  (pelvis frame, m/s)      15  vy
    16    dyaw (rad, relative to the current heading)
    17    dz  (pelvis height relative to standing; 0 = stand, <0 = squat)
"""
from __future__ import annotations

import math
import os

import numpy as np

from arm_ik import ArmIK
from g1_joints import ARM_MJ
from primitives import upper_body_from_mj

# Prelude station-keeping bands, read ONCE at import (set them in the
# environment before launching, not mid-run). See Action18Sender.hold.
HOLD_IN = float(os.environ.get("TF_HOLD_IN", "0.02"))      # stop walking below
HOLD_OUT = float(os.environ.get("TF_HOLD_OUT", "0.06"))    # start walking above
HOLD_SPEED = float(os.environ.get("TF_HOLD_SPEED", "0.215"))

SQUAT_DZ = -0.02          # below this the base command becomes IDEL_SQUAT
WALK_V = 0.03             # below this speed the base command becomes IDLE
PELVIS_STAND = 0.793
IK_RES_WARN = 0.06


def rotvec_to_mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-8:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


class Action18Sender:
    """Stateful because IK is: each solve is seeded from the joint vector we
    last commanded, which is what keeps the arm on one branch instead of
    flipping elbow-up/elbow-down between frames."""

    def __init__(self, p, sides=("right", "left")):
        self.p = p
        self.sides = sides
        self.ik = ArmIK()
        self.q = np.asarray(p.obs()["joints"], dtype=float).copy()
        self.res = {s: 0.0 for s in sides}
        self.n_bad = 0
        self._holding = False      # hold()'s hysteresis latch

    def send(self, a18, frame_index: int | None = None) -> None:
        a = np.asarray(a18, dtype=float).ravel()
        if a.size != 18:
            raise ValueError(f"action must be 18-D, got {a.size}")
        s = self.p.obs()
        q = self.q.copy()

        for k, side in (("right", 0), ("left", 7)):
            if k not in self.sides:
                continue
            pos = a[side:side + 3]
            rot = rotvec_to_mat(a[side + 3:side + 6])
            q_sol, res, _ = self.ik.solve(k, pos, rot, q_mj29=q,
                                          use_waist=(k == "right"))
            self.res[k] = float(res)
            if res < IK_RES_WARN:
                q[ARM_MJ[k]] = q_sol[ARM_MJ[k]]
                if k == "right":
                    q[12:15] = q_sol[12:15]        # the waist lean the servo needs
            else:
                self.n_bad += 1
        self.q = q

        # upper body + hands
        self.p.cmd.set_upper_body(upper_body_from_mj(q))
        rh = np.zeros(7, dtype=np.float32); rh[0] = rh[1] = float(a[6])
        lh = np.zeros(7, dtype=np.float32); lh[0] = lh[1] = float(a[13])
        self.p.cmd.set_hands(left=lh, right=rh)

        # base: pelvis-frame velocity and a heading offset -> world planner frame
        from primitives import LocomotionMode
        yaw = float(self.p.pelvis_yaw(s))
        vx, vy, dyaw, dz = (float(a[14]), float(a[15]), float(a[16]), float(a[17]))
        face = yaw + float(np.clip(dyaw, -0.6, 0.6))
        fv = (math.cos(face), math.sin(face), 0.0)
        spd = float(math.hypot(vx, vy))
        if dz < SQUAT_DZ:
            # height is an ABSOLUTE pelvis height and is honoured only in
            # IDEL_SQUAT (docs memory: sonic-planner-depth-channel)
            self.p.cmd.frame(mode=LocomotionMode.IDEL_SQUAT, facing=fv,
                             height=PELVIS_STAND + float(np.clip(dz, -0.20, 0.0)))
        elif spd >= WALK_V:
            c, sn = math.cos(yaw), math.sin(yaw)
            mv = (c * vx - sn * vy, sn * vx + c * vy, 0.0)
            n = math.hypot(mv[0], mv[1]) or 1.0
            self.p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=fv,
                             movement=(mv[0] / n, mv[1] / n, 0.0),
                             speed=float(np.clip(spd, 0.0, 0.5)))
        else:
            self.p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)

    def hold(self, anchor_xy, anchor_yaw, left_hand, right_hand,
             band_in: float = HOLD_IN, band_out: float = HOLD_OUT,
             speed: float = HOLD_SPEED) -> float:
        """Hold the prelude pose. -> current xy error from the anchor (m).

        Heading is held closed against `anchor_yaw`; position is held closed
        against `anchor_xy` with hysteresis -- the loop starts walking only
        past `band_out` and stops inside `band_in`, because a walking gait
        cannot servo a 3 cm error and a tight band just makes it hunt.

        TWO MEASUREMENTS, AND THE SECOND IS THE ONE THAT COUNTS.

        diag_station_keep.py (2026-09-04, 20 s per arm) reported planner IDLE
        holding to 0.7 cm and this loop to 1.2 cm -- but it reset to the SCENE
        DEFAULT pose: yaw 0, x -1.5, far from the table. That is not an eval
        spawn. In the real thing (outputs/eval/a18_smoke5, 2026-09-05, 4
        rounds, the eval's own spawn distribution) the prelude's hold segment
        moved the robot 0.27-0.46 m table-ward in ~1.2 s, EVERY round --
        exactly the "~0.15 m/s through any planner hold" that grasp_gym
        documents for drift-prone spawn configs. Do not quote the 0.7 cm as
        this path's drift: it was measured in a configuration the eval never
        visits.

        Hence tight bands (0.06 engage / 0.02 stop), overridable via
        TF_HOLD_OUT / TF_HOLD_IN / TF_HOLD_SPEED: with ~1.2 s of prelude and
        a drift of that rate, the original 0.20 m dead band spent the entire
        budget before doing anything. Straight back, never turn-and-walk:
        turning is what a prelude must not do.

        WHAT ACTUALLY MATTERS HERE is the heading anchor. `frame()` subtracts
        `heading_offset` from facing AND movement (one shared frame), and in a
        PURE-PLANNER process -- which this client is, start to finish -- psi0
        is neither 0 nor the landing yaw; it measured -106 deg. grasp_gym.py:924
        stores the post-settle yaw and gets away with it only because a
        collection round runs a streamed grasp segment that re-anchors psi0
        every round. This path has no such segment. Calibrate: command a known
        facing with zero compensation, let it settle, and the yaw it settles at
        is psi0.

        HISTORY, because this took three wrong measurements to get right and
        each wrong one was briefly written down as fact: (1) "planner IDLE
        settles 23 cm off" -- that was the teleport landing transient, the
        robot dropping ~10 cm out of model.qpos0 and catching itself in 0.2 s;
        (2) "closing the loop diverges to 207 cm because movement is not
        frame-compensated" -- both halves false, the probe had left
        `heading_offset` unset after its reset so the robot chased a phantom
        heading; (3) "the heading channel cannot hold, 60-90 deg per 20 s" --
        the same probe anchoring itself inside the landing transient. With the
        transient waited out and psi0 calibrated the base holds to 0.7 cm and
        1.5 deg. Any probe that resets must do BOTH before it measures.
        """
        from primitives import LocomotionMode
        s = self.p.obs()
        xy = np.asarray(self.p.pelvis_xy(s), dtype=float)
        yaw = float(self.p.pelvis_yaw(s))
        d = float(anchor_yaw) - yaw
        d = (d + math.pi) % (2 * math.pi) - math.pi
        face = yaw + float(np.clip(d, -0.3, 0.3))
        fv = (math.cos(face), math.sin(face), 0.0)
        self.p.cmd.set_upper_body(upper_body_from_mj(self.q))
        self.p.cmd.set_hands(left=left_hand, right=right_hand)

        e = np.asarray(anchor_xy, dtype=float) - xy
        r = float(np.linalg.norm(e))
        self._holding = r > band_out if not getattr(self, "_holding", False) \
            else r > band_in
        if self._holding:
            self.p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=fv,
                             movement=(e[0] / r, e[1] / r, 0.0), speed=speed)
        else:
            self.p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
        return r

    def stats(self) -> str:
        return (f"ik res right {self.res.get('right', 0):.3f} "
                f"left {self.res.get('left', 0):.3f}, {self.n_bad} rejected")
