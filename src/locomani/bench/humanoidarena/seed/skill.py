"""The seed skill of the HumanoidArena package: a scripted bimanual box carry in the native action space (TODO 49.5b).

The native action is the 40-D reference the TWIST2 tracker follows (`env.hold`): the reference root's step, its height and heading, the
29 canonical joint positions, the two hand values. The tracker turns it into leg and waist motion; the arms follow their joint
targets. What the robot knows of the scene comes through `env.view.goal()` (the truth view); of itself through `env.me` (the
pelvis, the wrists, the arm Jacobians).

pp_box (pick the box off the table, put it on the shelf layer the benchmark's check names):
  1 settle     stand
  2 approach   walk to the table's edge, one thigh's depth short of it, level with the box, facing it
  3 reach      both wrists to the box's two sides, a palm's length behind the box's middle (arms raised by the Jacobian servo)
  4 squeeze    close in on the sides a few centimetres, then hold the joint targets (the tracker keeps the squeeze)
  5 lift       raise both wrists
  6 carry      back off from the table, turn toward the shelf, walk to the layer; the arms' joint targets stay as they are
  7 place      lower to the layer's height, step in, open the arms and hands, back away
Every speed, distance and wait is a key of K. Units: m, m/s, s, rad.
"""

import math

import numpy as np

K = dict(
    settle_steps=30,
    heading_gain=1.0,  # rad/s of reference heading per rad of heading error while walking
    ramp_s=0.6,  # the legs blend between standing and the stride over this long
    slow_s=1.0,  # slow down to arrive: speed = distance / slow_s
    turn_speed=0.6,  # rad/s
    pos_tol=0.06,
    align_tol=0.025,  # level with the box to this (the arms must be symmetric about the pelvis)
    yaw_tol=0.06,
    walk_limit_s=9.0,
    edge_gap=0.17,  # the pelvis this far short of the table's edge
    palm=0.09,  # the wrist is this far behind the palm's middle (m, along the forearm)
    side_gap=0.012,  # the wrists' distance from the box's sides when they arrive (m)
    reach_z=0.0,  # the wrists' height above the box's middle
    squeeze=0.03,  # then close in this far on each side
    squeeze_steps=45,
    arm_step=0.006,  # rad per control step, the servo's largest joint change
    servo_damp=0.02,
    servo_tol=0.012,
    reach_limit=300,
    lift=0.05,
    lift_steps=60,
    back_off=0.55,  # m, backward from the stand
    place_y_margin=0.07,  # the box's centre this far inside the layer's front edge
    lower_steps=90,
    open_gap=0.08,  # each wrist moves this far outward to let go
    release_steps=50,
    hold_end=1.0,
    close_hands=True,  # the hands close when the wrists squeeze
)


# ---- small geometry


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Ctl:
    """The skill's running state: the 29 joint targets, the reference heading (relative to the start), the hands, the stride's phase."""

    def __init__(self, env):
        self.env = env
        self.q = env.me.joints().copy()
        self.stand = self.q.copy()
        self.yaw0 = env.me.pelvis()[1]
        self.yaw = 0.0  # the reference heading relative to the start heading
        self.hands = (0.0, 0.0)
        self.height = env.start_height
        self.phase = 0.0
        self.gain = 0.0  # 0 standing .. 1 walking: how far the legs follow the stride

    def step(self, dxy_world=(0.0, 0.0), walking=False, K=None):
        """One control step. While `walking` the legs follow the stride (blended in over ramp_s), else they return to the standing pose."""
        g = self.env.gait
        rate = 0.02 / (K["ramp_s"] if K else 0.6)
        self.gain = min(1.0, self.gain + rate) if walking else max(0.0, self.gain - rate)
        if self.gain > 0.0:
            self.phase = (self.phase + 0.02 / g.period) % 1.0
            legs = self.env.leg_ids
            self.q[legs] = (1 - self.gain) * self.stand[legs] + self.gain * g.pose(self.phase)[legs]
        else:
            self.q[self.env.leg_ids] = self.stand[self.env.leg_ids]
        d = rot2(-(self.yaw0 + self.yaw)) @ np.asarray(
            dxy_world, float
        )  # the action's step is in the reference heading frame
        self.env.drive(self.env.hold(joints=self.q, dxy=d, yaw=self.yaw, height=self.height, hands=self.hands))

    def stand_still(self, steps, K=None):
        for _ in range(steps):
            self.step(walking=False, K=K)

    def turn_to(self, yaw_world, K):
        """Turn the reference heading to yaw_world while stepping in place (the stride runs, the root does not move)."""
        for _ in range(int(K["walk_limit_s"] / 0.02)):
            cur = self.yaw0 + self.yaw
            err = wrap(yaw_world - cur)
            if abs(err) < K["yaw_tol"] and abs(wrap(yaw_world - self.env.me.pelvis()[1])) < 2 * K["yaw_tol"]:
                return True
            self.yaw += float(np.clip(err, -K["turn_speed"] * 0.02, K["turn_speed"] * 0.02))
            self.step(walking=True, K=K)
        return False

    def walk_to(self, goal_xy, K, tol=None):
        """Move the reference toward goal_xy (world) at the stride's speed, steered by the pelvis's own position; stop within pos_tol."""
        goal = np.asarray(goal_xy, float)
        tol = K["pos_tol"] if tol is None else tol
        hold = self.yaw0 + self.yaw  # the heading to keep: the legs' stride veers, the reference heading steers it back
        for _ in range(int(K["walk_limit_s"] / 0.02)):
            self.yaw += float(np.clip(K["heading_gain"] * wrap(hold - self.env.me.pelvis()[1]) * 0.02, -0.01, 0.01))
            p = self.env.me.pelvis()[0][:2]
            e = goal - p
            dist = float(np.linalg.norm(e))
            if dist < tol:
                break
            v = min(self.env.gait.speed, 1.0 * dist / K["slow_s"])
            self.step(e / dist * v * 0.02, walking=True, K=K)
        n = int(1.2 / 0.02)
        for _ in range(n):
            self.step(walking=False, K=K)
        return float(np.linalg.norm(goal - self.env.me.pelvis()[0][:2])) < 2 * tol

    def servo(self, targets, steps, tol, arm_step, damp, only=None):
        """Jacobian servo of the wrists to `targets` {side: xyz}: one joint step per control step. -> the largest remaining error (m)."""
        env = self.env
        err = 0.0
        for _ in range(steps):
            err = 0.0
            for side, tgt in targets.items():
                J, h = env.me.jac(side)
                e = np.asarray(tgt, float) - h
                err = max(err, float(np.linalg.norm(e)))
                dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), np.clip(e, -0.05, 0.05))
                dq = np.clip(dq, -arm_step, arm_step)
                ids = env.me.arm_ids(side)
                for k, i in enumerate(ids):
                    self.q[i] += dq[k]
            if err < tol:
                break
            self.step(walking=False)
        return err

    def wrists(self):
        return {s: self.env.me.wrist(s) for s in ("left", "right")}


def run(env):
    task = env.task
    if task != "pp_box":
        raise NotImplementedError(f"the seed skill has a flow for pp_box only, not {task}")
    return run_pp_box(env)


def note(env, name):
    """Mark a moment with the robot's own readings (the pelvis, the wrists) for the record."""
    p, yaw = env.me.pelvis()
    w = {k: [round(float(x), 3) for x in v] for k, v in (("L", env.me.wrist("left")), ("R", env.me.wrist("right")))}
    env.mark(name, dict(t=round(env.t, 2), pelvis=[round(float(x), 3) for x in p], yaw=round(float(yaw), 3), wrists=w))


def approach(env, c, g):
    """2: walk to the table's edge, level with the box, facing +x."""
    Kp = env.K
    table = g["table"]
    box = g["box_center"]
    x_stand = (table[0][0] - Kp["edge_gap"]) if table is not None else box[0] - 0.45
    c.turn_to(c.yaw0, Kp)
    c.walk_to((x_stand, box[1]), Kp, tol=Kp["align_tol"])
    c.stand_still(15)
    note(env, "at_table")


def reach(env, c, g):
    """3: both wrists beside the box, a palm's length behind its middle. -> the remaining error (m)."""
    Kp = env.K
    p = env.me.pelvis()[0]
    box = env.view.goal()["box_center"]
    xw, zw = box[0] - Kp["palm"], box[2] + Kp["reach_z"]
    left = np.array([xw, box[1] + g["half"] + Kp["side_gap"], zw])
    right = np.array([xw, box[1] - g["half"] - Kp["side_gap"], zw])
    # raise first with the elbows in, so the forearms clear the table's edge
    mid = {"left": np.array([p[0] + 0.30, box[1] + 0.20, zw]), "right": np.array([p[0] + 0.30, box[1] - 0.20, zw])}
    c.servo(mid, Kp["reach_limit"], Kp["servo_tol"], Kp["arm_step"], Kp["servo_damp"])
    note(env, "mid")
    c.stand_still(10)
    err = c.servo({"left": left, "right": right}, Kp["reach_limit"], Kp["servo_tol"], Kp["arm_step"], Kp["servo_damp"])
    note(env, "reached")
    return err, left, right


def grasp_and_lift(env, c, left, right):
    """4, 5: close in on the sides, hold the joint targets (the tracker keeps the squeeze), raise both wrists."""
    Kp = env.K
    sq = {"left": left - np.array([0, Kp["squeeze"], 0]), "right": right + np.array([0, Kp["squeeze"], 0])}
    c.servo(sq, Kp["squeeze_steps"], 0.0, 0.01, Kp["servo_damp"])
    c.hands = (1.0, 1.0) if Kp["close_hands"] else (0.0, 0.0)
    c.stand_still(15)
    note(env, "squeezed")
    w = c.wrists()
    up = {s: w[s] + np.array([0, 0, Kp["lift"]]) for s in w}
    c.servo(up, Kp["lift_steps"], 0.01, 0.01, Kp["servo_damp"])
    note(env, "lifted")


def carry(env, c):
    """6: the joint targets stay; the reference walks back, turns toward the shelf and walks to the layer."""
    Kp = env.K
    p = env.me.pelvis()[0]
    c.walk_to((p[0] - Kp["back_off"], p[1]), Kp)
    c.turn_to(c.yaw0 + math.pi / 2, Kp)
    ahead = float(
        np.linalg.norm(env.view.goal()["box_center"][:2] - env.me.pelvis()[0][:2])
    )  # how far ahead of the pelvis the box rides
    goal = env.view.goal()
    rect = goal["place_rect"]
    bx = float(np.clip(goal["place_xy"][0], rect[0] + 0.15, rect[1] - 0.15))
    by = rect[2] + goal["half"] + Kp["place_y_margin"]
    c.walk_to((bx, by - ahead), Kp)
    c.stand_still(15)
    note(env, "at_shelf")
    return goal


def place(env, c, goal):
    """7: lower to the layer, let go, back away."""
    Kp = env.K
    dz = (goal["place_bottom_z"] + goal["half"] + 0.01) - env.view.goal()["box_center"][2]
    w = c.wrists()
    c.servo({s: w[s] + np.array([0, 0, dz]) for s in w}, Kp["lower_steps"], 0.01, 0.01, Kp["servo_damp"])
    note(env, "lowered")
    c.hands = (0.0, 0.0)
    w = c.wrists()
    lat = rot2(env.me.pelvis()[1]) @ np.array([0.0, Kp["open_gap"]])  # the pelvis's left is its +y
    out = {"left": w["left"] + np.array([lat[0], lat[1], 0.0]), "right": w["right"] - np.array([lat[0], lat[1], 0.0])}
    c.servo(out, Kp["release_steps"], 0.01, 0.02, Kp["servo_damp"])
    note(env, "released")
    p = env.me.pelvis()[0][:2]
    c.walk_to((p[0], p[1] - 0.3), Kp)
    c.stand_still(int(Kp["hold_end"] / 0.02))


def run_pp_box(env):
    g = env.view.goal()
    c = Ctl(env)
    c.stand_still(env.K["settle_steps"])
    note(env, "settled")
    approach(env, c, g)
    err, left, right = reach(env, c, g)
    grasp_and_lift(env, c, left, right)
    place(env, c, carry(env, c))
    return dict(reach_err=float(err))
