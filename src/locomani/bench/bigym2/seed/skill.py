"""The seed skill of the BiGym 2.0 package: a scripted reach and a scripted drawer pull in the native action space (TODO 49.5).

BiGym's action is the lower body's command (vx, vy, pelvis height, wz) plus absolute arm joint targets and two gripper
commands, 50 Hz. What the robot knows of the scene comes through `env.view.goal()` (the truth view); of itself through `env.me`
(joints, the pinch centres, the pelvis, an arm solution for a hand target).

Two flows, one per task family:

reach   the sphere's centre is half a metre ahead of the settled pelvis, a few centimetres out of the left arm's reach: walk
        forward until the arm solution reaches it, stop, move the hand there and hold (the check wants the pinch centre in the ball
        for one second).
drawer  the handle is a horizontal bar on the drawer's front, the drawer slides out toward the robot. Walk to the bar, turn the
        hand so the jaws open vertically (above and below the bar), put the pinch centre on the bar, close, walk backwards pulling the
        drawer out, let go.

Every speed, distance and wait is a key of K. Units: m, m/s, s, rad, rad/s.
"""

import math

import numpy as np

K = dict(
    # settling and walking
    start_hold=0.6,  # standing before anything
    walk_speed=0.20,  # forward speed to the stand-off
    stand_tol=0.012,  # the arm solution must be this close to the target to stop walking
    stop_hold=0.8,  # standing after a walk before the arm moves
    heading_gain=1.2,  # rad/s per rad of heading error while walking
    walk_limit_s=8.0,
    shoulder_off=0.10,  # the shoulder is this far out of the pelvis's midline (drawer: sidestep until it is level with the bar)
    lat_gain=2.0,  # m/s per metre of sideways error
    lat_speed=0.15,
    lat_tol=0.015,
    # the arm
    arm_speed=1.4,  # rad/s, largest joint change rate
    bar_speed=0.35,  # rad/s, the last approach to the bar (a fast one overshoots into the drawer face)
    settle_steps=12,  # the hand must stay within reach_tol this many steps before the gripper closes
    servo_gain=0.3,  # the share of the measured hand error added to the aim every five steps
    reach_tol=0.008,  # a reach leg is done when the pinch centre is this close to its target
    leg_limit_s=5.0,
    hold_s=1.6,  # holding the hand in the sphere
    # the drawer
    pre=0.10,  # the hand's pregrasp, in front of the bar
    bar_dx=0.040,  # the grasp point is this far toward the robot of the handle box centre: the pads (6.6 cm, centred on the pinch centre) reach 3 cm past it, and the fingertips must stop short of the drawer face
    bar_dy=0.0,  # shift along the bar
    bar_dz=0.0,  # shift up
    close_s=1.0,
    pull_speed=0.15,  # m/s backwards
    pull_frac=0.96,  # stop pulling at this much open
    pull_limit_s=6.0,
    grip=1.0,
    stand_x=0.38,  # the bar this far ahead of the pelvis (along the heading) when the walk ends
    dist_tol=0.015,
    height=0.68,  # drawer: the crouch that brings the shoulder down to the bar's level (the arm cannot reach down 0.3 m standing; below about 0.66 the legs no longer walk, so the pull would not move)
    crouch_s=1.5,
)


# ---- small geometry


def quat_of(R) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def facing(forward, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The pinch centre's frame with the fingers along `forward` and the jaws opening along `up` (columns x, y, z)."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    return np.stack([f, u, np.cross(f, u)], axis=1)


def yaw_of(env) -> float:
    return env.me.pelvis()[1]


# ---- moves


class Arms:
    """Holds the arms' joint targets across steps (an arm not given is held where this skill last put it)."""

    def __init__(self, env):
        self.env = env
        self.q = {s: env.me.arm(s) for s in ("left", "right")}
        self.grip = np.zeros(2)
        self.bias: dict = {}  # the aim correction per arm (hand_to)
        self.height = None  # the pelvis height command (m); None leaves the controller's default

    def step(self, vx=0.0, vy=0.0, wz=0.0) -> None:
        self.env.drive(
            vx=vx, vy=vy, wz=wz, height=self.height, left=self.q["left"], right=self.q["right"], grip=self.grip
        )

    def wait(self, seconds: float, vx=0.0, vy=0.0, wz=0.0) -> None:
        for _ in range(int(round(seconds / 0.02))):
            self.step(vx, vy, wz)

    def toward(self, side: str, q_goal, speed: float) -> float:
        """One step of the joint targets toward `q_goal`, at most `speed` rad/s on the largest joint. -> the largest remaining change."""
        d = np.asarray(q_goal, float) - self.q[side]
        m = float(np.max(np.abs(d)))
        lim = speed * 0.02
        self.q[side] = self.q[side] + (d if m <= lim else d * (lim / m))
        return m

    def hand_to(self, side: str, point, quat, K, speed=None, tol=None, limit=None, settle=1) -> float:
        """Move the pinch centre to `point` (facing `quat` or free): the joint targets move toward an arm solution at `speed`, which is
        re-solved every five steps from the arm's own readings and aimed past the point by the tracking lag measured so far (gravity on a
        PD joint). Ends when the hand has been within `tol` for `settle` steps in a row, or after `limit` s. -> the final error (m)."""
        speed = K["arm_speed"] if speed is None else speed
        tol = K["reach_tol"] if tol is None else tol
        limit = K["leg_limit_s"] if limit is None else limit
        err, inside = 9.0, 0
        point = np.asarray(point, float)
        bias = self.bias.setdefault(side, np.zeros(3))
        q_goal, _ = self.env.me.ik(side, point + bias, quat, self.q[side])
        for i in range(int(limit / 0.02)):
            if i % 5 == 0:
                if (
                    float(np.max(np.abs(q_goal - self.q[side]))) < 0.02
                ):  # only once the joint targets have arrived: lag while moving is no bias
                    bias = self.bias[side] = np.clip(
                        bias + K["servo_gain"] * (point - self.env.me.hand(side)), -0.06, 0.06
                    )
                q_goal, _ = self.env.me.ik(side, point + bias, quat, self.q[side])
            left = self.toward(side, q_goal, speed)
            self.step()
            err = float(np.linalg.norm(self.env.me.hand(side) - point))
            inside = inside + 1 if (err < tol and left < 0.02) else 0
            if inside >= settle:
                break
        return err

    def hold_hand(self, side: str, point, quat, K, seconds: float) -> float:
        """Keep servoing the hand to `point` for `seconds` (the grippers may move meanwhile). -> the final error (m)."""
        return self.hand_to(side, point, quat, K, limit=seconds, tol=-1.0)


def walk_until_reachable(env, arms: Arms, side: str, point, ahead, K) -> dict:
    """Walk forward (steering at the heading `ahead`, a world vector) until the arm solution reaches `point`; stop and stand."""
    info = dict(walked_s=0.0, err=None)
    for _ in range(int(K["walk_limit_s"] / 0.02)):
        _, err = env.me.ik(side, point, None, arms.q[side])
        info["err"] = err
        if err < K["stand_tol"]:
            break
        dyaw = (math.atan2(ahead[1], ahead[0]) - yaw_of(env) + math.pi) % (2 * math.pi) - math.pi
        arms.step(vx=K["walk_speed"], wz=float(np.clip(K["heading_gain"] * dyaw, -0.5, 0.5)))
        info["walked_s"] += 0.02
    arms.wait(K["stop_hold"])
    return info


def walk_to_stand(env, arms: Arms, side: str, point, forward, K) -> dict:
    """Walk (standing) until `point` is `stand_x` ahead of the pelvis along `forward` and the shoulder of `side` is level with it."""
    info = dict(walked_s=0.0)
    sign = 1.0 if side == "left" else -1.0
    fwd = np.asarray(forward, float)
    for _ in range(int(K["walk_limit_s"] / 0.02)):
        pel, yaw = env.me.pelvis()
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        d = float(np.dot(np.asarray(point, float) - pel, fwd)) - K["stand_x"]
        e_lat = float(np.dot(np.asarray(point, float) - (pel + sign * K["shoulder_off"] * left), left))
        if d < K["dist_tol"] and abs(e_lat) < K["lat_tol"]:
            break
        dyaw = (math.atan2(fwd[1], fwd[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        vx = float(np.clip(1.5 * d, 0.0, K["walk_speed"]))
        arms.step(
            vx=vx,
            vy=float(np.clip(K["lat_gain"] * e_lat, -K["lat_speed"], K["lat_speed"])),
            wz=float(np.clip(K["heading_gain"] * dyaw, -0.5, 0.5)),
        )
        info["walked_s"] += 0.02
    arms.wait(K["stop_hold"])
    return info


# ---- the flows


def reach(env, K) -> dict:
    g = env.view.goal()
    side, point = g["side"], g["point"]
    arms = Arms(env)
    arms.wait(K["start_hold"])
    env.mark("start")
    ahead = point - env.me.pelvis()[0]
    info = walk_until_reachable(env, arms, side, point, ahead, K)
    env.mark("stand")
    err = arms.hand_to(side, point, None, K, speed=K["arm_speed"])
    env.mark("reached")
    arms.wait(K["hold_s"])
    info.update(hand_err=err)
    return info


def drawer(env, K) -> dict:
    g = env.view.goal()
    side = "left"
    pull = np.asarray(g["pull"], float)
    pull[2] = 0.0
    pull /= np.linalg.norm(pull)
    forward = -pull
    bar = (
        np.asarray(g["handle"], float)
        + K["bar_dx"] * pull
        + K["bar_dy"] * np.array([-pull[1], pull[0], 0.0])
        + K["bar_dz"] * np.array([0, 0, 1.0])
    )
    quat = quat_of(facing(forward))
    arms = Arms(env)
    arms.wait(K["start_hold"])
    env.mark("start")
    pre = bar - K["pre"] * forward
    info = walk_to_stand(env, arms, side, bar, forward, K)
    arms.height = K["height"]
    arms.wait(K["crouch_s"])
    env.mark("stand")
    e1 = arms.hand_to(side, pre, quat, K)
    env.mark("pregrasp")
    e2 = arms.hand_to(side, bar, quat, K, speed=K["bar_speed"], limit=8.0, settle=K["settle_steps"])
    env.mark("at_bar")
    arms.grip[0 if side == "left" else 1] = K["grip"]
    arms.hold_hand(side, bar, quat, K, K["close_s"])
    env.mark("closed")
    pulled = 0.0
    for _ in range(int(K["pull_limit_s"] / 0.02)):
        frac = env.view.goal()["open_frac"]
        pulled = frac
        if frac >= K["pull_frac"]:
            break
        arms.step(vx=-K["pull_speed"])
    env.mark("pulled")
    arms.grip[:] = 0.0
    arms.wait(1.2)
    info.update(pre_err=e1, bar_err=e2, open_frac=pulled)
    return info


def run(env) -> dict:
    K_ = dict(env.K)
    if env.task.startswith("reach_target"):
        return reach(env, K_)
    if env.task.startswith("drawer_top"):
        return drawer(env, K_)
    raise ValueError(f"the seed package has no flow for {env.task}")
