"""The pick's motions with the Dex3 hand: reach behind the pregrasp, the pregrasp, the grasp, close three fingers by feel,
lift. Moved from simple_eval_20261112/pick.py (TODO 49.2) into the editable seed package: `touching`, `cont`, the two ways to
close the hand and `pick` stay here (the skill takes `cont`, `touching` and `FINGERS` from this module). `Body` (what the
odometry feeds read of the robot) is fixed infrastructure now: locomani.bench.simple.body. Left behind: the privileged
baseline's CLI, SIMPLE's stored grasp and GraspGen-X's library lookup (they read EP.OUT and torch; the loop path never
called them).
"""

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import WRIST
from locomani.bench.simple.body import PALM, Body  # noqa: F401  (Body: re-exported, the old pick.Body)

SIDE_OFF = -0.22  # m, the palm's target to the left of the pelvis at the stance (SIMPLE's grasp sits 0.22 to the right)
AHEAD = 0.7  # a hand pose the right arm takes: the cosine of the fingers with straight ahead of the robot,
ROLL = (-30.0, 110.0)  # and, in deg, the palm facing between left (the back of the hand up) and down
CLEAR = 0.06  # m, the palm link above what the object stands on: the middle finger hangs 4 cm below the palm link
PRE = 0.08  # m, GraspGen-X's pregrasp: back from the grasp against the palm's normal (the palm link's y)
BACK = 0.06  # m, the reach starts this far behind the pregrasp, on the line from the grasp through the pregrasp
FINGERS = {"thumb": [0, 1, 2], "index": [3, 4], "middle": [5, 6]}  # a finger's joints in runtime.DEX3_JOINTS
PRESS = 0.3  # rad, how far past the angle at which it touched a finger's joints are driven


def T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def touching(sim, body: int) -> list:
    """The hand links in contact with the body."""
    m, d, out = sim.m, sim.d, set()
    for i in range(d.ncon):
        a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        other = b if a == body else a if b == body else None
        if other is not None and ("_hand_" in m.body(other).name or "wrist" in m.body(other).name):
            out.add(m.body(other).name.split("_hand_")[-1])
    return sorted(out)


def cont(mo, side: str) -> dict:
    """Motor.reach keywords that start a reach where the last one's command ended, in its posture family. A reach that
    starts at the measured palm instead drops the servo's lead in one step, and the hand recoils (it knocked objects)."""
    p, q = mo.last_fut[WRIST[side]]
    return dict(start=p[0] + R.mat(q[0]) @ R.OFF, start_rot=R.mat(q[0]), ik_from_ref=True)


def close_on(sim, side: str, body: int, f_close, f_pre, seconds: float = 1.0) -> None:
    """Close the hand finger by finger: each finger stops PRESS past where it first touches the body (the Dex3's
    fingertip pressure sensors give that touch), so fingers that keep curling do not turn the object in the hand."""
    m, d = sim.m, sim.d
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    cmd, free = np.array(f_pre, float), dict(FINGERS)
    for u in np.linspace(0, 1, int(seconds / 0.02)):
        touched = set()
        for i in range(d.ncon):
            a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            other = b if a == body else a if b == body else None
            if other is not None:
                touched |= {f for f in free if f"{side}_hand_{f}_" in m.body(other).name}
        for f in touched:
            j = free.pop(f)
            cmd[j] = d.qpos[q[j]] + PRESS * np.sign(f_close[j] - f_pre[j])
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        sim.fingers_cmd[side] = cmd.copy()
        yield


def close_by_feel(sim, side: str, f_close, f_pre, seconds: float = 1.0, lag: float = 0.4, slow: float = 0.3, log=None):
    """Close the hand with the finger joints' readings alone: a finger whose joints stop following the closing command
    (they moved less than `slow` of what a free finger moves in 0.1 s; a free joint follows with a time constant of
    `lag`) is pressing the object against the other fingers, and every finger stops PRESS past where it is."""
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    f_close, f_pre = np.asarray(f_close, float), np.asarray(f_pre, float)
    cmd, free, hist = f_pre.copy(), dict(FINGERS), []
    for k, u in enumerate(np.linspace(0, 1, int(seconds / 0.02))):
        now = sim.d.qpos[q].copy()  # the finger encoders
        hist.append(now)
        if k >= 10:
            moved = np.abs(now - hist[k - 5])
            want = np.abs(f_close - f_pre) / seconds * 0.1 * (1 - np.exp(-(k - 2.5) * 0.02 / lag))
            for f, j in list(free.items()):
                jj = [i for i in j if want[i] > 2e-3]
                if log is not None:
                    log.append((k, f, float(moved[jj].sum() / max(1e-9, want[jj].sum())) if jj else None))
                if jj and moved[jj].sum() < slow * want[jj].sum():
                    for j in free.values():  # the object is held between this finger and the others: all stop
                        cmd[j] = now[j] + PRESS * np.sign(f_close[j] - f_pre[j])
                    free.clear()
                    break
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        sim.fingers_cmd[side] = cmd.copy()
        yield


def pick(
    mo, sim, side: str, Tg, Tp, f_close, f_pre, squeeze: float = 0.6, close_s: float = 0.5, body=None, feel=None
) -> dict:
    """Reach to the pregrasp, then the grasp, close the fingers, lift 15 cm. The fingers are driven past the closed
    posture by `squeeze`, or with `body` given each one stops where it touches that body (close_on)."""
    step = Tp[:3, 3] - Tg[:3, 3]
    sim.fingers_cmd[side] = f_pre
    mo.reach(side, Tp[:3, 3] + BACK * step / max(1e-6, np.linalg.norm(step)), Tp[:3, :3])
    mo.reach(side, Tp[:3, 3], Tp[:3, :3], speed=0.1, **cont(mo, side))
    mo.reach(side, Tg[:3, 3], Tg[:3, :3], speed=0.08, **cont(mo, side))
    p, Rm = mo.jaw_centre(side)  # the true palm, for the record
    err = 100 * np.linalg.norm(p - Tg[:3, 3])
    ang = np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3].T @ Rm) - 1) / 2, -1, 1)))
    if feel is not None:
        for _ in close_by_feel(sim, side, f_close, f_pre, log=feel):
            mo.hold(0.02)
    elif body is not None:
        for _ in close_on(sim, side, body, f_close, f_pre):
            mo.hold(0.02)
    for u in np.linspace(0, 1, int(close_s / 0.02)) if body is None and feel is None else ():
        sim.fingers_cmd[side] = f_pre + u * (f_close + squeeze * (f_close - f_pre) - f_pre)
        mo.hold(0.02)
    mo.hold(0.5)
    p, Rm = (getattr(mo, "jaw_source", None) or mo.jaw_centre)(side)
    mo.reach(side, p + [0, 0, 0.15], Rm, speed=0.1, **cont(mo, side))
    mo.hold(1.0)
    return dict(palm_cm=float(err), palm_deg=float(ang))
