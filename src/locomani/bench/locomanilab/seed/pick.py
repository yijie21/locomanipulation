"""The pick's small helpers with the Dex3 hand (moved from seed_pkg_lml/pick.py, TODO 49.4): `cont` (the Motor.reach keywords that
continue a reach), the fingers' joint groups and the pose helper `T`. The old file also carried the simulator-reading `touching`
and the privileged baseline's CLI (the SIMPLE pick script); the loop path never used the CLI, and `touching` is the bed's now.
`Body` (what the odometry feeds read of the robot) is fixed infrastructure: locomani.bench.simple.body.
"""

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import WRIST
from locomani.bench.simple.body import PALM, Body  # noqa: F401  (re-exported: the old pick.Body, pick.PALM)

FINGERS = {"thumb": [0, 1, 2], "index": [3, 4], "middle": [5, 6]}  # a finger's joints in runtime.DEX3_JOINTS


def T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def cont(mo, side: str) -> dict:
    """Motor.reach keywords that start a reach where the last one's command ended, in its posture family. A reach that
    starts at the measured palm instead drops the servo's lead in one step, and the hand recoils (it knocked objects)."""
    p, q = mo.last_fut[WRIST[side]]
    return dict(start=p[0] + R.mat(q[0]) @ R.OFF, start_rot=R.mat(q[0]), ik_from_ref=True)
