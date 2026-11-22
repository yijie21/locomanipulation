"""selfmodel_via_20261120, third step: use the prediction -- shift a path's via points and goal against the offset the
point model predicts, before the command is sent.

gen_reach.py calls it for a replayed path when REACH_VIA_COMP=<model dir> is set (with REACH_VIA_REPLAY):

    comp = Comp(<model dir with via_point.pt>, <scene dir>)
    via_p, goal, info = comp.compensate(m, rec, side, goal_w, rot_w, pelvis_z, via_p, via_R, speed, p, Rh)

What is wanted is the sweep the ORIGINAL command's reference plan makes: where a planner believes the arm will go. For a
candidate command the motor plans without moving (the plan is caught at Motor._blend), the plan's 24 spheres are computed by
forward kinematics at via_model.py's Q moments, and via_point.py's ensemble predicts the real sweep from the state recorded
at the command and that reference -- the same features, built by the same functions, as in training. At each via point's
moment in the plan and at the settled end, the predicted palm is compared with the wanted palm and that via point (or the
goal) is moved by the difference, the other way. ITERS rounds, each point moved at most CAP in all. Wrist attitudes are not
touched. Only the hand's points can be commanded: the elbow and the shoulder are the controller's whole-body IK.

info (heading frame of the command, m): ref0 (Q, 24, 3) the wanted sweep, off0 (Q + 1, 24, 3) the offsets predicted for
the original command, ref / off the same for the command sent, shift (17, 3) what each via point (16 padded) and the goal
(last row) were moved by, goal0 the original goal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402
from arm_fk import ArmFK  # noqa: E402

Q, ITERS, CAP = VM.Q, 2, 0.08


class _Stop(Exception):
    pass


def plan(m, side: str, target_w, rot_w, pelvis_z, kw: dict):
    """Motor.reach's whole-body plan (poses, durations) for this command from the current state, without moving."""
    blend, got = m._blend, {}

    def stop(poses, durations):
        got["plan"] = (poses, durations)
        raise _Stop

    m._blend = stop
    try:
        m.reach(side, target_w, rot_w, pelvis_z=pelvis_z, servo=False, **kw)
    except _Stop:
        pass
    finally:
        m._blend = blend
    return got["plan"]


def at(P, u: float) -> np.ndarray:
    """A path sampled at Q equal steps of its duration, at the fraction u of it."""
    x = float(np.clip(u, 0, 1)) * (len(P) - 1)
    j = min(int(x), len(P) - 2)
    return P[j] + (x - j) * (P[j + 1] - P[j])


class Comp:
    def __init__(self, model: Path, scene: Path):
        import torch

        self.M = torch.load(Path(model) / "via_point.pt", map_location="cpu", weights_only=False)["point"]
        self.fk = ArmFK(scene)
        link = self.fk.links
        self.key = np.array([link.index(x) for x in VM.KEY_LINKS])
        self.palm = link.index("hand_palm_link")
        self.hand2 = np.array([self.palm + 1, link.index("hand_index_1_link")])

    def sweep(self, m, st: dict, side: str, goal_w, rot_w, pelvis_z, via_p, via_R, speed):
        """The reference sweep (Q, 24, 3), the predicted offsets (Q + 1, 24, 3) and each via point's and the goal's moment
        (fractions of the plan's duration) for one candidate command; heading frame."""
        kw: dict = dict(via=list(via_p), via_rot=list(via_R)) if len(via_p) else {}
        if speed is not None:
            kw["speed"] = speed
        poses, durations = plan(m, side, goal_w, rot_w, pelvis_z, kw)
        RT = np.r_[0.0, np.cumsum(durations)]
        XR = np.array(
            [st["head"](self.fk.spheres(side, q.pos, q.quat, q.joints, "mj", st["fingers"])[0]) for q in poses]
        )
        ref = VM.interp(RT, XR, np.linspace(0.0, 1.0, Q) * RT[-1]).astype(np.float32)
        # the features of via_model.build's x that via_point.py reads (G_COLS), in its layout
        x = np.zeros(499, np.float32)
        rot_h = st["Rh"].T @ np.asarray(rot_w, float)
        x[:213] = np.r_[st["S3"], st["jaw"], st["R0"][:, 0], st["R0"][:, 1], st["head"](np.asarray(goal_w)[None])[0],
                        rot_h[:, 0], rot_h[:, 1]]  # fmt: skip
        pz = -1.0 if pelvis_z is None else float(pelvis_z)
        x[373:379] = len(via_p) / VM.VIA_MAX, speed or m.cfg.reach_speed, float(pz > 0), max(pz, 0.0), st["sd"], RT[-1]
        T = dict(x=x[None], ref=ref[None], key=self.key, hand2=self.hand2)
        off = VP.predict(self.M, T, np.array([0]))[0]
        # Motor.reach: a leg lasts max(reach_min_s, its length / speed)
        pts = np.vstack([st["jaw_w"], *via_p, goal_w]) if len(via_p) else np.vstack([st["jaw_w"], goal_w])
        leg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.cumsum(np.maximum(m.cfg.reach_min_s, leg / (speed or m.cfg.reach_speed)))
        return ref, off, t / t[-1]

    def compensate(self, m, rec, side: str, goal_w, rot_w, pelvis_z, via_p, via_R, speed, p, Rh, fingers):
        sd = 1.0 if side == "right" else -1.0
        r6 = np.asarray(rec.J[-1], float)
        R0 = Rh.T @ np.c_[r6[:3], r6[3:6], np.cross(r6[:3], r6[3:6])]
        head = lambda X: np.c_[(np.asarray(X)[:, :2] - p[:2]) @ Rh[:2, :2], np.asarray(X)[:, 2]]  # noqa: E731
        jaw_rec = np.asarray(rec.W[-1][4:7] if sd > 0 else rec.W[-1][7:10], float)
        st = dict(S3=np.r_[rec.S[-1], rec.S[-2], rec.S[-3]], jaw=head(jaw_rec[None])[0], R0=R0, Rh=Rh, head=head, sd=sd,
                  fingers=fingers, jaw_w=m.jaw_centre(side)[0])  # fmt: skip
        via0, goal0 = np.asarray(via_p, float).reshape(-1, 3), np.asarray(goal_w, float)
        ref0, off0, u0 = self.sweep(m, st, side, goal0, rot_w, pelvis_z, via0, via_R, speed)
        want = [at(ref0[:, self.palm], u) for u in u0[:-1]] + [ref0[-1, self.palm]]  # the wanted palm at each point
        shift = np.zeros((len(via0) + 1, 3))  # heading frame
        ref, off, u = ref0, off0, u0
        for _ in range(ITERS):
            pred = np.concatenate([ref, ref[-1:]]) + off
            now = [at(pred[:Q, self.palm], x) for x in u[:-1]] + [pred[Q, self.palm]]
            shift -= np.array(now) - np.array(want)
            n = np.linalg.norm(shift, axis=1, keepdims=True)
            shift *= np.minimum(1.0, CAP / np.maximum(n, 1e-9))
            sw = shift @ Rh.T  # heading frame -> world (rows)
            via, goal = via0 + sw[:-1], goal0 + sw[-1]
            ref, off, u = self.sweep(m, st, side, goal, rot_w, pelvis_z, via, via_R, speed)
        out = np.zeros((VM.VIA_MAX + 1, 3), np.float32)
        out[: len(via0)], out[-1] = shift[:-1], shift[-1]
        return via, goal, dict(ref0=ref0, off0=off0.astype(np.float32), ref=ref, off=off.astype(np.float32), shift=out,
                               goal0=head(goal0[None])[0].astype(np.float32))  # fmt: skip
