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

REACH_VIA_COMP_MODE=const moves every path by the training paths' mean offset instead (one round): the constant
correction, an opponent in the narrow-gap test.
REACH_VIA_COMP_MODE=goal is the default mode with the via points left where they are: only the goal is moved (same
rounds, same cap). An opponent that tells what the correction buys along the way from what it buys at the end.

REACH_VIA_COMP_MODE=pose also turns the wrist commands. The palm and the fingers are one rigid body during a reach (15
spheres). At each via point's moment and at the settled end, the rotation that takes the wanted hand onto the predicted hand
(least squares over the 15 spheres) is undone on that point's wrist command, and the position is moved as before -- but a
share (gain, 0.5) of the remaining correction a round, over 6 rounds, and at most 40 deg of turn a point. The first version
took the whole correction each round (4 rounds, up to 60 deg): on 8 trial replays its commands ended where the model had
seen nothing like them, its own error rose from 2.5 to 5.5 cm and the settled end got worse.
The model has to know the commands this sends. One trained on random paths alone does not, on a planner's routes, once a
wrist command is turned 20 deg or more (its error on the run 2.8 -> 6.9 cm, 41 development routes, 2026-11-22); one that
also saw compensated runs on other planner routes does (7.2 -> 2.5 cm on routes held out). So this mode is for a model
trained that way.
REACH_VIA_COMP_MODE=arm is pose with one change in how the turned command is planned (Motor.reach's body_rot): the body is
planned for the ORIGINAL wrist attitudes at the moved points, and the reaching arm's seven joints alone turn the wrist. In
mode pose the whole-body IK paid for the turns partly with the torso: the plan's shoulder went 1 cm forward where the
position-only plan takes it 2 cm back (which cancels the controller's own 2 cm lead), and the robot's shoulder ended 7.5 cm
off where it was wanted against 3.8 cm (41 development routes, 2026-11-23). info: arm_only, body_via_R, body_rot.
REACH_VIA_COMP_MODE=body is arm with one more thing moved: the points the BODY is planned for (Motor.reach's body_rot with
points). The controller itself leans further forward when the hand is turned (its shoulder 3.2 cm ahead of its plan
against 1.9 cm), which no plan of the hand undoes; the plan's shoulder follows the body's points about 1 : 1, so these
are moved against the shoulder's predicted offset, as the hand's points are against the palm's. info: bshift (17, 3),
body_points, body_via_p, body_goal.
An optional trust rule (off by default: REACH_VIA_COMP_TRUST=inf, REACH_VIA_COMP_TRUST_ABS=inf): a point's new correction
is kept only while the ensemble's disagreement about the hand at that point stays within `trust` times what it was for the
original command and below `trust_abs` (m); a refused point keeps its last accepted correction and its gain is halved. At
1.5 and 0.012 it refused corrections on 91 % of 200 random paths and cost the palm more than it saved; the ensemble's
disagreement (1.0 cm) also understates the error where the model is wrong (3 cm).
info then also has turn (17, 3), each point's wrist correction as a rotation vector in the heading frame.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_gen as VG  # noqa: E402
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402
from arm_fk import ArmFK  # noqa: E402

Q, ITERS = VM.Q, 2
CAP = float(os.environ.get("REACH_VIA_COMP_CAP", "0.08"))  # m: the most a point (or the body's point) is moved
# REACH_VIA_COMP_MODE=pose: rounds, the share of the remaining correction taken a round, the most a wrist command is turned
# (deg), and the trust rule's two numbers (see the docstring). Overridable for development runs.
POSE = dict(iters=int(os.environ.get("REACH_VIA_COMP_ITERS", "6")), gain=float(os.environ.get("REACH_VIA_COMP_GAIN", "0.5")),
            cap_turn=float(os.environ.get("REACH_VIA_COMP_CAP_TURN", "40")), trust=float(os.environ.get("REACH_VIA_COMP_TRUST", "inf")),
            trust_abs=float(os.environ.get("REACH_VIA_COMP_TRUST_ABS", "inf")),
            goal_turn=os.environ.get("REACH_VIA_COMP_GOAL_TURN", "1") == "1")  # fmt: skip


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


def kabsch(A, B) -> np.ndarray:
    """The rotation R that takes the points A (n, 3) onto B about their centroids, least squares: R (A - mean) ~ B - mean."""
    U, _, Vt = np.linalg.svd((A - A.mean(0)).T @ (B - B.mean(0)))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def at(P, u: float) -> np.ndarray:
    """A path sampled at Q equal steps of its duration, at the fraction u of it."""
    x = float(np.clip(u, 0, 1)) * (len(P) - 1)
    j = min(int(x), len(P) - 2)
    return P[j] + (x - j) * (P[j + 1] - P[j])


class Comp:
    def __init__(self, model: Path, scene: Path):
        import torch

        full = torch.load(Path(model) / "via_point.pt", map_location="cpu", weights_only=False)
        self.M = full["point"]
        # REACH_VIA_COMP_MODE=const (the narrow-gap test's opponent): every path is moved by the same mean offset, the
        # training paths' per hand -- a correction that does not look at the path; the model is still run, for the record
        self.mode = os.environ.get("REACH_VIA_COMP_MODE", "point")
        self.const = dict(right=np.asarray(full["const_right"], float), left=np.asarray(full["const_left"], float))
        self.fk = ArmFK(scene)
        link = self.fk.links
        self.key = np.array([link.index(x) for x in VM.KEY_LINKS])
        self.palm = link.index("hand_palm_link")
        self.hand2 = np.array([self.palm + 1, link.index("hand_index_1_link")])
        self.hand = np.array([i for i, x in enumerate(link) if x.startswith("hand_")])
        self.shoulder = np.array([i for i, x in enumerate(link) if "shoulder" in x])
        self.nets: dict = {}

    def members(self, T: dict) -> np.ndarray:
        """Each member's predicted offsets for the one row of T: (members, Q + 1, 24, 3). Their mean is via_point.predict's."""
        import torch

        P = self.M
        X = torch.as_tensor((VP.features(T, np.array([0])) - P["mu_x"]) / P["sd_x"])
        out = []
        for mem in P["members"]:
            f = self.nets.setdefault(id(mem), VP.net(P["din"], P["width"], P["depth"], P["dropout"]))
            f.load_state_dict(mem["state"])
            f.eval()
            with torch.no_grad():
                out.append((P["mu_y"] + P["sd_y"] * f(X).numpy()).reshape(Q + 1, 24, 3))
        return np.array(out)

    def sweep(self, m, st: dict, side: str, goal_w, rot_w, pelvis_z, via_p, via_R, speed, body=None):
        """The reference sweep (Q, 24, 3), the predicted offsets (Q + 1, 24, 3) and each via point's and the goal's moment
        (fractions of the plan's duration) for one candidate command, and the ensemble's disagreement about the hand at
        the Q + 1 points; heading frame. body: Motor.reach's body_rot (the attitudes the body is planned for)."""
        kw: dict = dict(via=list(via_p), via_rot=list(via_R)) if len(via_p) or body is not None else {}
        if body is not None:
            kw["body_rot"] = body
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
        off = VP.predict(self.M, T, np.array([0]))[
            0
        ]  # the ensemble's mean, exactly as the offline evaluation computes it
        spread = np.zeros(Q + 1)
        if self.mode in ("pose", "arm"):
            ys = self.members(T)
            spread = np.linalg.norm(ys - ys.mean(0), axis=-1).mean(0)[:, self.hand].mean(1)  # (Q + 1,), m
        # Motor.reach: a leg lasts max(reach_min_s, its length / speed)
        pts = np.vstack([st["jaw_w"], *via_p, goal_w]) if len(via_p) else np.vstack([st["jaw_w"], goal_w])
        leg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.cumsum(np.maximum(m.cfg.reach_min_s, leg / (speed or m.cfg.reach_speed)))
        return ref, off, t / t[-1], spread

    def compensate(self, m, rec, side: str, goal_w, rot_w, pelvis_z, via_p, via_R, speed, p, Rh, fingers):
        sd = 1.0 if side == "right" else -1.0
        r6 = np.asarray(rec.J[-1], float)
        R0 = Rh.T @ np.c_[r6[:3], r6[3:6], np.cross(r6[:3], r6[3:6])]
        head = lambda X: np.c_[(np.asarray(X)[:, :2] - p[:2]) @ Rh[:2, :2], np.asarray(X)[:, 2]]  # noqa: E731
        jaw_rec = np.asarray(rec.W[-1][4:7] if sd > 0 else rec.W[-1][7:10], float)
        st = dict(S3=np.r_[rec.S[-1], rec.S[-2], rec.S[-3]], jaw=head(jaw_rec[None])[0], R0=R0, Rh=Rh, head=head, sd=sd,
                  fingers=fingers, jaw_w=m.jaw_centre(side)[0])  # fmt: skip
        via0, goal0 = np.asarray(via_p, float).reshape(-1, 3), np.asarray(goal_w, float)
        viaR0, rot0 = [np.asarray(r, float) for r in via_R], np.asarray(rot_w, float)
        pose = self.mode in ("pose", "arm", "body")
        gain = np.full(len(via0) + 1, POSE["gain"] if pose else 1.0)
        cap_turn = np.radians(POSE["cap_turn"])

        def command(shift, turn):
            sw = shift @ Rh.T  # heading frame -> world (rows)
            return (via0 + sw[:-1], goal0 + sw[-1], [Rh @ turn[k] @ Rh.T @ viaR0[k] for k in range(len(viaR0))],
                    Rh @ turn[-1] @ Rh.T @ rot0)  # fmt: skip

        def body_of(bshift):
            """Motor.reach's body_rot. arm: the body is planned for the original attitudes at the hand's moved points and
            the arm alone turns the wrist; body: the body has its own points too, the original ones moved by bshift."""
            if self.mode == "arm":
                return (viaR0, rot0)
            if self.mode == "body":
                bw = bshift @ Rh.T
                return (viaR0, rot0, list(via0 + bw[:-1]), goal0 + bw[-1])
            return None

        def run(shift, turn, bshift, plain=False):  # plain: the original command exactly as a direct run plans it
            via, goal, viaR, rot = command(shift, turn)
            body = None if plain else body_of(bshift)
            ref, off, u, sp = self.sweep(m, st, side, goal, rot, pelvis_z, via, viaR, speed, body)
            return ref, off, u, np.array([at(sp[:Q], x) for x in u[:-1]] + [sp[Q]])

        shift = np.zeros((len(via0) + 1, 3))  # heading frame
        bshift = np.zeros((len(via0) + 1, 3))  # mode body: how far the body's own points are moved, heading frame
        turn = [np.eye(3) for _ in range(len(via0) + 1)]  # each point's wrist correction, heading frame
        ref0, off0, u0, sp0 = run(shift, turn, bshift, plain=True)
        sh0 = ref0[:, self.shoulder].mean(1)  # the wanted shoulder (its spheres' centre) along the sweep
        want_sh = [at(sh0, x) for x in u0[:-1]] + [sh0[-1]]
        want = [at(ref0[:, self.palm], u) for u in u0[:-1]] + [ref0[-1, self.palm]]  # the wanted palm at each point
        want_hand = [at(ref0[:, self.hand], u) for u in u0[:-1]] + [
            ref0[-1, self.hand]
        ]  # and the wanted hand's 15 spheres
        allowed = np.full(len(sp0), np.inf)
        if np.isfinite(POSE["trust"]) or np.isfinite(POSE["trust_abs"]):
            allowed = np.maximum(np.minimum(POSE["trust"] * np.maximum(sp0, 1e-9), POSE["trust_abs"]), sp0)
        ref, off, u = ref0, off0, u0
        refused = 0
        for _ in range(dict(const=1).get(self.mode, POSE["iters"] if pose else ITERS)):
            pred = np.concatenate([ref, ref[-1:]]) + (self.const[side] if self.mode == "const" else off)
            now = [at(pred[:Q, self.palm], x) for x in u[:-1]] + [pred[Q, self.palm]]
            new_shift = shift - gain[:, None] * (np.array(now) - np.array(want))
            n = np.linalg.norm(new_shift, axis=1, keepdims=True)
            new_shift *= np.minimum(1.0, CAP / np.maximum(n, 1e-9))
            if self.mode == "goal":  # the via points stay; only the goal (last row) is moved
                new_shift[:-1] = 0.0
            new_turn = list(turn)
            if pose:
                now_hand = [at(pred[:Q, self.hand], x) for x in u[:-1]] + [pred[Q, self.hand]]
                for k in range(len(turn) if POSE["goal_turn"] else len(turn) - 1):
                    step = VG.expm(gain[k] * VG.logm(kabsch(want_hand[k], now_hand[k]).T))
                    v = VG.logm(step @ turn[k])
                    new_turn[k] = VG.expm(v * min(1.0, cap_turn / max(float(np.linalg.norm(v)), 1e-9)))
            new_bshift = bshift
            if self.mode == "body":  # the body's points against the shoulder's predicted offset
                sh = pred[:, self.shoulder].mean(1)
                now_sh = [at(sh[:Q], x) for x in u[:-1]] + [sh[Q]]
                new_bshift = bshift - gain[:, None] * (np.array(now_sh) - np.array(want_sh))
                nb = np.linalg.norm(new_bshift, axis=1, keepdims=True)
                new_bshift = new_bshift * np.minimum(1.0, CAP / np.maximum(nb, 1e-9))
            ref_n, off_n, u_n, sp_n = run(new_shift, new_turn, new_bshift)
            ok = sp_n <= allowed if pose else np.ones(len(gain), bool)
            if ok.all():
                shift, turn, bshift, ref, off, u = new_shift, new_turn, new_bshift, ref_n, off_n, u_n
                continue
            refused += int((~ok).sum())
            gain[~ok] *= 0.5
            shift = np.where(ok[:, None], new_shift, shift)
            bshift = np.where(ok[:, None], new_bshift, bshift)
            turn = [new_turn[k] if ok[k] else turn[k] for k in range(len(turn))]
            ref, off, u, _ = run(shift, turn, bshift)
        via, goal, viaR, rot = command(shift, turn)
        out = np.zeros((VM.VIA_MAX + 1, 3), np.float32)
        out[: len(via0)], out[-1] = shift[:-1], shift[-1]
        tv = np.zeros((VM.VIA_MAX + 1, 3), np.float32)
        if len(via0):
            tv[: len(via0)] = [VG.logm(r) for r in turn[:-1]]
        tv[-1] = VG.logm(turn[-1])
        info = dict(ref0=ref0, off0=off0.astype(np.float32), ref=ref, off=off.astype(np.float32), shift=out, turn=tv,
                    goal0=head(goal0[None])[0].astype(np.float32), refused=refused)  # fmt: skip
        # what gen_reach.py hands to Motor.reach as body_rot; float64, so the run plans exactly what was planned here
        bR = np.zeros((VM.VIA_MAX, 3, 3))
        if len(viaR0):
            bR[: len(viaR0)] = viaR0
        bs = np.zeros((VM.VIA_MAX + 1, 3), np.float32)
        bs[: len(via0)], bs[-1] = bshift[:-1], bshift[-1]
        bP, bw = np.zeros((VM.VIA_MAX, 3)), bshift @ Rh.T
        if len(via0):
            bP[: len(via0)] = via0 + bw[:-1]
        info.update(arm_only=int(self.mode in ("arm", "body")), body_via_R=bR, body_rot=rot0.copy(), bshift=bs,
                    body_points=int(self.mode == "body"), body_via_p=bP, body_goal=goal0 + bw[-1])  # fmt: skip
        return via, np.array(viaR).reshape(-1, 3, 3), goal, rot, info
