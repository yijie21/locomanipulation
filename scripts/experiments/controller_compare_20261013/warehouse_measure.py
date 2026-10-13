"""Warehouse L1 measurements (TODO 25.3, plan §4.2 and §15): W1-W8 from the round files of warehouse_states.py, the same
arithmetic for both controllers. Contacts are MuJoCo's own on the recorded full scene state (`qpos`) of the mars model.

* W1 stance, read where the reach starts from (SONIC: `handoff`, after its expert's settle; a ScaleBFM walk round:
  its last tick, 3 s after the reference stopped): low gap = the edge line minus the frontmost robot point reaching below the platform top (the crate
  guard's own measure, FootWatch.frontmost_low) and its minimum over the walk (< 0 = crossed); stand-off = the pelvis
  before the line along the walk-in axis; lateral = the pelvis left of the box centre; yaw error to square-on
  (warehouse_states.square_yaw); still = seconds from `brake` to the still tick (B2's rule: 0.5 s of pelvis
  < 0.03 m/s and legs < 0.2 rad/s).
* W2 bend over [reach start, end]: peak torso pitch, lowest pelvis, peak waist pitch, and the waist-pitch static demand
  -- gravity on the robot (qfrc_bias at zero velocity) plus the box's weight about the joint axis while it is off the
  platform -- p95 and max. Torso pitch and pelvis height at `close` too.
* W3 hands over [close, end] (both wrists commanded): hand-separation error (|L-R| measured minus commanded) mean and
  max, common drift (the mean of the two wrists' errors) mean and max; over [descend, close] the same common drift
  (the stall's source). The clamp is each jaw pinching a wall (the force is the jaw law's, the same for both
  controllers), so from 0.5 s after `close`: the share of ticks with both fingers of both hands on the box (any one
  hand; both palms) stands in for the plan's wall depth -> clamp force.
* W4 lift: lifted = the box 5 cm above its start for 2 s (level: and tilted <= 15 deg at the end, the expert's own
  max_tilt_deg -- one jaw alone also lifts the box, hanging); at the end its rise, the commanded rise minus the box's
  (the wrists' mean commanded z from `lift` to the end), box tilt, torso pitch, and the pelvis's drop from `lift`.
* W5 / W6 carry (`carry`: a carry plan's or ScaleBFM's carry, from 0.5 s before the walk starts -- 0.5 s on end of
  > 0.1 m/s or > 0.2 rad/s): the rim's centre (box top, where the jaws hold; the box's own origin is its bottom, 0.33 m
  below, which turns a swing into decimetres) relative to the wrists' midpoint in the pelvis-yaw frame, its sag
  (sliding down the jaws, unaffected by the arms raising the box) per metre walked, the box's peak tilt -- over the
  first 0.3 m of a heading-kept carry (W5) and the whole carry; net travel and turn, and held at the end (box still
  5 cm above its start and both hands on it -- the expert's own `touching`).
* W7 put down (put-down rounds with a `release`): at the end, box bottom over the platform top, tilt, travel from its start,
  on the platform footprint, a hand still on the box; peak box tilt from 3 s before the release to the end.
* W8 contacts over the round, counted as onsets (a pair class touching after >= 0.1 s apart): arm links other than the
  hands on the box, any robot geom on the platform, on the room (static geoms other than floor and platform), and on
  the floor other than the feet.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/warehouse_measure.py \\
        outputs/controller_compare_20261013/warehouse/sonic/states [more state dirs ...] --out outputs/controller_compare_20261013/warehouse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from bottle_measure import DT, LEGS, WAIST, fmt, mat, window, wrap, yaw_of  # noqa: E402
from warehouse_states import BOX, SCENE, Fk, square_yaw  # noqa: E402

from locomani.primitives.foot_watch import FootWatch  # noqa: E402

G = 9.81
MAX_TILT_DEG = 15.0  # the crate expert's own acceptance (visual.max_tilt_deg)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def tilt_deg(q) -> float:
    """The box's lean: its z axis from vertical (its yaw, e.g. a 180 deg carry turn, is not a tilt)."""
    return float(np.degrees(np.arccos(np.clip(mat(q)[2, 2], -1.0, 1.0))))


class Scene:
    """The mars warehouse model and its geom classes for the contact counts."""

    def __init__(self):
        self.fk = Fk(SCENE)
        m = self.m = self.fk.m
        self.d = mujoco.MjData(m)
        self.watch = FootWatch(model=m)
        self.box_body = m.body(BOX).id
        self.waist = m.joint("waist_pitch_joint").id
        self.waist_dof = int(m.jnt_dofadr[self.waist])
        root = lambda b: b if m.body_parentid[b] == 0 else root(int(m.body_parentid[b]))
        under = lambda b, top: b == top or (b != 0 and under(int(m.body_parentid[b]), top))
        pelvis = m.body("pelvis").id
        wrists = [m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
        feet = [m.body(n).id for n in FEET]
        platform = m.body("logistics_platform").id
        floor = m.geom("floor").id
        self.cls = np.full(m.ngeom, "", dtype=object)
        self.hand = np.full(m.ngeom, -1)
        self.part = np.full(m.ngeom, "", dtype=object)  # hand geoms: finger_1 / finger_2 / palm (the rest of the hand)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if root(b) == pelvis:
                k = next((i for i, w in enumerate(wrists) if under(b, w)), -1)
                self.hand[g] = k
                name = m.body(b).name
                self.part[g] = "finger_1" if "finger_1" in name else "finger_2" if "finger_2" in name else "palm"
                self.cls[g] = "hand" if k >= 0 else ("foot" if any(under(b, f) for f in feet) else "robot")
            elif b == self.box_body:
                self.cls[g] = "box"
            elif g == floor:
                self.cls[g] = "floor"
            elif under(b, platform):
                self.cls[g] = "platform"
            elif b != 0 or g != floor:
                self.cls[g] = "room"

    def contacts(self, q):
        """(pair classes touching, per hand the set of its parts on the box: finger_1 / finger_2 / palm)."""
        m, d = self.m, self.d
        d.qpos[:] = q
        mujoco.mj_fwdPosition(m, d)
        pairs, touch = set(), (set(), set())
        for c in d.contact[: d.ncon]:
            a, b = self.cls[c.geom1], self.cls[c.geom2]
            robot = {"hand", "robot", "foot"}
            if a not in robot:
                a, b = b, a
                g = c.geom2
            else:
                g = c.geom1
            if a not in robot:
                continue
            if b == "box":
                if a == "hand":
                    touch[self.hand[g]].add(self.part[g])
                else:
                    pairs.add("arm_box")
            elif b == "platform":
                pairs.add("platform")
            elif b == "room":
                pairs.add("room")
            elif b == "floor" and a != "foot":
                pairs.add("floor")
        return pairs, touch

    def waist_demand(self, q, box_mass, box_up):
        """|static torque| the waist pitch must hold: gravity on the robot plus the box's weight while it is up."""
        m, d = self.m, self.d
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        tau = float(d.qfrc_bias[self.waist_dof])
        if box_up:
            r = d.xipos[self.box_body] - d.xanchor[self.waist]
            tau += float(d.xaxis[self.waist] @ np.cross(r, np.array([0.0, 0.0, -box_mass * G])))
        return abs(tau)


class Round:
    def __init__(self, npz: Path, sc: Scene):
        self.z = {k: v for k, v in np.load(npz).items()}
        self.f = json.loads(npz.with_suffix(".json").read_text())
        self.label = npz.stem
        self.t = self.z["t"]
        self.ev = self.f["events"]
        self.sc = sc
        self._body = {}
        self._contacts = None

    def body(self, name):
        if name not in self._body:
            out = np.empty((len(self.t), 7))
            for i in range(len(self.t)):
                self.sc.fk.put(self.z["pelvis"][i], self.z["joints"][i])
                out[i] = self.sc.fk.pose(name)
            self._body[name] = out
        return self._body[name]

    def contacts(self):
        if self._contacts is None:
            self._contacts = [self.sc.contacts(q) for q in self.z["qpos"]]
        return self._contacts

    def state(self, i):
        p = self.z["pelvis"][i]
        return dict(pelvis=dict(pos=p[:3], quat=p[3:]), joints=self.z["joints"][i])

    def torso_pitch(self, idx):
        torso = self.body("torso_link")
        return np.array([np.degrees(np.arcsin(-mat(torso[i, 3:])[2, 0])) for i in idx])

    # -- W1 -------------------------------------------------------------------
    def still_tick(self):
        if "brake" not in self.ev:
            return None
        v = np.linalg.norm(np.diff(self.z["pelvis"][:, :2], axis=0), axis=1) / DT
        leg = np.max(np.abs(np.diff(self.z["joints"][:, LEGS], axis=0)), axis=1) / DT
        quiet = (v < 0.03) & (leg < 0.2)
        n = int(0.5 / DT)
        start = int(np.searchsorted(self.t, self.ev["brake"]))
        for i in range(start, len(quiet) - n):
            if quiet[i : i + n].all():
                return i
        return None

    def w1(self):
        i = self.still_tick()
        if i is None:
            return {}
        axis, line = np.asarray(self.f["edge"]["axis"], float), float(self.f["edge"]["line"])
        across = np.array([-axis[1], axis[0]])
        # the stance the reach starts from: SONIC's `handoff` (its expert's settle done), a walk round's last tick
        k = int(np.searchsorted(self.t, self.ev["handoff"])) if "handoff" in self.ev else len(self.t) - 1
        top = self.f["platform_top"]
        gap = lambda j: line - self.sc.watch.frontmost_low(self.state(j), np.zeros(2), axis, top)
        walk = window(self.t, self.ev.get("walk", self.t[0]), self.t[k])
        p, b = self.z["pelvis"][k], self.z["box"][k]
        square = square_yaw(self.f["heading"], yaw_of(b[3:]))
        return dict(
            still_s=float(self.t[i] - self.ev["brake"]),
            low_gap_cm=100 * gap(k),
            low_gap_min_cm=100 * min(gap(j) for j in walk[:: max(1, len(walk) // 200)]),
            standoff_cm=100 * float(line - axis @ p[:2]),
            lateral_cm=100 * float(across @ (p[:2] - b[:2])),
            yaw_err_deg=float(np.degrees(wrap(yaw_of(p[3:]) - square))),
        )

    # -- W2 -------------------------------------------------------------------
    def w2(self):
        start = self.ev.get("reach", self.ev.get("close"))
        if start is None or "close" not in self.ev:
            return {}
        if self.f["controller"] == "scalebfm":
            start = max(start, 0.0)
        w = window(self.t, start, self.t[-1])
        q, p = self.z["joints"], self.z["pelvis"]
        b = self.z["box"]
        mass = self.f.get("box_mass") or 1.2
        top = self.f["platform_top"]
        dem = np.array([self.sc.waist_demand(self.z["qpos"][i], mass, b[i, 2] - b[0, 2] > 0.01) for i in w[::2]])
        c = int(np.searchsorted(self.t, self.ev["close"]))
        return dict(
            torso_pitch_max_deg=float(self.torso_pitch(w).max()),
            torso_pitch_close_deg=float(self.torso_pitch([c])[0]),
            pelvis_z_min_cm=100 * float(p[w, 2].min()),
            pelvis_z_close_cm=100 * float(p[c, 2]),
            waist_pitch_max_deg=float(np.degrees(q[w, WAIST["pitch"]].max())),
            waist_demand_p95_nm=float(np.percentile(dem, 95)),
            waist_demand_max_nm=float(dem.max()),
            platform_top=top,
        )

    # -- W3 -------------------------------------------------------------------
    def w3(self):
        if "close" not in self.ev:
            return {}
        cmd = self.z["cmd_wrist"][:, :, :3]
        meas = np.stack([self.body("left_wrist_yaw_link")[:, :3], self.body("right_wrist_yaw_link")[:, :3]], 1)
        ok = np.all(np.isfinite(cmd[:, :, 0]), 1)
        out = {}
        h = window(self.t, self.ev["close"], self.t[-1])
        h = h[ok[h]]
        if len(h) > 10:
            sep = np.abs(
                np.linalg.norm(meas[h, 0] - meas[h, 1], axis=1) - np.linalg.norm(cmd[h, 0] - cmd[h, 1], axis=1)
            )
            common = np.linalg.norm((meas[h] - cmd[h]).mean(1), axis=1)
            hold = h[self.t[h] >= self.ev["close"] + 0.5]
            touch = [self.contacts()[i][1] for i in hold]
            pinch = np.array([[{"finger_1", "finger_2"} <= x[k] for k in range(2)] for x in touch])
            palm = np.array([["palm" in x[k] for k in range(2)] for x in touch])
            out.update(
                sep_err_cm=100 * float(sep.mean()),
                sep_err_max_cm=100 * float(sep.max()),
                common_drift_cm=100 * float(common.mean()),
                common_drift_max_cm=100 * float(common.max()),
                pinch_both=float(np.mean(np.all(pinch, 1))) if len(hold) else None,
                pinch_any=float(np.mean(np.any(pinch, 1))) if len(hold) else None,
                palm_on_box=float(np.mean(np.all(palm, 1))) if len(hold) else None,
            )
        if "descend" in self.ev:
            dsc = window(self.t, self.ev["descend"], self.ev["close"])
            dsc = dsc[ok[dsc]]
            if len(dsc) > 5:
                out["descend_drift_max_cm"] = 100 * float(np.linalg.norm((meas[dsc] - cmd[dsc]).mean(1), axis=1).max())
        return out

    # -- W4 -------------------------------------------------------------------
    def w4(self):
        if "close" not in self.ev:
            return {}
        b = self.z["box"]
        rise = b[:, 2] - b[0, 2]
        run = best = 0
        for u in rise > 0.05:
            run = run + 1 if u else 0
            best = max(best, run)
        tilt = lambda i: tilt_deg(b[i, 3:])
        if "carry" in self.ev or "release" in self.ev:
            return {}  # a carry / put-down round: W6 / W7 read it
        lifted = bool(best * DT >= 2.0)
        out = dict(lifted=lifted, lifted_level=lifted and tilt(-1) <= MAX_TILT_DEG, rise_end_cm=100 * float(rise[-1]))
        if "lift" not in self.ev:
            return out
        j = int(np.searchsorted(self.t, self.ev["lift"]))
        cmd = self.z["cmd_wrist"][:, :, 2].mean(1)
        p = self.z["pelvis"]
        out.update(
            box_tilt_end_deg=tilt(-1),
            torso_pitch_end_deg=float(self.torso_pitch([len(self.t) - 1])[0]),
            pelvis_drop_cm=100 * float(p[j, 2] - p[-1, 2]),
        )
        if np.isfinite(cmd[j]) and np.isfinite(cmd[-1]):
            out["lift_shortfall_cm"] = 100 * float((cmd[-1] - cmd[j]) - (b[-1, 2] - b[j, 2]))
        return out

    # -- W5 / W6 ---------------------------------------------------------------
    def carry(self):
        if "carry" not in self.ev:
            return {}
        attempted = self.f.get("steps_attempted")
        if attempted is not None and not {"walk_to", "carry_to"} & set(attempted):
            return {}
        p = self.z["pelvis"]
        i0 = int(np.searchsorted(self.t, self.ev["carry"]))
        # the walk's onset: 0.5 s on end of walking (> 0.1 m/s) or turning (> 0.2 rad/s) -- the stand-up after a lift
        # shifts the pelvis by centimetres, which a displacement threshold would take for the carry
        speed = np.linalg.norm(np.diff(p[i0:, :2], axis=0), axis=1) / DT
        yaw = np.unwrap([yaw_of(x[3:]) for x in p[i0:]])
        going = (speed > 0.1) | (np.abs(np.diff(yaw)) / DT > 0.2)
        n = int(0.5 / DT)
        onset = next((i for i in range(len(going) - n) if going[i : i + n].all()), None)
        if onset is None:
            return {}
        h = window(self.t, float(self.t[i0 + onset]) - 0.5, self.t[-1])
        lw, rw = self.body("left_wrist_yaw_link"), self.body("right_wrist_yaw_link")
        b = self.z["box"]
        height = 2 * self.f["box_half"][2]
        rel, sag, tilt = [], [], []
        for i in h:
            mid = (lw[i, :3] + rw[i, :3]) / 2
            top = b[i, :3] + mat(b[i, 3:]) @ np.array([0.0, 0.0, height])  # the rim's centre, where the jaws hold
            y = yaw_of(p[i, 3:])
            rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            rel.append(rz.T @ (top - mid))
            sag.append(mid[2] - top[2])
            tilt.append(tilt_deg(b[i, 3:]))
        rel, sag = np.array(rel), np.array(sag) - sag[0]
        slip = np.linalg.norm(rel - rel[0], axis=1)
        travel = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p[h, :2], axis=0), axis=1))])
        first = travel <= 0.3
        dist = float(travel[-1])
        w5 = {}
        if "back1" in self.f["program"]:  # W5 is the backing off: the first 0.3 m of a heading-kept carry
            w5 = dict(w5_slip_cm=100 * float(slip[first].max()), w5_sag_cm=100 * float(sag[first].max()))
        return dict(
            **w5,
            carry_slip_cm=100 * float(slip.max()),
            carry_sag_cm_per_m=None if dist < 0.2 else 100 * float(sag[-1]) / dist,
            carry_tilt_max_deg=float(max(tilt)),
            carry_dist_cm=100 * dist,
            carry_turn_deg=float(np.degrees(abs(wrap(yaw_of(p[h[-1], 3:]) - yaw_of(p[h[0], 3:]))))),
            held_at_end=bool(b[-1, 2] - b[0, 2] > 0.05 and all(self.contacts()[-1][1])),
        )

    # -- W7 -------------------------------------------------------------------
    def w7(self):
        if "release" not in self.ev or "put_down" not in self.f["program"]:
            return {}  # a carry round's jaws also open, when the expert gives up
        b = self.z["box"]
        top = self.f["platform_top"]
        meta = json.loads((SCENE / "scene_meta.json").read_text())["surfaces"]["A"]
        inside = np.all(np.abs(b[-1, :2] - np.asarray(meta["xy"][:2])) <= np.asarray(meta["flat_half_extent"][:2]))
        r = window(self.t, self.ev["release"] - 3.0, self.t[-1])
        tilt = np.array([tilt_deg(b[i, 3:]) for i in r])
        return dict(
            place_height_cm=100 * float(b[-1, 2] - top),
            place_tilt_deg=float(tilt[-1]),
            place_travel_cm=100 * float(np.linalg.norm(b[-1, :2] - b[0, :2])),
            placed=bool(inside and abs(b[-1, 2] - top) < 0.02 and tilt[-1] < 5.0),
            hand_on_box_end=bool(any(self.contacts()[-1][1])),
            place_tilt_max_deg=float(tilt.max()),
        )

    # -- W8 -------------------------------------------------------------------
    def w8(self):
        gap = int(0.1 / DT)
        out = {}
        for key in ("arm_box", "platform", "room", "floor"):
            on = np.array([key in c[0] for c in self.contacts()])
            n, last = 0, -(10**9)
            for i in np.flatnonzero(on):
                if i - last > gap:
                    n += 1
                last = i
            out[f"hits_{key}"] = n
        return out

    def measure(self):
        r = dict(
            label=self.label,
            controller=self.f["controller"],
            program=self.f["program"],
            ok=self.f.get("ok"),
            sonic_ok=self.f.get("sonic_ok", self.f.get("ok")),
            rtf=self.f.get("rtf"),
        )
        for part in (self.w1, self.w2, self.w3, self.w4, self.carry, self.w7, self.w8):
            r.update(part())
        return r


COLUMNS = [
    ("W1", "low_gap_cm", "低位最前点到边线 cm"),
    ("W1", "low_gap_min_cm", "走近途中最小 cm（< 0 越线）"),
    ("W1", "standoff_cm", "骨盆到边线 cm"),
    ("W1", "lateral_cm", "相对箱心横向 cm"),
    ("W1", "yaw_err_deg", "相对正对箱子的朝向误差 °"),
    ("W1", "still_s", "刹车到静止 s"),
    ("W2", "torso_pitch_close_deg", "合爪时躯干俯仰 °"),
    ("W2", "torso_pitch_max_deg", "躯干最大俯仰 °"),
    ("W2", "pelvis_z_close_cm", "合爪时骨盆高度 cm"),
    ("W2", "pelvis_z_min_cm", "骨盆最低 cm"),
    ("W2", "waist_pitch_max_deg", "腰 pitch 最大 °"),
    ("W2", "waist_demand_p95_nm", "腰 pitch 静态力矩 p95 N·m"),
    ("W2", "waist_demand_max_nm", "腰 pitch 静态力矩最大 N·m"),
    ("W3", "descend_drift_max_cm", "下探段两手共同漂移最大 cm"),
    ("W3", "sep_err_cm", "夹持段两手间距误差 cm"),
    ("W3", "sep_err_max_cm", "夹持段两手间距误差最大 cm"),
    ("W3", "common_drift_cm", "夹持段两手共同漂移 cm"),
    ("W3", "common_drift_max_cm", "夹持段两手共同漂移最大 cm"),
    ("W3", "pinch_both", "合爪 0.5 s 后两手都两指夹住箱壁的时间占比"),
    ("W3", "pinch_any", "至少一只手两指夹住的时间占比"),
    ("W3", "palm_on_box", "两手掌部都贴箱的时间占比"),
    ("W4", "rise_end_cm", "结束时箱子抬高 cm"),
    ("W4", "lift_shortfall_cm", "抬起量指令减实际 cm"),
    ("W4", "box_tilt_end_deg", "结束时箱子倾角 °"),
    ("W4", "torso_pitch_end_deg", "结束时躯干俯仰 °"),
    ("W4", "pelvis_drop_cm", "提起后骨盆下沉 cm"),
    ("W5", "w5_slip_cm", "后退前 0.3 m 箱沿相对双手中点的位移 cm"),
    ("W5", "w5_sag_cm", "后退前 0.3 m 箱沿相对双手下滑 cm"),
    ("W6", "carry_slip_cm", "携带全程箱沿相对双手中点的位移 cm"),
    ("W6", "carry_sag_cm_per_m", "箱沿相对双手每米下滑 cm/m"),
    ("W6", "carry_tilt_max_deg", "携带中箱子最大倾角 °"),
    ("W6", "carry_dist_cm", "携带走过的路程 cm"),
    ("W6", "carry_turn_deg", "携带净转向 |°|"),
    ("W7", "place_height_cm", "箱底到台面 cm"),
    ("W7", "place_tilt_deg", "放下后倾角 °"),
    ("W7", "place_tilt_max_deg", "放下过程最大倾角 °"),
    ("W7", "place_travel_cm", "箱子相对起点平移 cm"),
    ("W8", "hits_arm_box", "手以外的臂段碰箱次数"),
    ("W8", "hits_platform", "身体碰平台次数"),
    ("W8", "hits_room", "身体碰房间次数"),
    ("W8", "hits_floor", "脚以外碰地次数"),
]
COUNTS = [
    ("W4", "lifted", "抬起 ≥ 5 cm 持续 2 s（分母：合了爪的局）"),
    ("W4", "lifted_level", "抬起且结束时箱子倾角 ≤ 15°（专家自己的 max_tilt_deg）"),
    ("W6", "held_at_end", "携带结束仍拿着（箱子高于起点 5 cm，两只手都碰着箱子）"),
    ("W7", "placed", "放稳在台面上（箱底 ≤ 2 cm、倾角 < 5°、在台面范围内）"),
    ("W7", "hand_on_box_end", "结束时手仍贴着箱子"),
]


def table(rows, title):
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['controller']} / {r['program']}", []).append(r)
    lines = [
        f"# {title}",
        "",
        "mean ± sd；每组的局数见表头。生成：`warehouse_measure.py`。",
        "",
        "| 项 | 指标 | " + " | ".join(f"{g}（{len(v)} 局）" for g, v in groups.items()) + " |",
        "|---|---|" + "---|" * len(groups),
    ]
    for item, key, name in COLUMNS:
        cells = [fmt([r.get(key) for r in v]) for v in groups.values()]
        if all(c == "–" for c in cells):
            continue
        lines.append(f"| {item} | {name} | " + " | ".join(cells) + " |")
    for item, key, name in COUNTS:
        cells = []
        for v in groups.values():
            n = sum(1 for r in v if key in r)
            cells.append(f"{sum(1 for r in v if r.get(key))}/{n}" if n else "–")
        if all(c == "–" for c in cells):
            continue
        lines.append(f"| {item} | {name} | " + " | ".join(cells) + " |")
    falls = [f"{sum(1 for r in v if not r.get('ok', True) and r['controller'] == 'scalebfm')}" for v in groups.values()]
    lines.append("| – | ScaleBFM 摔倒局数 | " + " | ".join(falls) + " |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="warehouse_l1")
    ap.add_argument("--title", default="warehouse L1（TODO 25.3）")
    a = ap.parse_args(argv)
    sc = Scene()
    rows = []
    for d in a.states:
        for npz in sorted(d.glob("*.npz")):
            r = Round(npz, sc).measure()
            rows.append(r)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.json").write_text(json.dumps(rows, indent=1))
    lines = table(rows, a.title)
    (a.out / f"{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
