"""Bottle L1 measurements (TODO 25.2, plan §4.1 and §10): B1 B2 B4 B5 B6 B8 from the round files of bottle_states.py,
the same arithmetic for both controllers.

Windows come from the round's events (simulation seconds): `walk` (first moving command), `brake` (the stop
decision: SONIC's guard brake, ScaleBFM's reference deceleration), `ready` (arms leave the walking posture),
`close`, `lift`. A metric whose events are missing is left out (a replay round has no walk; a walk round no reach).

* B1 stand (read 0.5 s after B2's still tick): band gap = table edge line minus the frontmost robot point within the
  table-top band (EdgeGuard's own measure, foot_watch.FootWatch.frontmost_band), its minimum over the whole walk
  (< 0 = crossed), pelvis lateral offset from the planned stand along the edge, yaw error to the stand heading psi.
* B2 still: seconds from `brake` to the first tick that starts 0.5 s with pelvis planar speed < 0.03 m/s and every
  leg joint < 0.2 rad/s (stricter than the expert's takeover gate, 0.10 / 0.50).
* B4 wrist tracking over [ready, close]: |commanded - measured| right wrist position (mean, p95) and orientation, and
  the lag (0-0.6 s) that minimises the mean position error of measured(t) against commanded(t - lag).
* B5 body during [ready, close]: pelvis planar excursion from its pose at `ready`, foot slip (per foot, the net
  horizontal travel of a sole point over each stance span, summed; see foot_slip), waist yaw / roll / pitch
  excursion, peak torso pitch.
* B6 over [close, close + 1 s]: right-wrist jitter RMS (position minus its 0.2 s centred mean); from close + 1 s to
  lift + 3 s the bottle's largest displacement in the right-wrist frame; lifted = bottle 6 cm above its start for
  2 s (sonic_lift6cm_hold2s_v1).
* B7 the carry (the production expert's `carry_and_verify`, or `carry` of a carry plan / ScaleBFM's carry) to the
  record's end: the bottle's largest displacement in the right-wrist frame, the pelvis's net travel and turn, and
  whether the bottle is still 6 cm up at the end.
* B8 head camera (fixed on torso_link): angular speed mean / p95 during the walk, the brake (brake - 0.5 .. + 1.5 s)
  and the reach [ready, close].

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/bottle_measure.py \\
        outputs/controller_compare_20261013/bottle/sonic/states [more state dirs ...] --out outputs/controller_compare_20261013/bottle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bottle_states import Fk  # noqa: E402

from locomani.primitives.foot_watch import FootWatch  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

DT = 0.02
LEGS = [i for i, n in enumerate(MUJOCO_JOINT_NAMES) if any(k in n for k in ("hip", "knee", "ankle"))]
WAIST = {k: MUJOCO_JOINT_NAMES.index(f"waist_{k}_joint") for k in ("yaw", "roll", "pitch")}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def quat_angle(q0, q1):
    return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0, 1))


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def mat(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    return r.reshape(3, 3)


def window(t, a, b):
    return np.flatnonzero((t >= a) & (t <= b))


class Round:
    def __init__(self, npz: Path, fk: Fk, watch: FootWatch):
        self.z = {k: v for k, v in np.load(npz).items()}
        self.f = json.loads(npz.with_suffix(".json").read_text())
        self.label = npz.stem
        self.t = self.z["t"]
        self.ev = self.f["events"]
        self.fk, self.watch = fk, watch
        self._body = {}

    def body(self, name):
        """(T,7) world pose of a robot body, FK of the recorded pelvis + joints (cached)."""
        if name not in self._body:
            out = np.empty((len(self.t), 7))
            for i in range(len(self.t)):
                self.fk.put(self.z["pelvis"][i], self.z["joints"][i])
                out[i] = self.fk.pose(name)
            self._body[name] = out
        return self._body[name]

    def state(self, i):
        p = self.z["pelvis"][i]
        return dict(pelvis=dict(pos=p[:3], quat=p[3:]), joints=self.z["joints"][i])

    # -- B2 / B1 --------------------------------------------------------------
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

    def b1_b2(self):
        i = self.still_tick()
        if i is None:
            return dict(still_s=None)
        out = dict(still_s=float(self.t[i] - self.ev["brake"]))
        edge = self.f["edge"]
        if edge.get("axis") is None:
            return out
        axis, line = np.asarray(edge["axis"], float), float(edge["line"])
        band = self.f["band"]
        k = min(i + int(0.5 / DT), len(self.t) - 1)
        gap = lambda j: line - self.watch.frontmost_band(self.state(j), np.zeros(2), axis, *band)[0]
        walk = window(self.t, self.ev.get("walk", self.t[0]), self.t[k])
        across = np.array([-axis[1], axis[0]])
        p = self.z["pelvis"][k]
        out.update(
            band_gap_cm=100 * gap(k),
            band_gap_min_cm=100 * min(gap(j) for j in walk[:: max(1, len(walk) // 200)]),
            lateral_cm=100 * float((p[:2] - np.asarray(self.f["stand_xy"])) @ across),
            yaw_err_deg=float(np.degrees(wrap(yaw_of(p[3:]) - self.f["psi"]))),
            pelvis_gap_cm=100 * float(line - axis @ p[:2]),
        )
        return out

    # -- B4 / B5 / B6 ---------------------------------------------------------
    def b4(self):
        if "ready" not in self.ev or "close" not in self.ev:
            return {}
        w = window(self.t, self.ev["ready"], self.ev["close"])
        cmd = self.z["cmd_wrist"][:, 1]
        w = w[np.isfinite(cmd[w, 0])]
        if len(w) < 10:
            return {}
        meas = self.body("right_wrist_yaw_link")
        e = np.linalg.norm(meas[w, :3] - cmd[w, :3], axis=1)
        lags = range(0, int(0.6 / DT) + 1)
        mean_at = [np.mean(np.linalg.norm(meas[w[w >= s], :3] - cmd[w[w >= s] - s, :3], axis=1)) for s in lags]
        return dict(
            wrist_err_cm=100 * float(e.mean()),
            wrist_err_p95_cm=100 * float(np.percentile(e, 95)),
            wrist_err_deg=float(np.degrees(np.mean(quat_angle(meas[w, 3:], cmd[w, 3:])))),
            lag_s=float(np.argmin(mean_at) * DT),
            wrist_err_lagged_cm=100 * float(min(mean_at)),
        )

    def foot_slip(self, idx):
        """Per foot, summed over its stance spans in idx (floor contact on every tick): the horizontal distance the
        span's first contact centroid -- a point fixed in the foot -- travels from the span's first tick to its last.
        Net per span, so contact jitter does not accumulate and a step (contact lost) starts a new span."""
        m, d = self.fk.m, self.fk.d
        feet = [m.body(n).id for n in FEET]
        floor = m.geom("floor").id
        span = [None, None]  # (local point, world xy at the span start, last world xy)
        slip = np.zeros(2)
        for i in idx:
            self.fk.put(self.z["pelvis"][i], self.z["joints"][i])
            mujoco.mj_forward(m, d)
            pts = [[], []]
            for c in d.contact[: d.ncon]:
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if o == floor and m.geom_bodyid[g] in feet:
                        pts[feet.index(m.geom_bodyid[g])].append(c.pos.copy())
            for k, b in enumerate(feet):
                R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
                if not pts[k]:
                    if span[k] is not None:
                        slip[k] += np.linalg.norm(span[k][2] - span[k][1])
                    span[k] = None
                    continue
                if span[k] is None:
                    local = R.T @ (np.mean(pts[k], axis=0) - x)
                    span[k] = [local, (R @ local + x)[:2].copy(), None]
                span[k][2] = (R @ span[k][0] + x)[:2].copy()
        for k in range(2):
            if span[k] is not None:
                slip[k] += np.linalg.norm(span[k][2] - span[k][1])
        return slip

    def b5(self):
        if "ready" not in self.ev or "close" not in self.ev:
            return {}
        w = window(self.t, self.ev["ready"], self.ev["close"])
        p, q = self.z["pelvis"], self.z["joints"]
        torso = self.body("torso_link")
        pitch = [np.degrees(np.arcsin(-mat(x[3:])[2, 0])) for x in torso[w]]
        slip = self.foot_slip(w)
        return dict(
            pelvis_excursion_cm=100 * float(np.max(np.linalg.norm(p[w, :2] - p[w[0], :2], axis=1))),
            foot_slip_cm=100 * float(slip.max()),
            waist_yaw_deg=float(np.degrees(np.max(np.abs(q[w, WAIST["yaw"]] - q[w[0], WAIST["yaw"]])))),
            waist_roll_deg=float(np.degrees(np.max(np.abs(q[w, WAIST["roll"]] - q[w[0], WAIST["roll"]])))),
            waist_pitch_deg=float(np.degrees(np.max(np.abs(q[w, WAIST["pitch"]] - q[w[0], WAIST["pitch"]])))),
            torso_pitch_max_deg=float(np.max(pitch)),
        )

    def b6(self):
        if "close" not in self.ev:
            return {}
        out = {}
        wr = self.body("right_wrist_yaw_link")
        c = window(self.t, self.ev["close"], self.ev["close"] + 1.0)
        if len(c) > 12:
            k = int(0.1 / DT)
            pos = wr[:, :3]
            smooth = np.array([pos[max(0, i - k) : i + k + 1].mean(0) for i in c])
            out["close_jitter_rms_mm"] = 1000 * float(np.sqrt(np.mean(np.sum((pos[c] - smooth) ** 2, axis=1))))
        b = self.z["bottle"]
        rise = b[:, 2] - b[0, 2]
        up = rise > 0.06
        run, best = 0, 0
        for u in up:
            run = run + 1 if u else 0
            best = max(best, run)
        out["lifted"] = bool(best * DT >= 2.0)
        out["rise_max_cm"] = 100 * float(rise.max())
        if "lift" in self.ev:
            h = window(self.t, self.ev["close"] + 1.0, self.ev["lift"] + 3.0)
            if len(h) > 5:
                rel = np.array([mat(wr[i, 3:]).T @ (b[i, :3] - wr[i, :3]) for i in h])
                out["in_hand_slip_cm"] = 100 * float(np.max(np.linalg.norm(rel - rel[0], axis=1)))
        return out

    def b7(self):
        """The carry: from `carry_and_verify` (the production expert: carry pose, 0.35 m back-off, terminal hold) or
        `carry` (a carry plan / ScaleBFM's carry) to the record's end: the bottle's largest displacement in the
        right-wrist frame, the pelvis's net travel and turn, and whether the bottle is still 6 cm up at the end."""
        start = self.ev.get("carry_and_verify", self.ev.get("carry"))
        if start is None:
            return {}
        attempted = self.f.get("steps_attempted")
        if attempted is not None and not {"walk_to", "turn_to", "carry_to"} & set(attempted):
            return {}  # a plan round that ended before its carry step (a precondition refused it)
        p = self.z["pelvis"]
        if "carry_and_verify" not in self.ev:
            # a carry: from 0.5 s before the base starts to move (3 cm or 3 deg from its pose at `carry`)
            i0 = int(np.searchsorted(self.t, start))
            moved = np.flatnonzero(
                (np.linalg.norm(p[i0:, :2] - p[i0, :2], axis=1) > 0.03)
                | (np.abs([wrap(yaw_of(x[3:]) - yaw_of(p[i0, 3:])) for x in p[i0:]]) > np.radians(3))
            )
            if not len(moved):
                return {}
            start = float(self.t[i0 + moved[0]]) - 0.5
        h = window(self.t, start, self.t[-1])
        if len(h) < 5:
            return {}
        wr = self.body("right_wrist_yaw_link")
        b = self.z["bottle"]
        rel = np.array([mat(wr[i, 3:]).T @ (b[i, :3] - wr[i, :3]) for i in h])
        return dict(
            carry_slip_cm=100 * float(np.max(np.linalg.norm(rel - rel[0], axis=1))),
            carry_back_cm=100 * float(np.linalg.norm(p[h[-1], :2] - p[h[0], :2])),
            carry_turn_deg=float(np.degrees(abs(wrap(yaw_of(p[h[-1], 3:]) - yaw_of(p[h[0], 3:]))))),
            held_at_end=bool(b[-1, 2] - b[0, 2] > 0.06),
        )

    def b8(self):
        torso = self.body("torso_link")
        w = np.array([np.degrees(quat_angle(torso[i, 3:], torso[i + 1, 3:])) / DT for i in range(len(self.t) - 1)])
        out = {}
        spans = {
            "walk": (self.ev.get("walk"), self.ev.get("brake")),
            "brake": (None if "brake" not in self.ev else self.ev["brake"] - 0.5, None),
            "reach": (self.ev.get("ready"), self.ev.get("close")),
        }
        if spans["brake"][0] is not None:
            spans["brake"] = (spans["brake"][0], spans["brake"][0] + 2.0)
        for k, (a, b) in spans.items():
            if a is None or b is None:
                continue
            i = window(self.t[:-1], a, b)
            if len(i):
                out[f"cam_{k}_deg_s"] = float(w[i].mean())
                out[f"cam_{k}_p95_deg_s"] = float(np.percentile(w[i], 95))
        return out

    def measure(self):
        r = dict(label=self.label, controller=self.f["controller"], program=self.f["program"], ok=self.f.get("ok"))
        for part in (self.b1_b2, self.b4, self.b5, self.b6, self.b7, self.b8):
            r.update(part())
        return r


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    a = np.asarray(vals, float)
    return f"{a.mean():.1f} ± {a.std():.1f}" if len(a) > 1 else f"{a[0]:.1f}"


COLUMNS = [
    ("B1", "band_gap_cm", "桌边带最前点到桌边 cm"),
    ("B1", "band_gap_min_cm", "走近途中最小 cm（< 0 越线）"),
    ("B1", "lateral_cm", "沿桌边横向 cm"),
    ("B1", "yaw_err_deg", "朝向误差 °"),
    ("B2", "still_s", "刹车到静止 s"),
    ("B4", "wrist_err_cm", "右腕误差 cm"),
    ("B4", "wrist_err_p95_cm", "右腕误差 p95 cm"),
    ("B4", "wrist_err_deg", "右腕姿态误差 °"),
    ("B4", "lag_s", "滞后 s"),
    ("B5", "pelvis_excursion_cm", "骨盆偏移 cm"),
    ("B5", "foot_slip_cm", "脚滑 cm"),
    ("B5", "waist_yaw_deg", "腰 yaw 变化 °"),
    ("B5", "waist_roll_deg", "腰 roll 变化 °"),
    ("B5", "torso_pitch_max_deg", "躯干最大俯仰 °"),
    ("B6", "close_jitter_rms_mm", "合爪手抖 RMS mm"),
    ("B6", "in_hand_slip_cm", "提起保持掌系滑移 cm"),
    ("B7", "carry_slip_cm", "携带掌系滑移 cm"),
    ("B7", "carry_back_cm", "携带位移 cm"),
    ("B7", "carry_turn_deg", "携带净转向 |°|"),
    ("B8", "cam_walk_p95_deg_s", "走路相机角速度 p95 °/s"),
    ("B8", "cam_brake_p95_deg_s", "刹车相机角速度 p95 °/s"),
    ("B8", "cam_reach_p95_deg_s", "伸手相机角速度 p95 °/s"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    fk = Fk()
    watch = FootWatch(model=fk.m)
    rows = []
    for d in a.states:
        for npz in sorted(d.glob("*.npz")):
            r = Round(npz, fk, watch).measure()
            rows.append(r)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bottle_l1.json").write_text(json.dumps(rows, indent=1))
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['controller']} / {r['program']}", []).append(r)
    lines = [
        "# bottle L1（TODO 25.2）",
        "",
        "mean ± sd；每组的局数见表头。生成：`bottle_measure.py`。",
        "",
        "| 项 | 指标 | " + " | ".join(f"{g}（{len(v)} 局）" for g, v in groups.items()) + " |",
        "|---|---|" + "---|" * len(groups),
    ]
    for item, key, name in COLUMNS:
        cells = [fmt([r.get(key) for r in v]) for v in groups.values()]
        if all(c == "–" for c in cells):
            continue
        lines.append(f"| {item} | {name} | " + " | ".join(cells) + " |")

    def count(v, key):
        n = sum(1 for r in v if key in r)
        return f"{sum(1 for r in v if r.get(key))}/{n}" if n else "–"

    lifted = [count(v, "lifted") for v in groups.values()]
    lines.append("| B6 | 抬起 ≥ 6 cm 持续 2 s（分母：走到合爪的局） | " + " | ".join(lifted) + " |")
    held = [count(v, "held_at_end") for v in groups.values()]
    lines.append("| B7 | 携带结束瓶子仍在手里（高于起点 6 cm） | " + " | ".join(held) + " |")
    (a.out / "bottle_l1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
