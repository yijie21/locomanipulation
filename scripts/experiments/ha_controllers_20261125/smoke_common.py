"""Controller smoke tests run identically in LocoManiLab (lml_ref.py) and HumanoidArena (probe_ctrl.py), 2026-11-25.

Every phase drives a host through lml_host.HostDriver's surface and nothing else -- reset(), start_frames(), push(world
qpos frame), step(), sample(), pl (lml_selfmodel.Planner on robot_fk), to_mj (ScaleBFM order -> MuJoCo order),
ref_delay_s -- so both simulators get the same references from the same code: LML's stance (robot_fk's default pose at
0.8 m, at the measured spawn xy / heading), d3_collect's planning (whole-body IK from the measured state, min-jerk
blend, the reference shifted onto the current reference root), d3_collect's settle rule and lag measurement, and
hg_walk's gait reference.

Phases:
  stand  the stance reference held `seconds`.
  lag    d3_collect.lag's six single-joint ramps (1 s min-jerk, then 3 s hold); lag = frames from the reference crossing
         half its ramp to the joint crossing half its own move (and at 90 %), reached share = joint move / ramp.
  reach  the right hand point (13 cm ahead of wrist_yaw_link) 30 cm ahead along the pelvis heading, same height, the
         wrist attitude kept; commanded on the first settled row after 0.7 s (data3's first reach), planned at
         0.15 m/s (Planner.reach: 2 s). Labelled where data3 labels: the first row, after the reference end plus the
         buffer delay, closing 0.5 s of pelvis and both hand points under 2 cm/s (0.1 s displacements); at most 3 s.
         Error = hand point at the label minus the target (world, and in the heading frame at the command). Joint lag
         as `lag` on every joint the reference moves by more than 0.1 rad.
  walk   hg_walk.walk_frames to a goal 0.5 m ahead (heading kept) from the first settled row after 0.7 s, then 3 s
         hold. Reports the pelvis displacement and heading change against the reference's, the largest deviation of
         the pelvis from the reference path (both measured from the command row) and the final error to the goal.
A fall is the pelvis below 0.45 m (lml_selfmodel.FALL_Z); hosts may add their own detector in `extra()`.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

W = Path(os.environ.get("LOCO_W", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
for p in (W / "src", W / "scripts/experiments/locomanilab_20261117", W / "scripts/experiments/hardgrasp_20261125"):
    if str(p) not in sys.path:
        sys.path.append(str(p))
import lml_selfmodel as L  # noqa: E402

FPS = L.FPS
FALL_Z = L.FALL_Z
HAND_OFF = L.HAND_OFF
LAG_MOVES = ((22, -0.5), (25, -0.4), (23, -0.3), (15, -0.5), (18, -0.4), (16, 0.3))  # d3_collect.LAG_MOVES (MuJoCo order)
SETTLE_V, SETTLE_F, SETTLE_MAX_F = 0.02, int(round(0.5 * FPS)), int(round(3.0 * FPS))  # d3_collect
FIRST_OPEN_F = int(round(L.START_HOLD_S * FPS)) - SETTLE_F  # 10: the first command no earlier than 0.7 s
HOLD_S = 4.5  # d3_collect.HOLD_S


def mj_names():
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    return list(MUJOCO_JOINT_NAMES)


def hand_point(x: dict, side: str) -> np.ndarray:
    p, R = x["wr"][side]
    return np.asarray(p) + np.asarray(R) @ HAND_OFF


def yaw_of_root(x: dict) -> float:
    return L.yaw_of(L._qmat(x["root"][3:]))


def tilt_deg(root7) -> float:
    R = L._qmat(root7[3:])
    return float(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))


class Settle:
    """d3_collect's settle rule: pelvis and both hand points slower than 2 cm/s (displacement over the last 5 steps)
    for 25 consecutive steps; counted from `open_f`, given up after SETTLE_MAX_F."""

    def __init__(self):
        self.hist = []
        self.run = 0

    def push(self, x: dict) -> float:
        pts = np.concatenate([x["root"][:3], hand_point(x, "right"), hand_point(x, "left")])
        self.hist.append(pts)
        if len(self.hist) <= 5:
            v = np.inf
        else:
            d = (self.hist[-1] - self.hist[-6]).reshape(3, 3)
            v = float(np.linalg.norm(d, axis=1).max() / (5 / FPS))
        self.v = v
        return v

    def quiet(self) -> bool:
        return self.v < SETTLE_V


def _run_until_settled(h, frames, f, open_f, rows, rec):
    """Push / step from frame f until settled (the 0.5 s window entirely after open_f) or SETTLE_MAX_F after open_f.
    Returns (f of the settled row or None if it ran out, fell_at or None, f_next)."""
    st = Settle()
    run = 0
    while True:
        h.push(frames[min(f, len(frames) - 1)])
        h.step()
        x = rows[f] = h.sample()
        rec(f, x)
        v = st.push(x)
        if float(x["root"][2]) < FALL_Z:
            return None, f, f + 1
        run = run + 1 if (f >= open_f and v < SETTLE_V) else 0
        if run >= SETTLE_F:
            return f, None, f + 1
        if f - open_f >= SETTLE_MAX_F:
            return None, None, f + 1
        f += 1


def crossing_lag(ref: np.ndarray, act: np.ndarray, f0: int, lvl: float):
    """Frames from the reference crossing `lvl` of its own move (after f0) to the joint crossing `lvl` of its own."""

    def cross(v):
        d = v[-1] - v[f0 - 1]
        if abs(d) < 1e-6:
            return None
        s = (v[f0 - 1 :] - v[f0 - 1]) / d
        idx = np.flatnonzero(s >= lvl)
        return int(idx[0]) if idx.size else None

    a, b = cross(ref), cross(act)
    return None if a is None or b is None else b - a


class Recorder:
    def __init__(self, h):
        self.h = h
        self.rows = {}
        self.ext = {}

    def __call__(self, f, x):
        e = getattr(self.h, "extra", None)
        if e is not None:
            self.ext[f] = e()


# -- phases -----------------------------------------------------------------------------------------------------------
def phase_stand(h, seconds: float) -> dict:
    h.reset()
    frames = h.start_frames()
    n = int(round(seconds * FPS))
    t0 = time.time()
    zs, tilts, xy, jerr, fell = [], [], [], [], None
    to_mj = h.to_mj
    ext = []
    for f in range(n):
        fr = frames[min(f, len(frames) - 1)]
        h.push(fr)
        h.step()
        x = h.sample()
        zs.append(float(x["root"][2]))
        tilts.append(tilt_deg(x["root"]))
        xy.append(np.asarray(x["root"][:2], float))
        jerr.append(np.abs(x["q"][to_mj] - fr[7:]))
        e = getattr(h, "extra", None)
        if e is not None:
            ext.append(e())
        if fell is None and zs[-1] < FALL_Z:
            fell = f
            break
    xy = np.array(xy)
    jerr = np.array(jerr)
    out = dict(seconds=seconds, steps=len(zs), wall_s=time.time() - t0, fell_at_s=None if fell is None else fell / FPS,
               root_z_start=zs[0], root_z_min=min(zs), root_z_end=zs[-1], tilt_max_deg=max(tilts),
               tilt_end_deg=tilts[-1], xy_drift_m=float(np.linalg.norm(xy[-1] - xy[0])),
               xy_drift_after_2s_m=float(np.linalg.norm(xy[-1] - xy[min(len(xy) - 1, 2 * FPS)])),
               joint_err_vs_ref_end_max_rad=float(jerr[-1].max()), joint_err_vs_ref_end_mean_rad=float(jerr[-1].mean()),
               joint_err_vs_ref_end_worst=mj_names()[int(jerr[-1].argmax())])  # fmt: skip
    if ext:
        out["host"] = summarize_extra(ext)
    return out


def phase_lag(h) -> dict:
    """d3_collect.lag on this host."""
    names = mj_names()
    to_mj = h.to_mj
    out = []
    for j, amp in LAG_MOVES:
        h.reset()
        fr = h.start_frames()
        f0, n = len(fr) + 10, FPS
        fr = np.concatenate([fr, np.repeat(fr[-1:], 10 + n + 3 * FPS, 0)])
        s_ = np.arange(n + 1) / n
        fr[f0 : f0 + n + 1, 7 + j] += amp * (10 * s_**3 - 15 * s_**4 + 6 * s_**5)
        fr[f0 + n + 1 :, 7 + j] += amp
        q, fell = [], None
        for f in range(len(fr)):
            h.push(fr[f])
            h.step()
            x = h.sample()
            q.append(float(x["q"][to_mj][j]))
            if fell is None and float(x["root"][2]) < FALL_Z:
                fell = f
        q = np.array(q)
        da = q[-1] - q[f0 - 1]
        l50 = crossing_lag(fr[:, 7 + j], q, f0, 0.5)
        l90 = crossing_lag(fr[:, 7 + j], q, f0, 0.9)
        out.append(dict(joint=names[j], amp=amp, lag50_ms=None if l50 is None else 1000 * l50 / FPS,
                        lag90_ms=None if l90 is None else 1000 * l90 / FPS, reached=float(da / amp),
                        end_err_rad=float(q[-1] - fr[-1, 7 + j]), fell=fell))  # fmt: skip
    l50s = [o["lag50_ms"] for o in out if o["lag50_ms"] is not None]
    l90s = [o["lag90_ms"] for o in out if o["lag90_ms"] is not None]
    return dict(moves=out, median_lag50_ms=float(np.median(l50s)) if l50s else None,
                median_lag90_ms=float(np.median(l90s)) if l90s else None,
                median_reached=float(np.median([o["reached"] for o in out])), ref_delay_s=float(h.ref_delay_s))  # fmt: skip


def plan_rel_reach(h, x: dict, ref_frame: np.ndarray, side: str, dist: float):
    """d3_collect.plan_reach for one hand with the target `dist` ahead of the hand point along the pelvis heading
    (same height) and the wrist attitude unchanged (both from the robot_fk FK of the measured state)."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    q_mj = x["q"][h.to_mj].astype(float)
    cur = h.pl.fk_world(p, quat, q_mj)
    a, R0 = cur[side]
    target_w = a + L.rot_z(yaw) @ np.array([dist, 0.0, 0.0])
    poses, durs, ik_err, dur = h.pl.reach((p, quat, q_mj), side, target_w, R0, None)
    delta = np.r_[ref_frame[:2] - p[:2], 0.0]
    ref_pose = (ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    meas_hand = hand_point(x, side)
    return L.blend_frames(shifted, durs), shifted[-1], dict(target_w=target_w, rot_w=R0, ik_err_cm=100 * ik_err,
                                                           reach_s=dur, fk_hand_minus_measured_cm=100 * float(np.linalg.norm(a - meas_hand)),
                                                           yaw=yaw, delta=delta)  # fmt: skip


def phase_reach(h, side: str = "right", dist: float = 0.30) -> dict:
    h.reset()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(HOLD_S * FPS) + SETTLE_MAX_F, 0)])  # as d3_collect
    rows, ext = {}, {}

    def rec(f, x):
        e = getattr(h, "extra", None)
        if e is not None:
            ext[f] = e()

    f_cmd, fell, f = _run_until_settled(h, frames, 0, FIRST_OPEN_F, rows, rec)
    if f_cmd is None:
        return dict(ok=False, why="fell before the command" if fell is not None else "never settled", fell_at_s=None if fell is None else fell / FPS)
    x0 = rows[f_cmd]
    blend, last, plan = plan_rel_reach(h, x0, frames[f_cmd], side, dist)
    frames = np.concatenate([frames[: f_cmd + 1], blend, L.hold(last, HOLD_S)])
    f_end = f_cmd + len(blend)
    open_f = f_end + int(round(h.ref_delay_s * FPS))
    # run to the reference end, then the settle wait (data3: the window opens at the reference end + buffer delay)
    fell = None
    while f < open_f:
        h.push(frames[f])
        h.step()
        x = rows[f] = h.sample()
        rec(f, x)
        if float(x["root"][2]) < FALL_Z:
            fell = f
            break
        f += 1
    label, fell2, f = (None, fell, f) if fell is not None else _run_until_settled(h, frames, f, open_f, rows, rec)
    fell = fell if fell is not None else fell2
    settled = label is not None
    if label is None:
        label = max(rows)
    x1 = rows[label]
    tgt = plan["target_w"]
    err_w = hand_point(x1, side) - tgt
    Rh = L.rot_z(plan["yaw"])
    err_h = Rh.T @ err_w
    moved_w = hand_point(x1, side) - hand_point(x0, side)
    # joint lag on the joints the reference moves (frames are pushed one per step: frame index = step)
    ks = sorted(k for k in rows if k >= f_cmd - 1)
    names = mj_names()
    ref = np.array([frames[k][7:] for k in ks])
    act = np.array([rows[k]["q"][h.to_mj] for k in ks])
    f0 = 1  # f_cmd sits at index 1 (index 0 is f_cmd - 1)
    lags = []
    for j in range(29):
        if abs(ref[-1, j] - ref[0, j]) > 0.1:
            l50 = crossing_lag(ref[:, j], act[:, j], f0 + 1, 0.5)
            l90 = crossing_lag(ref[:, j], act[:, j], f0 + 1, 0.9)
            lags.append(dict(joint=names[j], ref_move_rad=float(ref[-1, j] - ref[0, j]),
                             act_move_rad=float(act[-1, j] - act[0, j]),
                             lag50_ms=None if l50 is None else 1000 * l50 / FPS,
                             lag90_ms=None if l90 is None else 1000 * l90 / FPS,
                             end_err_rad=float(act[-1, j] - ref[-1, j])))  # fmt: skip
    l50s = [o["lag50_ms"] for o in lags if o["lag50_ms"] is not None]
    arm = [o for o in lags if any(k in o["joint"] for k in ("shoulder", "elbow", "wrist"))]
    # the hand point's own lag along the reach direction: reference hand (robot_fk FK of the frames) vs measured
    u = (tgt - hand_point(x0, side)) / max(float(np.linalg.norm(tgt - hand_point(x0, side))), 1e-9)
    ref_h = np.array([float(u @ h.pl.fk_world(frames[k][:3].astype(float), frames[k][3:7].astype(float),
                                             frames[k][7:].astype(float))[side][0]) for k in ks])  # fmt: skip
    act_h = np.array([float(u @ hand_point(rows[k], side)) for k in ks])
    hl50 = crossing_lag(ref_h, act_h, f0 + 1, 0.5)
    hl90 = crossing_lag(ref_h, act_h, f0 + 1, 0.9)
    out = dict(ok=fell is None, side=side, dist_m=dist, t_cmd_s=f_cmd / FPS, reach_s=plan["reach_s"],
               ik_err_cm=plan["ik_err_cm"], fk_hand_minus_measured_cm=plan["fk_hand_minus_measured_cm"],
               fell_at_s=None if fell is None else fell / FPS, settled=settled,
               label_after_ref_end_s=(label - f_end) / FPS, label_s=label / FPS,
               err_cm=100 * float(np.linalg.norm(err_w)), err_heading_cm=(100 * err_h).tolist(),
               hand_moved_heading_cm=(100 * (Rh.T @ moved_w)).tolist(),
               pelvis_moved_cm=100 * float(np.linalg.norm(x1["root"][:2] - x0["root"][:2])),
               pelvis_dz_cm=100 * float(x1["root"][2] - x0["root"][2]),
               joint_lags=lags, median_lag50_ms=float(np.median(l50s)) if l50s else None,
               arm_median_lag50_ms=float(np.median([o["lag50_ms"] for o in arm if o["lag50_ms"] is not None])) if arm else None,
               arm_median_lag90_ms=float(np.median([o["lag90_ms"] for o in arm if o["lag90_ms"] is not None])) if arm else None,
               hand_lag50_ms=None if hl50 is None else 1000 * hl50 / FPS, hand_lag90_ms=None if hl90 is None else 1000 * hl90 / FPS,
               hand_ref_move_cm=100 * float(ref_h[-1] - ref_h[0]), hand_act_move_cm=100 * float(act_h[-1] - act_h[0]),
               max_tilt_deg=max(tilt_deg(rows[k]["root"]) for k in ks))  # fmt: skip
    out["feet_moved_cm"] = feet_moved_cm(rows, f_cmd)
    if ext:
        out["host"] = summarize_extra([ext[k] for k in sorted(ext) if k >= f_cmd])
    out["_trace"] = trace(rows, frames, f_cmd - 1, ext)
    return out


def phase_walk(h, dist: float = 0.5) -> dict:
    import hg_walk as HW

    h.reset()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(HOLD_S * FPS) + SETTLE_MAX_F, 0)])  # as d3_collect
    rows, ext = {}, {}

    def rec(f, x):
        e = getattr(h, "extra", None)
        if e is not None:
            ext[f] = e()

    f_cmd, fell, f = _run_until_settled(h, frames, 0, FIRST_OPEN_F, rows, rec)
    if f_cmd is None:
        return dict(ok=False, why="fell before the command" if fell is not None else "never settled")
    x0 = rows[f_cmd]
    yaw0 = yaw_of_root(x0)
    goal_xy = x0["root"][:2] + (L.rot_z(yaw0) @ np.array([dist, 0.0, 0.0]))[:2]
    info = {}
    wf = HW.walk_frames(h, x0, frames[f_cmd], goal_xy, yaw0, info=info)
    frames = np.concatenate([frames[: f_cmd + 1], wf, np.repeat(wf[-1:], 3 * FPS, 0)])
    fell, dev_max = None, 0.0
    anchor = (x0["root"][:2].copy(), frames[f_cmd][:2].copy())
    delay_f = int(round(h.ref_delay_s * FPS))
    while f < len(frames):
        h.push(frames[f])
        h.step()
        x = rows[f] = h.sample()
        rec(f, x)
        if float(x["root"][2]) < FALL_Z:
            fell = f
            break
        fr = frames[max(f_cmd, f - delay_f)]  # the reference the robot is executing now (buffer delay)
        dev = (x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])
        dev_max = max(dev_max, float(np.linalg.norm(dev)))
        f += 1
    x1 = rows[max(rows)]
    Rh = L.rot_z(yaw0)
    moved = Rh.T @ np.r_[x1["root"][:2] - x0["root"][:2], 0.0]
    ref_moved = Rh.T @ np.r_[frames[-1][:2] - frames[f_cmd][:2], 0.0]
    out = dict(ok=fell is None, dist_m=dist, t_cmd_s=f_cmd / FPS, walk_s=info.get("walk_s"), n_steps=info.get("n_steps"),
               ik_err_cm=info.get("ik_err_cm"), fell_at_s=None if fell is None else fell / FPS,
               pelvis_moved_heading_cm=(100 * moved[:2]).tolist(), ref_moved_heading_cm=(100 * ref_moved[:2]).tolist(),
               yaw_change_deg=math.degrees(L.wrap(yaw_of_root(x1) - yaw0)) if hasattr(L, "wrap") else
               math.degrees((yaw_of_root(x1) - yaw0 + math.pi) % (2 * math.pi) - math.pi),
               goal_err_cm=100 * float(np.linalg.norm(x1["root"][:2] - goal_xy)), max_dev_from_ref_cm=100 * dev_max,
               max_tilt_deg=max(tilt_deg(rows[k]["root"]) for k in rows if k >= f_cmd),
               root_z_min=min(float(rows[k]["root"][2]) for k in rows if k >= f_cmd))  # fmt: skip
    out["feet_moved_cm"] = feet_moved_cm(rows, f_cmd)
    if ext:
        out["host"] = summarize_extra([ext[k] for k in sorted(ext) if k >= f_cmd])
    out["_trace"] = trace(rows, frames, f_cmd - 1, ext)
    return out


def trace(rows: dict, frames: np.ndarray, f_from: int, ext: dict | None = None) -> list[dict]:
    """Per-step record from f_from: pelvis, heading, tilt, both hand points, the reference root, the feet (if the host
    samples them) and the host extras."""
    out = []
    for k in sorted(rows):
        if k < f_from:
            continue
        x = rows[k]
        r = dict(f=k, root=np.round(x["root"][:3], 5).tolist(), yaw_deg=round(math.degrees(yaw_of_root(x)), 3),
                 tilt_deg=round(tilt_deg(x["root"]), 3), hand_r=np.round(hand_point(x, "right"), 5).tolist(),
                 hand_l=np.round(hand_point(x, "left"), 5).tolist(),
                 ref_root=np.round(frames[min(k, len(frames) - 1)][:3], 5).tolist())  # fmt: skip
        if "feet" in x:
            r["feet"] = {s: np.round(v, 5).tolist() for s, v in x["feet"].items()}
        if ext and k in ext:
            r["host"] = ext[k]
        out.append(r)
    return out


def feet_moved_cm(rows: dict, f0: int) -> dict | None:
    """Largest horizontal displacement of each foot (ankle_roll_link) from the command row: stepping or sliding."""
    if "feet" not in rows[f0]:
        return None
    out = {}
    for s in ("left", "right"):
        p0 = np.asarray(rows[f0]["feet"][s][:2])
        out[s] = 100 * max(float(np.linalg.norm(np.asarray(rows[k]["feet"][s][:2]) - p0)) for k in rows if k >= f0)
    return out


def summarize_extra(ext: list[dict]) -> dict:
    """Host-specific per-step extras (dicts of scalars / None): max of numbers, first non-empty of the rest."""
    out = {}
    for k in ext[0]:
        vals = [e.get(k) for e in ext]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums:
            out[k + "_max"] = float(max(nums))
        else:
            nz = [v for v in vals if v]
            out[k] = nz[0] if nz else None
    return out
