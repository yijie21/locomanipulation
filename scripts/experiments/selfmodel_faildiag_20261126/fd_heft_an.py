"""selfmodel_faildiag_20261126 question 3 (measurement only): what the pelvis and the feet do in a HEFT reach, the same
measure on HumanoidArena's step logs (selfmodel_hacollect's heft_video/steps) and on LML's (fd_heft.py).

Per reach, from the command frame f0, in the heading frame at f0 (x ahead, y left):
  shift(t)     pelvis xy minus the midpoint of the two ankle_roll_links, minus its value at f0 (the pelvis moving over
               its feet); `fwd` = its x, `lat` = its y
  feet still   both ankles within 1 cm (xy) of their f0 place; contact = normal force >= 10 N
  lift-off     a foot under 10 N for 3 steps in a row (the first after f0)
  slip         a foot in contact whose xy moved > 1 cm since f0 before any lift-off
  ref          the reference pelvis's xy change since f0 (LML: the pushed frame through the buffer's transform; HA: the
               log's ref_x / ref_y), not delayed
At 1.0 / 2.0 / 2.6 s and at the first lift-off (or the end of the window, 6 s): fwd, lat, |ref|.
LML only (b): the reference's torso pitch (the reference pelvis's pitch + waist_pitch) and the leg joints (hip pitch,
knee, ankle pitch) at f0 and at the reference's end, and their change; the same for the robot.

    python fd_heft_an.py --ha <steps dir> --lml <fd_heft out dir> [--lml ...] --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

FPS = 50
# the pushed frame's 29 joints are in MuJoCo order (locomani.primitives.g1_joints.MUJOCO_JOINT_NAMES; L.DEFAULT_Q); fd_heft's
# column names refq_<name> (SCALEBFM order) were wrong labels: the columns are taken by position
MJ_NAMES = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint")  # fmt: skip
F_N, LIFT_RUN, STILL_M, WIN_S = 10.0, 3, 0.01, 6.0


def yaw_q(q):  # wxyz
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pitch_q(q):
    w, x, y, z = q
    return math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def measure(pel, yaw0, fl, fr, Fl, Fr, ref, f0, f_end_ref, n, u=None):
    """pel / fl / fr (T, 2) world xy, Fl / Fr (T,) N, ref (T, 2) world xy or None; window f0 .. f0 + n."""
    f1 = min(len(pel), f0 + n)
    Rt = rot2(yaw0).T
    mid = (fl + fr) / 2
    rel = (pel - mid) @ Rt.T
    sh = rel[f0:f1] - rel[f0]
    dfl = np.linalg.norm(fl[f0:f1] - fl[f0], axis=1)
    dfr = np.linalg.norm(fr[f0:f1] - fr[f0], axis=1)
    low = [(F[f0:f1] < F_N) for F in (Fl, Fr)]

    def first_run(b):
        r = 0
        for i, v in enumerate(b):
            r = r + 1 if v else 0
            if r >= LIFT_RUN:
                return i - LIFT_RUN + 1
        return None

    lo = [first_run(b) for b in low]
    lift = min([v for v in lo if v is not None], default=None)
    stop = lift if lift is not None else f1 - f0
    slip = None
    for i in range(stop):
        if (dfl[i] > STILL_M and Fl[f0 + i] >= F_N) or (dfr[i] > STILL_M and Fr[f0 + i] >= F_N):
            slip = i
            break
    still = next((i for i in range(f1 - f0) if dfl[i] > STILL_M or dfr[i] > STILL_M), None)
    pd = (pel[f0:f1] - pel[f0]) @ Rt.T
    rd = None if ref is None else (ref[f0:f1] - ref[f0]) @ Rt.T

    def at(i):
        i = min(i, f1 - f0 - 1)
        return dict(t=i / FPS, fwd_cm=100 * float(sh[i, 0]), lat_cm=100 * float(sh[i, 1]), pel_fwd_cm=100 * float(pd[i, 0]),
                    pel_lat_cm=100 * float(pd[i, 1]), ref_cm=None if rd is None else 100 * float(np.linalg.norm(rd[i])))  # fmt: skip

    fwd_first2 = next((i for i in range(f1 - f0) if sh[i, 0] >= 0.02), None)
    mag = np.linalg.norm(sh, axis=1)
    mag_first2 = next((i for i in range(f1 - f0) if mag[i] >= 0.02), None)
    extra = dict(mag2cm_s=None if mag_first2 is None else mag_first2 / FPS, max_mag_before_lift_cm=100 * float(mag[:max(stop, 1)].max()),
                 mag_2_6s_cm=100 * float(mag[min(int(2.6 * FPS), f1 - f0 - 1)]))  # fmt: skip
    if u is not None and np.linalg.norm(u) > 1e-6:
        u = np.asarray(u, float) / np.linalg.norm(u)
        al = sh @ u
        i_pk = int(np.argmax(mag[:max(stop, 1)]))
        extra.update(along_2_6s_cm=100 * float(al[min(int(2.6 * FPS), f1 - f0 - 1)]), max_along_before_lift_cm=100 * float(al[:max(stop, 1)].max()),
                     cos_at_peak=float(al[i_pk] / max(mag[i_pk], 1e-9)))  # fmt: skip
    return dict(lift_s=None if lift is None else lift / FPS, lift_foot=None if lift is None else ("left" if lo[0] == lift else "right"),
                slip_s=None if slip is None else slip / FPS, feet_still_s=(still if still is not None else f1 - f0) / FPS,
                fwd2cm_s=None if fwd_first2 is None else fwd_first2 / FPS,
                at_1s=at(FPS), at_2s=at(2 * FPS), at_2_6s=at(int(2.6 * FPS)), at_lift=at(stop - 1 if stop else 0),
                max_fwd_before_lift_cm=100 * float(sh[:max(stop, 1), 0].max()), min_load_n=float(min(Fl[f0:f1].min(), Fr[f0:f1].min())),
                ref_end_s=None if f_end_ref is None else (f_end_ref - f0) / FPS, end=at(f1 - f0 - 1), **extra)  # fmt: skip


def ha_file(f: Path, ev: Path) -> list[dict]:
    z = np.load(f)
    E = json.loads(ev.read_text())
    reach_f = [(e["f"], e["what"]) for e in E if str(e.get("what", "")).startswith("reach ")]
    out = []
    n = int(WIN_S * FPS)
    pel = np.c_[z["x"], z["y"]]
    fl, fr = np.c_[z["ankle_x_left"], z["ankle_y_left"]], np.c_[z["ankle_x_right"], z["ankle_y_right"]]
    ref = np.c_[z["ref_x"], z["ref_y"]]
    for k, (f0, what) in enumerate(reach_f):
        r = measure(pel, float(z["yaw"][f0]), fl, fr, z["load_left"], z["load_right"], ref, int(f0), None, n)
        out.append(dict(src="HA", file=f.name, reach=what, f0=int(f0), **r))
    return out


def lml_file(f: Path) -> list[dict]:
    z = np.load(f, allow_pickle=True)
    rows, cols = z["rows"], [str(c) for c in z["cols"]]
    meta = json.loads(str(z["meta"]))
    C = {c: i for i, c in enumerate(cols)}
    g = lambda *ks: rows[:, [C[k] for k in ks]]  # noqa: E731
    pel, fl, fr = g("pel_x", "pel_y"), g("fl_x", "fl_y"), g("fr_x", "fr_y")
    Fl, Fr = rows[:, C["force_l"]], rows[:, C["force_r"]]
    # the reference pelvis in the world: our frame -> world by the buffer's transform (yaw offset about src -> dst)
    rx, ry = rows[:, C["ref_0"]], rows[:, C["ref_1"]]
    yo, sx, sy, dx, dy = (rows[:, C[k]] for k in ("tr_yaw", "tr_src_x", "tr_src_y", "tr_dst_x", "tr_dst_y"))
    c, s = np.cos(yo), np.sin(yo)
    ref = np.c_[dx + c * (rx - sx) - s * (ry - sy), dy + s * (rx - sx) + c * (ry - sy)]
    ref = np.where(np.isfinite(ref), ref, np.nan)
    if meta["mode"] == "ha":
        reaches = [dict(f_cmd=meta["f_cmd"], f_end=meta["f_end"], mode=meta["side"], pelvis_z=meta["pelvis_z"], hands={meta["side"]: dict(cmd_target=meta["cmd_target"])})]
    else:
        reaches = meta["reaches"]
    out = []
    n = int(WIN_S * FPS)
    qcol = lambda name: C[f"q_{name}"]  # noqa: E731
    i_ref0 = next(i for i, c_ in enumerate(cols) if c_.startswith("refq"))
    rcol = lambda name: i_ref0 + MJ_NAMES.index(name)  # noqa: E731
    legs = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")]
    for k, rc in enumerate(reaches):
        f0 = int(rc["f_cmd"])
        if f0 >= len(rows):
            continue
        # f0 is the command's frame index; the log's row i is the state after step i (same index as the frames)
        yaw0 = yaw_q(rows[f0, [C["pel_qw"], C["pel_qx"], C["pel_qy"], C["pel_qz"]]])
        fe = int(rc["f_end"]) if rc.get("f_end") is not None else None
        tg = [np.asarray(v["cmd_target"], float)[:2] for v in rc["hands"].values() if v.get("cmd_target") is not None]
        u = np.mean(tg, axis=0) if tg else None  # the reach's horizontal direction (heading frame, from the pelvis)
        r = measure(pel, yaw0, fl, fr, Fl, Fr, ref, f0, fe, n, u)
        # (b): the reference's torso pitch and legs at the command and at the reference's end (as pushed, not delayed)
        b = {}
        if fe is not None and fe < len(rows):
            for nm, fi in (("cmd", f0 + 1), ("ref_end", fe)):
                rq = rows[fi, [C["ref_3"], C["ref_4"], C["ref_5"], C["ref_6"]]]
                b[nm] = dict(ref_pelvis_pitch_deg=math.degrees(pitch_q(rq)), ref_waist_pitch_deg=math.degrees(rows[fi, rcol("waist_pitch_joint")]),
                             ref_legs_rad={j: float(rows[fi, rcol(j)]) for j in legs}, ref_pelvis_z=float(rows[fi, C["ref_2"]]),
                             robot_pelvis_pitch_deg=math.degrees(pitch_q(rows[fi, [C["pel_qw"], C["pel_qx"], C["pel_qy"], C["pel_qz"]]])),
                             robot_waist_pitch_deg=math.degrees(rows[fi, qcol("waist_pitch_joint")]), robot_legs_rad={j: float(rows[fi, qcol(j)]) for j in legs})  # fmt: skip
            b["d_ref_torso_pitch_deg"] = (b["ref_end"]["ref_pelvis_pitch_deg"] + b["ref_end"]["ref_waist_pitch_deg"]) - (b["cmd"]["ref_pelvis_pitch_deg"] + b["cmd"]["ref_waist_pitch_deg"])
            b["d_ref_legs_rad"] = {j: b["ref_end"]["ref_legs_rad"][j] - b["cmd"]["ref_legs_rad"][j] for j in legs}
            b["d_ref_pelvis_z_cm"] = 100 * (b["ref_end"]["ref_pelvis_z"] - b["cmd"]["ref_pelvis_z"])
        tgt = {s: v.get("cmd_target") for s, v in rc["hands"].items()}
        out.append(dict(src="LML", file=f.name, mode=meta["mode"], reach=k, f0=f0, walked=meta.get("walked", False), lost=meta.get("lost"),
                        hands=rc["mode"], pelvis_z=rc.get("pelvis_z"), target_h=tgt, reach_s=rc.get("reach_s", meta.get("reach_s")), pose=b, **r))  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ha", default="")
    ap.add_argument("--lml", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    if a.ha:
        for f in sorted(Path(a.ha).glob("heft_*.npz")):
            res += ha_file(f, f.with_name(f.stem + "_events.json"))
    for d in a.lml:
        for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
            res += lml_file(Path(f))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for r in res:
        s = r["at_2_6s"]
        print(f"{r['src']:3s} {r['file'][:34]:34s} {str(r['reach'])[:28]:28s} lift {r['lift_s']} slip {r['slip_s']} feet still {r['feet_still_s']:.2f} s; "
              f"2.6 s fwd {s['fwd_cm']:+.1f} lat {s['lat_cm']:+.1f} ref {s['ref_cm'] if s['ref_cm'] is None else round(s['ref_cm'], 1)} cm; "
              f"at lift fwd {r['at_lift']['fwd_cm']:+.1f} cm; max fwd before lift {r['max_fwd_before_lift_cm']:+.1f}; min load {r['min_load_n']:.0f} N")  # fmt: skip


if __name__ == "__main__":
    main()
