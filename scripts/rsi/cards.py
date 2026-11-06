"""Failure cards: what the Reviewer gets of the incumbent's diagnostic rounds (TODO 43; rules.md 32).

    python scripts/rsi/cards.py <diag round dir> <scene file or dir> --out <dir>    # cards.md, cards.jsonl, img/<label>.png

Offline: reads the round .json/.npz and the scene file, never steps a sim. `loop.py` writes them to iter_NN/cards after
the incumbent's diagnosis. cards.md opens with the pool summary (failure classes, first failing steps, success by task
and by object feature), then one card per failed round: identity (task, object features), outcome and the first failing
step with its check facts, the quantities at the failure moment, and one contrast with the nearest successful round.
Each card names a 2-panel schematic. Deterministic: no randomness, fixed fonts/dpi, no timestamps in the png.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from locomani.primitives.navigator import build as walk_map  # noqa: E402

JOINTS = ['left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
          'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
          'waist_yaw', 'waist_roll', 'waist_pitch',
          'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
          'left_wrist_pitch', 'left_wrist_yaw',
          'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll',
          'right_wrist_pitch', 'right_wrist_yaw']  # fmt: skip
SKIP_STEPS = {"settle", "hold"}
EDGE_N = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}  # a surface's edges, in its frame
JAW_LEAD_M = 0.1515  # jaw centre ahead of the wrist along the approach (the Reviewers' own constant)
CORRIDOR_HALF_W = 0.07
CORRIDOR_LEN = 0.45
FUR_MAX_HALF = 3.0  # skip the apartment shell
KNOCK_CM = 1.0


# ----------------------------------------------------------------------------------------------- scene geometry
def quat_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts) < 3:
        return np.array(pts)
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


class Scene:
    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            path = path / "scene_meta.json"
        self.meta = json.loads(path.read_text())
        self.objs = self.meta["objects"]
        self.surfaces = self.meta["surfaces"]
        self.fp = {}
        for name, o in self.objs.items():
            b = o.get("box") or {}
            he = np.asarray(b.get("half_extent") or (0, 0, 0), float)
            if not he.any() or he.max() > FUR_MAX_HALF or o.get("init_pos") is None:
                continue
            R = quat_R(o.get("init_quat") or (1, 0, 0, 0))
            c = np.asarray(b.get("centre") or (0, 0, 0), float)
            sg = np.array([[a, b_, c_] for a in (-1, 1) for b_ in (-1, 1) for c_ in (-1, 1)], float)
            cor = np.asarray(o["init_pos"], float) + (c + sg * he) @ R.T
            self.fp[name] = dict(name=name, poly=hull(cor[:, :2]), z0=float(cor[:, 2].min()), z1=float(cor[:, 2].max()),
                                 free=bool(o.get("free")), container=bool(o.get("container")),
                                 centre=cor.mean(0)[:3])  # fmt: skip

    def support_at(self, p, bottom_z):
        """The surface under the object's start (xy inside its rectangle, top within 8 cm below the object's bottom); None = floor."""
        best = None
        for nm, sf in self.surfaces.items():
            yw = sf.get("yaw", 0.0)
            d = np.asarray(p[:2]) - np.asarray(sf["xy"])
            lx, ly = math.cos(yw) * d[0] + math.sin(yw) * d[1], -math.sin(yw) * d[0] + math.cos(yw) * d[1]
            h = sf["half_extent"]
            if (
                abs(lx) <= h[0] + 0.02
                and abs(ly) <= h[1] + 0.02
                and -0.03 <= bottom_z - sf["top"] <= 0.08
                and sf["top"] > 0.05
            ):
                if best is None or sf["top"] > best["top"]:
                    best = dict(sf, _name=nm)
        return best

    def occ(self):
        """The walk map the runner plans on (navigator.build, static furniture)."""
        if not hasattr(self, "_occ"):
            self._occ = walk_map(self.meta, body_radius=0.35)[0]
        return self._occ

    def edges(self, sf, p0, pel, skip=None):
        """Edges of the support (its frame, flat top): depth from the object, a stand point 0.45 m out from the edge
        under the object, free on the walk map and outside every scene footprint, listed in approach_edges."""
        flat = np.asarray(sf.get("flat_half_extent") or sf["half_extent"], float)[:2]
        yw = sf.get("yaw", 0.0)
        R2 = np.array([[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]])
        loc = R2.T @ (np.asarray(p0[:2]) - np.asarray(sf["xy"]))
        out = []
        for nm, nv in EDGE_N.items():
            ax = 0 if nm[1] == "x" else 1
            sg = 1 if nm[0] == "+" else -1
            depth = flat[ax] - sg * loc[ax]
            q = loc.copy()
            q[ax] = sg * (flat[ax] + 0.45)
            stand = np.asarray(sf["xy"]) + R2 @ q
            free, blk = bool(self.occ().is_free(stand)), None
            if free:
                for f in self.fp.values():
                    if f["name"] in (skip,) or f["free"] or f["z1"] < 0.1 or f["z0"] > 1.2 or len(f["poly"]) < 3:
                        continue
                    if poly_dist(f["poly"], stand) < 0.0:
                        free, blk = False, f["name"]
                        break
            a0 = np.array(nv, float)
            ctr = np.asarray(sf["xy"]) + R2 @ (loc * (1 - abs(a0)) + a0 * flat)  # the edge's foot of the object
            out.append(
                dict(
                    edge=nm,
                    depth_cm=round(100 * float(depth), 1),
                    free=free,
                    blocker=blk,
                    listed=nm in (sf.get("approach_edges") or EDGE_N),
                    stand=stand,
                    foot=ctr,
                    used_d=float(np.linalg.norm(np.asarray(pel) - stand)),
                )
            )
        used = min(out, key=lambda e: e["used_d"])
        for e in out:
            e["used"] = e is used
        return out

    def obj_name(self, key: str) -> str:
        return key[4:-7] if key.endswith("_object") else key[4:]


def poly_dist(poly, p):
    """Distance from point p to polygon (0 inside)."""
    if len(poly) < 3:
        return float("inf")
    if MPath(poly).contains_point(p):
        return 0.0
    d = float("inf")
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) or 1.0), 0, 1)
        d = min(d, float(np.linalg.norm(p - (a + t * ab))))
    return d


def corridor_scan(scene: Scene, name: str, p0, u, z_bot, z_top, skip=None):
    """Bodies reaching into the hand corridor behind the object (strip of half width CORRIDOR_HALF_W, length
    CORRIDOR_LEN along -u from the object centre) at the hand's height band. Returns [(name, gap_cm, z0, z1)] by gap."""
    n = np.array([-u[1], u[0]])
    ss = np.arange(0.0, CORRIDOR_LEN + 1e-9, 0.02)
    ws = np.arange(-CORRIDOR_HALF_W, CORRIDOR_HALF_W + 1e-9, 0.02)
    pts = np.array([p0 - u * s + n * w for s in ss for w in ws])
    sv = np.array([s for s in ss for _ in ws])
    out = []
    for k, f in scene.fp.items():
        if k == name or k == skip or f["z1"] < z_bot + 0.02 or f["z0"] > z_top + 0.12:
            continue
        if f["poly"].shape[0] < 3:
            continue
        m = MPath(f["poly"]).contains_points(pts)
        if m.any():
            out.append((k, round(100 * float(sv[m].min()), 1), round(f["z0"], 2), round(f["z1"], 2)))
    return sorted(out, key=lambda r: r[1])


# ----------------------------------------------------------------------------------------------- round analysis
def first_fail(r):
    for s in r.get("steps") or []:
        if not s.get("ok"):
            return s
    return None


def scalars(s, skip=("step", "ok", "why", "t0", "t1")):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()
            if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)}  # fmt: skip


def shape(why):
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", why or "")[:70]


def yaw_pitch_roll(q):
    R = quat_R(q)
    return (math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(math.asin(max(-1, min(1, -R[2, 0])))),
            math.degrees(math.atan2(R[2, 1], R[2, 2])))  # fmt: skip


def goal_lookup():
    ids = {}
    for f in sorted((HERE / "families").glob("*_tasks.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                ids.setdefault(x["id"], x)
    return ids


def target_key(z, task):
    keys = [k for k in z.files if k.startswith("obj_")]
    for k in keys:
        if re.search(r"(^|_)" + re.escape(k[4:-7]) + r"(_|$)", task):
            return k
    return keys[0] if keys else None


def analyse(r, z, scene: Scene, goal):
    a = {"label": r["label"], "task": r["task"], "ok": bool(r["ok"]), "class": r.get("failure_class") or "",
         "fell": bool(r.get("fell")), "dropped": bool(r.get("dropped")), "goal": " and ".join(goal.get("goal", []))}  # fmt: skip
    steps = r.get("steps") or []
    st = {}
    for s in steps:
        st.setdefault(s["step"], s)
    ff = first_fail(r)
    a["first_fail"] = dict(step=ff["step"], why=ff.get("why", ""), t0=ff["t0"], t1=ff["t1"], facts=scalars(ff)) if ff else \
        dict(step="goal", why=r.get("why") or r.get("plan_why") or "", t0=r.get("t_end", 0), t1=r.get("t_end", 0), facts={})  # fmt: skip
    a["soft_flags"] = [f"{s['step']}: {s['why']}" for s in steps if s.get("ok") and s.get("why")]
    a["t_end"] = r.get("t_end", 0)
    a["timeline"] = [
        (s["step"], round(s["t0"]), round(s["t1"]), bool(s.get("ok"))) for s in steps if s["step"] not in SKIP_STEPS
    ]
    if z is None:
        return a
    t = z["t"].astype(float)
    idx = lambda x: int(min(max(np.searchsorted(t, x), 0), len(t) - 1))  # noqa: E731
    rob = z["robot"].astype(float)
    key = target_key(z, r["task"])
    name = scene.obj_name(key) if key else None
    o = scene.objs.get(name or "", {})
    # failure moment: the first failing step's end; a round whose steps all passed but whose goal failed: the end of its last real step
    real = [s for s in steps if s["step"] not in SKIP_STEPS]
    phase = ff or (real[-1] if real else None)
    a["phase"] = phase["step"] if phase else "goal"
    a["phase_t"] = [phase["t0"], phase["t1"]] if phase else [0.0, t[-1]]
    tf = phase["t1"] if phase else r.get("t_end", t[-1])
    fall_i = np.where(rob[:, 2] < 0.45)[0]
    fall_t = float(t[fall_i[0]]) if len(fall_i) else None
    if fall_t is not None and r.get("fell"):
        tf = min(tf, fall_t)
        a["fall_t"] = round(fall_t, 1)
    tf = float(min(tf, t[-1]))
    i_f = idx(tf)
    a["t_fail"] = round(tf, 1)
    # hand side from the motor calls
    side = 1
    for c in r.get("calls") or []:
        if c.get("name") in ("reach", "grip") and c.get("args") and c["args"][0] in ("left", "right"):
            side = 0 if c["args"][0] == "left" else 1
            break
    a["hand"] = "left" if side == 0 else "right"
    W, RW = z["wrists"][:, side].astype(float), z["ref_wrists"][:, side].astype(float)
    # object facts and trace
    if key:
        P = z[key].astype(float)
        i0 = idx(1.0)
        p0 = P[i0, :3]
        a["obj"] = dict(name=name, start=[round(v, 3) for v in p0], size_cm=[round(200 * v, 1) for v in (o.get("box") or {}).get("half_extent", [0, 0, 0])],  # fmt: skip
                        mass_kg=(o.get("semantic") or {}).get("mass_kg"), container=bool(o.get("container")), wall_cm=round(100 * (o.get("box") or {}).get("wall", 0) , 1) if o.get("container") else None)  # fmt: skip
        sf = scene.support_at(p0, float(p0[2] - (o.get("box") or {}).get("half_extent", [0, 0, 0])[2]))
        a["obj"]["support"] = dict(name=sf["_name"], furniture=sf.get("of_object"), top=sf.get("top")) if sf else None  # fmt: skip
        Rt = quat_R(P[i0, 3:7])
        cen = np.asarray((o.get("box") or {}).get("centre") or (0, 0, 0), float)
        hev = np.asarray((o.get("box") or {}).get("half_extent") or (0, 0, 0), float)
        sg = np.array([[x_, y_, z_] for x_ in (-1, 1) for y_ in (-1, 1) for z_ in (-1, 1)], float)
        cor_ = p0 + (cen + sg * hev) @ Rt.T
        tpoly, tz0, tz1 = hull(cor_[:, :2]), float(cor_[:, 2].min()), float(cor_[:, 2].max())
        dims = sorted(200 * np.asarray((o.get("box") or {}).get("half_extent", [0, 0, 0])))
        h = 200 * (o.get("box") or {}).get("half_extent", [0, 0, 0])[2]
        a["obj"]["shape"] = ("container" if o.get("container") else "flat" if dims[0] < 3.0 else "tall_narrow" if h > 2 * max(dims[1], 1e-3) and h > 12 else "wide" if dims[2] > 20 else "compact")  # fmt: skip
        top_z, bot_z = tz1, tz0
        cover = [(f["name"], round(f["z0"], 2)) for f in scene.fp.values() if not f["free"] and f["z0"] > top_z + 0.02 and f["name"] != name and len(f["poly"]) > 2 and MPath(f["poly"]).contains_point(p0[:2])]  # fmt: skip
        a["obj"]["cover_above"] = min(cover, key=lambda c: c[1]) if cover else None
        # approach axis from stance pelvis to the object
        pel = (st.get("stance") or {}).get("pelvis")
        pel = np.asarray(pel[:2], float) if pel else rob[idx((st.get("walk") or {}).get("t1", 1.0)), :2]
        u = p0[:2] - pel
        u = u / (np.linalg.norm(u) or 1.0)
        n = np.array([-u[1], u[0]])
        a["stance_pelvis"] = [round(float(v), 3) for v in pel]
        a["u"] = [round(float(v), 3) for v in u]
        # trace of the object
        d_obj = P[:, :2] - p0[:2]
        mv = np.linalg.norm(d_obj, axis=1)
        gs = st.get("grip")
        t_g = gs["t0"] if gs else float("inf")
        k = np.where((mv > KNOCK_CM / 100) & (t >= 1.0))[0]
        R_f = quat_R(P[i_f, 3:7])
        slip = None
        if gs and gs.get("ok") and idx(gs["t1"]) < i_f:  # distance object-to-wrist change since the grasp closed
            dd = np.linalg.norm(P[:, :3] - W, axis=1)
            slip = round(100 * float(dd[i_f] - dd[idx(gs["t1"])]), 1)
        a["objfail"] = dict(moved_cm=round(100 * float(mv[i_f]), 1), lift_cm=round(100 * float(P[i_f, 2] - p0[2]), 1), max_lift_cm=round(100 * float(P[:, 2].max() - p0[2]), 1),  # fmt: skip
                            tilt_deg=round(math.degrees(math.acos(max(-1, min(1, R_f[2, 2])))), 1), slip_cm=slip,
                            first_move_t=round(float(t[k[0]]), 1) if len(k) else None,
                            knocked_before_grip_cm=round(100 * float(mv[idx(t_g)]), 1) if math.isfinite(t_g) and mv[idx(t_g)] > KNOCK_CM / 100 else None)  # fmt: skip
        # hand vs object at the failure moment, in the approach frame
        jaw = W[i_f].copy()
        jaw[:2] += JAW_LEAD_M * u
        off = P[i_f, :3] - jaw
        a["hand_vs_obj"] = dict(object_ahead_of_jaw_cm=round(100 * float(off[:2] @ u), 1), object_lateral_cm=round(100 * float(off[:2] @ n), 1),  # fmt: skip
                                jaw_z_minus_obj_z_cm=round(-100 * float(off[2]), 1), obj_top_minus_jaw_z_cm=round(100 * float(top_z - jaw[2]), 1))  # fmt: skip
        a["edges"] = (
            [
                {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in e.items()}
                for e in scene.edges(sf, p0, pel, sf.get("of_object"))
            ]
            if sf
            else []
        )
        # finger travel over the grip: opening = 200 * finger + 4.6 cm (fits the recorded opening_cm); jaw = the command
        if gs:
            fi = slice(2 * side, 2 * side + 2)
            ia, ib = idx(gs["t0"]), idx(gs["t1"])
            op = lambda i: round(200 * float(z["fingers"][i, fi].mean()) + 4.6, 1)  # noqa: E731
            wid = float(np.ptp((tpoly - p0[:2]) @ n)) * 100
            a["fingers"] = dict(open_cm=op(ia), reached_cm=op(ib), cmd_open_cm=round(200 * float(z["jaw"][ia, side]) + 4.6, 1), cmd_end_cm=round(200 * float(z["jaw"][ib, side]) + 4.6, 1),
                                width_cm=round(wid, 1), stop_minus_width_cm=round(op(ib) - wid, 1), grip_ok=bool(gs.get("ok")), t=[round(gs["t0"], 1), round(gs["t1"], 1)])  # fmt: skip
        a["corridor"] = corridor_scan(scene, name, p0[:2], u, bot_z, top_z, (sf or {}).get("of_object"))[:3]
        a["neighbours"] = sorted([(f["name"], round(100 * poly_dist(f["poly"], p0[:2]), 1), round(f["z1"] - top_z, 2) if False else round(f["z1"], 2), "free" if f["free"] else "fixed")  # fmt: skip
                                  for f in scene.fp.values() if f["name"] != name and f["name"] != (sf or {}).get("of_object") and poly_dist(f["poly"], p0[:2]) < 0.40 and f["z1"] > bot_z + 0.02 and f["z0"] < top_z + 0.4], key=lambda x: x[1])[:3]  # fmt: skip
        a["_geom"] = dict(p0=p0, u=u, n=n, top_z=top_z, bot_z=bot_z, name=name, key=key, poly=tpoly)
    # wrist vs reference
    w0, w1 = a["phase_t"]
    i0, i1 = idx(w0), max(idx(w1), idx(w0) + 1)
    ok_ref = np.linalg.norm(RW, axis=1) > 1e-6
    e = np.linalg.norm(W - RW, axis=1)
    seg = e[i0 : i1 + 1][ok_ref[i0 : i1 + 1]]
    a["wrist_ref"] = dict(err_at_fail_cm=round(100 * float(e[i_f]), 1) if ok_ref[i_f] else None,
                          err_max_in_step_cm=round(100 * float(seg.max()), 1) if len(seg) else None,
                          err_mean_in_step_cm=round(100 * float(seg.mean()), 1) if len(seg) else None)  # fmt: skip
    if "_geom" in a and len(seg):
        g = a["_geom"]
        lat_w = (W[i0 : i1 + 1, :2] - g["p0"][:2]) @ g["n"]
        lat_r = ((RW[i0 : i1 + 1, :2] - g["p0"][:2]) @ g["n"])[ok_ref[i0 : i1 + 1]]
        a["wrist_ref"]["ref_swing_cm"] = round(100 * float(np.ptp(lat_r)), 1) if len(lat_r) else None
        a["wrist_ref"]["wrist_swing_cm"] = round(100 * float(np.ptp(lat_w)), 1)
    # body at the failure moment
    yaw, pit, rol = yaw_pitch_roll(rob[i_f, 3:7])
    wk = st.get("walk")
    iw = idx(wk["t1"]) if wk else idx(1.0)
    _, pit_w, _ = yaw_pitch_roll(rob[iw, 3:7])
    q = z["joints"][i_f].astype(float)
    jd = lambda nm: round(math.degrees(float(q[JOINTS.index(nm)])), 1)  # noqa: E731
    a["body"] = dict(pelvis_z=round(float(rob[i_f, 2]), 2), pitch_deg=round(pit, 1), pitch_after_walk_deg=round(pit_w, 1), roll_deg=round(rol, 1),  # fmt: skip
                     waist_pitch_deg=jd("waist_pitch"), waist_roll_deg=jd("waist_roll"), waist_yaw_deg=jd("waist_yaw"),
                     shoulder_pitch_deg=jd(a["hand"] + "_shoulder_pitch"), elbow_deg=jd(a["hand"] + "_elbow"),
                     pelvis_off_stance_cm=round(100 * float(np.linalg.norm(rob[i_f, :2] - np.asarray(a["stance_pelvis"]))), 1) if "stance_pelvis" in a else None)  # fmt: skip
    prim = z["primitives"][int(z["primitive"][i_f])] if "primitives" in z.files and len(z["primitives"]) else "?"
    a["controller"] = dict(primitive=str(prim), mode=int(z["mode"][i_f]))
    a["start_to_stance_m"] = round(float(np.linalg.norm(rob[0, :2] - np.asarray(a["stance_pelvis"]))), 1) if "stance_pelvis" in a else None  # fmt: skip
    a["_arr"] = dict(t=t, rob=rob, W=W, RW=RW, ok_ref=ok_ref, i_f=i_f, iw=iw)
    return a


# ----------------------------------------------------------------------------------------------- contrast
STEP_KEYS = ("jaw_err_cm", "attitude_err_deg", "ik_err_cm", "pred_p_miss")


def contrast(a, pool):
    cands = [b for b in pool if b["ok"] and b["task"] == a["task"]]
    how = "same object"
    if not cands:
        sh = (a.get("obj") or {}).get("shape")
        cands = [b for b in pool if b["ok"] and (b.get("obj") or {}).get("shape") == sh and sh]
        how = f"same shape class ({sh})"
    if not cands:
        return None

    def dist(b):
        sa, sb = a["_steps"].get("stance", {}), b["_steps"].get("stance", {})
        return abs(sa.get("normal_angle_deg", 0) - sb.get("normal_angle_deg", 0)) / 10 + abs((a["start_to_stance_m"] or 0) - (b["start_to_stance_m"] or 0)) / 3 + (0 if sa.get("kind") == sb.get("kind") else 1)  # fmt: skip

    b = min(cands, key=lambda b: (dist(b), b["label"]))
    sa, sb = a["_steps"], b["_steps"]
    rows = []

    def add(nm, va, vb):
        if va is None and vb is None:
            return
        rows.append((nm, va, vb))

    add("stance kind", sa.get("stance", {}).get("kind"), sb.get("stance", {}).get("kind"))
    for k in ("normal_angle_deg", "shoulder_to_obj"):
        add("stance." + k, sa.get("stance", {}).get(k), sb.get("stance", {}).get(k))
    add("approach_deg", sa.get("geom", {}).get("approach_deg"), sb.get("geom", {}).get("approach_deg"))
    add("walk stop_err_cm", sa.get("walk", {}).get("stop_err_cm"), sb.get("walk", {}).get("stop_err_cm"))
    add("walk end t", sa.get("walk", {}).get("t1"), sb.get("walk", {}).get("t1"))
    fs = a["phase"]
    if fs in sb:
        for k in STEP_KEYS:
            add(f"{fs}.{k}", sa.get(fs, {}).get(k), sb[fs].get(k))
        add(f"{fs} duration s", round(a["phase_t"][1] - a["phase_t"][0], 1), round(sb[fs]["t1"] - sb[fs]["t0"], 1))  # fmt: skip
    else:
        rows.append((f"{fs}", "failed", "not reached/absent"))
    add("wrist ref err max cm (failing step window)", a.get("wrist_ref", {}).get("err_max_in_step_cm"), _okerr(b, fs))
    return dict(ok_label=b["label"], how=how, rows=[(n, _r(x), _r(y)) for n, x, y in rows])


def _r(v):
    return round(v, 2) if isinstance(v, float) else v


def _okerr(b, step):
    if "_arr" not in b or step not in b["_steps"]:
        return None
    s = b["_steps"][step]
    A = b["_arr"]
    t = A["t"]
    i0, i1 = int(np.searchsorted(t, s["t0"])), int(np.searchsorted(t, s["t1"]))
    m = A["ok_ref"][i0 : i1 + 1]
    e = np.linalg.norm(A["W"] - A["RW"], axis=1)[i0 : i1 + 1][m]
    return round(100 * float(e.max()), 1) if len(e) else None


# ----------------------------------------------------------------------------------------------- image
def draw(a, scene: Scene, path: Path):
    g, A = a["_geom"], a["_arr"]
    p0, u, n = g["p0"][:2], g["u"], g["n"]
    t, rob, W, RW, i_f = A["t"], A["rob"], A["W"], A["RW"], A["i_f"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    pel = np.asarray(a["stance_pelvis"])
    sel = np.arange(0, i_f + 1, 4)
    iw0 = min(A["iw"], int(np.searchsorted(t, t[i_f] - 6.0)))
    pts = np.vstack([p0[None], pel[None], rob[iw0 : i_f + 1 : 4, :2], W[iw0 : i_f + 1 : 4, :2]])
    lo, hi = pts.min(0) - 0.55, pts.max(0) + 0.55
    c, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1.0)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    sf = scene.support_at(g["p0"], g["bot_z"])
    if sf:
        h = np.asarray(sf["half_extent"])
        yw = sf.get("yaw", 0.0)
        R2 = np.array([[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]])
        cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * h
        ax.add_patch(MPoly(np.asarray(sf["xy"]) + cs @ R2.T, fc="#e8d9a8", ec="#a08a3c", lw=0.8, zorder=1))
        fl = np.asarray(sf.get("flat_half_extent") or sf["half_extent"], float)[:2]
        for e in a.get("edges", []):
            nv = np.array(EDGE_N[e["edge"]])
            tg = np.array([-nv[1], nv[0]])
            c0 = np.asarray(sf["xy"]) + R2 @ (nv * fl)
            L2 = R2 @ (tg * fl)
            col = "#1a9850" if e["free"] else "#d73027"
            ax.plot(
                *zip(c0 - L2, c0 + L2),
                "-",
                color=col,
                lw=3.5 if e["used"] else 2.0,
                alpha=1.0 if e["listed"] else 0.4,
                zorder=3,
            )
            ax.plot(*e["stand"], "v" if e["used"] else ".", color=col, ms=6, zorder=3)
    nearest = []
    for f in scene.fp.values():
        if f["name"] == g["name"] or len(f["poly"]) < 3:
            continue
        if f["poly"].min(0).max() < -1e9:
            continue
        if (f["poly"][:, 0].max() < c[0] - half or f["poly"][:, 0].min() > c[0] + half or f["poly"][:, 1].max() < c[1] - half or f["poly"][:, 1].min() > c[1] + half):  # fmt: skip
            continue
        above = f["z0"] > g["top_z"] + 0.02 and not f["free"]
        ax.add_patch(MPoly(f["poly"], fc="#6b6b6b" if f["free"] else ("none" if above else "#c9c9c9"), ec="#555" if not above else "#b5381f", lw=0.8, hatch="//" if above else None, alpha=0.9, zorder=2))  # fmt: skip
        nearest.append((poly_dist(f["poly"], p0), f))
    for d, f in sorted(nearest, key=lambda x: x[0])[:5]:
        ctr = f["poly"].mean(0)
        if abs(ctr[0] - c[0]) < half and abs(ctr[1] - c[1]) < half:
            ax.text(ctr[0], ctr[1], f["name"][:14], fontsize=5.5, ha="center", va="center", zorder=6)
    # hand corridor
    s_len = CORRIDOR_LEN
    cor = np.array([p0 + n * CORRIDOR_HALF_W, p0 - n * CORRIDOR_HALF_W, p0 - n * CORRIDOR_HALF_W - u * s_len, p0 + n * CORRIDOR_HALF_W - u * s_len])  # fmt: skip
    ax.add_patch(MPoly(cor, fc="#ff9d3a", alpha=0.35, ec="#e07b00", lw=0.6, zorder=3))
    ax.plot(*zip(pel, p0 + u * 0.1), "k--", lw=0.8, zorder=4)
    # object
    obj = g
    ax.add_patch(MPoly(g["poly"], fc="#d62728", ec="k", lw=0.8, zorder=5))
    # robot path, stance, heading
    ax.plot(rob[sel, 0], rob[sel, 1], "-", color="#1f77b4", lw=1.0, zorder=4)
    ax.plot(*rob[0, :2], "o", color="#1f77b4", ms=3, zorder=4)
    ax.plot(*pel, "o", mfc="none", mec="#1f77b4", ms=8, zorder=5)
    yaw = yaw_pitch_roll(rob[A["iw"], 3:7])[0]
    ax.arrow(rob[A["iw"], 0], rob[A["iw"], 1], 0.3 * math.cos(math.radians(yaw)), 0.3 * math.sin(math.radians(yaw)), head_width=0.07, color="#1f77b4", zorder=5)  # fmt: skip
    ax.plot(rob[i_f, 0], rob[i_f, 1], "s", color="#1f77b4", ms=5, zorder=6)
    ok = A["ok_ref"][sel]
    iw = A["iw"]
    iw = iw0
    ws = np.arange(iw, i_f + 1, 3)
    ax.plot(W[ws, 0], W[ws, 1], "-", color="#2ca02c", lw=1.3, zorder=6)
    rws = ws[A["ok_ref"][ws]]
    ax.plot(RW[rws, 0], RW[rws, 1], "--", color="#2ca02c", lw=1.0, zorder=6)
    ax.plot(*W[i_f, :2], "x", color="k", ms=8, mew=2, zorder=7)
    ax.set_title(f"top-down  {a['label']}  [{a['class'] or 'fail'}]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.4)
    # side panel: along-approach coordinate s (object at 0), height z
    sc_ = lambda P: (P[:, :2] - p0) @ u  # noqa: E731
    top, bot = g["top_z"], g["bot_z"]
    hd = (g["poly"] - p0) @ u
    bx.add_patch(plt.Rectangle((hd.min(), bot), hd.max() - hd.min(), top - bot, fc="#d62728", ec="k", lw=0.8, zorder=3))
    if sf:
        bx.axhline(sf["top"], color="#a08a3c", lw=1.2, zorder=1)
        bx.text(
            0.02,
            sf["top"],
            " support top",
            transform=bx.get_yaxis_transform(),
            fontsize=6,
            va="bottom",
            color="#7a6a2a",
        )
    for name_, gap, z0, z1 in a.get("corridor", []):
        f = scene.fp[name_]
        s_ = (f["poly"] - p0) @ u
        bx.add_patch(
            plt.Rectangle(
                (s_.min(), z0), s_.max() - s_.min(), z1 - z0, fc="none", ec="#e07b00", hatch="xx", lw=0.8, zorder=2
            )
        )
        bx.text(s_.mean(), (z0 + z1) / 2, name_[:12], fontsize=5.5, ha="center", va="center")
    cv = a["obj"].get("cover_above")
    if cv:
        bx.axhline(cv[1], color="#b5381f", ls="-.", lw=1.0)
        bx.text(
            0.02,
            cv[1],
            f" cover {cv[0][:14]} z={cv[1]:.2f}",
            transform=bx.get_yaxis_transform(),
            fontsize=6,
            va="bottom",
            color="#b5381f",
        )
    ws2 = np.arange(iw, min(i_f + 40, len(t)), 3)
    bx.plot(sc_(W[ws2]), W[ws2, 2], "-", color="#2ca02c", lw=1.3, label="wrist (actual)")
    rws2 = ws2[A["ok_ref"][ws2]]
    bx.plot(sc_(RW[rws2]), RW[rws2, 2], "--", color="#2ca02c", lw=1.0, label="wrist reference")
    bx.plot(sc_(W[i_f : i_f + 1]), W[i_f : i_f + 1, 2], "x", color="k", ms=9, mew=2, label=f"failure t={a['t_fail']}s")
    bx.set_xlim(-0.75, 0.3)
    zs = np.concatenate([W[ws2, 2], [top, bot]])
    bx.set_ylim(max(zs.min() - 0.12, 0), zs.max() + 0.25)
    bx.set_xlabel("along approach from object centre (m)", fontsize=6)
    bx.set_ylabel("z (m)", fontsize=6)
    bx.set_title(f"side view: {a['hand']} wrist vs reference ({a['phase']})", fontsize=7)
    bx.tick_params(labelsize=6)
    bx.grid(alpha=0.25, lw=0.4)
    bx.legend(fontsize=5.5, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=75, metadata={"Software": None})
    plt.close(fig)
    try:  # palette-quantise to stay far below 100 KB, deterministic
        from PIL import Image

        im = Image.open(path).convert("RGB").quantize(colors=48, method=Image.Quantize.MEDIANCUT)
        im.save(path, optimize=True)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------- text
def fmt(v, unit=""):
    return "n/a" if v is None else f"{v}{unit}"


def card_text(a, ct):
    o = a.get("obj") or {}
    ff = a["first_fail"]
    sup = o.get("support")
    L = [
        f"### {a['label']}  [{a['class'] or 'unclassified'}]{' FELL' if a['fell'] else ''}{' DROPPED' if a['dropped'] else ''}"
    ]
    sz = " x ".join(str(x) for x in o.get("size_cm", []))
    L.append(f"- Task {a['goal'] or a['task']}: {o.get('name')} box {sz} cm ({o.get('shape')}), {fmt(o.get('mass_kg'), ' kg')}, "
             + (f"on {sup['furniture'] or sup['name']} top {sup['top']:.3f} m" if sup else "on the floor")
             + (f", container wall {o['wall_cm']} cm" if o.get("container") else "")
             + (f", cover above: {o['cover_above'][0]} at {o['cover_above'][1]:.2f} m" if o.get("cover_above") else ", no cover above")
             + f"; hand {a['hand']}")  # fmt: skip
    fc = ", ".join(f"{k} {v}" for k, v in list(ff["facts"].items())[:7])
    L.append(f"- Outcome FAIL at t={a.get('t_fail')} s of {a['t_end']:.0f} s. First failing step: {ff['step']} [{ff['t0']:.1f}-{ff['t1']:.1f}]: {ff['why'] or 'goal not met'}"
             + (f" ({fc})" if fc else "") + (f". Soft flags: {'; '.join(a['soft_flags'])}" if a["soft_flags"] else ""))  # fmt: skip
    if "wrist_ref" in a:
        sa = a["_steps"]
        _f = lambda d: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
        stn, gm, wk = _f(sa.get("stance", {})), _f(sa.get("geom", {})), _f(sa.get("walk", {}))
        L.append(f"- Stance: kind {stn.get('kind')}, normal angle {stn.get('normal_angle_deg')} deg, shoulder-to-object {stn.get('shoulder_to_obj')} m, corridor {stn.get('corridor_cm')} cm, blocker {stn.get('blocker')}; approach {gm.get('approach_deg')} deg, jaw z over top {gm.get('jaw_z_minus_top')} m; walk stop err {wk.get('stop_err_cm')} cm, yaw err {wk.get('yaw_err_deg')} deg, ended t={wk.get('t1')}")  # fmt: skip
        w, h, b, c = a["wrist_ref"], a.get("hand_vs_obj", {}), a["body"], a["objfail"]
        L.append(f"- At failure: wrist-ref err {fmt(w['err_at_fail_cm'])} cm (max {fmt(w['err_max_in_step_cm'])} in step; ref swing {fmt(w.get('ref_swing_cm'))} vs wrist {fmt(w.get('wrist_swing_cm'))} cm); "
                 f"object {h.get('object_ahead_of_jaw_cm')} cm ahead of jaw, {h.get('object_lateral_cm'):+} lat, obj top {h.get('obj_top_minus_jaw_z_cm'):+} cm over jaw; "
                 f"object moved {c['moved_cm']} cm, lift {c['lift_cm']}, tilt {c['tilt_deg']} deg, first move t={fmt(c['first_move_t'])}" + (f", slip {c['slip_cm']} cm since the grip closed" if c.get("slip_cm") is not None else "")
                 + (f", KNOCKED {c['knocked_before_grip_cm']} cm before grip" if c["knocked_before_grip_cm"] else ""))  # fmt: skip
        L.append(f"- Body/controller: pelvis z {b['pelvis_z']}, pitch {b['pitch_deg']} (after walk {b['pitch_after_walk_deg']}) roll {b['roll_deg']} deg, {fmt(b['pelvis_off_stance_cm'])} cm off the stance; waist p/r/y "
                 f"{b['waist_pitch_deg']}/{b['waist_roll_deg']}/{b['waist_yaw_deg']}, {a['hand']} shoulder-pitch {b['shoulder_pitch_deg']} elbow {b['elbow_deg']} deg; primitive {a['controller']['primitive']}; start-to-stance {a['start_to_stance_m']} m")  # fmt: skip
        cor = "; ".join(f"{n_} gap {g_} cm z {z0}-{z1}" for n_, g_, z0, z1 in a["corridor"]) or "none"
        nb = "; ".join(f"{n_} {d} cm top {z} ({k})" for n_, d, z, k in a["neighbours"]) or "none within 40 cm"
        L.append(f"- Hand corridor blockers: {cor}. Neighbours: {nb}")
    if ct:
        L.append(f"- Contrast with OK {ct['ok_label']} ({ct['how']}); fail | ok:")
        L.append("  " + "; ".join(f"{n_} {x} | {y}" for n_, x, y in ct["rows"] if x != y))
    tl = [
        x for x in a["timeline"] if (x[2] - x[1] >= 1 and x[0] != "walk") or not x[3]
    ]  # walk: its end is in the Stance line
    cut = next((i for i, x in enumerate(tl) if not x[3]), len(tl) - 1)
    pre = [x for x in tl[:cut] if x[3]]  # the failing step itself is in the Outcome line
    sline = "- " + ("S: " + " ".join(f"{n_} {x1}" for n_, x0, x1, k_ in pre) if pre else "")
    es = [
        e for e in a.get("edges", []) if e["used"] or (e["listed"] and e["depth_cm"] < 100)
    ]  # all edges stay in the json
    L.append(sline + ((" | " if pre else "") + "E: " + " ".join(f"{e['edge']} {e['depth_cm']:.0f}{'B' if not e['free'] else ''}{'*' if e['used'] else ''}" for e in es) if es else ""))  # fmt: skip
    fg = a.get("fingers")
    if (
        fg and fg["t"][0] <= a["t_fail"] + 0.5 and (fg["grip_ok"] or a["phase"] == "grip")
    ):  # only when the grip is the failure or went through (else a consequence of an earlier failure; kept in the json)
        L.append(f"- Fingers {a['hand'][0].upper()} cm: {fg['open_cm']}>{fg['reached_cm']} w {fg['width_cm']} ({fg['stop_minus_width_cm']:+})")  # fmt: skip
    L.append(f"- Image: img/{a['label']}.png")
    return "\n".join(L)


def pool_summary(rounds, anal, name):
    fails = [a for a in anal if not a["ok"]]
    out = [f"# Failure cards: {name}", "",
           f"{len(rounds) - len(fails)}/{len(rounds)} succeed; fell {sum(bool(r.get('fell')) for r in rounds)}, dropped {sum(bool(r.get('dropped')) for r in rounds)}. One card per failed round ({len(fails)}); numbers are computed from the round .json/.npz and the scene file only.", ""]  # fmt: skip
    cls = collections.Counter(a["class"] or "unclassified" for a in fails)
    out.append("Failure classes: " + ", ".join(f"{k} {n}" for k, n in cls.most_common()))
    ffs = collections.Counter(f"{a['first_fail']['step']}: {shape(a['first_fail']['why'])}" for a in fails)
    out += ["First failing steps:"] + [f"- {n} x {k}" for k, n in ffs.most_common(8)]
    out.append(
        "Key: S passed steps after the walk, end s; Fingers = open>reached cm, w object width; E edge cm, B blocked, * used."
    )
    out.append("By task: " + ", ".join(f"{t} {sum(x['ok'] for x in anal if x['task'] == t)}/{sum(1 for x in anal if x['task'] == t)}" for t in sorted({x['task'] for x in anal})))  # fmt: skip

    def grp(title, fn):
        c = collections.defaultdict(lambda: [0, 0])
        for x in anal:
            if "obj" in x:
                k = fn(x)
                c[k][1] += 1
                c[k][0] += x["ok"]
        out.append(
            f"By {title}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))
        )

    grp("shape class", lambda x: x["obj"]["shape"])
    grp("support top", lambda x: "floor" if not x["obj"]["support"] else "<0.6 m" if x["obj"]["support"]["top"] < 0.6 else "0.6-0.9 m" if x["obj"]["support"]["top"] < 0.9 else ">=0.9 m")  # fmt: skip
    grp("cover above", lambda x: "covered" if x["obj"]["cover_above"] else "open")
    grp("corridor blocker", lambda x: "blocked" if x.get("corridor") else "clear")
    out.append("")
    return out


def build(diag: Path, scene_path: Path, out: Path, images: bool = True) -> int:
    """The cards of a diag round dir into out/ (cards.md, cards.jsonl, img/); returns the number of cards."""
    scene = Scene(scene_path)
    goals = goal_lookup()
    rounds = sorted((json.loads(p.read_text()) for p in diag.glob("*.json") if not p.name.startswith("_")), key=lambda r: r["label"])  # fmt: skip
    anal = []
    for r in rounds:
        npz = diag / f"{r['label']}.npz"
        z = np.load(npz, allow_pickle=True) if npz.exists() else None
        a = analyse(r, z, scene, goals.get(r["task"], {}))
        a["_steps"] = {s["step"]: s for s in r.get("steps") or []}
        anal.append(a)
    (out / "img").mkdir(parents=True, exist_ok=True)
    txt = pool_summary(rounds, anal, f"{diag.parents[1].name}/{diag.name}")
    js = []
    for a in anal:
        if a["ok"]:
            continue
        ct = contrast(a, anal) if "_arr" in a else None
        if images and "_geom" in a:
            draw(a, scene, out / "img" / f"{a['label']}.png")
        txt.append(card_text(a, ct) if "wrist_ref" in a else f"### {a['label']}\n- no trace: {a['first_fail']}")
        txt.append("")
        js.append({k: v for k, v in dict(a, contrast=ct).items() if not k.startswith("_")})
    (out / "cards.md").write_text("\n".join(txt))
    with open(out / "cards.jsonl", "w") as f:
        for x in js:
            f.write(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return len(js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diag", type=Path)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-images", action="store_true")
    a_ = ap.parse_args()
    n = build(a_.diag.resolve(), a_.scene, a_.out, not a_.no_images)
    print(f"{n} cards, {(a_.out / 'cards.md').stat().st_size} bytes of text -> {a_.out}")


if __name__ == "__main__":
    main()
