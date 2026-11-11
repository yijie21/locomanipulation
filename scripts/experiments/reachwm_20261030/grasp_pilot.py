"""Grasp pilot (2026-11-02, after the report): does the self model's command turn missed grasps into grasps?

A cabinet in front of the robot (our own MuJoCo scene dir: walk_mars + the cabinet + one box; the shared ADT scene is
not touched): front face at x = FRONT_X, 4 shelves (boards at SHELVES), side walls, back wall, top. One 6 x 6 x 15 cm box
per placement: shelf x depth from the front x hand, 24 placements. Each episode resets the robot 1.5-2.5 m back (seeded
per placement), walks to the stance STANCE_BACK_M in front of the cabinet, then side-grasps with the probe's procedure
(capability_probe.do_grip): a pre-grasp PRE_OUT_M outside the cabinet front at the grasp height, the entry to the grasp
point (1 cm past the box centre along +x), close,
lift LIFT_M, hold 3 s. Success = the box between the fingers and risen > 2 cm (clear of the board; v1 asked 5 cm,
which standing lifts of a gripped box missed by the lift's own shortfall). Ground-truth target (the box's settled
pose); both methods replay the same reset and walk, so every placement is a paired comparison:
    direct   the pre-grasp and the grasp point sent as the commands, servo off
    inv      each command from the far-turn model's position fixed point (wrist_invert.fixed_point, best by loss) at the
             state the command is issued from (after the PRE_S hold), servo off
The wanted wrist rotation is level with the approach along +x (in gen_reach's family: dyaw = -yaw, pitch 0); the
pelvis height follows the probe's rule clamped to the data's range. Also recorded: the jaw-centre error to the grasp
point when the fingers close, how far the box was pushed before the grip, robot-cabinet contacts per phase.

Tolerance curve (phase t): the inverted commands plus the late position-only servo pulling the jaw centre to the grasp
point + an offset (0-4 cm across the jaw, +-1.5/3 cm along the approach, +-2/4 cm vertical), then the same grip and lift:
grasp success against the offset the hand actually had.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/grasp_pilot.py s                  # write the scene dir
    $PY scripts/experiments/reachwm_20261030/grasp_pilot.py h --procs 6        # placements x methods
    $PY scripts/experiments/reachwm_20261030/grasp_pilot.py t --procs 6        # tolerance curve
    $PY scripts/experiments/reachwm_20261030/grasp_pilot.py d                  # report
Writes outputs/reachwm_20261030/grasp_pilot/.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_invert as WI  # noqa: E402

PV = WI.PV
RS, E, IE, PI, SG = PV.RS, PV.E, PV.IE, PV.PI, PV.SG
OUT = IE.OUT.parent / "grasp_pilot"
SCENE = OUT / "scene"
MODEL = IE.OUT / "wrist_invert2" / "wm_ref.pt"

FRONT_X = 1.0  # cabinet front face, world x
DEPTH_M, HALF_W, BOARD_T = 0.50, 0.60, 0.02
SHELVES = (0.25, 0.60, 0.95, 1.20)  # board centres (tops + BOARD_T / 2); the last is the cabinet top, nothing above it
TOP_Z = SHELVES[-1]
# v1 (2026-11-02 first run): boards 0.30/0.60/0.90/1.20 under a 1.50 top, the pre-grasp 13 cm behind the grasp point,
# i.e. inside the cabinet for depths >= 0.13: on the two upper shelves the hand rising from below caught the board's
# front edge (stuck 17-27 cm short, 11 cm low, hundreds of contact ticks) whatever the command. v2: 35 cm spacing, an
# open top shelf, the pre-grasp PRE_OUT_M outside the front face at the grasp height (capability_probe's via point
# outside the table edge), and a second, wider box: the 6 cm box with the 9.5 cm jaw held across-jaw offsets of 3 cm
WIDTHS = (0.06, 0.08)
PRE_OUT_M = 0.06
STANCE_BACK_M = 0.30  # stance pelvis this far in front of the cabinet front
DEPTHS = (0.10, 0.20, 0.30)  # box centre behind the front face
LATERAL_M = 0.15
BOX_HALF_H = 0.075
LIFT_M = 0.06  # v1 lifted 0.10: on the 0.95 m shelf the hand then met the board above
SEED = 4200000

CABINET = f"""
    <body name="cab" pos="0 0 0">
      <geom name="cab_plinth" type="box" pos="{FRONT_X + DEPTH_M / 2} 0 {(SHELVES[0] - BOARD_T / 2) / 2}"
            size="{DEPTH_M / 2} {HALF_W} {(SHELVES[0] - BOARD_T / 2) / 2}" rgba="0.55 0.45 0.35 1" friction="0.6 0.005 0.0001"/>
""" + "".join(f"""      <geom name="cab_board{i}" type="box" pos="{FRONT_X + DEPTH_M / 2} 0 {z}" size="{DEPTH_M / 2} {HALF_W} {BOARD_T / 2}"
            rgba="0.62 0.5 0.38 1" friction="0.6 0.005 0.0001"/>
""" for i, z in enumerate(SHELVES)) + f"""      <geom name="cab_back" type="box" pos="{FRONT_X + DEPTH_M + 0.01} 0 {TOP_Z / 2}" size="0.01 {HALF_W} {TOP_Z / 2}"
            rgba="0.5 0.4 0.3 1"/>
      <geom name="cab_left" type="box" pos="{FRONT_X + DEPTH_M / 2} {HALF_W + 0.01} {TOP_Z / 2}" size="{DEPTH_M / 2} 0.01 {TOP_Z / 2}"
            rgba="0.5 0.4 0.3 1"/>
      <geom name="cab_right" type="box" pos="{FRONT_X + DEPTH_M / 2} {-HALF_W - 0.01} {TOP_Z / 2}" size="{DEPTH_M / 2} 0.01 {TOP_Z / 2}"
            rgba="0.5 0.4 0.3 1"/>
    </body>
""" + "".join(f"""    <body name="pilot_box{int(round(100 * w))}" pos="{3.0 + i} 3.0 0.2">
      <freejoint name="pilot_box{int(round(100 * w))}_joint"/>
      <geom name="pilot_box{int(round(100 * w))}_geom" type="box" size="{w / 2} 0.03 {BOX_HALF_H}" mass="0.3" rgba="0.2 0.5 0.8 1"
            friction="0.6 0.005 0.0001"/>
    </body>
""" for i, w in enumerate(WIDTHS))  # fmt: skip


def write_scene() -> None:
    import capability_probe as cap

    xml = (cap.WALK / "scene_locomani.xml").read_text()
    assert xml.count("</worldbody>") == 1
    SCENE.mkdir(parents=True, exist_ok=True)
    (SCENE / "scene_meta.json").write_text((cap.WALK / "scene_meta.json").read_text())
    (SCENE / "scene_locomani.xml").write_text(xml.replace("</worldbody>", CABINET + "  </worldbody>"))
    print("wrote", SCENE)


def placements() -> list[dict]:
    """Box width x shelf x depth x hand; the start (and so the walk) is seeded by the position, not the width, so the
    two widths at one position share the reset and the walk."""
    out = []
    for w in WIDTHS:
        for si, z in enumerate(SHELVES):
            for di, d in enumerate(DEPTHS):
                for side in ("right", "left"):
                    sd = 1.0 if side == "right" else -1.0
                    pid = len(out)
                    rng = np.random.default_rng(SEED + pid % (len(SHELVES) * len(DEPTHS) * 2))
                    out.append(dict(pid=pid, width=w, body=f"pilot_box{int(round(100 * w))}", shelf=si, depth=d, side=side,
                                box=[FRONT_X + d, -sd * LATERAL_M, z + BOARD_T / 2 + BOX_HALF_H + 0.001],
                                start=[FRONT_X - STANCE_BACK_M - float(rng.uniform(1.5, 2.5)), float(rng.uniform(-0.2, 0.2))],
                                yaw0=float(math.radians(rng.uniform(-15, 15)))))  # fmt: skip
    return out


def make_env():
    """selfwm_20261029/env.Env on our scene dir (capability_probe.Probe's set-up with SCENE)."""
    import capability_probe as cap
    import mujoco

    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.backends.scalebfm.motor import Motor, MotorConfig
    from locomani.tasks.scene.world import GRIP_LAW

    policy = R.Policy("cuda", ScaleBfmBackendConfig())
    sim = R.Sim(policy.meta, backend=ScaleBfmBackendConfig(scene_dir=str(SCENE), object_name="", **GRIP_LAW))
    m = Motor(sim, policy, MotorConfig(), model_xml=str(cap.WALK / "scene_locomani.xml"))
    probe = cap.Probe.__new__(cap.Probe)
    probe.R, probe.kind, probe.m, probe.sim = R, "grip", m, sim
    env = E.Env.__new__(E.Env)
    env.cap, env.probe, env.m, env.sim, env.R = cap, probe, m, sim, R
    env.boxes = {w: sim.m.body(f"pilot_box{int(round(100 * w))}").id for w in WIDTHS}
    env.cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    return env


class Watch:
    """Per tick: robot-cabinet contacts (count of ticks with one, deepest penetration) under the current phase, and
    qpos every 2 ticks for later videos."""

    def __init__(self, env):
        m, sm = env.m, env.sim.m
        self.env, self.phase = env, "walk"
        root = sm.body_rootid[sm.body("pelvis").id]
        self.robot = np.array([sm.body_rootid[b] == root for b in range(sm.nbody)])
        cab = sm.body("cab").id
        self.cab = np.array([sm.geom_bodyid[g] == cab for g in range(sm.ngeom)])
        self.clear()
        orig = m.tick

        def tick():
            ok = orig()
            self.sample()
            return ok

        m.tick = tick

    def clear(self) -> None:
        self.ticks, self.contact, self.depth, self.q, self.marks, self.parts = 0, {}, {}, [], [], {}

    def mark(self, name: str) -> None:
        self.phase = name
        self.marks.append((name, len(self.q)))

    def sample(self) -> None:
        sim = self.env.sim
        d, sm = sim.d, sim.m
        self.ticks += 1
        hit, deep = False, 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for ga, gb in ((g1, g2), (g2, g1)):
                if self.cab[ga] and self.robot[sm.geom_bodyid[gb]]:
                    hit, deep = True, max(deep, -float(c.dist))
                    nm = sm.body(sm.geom_bodyid[gb]).name
                    key = f"{self.phase}:{nm}"
                    self.parts[key] = self.parts.get(key, 0) + 1
        if hit:
            self.contact[self.phase] = self.contact.get(self.phase, 0) + 1
            self.depth[self.phase] = max(self.depth.get(self.phase, 0.0), deep)
        if self.ticks % 2 == 0:
            self.q.append(d.qpos.astype(np.float32).copy())


def x0_here(rec, sd: float, walked: bool, start: np.ndarray, chain: int) -> np.ndarray:
    """restance.x0_now with the chain flag (the entry is the second reach of a chain)."""
    import data as DT

    seg = dict(S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=len(rec.S) - 1,
               side=sd, target=np.zeros(3, np.float32), dyaw=0.0, pitch=0.0, rot6=np.zeros(6, np.float32),
               pelvis_z=-1.0, start=start, walked=walked, chain=chain, fell=False, ok=True, jaw_err_cm=-1.0,
               att_err_deg=-1.0, ik_err_cm=-1.0, reach_s=0.0, reset=0)  # fmt: skip
    D = IE.pack([seg])
    D.update(ref_q=np.zeros((1, 29), np.float32), ref_pel=np.r_[0, 0, 0, 1, 0, 0, 0][None].astype(np.float32),
             ref_s=np.zeros(1, np.float32))  # fmt: skip
    DT.derive(D)
    return D["x0"][0, :206]


def state_here(env, rec, side: str, goal_w: np.ndarray, rot_w: np.ndarray, chain: int) -> dict:
    """wrist_invert's planning state for a command issued now (call right after the PRE_S hold)."""
    pv, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    g = Rh.T @ np.r_[goal_w[:2] - pv[:2], 0.0]
    goal = np.r_[g[:2], goal_w[2]].astype(np.float32).astype(np.float64)
    rot_h = Rh.T @ rot_w
    sd = 1.0 if side == "right" else -1.0
    start = np.r_[pv, yaw].astype(np.float32)
    return dict(sd=sd, p=pv, yaw=yaw, Rh=Rh, goal=goal, rot_w=rot_w, rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32),
                c2=dict(side=side, dyaw=float(math.atan2(rot_h[1, 0], rot_h[0, 0])), pitch=0.0), start=start, walked=True,
                x0b=x0_here(rec, sd, True, start, chain))  # fmt: skip


def to_w(st: dict, t) -> np.ndarray:
    t = np.asarray(t, float)
    return np.r_[st["p"][:2] + (st["Rh"] @ np.r_[t[:2], 0.0])[:2], t[2]]


def pelvis_for(z_g: float) -> float:
    """capability_probe.do_grip's rule (crouch under 0.60 m, 5 cm over the grasp), clamped to the data's 0.52-0.76."""
    return -1.0 if z_g >= 0.60 else float(np.clip(z_g + 0.05, 0.52, 0.76))


def reach(env, rec, model, method: str, side: str, goal_w, rot_w, pz: float, chain: int, servo_goal=None) -> dict:
    """One reach the way gen_reach.segment does it (PRE_S hold, reach, POST_S hold), its command per `method`."""
    import gen_reach as G

    m = env.m
    rec.start(side)
    m.hold(G.PRE_S)
    rec.on = False
    info = dict(goal_w=np.asarray(goal_w, float).tolist())
    if method == "inv":
        st = state_here(env, rec, side, np.asarray(goal_w, float), rot_w, chain)
        evs = WI.fixed_point(model, m, st, st["goal"], pz, None)
        best = min(evs, key=lambda ev: ev["l_pos"])
        target_w = to_w(st, best["t"])
        info.update(pred_err=float(best["pred_err"]), pred_std=float(best["pred_std"]), k=int(best["k"]),
                    shift_cm=float(100 * np.linalg.norm(target_w - np.asarray(goal_w))))  # fmt: skip
    else:
        target_w = np.asarray(goal_w, float)
    if servo_goal is not None:
        m.servo_target = IE.KeepTarget({side: np.asarray(servo_goal, float)})
    out = m.reach(side, target_w, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=servo_goal is not None)
    if m.fell_at is None:
        m.hold(G.POST_S)
    m.servo_target = {}
    jaw = m.jaw_centre(side)[0]
    info.update(target_w=target_w.tolist(), ok=bool(out.ok), err_cm=float(100 * np.linalg.norm(jaw - np.asarray(goal_w))),
                miss_w_cm=(100 * (jaw - np.asarray(goal_w))).tolist())  # fmt: skip
    return info


def episode(env, rec, watch, gate, model, pl: dict, method: str, offset=None) -> dict:
    m, sim = env.m, env.sim
    side = pl["side"]
    watch.clear()
    watch.mark("walk")
    m.reset(tuple(pl["start"]), pl["yaw0"])
    for w, b in env.boxes.items():  # the other box waits off stage
        park = np.array([3.0 + WIDTHS.index(w), 3.0, 0.2])
        m.place(
            f"pilot_box{int(round(100 * w))}",
            np.array(pl["box"]) if w == pl["width"] else park,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    m.hold(1.0)
    stance = np.array([FRONT_X - STANCE_BACK_M, 0.0])
    m.walk_to(stance, yaw=0.0)
    res = dict(
        pid=pl["pid"], method=method, offset=offset, width=pl["width"], shelf=pl["shelf"], depth=pl["depth"], side=side
    )
    if m.fell_at is not None:
        return dict(res, fell="walk", success=False)
    m.hold(0.5)
    ob = env.boxes[pl["width"]]
    box0 = sim.d.xpos[ob].copy()
    pv = m.pelvis()[0]
    res["stance_cm"] = (100 * (pv[:2] - stance)).tolist()
    grasp = np.r_[box0[0] + 0.01, box0[1], box0[2]]  # the probe's side grasp: 1 cm past the centre, at mid height
    pre = np.array([FRONT_X - PRE_OUT_M, grasp[1], grasp[2]])  # outside the front face, at the grasp height
    rot_w = np.eye(3)  # level, approach along +x (the cabinet's depth)
    pz = pelvis_for(float(grasp[2]))
    res["pz"] = pz
    cfg = m.cfg
    servo_goal = None
    if offset is not None:  # tolerance curve: the late position-only servo nails the entry to grasp + offset
        gate.mode, gate.radius = "late_pos", None
        gate.reset()
        m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
        servo_goal = grasp + np.asarray(offset, float)
    watch.mark("pregrasp")
    res["pregrasp"] = reach(env, rec, model, "inv" if method != "direct" else "direct", side, pre, rot_w, pz, 0)
    if m.fell_at is not None:
        m.cfg = cfg
        return dict(res, fell="pregrasp", success=False)
    watch.mark("entry")
    res["entry"] = reach(env, rec, model, "inv" if method != "direct" else "direct", side,
                         grasp if offset is None else servo_goal, rot_w, pz, 1, servo_goal=servo_goal)  # fmt: skip
    m.cfg = cfg
    gate.mode = "orig"
    if m.fell_at is not None:
        return dict(res, fell="entry", success=False)
    jaw = m.jaw_centre(side)[0]
    res["at_grip_cm"] = (100 * (jaw - grasp)).tolist()  # jaw centre to the grasp point when the fingers close
    res["pushed_cm"] = float(100 * np.linalg.norm(sim.d.xpos[ob][:2] - box0[:2]))
    watch.mark("grip")
    g = m.grip(side)
    res["grip_held"] = bool(g.ok)
    watch.mark("lift")
    lift = m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_M])
    m.reach(side, lift, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False)
    if m.fell_at is None:
        watch.mark("hold")
        m.hold(3.0)
    obj = sim.d.xpos[ob]
    gripped = side in m.gripping(ob)
    res.update(fell=None if m.fell_at is None else "grip/lift", gripped=bool(gripped),
               rise_cm=float(100 * (obj[2] - box0[2])), success=bool(gripped and obj[2] > box0[2] + 0.02 and m.fell_at is None),
               contacts=dict(watch.contact), contact_depth_mm={k: round(1000 * v, 1) for k, v in watch.depth.items()},
               contact_parts=dict(sorted(watch.parts.items(), key=lambda kv: -kv[1])[:6]))  # fmt: skip
    return res


OFFSETS = [(0, 0, 0)] + [(0, a * s, 0) for a in (0.01, 0.02, 0.03, 0.04) for s in (1, -1)] + \
          [(a, 0, 0) for a in (-0.03, -0.015, 0.015)] + [(0, 0, a) for a in (-0.04, -0.02, 0.02, 0.04)]  # fmt: skip
TOL_PIDS = (
    8,
    14,
    32,
    38,
)  # shelf 0.60 and 0.95, depth 0.20, right hand; the 6 cm box (8, 14) and the 8 cm box (32, 38)


def worker(phase: str, wid: int, jobs: list) -> None:
    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    env = make_env()
    rec = G.Recorder(env)
    watch = Watch(env)
    gate = SG.Gate(env.m)
    model = torch.load(MODEL, map_location="cpu", weights_only=False).eval()
    P = {p["pid"]: p for p in placements()}
    res = []
    for job in jobs:
        tic = time.time()
        pid, method, off = job
        r = episode(env, rec, watch, gate, model, P[pid], method, off)
        np.save(OUT / "q" / f"{phase}_{pid}_{method}_{'_'.join(str(round(100 * v, 1)) for v in off) if off else 'none'}.npy",
                np.array(watch.q, np.float32))  # fmt: skip
        r["marks"] = watch.marks
        res.append(r)
        e = r.get("entry", {})
        print(f"{phase}{wid} pid {pid} {method} off {off}: success {r['success']} fell {r.get('fell')} at_grip "
              f"{np.round(r.get('at_grip_cm', [np.nan] * 3), 1)} pushed {r.get('pushed_cm', float('nan')):.1f} cm "
              f"pred {e.get('pred_err', float('nan')):.2f}+-{e.get('pred_std', float('nan')):.2f} contacts {r.get('contacts')} "
              f"| {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        json.dump(res, open(OUT / f"{phase}_w{wid}.json", "w"))


def launch(phase: str, procs: int, gpus: str) -> None:
    if phase == "h":
        jobs = [(p["pid"], m, None) for p in placements() for m in ("direct", "inv")]
    else:
        jobs = [(pid, "inv", list(o)) for pid in TOL_PIDS for o in OFFSETS]
    (OUT / "q").mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"{phase}_w{w}.log", "w")
        ps.append(subprocess.Popen([sys.executable, __file__, "_w", phase, str(w), json.dumps(jobs[w::procs])], env=envv,
                                   stdout=log, stderr=subprocess.STDOUT))  # fmt: skip
    for p in ps:
        p.wait()
    print("done", [p.returncode for p in ps])


def report() -> None:
    def load(phase):
        out = []
        for f in sorted(OUT.glob(f"{phase}_w*.json")):
            out += json.load(open(f))
        return out

    H = load("h")
    by = {}
    for r in H:
        by.setdefault(r["pid"], {})[r["method"]] = r
    L = ["# Grasp pilot", "", f"{len(by)} placements, paired (same reset and walk); success = held and risen > 2 cm", "",
         "| box | method | success | jaw to grasp point at the grip, median cm | across the jaw median cm | box pushed > 1 cm | falls | episodes with cabinet contact |",
         "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for w, meth in [(w, m) for w in WIDTHS for m in ("direct", "inv")]:
        rs = [v[meth] for v in by.values() if meth in v and v[meth].get("width") == w]
        if not rs:
            continue
        ag = [np.linalg.norm(r["at_grip_cm"]) for r in rs if "at_grip_cm" in r]
        lat = [abs(r["at_grip_cm"][1]) for r in rs if "at_grip_cm" in r]
        L.append(f"| {int(round(100 * w))} cm | {meth} | {sum(r['success'] for r in rs)}/{len(rs)} | {np.median(ag):.2f} | {np.median(lat):.2f} | "
                 f"{sum(r.get('pushed_cm', 0) > 1 for r in rs)} | {sum(bool(r.get('fell')) for r in rs)} | "
                 f"{sum(bool(r.get('contacts')) for r in rs)} |")  # fmt: skip
    both = [v for v in by.values() if "direct" in v and "inv" in v]
    L += ["", f"paired: inv wins {sum(v['inv']['success'] and not v['direct']['success'] for v in both)}, "
          f"loses {sum(v['direct']['success'] and not v['inv']['success'] for v in both)}, "
          f"both {sum(v['direct']['success'] and v['inv']['success'] for v in both)}, "
          f"neither {sum(not v['direct']['success'] and not v['inv']['success'] for v in both)}", "",
          "per shelf (board centre m), successes direct / inv: " + "; ".join(
              f"{SHELVES[k]:.2f}: " + ", ".join(f"{int(round(100 * w))} cm {sum(v['direct']['success'] for v in both if v['direct']['shelf'] == k and v['direct'].get('width') == w)}"
                                               f" / {sum(v['inv']['success'] for v in both if v['inv']['shelf'] == k and v['inv'].get('width') == w)}" for w in WIDTHS)
              for k in range(len(SHELVES))), "",
          "| pid | shelf | depth | side | stance off cm | direct: at grip cm, pushed, ok | inv: at grip cm, pushed, ok, pred | contacts direct / inv |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for pid in sorted(by):
        v = by[pid]
        d, i = v.get("direct", {}), v.get("inv", {})
        f = lambda r: (
            f"{np.round(r['at_grip_cm'], 1).tolist()}, {r.get('pushed_cm', 0):.1f}, {'Y' if r['success'] else 'n'}"  # noqa: E731
            if "at_grip_cm" in r
            else f"fell {r.get('fell')}"
        )
        e = i.get("entry", {})
        L.append(f"| {pid} | {SHELVES[d.get('shelf', i.get('shelf', 0))]:.2f} | {d.get('depth', i.get('depth'))} | {d.get('side', i.get('side'))} | "
                 f"{np.round(d.get('stance_cm', [np.nan, np.nan]), 1).tolist()} | {f(d)} | {f(i)}, {e.get('pred_err', float('nan')):.2f}+-{e.get('pred_std', float('nan')):.2f} | "
                 f"{d.get('contacts')} / {i.get('contacts')} |")  # fmt: skip
    T = load("t")
    if T:
        L += [
            "",
            "## Tolerance curve (inverted commands + late position servo to grasp + offset)",
            "",
            "| pid | offset cm (approach, across, up) | actual at grip cm | success | pushed cm |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(T, key=lambda r: (r["pid"], r["offset"])):
            L.append(f"| {r['pid']} | {np.round(100 * np.array(r['offset']), 1).tolist()} | {np.round(r.get('at_grip_cm', [np.nan] * 3), 1).tolist()} | "
                     f"{'Y' if r['success'] else 'n'} | {r.get('pushed_cm', float('nan')):.1f} |")  # fmt: skip
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["s", "h", "t", "d", "_w"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "s":
        write_scene()
    elif a.phase in ("h", "t"):
        launch(a.phase, a.procs, a.gpus)
    elif a.phase == "d":
        report()
    else:
        ph, wid, jobs = a.rest[0], int(a.rest[1]), json.loads(a.rest[2])
        worker(ph, wid, [tuple(j) if j[2] is None else (j[0], j[1], tuple(j[2])) for j in jobs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
