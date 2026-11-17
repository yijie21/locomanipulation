"""Press-panel demo (2026-11-16): a wall of buttons, or a wall-mounted keyboard, 1 m in front of the robot. The robot
walks once to a stance in front of the panel, then presses a sequence of targets with the closed Dex1, both arms homed
between presses (gen_reach.home). Before each press it re-anchors where it stands and sights the panel, so every press
is a post-walk chain-0 reach like the self model's data (episode()). The closed Dex1's tip face is 1.8 x 1 cm, 3.3 cm
ahead of the jaw centre (measured on walk_mars, 2026-11-16); the tip is sent DEPTH past the cap's resting face.

Three ways to command each press, every one from the bit-identical start, the same walk, sightings and re-steps:
  direct     the controller alone: the jaw centre commanded where the tip should press
  servo      the same with the motor's jaw-centre servo on, no model. NB (review, 2026-11-17): the servo measures the
             jaw with Motor.jaw_centre, the simulator's wrist pose -- ScaleBFM unless m.jaw_source is set, AmoMotor
             (GR00T-WBC, AMO) always -- so this is a ground-truth (oracle) servo, not encoders + leg odometry
  selfmodel  the self model's position fixed point (wrist_invert.fixed_point) moves the command so the predicted landing
             is the press point; servo off
  servo_odo  the same servo measuring the jaw as a robot can: joint readings + gravity/gyro + leg odometry, anchored
             where the robot stands at the sighting (leg_odo.JawFeed via Motor.jaw_source)
  lookup     no learned model: the command moved by the median landing offset of similar reaches in the self model's own
             training data (same side, stand/squat, wrist level, target within a few cm; `table` phase)
  selfmodel_servo_odo  the self model's command as the reference and servo_odo pulling the jaw to the wanted point
             (invert_eval's aim_ff_servo), full settle

Phases
  scene   write outputs/press/scene_{wall,piano}/ (walk_mars + the panel); REACH_SCENE_DIR points the sim at one;
          PRESS_BTN_R sets the wall buttons' radius (round 1: 3.5 cm; round 2: 2 cm)
  table   the lookup method's offsets from REACHWM_ROOT's training split -> PRESS_OUT/lookup_<controller>.npz
  run     --layout wall|piano --seq N --methods ...: one episode per method -> press/<ctrl>_<layout>_<seq>/<method>.{json,npz}
  video   --tag <ctrl>_<layout>_<seq>: per method an mp4 (keyboard notes as audio) and the methods side by side

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/press_panel.py scene
    export BFM_ROOT=odo REACH_CONTROLLER=gr00t_wbc REACHWM_ROOT=$PWD/outputs/reachwm_gr00t_20261116 \\
        APPROACH_WM_POST=$PWD/outputs/reachwm_gr00t_20261116/invert/wrist_graspr_odo/wm_ref.pt \\
        WALK_SUFFIX=_odo WALK_DATA=approach_odo,approach0_odo REACH_SCENE_DIR=$PWD/outputs/press/scene_wall
    $PY scripts/experiments/reachwm_20261030/press_panel.py run --layout wall --seq 0
    MUJOCO_GL=egl $PY scripts/experiments/reachwm_20261030/press_panel.py video --tag gr00t_wbc_wall_0
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = HERE.parents[2]
OUT = Path(os.environ.get("PRESS_OUT") or REPO / "outputs" / "press")
WALK_SCENE = REPO / "outputs" / "scene" / "walk_mars"

PX = 1.00  # panel board's front face, world x (the robot starts at the origin facing +x)
TIP = 0.033  # closed Dex1: pad centre -> tip face along the jaw's x axis
# Long-throw plungers (11-16 smoke run: with a 2.5 cm cap and 2.2 cm travel a tip landing a little long bottomed the cap
# and a missed tip hit the board; the arm's PD then pushed the whole body back 5-24 cm and every later press was off).
# Now the cap stands 5 cm proud with 4 cm of travel: a press aimed DEPTH in never reaches the stop or the board unless it
# lands 2.4 cm long, and a missed tip presses air.
CAP_OUT, TRAVEL = 0.05, 0.04  # cap's protrusion from the board, its spring travel
DEPTH = 0.016  # the tip is sent this far past the cap's resting face; a press counts from PRESS_M
PRESS_M = 0.006  # cap displacement that counts as a press
SPRING, DAMP = 120.0, 2.0  # N/m (1.9 N at DEPTH, 4.8 N at full travel), N s/m
JAW_SHUT = -0.02  # Dex1 finger joints' lower limit: closed, no squeeze
RESTEP_M, RESTEP_DEG = 0.06, 8.0  # step back to the planned stance when the body has drifted this far
QFPS = 2  # one video frame every QFPS motor ticks (50 Hz motor -> 25 fps)
FPS = 25

WALL_Z = (1.10, 0.90, 0.70, 0.50)
WALL_Y = (0.30, 0.15, 0.0, -0.15, -0.30)
BTN_R = float(os.environ.get("PRESS_BTN_R", "0.035"))  # round 1: 7 cm arcade buttons, 15 cm apart
NOTES = ("C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5")
KEY_W, KEY_H, KEY_Z = 0.088, 0.15, 0.95  # big floor-piano style keys (11-16 smoke: 7 cm keys, 0.5 cm gaps, 9 neighbours hit)
KEY_PITCH = 0.10  # key centre spacing: 1.2 cm gaps, the 8 keys span y +-0.39 m
TWINKLE = "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4"
METHODS = ("direct", "lookup", "servo", "servo_odo", "selfmodel", "selfmodel_servo_odo")
ODO_SERVO = ("servo_odo", "selfmodel_servo_odo")
LOOKUP_KEYS = ("ref_odo", "graspr_odo")  # datasets whose target and landing share the command-time frame
LIT, HIT, MISS, IDLE = (0.10, 0.85, 0.30), (0.10, 0.45, 0.95), (0.95, 0.25, 0.20), (0.86, 0.86, 0.83)
KEY_IDLE = (0.96, 0.96, 0.93)


# -- the panel ------------------------------------------------------------------------------------------------------
def targets(layout: str) -> list[dict]:
    """The press points: name, world y and z of the cap centre."""
    if layout == "wall":
        return [dict(name=f"b{r}{c}", y=y, z=z) for r, z in enumerate(WALL_Z) for c, y in enumerate(WALL_Y)]
    y0 = KEY_PITCH * (len(NOTES) - 1) / 2  # low notes on the robot's left (+y)
    return [dict(name=n, y=y0 - k * KEY_PITCH, z=KEY_Z) for k, n in enumerate(NOTES)]


def panel_xml(layout: str) -> str:
    T = targets(layout)
    if layout == "wall":
        board = '<geom name="panel_board" type="box" pos="0.02 0 0.80" size="0.02 0.50 0.50" rgba="0.17 0.19 0.22 1"/>'
        cap = lambda t: f'<geom name="{t["name"]}_g" type="cylinder" size="{BTN_R} {CAP_OUT / 2}" pos="{-CAP_OUT / 2} 0 0" quat="0.7071068 0 0.7071068 0" rgba="{IDLE[0]} {IDLE[1]} {IDLE[2]} 1" mass="0.02"/>'  # noqa: E731,E501
    else:
        board = '<geom name="panel_board" type="box" pos="0.02 0 0.95" size="0.02 0.44 0.13" rgba="0.12 0.12 0.13 1"/>'
        cap = lambda t: f'<geom name="{t["name"]}_g" type="box" size="{CAP_OUT / 2} {KEY_W / 2} {KEY_H / 2}" pos="{-CAP_OUT / 2} 0 0" rgba="{KEY_IDLE[0]} {KEY_IDLE[1]} {KEY_IDLE[2]} 1" mass="0.03"/>'  # noqa: E731,E501
    out = [f'    <body name="panel" pos="{PX:.4f} 0 0">', "      " + board]
    for t in T:
        out.append(f'      <body name="{t["name"]}" pos="0 {t["y"]:.4f} {t["z"]:.4f}">')
        out.append(f'        <joint name="{t["name"]}_j" type="slide" axis="1 0 0" range="0 {TRAVEL}" stiffness="{SPRING}" damping="{DAMP}" armature="0.002"/>')  # fmt: skip
        out.append("        " + cap(t))
        out.append("      </body>")
    out.append("    </body>")
    out.append('    <light name="panel_light" pos="0.2 0 2.4" dir="0.3 0 -1" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" castshadow="false"/>')  # fmt: skip
    return "\n".join(out)


def write_scenes() -> None:
    import mujoco

    xml0 = (WALK_SCENE / "scene_locomani.xml").read_text()
    assert xml0.count("</worldbody>") == 1
    for layout in ("wall", "piano"):
        d = OUT / f"scene_{layout}"
        d.mkdir(parents=True, exist_ok=True)
        # the board is welded to the world, so MuJoCo's parent filter does not keep a cap off it (the 11-16 smoke run:
        # every cap held at its rest face); exclude each cap-board pair explicitly
        excl = "\n".join(f'    <exclude body1="panel" body2="{t["name"]}"/>' for t in targets(layout))
        xml = xml0.replace("</worldbody>", panel_xml(layout) + "\n  </worldbody>\n  <contact>\n" + excl + "\n  </contact>")
        (d / "scene_locomani.xml").write_text(xml)
        (d / "scene_meta.json").write_text((WALK_SCENE / "scene_meta.json").read_text())
        m = mujoco.MjModel.from_xml_path(str(d / "scene_locomani.xml"))
        dd = mujoco.MjData(m)
        k = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, targets(layout)[0]["name"] + "_j")]
        dd.qpos[k] = 0.015
        mujoco.mj_forward(m, dd)
        cap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, targets(layout)[0]["name"] + "_g")
        touching = [i for i in range(dd.ncon) if cap in (dd.contact[i].geom1, dd.contact[i].geom2)]
        assert not touching, f"{layout}: a pressed cap still touches {touching}"
        print(layout, d, "nq", m.nq, "targets", len(targets(layout)), "cap free to travel")


def sequence(layout: str, seq: int) -> list[str]:
    if layout == "piano":
        return TWINKLE.split()
    rng = np.random.default_rng(31000000 + seq)
    names, out = [t["name"] for t in targets("wall")], []
    while len(out) < 8:
        n = names[int(rng.integers(len(names)))]
        if not out or n != out[-1]:
            out.append(n)
    return out


# -- recording ------------------------------------------------------------------------------------------------------
class Tape:
    """Per motor tick: every target's cap displacement, the running max inside the current press window; every QFPS
    ticks the full qpos (the video), the lit target and the phase."""

    def __init__(self, env, names: list[str]):
        import mujoco

        sim, m = env.sim, env.m
        self.adr = np.array([sim.m.jnt_qposadr[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, n + "_j")] for n in names])
        self.sim, self.names, self.on, self.k = sim, names, False, 0
        self.q, self.lit, self.phase, self.lit_now, self.phase_now = [], [], [], -1, ""
        self.peak = np.zeros(len(names))
        orig = m.tick

        def tick():
            ok = orig()
            if self.on:
                disp = self.sim.d.qpos[self.adr]
                np.maximum(self.peak, disp, out=self.peak)
                self.k += 1
                if self.k % QFPS == 0:
                    self.q.append(self.sim.d.qpos.copy().astype(np.float32))
                    self.lit.append(self.lit_now)
                    self.phase.append(self.phase_now)
            return ok

        m.tick = tick


def shut_jaws(m) -> None:
    m.jaw[:] = JAW_SHUT
    if hasattr(m.sim, "grip_target"):
        m.sim.grip_target[:] = JAW_SHUT


def tip_of(m, side: str) -> np.ndarray:
    p, Rw = m.jaw_centre(side)
    return p + Rw[:, 0] * TIP


def to_w(p1, Rh1, t) -> np.ndarray:
    t = np.asarray(t, float)
    return np.r_[p1[:2] + (Rh1 @ np.r_[t[:2], 0.0])[:2], t[2]]


def write_lookup() -> None:
    """The lookup method's table from REACHWM_ROOT's training split (LOOKUP_KEYS): per reach the side, stand or the
    pelvis height asked, the wrist pitch and turn asked, the target and the landing offset (landing - target, heading
    frame at the command), nothing else. Written to PRESS_OUT/lookup_<REACH_CONTROLLER>.npz."""
    import data as DT

    D = DT.load(",".join(str(DT.OUT / f"data_{k}" / "all.npz") for k in LOOKUP_KEYS))
    tr = DT.splits(D)["train"]
    tr = tr[~D["fell"][tr].astype(bool)]
    out = dict(side=np.sign(D["side"][tr]).astype(np.int8), pz=D["pelvis_z"][tr].astype(np.float32),
               pitch=D["pitch"][tr].astype(np.float32), dyaw=D["dyaw"][tr].astype(np.float32),
               target=D["target"][tr].astype(np.float32), off=(D["pos_f"][tr] - D["target"][tr]).astype(np.float32))  # fmt: skip
    ctrl = os.environ.get("REACH_CONTROLLER", "scalebfm")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"lookup_{ctrl}.npz", **out)
    print(f"lookup_{ctrl}.npz: {len(tr)} training reaches from {LOOKUP_KEYS}", flush=True)


def lookup(T: dict, sd: float, pz: float, g: np.ndarray) -> tuple[np.ndarray, int]:
    """Median landing offset of the training reaches like this one: same hand, stand (or a pelvis within 5 cm), wrist
    level (pitch and turn under 20 deg), target within 5 cm in height and 10 cm across; the box doubles until 20 match."""
    base = (T["side"] == int(np.sign(sd))) & (np.abs(T["pitch"]) < np.radians(20)) & (np.abs(T["dyaw"]) < np.radians(20))
    base &= (T["pz"] < 0) if pz < 0 else (np.abs(T["pz"] - pz) < 0.05)
    t = T["target"]
    for k in (1.0, 2.0, 4.0, 8.0):
        m = base & (np.abs(t[:, 2] - g[2]) < 0.05 * k) & (np.abs(t[:, 0] - g[0]) < 0.10 * k) & (np.abs(t[:, 1] - g[1]) < 0.10 * k)
        if m.sum() >= 20:
            break
    return np.median(T["off"][m], axis=0).astype(np.float64), int(m.sum())


def episode(env, rec, tape: Tape, models: dict, layout: str, seq_names: list[str], method: str) -> dict:
    """One sequence. The robot walks once to the stance in front of the first target; before every press it re-anchors
    its stand reference where it stands (Motor._stand: the leg odometry restarts there, OdoRoot) and sights the panel
    (the panel's pose relative to the pelvis, exact here: a head camera's job), so each press is a fresh post-walk,
    chain-0 reach as in the self model's data. If the body has drifted more than RESTEP_M off the planned stance it
    first steps back (an open walk by the sighted offset). All three methods get the same walk, sightings and re-steps;
    they differ only in the reach command."""
    import approach_eval as AE
    import gen_reach as GR
    import select_eval as SE
    import wrist_invert as WI

    E = AE.E
    m = env.m
    T = {t["name"]: t for t in targets(layout)}
    env.reset(None)
    shut_jaws(m)
    m.hold(0.6)
    tape.q, tape.lit, tape.phase, tape.k, tape.lit_now, tape.on = [], [], [], 0, -1, True
    tape.phase_now = "sight"
    rec.start("right")
    m.hold(GR.PRE_S)
    p0, Rp0 = env.pelvis()
    R0 = E.rot_z(E.yaw_of(Rp0))
    first = T[seq_names[0]]
    stance_w = np.r_[PX - CAP_OUT + DEPTH - TIP - AE.X_NOM, 0.0]  # pelvis xy in front of the panel's centre line
    d0 = R0.T @ np.r_[stance_w - p0[:2], 0.0]
    walk = np.r_[d0[0], d0[1], 0.0]
    res = dict(method=method, layout=layout, seq=seq_names, walk=walk.tolist(), stance_w=stance_w.tolist(), presses=[],
               first=first["name"])  # fmt: skip
    tape.phase_now = "walk"
    GR.open_walk(m, float(walk[0]), float(walk[1]), float(walk[2]))
    if m.fell_at is not None:
        tape.on = False
        return dict(res, fell="walk", q=tape)
    rb = GR.rot_y(0.0)
    names = tape.names
    n_restep = 0
    for j, name in enumerate(seq_names):
        tw = T[name]
        side = "right" if tw["y"] <= 0.0 else "left"
        sd = 1.0 if side == "right" else -1.0
        tape.lit_now, tape.phase_now = names.index(name), "aim"
        p, Rp = env.pelvis()
        off = stance_w - p[:2]
        if np.linalg.norm(off) > RESTEP_M or abs(E.yaw_of(Rp)) > math.radians(RESTEP_DEG):
            tape.phase_now = "step"
            dh = E.rot_z(E.yaw_of(Rp)).T @ np.r_[off, 0.0]
            GR.open_walk(m, float(dh[0]), float(dh[1]), float(-E.yaw_of(Rp)))
            n_restep += 1
            if m.fell_at is not None:
                res["fell"] = f"restep {j}"
                break
            tape.phase_now = "aim"
        m._stand()  # re-anchor the stand reference and the leg odometry where the robot now stands
        rec.start(side)
        m.hold(GR.PRE_S)
        p1, Rp1 = env.pelvis()
        yaw1 = E.yaw_of(Rp1)
        Rh1 = E.rot_z(yaw1)
        jaw_w = np.array([PX - CAP_OUT + DEPTH - TIP, tw["y"], tw["z"]])
        g = np.r_[(Rh1.T @ np.r_[jaw_w[:2] - p1[:2], 0.0])[:2], jaw_w[2]]  # the sighting, heading frame
        pz = AE.pelvis_for(g[2])
        tb = g.copy()
        pr = dict(i=j, target=name, side=side, pz=pz, goal_h=g.tolist(), k0=len(tape.q), stance_off_cm=(100 * off).tolist())
        tic = time.time()
        if method == "lookup":
            off, n_nb = lookup(models["lookup"], sd, pz, g)
            tb = g - off
            pr.update(lookup_off_cm=(100 * off).tolist(), lookup_n=n_nb)
        if method in ("selfmodel", "selfmodel_servo_odo"):
            start = np.r_[p1, yaw1].astype(np.float32)
            st = dict(sd=sd, p=p1, yaw=yaw1, Rh=Rh1, goal=g.astype(np.float32).astype(np.float64), rot_w=Rh1 @ rb,
                      rot6=AE.rot6_of(rb), c2=dict(side=side, dyaw=0.0, pitch=0.0), start=start, walked=True,
                      x0b=SE.x0_side(rec, m, side, sd, True, start))  # fmt: skip
            evs = WI.fixed_point(models["post"], m, st, st["goal"], pz, None)
            best = min(evs, key=lambda ev: ev["l_pos"])
            pr.update(pred_naive_w=to_w(p1, Rh1, evs[0]["pos"]).tolist(), pred_w=to_w(p1, Rh1, best["pos"]).tolist(),
                      pred_std_cm=float(best["pred_std"]), pred_err_cm=float(best["pred_err"]),
                      pred_naive_err_cm=float(evs[0]["pred_err"]))  # fmt: skip
            tb = np.asarray(best["t"], float)
        pr["think_s"] = time.time() - tic
        target_w = to_w(p1, Rh1, tb)
        pr["cmd_w"] = target_w.tolist()
        tape.peak[:] = 0.0
        tape.phase_now = "press"
        cfg = m.cfg
        if method == "selfmodel_servo_odo":  # invert_eval's aim_ff_servo: the model's command as the reference, the
            # servo pulling the measured jaw centre to the wanted point; full settle, the jaw never meets the aimed point
            import dataclasses

            import invert_eval as IE

            m.servo_target = IE.KeepTarget({side: to_w(p1, Rh1, g)})
            m.cfg = dataclasses.replace(cfg, reach_tolerance=-1.0)
        jf = models.get("jf")
        if method in ODO_SERVO:  # the servo measures the jaw from joints + leg odometry, anchored at this sighting
            from locomani.backends.scalebfm import motor as MM

            jf.anchor(p1, yaw1, MM.WRIST, MM.R.OFF)
            m.jaw_source = jf.jaw
        servo = method in ("servo",) + ODO_SERVO
        out = m.reach(side, target_w, Rh1 @ rb, pelvis_z=None if pz < 0 else float(pz), servo=servo)
        if method in ODO_SERVO:
            pr["jaw_est_err_cm"] = 100 * float(np.linalg.norm(jf.jaw(side)[0] - m.jaw_centre(side)[0]))
            jf.on, m.jaw_source = False, None
        m.cfg, m.servo_target = cfg, {}
        if m.fell_at is None:
            m.hold(GR.POST_S)
        rec.on = False
        tip = tip_of(m, side)
        hit_k = names.index(name)
        pressed = [names[k] for k in np.flatnonzero(tape.peak >= PRESS_M)]
        pr.update(ok=bool(out.ok), tip_w=tip.tolist(), err_yz_cm=100 * float(np.hypot(tip[1] - tw["y"], tip[2] - tw["z"])),
                  depth_cm=100 * float(tape.peak[hit_k]), hit=bool(tape.peak[hit_k] >= PRESS_M),
                  wrong=[n for n in pressed if n != name], k1=len(tape.q), fell=m.fell_at is not None)  # fmt: skip
        tape.phase_now = "home"
        tape.lit_now = -1
        if m.fell_at is None:
            GR.home(m)
        late = [names[k] for k in np.flatnonzero(tape.peak >= PRESS_M)]
        pr["wrong"] = sorted(set(pr["wrong"]) | {n for n in late if n != name})
        res["presses"].append(pr)
        print(f"  {method:9s} {j:2d} {name:4s} {side:5s} hit {pr['hit']!s:5s} yz {pr['err_yz_cm']:5.2f} cm depth "
              f"{pr['depth_cm']:4.1f} cm wrong {pr['wrong']} stance off {np.round(100 * off, 1)} think {pr['think_s']:.1f}s",
              flush=True)  # fmt: skip
        if m.fell_at is not None:
            res["fell"] = f"press {j}"
            break
    res["n_restep"] = n_restep
    tape.on = False
    return dict(res, q=tape)


def run(layout: str, seq: int, methods: list[str]) -> None:
    import approach_eval as AE
    import gen_reach as GR
    import torch

    torch.set_num_threads(1)
    E = AE.E
    env = E.Env("open")
    names = [t["name"] for t in targets(layout)]
    rec, tape = GR.Recorder(env), Tape(env, names)
    models = {"post": torch.load(AE.WM_POST, map_location="cpu", weights_only=False).eval()}
    ctrl = os.environ.get("REACH_CONTROLLER", "scalebfm")
    if any(mm in ODO_SERVO for mm in methods):
        import leg_odo as LO

        models["jf"] = LO.JawFeed(env)
    if "lookup" in methods:
        models["lookup"] = dict(np.load(OUT / f"lookup_{ctrl}.npz"))
    env.reset(None)  # scalebfm-first-call-nondeterminism: warm up
    env.m.hold(1.0)
    tag = f"{ctrl}_{layout}_{seq}"
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    seq_names = sequence(layout, seq)
    print(f"{tag}: {' '.join(seq_names)}", flush=True)
    for meth in methods:
        tic = time.time()
        r = episode(env, rec, tape, models, layout, seq_names, meth)
        t = r.pop("q")
        np.savez_compressed(d / f"{meth}.npz", q=np.array(t.q), lit=np.array(t.lit), phase=np.array(t.phase))
        P = r["presses"]
        r.update(controller=ctrl, wall_s=time.time() - tic, n_hit=sum(p["hit"] for p in P), n_wrong=sum(len(p["wrong"]) for p in P),
                 med_err_yz_cm=float(np.median([p["err_yz_cm"] for p in P])) if P else None)  # fmt: skip
        json.dump(r, open(d / f"{meth}.json", "w"), indent=1)
        print(f"{tag} {meth}: hit {r['n_hit']}/{len(seq_names)} wrong {r['n_wrong']} median yz {r['med_err_yz_cm']} cm "
              f"fell {r.get('fell')} | {r['wall_s']:.0f} s", flush=True)  # fmt: skip


# -- video ----------------------------------------------------------------------------------------------------------
def tone(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    env_ = np.exp(-3.2 * t) * np.minimum(1.0, t / 0.005)
    return env_ * sum(a * np.sin(2 * math.pi * freq * h * t) for h, a in ((1, 1.0), (2, 0.45), (3, 0.2), (4, 0.1)))


FREQ = {"C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25}


GHOST_PLAIN, GHOST_OURS = (0.95, 0.30, 0.20, 0.45), (0.15, 0.80, 0.35, 0.55)


def ball(scn, pos, rad: float, rgba) -> None:
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))  # fmt: skip
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def video(tag: str, speed: float) -> None:
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    d = OUT / tag
    layout = "piano" if "_piano_" in tag else "wall"
    T = targets(layout)
    names = [t["name"] for t in T]
    m = mujoco.MjModel.from_xml_path(str(OUT / f"scene_{layout}" / "scene_locomani.xml"))
    data = mujoco.MjData(m)
    gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n + "_g") for n in names]
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_j")] for n in names]
    idle = KEY_IDLE if layout == "piano" else IDLE
    W, H, STRIP = 640, 480, 64
    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # from the robot's right, ahead of its torso, so both hands on the panel are in view (11-17: the first camera sat
    # behind the robot and its body hid the keyboard)
    look, dist, az, el = {"wall": ((0.80, 0.0, 0.80), 2.4, 60.0, -14.0), "piano": ((0.85, 0.0, 0.92), 2.0, 62.0, -20.0)}[layout]
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    ctrl = tag.rsplit("_", 2)[0]
    # the servo measures the jaw with the simulator's wrist pose (Motor.jaw_centre): an oracle, labelled as one
    title = {"direct": "Controller alone", "servo": "Servo on sim ground truth (oracle)", "selfmodel": "Self-model (ours)",
             "servo_odo": "Servo on joints + leg odometry", "lookup": "Offset lookup (no model)",
             "selfmodel_servo_odo": "Self-model + odometry servo"}
    sr, outs = 44100, []
    for meth in METHODS:
        if not (d / f"{meth}.npz").exists():
            continue
        z = np.load(d / f"{meth}.npz")
        R = json.load(open(d / f"{meth}.json"))
        q, lit = z["q"], z["lit"]
        done = {p["k1"]: p for p in R["presses"]}
        hits, tries, onsets = 0, 0, []
        sfx = "" if speed == 1 else f"_x{speed:g}"
        mp4, vid = d / f"{meth}{sfx}.mp4", d / f"{meth}{sfx}_video.mp4"
        # frames go straight into ffmpeg (11-17: a raw file of 1.5 GB per method filled a800's root disk)
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + STRIP}",
                                "-r", str(FPS * speed), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(vid)],
                               stdin=subprocess.PIPE)  # fmt: skip
        was = np.zeros(len(names), bool)
        for k in range(len(q)):
            data.qpos[:] = q[k]
            mujoco.mj_forward(m, data)
            disp = np.array([q[k][a] for a in adr])
            down = disp >= PRESS_M
            for i in np.flatnonzero(down & ~was):
                onsets.append((k / FPS, names[i]))
            was = down
            for i, g in enumerate(gid):
                c = idle
                if down[i]:
                    c = HIT if i == lit[k] else MISS
                elif i == lit[k]:
                    c = LIT
                m.geom_rgba[g, :3] = c
            r.update_scene(data, camera=cam)
            for p in R["presses"]:  # the self model's two predictions while this press is on: plain command, corrected
                if "pred_w" in p and p["k0"] <= k < p["k1"]:
                    for w, rgba in ((p["pred_naive_w"], GHOST_PLAIN), (p["pred_w"], GHOST_OURS)):
                        ball(r.scene, np.asarray(w, float) + np.array([TIP, 0.0, 0.0]), 0.018, rgba)
            img = Image.new("RGB", (W, H + STRIP), (250, 250, 248))
            img.paste(Image.fromarray(r.render()), (0, STRIP))
            if k in done:
                tries += 1
                hits += int(done[k]["hit"])
            dr = ImageDraw.Draw(img)
            dr.text((12, 8), title[meth], font=bold, fill=(20, 20, 20))
            dr.text((12, 36), f"{ctrl} | {'hits' if layout == 'wall' else 'right notes'} {hits}/{tries} of {len(R['seq'])}",
                    font=font, fill=(60, 60, 60))  # fmt: skip
            if meth.startswith("selfmodel"):
                dr.text((W - 300, 10), "red: predicted landing of the plain command", font=small, fill=(200, 60, 40))
                dr.text((W - 300, 32), "green: predicted landing after correction", font=small, fill=(30, 140, 60))
            enc.stdin.write(np.asarray(img).tobytes())
        enc.stdin.close()
        assert enc.wait() == 0, f"ffmpeg failed on {meth}"
        n_frames = len(q)
        if layout == "piano" and onsets:
            n = int((n_frames / FPS / speed + 2.0) * sr)
            a = np.zeros(n)
            for t0, note in onsets:
                s = tone(FREQ[note], 1.2, sr)
                i0 = int(t0 / speed * sr)
                a[i0 : i0 + len(s)] += s[: max(0, n - i0)]
            a = (0.3 * a / max(1e-6, np.abs(a).max()) * 32767).astype(np.int16)
            wav = d / f"{meth}.pcm"
            wav.write_bytes(a.tobytes())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", str(wav),
                            "-c:v", "copy", "-c:a", "aac", "-shortest", str(mp4)], check=True)  # fmt: skip
            wav.unlink()
            vid.unlink()
        else:
            vid.replace(mp4)
        outs.append(mp4)
        print(meth, mp4, n_frames, "frames", f"onsets {len(onsets)}", flush=True)
    if len(outs) > 1:
        side = d / f"side_by_side_x{speed:g}.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for o in outs:
            cmd += ["-i", str(o)]
        flt = "".join(f"[{i}:v]" for i in range(len(outs))) + f"hstack=inputs={len(outs)}[v]"
        cmd += ["-filter_complex", flt, "-map", "[v]"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(side)]
        subprocess.run(cmd, check=True)
        print("side by side", side)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["scene", "table", "run", "video"])
    ap.add_argument("--layout", default="wall", choices=["wall", "piano"])
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    if a.phase == "scene":
        write_scenes()
    elif a.phase == "table":
        write_lookup()
    elif a.phase == "run":
        run(a.layout, a.seq, a.methods.split(","))
    else:
        video(a.tag, a.speed)
