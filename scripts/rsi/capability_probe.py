"""TODO 27.3 (plan docs/handoff/scene-rsi-plan-20261016.md §9, D10): phase 1 -- closed-loop probes of the mars +
ScaleBFM motor layer (src/locomani/backends/scalebfm/motor.py) over the ranges the ADT apartment holds, for the
measured capability table. No skill code: each cell drives the primitives directly. One JSON per cell lands in
outputs/scene_rsi_20261016/capability/<kind>/<cell id>.json (a rerun skips cells already written).

Kinds:
  reach    walk_mars: one jaw centre to a point ahead of the pelvis -- side x wrist level / down x standing or after a
           squat to 0.62 / 0.52 m x height 0.05-1.55 m x forward 0.30-0.70 m; delivered error, attitude, fall
  posture  walk_mars: squat to a pelvis height, hold 10 s, stand up; lean references 10-50 deg
  loco     walk_mars: walks, back steps, sidesteps, turns, a 30 s stand
  grip     the probe scene (walk_mars + a table box + one free box, both sized per cell at runtime): stand facing the
           table edge EDGE_M ahead, pre-grasp PREGRASP_M short, enter, grip, lift 10 cm, hold 3 s; sweeps over width,
           mass (then step back, turn 90 deg, walk 1 m), support top (then put it back and release), inboard depth,
           object height, and a top-down pinch; three small placement jitters per cell

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/rsi/capability_probe.py run reach posture loco grip \\
        > /workspace/tmp/capability_probe.log 2>&1 &
    $PY scripts/rsi/capability_probe.py summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani import paths  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "capability"
WALK = REPO / "outputs" / "scene" / "walk_mars"
PROBE_SCENE = OUT / "probe_scene"
KINDS = ("reach", "posture", "loco", "grip", "envelope")

LATERAL_M = 0.15  # the reaching hand's target this far to its own side of the pelvis
EDGE_M = 0.28  # pelvis to the table edge in the grip probes (the pick skills stand 0.21-0.30 m off)
PREGRASP_M = 0.12  # jaw centre short of the object centre before the entry (pick.PREGRASP_BACK_M)
ENTRY_SPEED = 0.08  # m/s, the entry and the lift (pick.ENTRY_SPEED)
LIFT_M = 0.10
HOUSING_BELOW_M = 0.035  # side grasp: the jaw centre at least this far over the top (housing 3.1 cm below the jaw)
FINGERTIP_M = 0.033  # top-down: fingertips ahead of the jaw centre along the wrist x (pick.FINGERTIP_M)
TIP_CLEAR_M = 0.004  # top-down: fingertips this far over the surface at the grasp
VIA_OUT_M = 0.08  # pick.VIA_OUT_M
TILT_UPRIGHT_DEG = 15.0
JITTER = ((0.0, 0.0, 0.0), (0.01, -0.01, 5.0), (-0.01, 0.01, -5.0))  # object dx, dy (m), dyaw (deg)
GRIP_BASE = dict(top=0.75, width=0.06, depth=0.06, height=0.15, mass=0.3, friction=0.6, inboard=0.10, att="side",
                 carry=False, place=False)  # fmt: skip

PROBE_BODIES = """
    <body name="probe_table" pos="1.0 0 0.375">
      <geom name="probe_table_geom" type="box" size="0.3 0.6 0.375" rgba="0.55 0.45 0.35 1" friction="0.6 0.005 0.0001"/>
    </body>
    <body name="probe_object" pos="3.0 3.0 0.2">
      <freejoint name="probe_object_joint"/>
      <geom name="probe_object_geom" type="box" size="0.03 0.03 0.075" mass="0.3" rgba="0.2 0.5 0.8 1"
            friction="0.6 0.005 0.0001"/>
    </body>
"""


def _num(v: float, scale: int = 100, width: int = 3) -> str:
    return f"{int(round(v * scale)):0{width}d}"


def cells(kind: str) -> list[dict]:
    out: list[dict] = []
    if kind == "reach":
        for side in ("right", "left"):
            for att in ("level", "down"):
                for pel in (None, 0.62, 0.52):
                    zs = [0.05 + 0.1 * k for k in range(16 if pel is None else 10)]
                    for z in zs:
                        for x in (0.30, 0.40, 0.50, 0.60, 0.70):
                            tag = "stand" if pel is None else f"p{_num(pel)}"
                            out.append(dict(id=f"{side}_{att}_{tag}_z{_num(z)}_x{_num(x, width=2)}", side=side, att=att,
                                            pelvis=pel, z=round(z, 2), x=x))  # fmt: skip
            # where the forward reach ends: level, standing, beyond the 0.70 m every height above 0.35 m reached
            for z in [0.25 + 0.1 * k for k in range(11)]:
                for x in (0.80, 0.90):
                    out.append(dict(id=f"{side}_level_stand_z{_num(z)}_x{_num(x, width=2)}", side=side, att="level",
                                    pelvis=None, z=round(z, 2), x=x))  # fmt: skip
    elif kind == "envelope":  # TODO 33.1: the body model's dense single-arm envelope (not read by capability_write)
        for side in ("right", "left"):
            out_y = -1.0 if side == "right" else 1.0  # toward the hand's own side
            for att in ("level", "down", "tilt45"):
                for pel in (None, 0.68, 0.62, 0.56, 0.52):
                    zs = [0.05 + 0.1 * k for k in range(15 if pel is None else 12)]
                    for y in (-0.05, 0.05, 0.15, 0.25, 0.35):  # + is outward of the hand's own side
                        for z in zs:
                            for x in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                                tag = "stand" if pel is None else f"p{_num(pel)}"
                                out.append(dict(id=f"{side}_{att}_{tag}_y{_num(y + 0.1)}_z{_num(z)}_x{_num(x, width=2)}",
                                                side=side, att=att, pelvis=pel, y=round(out_y * y, 2), z=round(z, 2),
                                                x=x))  # fmt: skip
    elif kind == "posture":
        for p in (0.74, 0.70, 0.66, 0.62, 0.58, 0.54, 0.50, 0.46, 0.42, 0.38):
            out.append(dict(id=f"squat_p{_num(p)}", pelvis=p))
        for deg in (10, 20, 30, 40, 50):
            out.append(dict(id=f"lean_{deg:02d}", lean_deg=deg))
    elif kind == "loco":
        for d in (0.5, 1.0, 2.0, 4.0):
            out.append(dict(id=f"walk_{_num(d)}", op="walk", d=d))
        for d in (0.2, 0.4):
            out.append(dict(id=f"back_{_num(d)}", op="back", d=d))
        for s, sign in (("left", 1), ("right", -1)):
            for d in (0.1, 0.2, 0.4):
                out.append(dict(id=f"side_{s}_{_num(d)}", op="side", d=sign * d))
        for deg in (45, -45, 90, -90, 180):
            out.append(dict(id=f"turn_{'p' if deg > 0 else 'm'}{abs(deg):03d}", op="turn", deg=deg))
        out.append(dict(id="walk_diag_turn", op="walk_turn"))
        out.append(dict(id="stand_30s", op="stand", seconds=30.0))
    elif kind == "grip":
        sweeps = {
            "width": [0.015, 0.02, 0.03, 0.045, 0.06, 0.075, 0.085, 0.09, 0.095],
            "mass": [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            "top": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30],
            "inboard": [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
            "height": [0.03, 0.05, 0.08, 0.12, 0.20, 0.30],
        }
        extra = dict(mass=dict(carry=True), top=dict(place=True))
        for key, vals in sweeps.items():
            for v in vals:
                for j, jit in enumerate(JITTER):
                    c = dict(GRIP_BASE, **extra.get(key, {}), sweep=key, jitter=list(jit))
                    c[key] = v
                    if key == "width":
                        c["depth"] = max(0.04, min(v, 0.06))
                    out.append(dict(c, id=f"{key}_{_num(v, 1000, 4)}_j{j}"))
        for top in (0.40, 0.60, 0.75, 0.90, 1.10):
            for h in (0.02, 0.05):
                for j, jit in enumerate(JITTER):
                    c = dict(GRIP_BASE, sweep="top_down", top=top, height=h, width=0.05, depth=0.08, att="down",
                             jitter=list(jit))  # fmt: skip
                    out.append(dict(c, id=f"topdown_t{_num(top)}_h{_num(h)}_j{j}"))
    else:
        raise SystemExit(f"unknown kind {kind}")
    return out


def probe_scene() -> Path:
    """walk_mars plus the probe table and object (sized per cell at runtime). Written once, before the workers start."""
    xml = (WALK / "scene_locomani.xml").read_text()
    if xml.count("</worldbody>") != 1:
        raise SystemExit("walk_mars scene: expected one </worldbody>")
    new = xml.replace("</worldbody>", PROBE_BODIES + "  </worldbody>")
    PROBE_SCENE.mkdir(parents=True, exist_ok=True)
    for name, text in (("scene_meta.json", (WALK / "scene_meta.json").read_text()), ("scene_locomani.xml", new)):
        f = PROBE_SCENE / name
        if not f.exists() or f.read_text() != text:  # atomic: workers may start beside a rewrite
            tmp = f.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(text)
            tmp.rename(f)
    return PROBE_SCENE


# -- worker side (sim venv) ------------------------------------------------------------------------------------------


class Probe:
    def __init__(self, kind: str, controller: str = "scalebfm", sonic=None, scene_dir=None, dynamics: str = "scene"):
        """`controller` "scalebfm" (the default), "amo", "gr00t_wbc", or "sonic": SONIC v1.1 in process (backends.sonic_onnx), same scene,
        robot, joint order and Motor interface; `sonic` its SonicConfig (None: the defaults). `scene_dir`: the sim's
        scene instead of walk_mars (the whole-body IK keeps walk_mars' robot); `dynamics`: SonicSim's joint dynamics."""
        sys.path[:0] = [str(REPO / "src")]
        from locomani.backends.scalebfm import runtime as R
        from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
        from locomani.backends.scalebfm.motor import Motor, MotorConfig
        from locomani.tasks.scene.world import GRIP_LAW

        self.R, self.kind, self.controller = R, kind, controller
        scene = probe_scene() if kind == "grip" else (Path(scene_dir) if scene_dir else WALK)
        backend = ScaleBfmBackendConfig(scene_dir=str(scene), object_name="", **GRIP_LAW)
        if controller == "sonic":
            from locomani.backends.sonic_onnx.motor import SonicMotor
            from locomani.backends.sonic_onnx.runtime import SonicPolicy, SonicSim

            meta = json.load(open(R.META))
            policy = SonicPolicy(meta["joint_names"], sonic)
            sim = SonicSim(meta, backend=backend, dynamics=dynamics)
            self.m = SonicMotor(sim, policy, MotorConfig(), model_xml=str(WALK / "scene_locomani.xml"))
        elif controller == "amo":  # AMO's legs and waist, our arm IK per tick (backends.amo); AMO's own motor defaults
            from locomani.backends.amo.motor import AmoMotor
            from locomani.backends.amo.runtime import AmoPolicy, AmoSim

            policy = AmoPolicy("cuda")
            sim = AmoSim(backend, order="scalebfm")
            self.m = AmoMotor(sim, policy, None, model_xml=str(WALK / "scene_locomani.xml"))
        elif controller == "gr00t_wbc":  # NVIDIA's gear WBC (backends.gr00t_wbc): legs and waist, arms as AMO's
            from locomani.backends.gr00t_wbc.motor import VARIANTS, Gr00tMotor
            from locomani.backends.gr00t_wbc.runtime import Gr00tPolicy, Gr00tSim

            policy = Gr00tPolicy()
            sim = Gr00tSim(backend, order="scalebfm")
            self.m = Gr00tMotor(
                sim, policy, None, model_xml=str(WALK / "scene_locomani.xml"), gr00t=VARIANTS["gr00t_wbc"]
            )
        else:
            assert controller == "scalebfm", controller
            policy = R.Policy("cuda", ScaleBfmBackendConfig())
            sim = R.Sim(policy.meta, backend=backend)
            self.m = Motor(sim, policy, MotorConfig(), model_xml=str(WALK / "scene_locomani.xml"))
        self.sim = sim

    # small helpers
    def pelvis(self) -> dict:
        p, q = self.m.pelvis()
        return dict(xy=[round(float(v), 3) for v in p[:2]], z=round(float(p[2]), 3),
                    yaw_deg=round(math.degrees(self.R.yaw_of(q)), 1))  # fmt: skip

    def torso_pitch_deg(self) -> float:
        from locomani.backends.scalebfm.motor import tilt

        return round(math.degrees(tilt(self.sim.body("torso_link")[1])), 1)

    def feet_cm(self) -> float:
        from locomani.backends.scalebfm.motor import ANKLE

        a = [self.sim.body(ANKLE[s])[0] for s in ("left", "right")]
        return round(100 * float(np.linalg.norm(np.asarray(a[0])[:2] - np.asarray(a[1])[:2])), 1)

    def step(self, rows: list, name: str, out) -> bool:
        rows.append(dict(name=name, ok=out.ok, why=out.why, seconds=round(out.t1 - out.t0, 2), **out.info))
        return self.m.fell_at is None

    def run(self, c: dict) -> dict:
        return getattr(self, f"do_{self.kind}")(c)

    # kinds
    def do_reach(self, c: dict) -> dict:
        m, R = self.m, self.R
        m.reset((0.0, 0.0), 0.0)
        rows: list = []
        self.step(rows, "hold", m.hold(1.0))
        if c["pelvis"] is not None and not self.step(rows, "posture", m.posture(pelvis_z=c["pelvis"])):
            return dict(ok=False, fell=True, steps=rows, pelvis=self.pelvis())
        y = -LATERAL_M if c["side"] == "right" else LATERAL_M
        target = np.array([c["x"], y, c["z"]])
        rot = R.rot_y(np.pi / 2) if c["att"] == "down" else np.eye(3)
        out = m.reach(c["side"], target, rot, pelvis_z=c["pelvis"])
        alive = self.step(rows, "reach", out)
        if alive:
            self.step(rows, "hold", m.hold(1.5))
        jaw, Rj = m.jaw_centre(c["side"])
        from locomani.backends.scalebfm.motor import rotvec

        err_after = float(np.linalg.norm(jaw - target))
        att_after = math.degrees(float(np.linalg.norm(rotvec(rot @ Rj.T))))
        fell = m.fell_at is not None
        return dict(ok=bool(out.ok and not fell and err_after <= 0.02), fell=fell,
                    jaw_err_cm=out.info.get("jaw_err_cm"), attitude_err_deg=out.info.get("attitude_err_deg"),
                    ik_err_cm=out.info.get("ik_err_cm"), jaw_err_after_hold_cm=round(100 * err_after, 2),
                    attitude_after_hold_deg=round(att_after, 1), pelvis=self.pelvis(),
                    torso_pitch_deg=self.torso_pitch_deg(), steps=rows)  # fmt: skip

    def do_envelope(self, c: dict) -> dict:
        """do_reach on the dense grid: the target `y` given in the pelvis frame, and a 45 deg tilted attitude too."""
        m, R = self.m, self.R
        m.reset((0.0, 0.0), 0.0)
        rows: list = []
        self.step(rows, "hold", m.hold(1.0))
        if c["pelvis"] is not None and not self.step(rows, "posture", m.posture(pelvis_z=c["pelvis"])):
            return dict(ok=False, fell=True, steps=rows, pelvis=self.pelvis())
        target = np.array([c["x"], c["y"], c["z"]])
        rot = {"level": np.eye(3), "down": R.rot_y(np.pi / 2), "tilt45": R.rot_y(np.pi / 4)}[c["att"]]
        out = m.reach(c["side"], target, rot, pelvis_z=c["pelvis"])
        alive = self.step(rows, "reach", out)
        if alive:
            self.step(rows, "hold", m.hold(1.5))
        jaw, Rj = m.jaw_centre(c["side"])
        from locomani.backends.scalebfm.motor import rotvec

        err_after = float(np.linalg.norm(jaw - target))
        att_after = math.degrees(float(np.linalg.norm(rotvec(rot @ Rj.T))))
        fell = m.fell_at is not None
        return dict(ok=bool(out.ok and not fell and err_after <= 0.02), fell=fell,
                    jaw_err_cm=out.info.get("jaw_err_cm"), attitude_err_deg=out.info.get("attitude_err_deg"),
                    ik_err_cm=out.info.get("ik_err_cm"), jaw_err_after_hold_cm=round(100 * err_after, 2),
                    attitude_after_hold_deg=round(att_after, 1), pelvis=self.pelvis(),
                    torso_pitch_deg=self.torso_pitch_deg(), feet_cm=self.feet_cm(), steps=rows)  # fmt: skip

    def do_posture(self, c: dict) -> dict:
        m = self.m
        m.reset((0.0, 0.0), 0.0)
        rows: list = []
        self.step(rows, "hold", m.hold(1.0))
        if "pelvis" in c:
            alive = self.step(rows, "posture", m.posture(pelvis_z=c["pelvis"]))
            at = dict(pelvis=self.pelvis(), torso_pitch_deg=self.torso_pitch_deg(), feet_cm=self.feet_cm())
            if alive:
                alive = self.step(rows, "hold_10s", m.hold(10.0))
            held = dict(pelvis=self.pelvis(), torso_pitch_deg=self.torso_pitch_deg(), feet_cm=self.feet_cm())
            if alive:
                alive = self.step(rows, "stand_up", m.posture())
            fell = m.fell_at is not None
            return dict(ok=not fell and abs(at["pelvis"]["z"] - c["pelvis"]) <= 0.03, fell=fell, at=at, after_10s=held,
                        end=dict(pelvis=self.pelvis(), torso_pitch_deg=self.torso_pitch_deg()), steps=rows)  # fmt: skip
        alive = self.step(rows, "lean", m.posture(torso_pitch=math.radians(c["lean_deg"])))
        at = dict(pelvis=self.pelvis(), torso_pitch_deg=self.torso_pitch_deg())
        if alive:
            self.step(rows, "stand_up", m.posture())
        fell = m.fell_at is not None
        return dict(ok=not fell, fell=fell, at=at, steps=rows)

    def do_loco(self, c: dict) -> dict:
        m = self.m
        m.reset((0.0, 0.0), 0.0)
        rows: list = []
        self.step(rows, "hold", m.hold(1.0))
        start = self.pelvis()
        op = c["op"]
        if op == "walk":
            goal, yaw = (c["d"], 0.0), 0.0
            out = m.walk_to(goal, 0.0)
        elif op == "back":
            goal, yaw = (-c["d"], 0.0), 0.0
            out = m.sidestep(forward=-c["d"])
        elif op == "side":
            goal, yaw = (0.0, c["d"]), 0.0
            out = m.sidestep(left=c["d"])
        elif op == "turn":
            goal, yaw = (0.0, 0.0), math.radians(c["deg"])
            out = m.turn(yaw)
        elif op == "walk_turn":
            goal, yaw = (1.0, 1.5), math.pi / 2
            out = m.walk_to(goal, yaw)
        else:
            goal, yaw = tuple(start["xy"]), 0.0
            out = m.hold(c["seconds"])
        self.step(rows, op, out)
        if m.fell_at is None:
            self.step(rows, "settle", m.hold(1.0))
        end = self.pelvis()
        fell = m.fell_at is not None
        pos_err = float(np.hypot(end["xy"][0] - goal[0], end["xy"][1] - goal[1]))
        yaw_err = abs((end["yaw_deg"] - math.degrees(yaw) + 180) % 360 - 180)
        return dict(ok=bool(out.ok and not fell), fell=fell, pos_err_cm=round(100 * pos_err, 1),
                    yaw_err_deg=round(yaw_err, 1), seconds=round(out.t1 - out.t0, 2), start=start, end=end,
                    feet_cm=self.feet_cm(), steps=rows)  # fmt: skip

    def _size(self, c: dict) -> dict:
        """Set the table (top at c['top'], front edge EDGE_M ahead) and the object box (width across the jaw = world
        y, depth along the approach, height, mass, friction) in the model."""
        import mujoco

        m, d = self.sim.m, self.sim.d
        tb, tg = m.body("probe_table").id, m.geom("probe_table_geom").id
        ob, og = m.body("probe_object").id, m.geom("probe_object_geom").id
        th = c["top"] / 2
        m.body_pos[tb] = [EDGE_M + 0.3, 0.0, th]
        m.geom_size[tg] = [0.3, 0.6, th]
        half = np.array([c["depth"] / 2, c["width"] / 2, c["height"] / 2])
        m.geom_size[og] = half
        for g, h in ((tg, np.array([0.3, 0.6, th])), (og, half)):
            m.geom_rbound[g] = float(np.linalg.norm(h))
            m.geom_aabb[g] = np.r_[0.0, 0.0, 0.0, h]
        m.geom_friction[og][0] = c["friction"]
        m.body_mass[ob] = c["mass"]
        a, b, h = 2 * half
        m.body_inertia[ob] = c["mass"] / 12 * np.array([b * b + h * h, a * a + h * h, a * a + b * b])
        mujoco.mj_setConst(m, d)
        return dict(body=ob)

    def do_grip(self, c: dict) -> dict:
        R, m = self.R, self.m
        info = self._size(c)
        ob = info["body"]
        side = "right"
        m.reset((0.0, 0.0), 0.0)
        dx, dy, dyaw = c["jitter"]
        centre = np.array([EDGE_M + c["inboard"] + dx, -LATERAL_M + dy, c["top"] + c["height"] / 2 + 0.001])
        m.place("probe_object", centre, R.quat_wxyz(R.rot_z(math.radians(dyaw))))
        rows: list = []
        self.step(rows, "settle", m.hold(1.0))
        start = self.sim.d.xpos[ob].copy()
        if c["att"] == "down":
            rot = R.rot_y(np.pi / 2)
            z_g = c["top"] + FINGERTIP_M + TIP_CLEAR_M
            grasp = np.r_[start[:2], z_g]
            pre = grasp + np.array([0.0, 0.0, 0.10])
        else:
            rot = np.eye(3)
            z_g = c["top"] + max(HOUSING_BELOW_M, min(c["height"] / 2, 0.10))
            grasp = np.r_[start[0] + 0.01, start[1], z_g]
            pre = grasp - np.array([PREGRASP_M + 0.01, 0.0, 0.0])
        pel = None if z_g >= 0.60 else max(0.42, round(z_g + 0.05, 2))
        alive = True
        if pel is not None:
            alive = self.step(rows, "posture", m.posture(pelvis_z=pel))
        speed = 0.05 if c["att"] == "down" else ENTRY_SPEED
        # a pre-grasp over the table is reached through a point VIA_OUT_M outside the edge at its height, as the pick
        # skills' edge_via does: straight from the hanging hand it would cut through the edge
        via = [np.array([EDGE_M - VIA_OUT_M, pre[1], pre[2]])] if pre[0] > EDGE_M - VIA_OUT_M else []
        seq = [("pregrasp", lambda: m.reach(side, pre, rot, via=via, pelvis_z=pel)),
               ("enter", lambda: m.reach(side, grasp, rot, speed=speed, pelvis_z=pel)),
               ("grip", lambda: m.grip(side))]  # fmt: skip
        for name, fn in seq:
            if not alive:
                break
            alive = self.step(rows, name, fn())
        before_lift = self.sim.d.xpos[ob].copy()
        knocked = float(np.linalg.norm(before_lift[:2] - start[:2]))
        if alive:
            lift = m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_M])
            alive = self.step(rows, "lift", m.reach(side, lift, rot, speed=ENTRY_SPEED, pelvis_z=pel))
        rel0 = self.sim.d.xpos[ob] - m.jaw_centre(side)[0]
        if alive:
            alive = self.step(rows, "hold", m.hold(3.0))
        rel1 = self.sim.d.xpos[ob] - m.jaw_centre(side)[0]
        obj = self.sim.d.xpos[ob]
        gripped = side in m.gripping(ob)
        lifted = bool(gripped and obj[2] > start[2] + 0.05)
        res = dict(z_grasp=round(float(z_g), 3), pelvis_used=pel, knocked_cm=round(100 * knocked, 1),
                   rise_cm=round(100 * float(obj[2] - start[2]), 1), gripped=gripped, lifted=lifted,
                   slip_hold_cm=round(100 * float(np.linalg.norm(rel1 - rel0)), 2))  # fmt: skip
        if c["carry"] and lifted and alive:
            legs = []
            if pel is not None:
                legs.append(("stand_up", lambda: m.posture()))
            legs += [("step_back", lambda: m.sidestep(forward=-0.5)), ("turn_90", lambda: m.turn(math.pi / 2)),
                     ("walk_1m", lambda: m.walk_to(np.asarray(m.pelvis()[0][:2]) + np.array([0.0, 1.0]), math.pi / 2))]  # fmt: skip
            for name, fn in legs:
                rel_a = self.sim.d.xpos[ob] - m.jaw_centre(side)[0]
                alive = self.step(rows, name, fn())
                rel_b = self.sim.d.xpos[ob] - m.jaw_centre(side)[0]
                rows[-1].update(
                    held=side in m.gripping(ob), slip_cm=round(100 * float(np.linalg.norm(rel_b - rel_a)), 2)
                )
                if not alive or side not in m.gripping(ob):
                    break
            res["carried"] = bool(alive and side in m.gripping(ob) and self.sim.d.xpos[ob][2] > 0.3)
        if c["place"] and lifted and alive:
            down = m.jaw_centre(side)[0] - np.array([0.0, 0.0, LIFT_M - 0.005])
            for name, fn in (("lower", lambda: m.reach(side, down, rot, speed=0.05, pelvis_z=pel)),
                             ("release", lambda: m.release(side)),
                             ("retract", lambda: m.reach(side, m.jaw_centre(side)[0] - np.array([PREGRASP_M, 0.0, -0.03]),
                                                         rot, pelvis_z=pel))):  # fmt: skip
                alive = self.step(rows, name, fn())
                if not alive:
                    break
            from locomani.backends.scalebfm.motor import tilt

            q = self.sim.d.xquat[ob]
            rest = self.sim.d.xpos[ob]
            on_table = abs(float(rest[2]) - (c["top"] + c["height"] / 2)) < 0.02
            res["placed"] = bool(alive and on_table and math.degrees(tilt(q)) < TILT_UPRIGHT_DEG and not m.touching(ob))
            res["place_tilt_deg"] = round(math.degrees(tilt(q)), 1)
        fell = m.fell_at is not None
        res.update(ok=bool(lifted and not fell), fell=fell, pelvis=self.pelvis(), steps=rows)
        return res


def worker(kind: str, ids: list[str]) -> None:
    by = {c["id"]: c for c in cells(kind)}
    out = OUT / kind
    out.mkdir(parents=True, exist_ok=True)
    pr = Probe(kind)
    for i in ids:
        f = out / f"{i}.json"
        if f.exists():
            continue
        t0 = time.time()
        try:
            rec = pr.run(by[i])
        except Exception as e:  # noqa: BLE001 -- a cell that raises is recorded, the shard goes on
            rec = dict(ok=False, fell=False, error=f"{type(e).__name__}: {e}")
        rec.update(cell=by[i], wall_s=round(time.time() - t0, 1))
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, default=float))
        tmp.rename(f)
        print(i, "ok" if rec.get("ok") else "fail", rec.get("error", ""), flush=True)


# -- driver side ------------------------------------------------------------------------------------------------------


def run(kinds: list[str], procs: int, gpus: str, shard: int) -> None:
    probe_scene()
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    env.setdefault("TMPDIR", paths.TMP)
    queue = []
    for kind in kinds:
        todo = [c["id"] for c in cells(kind) if not (OUT / kind / f"{c['id']}.json").exists()]
        for k in range(0, len(todo), shard):
            queue.append(
                [str(SIM_PY), str(Path(__file__).resolve()), "worker", kind, "--ids", ",".join(todo[k : k + shard])]
            )
    devices = gpus.split(",")
    running: list = []
    log = OUT / "run.log"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(queue)} shards", flush=True)
    with log.open("a") as fh:
        while queue or running:
            while queue and len(running) < procs:
                dev = min(devices, key=lambda x: sum(g == x for _, g in running))
                p = subprocess.Popen(queue.pop(0), cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": dev}, stdout=fh,
                                     stderr=subprocess.STDOUT)  # fmt: skip
                running.append((p, dev))
            time.sleep(1.0)
            running = [(p, g) for p, g in running if p.poll() is None]
    print("done", flush=True)


def load(kind: str) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT / kind).glob("*.json"))}


def summary() -> dict:
    """Per kind, the numbers the capability table takes (capability_write.py reads this file)."""
    s: dict = {}
    r = load("reach")
    if r:
        env: dict = {}
        for rec in r.values():
            c = rec["cell"]
            tag = "stand" if c["pelvis"] is None else f"p{_num(c['pelvis'])}"
            env.setdefault(f"{c['side']}_{c['att']}_{tag}", {}).setdefault(f"{c['z']:.2f}", {})[f"{c['x']:.2f}"] = (
                "F" if rec.get("fell") else ("ok" if rec["ok"] else f"{rec.get('jaw_err_after_hold_cm', '?')}")
            )
        s["reach"] = dict(n=len(r), ok=sum(x["ok"] for x in r.values()), fell=sum(bool(x.get("fell")) for x in r.values()),
                          grid=env)  # fmt: skip
    p = load("posture")
    if p:
        s["posture"] = {
            k: dict(ok=v["ok"], fell=v["fell"], at=v.get("at"), after_10s=v.get("after_10s")) for k, v in p.items()
        }
    lo = load("loco")
    if lo:
        s["loco"] = {k: dict(ok=v["ok"], fell=v["fell"], pos_err_cm=v.get("pos_err_cm"), yaw_err_deg=v.get("yaw_err_deg"),
                             seconds=v.get("seconds"), feet_cm=v.get("feet_cm")) for k, v in lo.items()}  # fmt: skip
    g = load("grip")
    if g:
        by: dict = {}
        for k, v in g.items():
            c = v["cell"]
            key = c["sweep"]
            val = {"width": c["width"], "mass": c["mass"], "top": c["top"], "inboard": c["inboard"], "height": c["height"],
                   "top_down": f"{c['top']:.2f}/{c['height']:.2f}"}[key]  # fmt: skip
            row = by.setdefault(key, {}).setdefault(str(val), dict(n=0, lifted=0, fell=0, carried=0, placed=0, why=[]))
            row["n"] += 1
            row["lifted"] += bool(v.get("lifted"))
            row["fell"] += bool(v.get("fell"))
            row["carried"] += bool(v.get("carried"))
            row["placed"] += bool(v.get("placed"))
            if not v.get("ok"):
                bad = next((x for x in v.get("steps", []) if not x["ok"]), None)
                row["why"].append(v.get("error") or (f"{bad['name']}: {bad['why']}" if bad else "not lifted"))
        s["grip"] = by
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("kinds", nargs="+", choices=KINDS)
    r.add_argument("--procs", type=int, default=12)
    r.add_argument("--gpus", default="0,1")
    r.add_argument("--shard", type=int, default=12, help="cells per worker process")
    w = sub.add_parser("worker")
    w.add_argument("kind", choices=KINDS)
    w.add_argument("--ids", required=True)
    sub.add_parser("summary")
    c = sub.add_parser("cells")
    c.add_argument("kind", choices=KINDS)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.kinds, a.procs, a.gpus, a.shard)
    elif a.cmd == "worker":
        worker(a.kind, a.ids.split(","))
    elif a.cmd == "cells":
        print(json.dumps([x["id"] for x in cells(a.kind)]))
    else:
        s = summary()
        print(json.dumps({k: (v if k != "reach" else {kk: vv for kk, vv in v.items() if kk != "grid"}) for k, v in s.items()},
                         indent=1)[:6000])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
