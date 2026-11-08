"""The hug-lift height probe's skills package (scripts/rsi/probe_hug_lift.py, 2026-11-07): not a seed and not in any
sandbox. It answers "is the two-hand put-in's raise limited by the load's mass or by the hands' height?": the box is
given MASS_KG, hugged with `bimanual.pick` of the main library or of LIBRARY (an RSI incumbent's skills) (stance, crouch, pre-grasp, entry, squeeze, the
15 cm lift), then both jaw centres are raised rung by rung to the absolute heights in RUNGS_Z with the move the put-in's
raise uses (`bimanual._shift_both`: wrists as they are, the squeeze kept, no pelvis target, the library's raise speed), no re-squeeze
and no retry. After each rung the hands hold HOLD_S and one `rung` step is noted: the jaw centres' height and error,
the reach's attitude and IK error, the box's underside, its tilt and its slip against the hands, which hands still
touch it, the pelvis height. The ladder stops when a hand leaves the box or the robot falls.

Loaded by scripts/rsi/scene_round.py like any skills package (`plan_for`, `apply_set`, `parameters`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene.skills import bimanual as MAIN_BM

LIBRARY = ""  # a skills package's directory whose bimanual.py hugs (an RSI incumbent's); "" is the main library
MASS_KG = 0.8  # the box's mass for this run (--set '{"MASS_KG": 0.3}'); its inertia is scaled with it
RUNGS_Z = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]  # mean jaw-centre heights (world z), skipped while below the hands
HOLD_S = 1.0
SIDES = MAIN_BM.SIDES

_orig: dict = {}  # body id -> (mass, inertia) as the scene had them (the world is reused across rounds)


def _set_mass(world, body: int, mass: float) -> dict:
    m = world.sim.m
    for b, (mass0, inertia0) in _orig.items():  # every body an earlier round changed goes back first
        m.body_mass[b], m.body_inertia[b] = mass0, inertia0
    if body not in _orig:
        _orig[body] = (float(m.body_mass[body]), m.body_inertia[body].copy())
    mass0, inertia0 = _orig[body]
    m.body_mass[body] = mass
    m.body_inertia[body] = inertia0 * (mass / mass0)
    return dict(mass_scene_kg=round(mass0, 3), mass_kg=round(float(mass), 3))


def _library():
    """The bimanual module of LIBRARY (loaded once per path as package `probe_hug_lib`), or the main library's."""
    if not LIBRARY:
        return MAIN_BM
    name = "probe_hug_lib"
    pkg = sys.modules.get(name)
    if pkg is None or getattr(pkg, "_root", None) != str(LIBRARY):
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        root = Path(LIBRARY)
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
        pkg._root = str(LIBRARY)
    return pkg.bimanual


def _tilt_deg(quat) -> float:
    up = R.mat(np.asarray(quat, float))[:, 2]
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def _mean_jaw(m) -> np.ndarray:
    return np.mean([m.jaw_centre(s)[0] for s in SIDES], axis=0)


def plan_for(task):
    return _hug_ladder if task.objects else None


def _hug_ladder(ctx) -> Outcome:
    world, m = ctx.world, ctx.motor
    BM = _library()
    key = ctx.task.objects[0]
    body = world.free[key]
    ctx.note("set_mass", Outcome(True, t0=m.t, t1=m.t, info=_set_mass(world, body, float(MASS_KG))))
    out = BM.pick(ctx, key)
    touching = sorted(m.touching(body))
    ctx.note("hugged", Outcome(out.ok, out.why, m.t, m.t), touching=touching, pick_ok=out.ok)
    if touching != list(SIDES) or m.fell_at is not None:
        return Outcome(False, f"no hug to raise: {out.why or 'a hand is off the box'}", out.t0, out.t1)

    rel0 = BM.box_centre(world, key) - _mean_jaw(m)  # the box against the hands at the end of the hug pick
    jaw0, under0 = _mean_jaw(m), BM.underside(world, key)
    top = dict(jaw_z=float(jaw0[2]), under_z=under0, rung=None)
    # the put-in's raise speed: the library's per-object choice (`hold`, from the scene file) when it has one
    speed = float(BM.hold(world, key)["speed"]) if hasattr(BM, "hold") else BM.LIFT_SPEED
    ctx.note("ladder_plan", Outcome(True, t0=m.t, t1=m.t), speed=speed, library=LIBRARY or "main")
    for target in RUNGS_Z:
        if target <= float(_mean_jaw(m)[2]) + 0.02:
            continue
        dz = target - float(_mean_jaw(m)[2])
        goal = {s: m.jaw_centre(s)[0] + np.array([0.0, 0.0, dz]) for s in SIDES}
        o = BM._reach_both(m, goal, None, keep_squeeze=True, speed=speed)
        if m.fell_at is None:
            m.hold(HOLD_S)
        jaws = {s: m.jaw_centre(s)[0] for s in SIDES}
        touching = sorted(m.touching(body))
        rel = BM.box_centre(world, key) - _mean_jaw(m)
        under = BM.underside(world, key)
        values = dict(
            target_z=round(target, 3),
            jaw_z=round(float(_mean_jaw(m)[2]), 4),
            jaw_dz_cm=round(100 * (float(_mean_jaw(m)[2]) - target), 2),  # achieved minus asked, height only
            **{f"{s}_jaw_err_cm": round(100 * float(np.linalg.norm(jaws[s] - goal[s])), 2) for s in SIDES},
            **{f"{s}_dz_cm": round(100 * float(jaws[s][2] - goal[s][2]), 2) for s in SIDES},  # height part
            **{f"{s}_dxy_cm": round(100 * float(np.linalg.norm(jaws[s][:2] - goal[s][:2])), 2) for s in SIDES},
            reach_jaw_err_cm=o.info.get("jaw_err_cm"),
            attitude_err_deg=o.info.get("attitude_err_deg"),
            ik_err_cm=o.info.get("ik_err_cm"),
            under_z=round(under, 4),
            box_rise_cm=round(100 * (under - under0), 2),
            hands_rise_cm=round(100 * (float(_mean_jaw(m)[2]) - float(jaw0[2])), 2),
            slip_cm=round(100 * float(np.linalg.norm(rel - rel0)), 2),
            slip_dz_cm=round(100 * float(rel[2] - rel0[2]), 2),  # negative: the box sank in the hands
            tilt_deg=round(_tilt_deg(world.object_pose(key)[1]), 1),
            hand_gap_cm=round(100 * float(np.linalg.norm(jaws["left"] - jaws["right"])), 2),
            pelvis_z=round(float(m.pelvis()[0][2]), 3),
            touching=touching,
            fell=m.fell_at is not None,
        )
        held = touching == list(SIDES) and m.fell_at is None
        ctx.note("rung", Outcome(held and o.ok, o.why, o.t0, m.t), **values)
        if not held:
            break
        top = dict(jaw_z=float(_mean_jaw(m)[2]), under_z=under, rung=target)
    ctx.note("ladder_top", Outcome(True, t0=m.t, t1=m.t),
             jaw_z=round(top["jaw_z"], 4), under_z=round(top["under_z"], 4), rung=top["rung"])  # fmt: skip
    return Outcome(True, "", m.t, m.t)


def _overridable(v) -> bool:
    return isinstance(v, (int, float, str, bool, list, tuple, dict)) or v is None


def apply_set(package, overrides: dict) -> list[str]:
    done = []
    for key, value in (overrides or {}).items():
        if not hasattr(package, key) or not key.isupper():
            raise KeyError(f"{key}: no parameter {key!r} in {package.__name__}")
        setattr(package, key, value)
        done.append(key)
    return done


def parameters(package) -> dict:
    return {k: v for k, v in vars(package).items() if k.isupper() and _overridable(v)}
