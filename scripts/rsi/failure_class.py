"""TODO 33.15 (docs/handoff/scene-rsi-strategy-plan-20261021.md §3.5): every round's failure class and its events,
computed by code from the 50 Hz trace and the object's geometry, written into the round record
(`rec["failure_class"]`, `rec["events"]`; scene_round.run_round). A Reviewer reads the classes instead of re-deriving
them from curves, and looks hard only at "other" and at rounds whose events disagree with their class.

Events (seconds of simulated time; distances cm), from the trace first -- motor primitives, wrists, the jaw command,
the finger joints, the object's pose -- so skills that name their steps differently still classify. Two numbers come
from the motor layer's own outcomes in the round's steps, whatever the step is called: a walk's `stop_err_cm` and a
reach's `jaw_err_cm` (the measured jaw centre against its target; the trace has no reach targets).

Classes, in this order: ok; no_plan; time_limit; fell_walk (fell before the first reach) / fell; walk_off (the stance
walk stopped > WALK_OFF_CM off); pinch_reach / side_reach (the last reach before the grip ended > REACH_MISS_CM off;
pinch when the wrist points down at the grip); too_wide (graspability: the section across the jaw's closing axis at
the grip does not fit, and the object was pushed or tipped before the grip or the jaw closed on nothing); knocked
(moved > MOVED_CM or tipped > TILT_DEG before the grip); slip (gripped, then not held); miss_near (the jaw closed on
nothing near the object); other. Rounds without a trace classify from their steps alone (`events.source` "steps").

    from failure_class import classify      # scripts/rsi on sys.path
    cls, events = classify(rec, arrays, world, key)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import graspability as G  # noqa: E402

WALK_OFF_CM = 5.0
REACH_MISS_CM = 5.0
REACH_GAP_CM = 8.0  # no reach outcome in the steps: the wrist reference this far ahead of the wrist counts as a miss
MOVED_CM = 2.0
TILT_DEG = 15.0
EMPTY_CM = 0.5  # a jaw opening below this after the grip closed on nothing
SETTLE_T = 1.5  # the object's reference pose: after the round's settle hold
GRIP_OPEN = 0.0245  # runtime.GRIP_OPEN: the jaw command while open
CLOSE_DROP = 0.004  # a command this far below open is a close
OFF = np.array([0.1515, 0.0, 0.0])  # runtime.OFF: wrist frame -> jaw centre
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
SIDES = ("left", "right")
FK_STRIDE = 5


def _quat_mat(q) -> np.ndarray:
    return G._quat_mat(q)


class FK:
    """Wrist poses from a trace row (pelvis xyz + quat, the 29 joints in MUJOCO_JOINT_NAMES order, as Motor.joints
    writes them), on a private MjData of `model` (the scene's or robot.xml)."""

    def __init__(self, model):
        import mujoco

        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.mj = mujoco
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.bid = [model.body(n).id for n in WRISTS]

    def __call__(self, robot, joints) -> list[tuple[np.ndarray, np.ndarray]]:
        self.d.qpos[:3], self.d.qpos[3:7], self.d.qpos[self.qadr] = robot[:3], robot[3:7], joints
        self.mj.mj_kinematics(self.m, self.d)
        return [(self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()) for b in self.bid]


def fk_of(world) -> FK | None:
    """The world's FK, made once and kept on it (a SceneWorld's scene model, or a StaticScene's `fk`)."""
    if world is None:
        return None
    fk = getattr(world, "fk", None) or getattr(world, "_failure_fk", None)
    if fk is None and hasattr(world, "sim"):
        fk = FK(world.sim.m)
        world._failure_fk = fk
    return fk


class _At:
    """A world whose object poses are the trace's at one moment (graspability reads `object_pose`)."""

    def __init__(self, world, poses: dict):
        self.meta, self._w, self._p = world.meta, world, poses
        if hasattr(world, "sim"):
            self.sim = world.sim
        else:
            self.m = world.m

    def object_pose(self, key):
        return self._p[key] if key in self._p else self._w.object_pose(key)


def _arrays(arrays) -> dict | None:
    if arrays is None:
        return None
    keys = arrays.files if hasattr(arrays, "files") else list(arrays.keys())
    return {k: np.asarray(arrays[k]) for k in keys}


def _tilt(q0, q1) -> float:
    z0, z1 = _quat_mat(q0)[:, 2], _quat_mat(q1)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z0 @ z1, -1.0, 1.0))))


def _last_before(steps: list[dict], t: float | None, field: str) -> dict | None:
    """The last step that carries `field` and started before `t` (any step name)."""
    cand = [s for s in steps if s.get(field) is not None and (t is None or s.get("t0", 0.0) <= t + 1e-6)]
    return cand[-1] if cand else None


def _first_reach_t(steps: list[dict]) -> float | None:
    s = next((s for s in steps if s.get("jaw_err_cm") is not None), None)
    return None if s is None else float(s.get("t0", 0.0))


def _segments(names: np.ndarray, want: str) -> list[tuple[int, int]]:
    """[i0, i1) runs of frames whose primitive is `want`."""
    on = names == want
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def trace_events(rec: dict, z: dict, world, key: str) -> dict:
    t = z["t"].astype(float)
    names = z["primitives"][z["primitive"]] if len(z["primitives"]) else np.array([""] * len(t))
    ev: dict = dict(source="trace")
    body = world.meta["objects"][key]["body"] if world is not None and key in world.meta["objects"] else None
    obj = z.get(f"obj_{body}") if body is not None else None
    jaw = z["jaw"]
    closing = np.nonzero(jaw.min(1) < GRIP_OPEN - CLOSE_DROP)[0]
    gi = int(closing[0]) if len(closing) else None
    side = int(np.argmin(jaw[gi])) if gi is not None else 1
    ev["grip_t"] = None if gi is None else round(float(t[gi]), 2)
    ev["grip_side"] = SIDES[side]
    end = gi if gi is not None else len(t)
    walks = [s for s in _segments(names, "walk_to") if s[0] < end]
    reaches = [s for s in _segments(names, "reach") if s[0] < end]
    ev["walk_end_t"] = round(float(t[walks[-1][1] - 1]), 2) if walks else None
    ev["last_reach_end_t"] = round(float(t[reaches[-1][1] - 1]), 2) if reaches else None
    if gi is not None:
        after = [s for s in _segments(names, "reach") if s[0] > gi]
        ev["lift_t"] = round(float(t[after[0][0]]), 2) if after else None
        k = min(len(t) - 1, int(np.searchsorted(t, t[gi] + 1.0)))
        f = z["fingers"][k, 2 * side : 2 * side + 2]
        ev["opening_at_grip_cm"] = round(float(2 * (np.mean(f) + 0.023) * 100), 2)
    if reaches:
        i0, i1 = reaches[-1][0], reaches[-1][1] - 1
        w, r = z["wrists"][:, side], z["ref_wrists"][:, side]
        cmd = float(np.linalg.norm(r[i1] - r[i0]))
        ev["reach_track_ratio"] = round(float(np.linalg.norm(w[i1] - w[i0])) / cmd, 3) if cmd > 0.02 else None
        ev["reach_gap_cm"] = round(float(np.linalg.norm(r[i1] - w[i1]) * 100), 2)
    fk = fk_of(world)
    if obj is not None:
        s0 = min(len(t) - 1, int(np.searchsorted(t, SETTLE_T)))
        p0, q0 = obj[s0, :3], obj[s0, 3:7]
        d = np.linalg.norm(obj[:, :3] - p0, axis=1)
        d[:s0] = 0.0
        moved = np.nonzero(d > MOVED_CM / 100)[0]
        ev["obj_first_move_t"] = round(float(t[moved[0]]), 2) if len(moved) else None
        ev["obj_tilt_end_deg"] = round(_tilt(q0, obj[-1, 3:7]), 1)
        ev["obj_dz_end_cm"] = round(float((obj[-1, 2] - p0[2]) * 100), 2)
        pre = (gi if gi is not None else len(t)) - 1
        ev["obj_moved_before_grip_cm"] = round(float(d[: pre + 1].max() * 100), 2) if pre >= s0 else 0.0
        ev["obj_tilt_before_grip_deg"] = round(_tilt(q0, obj[max(pre, s0), 3:7]), 1)
        if len(moved):
            i = int(moved[0])
            dist = {"body": float(np.linalg.norm(z["robot"][i, :2] - obj[i, :2]))}
            if fk is not None:
                for s, (pw, Rw) in zip(SIDES, fk(z["robot"][i], z["joints"][i]), strict=True):
                    dist[f"{s}_hand"] = float(np.linalg.norm(pw + Rw @ OFF - obj[i, :3]))
            else:
                for j, s in enumerate(SIDES):
                    dist[f"{s}_hand"] = float(np.linalg.norm(z["wrists"][i, j] - obj[i, :3])) - 0.15
            hands = {k: v for k, v in dist.items() if k.endswith("hand")}
            hk = min(hands, key=hands.get)
            ev["obj_first_move_by"] = hk if hands[hk] < 0.12 else ("body" if dist["body"] < 0.6 else "other")
            ev["obj_first_move_dist_cm"] = {k: round(v * 100, 1) for k, v in dist.items()}
        if fk is not None:
            lo = walks[-1][1] - 1 if walks else 0
            hi = min(len(t), (gi if gi is not None else len(t) - 1) + 26)
            best = np.inf
            for i in range(lo, hi, FK_STRIDE):
                pw, Rw = fk(z["robot"][i], z["joints"][i])[side]
                best = min(best, float(np.linalg.norm(pw + Rw @ OFF - obj[i, :3])))
            ev["min_jaw_obj_cm"] = round(best * 100, 2) if np.isfinite(best) else None
            if gi is not None:
                pw, Rw = fk(z["robot"][gi], z["joints"][gi])[side]
                ev["jaw_obj_at_grip_cm"] = round(float(np.linalg.norm(pw + Rw @ OFF - obj[gi, :3])) * 100, 2)
                ev["pinch"] = bool(Rw[2, 0] < -0.7)
                cax = Rw[:2, 1]
                if np.linalg.norm(cax) > 0.3:
                    ref = reaches[-1][0] if reaches else max(s0, gi - 50)
                    at = _At(world, {key: (obj[ref, :3].astype(float), obj[ref, 3:7].astype(float))})
                    fit = G.fit_axis(at, key, cax)
                    ev["fit"] = {k: fit[k] for k in ("verdict", "width_m", "margin_m", "z")}
    return ev


def steps_events(rec: dict, world, key: str) -> dict:
    """What a record without a trace still tells: the grip's opening, the reach and walk errors, the approach's fit."""
    steps = rec.get("steps") or []
    ev: dict = dict(source="steps")
    grip = next((s for s in steps if s.get("opening_cm") is not None), None)
    if grip is not None:
        ev["grip_t"] = grip.get("t0")
        ev["opening_at_grip_cm"] = grip.get("opening_cm")
        ev["gripped_step"] = bool(grip.get("held"))
    st = next((s for s in steps if s.get("step") == "stance"), {})
    geom = next((s for s in steps if s.get("approach_deg") is not None), None)
    ev["pinch"] = bool(st.get("flat")) or any(s.get("step") == "descend" for s in steps)
    if world is not None and key in world.meta["objects"]:
        if ev["pinch"]:
            tp = G.top_pinch(world, key)
            if tp is not None:
                ev["fit"] = dict(verdict="fits" if tp["fits"] else "tight" if tp["margin_m"] >= 0 else "too_wide",
                                 width_m=tp["width_m"], margin_m=tp["margin_m"])  # fmt: skip
        elif geom is not None:
            fit = G.fit(world, key, np.radians(float(geom["approach_deg"])))
            ev["fit"] = {k: fit[k] for k in ("verdict", "width_m", "margin_m", "z")}
    return ev


def classify(rec: dict, arrays, world, key: str) -> tuple[str, dict]:
    """(failure class, events) of one round; see the module docstring."""
    steps = rec.get("steps") or []
    z = _arrays(arrays)
    if z is not None and len(z.get("t", [])) > 1:
        ev = trace_events(rec, z, world, key)
        hint = steps_events(rec, None, key)
        ev.update({k: v for k, v in hint.items() if k not in ev and k != "source"})
        ev["pinch"] = bool(ev.get("pinch")) or hint["pinch"]  # a pinch whose wrist never pointed down is still one
    else:
        ev = steps_events(rec, world, key)
    grip_t = ev.get("grip_t")
    walk = _last_before(steps, _first_reach_t(steps), "stop_err_cm")
    reach = _last_before(steps, grip_t, "jaw_err_cm") if grip_t is not None else _last_before(steps, None, "jaw_err_cm")
    ev["stop_err_cm"] = None if walk is None else walk.get("stop_err_cm")
    ev["reach_err_cm"] = None if reach is None else reach.get("jaw_err_cm")
    why = f"{rec.get('why') or ''} {rec.get('plan_why') or ''} {rec.get('error') or ''}"
    if rec.get("ok"):
        return "ok", ev
    if "no skill covers" in why or "plan_for raised" in why:
        return "no_plan", ev
    if "episode time limit" in why:
        return "time_limit", ev
    if rec.get("fell"):
        first = _first_reach_t(steps)
        fall_t = rec.get("t_end")
        last = steps[-1].get("step") if steps else ""
        early = (first is None and last in ("walk", "settle", "stance", "turn", "door")) or (
            first is not None and fall_t is not None and fall_t < first
        )
        return ("fell_walk" if early or (first is None and ev.get("grip_t") is None and "walk" in last) else "fell"), ev
    if ev["stop_err_cm"] is not None and ev["stop_err_cm"] > WALK_OFF_CM:
        return "walk_off", ev
    pushed = (ev.get("obj_moved_before_grip_cm") or 0.0) > MOVED_CM or (
        ev.get("obj_tilt_before_grip_deg") or 0.0
    ) > TILT_DEG
    empty = ev.get("opening_at_grip_cm") is not None and ev["opening_at_grip_cm"] < EMPTY_CM
    miss = ev["reach_err_cm"] is not None and ev["reach_err_cm"] > REACH_MISS_CM
    if ev["reach_err_cm"] is None and ev.get("reach_gap_cm") is not None:
        miss = ev["reach_gap_cm"] > REACH_GAP_CM
    if miss and not (pushed and ev.get("obj_first_move_t") is not None and grip_t is not None and ev["obj_first_move_t"] < grip_t
                     and (ev.get("fit") or {}).get("verdict") in ("tight", "too_wide")):  # fmt: skip
        return ("pinch_reach" if ev.get("pinch") else "side_reach"), ev
    fit_bad = (ev.get("fit") or {}).get("verdict") in ("tight", "too_wide")
    if fit_bad and (pushed or empty):
        return "too_wide", ev
    if pushed and (empty or grip_t is None):
        return "knocked", ev
    gripped = (ev.get("opening_at_grip_cm") or 0.0) >= EMPTY_CM or bool(ev.get("gripped_step"))
    if gripped and grip_t is not None:
        return "slip", ev
    if empty or grip_t is not None:
        return "miss_near", ev
    return "other", ev
