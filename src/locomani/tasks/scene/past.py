"""TODO 39 (plan docs/handoff/past-scene-graph-plan-20261030.md): the PAST recorder -- one local, action-conditioned
scene-graph record per skill step and retry of a round, built after the round from the step log, the primitive call log,
the trace (with the motor layer's contact channel, TODO 39.1) and the scene file. Pure functions: no simulator, no motor
layer; the builder never runs during a round and nothing in the skills package may read its output (rules.md 38).

Records separate observed facts (contacts, poses), the goal predicates (predicates.py) and derived geometry
(support, spatial and relative-pose edges); a relation that could not be computed is `unknown`, never false. Entities
enter a record's scope by relevance to the action (roles, neighbours, stance, corridor, overhang, regions), not by
contact; bodies touched during the step join with a timestamp.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locomani.scene_discovery import predicates as PR

SCHEMA = "PAST-locomani-0.1"
NEIGHBOUR_M = 0.30  # a movable object this close (xy) to the target is in scope
OVERHANG_M = 0.50  # a surface this far above the target is in scope
STANCE_M = 0.35  # the stance footprint's radius around walk_to's goal
CORRIDOR_HALF_M = 0.08  # the approach corridor's half width (pregrasp target -> object)
MOVED_CM = 2.0  # an object this far from where it was at the step's start has moved (failure_class.MOVED_CM)
SLIP_CM = 1.0  # a held object this much farther from the wrist slipped (digest.py's slip)
SUPPORT_BAND_M = 0.02  # an underside this close to a surface's top may rest on it
REACH_TOL_CM = 3.0  # a reach lands when the jaw centre is this close to its target (body.reach's miss)
WALK_TOL_CM = 5.0  # a walk lands when the pelvis is this close to the stance (failure_class.WALK_OFF_CM)
PREDICATE_DT_TICKS = 5  # predicates and derived edges are re-evaluated every 0.1 s (episode.JUDGE_DT)
SIDES = ("left", "right")
HAND = {s: f"{s}_hand" for s in SIDES}
FINGER = {s: (f"{s}_dex1_finger_1_link", f"{s}_dex1_finger_2_link") for s in SIDES}
WRIST = {s: f"{s}_wrist_yaw_link" for s in SIDES}
ARM_JOINTS = {s: [f"{s}_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                              "wrist_pitch", "wrist_yaw")] for s in SIDES}  # fmt: skip
WALK_KINDS = ("walk", "turn", "sidestep", "posture", "hold")
KIND_OF_CALL = {"walk_to": "walk", "turn": "turn", "sidestep": "sidestep", "posture": "posture", "hold": "hold",
                "reach": "reach", "grip": "grip", "release": "release", "squeeze": "squeeze"}  # fmt: skip


# -- steps, calls, kinds and roles --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    index: int
    attempt: int
    step: str
    t0: float
    t1: float
    row: dict


def intervals(steps: list[dict]) -> list[Interval]:
    """One interval per step row, in order; a stage() retry is its own row with `attempt`."""
    return [Interval(i, int(s.get("attempt") or 1), str(s.get("step", "")), float(s.get("t0", 0.0)),
                     float(s.get("t1", s.get("t0", 0.0))), s) for i, s in enumerate(steps)]  # fmt: skip


def calls_in(calls: list[dict], t0: float, t1: float) -> list[dict]:
    """The primitive calls whose start lies in [t0, t1)."""
    return [c for c in calls if t0 <= float(c.get("t0", -1)) < t1]


def kind_of(calls: list[dict]) -> str:
    kinds = {KIND_OF_CALL.get(c.get("name"), "mixed") for c in calls}
    if not kinds:
        return "none"
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed"


def side_of(call: dict) -> str | None:
    if call.get("name") not in ("reach", "grip", "release", "hand"):
        return None
    args = call.get("args") or []
    s = args[0] if args else (call.get("kwargs") or {}).get("side")
    return s if s in (*SIDES, "both") else None


def _effector(side: str | None) -> str | None:
    return None if side is None else ("both_hands" if side == "both" else HAND[side])


def roles_of(kind: str, calls: list[dict], task_objects: list[str], *, held_before: dict, positions: dict) -> dict:
    """target / effector / carried from the calls alone (rules.md 38: no step names, no skill declarations).
    `held_before[obj]` is the hand name holding obj at the step's start or None; `positions[obj]` its xyz then."""
    sides = [side_of(c) for c in calls if side_of(c)]
    effector = _effector(sides[0]) if sides else None
    carried = next((o for o in task_objects if held_before.get(o)), None)
    target = None
    if kind in ("release", "squeeze") or (kind == "reach" and carried):
        target = carried
    if target is None and kind == "reach":
        goal = next(((c.get("args") or [None, None]) + [None])[1] for c in calls if c.get("name") == "reach")
        if goal is not None and task_objects:
            target = min(
                task_objects,
                key=lambda o: float(np.linalg.norm(np.asarray(positions[o][:3], float) - np.asarray(goal[:3], float))),
            )
    if target is None and task_objects:
        target = task_objects[0]
    return dict(target=target, effector=effector, carried=carried)


# -- one tick of the trace ----------------------------------------------------------------------------------------------
def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrist_attitude(q) -> str:
    """level / tilted / down from the pitch of the wrist's x axis (the jaw points along +x): the body table's bands."""
    pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, -float(_quat_to_mat(q)[2, 0])))))
    return "down" if pitch_deg < -60 else ("tilted" if pitch_deg < -20 else "level")


class Frame:
    """Tick `k` of a round's arrays, read as the recorder needs it. Object keys are scene-file keys; `arrays` holds
    `obj_<body>` per tracked body. Without the contact channel (an older round) `has_contacts` is False and every
    contact question answers None."""

    def __init__(self, arrays: dict, meta: dict, k: int):
        self.a, self.meta, self.k = arrays, meta, int(k)
        self.t = float(arrays["t"][self.k])
        self.has_contacts = all(key in arrays for key in ("contact_k", "contact_a", "contact_b", "bodies"))
        self._bodies = [str(b) for b in arrays["bodies"]] if self.has_contacts else []
        self._arm_idx = self._arm_indices(arrays)
        self._contacts: set | None = None

    @staticmethod
    def _arm_indices(arrays) -> dict | None:
        try:
            from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
        except Exception:
            return None
        j = arrays.get("joints")
        if j is None or getattr(j, "ndim", 0) != 2 or j.shape[1] != len(MUJOCO_JOINT_NAMES):
            return None
        return {s: [MUJOCO_JOINT_NAMES.index(n) for n in ARM_JOINTS[s]] for s in SIDES}

    def body_name(self, key: str) -> str | None:
        return (self.meta.get("objects") or {}).get(key, {}).get("body")

    def obj_pose(self, key: str):
        arr = self.a.get(f"obj_{self.body_name(key)}")
        if arr is None:
            o = self.meta["objects"][key]
            return np.asarray(o.get("init_pos", (0, 0, 0)), float), np.asarray(o.get("init_quat", (1, 0, 0, 0)), float)
        return np.asarray(arr[self.k][:3], float), np.asarray(arr[self.k][3:7], float)

    def obj_vel(self, key: str) -> tuple[float, float]:
        """(linear m/s, angular rad/s) from the previous tick."""
        arr = self.a.get(f"obj_{self.body_name(key)}")
        if arr is None or self.k == 0:
            return 0.0, 0.0
        dt = float(self.a["t"][self.k] - self.a["t"][self.k - 1]) or 0.02
        v = float(np.linalg.norm(arr[self.k][:3] - arr[self.k - 1][:3]) / dt)
        r1, r0 = _quat_to_mat(arr[self.k][3:7]), _quat_to_mat(arr[self.k - 1][3:7])
        ang = math.acos(max(-1.0, min(1.0, (float(np.trace(r1 @ r0.T)) - 1) / 2)))
        return round(v, 4), round(ang / dt, 4)

    def hand_pos(self, side: str) -> np.ndarray:
        return np.asarray(self.a["wrists"][self.k][SIDES.index(side)], float)

    def ref_wrist(self, side: str) -> np.ndarray:
        return np.asarray(self.a["ref_wrists"][self.k][SIDES.index(side)], float)

    def jaw_m(self, side: str) -> float:
        i = SIDES.index(side)
        return float(2 * (np.mean(self.a["fingers"][self.k][2 * i : 2 * i + 2]) + 0.023))

    def arm_q(self, side: str):
        if not self._arm_idx:
            return None
        return [round(float(v), 4) for v in self.a["joints"][self.k][self._arm_idx[side]]]

    def pelvis(self):
        r = self.a["robot"][self.k]
        return np.asarray(r[:3], float), np.asarray(r[3:7], float)

    def contacts(self) -> set:
        if self._contacts is None:
            self._contacts = set()
            if self.has_contacts:
                for i in np.nonzero(np.asarray(self.a["contact_k"]) == self.k)[0]:
                    a, b = self._bodies[int(self.a["contact_a"][i])], self._bodies[int(self.a["contact_b"][i])]
                    self._contacts.add((a, b))
                    self._contacts.add((b, a))
        return self._contacts

    def touches(self, body_a: str | None, body_b: str | None) -> bool | None:
        if not self.has_contacts:
            return None
        if not body_a or not body_b:
            return None
        return (body_a, body_b) in self.contacts()

    def touching(self, key: str) -> set:
        b = self.body_name(key)
        return {s for s in SIDES if b and any(self.touches(b, f) for f in FINGER[s])}

    def gripped(self, key: str) -> set:
        b = self.body_name(key)
        return {s for s in SIDES if b and all(self.touches(b, f) for f in FINGER[s])}

    def state(self, keys) -> dict:
        """The predicates' state dict (predicates.py) for `keys`."""
        objects = {}
        for k in keys:
            pos, quat = self.obj_pose(k)
            objects[k] = dict(pos=pos, quat=quat, touching=sorted(self.touching(k)), gripped=sorted(self.gripped(k)))
        pos, quat = self.pelvis()
        return dict(objects=objects, pelvis=dict(pos=pos, quat=quat), fallen=False)


# -- scope: which entities a record holds ---------------------------------------------------------------------------------
def _surface_body(meta: dict, s: str) -> str | None:
    surf = meta["surfaces"][s]
    if surf.get("body"):
        return surf["body"]
    of = surf.get("of_object")
    return (meta["objects"].get(of) or {}).get("body") if of else None


def _footprint(xy, surf: dict) -> bool:
    half = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    return bool(np.all(np.abs(PR.to_surface_frame(np.asarray(xy, float), surf)) <= half + SUPPORT_BAND_M))


def support_of(frame: Frame, key: str, meta: dict) -> tuple[str | None, str]:
    """(surface key, src) the object rests on: the highest surface whose top is within SUPPORT_BAND_M of its underside
    and whose footprint holds it; `derived` when the contact channel can confirm the touch, `approx` when it cannot."""
    scene = PR._Scene(meta)
    xy, bottom = scene.underside(key, frame.state([key]))
    best = None
    for s, surf in meta["surfaces"].items():
        top = float(surf["top"])
        if top - SUPPORT_BAND_M <= bottom <= top + SUPPORT_BAND_M and _footprint(xy, surf):
            if best is None or float(meta["surfaces"][best]["top"]) < top:
                best = s
    if best is None:
        return None, "derived" if frame.has_contacts else "approx"
    touched = frame.touches(frame.body_name(key), _surface_body(meta, best))
    return best, "derived" if touched is not None else "approx"


def _entity_object(meta: dict, key: str) -> dict:
    o = meta["objects"][key]
    sem, box = o.get("semantic") or {}, o.get("box") or {}
    return dict(type="object", body=o.get("body"), mass_kg=sem.get("mass_kg"), half_extent_m=box.get("half_extent"),
                com_m=box.get("centre"), movable=bool(sem.get("movable", o.get("free"))),
                container=bool(o.get("container")), upright_axis=sem.get("upright_axis"))  # fmt: skip


def _entity_surface(meta: dict, s: str) -> dict:
    surf = meta["surfaces"][s]
    return dict(type="surface", of=surf.get("of_object"), top_m=surf["top"], half_extent_m=surf["half_extent"],
                yaw=surf.get("yaw", 0.0), approach_edges=surf.get("approach_edges"), fixed=True)  # fmt: skip


def _entity_furniture(meta: dict, key: str) -> dict:
    o = meta["objects"].get(key) or {}
    return dict(type="furniture", box_m=o.get("box"), fixed=True)


def _goal_targets(goal: list[str]) -> tuple[str | None, str | None]:
    """(surface of on(X, S), container of in(X, C)) named by the goal."""
    surface = container = None
    for g in goal or ():
        p = PR.parse(g)
        if p.name == "on":
            surface = p.args[1]
        elif p.name == "in":
            container = p.args[1]
    return surface, container


def _reach_goal(calls: list[dict]) -> np.ndarray | None:
    for c in calls:
        args = c.get("args") or []
        if c.get("name") == "reach" and len(args) > 1 and args[1] is not None:
            return np.asarray(args[1][:3], float)
    return None


def _walk_goal(calls: list[dict]) -> np.ndarray | None:
    for c in calls:
        if c.get("name") == "walk_to" and (c.get("args") or []):
            return np.asarray(c["args"][0][:2], float)
    return None


def _xy_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))


def _in_corridor(p, a, b, half: float) -> bool:
    """Point p (xy) within `half` of the segment a-b (xy)."""
    a, b, p = (np.asarray(v, float)[:2] for v in (a, b, p))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab))) <= half


def _static_xy(meta: dict, key: str) -> np.ndarray:
    o = meta["objects"][key]
    box = o.get("box") or {}
    return np.asarray(box.get("centre", (0, 0, 0)), float)[:2] + np.asarray(o.get("init_pos", (0, 0, 0)), float)[:2]


def scope_of(frame: Frame, roles: dict, kind: str, task: dict, meta: dict, calls: list[dict]) -> tuple[dict, dict]:
    """Spec section 4.4: the entities a record starts with, selected by relevance at the step's start, each with why."""
    entities, why = {}, {}
    objs = meta.get("objects") or {}
    tgt = roles.get("target")
    walk = kind in WALK_KINDS

    def add(eid: str, ent: dict, reason: str):
        if eid not in entities:
            entities[eid], why[eid] = ent, reason

    for o in task.get("objects") or ():
        if o in objs:
            add(o, _entity_object(meta, o), "task object")
    add("robot", dict(type="robot"), "robot")
    if not walk:
        for s in SIDES:
            add(HAND[s], dict(type="effector", bodies=[WRIST[s], *FINGER[s]], jaw_max_m=0.095), "hand")
    for o in task.get("objects") or ():
        if o not in objs:
            continue
        s, _ = support_of(frame, o, meta)
        s = s or objs[o].get("surface")
        if s and s in meta["surfaces"]:
            add(s, _entity_surface(meta, s), f"support of {o}")
            of = meta["surfaces"][s].get("of_object")
            if of and of in objs:
                add(of, _entity_furniture(meta, of), "furniture of support")
    g_surface, g_container = _goal_targets(task.get("goal") or [])
    if g_surface in meta["surfaces"]:
        add(g_surface, _entity_surface(meta, g_surface), "goal surface")
    if g_container in objs:
        add(g_container, _entity_object(meta, g_container), "goal container")
    tpos = frame.obj_pose(tgt)[0] if tgt in objs else None
    reach_goal, walk_goal = _reach_goal(calls), _walk_goal(calls)
    if not walk and tpos is not None:
        for o, od in objs.items():
            if o in entities or not od.get("free"):
                continue
            p = frame.obj_pose(o)[0]
            d = _xy_dist(p, tpos)
            half = max((od.get("box") or {}).get("half_extent", (0, 0))[:2])
            if d <= NEIGHBOUR_M:
                add(o, _entity_object(meta, o), f"neighbour {d:.2f} m")
            elif reach_goal is not None and _in_corridor(p, reach_goal, tpos, CORRIDOR_HALF_M + half):
                add(o, _entity_object(meta, o), "in corridor")
        tbox = objs[tgt].get("box") or {}
        top_t = float(tpos[2] + tbox.get("centre", (0, 0, 0))[2] + tbox.get("half_extent", (0, 0, 0))[2])
        for s, surf in meta["surfaces"].items():
            if s in entities:
                continue
            gap = float(surf["top"]) - top_t
            if 0.0 < gap <= OVERHANG_M and _footprint(tpos[:2], surf):
                add(s, _entity_surface(meta, s), f"above target {gap:.2f} m")
    if walk_goal is not None:
        add("region:stance", dict(type="region", shape="disc", centre=[float(v) for v in walk_goal], radius_m=STANCE_M),
            "walk_to goal")  # fmt: skip
        ppos = frame.pelvis()[0]
        for key, od in objs.items():
            if key in entities or od.get("free") or not od.get("box"):
                continue
            r = float(np.hypot(*od["box"]["half_extent"][:2]))
            if _in_corridor(_static_xy(meta, key), ppos, walk_goal, STANCE_M + r):
                add(key, _entity_furniture(meta, key), "on the walk path")
    if reach_goal is not None and not walk:
        add(
            "region:pregrasp_target", dict(type="region", shape="point", p=[float(v) for v in reach_goal]), "reach goal"
        )
        if tpos is not None:
            add("region:corridor", dict(type="region", shape="segment", a=[float(v) for v in reach_goal],
                                        b=[float(v) for v in tpos], half_width_m=CORRIDOR_HALF_M), "approach corridor")  # fmt: skip
    if g_surface in meta["surfaces"] and kind in ("release", "reach", "mixed"):
        surf = meta["surfaces"][g_surface]
        add("region:place", dict(type="region", shape="rect", centre=[*surf["xy"], surf["top"]],
                                 half_extent_m=surf["half_extent"], yaw=surf.get("yaw", 0.0)), "goal placement")  # fmt: skip
    selector = dict(NEIGHBOUR_M=NEIGHBOUR_M, OVERHANG_M=OVERHANG_M, STANCE_M=STANCE_M, CORRIDOR_HALF_M=CORRIDOR_HALF_M)
    return entities, dict(selector=selector, initial=list(entities), initial_why=why, added=[])


# -- the graph at one tick -------------------------------------------------------------------------------------------------
def _edge(s, r, o, v, src, **num) -> dict:
    return dict(s=s, r=r, o=o, v=v, src=src, **num)


def _blocking(frame: Frame, eid: str, ent: dict, region: dict, meta: dict) -> tuple[bool | None, float]:
    """Does entity `eid`'s box intersect the region (disc or segment corridor)? (value, depth_m)."""
    if ent["type"] == "object":
        c = frame.obj_pose(eid)[0][:2]
        r = float(np.hypot(*((meta["objects"][eid].get("box") or {}).get("half_extent", (0, 0))[:2])))
    elif ent["type"] == "furniture" and ent.get("box_m"):
        c = _static_xy(meta, eid)
        r = float(np.hypot(*ent["box_m"]["half_extent"][:2]))
    else:
        return None, 0.0
    if region["shape"] == "disc":
        depth = region["radius_m"] + r - _xy_dist(c, region["centre"])
    elif region["shape"] == "segment":
        a, b = np.asarray(region["a"], float)[:2], np.asarray(region["b"], float)[:2]
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        depth = region["half_width_m"] + r - float(np.linalg.norm(c - (a + t * ab)))
    else:
        return None, 0.0
    return depth > 0.0, round(max(depth, 0.0), 3)


def _side_of(target_xy, other_xy, yaw: float) -> str:
    """Which face of the target (in its own yaw frame) the other point lies off."""
    d = np.asarray(other_xy, float) - np.asarray(target_xy, float)
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y = c * d[0] - s * d[1], s * d[0] + c * d[1]
    return ("+x" if x > 0 else "-x") if abs(x) >= abs(y) else ("+y" if y > 0 else "-y")


def graph_of(frame: Frame, entities: dict, roles: dict, meta: dict) -> dict:
    """Spec section 4.4.1: states of every entity in scope and the edges around the roles (target, carried object,
    effector, robot, regions); never all pairs."""
    objs = meta["objects"]
    scene = PR._Scene(meta)
    states, edges = {}, []
    tgt, eff, carried = roles.get("target"), roles.get("effector"), roles.get("carried")
    ppos, pquat = frame.pelvis()
    yaw = _yaw(pquat)
    for eid, ent in entities.items():
        if ent["type"] == "object":
            p, q = frame.obj_pose(eid)
            v, w = frame.obj_vel(eid)
            half, centre, _ = scene.box(eid)
            states[eid] = dict(p=[round(float(x), 4) for x in p], q=[round(float(x), 5) for x in q], v_mps=v, w_radps=w,
                               bottom_z_m=round(float(p[2] + centre[2] - half[2]), 4), yaw=round(_yaw(q), 3))  # fmt: skip
        elif ent["type"] == "effector" and eid != "both_hands":
            side = eid.split("_")[0]
            hp = frame.hand_pos(side)
            st = dict(p=[round(float(x), 4) for x in hp], jaw_m=round(frame.jaw_m(side), 4),
                      ref_err_cm=round(float(np.linalg.norm(frame.ref_wrist(side) - hp)) * 100, 2), att="level")  # fmt: skip
            aq = frame.arm_q(side)
            if aq is not None:
                st["arm_q"] = aq
            states[eid] = st
        elif ent["type"] == "robot":
            prim = frame.a["primitives"][int(frame.a["primitive"][frame.k])] if "primitives" in frame.a else None
            states[eid] = dict(pelvis_p=[round(float(x), 4) for x in ppos], pelvis_z_m=round(float(ppos[2]), 3),
                               yaw=round(yaw, 3), fell=False, primitive=None if prim is None else str(prim),
                               mode=int(frame.a["mode"][frame.k]) if "mode" in frame.a else None)  # fmt: skip
    hands = [e for e, ent in entities.items() if ent["type"] == "effector" and e != "both_hands"]
    objects_in = [e for e, ent in entities.items() if ent["type"] == "object"]
    surfaces_in = [e for e, ent in entities.items() if ent["type"] == "surface"]
    regions = {e: ent for e, ent in entities.items() if ent["type"] == "region"}
    # contact, holding, support, containment and the goal predicates around the target and the carried object
    for o in objects_in:
        if o not in (tgt, carried):
            continue
        ob = frame.body_name(o)
        st = frame.state([o])
        for h in hands:
            side = h.split("_")[0]
            t = None if not frame.has_contacts else side in frame.touching(o)
            edges.append(_edge(o, "touching", h, t, "observed" if t is not None else "unknown"))
            # gripped: both fingers on it (what a grip step is for); held: the goal predicate, gripped and off every
            # surface (what a lift is for)
            gripped = None if not frame.has_contacts else side in st["objects"][o]["gripped"]
            edges.append(_edge(o, "gripped", h, gripped, "observed" if gripped is not None else "unknown"))
            held = None if not frame.has_contacts else bool(PR._held(scene, o, st)[0] and gripped)
            edges.append(_edge(o, "held", h, held, "predicate" if held is not None else "unknown"))
            rel = [round(float(x), 4) for x in frame.hand_pos(side) - frame.obj_pose(o)[0]] if gripped else None
            edges.append(_edge(o, "rel_pose", h, rel, "derived"))
        sup, src = support_of(frame, o, meta)
        for cand in surfaces_in:
            edges.append(_edge(o, "supported_by", cand, cand == sup, src))
        for c in objects_in:
            if c != o and objs[c].get("container"):
                edges.append(_edge(o, "in", c, bool(PR._in(scene, o, c, frame.state([o, c]))[0]), "predicate"))
        edges.append(_edge(o, "upright", None, bool(PR._upright(scene, o, st)[0]), "predicate"))
        for other in objects_in:
            if other != o:
                t = frame.touches(ob, frame.body_name(other))
                edges.append(_edge(o, "touching", other, t, "observed" if t is not None else "unknown"))
    # spatial edges around the target
    if tgt in objects_in:
        tp, tq = frame.obj_pose(tgt)
        tbox = objs[tgt].get("box") or {}
        thalf = np.asarray(tbox.get("half_extent", (0, 0, 0)), float)
        top_t = float(tp[2] + tbox.get("centre", (0, 0, 0))[2] + thalf[2])
        sup, _ = support_of(frame, tgt, meta)
        for s in surfaces_in:
            surf = meta["surfaces"][s]
            gap = float(surf["top"]) - top_t
            edges.append(_edge(s, "above", tgt, bool(0.0 < gap <= OVERHANG_M and _footprint(tp[:2], surf)), "derived",
                               gap_m=round(gap, 3)))  # fmt: skip
            if s == sup:
                local = PR.to_surface_frame(tp[:2], surf)
                half = np.asarray(surf["half_extent"], float)[:2]
                dists = {
                    "+x": half[0] - local[0],
                    "-x": half[0] + local[0],
                    "+y": half[1] - local[1],
                    "-y": half[1] + local[1],
                }
                e = min(dists, key=dists.get)
                edges.append(_edge(tgt, "near_edge", s, True, "derived", dist_m=round(float(dists[e]), 3), edge=e))
        for o in objects_in:
            if o == tgt:
                continue
            op = frame.obj_pose(o)[0]
            ohalf = np.asarray((objs[o].get("box") or {}).get("half_extent", (0, 0, 0)), float)
            d = _xy_dist(op, tp)
            gap = d - float(max(thalf[:2])) - float(max(ohalf[:2]))
            edges.append(_edge(o, "beside", tgt, bool(d <= NEIGHBOUR_M), "derived", gap_m=round(gap, 3),
                               side=_side_of(tp[:2], op[:2], _yaw(tq))))  # fmt: skip
        d = _xy_dist(ppos, tp)
        bearing = (math.degrees(math.atan2(tp[1] - ppos[1], tp[0] - ppos[0]) - yaw) + 180) % 360 - 180
        edges.append(_edge("robot", "dist", tgt, round(d, 3), "derived", bearing_deg=round(bearing, 1)))
        edges.append(_edge("robot", "facing", tgt, round(abs(bearing), 1), "derived"))
    # blocking: anything but the target and the robot against the stance and the corridor
    for rid, region in regions.items():
        if region["shape"] not in ("disc", "segment"):
            continue
        for eid, ent in entities.items():
            if eid in (tgt, carried, "robot") or ent["type"] not in ("object", "furniture"):
                continue
            v, depth = _blocking(frame, eid, ent, region, meta)
            if v is not None:
                edges.append(
                    _edge(eid, "blocking", rid, v, "derived" if frame.has_contacts else "approx", depth_m=depth)
                )
    if "region:stance" in regions:
        err = _xy_dist(ppos, regions["region:stance"]["centre"]) * 100
        edges.append(_edge("robot", "at", "region:stance", bool(err <= WALK_TOL_CM), "derived", err_cm=round(err, 1)))
    # relative pose: the hand against its target point, in the pelvis frame; the approach face
    if eff in hands and "region:pregrasp_target" in regions:
        side = eff.split("_")[0]
        hp = frame.hand_pos(side)
        goal = np.asarray(regions["region:pregrasp_target"]["p"], float)
        rel = goal - ppos
        c, s = math.cos(-yaw), math.sin(-yaw)
        fwd, left = c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]
        out = -left if side == "right" else left
        err = float(np.linalg.norm(hp - goal)) * 100
        edges.append(_edge(eff, "reach_vec", "region:pregrasp_target",
                           [round(float(fwd), 3), round(float(out), 3), round(float(goal[2] - ppos[2]), 3)], "derived"))  # fmt: skip
        edges.append(
            _edge(eff, "at", "region:pregrasp_target", bool(err <= REACH_TOL_CM), "derived", err_cm=round(err, 2))
        )
        if tgt in objects_in:
            tp, tq = frame.obj_pose(tgt)
            edges.append(_edge(eff, "approach", tgt, _side_of(tp[:2], hp[:2], _yaw(tq)), "derived",
                               wrist_att=states.get(eff, {}).get("att")))  # fmt: skip
    return dict(t=round(frame.t, 3), states=states, edges=edges)


# -- expected transitions, events, discrepancies -------------------------------------------------------------------------
def expected_of(kind: str, roles: dict, task: dict, g_before: dict, *, rising: bool = False) -> list[dict]:
    """Spec section 4.5: the transitions the primitive kind is for -- by primitive, never by object or step name. A
    fixed table, not a judgement (rules.md 38)."""
    tgt, eff, carried = roles.get("target"), roles.get("effector"), roles.get("carried")
    g_surface, g_container = _goal_targets(task.get("goal") or [])
    out: list[dict] = []
    support_now = [e["o"] for e in g_before.get("edges", []) if e["r"] == "supported_by" and e["s"] == tgt and e["v"]]
    if kind == "walk":
        out.append(dict(edge=["robot", "at", "region:stance"], to=True))
        out.append(dict(edge=["*", "blocking", "region:stance"], to=False))
    elif kind == "reach" and eff:
        if carried and tgt == carried:
            out.append(dict(edge=[tgt, "held" if rising else "gripped", eff], to=True))
            if rising:
                out += [dict(edge=[tgt, "supported_by", s], to=False) for s in support_now]
        else:
            out.append(dict(edge=[eff, "at", "region:pregrasp_target"], to=True, tol_cm=REACH_TOL_CM))
            if tgt:
                out.append(dict(edge=[tgt, "moved", None], to=False))
            out.append(dict(edge=["*", "blocking", "region:corridor"], to=False))
            out.append({"edge": ["*", "moved", None], "to": False, "scope": "neighbours", "except": tgt})
    elif kind in ("grip", "squeeze") and eff and tgt:
        hands = [HAND[s] for s in SIDES] if eff == "both_hands" else [eff]
        out += [dict(edge=[tgt, "gripped", h], to=True) for h in hands]
    elif kind == "release" and eff and tgt:
        hands = [HAND[s] for s in SIDES] if eff == "both_hands" else [eff]
        out += [dict(edge=[tgt, "gripped", h], to=False) for h in hands]
        if g_surface:
            out.append(dict(edge=[tgt, "supported_by", g_surface], to=True))
        if g_container:
            out.append(dict(edge=[tgt, "in", g_container], to=True))
    return out


def _edge_key(e: dict) -> tuple:
    return (e["s"], e["r"], e["o"])


def _bool_edges(g: dict) -> dict:
    return {_edge_key(e): e["v"] for e in g["edges"] if isinstance(e["v"], bool) or e["v"] is None}


def events_of(arrays: dict, meta: dict, k0: int, k1: int, entities: dict, roles: dict, task: dict) -> tuple[list, list]:
    """Spec section 4.6 step 5: boolean edge flips between ticks k0 and k1 (touching and gripped every tick, the rest
    every PREDICATE_DT_TICKS), moved / slipped, and the bodies that joined the scope by contact (added in place)."""
    events, added = [], []
    f0 = Frame(arrays, meta, k0)
    prev = _bool_edges(graph_of(f0, entities, roles, meta))
    p0 = {o: f0.obj_pose(o)[0] for o, e in entities.items() if e["type"] == "object"}
    moved = {o: False for o in p0}
    hands = [e for e, ent in entities.items() if ent["type"] == "effector" and e != "both_hands"]
    carried, eff = roles.get("carried"), roles.get("effector")
    d0 = None
    if carried in p0 and eff in hands:
        d0 = float(np.linalg.norm(f0.hand_pos(eff.split("_")[0]) - f0.obj_pose(carried)[0]))
    slipped = False
    body_to_key = {o.get("body"): k for k, o in meta["objects"].items() if o.get("body")}
    mine = {meta["objects"][o]["body"] for o in task.get("objects", ()) if meta["objects"][o].get("body")}
    mine |= {"pelvis", *(f for s in SIDES for f in FINGER[s]), *WRIST.values()}
    for k in range(k0 + 1, k1 + 1):
        f = Frame(arrays, meta, k)
        if f.has_contacts:
            for a, b in f.contacts():
                if (a in mine or a.startswith(SIDES)) and b not in mine:
                    key = body_to_key.get(b)
                    if key and key not in entities:
                        entities[key] = (_entity_object if meta["objects"][key].get("free") else _entity_furniture)(
                            meta, key
                        )
                        added.append(dict(id=key, t=round(f.t, 3), why=f"touched by {a}"))
        full = (k - k0) % PREDICATE_DT_TICKS == 0 or k == k1
        g = graph_of(f, entities, roles, meta)
        srcs = {_edge_key(e): e["src"] for e in g["edges"]}
        for key, v in _bool_edges(g).items():
            if key[1] not in ("touching", "gripped") and not full:
                continue
            was = prev.get(key)
            if was is not None and v is not None and was != v:
                events.append({"t": round(f.t, 3), "edge": list(key), "from": was, "to": v, "src": srcs[key]})
            prev[key] = v
        for o in p0:
            if not moved[o] and o != carried:
                cm = float(np.linalg.norm(f.obj_pose(o)[0] - p0[o])) * 100
                if cm > MOVED_CM:
                    moved[o] = True
                    events.append({"t": round(f.t, 3), "edge": [o, "moved", None], "from": False, "to": True,
                                   "cm": round(cm, 1), "src": "derived"})  # fmt: skip
        if d0 is not None and not slipped:
            d = float(np.linalg.norm(f.hand_pos(eff.split("_")[0]) - f.obj_pose(carried)[0]))
            if abs(d - d0) * 100 > SLIP_CM:
                slipped = True
                events.append({"t": round(f.t, 3), "edge": [carried, "slipped", eff], "from": False, "to": True,
                               "cm": round((d - d0) * 100, 1), "src": "derived"})  # fmt: skip
    return events, added


def discrepancies_of(expected: list[dict], g_after: dict, events: list[dict]) -> list[dict]:
    """The expected transitions that did not hold at the step's end (or, for moved / slipped, that happened)."""
    after = {_edge_key(e): e for e in g_after["edges"]}
    out = []
    for ex in expected:
        s, r, o = ex["edge"]
        if r in ("moved", "slipped"):
            if s == "*":
                hit = [e for e in events if e["edge"][1] == r and e["edge"][0] != ex.get("except")]
            else:
                hit = [e for e in events if e["edge"][1] == r and e["edge"][0] == s]
            if bool(hit) != ex["to"]:
                out.append(dict(ex, observed=bool(hit)))
            continue
        if s == "*":
            vals = [e["v"] for e in g_after["edges"] if e["r"] == r and e["o"] == o and e["v"] is not None]
            observed = any(vals) if vals else None
        else:
            e = after.get((s, r, o))
            observed = None if e is None else e["v"]
        if observed is not None and observed != ex["to"]:
            out.append(dict(ex, observed=observed))
    return out


# -- the round -> records ----------------------------------------------------------------------------------------------
def _tick_of(arrays: dict, t: float, last: bool = False) -> int:
    ts = np.asarray(arrays["t"], float)
    k = int(np.searchsorted(ts, t - 1e-6, side="left"))
    k = min(max(k, 0), len(ts) - 1)
    if last and k > 0 and ts[k] > t + 1e-6:
        k -= 1
    return k


def _task_dict(task) -> dict:
    if isinstance(task, dict):
        return dict(id=task.get("id"), objects=list(task.get("objects") or []), goal=list(task.get("goal") or []))
    return dict(id=getattr(task, "id", None), objects=list(getattr(task, "objects", ()) or ()),
                goal=list(getattr(task, "goal", ()) or ()))  # fmt: skip


def build(rec: dict, arrays: dict | None, meta: dict, task) -> list[dict]:
    """One record per step row of `rec` (spec section 4.6). No arrays (the plan was refused before the world moved):
    no records."""
    if arrays is None or "t" not in arrays or len(arrays["t"]) == 0:
        return []
    task = _task_dict(task)
    steps, calls = rec.get("steps") or [], rec.get("calls") or []
    label = str(rec.get("label", ""))
    g_surface, g_container = _goal_targets(task["goal"])
    held_state: dict = {o: None for o in task["objects"]}
    out = []
    for iv in intervals(steps):
        k0 = _tick_of(arrays, iv.t0)
        k1 = max(k0, _tick_of(arrays, iv.t1, last=True))
        cs = calls_in(calls, iv.t0, iv.t1)  # an empty interval holds no call
        kind = kind_of(cs)
        f0, f1 = Frame(arrays, meta, k0), Frame(arrays, meta, k1)
        positions = {o: f0.obj_pose(o)[0] for o in task["objects"] if o in meta["objects"]}
        roles = roles_of(kind, cs, task["objects"], held_before=held_state, positions=positions)
        entities, scope = scope_of(f0, roles, kind, task, meta, cs)
        g_before = graph_of(f0, entities, roles, meta)
        rising = False
        if kind == "reach" and roles.get("carried") and roles.get("effector") in HAND.values():
            goal = _reach_goal(cs)
            rising = goal is not None and float(goal[2]) > float(f0.hand_pos(roles["effector"].split("_")[0])[2]) + 0.02
        expected = expected_of(kind, roles, task, g_before, rising=rising)
        events, added = events_of(arrays, meta, k0, k1, entities, roles, task)
        scope["added"] = added
        g_after = graph_of(f1, entities, roles, meta)
        sup = support_of(f0, roles["target"], meta)[0] if roles.get("target") in meta["objects"] else None
        roles_out = dict(roles, support=sup, goal_surface=g_surface, container=g_container,
                         stance="region:stance" if "region:stance" in entities else None,
                         corridor="region:corridor" if "region:corridor" in entities else None,
                         pregrasp="region:pregrasp_target" if "region:pregrasp_target" in entities else None)  # fmt: skip
        row = iv.row
        skip = ("step", "ok", "why", "t0", "t1", "check", "attempt")
        action = dict(t0=iv.t0, t1=iv.t1, ok=bool(row.get("ok")), why=str(row.get("why", "")), calls=cs,
                      check=row.get("check") or {}, info={k: v for k, v in row.items() if k not in skip})  # fmt: skip
        out.append(dict(
            schema=SCHEMA, record_id=f"{label}/{iv.index}.{iv.attempt}", label=label, task=task["id"],
            step_index=iv.index, attempt=iv.attempt, step=iv.step, kind=kind,
            skills=rec.get("skills"), set=rec.get("set") or {}, predicate_version="predicates.py",
            action=action, roles=roles_out, scope=scope, entities=entities,
            G_before=g_before, events=events, G_after=g_after, expected=expected,
            discrepancies=discrepancies_of(expected, g_after, events),
            outcome=dict(step_ok=bool(row.get("ok")), round_ok=bool(rec.get("ok")), round_why=rec.get("why", ""),
                         failure_class=rec.get("failure_class")),
            evidence=dict(npz=f"{label}.npz", k0=k0, k1=k1, json_step=iv.index),
        ))  # fmt: skip
        for o in task["objects"]:  # who holds what at the end of this step feeds the next step's roles
            g = f1.gripped(o) if f1.has_contacts else set()
            held_state[o] = HAND[next(iter(g))] if len(g) == 1 else ("both_hands" if len(g) == 2 else None)
    return out


def error_line(label: str, err: str) -> dict:
    return dict(schema=SCHEMA, label=label, error=str(err)[:300])


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.round(5).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def write(out_dir: Path, label: str, records: list[dict]) -> Path:
    """`<label>.past.jsonl`, one JSON object per line, written whole (tmp + rename) like the round's .json."""
    p = Path(out_dir) / f"{label}.past.jsonl"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in records))
    tmp.rename(p)
    return p


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
