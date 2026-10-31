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
            held = (
                None
                if not frame.has_contacts
                else bool(PR._held(scene, o, st)[0] and side in st["objects"][o]["gripped"])
            )
            edges.append(_edge(o, "held", h, held, "predicate" if held is not None else "unknown"))
            rel = [round(float(x), 4) for x in frame.hand_pos(side) - frame.obj_pose(o)[0]] if held else None
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
