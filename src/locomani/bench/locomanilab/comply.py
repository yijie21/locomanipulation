"""Did the robot do what a LocoManiLab task's instruction says? Beside the benchmark's own success check (2026-11-20).

The benchmark's checks are weaker than its instructions (a knocked cup passes "move the cup", an apple passes "put it
into the box" with the hand still on it, a lift needs no hand at all on one task). Simulator-side bookkeeping only, no
dependency on the bed: the bed calls it once per control step from the local model's contacts (as comply_simple does).

    snapshot(model, data, target_body, receptacle_body=None, robot_root="pelvis") -> dict
        one instant's contacts of the target (its body and every body below it): left / right (a hand), robot (any other
        body below `robot_root`: forearm, torso, legs), receptacle (the receptacle body and everything below it), other.
    Watcher(dt=0.02).feed(t, snap, check=False, fallen=False)   one call per counted control step
    verdict(task_family, success, state) -> dict(ok, why)       `why` is "" when ok

Bodies. The LocoManiLab robot is a Unitree G1 with a three-finger Dex3 hand per arm (the bed's model, lml_scene.py's
mirror of the benchmark's, with Unitree's names; the benchmark has no parallel gripper on this robot: the `_hand_`
test of bed_lml.py:150 and the DEX3 gains of lml_sim.py:63 say so). A hand is `left_wrist_yaw_link` or
`right_wrist_yaw_link` and every body below it (`<side>_hand_thumb_0/1/2_link`, `<side>_hand_middle_0/1_link`,
`<side>_hand_index_0/1_link`: the wrist-yaw link carries the palm). The robot is every other body below `pelvis`.

Families (the bed's `family`): navigation; lift, pick -> the lift rule; pick_place, receptacle, composite -> the place rule.
Rules (an episode whose check never held is `not_success`):
    navigation  fell     the robot had fallen at a step where the check held.
    lift        a step is HELD when the target touches a hand and nothing that is not the robot (not the support,
                not a receptacle); the episode is compliant when the run of HELD steps ending at a check-true step
                is at least HAND_S (0.2 s, ten steps) long at some check-true step: the last 0.2 s before the check
                holds (or inside it, when the caller keeps feeding after the check). Else, from the last check-true step:
                  short       a run, but under 0.2 s;
                  on_support  a hand, but also something that is not the robot;
                  no_hand     no hand on it.
    place       before the first check-true step a CARRY_S (0.1 s, five steps in a row) with the target on a hand
                and touching nothing that is not the robot, else `not_carried`; and at the first check-true step no
                hand on the target, else `hand_on`. (`not_carried` is tested first.)

"""

import numpy as np

L, R, ROBOT, RECEPT, OTHER = 1, 2, 4, 8, 16
NAV = ("navigation",)
LIFT = ("lift", "pick")
PLACE = ("pick_place", "receptacle", "composite")
ART = (
    "articulation",
    "navigation_articulation",
    "articulation_pnp",
)  # fork: the fridge tasks (bed2), the low drawers (bed3)
ART_WHY = ("fell before the door check held", "not genuine (too few steps)", "drawer not moved by a hand",
           "drawer moved by a leg")  # fmt: skip
LEG_PARTS = (
    "thigh",
    "knee",
    "shin",
    "foot",
)  # record.LEG_PARTS (copied: this file has no dependency on the bed; numpy only)
HAND_S = 0.2
CARRY_S = 0.1
_CACHE: dict = {}


def _id(model, b) -> int:
    return model.body(b).id if isinstance(b, str) else int(b)


def _below(model, i: int, a: int) -> bool:
    parent = model.body_parentid
    while i > 0:
        if i == a:
            return True
        i = int(parent[i])
    return False


def _classes(model, target: int, receptacle, robot_root) -> np.ndarray:
    """Per body: 0 scenery / other, L, R (hand links), ROBOT, RECEPT, -1 the target itself. Cached per model and ids."""
    key = (id(model), int(model.nbody), model.body_parentid.tobytes(), target, receptacle, robot_root)
    if _CACHE.get("key") == key:
        return _CACHE["cls"]
    wl, wr = _id(model, "left_wrist_yaw_link"), _id(model, "right_wrist_yaw_link")
    root = _id(model, robot_root)
    cls = np.zeros(model.nbody, dtype=int)
    for i in range(1, model.nbody):
        if _below(model, i, target):
            cls[i] = -1
        elif _below(model, i, wl):
            cls[i] = L
        elif _below(model, i, wr):
            cls[i] = R
        elif _below(model, i, root):
            cls[i] = ROBOT
        elif receptacle is not None and _below(model, i, receptacle):
            cls[i] = RECEPT
        else:
            cls[i] = OTHER
    _CACHE.update(key=key, cls=cls)
    return cls


def snapshot(model, data, target_body, receptacle_body=None, robot_root="pelvis") -> dict:
    """The target's contacts at this instant (a body id or name for the target, the receptacle and the robot's root)."""
    tb = _id(model, target_body)
    rb = None if receptacle_body is None else _id(model, receptacle_body)
    cls = _classes(model, tb, rb, _id(model, robot_root))
    bits = 0
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        ca, cb = cls[a], cls[b]
        if (ca == -1) == (cb == -1):  # neither touches the target, or the target touches itself
            continue
        bits |= cb if ca == -1 else ca
    return dict(left=bool(bits & L), right=bool(bits & R), robot=bool(bits & ROBOT), receptacle=bool(bits & RECEPT),
                other=bool(bits & OTHER))  # fmt: skip


_NONE = dict(left=False, right=False, robot=False, receptacle=False, other=False)


class Watcher:
    """Fed one snapshot per step (`snap` may be None for a navigation task). state() is a small dict of the facts."""

    def __init__(self, dt: float = 0.02, timeline_cap: int = 400, strict: bool = False):
        self.dt, self.cap = dt, timeline_cap
        self.strict = strict  # articulation: a fall at any step up to the first check-true step counts
        self.last, self.t_last = None, None
        self.t_check = None  # first check-true step
        self.hand_at_check = None  # a hand on the target at the first check-true step
        self.fell_at_check = False  # fallen at some check-true step
        self.t_carry = None  # the step a CARRY_S run (hand, nothing but the robot) was completed, before the check
        self.carry_n = 0
        self.run_n = 0  # the current run of HELD steps
        self.best_run = 0.0  # the longest run ending at a check-true step
        self.at_check = None  # (hand, nonrobot, run seconds) at the last check-true step
        self.timeline: list = []  # (t, bits) at every change

    def feed(self, t: float, snap: dict | None, check: bool = False, fallen: bool = False) -> None:
        s = snap or _NONE
        hand = s["left"] or s["right"]
        nonrobot = s["receptacle"] or s["other"]
        held = hand and not nonrobot
        self.last, self.t_last = dict(s), t
        b = L * s["left"] | R * s["right"] | ROBOT * s["robot"] | RECEPT * s["receptacle"] | OTHER * s["other"]
        if (not self.timeline or self.timeline[-1][1] != b) and len(self.timeline) < self.cap:
            self.timeline.append((round(t, 2), b))
        self.run_n = self.run_n + 1 if held else 0
        if self.strict and fallen and self.t_check is None:
            self.fell_at_check = True
        if check:
            if self.t_check is None:
                self.t_check, self.hand_at_check = t, hand
            self.fell_at_check = self.fell_at_check or bool(fallen)
            self.best_run = max(self.best_run, self.run_n * self.dt)
            self.at_check = (hand, nonrobot, self.run_n * self.dt)
        elif self.t_check is None:  # the carry facts are about the time before the check
            self.carry_n = self.carry_n + 1 if held else 0
            if self.t_carry is None and self.carry_n * self.dt >= CARRY_S - 1e-9:
                self.t_carry = t

    def state(self) -> dict:
        return dict(last=self.last, t_last=self.t_last, t_check=self.t_check, hand_at_check=self.hand_at_check,
                    fell_at_check=self.fell_at_check, t_carry=self.t_carry, best_run_s=round(self.best_run, 2),
                    at_check=self.at_check, timeline=list(self.timeline))  # fmt: skip


def verdict(task_family: str, success: bool, state: dict, genuine: bool = True, moved_by=None, hand_drives: bool = False) -> dict:  # fmt: skip
    """ok / why for one finished episode. `success`: the environment's check held at some counted step. `genuine`: the
    step count reached the task's genuine_min_steps (only articulation reads it). `moved_by`, `hand_drives`: (drawer
    tasks) the body parts in contact with the drawer while it moved (rec["art"]["moved_by"]) and whether the instruction
    wants a hand to work it; then a drawer moved by a leg, or by nothing that is a hand, is not what was asked."""
    if task_family not in NAV + LIFT + PLACE + ART:
        raise ValueError(f"unknown task family {task_family}")
    if not success:
        return dict(ok=False, why="not_success")
    if task_family in ART:  # the door check held, the robot had not fallen by then, and the episode was an attempt
        if state["fell_at_check"]:
            return dict(ok=False, why=ART_WHY[0])
        if not genuine:
            return dict(ok=False, why=ART_WHY[1])
        if hand_drives:
            by = set(moved_by or ())
            if by & set(LEG_PARTS):
                return dict(ok=False, why=ART_WHY[3])
            if "hand" not in by:
                return dict(ok=False, why=ART_WHY[2])
        if task_family == "articulation_pnp":  # and the apple: carried in a hand, no hand on it when the check holds
            if state["t_carry"] is None:
                return dict(ok=False, why="not_carried")
            if state["hand_at_check"]:
                return dict(ok=False, why="hand_on")
        return dict(ok=True, why="")
    if task_family in NAV:
        return dict(ok=not state["fell_at_check"], why="fell" if state["fell_at_check"] else "")
    if task_family in LIFT:
        if state["best_run_s"] >= HAND_S - 1e-9:
            return dict(ok=True, why="")
        hand, nonrobot, run = state["at_check"] or (False, False, 0.0)
        return dict(ok=False, why="short" if run > 0 else "on_support" if hand and nonrobot else "no_hand")
    if state["t_carry"] is None:
        return dict(ok=False, why="not_carried")
    if state["hand_at_check"]:
        return dict(ok=False, why="hand_on")
    return dict(ok=True, why="")


def _tl(spec: str, dt=0.02) -> Watcher:
    """'L:5 R:3 -:2' -> a watcher fed one snapshot per step; letters: L R r(obot) C(receptacle) O(ther), '-' none.
    A part starting with '+' has the check true, one starting with '!' has the robot fallen too."""
    w, t = Watcher(dt), 0.0
    keys = dict(L="left", R="right", r="robot", C="receptacle", O="other")
    for part in spec.split():
        sym, n = part.split(":")
        s = dict(_NONE)
        for ch in sym:
            if ch in keys:
                s[keys[ch]] = True
        for _ in range(int(n)):
            t += dt
            w.feed(t, s, "+" in sym, "!" in sym)
    return w
