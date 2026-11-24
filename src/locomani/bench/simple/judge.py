"""The bed's judge: comply_simple.py (did the robot do what the instruction says) and loss_simple.py (why a record is not a
success), moved together (TODO 49.2). numpy only, no locomani import: the environment server, which runs in SIMPLE's own
venv, loads this file BY PATH (server.py `_judge()`), and the bed imports it as `locomani.bench.simple.judge`.

Did the robot do what the task's instruction says? A per-task check beside SIMPLE's own success check (2026-11-19).

The benchmark's checks are weaker than its instructions (HandoverTeleop passes when the object touches the container,
whatever the hands did). This module is simulator-side bookkeeping only, with no dependency on the bed, so the same
three functions can be called from the mirror and from the physics server of the official environment.

    snapshot(model, data, target_body, container_body=None) -> dict
        one instant's contacts of the target: left / right (the Dex3 hand: the wrist-yaw link, which carries the palm,
        and everything below it: thumb, index, middle links), robot (any other robot body: forearm, torso, legs),
        container, other (neither robot nor container: table, floor, other objects).
    Watcher(dt=0.02).feed(t, snap)   one snapshot per control step; keeps a few times and a run-length timeline, not every step
    verdict(task, success, state, marks) -> dict(ok, why)

Bodies (G1 + Dex3 in the mirror's scene): robot = every body below `pelvis`; left hand = `left_wrist_yaw_link` and its
children (left_hand_thumb_0/1/2_link, left_hand_middle_0/1_link, left_hand_index_0/1_link), right hand likewise.
(SIMPLE's own `"hand" in name` test leaves the palm out; the palm is part of a hand here.)

Rules (`why` is "" when ok):
    not_success    the environment's check did not hold (within the limit).
    XMovePickTeleop, BendPickMP, TabletopGraspMP (rule changed 2026-11-20, see below): a step is HELD when the check holds at
      it AND the target touches a hand AND touches nothing that is neither robot nor container; the episode is compliant when
      held steps run for at least HAND_S (0.2 s, ten steps in a row) and the run's first step is inside the limit; the beds
      end the episode when a held run reaches HAND_S (not at the first check-true step) and record t_check (the first step the
      check held), t_success (the run's first step, else t_check). Otherwise why is
      short        held runs stayed under 0.2 s (a contact that flickers, an object flicked past a fingertip);
      no_hand      the last check-true step that was not held had the target in no hand;
      on_support   ... touching something that is neither robot nor container (it tipped on the table, say).
      (A watcher fed without the check flag, as the older callers do, is judged by the single last step as before.)
    LocomotionPickBetweenTablesTeleop: success, and earlier the target was at some step in a hand and touching nothing
      but the robot: no_hand (never in a hand), not_lifted (in a hand, but always touching a support or the container too).
    HandoverTeleop: success, and before the first target-container contact: a moment with the target on the right hand and
      nothing but the robot, and after it an interval of at least 0.2 s with the target on the left hand and neither the
      right hand nor anything that is not the robot: else no_handover.
2026-11-20 (the one change of the pick rules, asked by the coordinator): the single instant of the check was brittle: a real
grasp whose first 1.8 cm came from tipping the object ended the episode before the lift (on_support), and a flickering contact
decided the other cases. The per-step held test and the run length replace it; HandoverTeleop and LocomotionPickBetweenTables
are unchanged.
`marks` (the bed's marks dict) is accepted for the caller's evidence and not used by the rules.

    pytest tests/test_simple_judge.py   (the old --selftest cases)
"""

import numpy as np

L, R, ROBOT, CONT, OTHER = 1, 2, 4, 8, 16
PICK = ("XMovePickTeleop", "BendPickMP", "TabletopGraspMP")
BETWEEN = "LocomotionPickBetweenTablesTeleop"
HANDOVER = "HandoverTeleop"
HAND_S = 0.2
_CACHE: dict = {}

# The four articulated teleop tasks (2026-11-23, simple_art): task -> (candidate checked joints as (joint, sign toward success), the
# official threshold in rad of the joint value, signed like the joint). The oven's check reads joint_1 or joint_2 (either opens it).
ART = dict(
    OpenOvenTeleop=(("articulate_joint_1", 1), ("articulate_joint_2", 1)),
    CloseDoorTeleop=(("articulate_joint_1", -1),),
    OpenFaucetTeleop=(("articulate_joint_0", 1), ("articulate_joint_0", -1)),
    OpenTrashCanTeleop=(("articulate_joint_0", 1),),
)
ART_NEED = dict(
    OpenOvenTeleop=1.4, CloseDoorTeleop=-0.16, OpenFaucetTeleop=0.7, OpenTrashCanTeleop=0.5
)  # |value| the check reads
ART_MOVE_DEG = 3.0  # t_move: the joint this far past its start toward success
ART_HAND_DEG = 5.0  # compliance: advance made with a hand on a moving body


def _classes(model) -> np.ndarray:
    """Per body: 0 world/scenery, L, R (hand links), ROBOT (other robot bodies). Cached per model."""
    key = (id(model), int(model.nbody), model.body_parentid.tobytes())
    if _CACHE.get("key") == key:
        return _CACHE["cls"]
    names = [model.body(i).name for i in range(model.nbody)]
    parent = model.body_parentid
    root = names.index("pelvis")
    wrist = {"left": names.index("left_wrist_yaw_link"), "right": names.index("right_wrist_yaw_link")}

    def below(i: int, a: int) -> bool:
        while i > 0:
            if i == a:
                return True
            i = int(parent[i])
        return False

    cls = np.zeros(model.nbody, dtype=int)
    for i in range(1, model.nbody):
        if below(i, wrist["left"]):
            cls[i] = L
        elif below(i, wrist["right"]):
            cls[i] = R
        elif below(i, root):
            cls[i] = ROBOT
    _CACHE.update(key=key, cls=cls)
    return cls


def snapshot(model, data, target_body, container_body=None) -> dict:
    """The target's contacts at this instant (a body id or name for the target and the container)."""
    tb = model.body(target_body).id if isinstance(target_body, str) else int(target_body)
    cb = None
    if container_body is not None:
        cb = model.body(container_body).id if isinstance(container_body, str) else int(container_body)
    cls = _classes(model)
    bits = 0
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if a != tb and b != tb:
            continue
        o = b if a == tb else a
        if o == tb:
            continue
        bits |= cls[o] if cls[o] else (CONT if o == cb else OTHER)
    return dict(left=bool(bits & L), right=bool(bits & R), robot=bool(bits & ROBOT), container=bool(bits & CONT),
                other=bool(bits & OTHER))  # fmt: skip


def _bits(s: dict) -> int:
    return L * s["left"] | R * s["right"] | ROBOT * s["robot"] | CONT * s["container"] | OTHER * s["other"]


class Watcher:
    """Fed one snapshot per step. state() is a small dict: last snapshot, first times, the handover and carry facts."""

    def __init__(self, dt: float = 0.02, timeline_cap: int = 400):
        self.dt, self.cap = dt, timeline_cap
        self.last, self.t_last = None, None
        self.t_right = None  # first step: on the right hand and nothing but the robot
        self.t_carry = None  # first step: in a hand, nothing but the robot touching
        self.t_container = None  # first step touching the container
        self.t_hand = None  # first step in a hand
        self.run_t0, self.run_n, self.best_run = None, 0, 0.0  # the left-only run after t_right
        self.t_left_ok = None  # the step the left-hand interval reached HAND_S (before any container contact)
        self.timeline: list = []  # (t, bits) at every change
        self.checked = False  # fed with the check flag (the pick rule's per-step test)
        self.t_check, self.t_held, self.bad = (
            None,
            None,
            None,
        )  # first check-true step; first step of the first held run >= HAND_S
        self.held_t0, self.held_n, self.best_held = None, 0, 0.0

    def feed(self, t: float, snap: dict, check: bool | None = None) -> None:
        """One snapshot per counted step; `check` (the environment's check at this step) switches the pick rule's tests on."""
        if check is not None:
            self.checked = True
            if check and self.t_check is None:
                self.t_check = t
            hand = snap["left"] or snap["right"]
            if check and hand and not snap["other"]:
                if self.held_t0 is None:
                    self.held_t0, self.held_n = t, 0
                self.held_n += 1
                self.best_held = max(self.best_held, self.held_n * self.dt)
                if self.t_held is None and self.held_n * self.dt >= HAND_S - 1e-9:
                    self.t_held = self.held_t0
            else:
                self.held_t0, self.held_n = None, 0
                if check:
                    self.bad = "on_support" if hand else "no_hand"
        b = _bits(snap)
        self.last, self.t_last = dict(snap), t
        if not self.timeline or self.timeline[-1][1] != b:
            if len(self.timeline) < self.cap:
                self.timeline.append((round(t, 2), b))
        hand = snap["left"] or snap["right"]
        nonrobot = snap["other"] or snap["container"]
        if hand and self.t_hand is None:
            self.t_hand = t
        if hand and not nonrobot and self.t_carry is None:
            self.t_carry = t
        if snap["container"] and self.t_container is None:
            self.t_container = t
        if self.t_container is not None:
            return  # the handover facts are about the time before the first container contact
        if snap["right"] and not nonrobot and self.t_right is None:
            self.t_right = t
            return
        left_only = snap["left"] and not snap["right"] and not nonrobot
        if self.t_right is not None and left_only:
            if self.run_t0 is None:
                self.run_t0, self.run_n = t, 0
            self.run_n += 1
            self.best_run = max(self.best_run, self.run_n * self.dt)
            if self.t_left_ok is None and self.best_run >= HAND_S - 1e-9:
                self.t_left_ok = t
        else:
            self.run_t0, self.run_n = None, 0

    def state(self) -> dict:
        return dict(last=self.last, t_last=self.t_last, t_hand=self.t_hand, t_carry=self.t_carry, t_right=self.t_right,
                    t_container=self.t_container, t_left_ok=self.t_left_ok, best_left_s=round(self.best_run, 2),
                    checked=self.checked, t_check=self.t_check, t_held=self.t_held, best_held_s=round(self.best_held, 2),
                    bad=self.bad, timeline=list(self.timeline))  # fmt: skip


def verdict(task: str, success: bool, state: dict, marks: dict | None = None, limit_s: float | None = None) -> dict:
    """ok / why for one finished episode. `success`: the environment's check held at some counted step inside the limit;
    `limit_s`: the held run's first step must also be inside it (None: not tested)."""
    if not success:
        return dict(ok=False, why="not_success")
    last = state.get("last")
    if task in ART:  # the articulated tasks: joint check held, no body touch, a hand pushed (hand_deg >= ART_HAND_DEG)
        if state.get("t_body") is not None:
            return dict(ok=False, why="body_touch")
        if state.get("t_hand") is None or state.get("hand_deg", 0.0) < ART_HAND_DEG - 1e-9:
            return dict(ok=False, why="no_hand")
        return dict(ok=True, why="")
    if task in PICK and state.get("checked"):
        th = state.get("t_held")
        if th is not None and (limit_s is None or th <= limit_s + 1e-9):
            return dict(ok=True, why="")
        if state.get("best_held_s", 0.0) > 0.0 and th is None:
            return dict(ok=False, why="short")
        return dict(ok=False, why=state.get("bad") or "no_hand")
    if task in PICK:
        if last is None or not (last["left"] or last["right"]):
            return dict(ok=False, why="no_hand")
        if last["other"]:
            return dict(ok=False, why="on_support")
        return dict(ok=True, why="")
    if task == BETWEEN:
        if state.get("t_carry") is not None:
            return dict(ok=True, why="")
        return dict(ok=False, why="not_lifted" if state.get("t_hand") is not None else "no_hand")
    if task == HANDOVER:
        if state.get("t_left_ok") is not None:
            return dict(ok=True, why="")
        return dict(ok=False, why="no_handover")
    raise ValueError(f"unknown task {task}")


class ArtWatcher:
    """The articulated tasks' watcher, fed one step at a time up to the step the check first holds (the server feeds it).
    feed(t, q, hand, body, check): q = {joint name: value (rad)} of the asset's joints at the end of the step; hand = a hand link
    (wrist_yaw_link and below, either side) touched a moving body of the asset in some physics step of this control step;
    body = the names of the robot links outside the hands that did. Per candidate (joint, sign) it keeps the start value, the
    furthest advance toward success, the first time the advance reached ART_MOVE_DEG and the advance made during hand steps."""

    def __init__(self, task: str, dt: float = 0.02):
        self.task, self.dt, self.cands = task, dt, ART[task]
        self.q0: dict = {}
        self.q_prev: dict = {}
        self.q_end: dict = {}
        self.best = {c: 0.0 for c in self.cands}  # rad, advance toward success
        self.t_move = {c: None for c in self.cands}
        self.hand_adv = {c: 0.0 for c in self.cands}
        self.t_hand, self.t_body, self.body_links = None, None, set()
        self.t_check, self.t_last = None, None
        self.t_held = None  # (the server's step reply reads it; only the pick rule sets it)

    def start(self, q: dict) -> None:
        """The joints' values when the counted time begins (the step before the first fed one)."""
        self.q0, self.q_prev = dict(q), dict(q)

    def feed(self, t: float, q: dict, hand: bool, body, check: bool = False) -> None:
        if not self.q0:
            self.start(q)
        for c in self.cands:
            j, sg = c
            adv = sg * (q[j] - self.q0[j])
            self.best[c] = max(self.best[c], adv)
            if self.t_move[c] is None and adv >= np.radians(ART_MOVE_DEG):
                self.t_move[c] = t
            if hand:
                self.hand_adv[c] += sg * (q[j] - self.q_prev[j])
        if hand and self.t_hand is None:
            self.t_hand = t
        if body:
            if self.t_body is None:
                self.t_body = t
            self.body_links |= set(body)
        if check and self.t_check is None:
            self.t_check = t
        self.q_prev, self.q_end, self.t_last = dict(q), dict(q), t

    def chosen(self):
        """The candidate the record reports: the one that went furthest toward success."""
        return max(self.cands, key=lambda c: self.best[c])

    def state(self) -> dict:
        c = self.chosen()
        j, sg = c
        d = np.degrees
        q0, qe = self.q0.get(j, 0.0), self.q_end.get(j, self.q0.get(j, 0.0))
        tm = self.t_move[c]
        return dict(
            art=dict(joint=j, sign=sg, start_deg=round(float(d(q0)), 3), best_deg=round(float(d(q0 + sg * self.best[c])), 3),
                     end_deg=round(float(d(qe)), 3), need_deg=round(float(d(sg * abs(ART_NEED[self.task]))), 3),
                     t_move=None if tm is None else round(tm, 2)),
            art_touch=dict(t_hand=None if self.t_hand is None else round(self.t_hand, 2),
                           hand_deg=round(float(max(0.0, d(self.hand_adv[c]))), 3),
                           t_body=None if self.t_body is None else round(self.t_body, 2),
                           body_links=sorted(self.body_links)[:5]),
            t_hand=self.t_hand, t_body=self.t_body, hand_deg=round(float(max(0.0, d(self.hand_adv[c]))), 3),
            body_links=sorted(self.body_links)[:5], t_check=self.t_check, t_last=self.t_last,
        )  # fmt: skip


def _tl(spec: str, dt=0.02) -> Watcher:
    """'L:5 R:3 -:2' -> a watcher fed one snapshot per step; letters: L R r(obot) C(ontainer) O(ther), '-' none. A part
    starting with '+' ('+R:10') has the environment's check true; if any part does, the others have it false."""
    w, t = Watcher(dt), 0.0
    flagged = "+" in spec
    keys = dict(L="left", R="right", r="robot", C="container", O="other")
    for part in spec.split():
        sym, n = part.split(":")
        s = dict(left=False, right=False, robot=False, container=False, other=False)
        chk = sym.startswith("+")
        for ch in sym:
            if ch in keys:
                s[keys[ch]] = True
        for _ in range(int(n)):
            t += dt
            w.feed(t, s, chk if flagged else None)
    return w


def art_selftest() -> None:
    """Synthetic articulated episodes: a door-like (sign -1) and an oven-like (two joints, sign +1) timeline."""

    def run(task, steps, success=True):
        """steps: (dq_deg per step of the first joint, hand, body names) repeated n times: [(n, dq, hand, body), ...]."""
        w = ArtWatcher(task)
        (j0, sg), *rest = w.cands
        q = {j: 0.0 for j, _ in w.cands}
        q[j0] = np.radians(40.0) if task == "CloseDoorTeleop" else 0.0  # the door starts open
        w.start(q)
        t = 0.0
        for n, dq, hand, body in steps:
            for _ in range(n):
                t += 0.02
                q[j0] += sg * np.radians(dq)
                w.feed(t, q, hand, body)
        return w, verdict(task, success, w.state())

    # ok: a hand pushes the oven door 90 deg
    w, v = run("OpenOvenTeleop", [(20, 0.0, False, ()), (30, 3.0, True, ())])
    assert v == dict(ok=True, why="") and w.state()["art"]["joint"] == "articulate_joint_1", (v, w.state())
    assert abs(w.state()["art"]["best_deg"] - 90.0) < 1e-6 and w.state()["art_touch"]["t_body"] is None
    assert (
        abs(w.state()["art"]["t_move"] - 0.42) < 1e-6
    )  # 3 deg at the second step of the push: t = 20 + 1 steps -> 0.42
    # body_touch: the torso pushes the door shut, a hand never touched it
    w, v = run("CloseDoorTeleop", [(10, 0.0, False, ()), (20, 2.0, False, ("torso_link", "left_elbow_link"))])
    assert v == dict(ok=False, why="body_touch"), v
    st = w.state()
    assert st["art"]["sign"] == -1 and st["art"]["start_deg"] == 40.0 and st["art"]["best_deg"] == 0.0, st["art"]
    assert st["art_touch"]["t_body"] == 0.22 and st["art_touch"]["body_links"] == ["left_elbow_link", "torso_link"]
    # body_touch wins over a hand push (a leg bumped the door once)
    w, v = run("CloseDoorTeleop", [(2, 0.0, False, ("left_knee_link",)), (20, 2.0, True, ())])
    assert v["why"] == "body_touch"
    # no_hand: the hand touched but advanced under 5 deg, the door swung on by itself afterwards (coasting is fine when >= 5)
    w, v = run("OpenTrashCanTeleop", [(5, 0.5, True, ()), (40, 2.0, False, ())])
    assert v == dict(ok=False, why="no_hand") and abs(w.state()["art_touch"]["hand_deg"] - 2.5) < 1e-6, (v, w.state())
    w, v = run("OpenTrashCanTeleop", [(5, 1.0, True, ()), (40, 2.0, False, ())])
    assert (
        v == dict(ok=True, why="") and abs(w.state()["art_touch"]["hand_deg"] - 5.0) < 1e-6
    )  # coasting after the push
    # never touched: no_hand, t_hand None
    w, v = run("OpenOvenTeleop", [(30, 3.0, False, ())])
    assert v == dict(ok=False, why="no_hand") and w.state()["art_touch"]["t_hand"] is None
    # the faucet turns either way: the direction that went further is reported
    w = ArtWatcher("OpenFaucetTeleop")
    q = {"articulate_joint_0": 0.0}
    w.start(q)
    for k in range(10):
        q["articulate_joint_0"] -= np.radians(5.0)
        w.feed(0.02 * (k + 1), q, True, ())
    st = w.state()
    assert st["art"]["sign"] == -1 and st["art"]["best_deg"] == -50.0 and abs(st["art"]["need_deg"] + 40.107) < 1e-3, (
        st["art"]
    )
    assert verdict("OpenFaucetTeleop", True, st)["ok"]
    assert verdict("OpenFaucetTeleop", False, st)["why"] == "not_success"


def selftest() -> None:
    def v(task, spec, success=True):
        return verdict(task, success, _tl(spec).state())["why"]

    for task in PICK + (BETWEEN, HANDOVER):
        assert v(task, "R:3", success=False) == "not_success", task
    for task in PICK:
        assert v(task, "O:3 RO:3 R:2") == "" and v(task, "R:5 RC:2") == ""
        assert v(task, "O:3 -:2") == "no_hand" and v(task, "r:3") == "no_hand"
        assert v(task, "R:5 RO:1") == "on_support"
    for task in PICK:  # the per-step rule (2026-11-20)
        assert v(task, "-:5 +RO:2 +R:10") == ""  # tipped on the table, then lifted: ok at the later time
        assert _tl("-:5 +RO:2 +R:10").state()["t_held"] > _tl("-:5 +RO:2 +R:10").state()["t_check"]
        assert v(task, "+-:3 +R:1 +-:5") == "short"  # flicked past a finger for one step
        assert v(task, "-:35 +-:5") == "no_hand"  # the check at 0.7 s with nobody near, never held
        assert v(task, "+R:10") == "" and v(task, "+R:9") == "short"
        assert v(task, "+R:6 R:1 +R:6") == "short"  # a step with the check false breaks the run
        assert (
            v(task, "+O:2 +RO:3") == "on_support" and v(task, "+RO:3 +O:2") == "no_hand"
        )  # the last such step decides
    assert v(BETWEEN, "O:2 RO:5 R:2 C:3 RC:3") == ""
    assert v(BETWEEN, "O:5 R:4 C:3") == "" and v(BETWEEN, "Rr:3 C:2") == ""
    assert v(BETWEEN, "O:5 RO:4 RC:3 C:3") == "not_lifted"
    assert v(BETWEEN, "O:5 C:3") == "no_hand"
    ok = "R:5 Rr:2 RL:3 L:10 L:1 LC:3 C:12"  # right carries, both hold, left alone 11 steps (0.22 s), places
    assert v(HANDOVER, ok) == ""
    assert v(HANDOVER, "R:5 RL:3 L:9 C:12") == "no_handover"  # left alone only 0.18 s
    assert v(HANDOVER, "R:5 L:6 -:1 L:6 C:12") == "no_handover"  # interval broken by a step in the air
    assert v(HANDOVER, "O:5 C:12") == "no_handover" and v(HANDOVER, "R:5 RC:2 C:12") == "no_handover"  # the shortcut
    assert v(HANDOVER, "L:20 R:3 C:12") == "no_handover"  # left first, right only after: wrong order
    assert v(HANDOVER, "R:3 LO:6 L:5 C:12") == "no_handover"  # a table touch breaks it
    assert v(HANDOVER, "R:3 L:11 C:12") == ""
    assert v(HANDOVER, "R:3 L:6 C:2 L:10 C:5") == "no_handover"  # the long left run came after the container contact
    assert v(HANDOVER, "R:3 Lr:5 L:6 C:3") == ""  # the forearm (robot) may touch
    art_selftest()
    print("selftest ok")


# ---- loss_simple.py: the failure class of a record

_LOSS_DOC = """Articulated tasks (2026-11-23): collide = a robot link outside the hands touched a moving body, deliver = no hand touched it,
grasp = touched, advanced under 3 deg toward success, short = advanced 3 deg or more and still short; perceive: "no part seen".

Why a bed_simple record is not a success within SIMPLE's limit (escalation plan, section 7.0), from the record alone.

timeout   SIMPLE's check passed, but after the task's step limit
fall      the robot fell
perceive  the target or the basket was not found (camera track)
collide   the target was pushed 3 cm or tipped 20 deg before the fingers closed, or the basket was pushed 3 cm
          before the release
deliver   the palm was more than 2 cm from the grasp (or from the release point), or no grasp the hand could take
grasp     the fingers closed and the object did not come up, or was lost on the way
place     carried to the basket and let go, and it did not stay
"""

PLACE = ("LocomotionPickBetweenTablesTeleop", "HandoverTeleop")


def loss_of(r: dict) -> str | None:
    if "error" in r:
        return "error"
    t = r.get("t_success")
    if t is not None:
        return None if t <= r["limit_s"] else "timeout"
    if r.get("fell") is not None:
        return "fall"
    if r.get("why") in ("no target seen", "no basket seen", "no part seen"):
        return "perceive"
    if (
        "art" in r
    ):  # the four articulated tasks (the record carries rec["art"] / rec["art_touch"], bed_simple_official.run)
        a, at = r["art"], r.get("art_touch") or {}
        if at.get("t_body") is not None:
            return "collide"  # a link outside the hands touched a moving body
        if at.get("t_hand") is None:
            return "deliver"  # the hand never touched a moving body
        if a["sign"] * (a["best_deg"] - a["start_deg"]) < 3.0:
            return "grasp"  # touched, the part did not move toward success
        return "short"  # moved, but not far enough (or met and not compliant otherwise)
    m = r.get("marks", {})
    bc, al, bo = m.get("before_close"), m.get("after_lift"), m.get("before_open")
    if r.get("why") == "no grasp" or bc is None:
        return "deliver"
    if not (al is not None and al["held"] and al["rise_cm"] >= 5.0):  # the pick itself failed
        if bc["moved_cm"] > 3.0 or bc["tilt_deg"] > 20.0:
            return "collide"
        if r.get("palm_err_cm", 0.0) > 2.0 or not all(r.get("reach_ok", [True])):
            return "deliver"
        return "grasp"
    if r["episode_id"].split("/")[0] not in PLACE or (bo is not None and not bo["held"]):
        return "grasp"
    if bo is None:
        return "deliver"
    if (bo.get("basket_cm") or 0.0) > 3.0:
        return "collide"
    if r.get("release_cm", 0.0) > 2.0:
        return "deliver"
    return "place"
