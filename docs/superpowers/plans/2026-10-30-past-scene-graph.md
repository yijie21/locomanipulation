# PAST Scene-Graph Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, for every skill step and retry of a round, a local action-conditioned scene graph (`<label>.past.jsonl`) built from privileged simulation state, and give the Reviewer a digest section and a CLI to read it.

**Architecture:** The motor layer gains one trace field (per-tick contact pairs) exported in the NPZ; a pure function `locomani.tasks.scene.past.build(rec, arrays, meta, task)` turns a round's step log, call log, trace and scene file into records; `scene_round.py` writes them beside the round's `.json`; `digest.py` and `scripts/rsi/past.py` read them. Nothing reads the records at run time; no round outcome changes.

**Tech Stack:** Python 3.10, numpy, MuJoCo (contacts only in the motor layer), pytest via `make test TESTS=...` (`third_party/GR00T-WholeBodyControl/.venv_sim/bin/python`), ruff via `make check`.

**Spec:** `docs/handoff/past-scene-graph-plan-20261030.md` (sections 4.1–4.10 are normative; TODO 39.1–39.4).

## Global Constraints

- Branch `henry-dev`; commit after every task with explicit paths (`git add <paths>`), never `git add -A`; commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Code and comments in English; the docs this plan edits are Chinese (match the surrounding text).
- The recorder only writes: no module under `src/locomani/tasks/scene/skills/` imports `past`; `past.py` imports neither `mujoco` nor the motor layer; the motor change only appends to `self.trace` rows and `arrays()`.
- A failing builder never costs a round: `scene_round.py` catches every exception from `past.build` and writes one error line.
- Schema string `"PAST-locomani-0.1"`; quaternions `wxyz` (MuJoCo order, as the NPZ); units m, s, rad unless a key says `_cm` / `_deg`.
- Constants (spec §4.4): `NEIGHBOUR_M = 0.30`, `OVERHANG_M = 0.50`, `STANCE_M = 0.35`, `CORRIDOR_HALF_M = 0.08`; thresholds `MOVED_CM = 2.0`, `SLIP_CM = 1.0`, `SUPPORT_BAND_M = 0.02`, `REACH_TOL_CM = 3.0`, `PREDICATE_DT_TICKS = 5`.
- Simulator runs (the 50-round invariance rerun, any RSI run) need the user's explicit go-ahead first (user 2026-10-30). Unit tests and `make check` do not.
- `make check` must pass before each commit (ruff format/check, size lint, doc lint).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/locomani/backends/scalebfm/motor.py` (modify) | `ARM_LINKS`, `self.arm_bodies`, `contacts()` per tick into the trace row, three int arrays + `bodies` table in `arrays()` |
| `src/locomani/tasks/scene/past.py` (create) | Pure builder: step intervals, kinds and roles from calls, scope, entities, states, edges, events, expected, discrepancies, `build()`, `write()` |
| `scripts/rsi/scene_round.py` (modify) | Call `past.build` + `past.write` after the round record |
| `scripts/rsi/batch.py` (modify) | `retire()` also renames `.past.jsonl` |
| `scripts/rsi/digest.py` (modify) | New section from the records (per-task first unexpected transition; per failed round flip timeline) |
| `scripts/rsi/past.py` (create) | CLI `show`, `compare` |
| `scripts/rsi/prompts/reviewer.md`, `scripts/rsi/rules.md` (modify) | Reading order, rule 38 (version 23) |
| `docs/rsi_design.md`, `docs/rsi.md`, `TODO.md`, spec §7 (modify) | Pointers, command, ticks, log |
| `tests/test_scalebfm_motor.py` (modify), `tests/test_scene_past.py`, `tests/test_rsi_past_tools.py` (create) | Tests |

Facts the implementer needs (verified 2026-10-30):
- `Motor.__init__` has `self.track: dict[name, body id]`, `self.fingers[side] = (id1, id2)`, `self.foot_bodies`; `tick()` appends a dict row to `self.trace`; `arrays()` builds the NPZ dict (`t, mode, robot[7] (pelvis pos + quat wxyz), joints, ref_pelvis, ref_wrists[2,3], wrists[2,3], jaw[2], fingers[4], primitive (index), primitives (names), obj_<body>[7]`). Body 0 is `world`; the floor is a geom of body 0.
- Contact loop pattern (motor.py `between_fingers`): `for i in range(d.ncon): b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])`.
- `_logged` call rows: `dict(name, t0, args (list via _plain), kwargs, t1, ok, why)`; `reach(side, pos, ...)` → `args[0]` side, `args[1]` pos; `grip(side)`, `release(side)` → `args[0]`; `walk_to(xy, yaw)` → `args[0]` xy; `turn(yaw)`; `sidestep(left, forward)`; `posture(...)`; `hold(seconds)`; `squeeze(inward_m, seconds)`. Nested calls are not logged.
- Step rows (`SkillContext.note`): `dict(step, ok, why, t0, t1, **info, check?, attempt?)`.
- Round record `rec` (scene_round.py) has `label, task, ok, why, fell, dropped, plan_ok, plan_why, error, steps, calls, t_end, seated, failure_class, events, skills (hash), set (overrides), rule, controller, motor`.
- `predicates._Scene(meta)`, `_on`, `_in`, `_held`, `_upright`, `_near`, `_lifted`, `to_surface_frame(xy, surface)` and the state dict `{"objects": {k: {pos, quat, touching: [sides], gripped: [sides]}}, "pelvis": {pos, quat}, "fallen": bool}`.
- `scene_meta.json`: `objects[k]` has `surface, init_pos, init_quat, box{half_extent, centre, wall}, container, free, body, semantic{mass_kg, movable, upright_axis, ...}`; `surfaces[s]` has `xy, top, half_extent, flat_half_extent?, yaw?, of_object?, body?, approach_edges?`.
- Test fixture `tests/fixtures/geometry/`: scene `scene_locomani.xml` with body `water_bottle_object` on surface `A` (`dining_table_dining_table`, top 0.62) and `B`; `tests/test_scalebfm_motor.py::motor()` builds a CPU `Motor` with a stand-in policy; a tick steps the sim.
- Arm link names per side: `{side}_shoulder_pitch_link, {side}_shoulder_roll_link, {side}_shoulder_yaw_link, {side}_elbow_link, {side}_wrist_roll_link, {side}_wrist_pitch_link, {side}_wrist_yaw_link, {side}_dex1_finger_1_link, {side}_dex1_finger_2_link`.
- `diag_pool.apply` renames retired rounds with `d.glob(f"{x}.*")`, which already matches `<x>.past.jsonl`; `batch.retire` renames only `.json` and `.npz` and needs the third suffix. `batch.outcomes` reads `*.json` only, so `.past.jsonl` never shows up as a round.

---

### Task 1: Contact channel in the motor trace

**Files:**
- Modify: `src/locomani/backends/scalebfm/motor.py` (constants near line 48, `__init__` near line 254, `tick()` near line 448, `arrays()` near line 950)
- Test: `tests/test_scalebfm_motor.py`

**Interfaces:**
- Produces: trace rows gain `contacts: list[tuple[int, int]]`; `arrays()` gains `contact_k`, `contact_a`, `contact_b` (`int32`, same length) and `bodies` (`np.array` of str, index = body id). Later tasks read exactly these keys.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scalebfm_motor.py`:

```python
class ContactChannel(unittest.TestCase):
    """TODO 39.1: the per-tick contact pairs for the PAST recorder (recording only)."""

    def test_rows_carry_contacts_and_arrays_export_them(self):
        meta = json.loads(Path(R.META).read_text())
        sim = R.Sim(meta, backend=ScaleBfmBackendConfig(scene_dir=str(DIRECTORY)))
        m = Motor(sim, StandInPolicy(meta), MotorConfig(), model_xml=str(DIRECTORY / "scene_locomani.xml"),
                  track=("water_bottle_object",))
        m.reset((-1.5, 0.0), 0.0, warm_up=0)
        for _ in range(3):
            self.assertTrue(m.tick())
        self.assertTrue(all("contacts" in r for r in m.trace))
        out = m.arrays()
        for k in ("contact_k", "contact_a", "contact_b"):
            self.assertEqual(out[k].dtype, np.int32)
        self.assertEqual(len(out["contact_k"]), len(out["contact_a"]))
        self.assertEqual(len(out["bodies"]), m.sim.m.nbody)
        self.assertEqual(str(out["bodies"][0]), "world")
        watched = set(m.track.values()) | m.arm_bodies
        for a, b in zip(out["contact_a"], out["contact_b"]):
            self.assertLess(a, b)
            self.assertNotEqual(a, 0)  # the floor (body 0) is never recorded
            self.assertTrue(a in watched or b in watched)

    def test_arm_bodies_are_the_eighteen_arm_and_finger_links(self):
        m = motor()
        names = {m.sim.m.body(i).name for i in m.arm_bodies}
        self.assertEqual(len(names), 18)
        self.assertIn("left_elbow_link", names)
        self.assertIn("right_dex1_finger_2_link", names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test TESTS="tests/test_scalebfm_motor.py -k ContactChannel"`
Expected: FAIL — `AttributeError: 'Motor' object has no attribute 'arm_bodies'`.

- [ ] **Step 3: Implement**

In `motor.py`, after `FINGERS = ...` (line 48):

```python
# TODO 39.1: the arm and hand links whose contacts the trace records for the PAST recorder (tasks/scene/past.py)
ARM_LINKS = {
    s: tuple(
        f"{s}_{n}_link"
        for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
                  "dex1_finger_1", "dex1_finger_2")
    )
    for s in SIDES
}  # fmt: skip
```

In `__init__`, after `self.fingers = ...`:

```python
        self.arm_bodies = {sim.m.body(n).id for s in SIDES for n in ARM_LINKS[s]}
```

Add a method next to `touching()`:

```python
    def contacts(self) -> list[tuple[int, int]]:
        """TODO 39.1: body pairs in contact this tick with one side a tracked object or an arm / hand link and the other
        not the world body (the floor). Recording only; sorted, no duplicates."""
        m, d = self.sim.m, self.sim.d
        watched = set(self.track.values()) | self.arm_bodies
        out = set()
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            if b1 == b2 or 0 in (b1, b2) or not (b1 in watched or b2 in watched):
                continue
            out.add((min(b1, b2), max(b1, b2)))
        return sorted(out)
```

In `tick()`, add to the `self.trace.append(dict(...))` row: `contacts=self.contacts(),`.

In `arrays()`, before `return out`:

```python
        # TODO 39.1: the contact channel as three flat int columns (tick index, body a, body b) and the body-name table
        k = [i for i, r in enumerate(rows) for _ in r.get("contacts", ())]
        pairs = [p for r in rows for p in r.get("contacts", ())]
        out["contact_k"] = np.array(k, dtype=np.int32)
        out["contact_a"] = np.array([p[0] for p in pairs], dtype=np.int32)
        out["contact_b"] = np.array([p[1] for p in pairs], dtype=np.int32)
        out["bodies"] = np.array([self.sim.m.body(i).name for i in range(self.sim.m.nbody)])
```

Update the `arrays()` docstring to list the four keys.

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_scalebfm_motor.py"`
Expected: PASS (all classes, including the pre-existing ones).

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add src/locomani/backends/scalebfm/motor.py tests/test_scalebfm_motor.py
git commit -m "motor: per-tick contact pairs in the trace and NPZ for the PAST recorder (TODO 39.1)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `past.py` — step intervals, kinds and roles from the call log

**Files:**
- Create: `src/locomani/tasks/scene/past.py`
- Test: `tests/test_scene_past.py`

**Interfaces:**
- Produces:
  - `SCHEMA = "PAST-locomani-0.1"`, the constants of Global Constraints.
  - `intervals(steps: list[dict]) -> list[Interval]` with `Interval(index: int, attempt: int, step: str, t0: float, t1: float, row: dict)`; `settle` and `hold` runner steps are kept (they are steps) but get `kind "none"` / `"hold"`.
  - `calls_in(calls: list[dict], t0: float, t1: float) -> list[dict]` (a call belongs to the interval that contains its `t0`).
  - `kind_of(calls: list[dict]) -> str` ∈ `{"walk","turn","sidestep","posture","hold","reach","grip","release","squeeze","mixed","none"}`.
  - `side_of(call: dict) -> str | None` (`"left" | "right" | "both"`).
  - `roles_of(kind, calls, task_objects: list[str], held_before: dict[str, str | None]) -> dict` with keys `target, effector, carried` (others are filled in Task 3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_past.py`:

```python
"""TODO 39: the PAST recorder's builder (src/locomani/tasks/scene/past.py) on hand-made rounds, no simulation."""

from __future__ import annotations

import unittest

from locomani.tasks.scene import past as P


def step(name, t0, t1, ok=True, **extra):
    return dict(step=name, ok=ok, why="" if ok else "x", t0=t0, t1=t1, **extra)


def call(name, t0, t1, *args, ok=True, **kwargs):
    return dict(name=name, t0=t0, t1=t1, args=list(args), kwargs=kwargs, ok=ok, why="")


class Intervals(unittest.TestCase):
    def test_steps_become_intervals_and_retries_keep_their_attempt(self):
        steps = [step("settle", 0.0, 1.0), step("walk", 1.0, 5.0), step("pregrasp", 5.0, 7.0, ok=False, attempt=1),
                 step("pregrasp", 7.0, 9.0, attempt=2), step("grip", 9.0, 10.0)]
        iv = P.intervals(steps)
        self.assertEqual([(i.step, i.attempt) for i in iv],
                         [("settle", 1), ("walk", 1), ("pregrasp", 1), ("pregrasp", 2), ("grip", 1)])
        self.assertEqual([i.index for i in iv], [0, 1, 2, 3, 4])

    def test_calls_belong_to_the_interval_holding_their_start(self):
        calls = [call("walk_to", 1.0, 4.8, [1.0, 0.0]), call("reach", 5.1, 6.9, "right", [1.0, 0.1, 0.9]),
                 call("grip", 9.0, 9.8, "right")]
        self.assertEqual([c["name"] for c in P.calls_in(calls, 5.0, 7.0)], ["reach"])
        self.assertEqual(P.calls_in(calls, 7.0, 9.0), [])


class Kinds(unittest.TestCase):
    def test_kind_from_the_calls(self):
        self.assertEqual(P.kind_of([call("walk_to", 0, 1, [0, 0])]), "walk")
        self.assertEqual(P.kind_of([call("reach", 0, 1, "left", [0, 0, 0])]), "reach")
        self.assertEqual(P.kind_of([call("grip", 0, 1, "both")]), "grip")
        self.assertEqual(P.kind_of([call("reach", 0, 1, "left", [0, 0, 0]), call("grip", 1, 2, "left")]), "mixed")
        self.assertEqual(P.kind_of([]), "none")
        self.assertEqual(P.kind_of([call("hold", 0, 1, 2.0)]), "hold")

    def test_side_of_a_call(self):
        self.assertEqual(P.side_of(call("reach", 0, 1, "right", [0, 0, 0])), "right")
        self.assertEqual(P.side_of(call("grip", 0, 1, "both")), "both")
        self.assertIsNone(P.side_of(call("walk_to", 0, 1, [0, 0])))


class Roles(unittest.TestCase):
    def test_reach_binds_effector_and_nearest_object_as_target(self):
        calls = [call("reach", 0, 1, "right", [1.0, 0.0, 0.9])]
        r = P.roles_of("reach", calls, ["Mug", "Can"], held_before={"Mug": None, "Can": None},
                       positions={"Mug": [1.0, 0.05, 0.9], "Can": [2.0, 0.0, 0.9]})
        self.assertEqual(r["effector"], "right_hand")
        self.assertEqual(r["target"], "Mug")
        self.assertIsNone(r["carried"])

    def test_release_targets_the_object_held_before(self):
        calls = [call("release", 0, 1, "left")]
        r = P.roles_of("release", calls, ["Mug"], held_before={"Mug": "left_hand"}, positions={"Mug": [0, 0, 0]})
        self.assertEqual(r["target"], "Mug")
        self.assertEqual(r["carried"], "Mug")

    def test_walk_has_no_effector(self):
        r = P.roles_of("walk", [call("walk_to", 0, 1, [1.0, 2.0])], ["Mug"], held_before={"Mug": None},
                       positions={"Mug": [0, 0, 0]})
        self.assertIsNone(r["effector"])
        self.assertEqual(r["target"], "Mug")
```

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_scene_past.py"`
Expected: FAIL — `ModuleNotFoundError: No module named 'locomani.tasks.scene.past'`.

- [ ] **Step 3: Implement**

Create `src/locomani/tasks/scene/past.py`:

```python
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

SCHEMA = "PAST-locomani-0.1"
NEIGHBOUR_M = 0.30  # a movable object this close (xy) to the target is in scope
OVERHANG_M = 0.50  # a surface this far above the target is in scope
STANCE_M = 0.35  # the stance footprint's radius around walk_to's goal
CORRIDOR_HALF_M = 0.08  # the approach corridor's half width (pregrasp target -> object)
MOVED_CM = 2.0  # an object this far from where it was at the step's start has moved (failure_class.MOVED_CM)
SLIP_CM = 1.0  # a held object this much farther from the wrist slipped (digest.py's slip)
SUPPORT_BAND_M = 0.02  # an underside this close to a surface's top may rest on it
REACH_TOL_CM = 3.0  # a reach lands when the jaw centre is this close to its target (body.reach's miss)
PREDICATE_DT_TICKS = 5  # predicates and derived edges are re-evaluated every 0.1 s (episode.JUDGE_DT)
SIDES = ("left", "right")
HAND = {s: f"{s}_hand" for s in SIDES}
WALK_KINDS = ("walk", "turn", "sidestep", "posture", "hold")
KIND_OF_CALL = {"walk_to": "walk", "turn": "turn", "sidestep": "sidestep", "posture": "posture", "hold": "hold",
                "reach": "reach", "grip": "grip", "release": "release", "squeeze": "squeeze"}  # fmt: skip


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
    """The primitive calls whose start lies in [t0, t1); the last interval also takes a call starting exactly at t1."""
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
        goal = next((c.get("args") or [None, None])[1] for c in calls if c.get("name") == "reach")
        if goal is not None and task_objects:
            target = min(task_objects, key=lambda o: float(np.linalg.norm(np.asarray(positions[o][:3]) - np.asarray(goal[:3]))))
    if target is None and task_objects:
        target = task_objects[0]
    return dict(target=target, effector=effector, carried=carried)
```

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_scene_past.py"`
Expected: PASS (3 classes).

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add src/locomani/tasks/scene/past.py tests/test_scene_past.py
git commit -m "past: step intervals, kinds and roles from the primitive call log (TODO 39.2, part 1)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `past.py` — a frame of the trace, entities, scope and edges

**Files:**
- Modify: `src/locomani/tasks/scene/past.py`
- Test: `tests/test_scene_past.py`

**Interfaces:**
- Produces:
  - `Frame(arrays: dict, meta: dict, k: int)` — read-only view of tick `k`: `.t`, `.obj_pose(key) -> (pos, quat)`, `.obj_vel(key) -> (v_mps, w_radps)`, `.hand_pose(side) -> (pos, quat)` (wrist), `.jaw_m(side)`, `.pelvis() -> (pos, quat)`, `.feet() -> list | None`, `.contacts() -> set[tuple[str, str]]` of body names (empty + `.has_contacts False` when the channel is missing), `.touching(obj_key) -> set[str]` sides whose finger links touch it, `.gripped(obj_key) -> set[str]` sides with both fingers on it, `.state(keys) -> dict` in `predicates.py`'s format, `.arm_q(side) -> list[float] | None`.
  - `scope_of(frame, roles, kind, task, meta, calls) -> tuple[dict entities, dict scope]` — entities keyed by id (`object`, `effector`, `robot`, `surface`, `furniture`, `region`), `scope = {"selector": {...constants}, "initial": [...], "initial_why": {...}, "added": []}`.
  - `graph_of(frame, entities, roles, meta) -> dict` with `t`, `states`, `edges` (list of `{"s","r","o","v","src", ...numbers}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scene_past.py` a synthetic round builder and tests:

```python
import numpy as np


def arrays_for(n, objects, hands, contacts=(), bodies=None, dt=0.02):
    """A trace of n ticks. objects: {body_name: (n,7) poses}; hands: {side: (n,3) wrist pos}; contacts: [(k, a, b)]
    with body names; bodies: body-name table (index = id)."""
    bodies = list(bodies or ["world", "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link",
                             "left_dex1_finger_1_link", "left_dex1_finger_2_link",
                             "right_dex1_finger_1_link", "right_dex1_finger_2_link", *objects])
    out = dict(t=np.arange(n) * dt, mode=np.full(n, 2), robot=np.tile([0, 0, 0.79, 1, 0, 0, 0], (n, 1)).astype(float),
               ref_pelvis=np.zeros((n, 3)), ref_wrists=np.zeros((n, 2, 3)),
               wrists=np.stack([np.stack([hands["left"], hands["right"]], 1)]).reshape(n, 2, 3),
               jaw=np.full((n, 2), 0.0245), fingers=np.zeros((n, 4)), joints=np.zeros((n, 29)),
               primitive=np.zeros(n, dtype=np.int16), primitives=np.array(["reach"]), bodies=np.array(bodies))
    for name, poses in objects.items():
        out[f"obj_{name}"] = np.asarray(poses, dtype=float)
    ids = {b: i for i, b in enumerate(bodies)}
    out["contact_k"] = np.array([k for k, _, _ in contacts], dtype=np.int32)
    out["contact_a"] = np.array([min(ids[a], ids[b]) for _, a, b in contacts], dtype=np.int32)
    out["contact_b"] = np.array([max(ids[a], ids[b]) for _, a, b in contacts], dtype=np.int32)
    return out


META = {
    "objects": {
        "Mug": {"surface": "T", "free": True, "body": "Mug", "container": False,
                "box": {"half_extent": [0.04, 0.04, 0.05], "centre": [0, 0, 0]},
                "semantic": {"mass_kg": 0.3, "movable": True, "upright_axis": [0, 0, 1]}},
        "Can": {"surface": "T", "free": True, "body": "Can", "container": False,
                "box": {"half_extent": [0.03, 0.03, 0.06], "centre": [0, 0, 0]},
                "semantic": {"mass_kg": 0.4, "movable": True, "upright_axis": [0, 0, 1]}},
        "Far": {"surface": "T", "free": True, "body": "Far", "container": False,
                "box": {"half_extent": [0.03, 0.03, 0.03], "centre": [0, 0, 0]},
                "semantic": {"mass_kg": 0.1, "movable": True, "upright_axis": [0, 0, 1]}},
        "Table": {"free": False, "body": None, "container": False, "semantic": {"movable": False},
                  "box": {"half_extent": [0.6, 0.4, 0.35], "centre": [1.0, 0.0, 0.35]}},
    },
    "surfaces": {
        "T": {"xy": [1.0, 0.0], "top": 0.70, "half_extent": [0.6, 0.4], "yaw": 0.0, "of_object": "Table", "approach_edges": ["-x"]},
        "Shelf": {"xy": [1.0, 0.0], "top": 1.05, "half_extent": [0.6, 0.4], "yaw": 0.0, "of_object": "Table"},
    },
}


def mug_round(n=20):
    still = lambda p: np.tile([*p, 1, 0, 0, 0], (n, 1))  # noqa: E731
    objects = {"Mug": still([1.0, 0.0, 0.75]), "Can": still([1.0, 0.10, 0.76]), "Far": still([1.0, 0.39, 0.73])}
    hands = {"left": np.tile([0.2, 0.2, 0.9], (n, 1)), "right": np.linspace([0.3, -0.1, 0.9], [0.85, 0.0, 0.78], n)}
    return objects, hands


class Frames(unittest.TestCase):
    def test_frame_reads_poses_contacts_and_touching(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands, contacts=[(19, "Mug", "right_dex1_finger_1_link"),
                                                       (19, "Mug", "right_dex1_finger_2_link")])
        f = P.Frame(arr, META, 19)
        self.assertTrue(f.has_contacts)
        np.testing.assert_allclose(f.obj_pose("Mug")[0], [1.0, 0.0, 0.75])
        self.assertEqual(f.touching("Mug"), {"right"})
        self.assertEqual(f.gripped("Mug"), {"right"})
        self.assertEqual(f.touching("Can"), set())
        st = f.state(["Mug"])
        self.assertEqual(st["objects"]["Mug"]["gripped"], ["right"])

    def test_frame_without_contact_channel_reports_unknown(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands)
        for k in ("contact_k", "contact_a", "contact_b", "bodies"):
            del arr[k]
        f = P.Frame(arr, META, 0)
        self.assertFalse(f.has_contacts)
        self.assertEqual(f.touching("Mug"), set())


class Scope(unittest.TestCase):
    def test_reach_scope_has_roles_neighbour_overhang_and_regions_not_the_far_object(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands)
        f = P.Frame(arr, META, 0)
        roles = dict(target="Mug", effector="right_hand", carried=None)
        calls = [call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78])]
        task = dict(objects=["Mug"], goal=["held(Mug)"])
        entities, scope = P.scope_of(f, roles, "reach", task, META, calls)
        ids = set(scope["initial"])
        for want in ("Mug", "right_hand", "left_hand", "robot", "T", "Table", "Can", "Shelf", "region:corridor", "region:pregrasp_target"):
            self.assertIn(want, ids, want)
        self.assertNotIn("Far", ids)  # 0.39 m away: outside NEIGHBOUR_M and the corridor
        self.assertEqual(scope["initial_why"]["Can"], "neighbour 0.10 m")
        self.assertTrue(scope["initial_why"]["Shelf"].startswith("above target"))
        self.assertEqual(entities["Mug"]["type"], "object")
        self.assertEqual(entities["Mug"]["mass_kg"], 0.3)
        self.assertEqual(entities["region:corridor"]["shape"], "segment")
        self.assertEqual(scope["selector"]["NEIGHBOUR_M"], 0.30)

    def test_walk_scope_has_stance_region_and_no_hands(self):
        objects, hands = mug_round()
        f = P.Frame(arrays_for(20, objects, hands), META, 0)
        roles = dict(target="Mug", effector=None, carried=None)
        entities, scope = P.scope_of(f, roles, "walk", dict(objects=["Mug"], goal=["held(Mug)"]), META,
                                     [call("walk_to", 0.0, 0.38, [0.4, 0.0])])
        self.assertIn("region:stance", scope["initial"])
        self.assertNotIn("right_hand", scope["initial"])
        self.assertEqual(entities["region:stance"]["radius_m"], 0.35)


class Graph(unittest.TestCase):
    def edges(self, g, r):
        return [e for e in g["edges"] if e["r"] == r]

    def test_graph_edges_before_the_grasp(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands, contacts=[(0, "Mug", "Table")],
                         bodies=["world", "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link",
                                 "left_dex1_finger_1_link", "left_dex1_finger_2_link",
                                 "right_dex1_finger_1_link", "right_dex1_finger_2_link", "Mug", "Can", "Far", "Table"])
        meta = json.loads(json.dumps(META))
        meta["objects"]["Table"]["body"] = "Table"
        f = P.Frame(arr, meta, 0)
        roles = dict(target="Mug", effector="right_hand", carried=None)
        calls = [call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78])]
        entities, scope = P.scope_of(f, roles, "reach", dict(objects=["Mug"], goal=["held(Mug)"]), meta, calls)
        g = P.graph_of(f, entities, roles, meta)
        sup = self.edges(g, "supported_by")
        self.assertEqual([(e["s"], e["o"], e["v"], e["src"]) for e in sup if e["s"] == "Mug"], [("Mug", "T", True, "derived")])
        held = [e for e in self.edges(g, "held") if e["s"] == "Mug" and e["o"] == "right_hand"]
        self.assertEqual(held[0]["v"], False)
        self.assertEqual(held[0]["src"], "predicate")
        above = [e for e in self.edges(g, "above") if e["s"] == "Shelf"]
        self.assertTrue(above[0]["v"])
        self.assertAlmostEqual(above[0]["gap_m"], 1.05 - 0.80, places=3)
        beside = [e for e in self.edges(g, "beside") if e["s"] == "Can"]
        self.assertAlmostEqual(beside[0]["gap_m"], 0.10 - 0.04 - 0.03, places=3)
        self.assertEqual(beside[0]["side"], "+y")
        edge = [e for e in self.edges(g, "near_edge") if e["s"] == "Mug"][0]
        self.assertEqual(edge["edge"], "+y")  # the mug sits at y 0 on a 0.4 m half-width table: 0.40 to +y and -y, 0.60 to x
        reach = [e for e in self.edges(g, "reach_vec")][0]
        self.assertEqual(len(reach["v"]), 3)
        self.assertEqual(g["states"]["right_hand"]["att"], "level")
        self.assertIn("v_mps", g["states"]["Mug"])
        self.assertEqual(g["states"]["robot"]["pelvis_z_m"], 0.79)

    def test_unknown_contacts_make_support_approx_and_touching_unknown(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands)
        for k in ("contact_k", "contact_a", "contact_b", "bodies"):
            del arr[k]
        f = P.Frame(arr, META, 0)
        roles = dict(target="Mug", effector="right_hand", carried=None)
        entities, _ = P.scope_of(f, roles, "reach", dict(objects=["Mug"], goal=["held(Mug)"]), META,
                                 [call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78])])
        g = P.graph_of(f, entities, roles, META)
        touching = [e for e in g["edges"] if e["r"] == "touching" and e["s"] == "Mug" and e["o"] == "right_hand"][0]
        self.assertIsNone(touching["v"])
        self.assertEqual(touching["src"], "unknown")
        sup = [e for e in g["edges"] if e["r"] == "supported_by" and e["s"] == "Mug"][0]
        self.assertEqual((sup["v"], sup["src"]), (True, "approx"))
```

Add `import json` at the top of the test file.

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_scene_past.py"`
Expected: FAIL — `AttributeError: module ... has no attribute 'Frame'`.

- [ ] **Step 3: Implement**

Append to `past.py`:

```python
from locomani.scene_discovery import predicates as PR  # noqa: E402  (pure numpy; no simulator)

FINGER = {s: (f"{s}_dex1_finger_1_link", f"{s}_dex1_finger_2_link") for s in SIDES}
WRIST = {s: f"{s}_wrist_yaw_link" for s in SIDES}
ARM_JOINTS = {s: [f"{s}_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                              "wrist_pitch", "wrist_yaw")] for s in SIDES}  # fmt: skip


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrist_attitude(q) -> str:
    """level / tilted / down from the wrist's x axis pitch (the jaw points along +x): the body table's three bands."""
    pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, -float(_quat_to_mat(q)[2, 0])))))
    return "down" if pitch_deg < -60 else ("tilted" if pitch_deg < -20 else "level")


class Frame:
    """Tick `k` of a round's arrays, read as the recorder needs it. Object keys are scene-file keys; `arrays` holds
    `obj_<body>` per tracked body."""

    def __init__(self, arrays: dict, meta: dict, k: int):
        self.a, self.meta, self.k = arrays, meta, int(k)
        self.t = float(arrays["t"][self.k])
        self.has_contacts = all(key in arrays for key in ("contact_k", "contact_a", "contact_b", "bodies"))
        self._bodies = [str(b) for b in arrays["bodies"]] if self.has_contacts else []
        self._body_of = {o.get("body"): key for key, o in (meta.get("objects") or {}).items() if o.get("body")}
        self._arm_idx = self._arm_indices(arrays)
        self._contacts: set | None = None

    @staticmethod
    def _arm_indices(arrays) -> dict | None:
        try:
            from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
        except Exception:
            return None
        j = arrays.get("joints")
        if j is None or j.ndim != 2 or j.shape[1] != len(MUJOCO_JOINT_NAMES):
            return None
        return {s: [MUJOCO_JOINT_NAMES.index(n) for n in ARM_JOINTS[s]] for s in SIDES}

    # -- bodies and objects --
    def body_name(self, key: str) -> str | None:
        return (self.meta.get("objects") or {}).get(key, {}).get("body")

    def obj_pose(self, key: str):
        arr = self.a.get(f"obj_{self.body_name(key)}")
        if arr is None:
            o = self.meta["objects"][key]
            return np.asarray(o.get("init_pos", (0, 0, 0)), float), np.asarray(o.get("init_quat", (1, 0, 0, 0)), float)
        return np.asarray(arr[self.k][:3], float), np.asarray(arr[self.k][3:7], float)

    def obj_vel(self, key: str) -> tuple[float, float]:
        arr = self.a.get(f"obj_{self.body_name(key)}")
        if arr is None or self.k == 0:
            return 0.0, 0.0
        dt = float(self.a["t"][self.k] - self.a["t"][self.k - 1]) or 0.02
        v = float(np.linalg.norm(arr[self.k][:3] - arr[self.k - 1][:3]) / dt)
        q1, q0 = _quat_to_mat(arr[self.k][3:7]), _quat_to_mat(arr[self.k - 1][3:7])
        ang = math.acos(max(-1.0, min(1.0, (np.trace(q1 @ q0.T) - 1) / 2)))
        return round(v, 4), round(ang / dt, 4)

    def hand_pose(self, side: str):
        return np.asarray(self.a["wrists"][self.k][SIDES.index(side)], float), None

    def jaw_m(self, side: str) -> float:
        return float(2 * (np.mean(self.a["fingers"][self.k][2 * SIDES.index(side): 2 * SIDES.index(side) + 2]) + 0.023))

    def arm_q(self, side: str):
        return None if not self._arm_idx else [round(float(v), 4) for v in self.a["joints"][self.k][self._arm_idx[side]]]

    def pelvis(self):
        r = self.a["robot"][self.k]
        return np.asarray(r[:3], float), np.asarray(r[3:7], float)

    def ref_wrist(self, side: str):
        return np.asarray(self.a["ref_wrists"][self.k][SIDES.index(side)], float)

    # -- contacts --
    def contacts(self) -> set:
        if self._contacts is None:
            self._contacts = set()
            if self.has_contacts:
                ks = self.a["contact_k"]
                for i in np.nonzero(ks == self.k)[0]:
                    self._contacts.add((self._bodies[int(self.a["contact_a"][i])], self._bodies[int(self.a["contact_b"][i])]))
        return self._contacts

    def touches(self, body_a: str, body_b: str) -> bool | None:
        if not self.has_contacts:
            return None
        return (min(body_a, body_b), max(body_a, body_b)) in self.contacts() or (body_a, body_b) in self.contacts() \
            or (body_b, body_a) in self.contacts()

    def touching(self, key: str) -> set:
        b = self.body_name(key)
        return {s for s in SIDES if b and any(self.touches(b, f) for f in FINGER[s])}

    def gripped(self, key: str) -> set:
        b = self.body_name(key)
        return {s for s in SIDES if b and all(self.touches(b, f) for f in FINGER[s])}

    def state(self, keys) -> dict:
        objects = {}
        for k in keys:
            pos, quat = self.obj_pose(k)
            objects[k] = dict(pos=pos, quat=quat, touching=sorted(self.touching(k)), gripped=sorted(self.gripped(k)))
        pos, quat = self.pelvis()
        return dict(objects=objects, pelvis=dict(pos=pos, quat=quat), fallen=False)
```

Scope and entities:

```python
def _surface_body(meta: dict, s: str) -> str | None:
    surf = meta["surfaces"][s]
    if surf.get("body"):
        return surf["body"]
    of = surf.get("of_object")
    return (meta["objects"].get(of) or {}).get("body") if of else None


def _footprint(xy, surf) -> bool:
    half = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    return bool(np.all(np.abs(PR.to_surface_frame(np.asarray(xy, float), surf)) <= half + SUPPORT_BAND_M))


def support_of(frame: Frame, key: str, meta: dict) -> tuple[str | None, str]:
    """(surface key, src) the object rests on: the surface whose top is within SUPPORT_BAND_M under its underside and
    whose footprint holds it; `derived` when the contact channel confirms the touch, `approx` when it cannot."""
    scene = PR._Scene(meta)
    st = frame.state([key])
    xy, bottom = scene.underside(key, st)
    best = None
    for s, surf in meta["surfaces"].items():
        top = float(surf["top"])
        if top - SUPPORT_BAND_M <= bottom <= top + SUPPORT_BAND_M and _footprint(xy, surf):
            best = s if best is None or float(meta["surfaces"][best]["top"]) < top else best
    if best is None:
        return None, "derived" if frame.has_contacts else "approx"
    sb, ob = _surface_body(meta, best), frame.body_name(key)
    touched = frame.touches(ob, sb) if (sb and ob) else None
    return best, ("derived" if touched is not None else "approx")


def _entity_object(meta, key) -> dict:
    o = meta["objects"][key]
    sem = o.get("semantic") or {}
    box = o.get("box") or {}
    return dict(type="object", body=o.get("body"), mass_kg=sem.get("mass_kg"), half_extent_m=box.get("half_extent"),
                com_m=box.get("centre"), movable=bool(sem.get("movable", o.get("free"))), container=bool(o.get("container")),
                upright_axis=sem.get("upright_axis"))  # fmt: skip


def _entity_surface(meta, s) -> dict:
    surf = meta["surfaces"][s]
    return dict(type="surface", of=surf.get("of_object"), top_m=surf["top"], half_extent_m=surf["half_extent"],
                yaw=surf.get("yaw", 0.0), approach_edges=surf.get("approach_edges"), fixed=True)  # fmt: skip


def _entity_furniture(meta, key) -> dict:
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


def _reach_goal(calls) -> np.ndarray | None:
    for c in calls:
        if c.get("name") == "reach" and len(c.get("args") or []) > 1 and c["args"][1] is not None:
            return np.asarray(c["args"][1][:3], float)
    return None


def _walk_goal(calls) -> np.ndarray | None:
    for c in calls:
        if c.get("name") == "walk_to" and (c.get("args") or []):
            return np.asarray(c["args"][0][:2], float)
    return None


def _xy_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))


def _in_corridor(p, a, b, half) -> bool:
    """Point p (xy) within `half` of segment a-b (xy)."""
    a, b, p = (np.asarray(v, float)[:2] for v in (a, b, p))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab))) <= half


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
            if of:
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
            if d <= NEIGHBOUR_M:
                add(o, _entity_object(meta, o), f"neighbour {d:.2f} m")
            elif reach_goal is not None and _in_corridor(p, reach_goal, tpos, CORRIDOR_HALF_M + max(od["box"]["half_extent"][:2])):
                add(o, _entity_object(meta, o), "in corridor")
        top_t = float(tpos[2] + objs[tgt]["box"]["centre"][2] + objs[tgt]["box"]["half_extent"][2])
        for s, surf in meta["surfaces"].items():
            if s in entities:
                continue
            gap = float(surf["top"]) - top_t
            if 0.0 < gap <= OVERHANG_M and _footprint(tpos[:2], surf):
                add(s, _entity_surface(meta, s), f"above target {gap:.2f} m")
    if walk_goal is not None:
        add("region:stance", dict(type="region", shape="disc", centre=[float(v) for v in walk_goal], radius_m=STANCE_M), "walk_to goal")
        ppos = frame.pelvis()[0]
        for key, od in objs.items():
            if key in entities or od.get("free") or not od.get("box"):
                continue
            c = np.asarray(od["box"]["centre"], float)[:2] + (np.asarray(od.get("init_pos", (0, 0, 0)), float)[:2])
            r = float(np.hypot(*od["box"]["half_extent"][:2]))
            if _in_corridor(c, ppos, walk_goal, STANCE_M + r):
                add(key, _entity_furniture(meta, key), "on the walk path")
    if reach_goal is not None and not walk:
        add("region:pregrasp_target", dict(type="region", shape="point", p=[float(v) for v in reach_goal],
                                           wrist_att=None), "reach goal")  # fmt: skip
        if tpos is not None:
            add("region:corridor", dict(type="region", shape="segment", a=[float(v) for v in reach_goal],
                                        b=[float(v) for v in tpos], half_width_m=CORRIDOR_HALF_M), "approach corridor")  # fmt: skip
    if g_surface in meta["surfaces"] and kind in ("release", "reach", "mixed"):
        surf = meta["surfaces"][g_surface]
        add("region:place", dict(type="region", shape="rect", centre=[*surf["xy"], surf["top"]], half_extent_m=surf["half_extent"],
                                 yaw=surf.get("yaw", 0.0)), "goal placement")  # fmt: skip
    selector = dict(NEIGHBOUR_M=NEIGHBOUR_M, OVERHANG_M=OVERHANG_M, STANCE_M=STANCE_M, CORRIDOR_HALF_M=CORRIDOR_HALF_M)
    return entities, dict(selector=selector, initial=list(entities), initial_why=why, added=[])
```

The graph:

```python
def _edge(s, r, o, v, src, **num) -> dict:
    return dict(s=s, r=r, o=o, v=v, src=src, **num)


def _blocking(frame: Frame, eid: str, ent: dict, region: dict, meta: dict) -> tuple[bool | None, float]:
    """Does entity `eid`'s box intersect the region (disc or segment corridor)? (value, depth_m)."""
    if ent["type"] == "object":
        c = frame.obj_pose(eid)[0][:2]
        r = float(np.hypot(*(meta["objects"][eid]["box"]["half_extent"][:2])))
    elif ent["type"] == "furniture" and ent.get("box_m"):
        o = meta["objects"][eid]
        c = np.asarray(ent["box_m"]["centre"], float)[:2] + np.asarray(o.get("init_pos", (0, 0, 0)), float)[:2]
        r = float(np.hypot(*ent["box_m"]["half_extent"][:2]))
    else:
        return None, 0.0
    if region["shape"] == "disc":
        d = _xy_dist(c, region["centre"])
        depth = region["radius_m"] + r - d
    elif region["shape"] == "segment":
        a, b = np.asarray(region["a"], float)[:2], np.asarray(region["b"], float)[:2]
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        depth = region["half_width_m"] + r - float(np.linalg.norm(c - (a + t * ab)))
    else:
        return None, 0.0
    return depth > 0.0, round(max(depth, 0.0), 3)


def _side_of(target_xy, other_xy, yaw) -> str:
    d = np.asarray(other_xy, float) - np.asarray(target_xy, float)
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y = c * d[0] - s * d[1], s * d[0] + c * d[1]
    return ("+x" if x > 0 else "-x") if abs(x) >= abs(y) else ("+y" if y > 0 else "-y")


def graph_of(frame: Frame, entities: dict, roles: dict, meta: dict) -> dict:
    """Spec section 4.4.1: states of every entity in scope and the edges around the roles (target, effector, robot,
    regions); never all pairs."""
    objs = meta["objects"]
    scene = PR._Scene(meta)
    states, edges = {}, []
    tgt, eff = roles.get("target"), roles.get("effector")
    ppos, pquat = frame.pelvis()
    yaw = _yaw(pquat)
    for eid, ent in entities.items():
        if ent["type"] == "object":
            p, q = frame.obj_pose(eid)
            v, w = frame.obj_vel(eid)
            half, centre, _ = scene.box(eid)
            states[eid] = dict(p=[round(float(x), 4) for x in p], q=[round(float(x), 5) for x in q], v_mps=v, w_radps=w,
                               bottom_z_m=round(float(p[2] + centre[2] - half[2]), 4), yaw=round(_yaw(q), 3))  # fmt: skip
        elif ent["type"] == "effector":
            if eid == "both_hands":
                continue
            side = eid.split("_")[0]
            hp, _ = frame.hand_pose(side)
            st = dict(p=[round(float(x), 4) for x in hp], jaw_m=round(frame.jaw_m(side), 4),
                      ref_err_cm=round(float(np.linalg.norm(frame.ref_wrist(side) - hp)) * 100, 2))  # fmt: skip
            aq = frame.arm_q(side)
            if aq is not None:
                st["arm_q"] = aq
            st["att"] = "level"  # the wrist quaternion is not in the trace; refined when ref_wrists carries rotation
            states[eid] = st
        elif ent["type"] == "robot":
            states[eid] = dict(pelvis_p=[round(float(x), 4) for x in ppos], pelvis_z_m=round(float(ppos[2]), 3),
                               yaw=round(yaw, 3), fell=False,
                               primitive=str(frame.a["primitives"][int(frame.a["primitive"][frame.k])]) if "primitives" in frame.a else None,
                               mode=int(frame.a["mode"][frame.k]) if "mode" in frame.a else None)  # fmt: skip
    hands = [e for e, ent in entities.items() if ent["type"] == "effector" and e != "both_hands"]
    objects_in = [e for e, ent in entities.items() if ent["type"] == "object"]
    surfaces_in = [e for e, ent in entities.items() if ent["type"] == "surface"]
    regions = {e: ent for e, ent in entities.items() if ent["type"] == "region"}
    # contact, holding, support (around the target and the hands)
    for o in objects_in:
        if o != tgt and o != roles.get("carried"):
            continue
        ob = frame.body_name(o)
        for h in hands:
            side = h.split("_")[0]
            t = None if not frame.has_contacts else bool(frame.touching(o) & {side})
            edges.append(_edge(o, "touching", h, t, "observed" if t is not None else "unknown"))
            st = frame.state([o])
            held = None if not frame.has_contacts else PR._held(scene, o, st)[0] and side in st["objects"][o]["gripped"]
            edges.append(_edge(o, "held", h, held, "predicate" if held is not None else "unknown"))
            if held:
                rel = frame.hand_pose(side)[0] - frame.obj_pose(o)[0]
                edges.append(_edge(o, "rel_pose", h, [round(float(x), 4) for x in rel], "derived"))
            else:
                edges.append(_edge(o, "rel_pose", h, None, "derived"))
        s, src = support_of(frame, o, meta)
        for cand in surfaces_in:
            edges.append(_edge(o, "supported_by", cand, cand == s, src))
        for c in objects_in:
            if c != o and objs[c].get("container"):
                edges.append(_edge(o, "in", c, PR._in(scene, o, c, frame.state([o, c]))[0], "predicate"))
        edges.append(_edge(o, "upright", None, PR._upright(scene, o, frame.state([o]))[0], "predicate"))
        if ob:
            for other in objects_in:
                if other != o:
                    t = frame.touches(ob, frame.body_name(other)) if frame.body_name(other) else None
                    edges.append(_edge(o, "touching", other, t, "observed" if t is not None else "unknown"))
    # spatial (around the target)
    if tgt in objects_in:
        tp = frame.obj_pose(tgt)[0]
        thalf = np.asarray(objs[tgt]["box"]["half_extent"], float)
        top_t = float(tp[2] + objs[tgt]["box"]["centre"][2] + thalf[2])
        for s in surfaces_in:
            surf = meta["surfaces"][s]
            gap = float(surf["top"]) - top_t
            edges.append(_edge(s, "above", tgt, 0.0 < gap <= OVERHANG_M and _footprint(tp[:2], surf), "derived", gap_m=round(gap, 3)))
            sup, _ = support_of(frame, tgt, meta)
            if s == sup:
                local = PR.to_surface_frame(tp[:2], surf)
                half = np.asarray(surf["half_extent"], float)[:2]
                dists = {"+x": half[0] - local[0], "-x": half[0] + local[0], "+y": half[1] - local[1], "-y": half[1] + local[1]}
                e = min(dists, key=dists.get)
                edges.append(_edge(tgt, "near_edge", s, True, "derived", dist_m=round(float(dists[e]), 3), edge=e))
        for o in objects_in:
            if o == tgt:
                continue
            op = frame.obj_pose(o)[0]
            ohalf = np.asarray(objs[o]["box"]["half_extent"], float)
            d = _xy_dist(op, tp)
            gap = d - float(max(thalf[:2])) - float(max(ohalf[:2]))
            edges.append(_edge(o, "beside", tgt, d <= NEIGHBOUR_M, "derived", gap_m=round(gap, 3), side=_side_of(tp[:2], op[:2], _yaw(frame.obj_pose(tgt)[1]))))
        d = _xy_dist(ppos, tp)
        bearing = math.degrees(math.atan2(tp[1] - ppos[1], tp[0] - ppos[0]) - yaw)
        bearing = (bearing + 180) % 360 - 180
        edges.append(_edge("robot", "dist", tgt, round(d, 3), "derived", bearing_deg=round(bearing, 1)))
        edges.append(_edge("robot", "facing", tgt, round(abs(bearing), 1), "derived"))
    for rid, region in regions.items():
        if region["shape"] not in ("disc", "segment"):
            continue
        for eid, ent in entities.items():
            if eid in (tgt, "robot") or ent["type"] not in ("object", "furniture"):
                continue
            v, depth = _blocking(frame, eid, ent, region, meta)
            if v is None:
                continue
            edges.append(_edge(eid, "blocking", rid, v, "derived" if frame.has_contacts else "approx", depth_m=depth))
    # relative pose (hand -> target region, robot frame)
    if eff in hands and "region:pregrasp_target" in regions:
        hp = frame.hand_pose(eff.split("_")[0])[0]
        goal = np.asarray(regions["region:pregrasp_target"]["p"], float)
        rel = goal - ppos
        c, s = math.cos(-yaw), math.sin(-yaw)
        fwd, left = c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]
        out = -left if eff.startswith("right") else left
        edges.append(_edge(eff, "reach_vec", "region:pregrasp_target", [round(float(fwd), 3), round(float(out), 3), round(float(goal[2] - ppos[2]), 3)], "derived"))
        edges.append(_edge(eff, "at", "region:pregrasp_target", float(np.linalg.norm(hp - goal)) * 100 <= REACH_TOL_CM, "derived",
                           err_cm=round(float(np.linalg.norm(hp - goal)) * 100, 2)))  # fmt: skip
        if tgt in objects_in:
            edges.append(_edge(eff, "approach", tgt, _side_of(frame.obj_pose(tgt)[0][:2], hp[:2], _yaw(frame.obj_pose(tgt)[1])), "derived",
                               wrist_att=states.get(eff, {}).get("att")))  # fmt: skip
    return dict(t=round(frame.t, 3), states=states, edges=edges)
```

Note for the implementer: the trace holds the wrist *position* only (`wrists`), so `att` is `"level"` with `src` implicit; spec §4.3 lists `att` and the plan keeps the key so the schema is stable. If `arrays` later gains wrist quaternions, `wrist_attitude()` is already there. Do not add a quaternion to the trace in this task.

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_scene_past.py"`
Expected: PASS. If `near_edge` picks `-y` instead of `+y` (tie at y 0), change the assertion to `assertIn(edge["edge"], ("+y", "-y"))` — the tie is real.

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add src/locomani/tasks/scene/past.py tests/test_scene_past.py
git commit -m "past: frames, relevance scope, entities and the five relation classes (TODO 39.2, part 2)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `past.py` — events, expected transitions, `build()` and `write()`

**Files:**
- Modify: `src/locomani/tasks/scene/past.py`
- Test: `tests/test_scene_past.py`

**Interfaces:**
- Produces:
  - `expected_of(kind, roles, task, g_before) -> list[dict]` (spec §4.5 table).
  - `events_of(arrays, meta, k0, k1, entities, roles) -> tuple[list[dict] events, list[dict] added]`.
  - `discrepancies_of(expected, g_after, events) -> list[dict]`.
  - `build(rec: dict, arrays: dict | None, meta: dict, task) -> list[dict]` where `task` has `.id`, `.objects`, `.goal` (the scene `Task`) or is a dict with those keys.
  - `write(out_dir: Path, label: str, records: list[dict]) -> Path` → `<out_dir>/<label>.past.jsonl` via `.tmp` + rename.
  - `error_line(label: str, err: str) -> dict`.
  - `read(path: Path) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scene_past.py`:

```python
import tempfile
from pathlib import Path


def grasp_round_arrays(n=40):
    """Reach (ticks 0-19), grip (20-29), lift (30-39): the mug is touched at 22, gripped from 24, leaves the table at 32."""
    still = lambda p, m: np.tile([*p, 1, 0, 0, 0], (m, 1))  # noqa: E731
    mug = np.vstack([still([1.0, 0.0, 0.75], 32), np.hstack([np.tile([1.0, 0.0], (8, 1)), np.linspace(0.75, 0.85, 8)[:, None], np.tile([1, 0, 0, 0], (8, 1))])])
    can = still([1.0, 0.10, 0.76], n)
    right = np.vstack([np.linspace([0.3, -0.1, 0.9], [0.85, 0.0, 0.78], 20), np.tile([0.85, 0.0, 0.78], (12, 1)),
                       np.hstack([np.tile([0.85, 0.0], (8, 1)), np.linspace(0.78, 0.88, 8)[:, None]])])
    left = np.tile([0.2, 0.2, 0.9], (n, 1))
    bodies = ["world", "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_dex1_finger_1_link",
              "left_dex1_finger_2_link", "right_dex1_finger_1_link", "right_dex1_finger_2_link", "Mug", "Can", "Far", "Table"]
    contacts = [(k, "Mug", "Table") for k in range(0, 32)]
    contacts += [(k, "Mug", "right_dex1_finger_1_link") for k in range(22, n)]
    contacts += [(k, "Mug", "right_dex1_finger_2_link") for k in range(24, n)]
    arr = arrays_for(n, {"Mug": mug, "Can": can, "Far": still([1.0, 0.39, 0.73], n)}, {"left": left, "right": right},
                     contacts=contacts, bodies=bodies)
    return arr


def grasp_round_rec(ok=True):
    return dict(label="pick_Mug_s1", task="pick_Mug", ok=ok, why="" if ok else "[goal] held(Mug): no hand touches it",
                fell=False, dropped=False, plan_ok=ok, plan_why="", error="", skills="abc123", set={"X": 1},
                failure_class="" if ok else "slip", events={},
                steps=[step("settle", 0.0, 0.0), step("pregrasp", 0.0, 0.40), step("grip", 0.40, 0.60, check={"opening_cm": 6.1}, attempt=1),
                       step("lift", 0.60, 0.78)],
                calls=[call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78]), call("grip", 0.40, 0.58, "right"),
                       call("reach", 0.60, 0.78, "right", [0.85, 0.0, 0.88])])


META_T = json.loads(json.dumps(META))
META_T["objects"]["Table"]["body"] = "Table"
TASK = dict(id="pick_Mug", objects=["Mug"], goal=["held(Mug)"])


class Expected(unittest.TestCase):
    def test_grip_expects_held_true_and_reach_expects_landing_and_no_blocking(self):
        roles = dict(target="Mug", effector="right_hand", carried=None)
        ex = P.expected_of("grip", roles, TASK, dict(edges=[]))
        self.assertIn({"edge": ["Mug", "held", "right_hand"], "to": True}, ex)
        ex = P.expected_of("reach", roles, TASK, dict(edges=[]))
        self.assertIn({"edge": ["right_hand", "at", "region:pregrasp_target"], "to": True, "tol_cm": 3.0}, ex)
        self.assertIn({"edge": ["*", "blocking", "region:corridor"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "moved", None], "to": False}, ex)

    def test_lift_reach_while_held_expects_support_to_drop(self):
        roles = dict(target="Mug", effector="right_hand", carried="Mug")
        g = dict(edges=[dict(s="Mug", r="supported_by", o="T", v=True, src="derived")])
        ex = P.expected_of("reach", roles, TASK, g, rising=True)
        self.assertIn({"edge": ["Mug", "supported_by", "T"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "held", "right_hand"], "to": True}, ex)

    def test_release_toward_goal_surface(self):
        task = dict(id="put", objects=["Mug"], goal=["on(Mug, T)"])
        roles = dict(target="Mug", effector="right_hand", carried="Mug")
        ex = P.expected_of("release", roles, task, dict(edges=[]))
        self.assertIn({"edge": ["Mug", "held", "right_hand"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "supported_by", "T"], "to": True}, ex)


class Build(unittest.TestCase):
    def test_build_makes_one_record_per_step_with_events_and_discrepancies(self):
        recs = P.build(grasp_round_rec(ok=True), grasp_round_arrays(), META_T, TASK)
        self.assertEqual([r["step"] for r in recs], ["settle", "pregrasp", "grip", "lift"])
        self.assertEqual([r["kind"] for r in recs], ["none", "reach", "grip", "reach"])
        self.assertTrue(all(r["schema"] == P.SCHEMA for r in recs))
        self.assertEqual(recs[2]["record_id"], "pick_Mug_s1/2.1")
        grip = recs[2]
        flips = [(e["edge"], e["to"]) for e in grip["events"]]
        self.assertIn((["Mug", "touching", "right_hand"], True), flips)
        self.assertIn((["Mug", "held", "right_hand"], True), flips)
        self.assertEqual(grip["discrepancies"], [])
        lift = recs[3]
        self.assertIn((["Mug", "supported_by", "T"], False), [(e["edge"], e["to"]) for e in lift["events"]])
        self.assertEqual(lift["outcome"]["round_ok"], True)
        self.assertEqual(lift["evidence"]["npz"], "pick_Mug_s1.npz")
        self.assertEqual((lift["evidence"]["k0"], lift["evidence"]["k1"]), (30, 39))
        self.assertEqual(grip["action"]["check"], {"opening_cm": 6.1})
        self.assertEqual(grip["roles"]["effector"], "right_hand")

    def test_grip_that_never_closes_shows_the_discrepancy(self):
        arr = grasp_round_arrays()
        keep = arr["contact_a"] != list(arr["bodies"]).index("right_dex1_finger_2_link")
        keep &= arr["contact_b"] != list(arr["bodies"]).index("right_dex1_finger_2_link")
        for k in ("contact_k", "contact_a", "contact_b"):
            arr[k] = arr[k][keep]
        recs = P.build(grasp_round_rec(ok=False), arr, META_T, TASK)
        grip = recs[2]
        self.assertEqual([d["edge"] for d in grip["discrepancies"]], [["Mug", "held", "right_hand"]])
        self.assertEqual(grip["outcome"]["failure_class"], "slip")

    def test_build_without_arrays_returns_no_records_and_error_line_is_a_dict(self):
        self.assertEqual(P.build(dict(label="x", steps=[], calls=[]), None, META_T, TASK), [])
        e = P.error_line("x", "boom")
        self.assertEqual((e["schema"], e["label"], e["error"]), (P.SCHEMA, "x", "boom"))

    def test_write_and_read_roundtrip(self):
        recs = P.build(grasp_round_rec(), grasp_round_arrays(), META_T, TASK)
        with tempfile.TemporaryDirectory() as d:
            p = P.write(Path(d), "pick_Mug_s1", recs)
            self.assertEqual(p.name, "pick_Mug_s1.past.jsonl")
            self.assertFalse((Path(d) / "pick_Mug_s1.past.jsonl.tmp").exists())
            back = P.read(p)
        self.assertEqual(len(back), len(recs))
        self.assertEqual(back[1]["record_id"], recs[1]["record_id"])
```

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_scene_past.py -k 'Expected or Build'"`
Expected: FAIL — `AttributeError: ... 'expected_of'`.

- [ ] **Step 3: Implement**

Append to `past.py`:

```python
def expected_of(kind: str, roles: dict, task: dict, g_before: dict, *, rising: bool = False) -> list[dict]:
    """Spec section 4.5: the transitions the primitive kind is for, by primitive, never by object or step name."""
    tgt, eff, carried = roles.get("target"), roles.get("effector"), roles.get("carried")
    g_surface, g_container = _goal_targets(task.get("goal") or [])
    out: list[dict] = []
    support_now = [e["o"] for e in g_before.get("edges", []) if e["r"] == "supported_by" and e["s"] == tgt and e["v"]]
    if kind == "walk":
        out.append(dict(edge=["robot", "at", "region:stance"], to=True))
        out.append(dict(edge=["*", "blocking", "region:stance"], to=False))
    elif kind == "reach" and eff:
        if carried and tgt == carried:
            out.append(dict(edge=[tgt, "held", eff], to=True))
            if rising:
                for s in support_now:
                    out.append(dict(edge=[tgt, "supported_by", s], to=False))
        else:
            out.append(dict(edge=[eff, "at", "region:pregrasp_target"], to=True, tol_cm=REACH_TOL_CM))
            if tgt:
                out.append(dict(edge=[tgt, "moved", None], to=False))
            out.append(dict(edge=["*", "blocking", "region:corridor"], to=False))
            out.append(dict(edge=["*", "moved", None], to=False, scope="neighbours"))
    elif kind in ("grip", "squeeze") and eff and tgt:
        if eff == "both_hands":
            out += [dict(edge=[tgt, "held", HAND[s]], to=True) for s in SIDES]
        else:
            out.append(dict(edge=[tgt, "held", eff], to=True))
    elif kind == "release" and eff and tgt:
        out.append(dict(edge=[tgt, "held", eff], to=False))
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
    """Spec section 4.6 step 5: boolean edge flips between ticks k0 and k1 (touching every tick, predicates and derived
    edges every PREDICATE_DT_TICKS), moved / slipped, and the bodies that joined the scope by contact."""
    events, added = [], []
    f0 = Frame(arrays, meta, k0)
    prev = _bool_edges(graph_of(f0, entities, roles, meta))
    p0 = {o: f0.obj_pose(o)[0] for o, e in entities.items() if e["type"] == "object"}
    moved = {o: False for o in p0}
    hands = [e for e, ent in entities.items() if ent["type"] == "effector" and e != "both_hands"]
    carried, eff = roles.get("carried"), roles.get("effector")
    d0 = None
    if carried and eff in hands:
        d0 = float(np.linalg.norm(f0.hand_pose(eff.split("_")[0])[0] - f0.obj_pose(carried)[0]))
    slipped = False
    known = set(entities)
    body_to_key = {o.get("body"): k for k, o in meta["objects"].items() if o.get("body")}
    watched = {meta["objects"][o]["body"] for o in task.get("objects", ()) if meta["objects"][o].get("body")}
    for k in range(k0 + 1, k1 + 1):
        f = Frame(arrays, meta, k)
        if f.has_contacts:
            for a, b in f.contacts():
                for mine, other in ((a, b), (b, a)):
                    if (mine in watched or any(mine.startswith(s) for s in SIDES) or mine == "pelvis") and other not in watched:
                        key = body_to_key.get(other, other)
                        if key not in known and key in meta["objects"]:
                            known.add(key)
                            entities[key] = _entity_object(meta, key) if meta["objects"][key].get("free") else _entity_furniture(meta, key)
                            added.append(dict(id=key, t=round(f.t, 3), why=f"touched by {mine}"))
        full = (k - k0) % PREDICATE_DT_TICKS == 0 or k == k1
        g = graph_of(f, entities, roles, meta)
        now = _bool_edges(g)
        for key, v in now.items():
            if key[1] != "touching" and not full:
                continue
            if key in prev and prev[key] != v and v is not None and prev[key] is not None:
                events.append(dict(t=round(f.t, 3), edge=list(key), **{"from": prev[key]}, to=v,
                                   src=next(e["src"] for e in g["edges"] if _edge_key(e) == key)))  # fmt: skip
            prev[key] = v
        for o in p0:
            if not moved[o]:
                cm = float(np.linalg.norm(f.obj_pose(o)[0] - p0[o])) * 100
                if cm > MOVED_CM and not (o == carried):
                    moved[o] = True
                    events.append(dict(t=round(f.t, 3), edge=[o, "moved", None], **{"from": False}, to=True, cm=round(cm, 1), src="derived"))
        if d0 is not None and not slipped:
            d = float(np.linalg.norm(f.hand_pose(eff.split("_")[0])[0] - f.obj_pose(carried)[0]))
            if abs(d - d0) * 100 > SLIP_CM:
                slipped = True
                events.append(dict(t=round(f.t, 3), edge=[carried, "slipped", eff], **{"from": False}, to=True, cm=round((d - d0) * 100, 1), src="derived"))
    return events, added


def discrepancies_of(expected: list[dict], g_after: dict, events: list[dict]) -> list[dict]:
    after = {_edge_key(e): e for e in g_after["edges"]}
    out = []
    for ex in expected:
        s, r, o = ex["edge"]
        if r in ("moved", "slipped"):
            hit = [e for e in events if e["edge"][1] == r and (s == "*" or e["edge"][0] == s)]
            if bool(hit) != ex["to"]:
                out.append(dict(**ex, observed=bool(hit)))
            continue
        if s == "*":
            vals = [e["v"] for e in g_after["edges"] if e["r"] == r and e["o"] == o and e["v"] is not None]
            observed = any(vals) if vals else None
        else:
            e = after.get((s, r, o))
            observed = None if e is None else e["v"]
        if observed is not None and observed != ex["to"]:
            out.append(dict(**ex, observed=observed))
    return out


def _tick_of(arrays: dict, t: float, last: bool = False) -> int:
    ts = arrays["t"]
    k = int(np.searchsorted(ts, t - 1e-9, side="left"))
    k = min(max(k, 0), len(ts) - 1)
    if last and k > 0 and ts[k] > t + 1e-9:
        k -= 1
    return k


def _task_dict(task) -> dict:
    if isinstance(task, dict):
        return dict(id=task.get("id"), objects=list(task.get("objects") or []), goal=list(task.get("goal") or []))
    return dict(id=getattr(task, "id", None), objects=list(getattr(task, "objects", ()) or ()), goal=list(getattr(task, "goal", ()) or ()))


def build(rec: dict, arrays: dict | None, meta: dict, task) -> list[dict]:
    """One record per step row of `rec` (spec section 4.6). No arrays (the plan was refused before the world moved):
    no records."""
    if arrays is None or "t" not in arrays or len(arrays["t"]) == 0:
        return []
    task = _task_dict(task)
    steps, calls = rec.get("steps") or [], rec.get("calls") or []
    label = rec.get("label", "")
    out = []
    held_state: dict = {o: None for o in task["objects"]}
    for iv in intervals(steps):
        k0, k1 = _tick_of(arrays, iv.t0), _tick_of(arrays, iv.t1, last=True)
        k1 = max(k0, k1)
        cs = calls_in(calls, iv.t0, iv.t1 if iv.t1 > iv.t0 else iv.t0 + 1e-6)
        kind = kind_of(cs)
        f0, f1 = Frame(arrays, meta, k0), Frame(arrays, meta, k1)
        positions = {o: f0.obj_pose(o)[0] for o in task["objects"] if o in meta["objects"]}
        roles = roles_of(kind, cs, task["objects"], held_before=held_state, positions=positions)
        entities, scope = scope_of(f0, roles, kind, task, meta, cs)
        g_before = graph_of(f0, entities, roles, meta)
        rising = False
        if kind == "reach" and roles.get("carried"):
            goal = _reach_goal(cs)
            rising = goal is not None and float(goal[2]) > float(f0.hand_pose(roles["effector"].split("_")[0])[0][2]) + 0.02
        expected = expected_of(kind, roles, task, g_before, rising=rising)
        events, added = events_of(arrays, meta, k0, k1, entities, roles, task)
        scope["added"] = added
        g_after = graph_of(f1, entities, roles, meta)
        disc = discrepancies_of(expected, g_after, events)
        # support / goal surface / container roles for the record (spec 4.3)
        sup, _ = support_of(f0, roles["target"], meta) if roles.get("target") in meta["objects"] else (None, "")
        g_surface, g_container = _goal_targets(task["goal"])
        roles_out = dict(roles, support=sup, goal_surface=g_surface, container=g_container,
                         stance="region:stance" if "region:stance" in entities else None,
                         corridor="region:corridor" if "region:corridor" in entities else None,
                         pregrasp="region:pregrasp_target" if "region:pregrasp_target" in entities else None)  # fmt: skip
        row = iv.row
        action = dict(t0=iv.t0, t1=iv.t1, ok=bool(row.get("ok")), why=str(row.get("why", "")),
                      calls=[{k: v for k, v in c.items() if k != "kwargs" or v} for c in cs],
                      check=row.get("check") or {},
                      info={k: v for k, v in row.items() if k not in ("step", "ok", "why", "t0", "t1", "check", "attempt")})  # fmt: skip
        out.append(dict(
            schema=SCHEMA, record_id=f"{label}/{iv.index}.{iv.attempt}", label=label, task=task["id"],
            step_index=iv.index, attempt=iv.attempt, step=iv.step, kind=kind,
            skills=rec.get("skills"), set=rec.get("set") or {}, predicate_version="predicates.py",
            action=action, roles=roles_out, scope=scope, entities=entities,
            G_before=g_before, events=events, G_after=g_after, expected=expected, discrepancies=disc,
            outcome=dict(step_ok=bool(row.get("ok")), round_ok=bool(rec.get("ok")), round_why=rec.get("why", ""),
                         failure_class=rec.get("failure_class")),
            evidence=dict(npz=f"{label}.npz", k0=k0, k1=k1, json_step=iv.index),
        ))  # fmt: skip
        # who holds what at the end of this step feeds the next step's roles
        for o in task["objects"]:
            g = f1.gripped(o) if f1.has_contacts else set()
            held_state[o] = HAND[next(iter(g))] if len(g) == 1 else ("both_hands" if len(g) == 2 else None)
    return out


def error_line(label: str, err: str) -> dict:
    return dict(schema=SCHEMA, label=label, error=str(err)[:300])


def write(out_dir: Path, label: str, records: list[dict]) -> Path:
    """`<label>.past.jsonl`, one JSON object per line, written whole (tmp + rename) like the round's .json."""
    p = Path(out_dir) / f"{label}.past.jsonl"
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in records))
    tmp.rename(p)
    return p


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.round(5).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
```

Also in Task 3's `graph_of`, make sure `held` for `both_hands` and the `at` edge are covered: the `at` edge is emitted only when `region:pregrasp_target` is in scope (done above); for `walk`, emit in `graph_of` when `region:stance` is in `regions`:

```python
    if "region:stance" in regions:
        c = np.asarray(regions["region:stance"]["centre"], float)
        err = _xy_dist(ppos, c) * 100
        edges.append(_edge("robot", "at", "region:stance", err <= 5.0, "derived", err_cm=round(err, 1)))  # failure_class.WALK_OFF_CM
```

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_scene_past.py"`
Expected: PASS. If the `(k0, k1) == (30, 39)` assertion is off by one, check `_tick_of`: `t=0.60` at `dt 0.02` is tick 30, `t=0.78` is tick 39.

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add src/locomani/tasks/scene/past.py tests/test_scene_past.py
git commit -m "past: events, expected transitions, discrepancies, build() and the .past.jsonl writer (TODO 39.2)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Write the records from `scene_round.py`; retire them with the round

**Files:**
- Modify: `scripts/rsi/scene_round.py` (imports near the top; `run_round` after `rec.update(...)`, before `tmp = f.with_suffix(".tmp")`)
- Modify: `scripts/rsi/batch.py:118-127` (`retire`)
- Test: `tests/test_rsi_past_tools.py` (create)

**Interfaces:**
- Consumes: `past.build`, `past.write`, `past.error_line` (Task 4).
- Produces: `<out>/<label>.past.jsonl` beside every round; `batch.retire(f)` renames `.past.jsonl` with the same `_<tag>_` prefix.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rsi_past_tools.py`:

```python
"""TODO 39.3 / 39.4: the recorder's side files in the loop's tools (batch.retire, digest, past.py CLI)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import batch as B  # noqa: E402


class Retire(unittest.TestCase):
    def test_retire_renames_the_past_side_file_too(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pick_s1.json"
            f.write_text(json.dumps(dict(label="pick_s1", ok=True, rule=B.SUCCESS_RULE, motor="old")))
            (Path(d) / "pick_s1.npz").write_bytes(b"x")
            (Path(d) / "pick_s1.past.jsonl").write_text("{}\n")
            B.retire(f)
            names = sorted(p.name for p in Path(d).iterdir())
            self.assertEqual(names, ["_motorold_pick_s1.json", "_motorold_pick_s1.npz", "_motorold_pick_s1.past.jsonl"])
```

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_rsi_past_tools.py"`
Expected: FAIL — the `.past.jsonl` keeps its name.

- [ ] **Step 3: Implement**

`batch.py` `retire`: change the loop to

```python
    for p in (f, f.with_suffix(".npz"), f.with_name(f"{f.stem}.past.jsonl")):
```

`scene_round.py`: add near the other `locomani` imports

```python
from locomani.tasks.scene import past as PAST  # TODO 39: the per-step scene-graph records
```

and in `run_round`, right after the `rec.update(skills=h, ...)` block and before `tmp = f.with_suffix(".tmp")`:

```python
    try:  # TODO 39.3: the PAST records beside the round; a recorder bug never costs a round (like the classifier)
        records = PAST.build(rec, arrays, world.meta, task) if plan is not None and not plan_err else []
        PAST.write(out, label, records)
    except Exception as e:
        PAST.write(out, label, [PAST.error_line(label, f"{type(e).__name__}: {e}")])
```

Check that `arrays` is defined on every path in `run_round` (it is set to `None` before the `try` and assigned when `run_episode` runs); `world.meta` is the scene file dict (`SceneWorld.meta`); `task` is the `Task` object.

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_rsi_past_tools.py tests/test_scene_past.py"`
Expected: PASS.

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add scripts/rsi/scene_round.py scripts/rsi/batch.py tests/test_rsi_past_tools.py
git commit -m "scene_round: write <label>.past.jsonl beside each round; batch.retire keeps it with the round (TODO 39.3)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The 50-round invariance rerun of TODO 39.3 is Task 9 (gated).

---

### Task 6: Digest section from the records

**Files:**
- Modify: `scripts/rsi/digest.py` (new function `past_lines`, called in `build()` after `class_lines(rounds)`)
- Test: `tests/test_rsi_past_tools.py`

**Interfaces:**
- Consumes: `past.read` (Task 4).
- Produces: `past_lines(diag: Path, rounds: list[dict]) -> list[str]` — empty when no `.past.jsonl` exists; otherwise ≤ `PAST_MAX_LINES = 115` lines.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rsi_past_tools.py`:

```python
import digest as DG  # noqa: E402


def past_record(label, step, kind, disc, events=(), ok=False):
    return dict(schema="PAST-locomani-0.1", record_id=f"{label}/1.1", label=label, task=label.rsplit("_s", 1)[0],
                step_index=1, attempt=1, step=step, kind=kind, events=list(events), expected=[], discrepancies=disc,
                outcome=dict(step_ok=not disc, round_ok=ok, round_why="", failure_class=None))


class DigestSection(unittest.TestCase):
    def test_section_counts_first_unexpected_transition_per_task_and_lists_failed_timelines(self):
        with tempfile.TemporaryDirectory() as d:
            diag = Path(d)
            rounds = []
            for i, ok in enumerate([False, False, True]):
                label = f"pick_Mug_s{i}"
                rounds.append(dict(label=label, task="pick_Mug", ok=ok, steps=[]))
                (diag / f"{label}.json").write_text(json.dumps(rounds[-1]))
                disc = [] if ok else [dict(edge=["Mug", "held", "right_hand"], to=True, observed=False)]
                ev = [dict(t=9.1, edge=["Mug", "touching", "right_hand"], **{"from": False}, to=True, src="observed")]
                recs = [past_record(label, "pregrasp", "reach", [], ok=ok), past_record(label, "grip", "grip", disc, ev, ok=ok)]
                (diag / f"{label}.past.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            lines = DG.past_lines(diag, rounds)
        text = "\n".join(lines)
        self.assertIn("## Scene-graph transitions", text)
        self.assertIn("pick_Mug: 2 x grip: Mug held right_hand expected true, observed false", text)
        self.assertIn("successes: 1/1 true at the same kind", text)
        self.assertIn("pick_Mug_s0", text)
        self.assertIn("9.1  grip.grip  Mug touching right_hand false->true", text)
        self.assertLessEqual(len(lines), DG.PAST_MAX_LINES)

    def test_no_side_files_means_no_section(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(DG.past_lines(Path(d), [dict(label="a", task="t", ok=False)]), [])
```

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_rsi_past_tools.py -k DigestSection"`
Expected: FAIL — `AttributeError: module 'digest' has no attribute 'past_lines'`.

- [ ] **Step 3: Implement**

In `digest.py`, add after `RELEASE_STEPS`:

```python
PAST_MAX_LINES = 115  # TODO 39.4: the scene-graph section stays under a third of the 344-line digest of 2026-10-30
PAST_TIMELINE_MAX = 6  # flips shown per failed round before folding
```

Add the import `from locomani.tasks.scene import past as PAST  # noqa: E402` next to the other `locomani` imports, and the function:

```python
def _edge_text(edge: list) -> str:
    s, r, o = edge
    return f"{s} {r} {o}" if o is not None else f"{s} {r}"


def _first_unexpected(recs: list[dict]) -> tuple | None:
    for r in recs:
        if r.get("discrepancies"):
            d = r["discrepancies"][0]
            return r["kind"], tuple(d["edge"]), d.get("to"), d.get("observed")
    return None


def past_lines(diag: Path, rounds: list[dict]) -> list[str]:
    """TODO 39.4 (plan docs/handoff/past-scene-graph-plan-20261030.md section 4.8): per task, the first unexpected
    transition of the failed rounds with the successes' value at the same kind; per failed round, the flip timeline."""
    recs_by = {}
    for r in rounds:
        p = diag / f"{r['label']}.past.jsonl"
        if p.exists():
            recs = [x for x in PAST.read(p) if "record_id" in x]
            if recs:
                recs_by[r["label"]] = recs
    if not recs_by:
        return []
    out = ["## Scene-graph transitions (TODO 39; `past.py show <round>` for a record, `compare <fail> <ok>` for two)", ""]
    by_task = collections.defaultdict(list)
    for r in rounds:
        if r["label"] in recs_by:
            by_task[r["task"]].append(r)
    for task, rows in sorted(by_task.items()):
        firsts = collections.Counter()
        for r in rows:
            if not r["ok"]:
                f = _first_unexpected(recs_by[r["label"]])
                if f:
                    firsts[f] += 1
        for (kind, edge, to, obs), n in firsts.most_common(3):
            same = [x for r in rows if r["ok"] for x in recs_by[r["label"]] if x["kind"] == kind]
            vals = []
            for x in same:
                e = next((e for e in x["G_after"]["edges"] if tuple((e["s"], e["r"], e["o"])) == edge), None) if "G_after" in x else None
                if e is not None and e["v"] is not None:
                    vals.append(bool(e["v"]))
            tail = f"; successes: {sum(v == to for v in vals)}/{len(vals)} {str(to).lower()} at the same kind" if vals or same else ""
            out.append(f"- {task}: {n} x {kind}: {_edge_text(list(edge))} expected {str(to).lower()}, observed {str(obs).lower()}{tail}")
    out.append("")
    out.append("Failed rounds, flips from the first manipulation step (t  step.kind  edge from->to [unexpected *]):")
    for r in rounds:
        if r["ok"] or r["label"] not in recs_by:
            continue
        lines = []
        for x in recs_by[r["label"]]:
            if x["kind"] in ("none", "walk", "turn", "sidestep", "posture", "hold") and not x.get("discrepancies"):
                continue
            bad = {tuple(d["edge"]) for d in x.get("discrepancies") or []}
            for e in x.get("events") or []:
                mark = " *" if tuple(e["edge"]) in bad else ""
                lines.append(f"  {e['t']:.1f}  {x['step']}.{x['kind']}  {_edge_text(e['edge'])} {str(e['from']).lower()}->{str(e['to']).lower()}{mark}")
            for d in x.get("discrepancies") or []:
                lines.append(f"  {x['action']['t1'] if 'action' in x else '':>5}  {x['step']}.{x['kind']}  {_edge_text(d['edge'])} expected {str(d['to']).lower()} *")
        if len(lines) > PAST_TIMELINE_MAX:
            lines = lines[:PAST_TIMELINE_MAX] + [f"  ... {len(lines) - PAST_TIMELINE_MAX} more: past.py show {diag / r['label']}"]
        out += [f"- {r['label']}"] + lines
        if len(out) >= PAST_MAX_LINES - 2:
            out.append(f"- ... more failed rounds; `past.py show`")
            break
    out.append("")
    return out[:PAST_MAX_LINES]
```

In `build()`, after `out += class_lines(rounds)` add `out += past_lines(diag, rounds)`.

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_rsi_past_tools.py"`
Expected: PASS. Adjust the exact strings in the test to the implementation if a space differs — but keep the asserted content (task line with counts and the successes' tail; the timeline row).

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add scripts/rsi/digest.py tests/test_rsi_past_tools.py
git commit -m "digest: scene-graph transitions section from the PAST records (TODO 39.4)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `scripts/rsi/past.py` — `show` and `compare`

**Files:**
- Create: `scripts/rsi/past.py`
- Test: `tests/test_rsi_past_tools.py`

**Interfaces:**
- Consumes: `past.read`.
- Produces: `show(path_or_prefix: Path, step: int | None, as_json: bool) -> str`, `compare(a: Path, b: Path) -> str`, `main(argv) -> int`. A path may be `<dir>/<label>` (no suffix), `<label>.json` or `<label>.past.jsonl`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rsi_past_tools.py`:

```python
import past_cli  # noqa: E402  (scripts/rsi/past.py imported under this name below)
```

Because `scripts/rsi/past.py` would shadow `locomani.tasks.scene.past` only by module name inside `scripts/rsi` (both are importable; the script imports the package one as `PAST`), import it in the test with:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("past_cli", REPO / "scripts" / "rsi" / "past.py")
past_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(past_cli)
```

and the tests:

```python
class Cli(unittest.TestCase):
    def records(self, label, ok):
        g0 = dict(t=9.0, states={}, edges=[dict(s="Mug", r="held", o="right_hand", v=False, src="predicate"),
                                            dict(s="Mug", r="supported_by", o="T", v=True, src="derived")])
        g1 = dict(t=9.6, states={}, edges=[dict(s="Mug", r="held", o="right_hand", v=ok, src="predicate"),
                                            dict(s="Mug", r="supported_by", o="T", v=True, src="derived")])
        ev = [dict(t=9.2, edge=["Mug", "touching", "right_hand"], **{"from": False}, to=True, src="observed")]
        disc = [] if ok else [dict(edge=["Mug", "held", "right_hand"], to=True, observed=False)]
        return [dict(schema="PAST-locomani-0.1", record_id=f"{label}/0.1", label=label, task="pick_Mug", step_index=0,
                     attempt=1, step="pregrasp", kind="reach", action=dict(t0=8.0, t1=9.0, ok=True, why="", calls=[], check={}),
                     roles=dict(target="Mug", effector="right_hand"), scope=dict(initial=["Mug"], added=[]), entities={},
                     G_before=g0, events=[], G_after=g0, expected=[], discrepancies=[],
                     outcome=dict(step_ok=True, round_ok=ok), evidence=dict(npz=f"{label}.npz", k0=400, k1=450)),
                dict(schema="PAST-locomani-0.1", record_id=f"{label}/1.1", label=label, task="pick_Mug", step_index=1,
                     attempt=1, step="grip", kind="grip", action=dict(t0=9.0, t1=9.6, ok=ok, why="", calls=[], check={"opening_cm": 6.1}),
                     roles=dict(target="Mug", effector="right_hand"), scope=dict(initial=["Mug"], added=[]), entities={},
                     G_before=g0, events=ev, G_after=g1, expected=[dict(edge=["Mug", "held", "right_hand"], to=True)],
                     discrepancies=disc, outcome=dict(step_ok=ok, round_ok=ok), evidence=dict(npz=f"{label}.npz", k0=450, k1=480))]

    def write(self, d, label, ok):
        (Path(d) / f"{label}.past.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records(label, ok)))

    def test_show_prints_steps_edges_events_and_expectations(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "pick_Mug_s0", False)
            text = past_cli.show(Path(d) / "pick_Mug_s0", step=None, as_json=False)
        self.assertIn("1.1 grip [grip] 9.0-9.6 s FAIL", text)
        self.assertIn("Mug —held→ right_hand  false→false  (expected true ✗)", text)
        self.assertIn("9.2  Mug —touching→ right_hand  false→true", text)
        self.assertIn("Mug —supported_by→ T  true→true", text)

    def test_show_one_step_as_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "pick_Mug_s0", False)
            text = past_cli.show(Path(d) / "pick_Mug_s0.past.jsonl", step=1, as_json=True)
        self.assertEqual(json.loads(text)["record_id"], "pick_Mug_s0/1.1")

    def test_compare_aligns_by_kind_and_marks_differences(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "pick_Mug_s0", False)
            self.write(d, "pick_Mug_s1", True)
            text = past_cli.compare(Path(d) / "pick_Mug_s0", Path(d) / "pick_Mug_s1")
        self.assertIn("grip | grip", text)
        self.assertIn("Mug —held→ right_hand  A false  B true  <-- differs", text)

    def test_main_show(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "pick_Mug_s0", False)
            self.assertEqual(past_cli.main(["show", str(Path(d) / "pick_Mug_s0")]), 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `make test TESTS="tests/test_rsi_past_tools.py -k Cli"`
Expected: FAIL — file not found.

- [ ] **Step 3: Implement**

Create `scripts/rsi/past.py`:

```python
"""TODO 39.4 (plan docs/handoff/past-scene-graph-plan-20261030.md section 4.9): read a round's PAST records.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/past.py show <diag dir>/<label> [--step N] [--json]
    PYTHONPATH=src $PY scripts/rsi/past.py compare <failed round> <successful round>

`show` prints every step: G_before -> events -> G_after, one edge per line with its expectation; `compare` aligns the
two rounds' steps by kind (longest common subsequence) and prints the edges whose values differ. Reading only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani.tasks.scene import past as PAST  # noqa: E402


def side_file(p: Path) -> Path:
    p = Path(p)
    if p.name.endswith(".past.jsonl"):
        return p
    if p.suffix == ".json":
        return p.with_name(p.stem + ".past.jsonl")
    return p.with_name(p.name + ".past.jsonl")


def _b(v) -> str:
    return "unknown" if v is None else (str(v).lower() if isinstance(v, bool) else json.dumps(v))


def _edge(e: dict) -> str:
    return f"{e['s']} —{e['r']}→ {e['o']}" if e.get("o") is not None else f"{e['s']} —{e['r']}"


def _key(e: dict) -> tuple:
    return (e["s"], e["r"], e["o"])


def record_lines(r: dict) -> list[str]:
    a = r.get("action") or {}
    head = f"{r['step_index']}.{r['attempt']} {r['step']} [{r['kind']}] {a.get('t0', 0):.1f}-{a.get('t1', 0):.1f} s {'OK' if a.get('ok') else 'FAIL'}"
    if a.get("why"):
        head += f"  {a['why']}"
    out = [head]
    roles = {k: v for k, v in (r.get("roles") or {}).items() if v}
    out.append("  roles " + ", ".join(f"{k}={v}" for k, v in roles.items()))
    if a.get("check"):
        out.append("  check " + json.dumps(a["check"]))
    exp = {tuple(x["edge"]): x for x in r.get("expected") or []}
    before = {_key(e): e for e in r["G_before"]["edges"]}
    after = {_key(e): e for e in r["G_after"]["edges"]}
    bad = {tuple(d["edge"]) for d in r.get("discrepancies") or []}
    for k in sorted(set(before) | set(after), key=lambda k: (k[1], k[0], str(k[2]))):
        b, f = before.get(k), after.get(k)
        if b is not None and f is not None and not isinstance(b["v"], bool) and b["v"] is not None:
            nums = ", ".join(f"{kk} {vv}" for kk, vv in f.items() if kk not in ("s", "r", "o", "v", "src"))
            out.append(f"  {_edge(f)}  {_b(b['v'])}→{_b(f['v'])}" + (f"  ({nums})" if nums else ""))
            continue
        e = f or b
        line = f"  {_edge(e)}  {_b(b['v'] if b else None)}→{_b(f['v'] if f else None)}"
        if k in exp:
            line += f"  (expected {_b(exp[k]['to'])} {'✗' if k in bad else '✓'})"
        out.append(line)
    for d in r.get("discrepancies") or []:
        if tuple(d["edge"]) not in before and tuple(d["edge"]) not in after:
            out.append(f"  {d['edge'][0]} —{d['edge'][1]}→ {d['edge'][2]}  expected {_b(d['to'])}, observed {_b(d.get('observed'))} ✗")
    if r.get("events"):
        out.append("  events:")
        for e in r["events"]:
            extra = "".join(f"  {k} {v}" for k, v in e.items() if k not in ("t", "edge", "from", "to", "src"))
            out.append(f"    {e['t']:.1f}  {e['edge'][0]} —{e['edge'][1]}→ {e['edge'][2]}  {_b(e['from'])}→{_b(e['to'])}{extra}")
    added = (r.get("scope") or {}).get("added") or []
    if added:
        out.append("  joined: " + ", ".join(f"{x['id']} at {x['t']:.1f} s ({x['why']})" for x in added))
    ev = r.get("evidence") or {}
    out.append(f"  evidence {ev.get('npz')} ticks {ev.get('k0')}-{ev.get('k1')}")
    return out


def show(path: Path, step: int | None = None, as_json: bool = False) -> str:
    recs = [r for r in PAST.read(side_file(path)) if "record_id" in r]
    if step is not None:
        recs = [r for r in recs if r["step_index"] == step]
    if as_json:
        return "\n".join(json.dumps(r, indent=1) for r in recs)
    if not recs:
        return "no records"
    out = [f"{recs[0]['label']}  task {recs[0]['task']}  round {'OK' if recs[0]['outcome'].get('round_ok') else 'FAIL'}"]
    for r in recs:
        out += [""] + record_lines(r)
    return "\n".join(out)


def _lcs(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append((i, None))
            i += 1
        else:
            out.append((None, j))
            j += 1
    out += [(k, None) for k in range(i, n)] + [(None, k) for k in range(j, m)]
    return out


def compare(a: Path, b: Path) -> str:
    ra = [r for r in PAST.read(side_file(a)) if "record_id" in r]
    rb = [r for r in PAST.read(side_file(b)) if "record_id" in r]
    out = [f"A {ra[0]['label'] if ra else a}  B {rb[0]['label'] if rb else b}  (aligned by kind)"]
    for i, j in _lcs([r["kind"] for r in ra], [r["kind"] for r in rb]):
        left = f"{ra[i]['step']} [{ra[i]['kind']}]" if i is not None else "-"
        right = f"{rb[j]['step']} [{rb[j]['kind']}]" if j is not None else "-"
        out.append(f"\n{ra[i]['kind'] if i is not None else rb[j]['kind']} | {rb[j]['kind'] if j is not None else '-'}   A {left}   B {right}")
        if i is None or j is None:
            continue
        ea = {_key(e): e for e in ra[i]["G_after"]["edges"]}
        eb = {_key(e): e for e in rb[j]["G_after"]["edges"]}
        for k in sorted(set(ea) | set(eb), key=lambda k: (k[1], k[0], str(k[2]))):
            va, vb = (ea.get(k) or {}).get("v"), (eb.get(k) or {}).get("v")
            if va != vb:
                out.append(f"  {ea.get(k, eb.get(k))['s']} —{k[1]}→ {k[2]}  A {_b(va)}  B {_b(vb)}  <-- differs")
        fa = [(e["edge"][1], e["to"]) for e in ra[i].get("events") or []]
        fb = [(e["edge"][1], e["to"]) for e in rb[j].get("events") or []]
        if fa != fb:
            out.append(f"  events A {fa}  B {fb}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("round", type=Path)
    s.add_argument("--step", type=int)
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    print(show(a.round, a.step, a.json) if a.cmd == "show" else compare(a.a, a.b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run: `make test TESTS="tests/test_rsi_past_tools.py"`
Expected: PASS. The exact strings in the tests must match the implementation's formatting; adjust the implementation (not the asserted content) if an arrow or spacing differs.

- [ ] **Step 5: `make check`, commit**

```bash
make check
git add scripts/rsi/past.py tests/test_rsi_past_tools.py
git commit -m "past.py: show and compare for a round's scene-graph records (TODO 39.4)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Prompt, rules, docs

**Files:**
- Modify: `scripts/rsi/prompts/reviewer.md` (item 4 and the "Data you may read" list)
- Modify: `scripts/rsi/rules.md` (title line: version 22 → 23; append rule 38)
- Modify: `docs/rsi_design.md` (§5 Reviewer paragraph; §8 version list)
- Modify: `docs/rsi.md` (a command line next to the `records.py` line, ~line 119)
- Modify: `TODO.md` (tick 39.1–39.4), `docs/handoff/past-scene-graph-plan-20261030.md` §7 log

No code; verified by `make check` (doc lint) and by reading.

- [ ] **Step 1: `reviewer.md`**

In item 4 (the `{DIGEST}` paragraph), replace the final clause `Start from it rather than re-deriving these views; open a round's .json / .npz for what it does not show.` with:

```
Start from it rather than re-deriving these views. For a round, read next its scene-graph records `{DIAG_INC}/<task>_s<seed>.past.jsonl` (`PYTHONPATH=src {SIM_PY} scripts/rsi/past.py show {DIAG_INC}/<label>`, `compare <failed> <successful>`; rules.md 38): one record per skill step and retry with the entities around the action (target, hands, support, goal surface or container, neighbours within 0.30 m, furniture in the stance footprint or the approach corridor, surfaces above the target), their states (pose, velocity, jaw, arm joints, pelvis, feet), the relations before and after (touching, supported_by, held, above, beside, near_edge, blocking, dist, reach_vec, approach, and the goal predicates), the flips in between with their times, and the transitions the primitive is for (`expected`) against what happened (`discrepancies`). `expected` is a fixed table by primitive, not a judgement; `src` says whether a value was observed, a predicate, derived geometry, or `unknown` (an old round without the contact channel). Open the .npz last, for what the records do not show (the evidence block gives the tick range).
```

- [ ] **Step 2: `rules.md`**

Title line: `# Rules for the scene RSI loop (TODO 27, version 23, 2026-10-30)`. Append after rule 37:

```
38. Scene-graph records (TODO 39, version 23; plan docs/handoff/past-scene-graph-plan-20261030.md). The runner writes
    `<label>.past.jsonl` beside every round: one record per skill step and retry, built after the round from the step
    log, the primitive calls, the trace with its contact channel and the scene file (`locomani.tasks.scene.past`). It is
    a measuring instrument: the skills package may not import it, read a `.past.jsonl`, or call `scripts/rsi/past.py`
    -- a draft whose skills code names `past.jsonl`, `scene.past` or `past.py` is void, like rule 30's object names
    (code: loop.py's draft check). The records never enter the gate, the sentinel or the regression; they are data for
    the Reviewer (prompt: digest section "Scene-graph transitions", `past.py show` / `compare`). Their `expected`
    transitions are a fixed table by primitive (walk lands and nothing blocks the stance; reach lands within 3 cm,
    the target and its neighbours do not move, nothing blocks the corridor; grip -> held; a rising reach while holding
    -> off its support; release -> not held and, when the goal names it, on the goal surface or in the container), not
    evidence that a change should stay (rule 7).
```

Then add the draft check it names: in `scripts/rsi/loop.py`, find the rule-30 object-name check of a draft (search `rule 30` / `object names` / the function that voids a draft for naming scene objects) and extend its forbidden-token list with `("past.jsonl", "scene.past", "past.py")` for files under the skills package, reusing the same void path and ledger reason (`"names the recorder (rule 38)"`). Add one unit test next to the existing rule-30 test (search `tests/` for the rule-30 test; add a sibling asserting a draft containing `from locomani.tasks.scene import past` is void). If no such check function exists in `loop.py`, write the rule as **prompt**-enforced instead (`(prompt)` in place of `(code: ...)`) and skip the code change — say so in the commit message.

- [ ] **Step 3: `docs/rsi_design.md`**

In §5's Reviewer item, after the sentence ending `一轮 79 分钟里约 25% 是在等这些。`, append:

```
每局另有一份场景图记录 `<label>.past.jsonl`（TODO 39，[方案](../../handoff/past-scene-graph-plan-20261030.md)）：每个技能步骤、每次重试一条，记动作周围的实体（目标、两手、支撑面、目标台面或容器、0.30 m 内的邻居、站位足迹和接近走廊里的家具、目标上方的面）、它们的状态、前后的关系（接触、支撑、持握、空间、相对位姿、谓词）、中间的翻转和时间、按原语的期待转移与实际的差异；由冻结层在局结束后从步骤日志、原语调用、带接触通道的轨迹和场景文件构造，技能不能读它（规则第 38 条）。摘要多一节"Scene-graph transitions"，`scripts/rsi/past.py show` / `compare` 看单局和对照两局；读法是摘要 → 记录 → NPZ 最后手段。
```

(When pasting into `docs/rsi_design.md`, the link target is `handoff/past-scene-graph-plan-20261030.md` — relative to `docs/`; the `../../` form above only keeps this plan file's own link check green.)

In §8's version list, after `第 22 版（10-26，合并 main 时改号）是严格判据 \`lifted\` 和判定窗口（TODO 38.19，第 37 条）` add `，第 23 版（10-30）是场景图记录（TODO 39，第 38 条）`.

- [ ] **Step 4: `docs/rsi.md`**

After the `records.py` bullet (line ~119) add:

```
- `PYTHONPATH=src $PY scripts/rsi/past.py show <diag 目录>/<label>` 和 `past.py compare <失败局> <成功局>`：一局的场景图记录（每个步骤、每次重试一条：实体、前后关系、翻转事件、期待与差异），和按步骤种类对齐两局的差异。记录由运行器写在每局的 `.json` 旁边（`<label>.past.jsonl`，TODO 39）；摘要里的"Scene-graph transitions"一节从它来。
```

- [ ] **Step 5: TODO and spec log**

Tick `39.1`–`39.4` in `TODO.md` (`- [x]`), appending to each a short note of what was done (file names, test names). In the spec §7 log add `- <date>: 39.1–39.4 代码和测试提交（<hashes>）；50 局不变性验证和新运行待用户批准。`

- [ ] **Step 6: Lint and commit**

```bash
make check
git add scripts/rsi/prompts/reviewer.md scripts/rsi/rules.md scripts/rsi/loop.py tests/<rule-38 test file> docs/rsi_design.md docs/rsi.md TODO.md docs/handoff/past-scene-graph-plan-20261030.md
git commit -m "RSI rules v23: scene-graph records are the Reviewer's data, never the skills' (rule 38); prompt, design and runbook pointers (TODO 39.4)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(Drop `scripts/rsi/loop.py` and the test from `git add` if Step 2 ended with a prompt-enforced rule.)

---

### Task 9 (GATED — ask the user first): 50-round invariance rerun

**Do not start this task without the user's explicit go-ahead in this session** (user 2026-10-30: 开始正式测试前先问我). It runs the simulator for about 4 minutes on 12 processes / 2 GPUs.

**Files:**
- No code changes; results into `docs/handoff/past-scene-graph-plan-20261030.md` §7 and `TODO.md` 39.3.

- [ ] **Step 1: Ask the user**, naming: sandbox `pick-repro`, the incumbent to rerun, the output directory (`/home/henry.zhao/workspace/tmp/past_invariance_<date>/`), the command, GPU use.

- [ ] **Step 2: Find the incumbent** — in `~/workspace/code/rsi-scene/pick-repro/outputs/scene_rsi_20261016/pick_single/`, the latest `diag/incumbent_<hash>/` whose hash matches the last accepted gate row in `loop/ledger.jsonl` (or `incumbents/<hash>/` for its skills package and `set`).

- [ ] **Step 3: Rerun** from the **main checkout** (it has the contact channel; the sandbox does not), with the sandbox's skills package and overrides:

```bash
cd /home/henry.zhao/workspace/locomanipulation
export LOCOMANI_WORKSPACE=/home/henry.zhao/workspace
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
SB=/home/henry.zhao/workspace/code/rsi-scene/pick-repro/outputs/scene_rsi_20261016/pick_single
OUT=/home/henry.zhao/workspace/tmp/past_invariance_$(date +%Y%m%d)
PYTHONPATH=src OMP_NUM_THREADS=2 $PY scripts/rsi/batch.py pick_single $SB/pool_diag <incumbent skills dir> $OUT --set '<incumbent set JSON>' --procs 12 --gpus 0,1 --npz
```

(Check `batch.py main`'s exact argument order with `--help` first; respect `RSI_GPUS` / `RSI_PROCS` if this machine is single-GPU per `docs/rsi.md` §0.)

- [ ] **Step 4: Compare** every `<label>.json` in `$OUT` with `$SB/diag/incumbent_<hash>/` on `ok, why, fell, dropped, t_end` (a short Python one-liner in the scratchpad, not in the repo), and the NPZ `robot` / `obj_*` max abs difference; confirm `$OUT/<label>.past.jsonl` exists for every round and has no `error` lines.

- [ ] **Step 5: Record** the result (50/50 identical or the list of differing rounds with their fields) in the spec §7 log and TODO 39.3; commit the two docs. If any round differs, stop and report — the contact channel is not allowed to change outcomes.

---

## Self-review (done while writing)

- **Spec coverage**: §4.2 → Task 1; §4.3–4.6 → Tasks 2–4; §4.7 → Task 5; §4.8 → Task 6; §4.9 → Task 7; §4.10 → Task 8; §5 unit tests → Tasks 1–7, invariance → Task 9 (gated); §6 new run and `past_monitor.py` → TODO 39.5, deliberately outside this plan (needs the user's go-ahead and belongs to a later plan).
- **Known simplifications** called out inline: the wrist attitude is `"level"` (the trace has wrist positions only); `blocking` for furniture uses `init_pos` + box centre (static furniture does not move). Both are in the spec's limitations list or added to §7's log when implemented.
- **Type consistency**: `roles_of(kind, calls, task_objects, *, held_before, positions)`; `scope_of(frame, roles, kind, task, meta, calls) -> (entities, scope)`; `graph_of(frame, entities, roles, meta)`; `expected_of(kind, roles, task, g_before, *, rising)`; `events_of(arrays, meta, k0, k1, entities, roles, task)`; `discrepancies_of(expected, g_after, events)`; `build(rec, arrays, meta, task)`; `write(out_dir, label, records)`; `read(path)`; `error_line(label, err)`; digest `past_lines(diag, rounds)`; CLI `show(path, step, as_json)`, `compare(a, b)`.
