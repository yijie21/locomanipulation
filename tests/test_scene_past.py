"""TODO 39: the PAST recorder's builder (src/locomani/tasks/scene/past.py) on hand-made rounds, no simulation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomani.tasks.scene import past as P


def step(name, t0, t1, ok=True, **extra):
    return dict(step=name, ok=ok, why="" if ok else "x", t0=t0, t1=t1, **extra)


def call(name, t0, t1, *args, ok=True, **kwargs):
    return dict(name=name, t0=t0, t1=t1, args=list(args), kwargs=kwargs, ok=ok, why="")


class Intervals(unittest.TestCase):
    def test_steps_become_intervals_and_retries_keep_their_attempt(self):
        steps = [step("settle", 0.0, 1.0), step("walk", 1.0, 5.0), step("pregrasp", 5.0, 7.0, ok=False, attempt=1),
                 step("pregrasp", 7.0, 9.0, attempt=2), step("grip", 9.0, 10.0)]  # fmt: skip
        iv = P.intervals(steps)
        self.assertEqual([(i.step, i.attempt) for i in iv],
                         [("settle", 1), ("walk", 1), ("pregrasp", 1), ("pregrasp", 2), ("grip", 1)])  # fmt: skip
        self.assertEqual([i.index for i in iv], [0, 1, 2, 3, 4])

    def test_calls_belong_to_the_interval_holding_their_start(self):
        calls = [call("walk_to", 1.0, 4.8, [1.0, 0.0]), call("reach", 5.1, 6.9, "right", [1.0, 0.1, 0.9]),
                 call("grip", 9.0, 9.8, "right")]  # fmt: skip
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
                       positions={"Mug": [1.0, 0.05, 0.9], "Can": [2.0, 0.0, 0.9]})  # fmt: skip
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
                       positions={"Mug": [0, 0, 0]})  # fmt: skip
        self.assertIsNone(r["effector"])
        self.assertEqual(r["target"], "Mug")


# -- a synthetic round: arrays as Motor.arrays() writes them, a scene file with a table, a shelf and three objects -----
BODIES = ["world", "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_dex1_finger_1_link",
          "left_dex1_finger_2_link", "right_dex1_finger_1_link", "right_dex1_finger_2_link", "Mug", "Can", "Far", "Table"]  # fmt: skip


def arrays_for(n, objects, hands, contacts=(), bodies=BODIES, dt=0.02):
    """A trace of n ticks. objects: {body: (n, 7) poses}; hands: {side: (n, 3) wrist positions}; contacts: [(k, a, b)]
    with body names; bodies: the body-name table (index = id)."""
    out = dict(t=np.arange(n) * dt, mode=np.full(n, 2), robot=np.tile([0, 0, 0.79, 1, 0, 0, 0], (n, 1)).astype(float),
               ref_pelvis=np.zeros((n, 3)), ref_wrists=np.zeros((n, 2, 3)),
               wrists=np.stack([np.asarray(hands["left"], float), np.asarray(hands["right"], float)], axis=1),
               jaw=np.full((n, 2), 0.0245), fingers=np.zeros((n, 4)), joints=np.zeros((n, 29)),
               primitive=np.zeros(n, dtype=np.int16), primitives=np.array(["reach"]), bodies=np.array(bodies))  # fmt: skip
    for name, poses in objects.items():
        out[f"obj_{name}"] = np.asarray(poses, dtype=float)
    ids = {b: i for i, b in enumerate(bodies)}
    out["contact_k"] = np.array([k for k, _, _ in contacts], dtype=np.int32)
    out["contact_a"] = np.array([min(ids[a], ids[b]) for _, a, b in contacts], dtype=np.int32)
    out["contact_b"] = np.array([max(ids[a], ids[b]) for _, a, b in contacts], dtype=np.int32)
    return out


def no_contact_channel(arr):
    for k in ("contact_k", "contact_a", "contact_b", "bodies"):
        del arr[k]
    return arr


def obj(half, mass, body, surface="T"):
    return {"surface": surface, "free": True, "body": body, "container": False, "init_pos": [0, 0, 0],
            "box": {"half_extent": list(half), "centre": [0, 0, 0]},
            "semantic": {"mass_kg": mass, "movable": True, "upright_axis": [0, 0, 1]}}  # fmt: skip


META = {
    "objects": {
        "Mug": obj([0.04, 0.04, 0.05], 0.3, "Mug"),
        "Can": obj([0.03, 0.03, 0.06], 0.4, "Can"),
        "Far": obj([0.03, 0.03, 0.03], 0.1, "Far"),
        "Table": {"free": False, "body": None, "container": False, "init_pos": [0, 0, 0],
                  "semantic": {"movable": False}, "box": {"half_extent": [0.6, 0.4, 0.35], "centre": [1.0, 0.0, 0.35]}},
    },
    "surfaces": {
        "T": {"xy": [1.0, 0.0], "top": 0.70, "half_extent": [0.6, 0.4], "yaw": 0.0, "of_object": "Table",
              "approach_edges": ["-x"]},
        "Shelf": {"xy": [1.0, 0.0], "top": 1.05, "half_extent": [0.6, 0.4], "yaw": 0.0, "of_object": "Table"},
    },
}  # fmt: skip
META_T = json.loads(json.dumps(META))
META_T["objects"]["Table"]["body"] = "Table"
META_T["surfaces"].pop(
    "Shelf"
)  # any surface whose footprint holds the mug makes held() "resting" (predicates.RESTING_GAP_M)
META_G = json.loads(json.dumps(META))  # the graph test keeps the shelf and gives the table a body
META_G["objects"]["Table"]["body"] = "Table"
TASK = dict(id="pick_Mug", objects=["Mug"], goal=["held(Mug)"])


def still(p, n):
    return np.tile([*p, 1, 0, 0, 0], (n, 1)).astype(float)


def mug_round(n=20):
    """The right hand approaches the mug; nothing moves."""
    objects = {
        "Mug": still([1.0, 0.0, 0.75], n),
        "Can": still([1.0, 0.10, 0.76], n),
        "Far": still([1.0, 0.39, 0.73], n),
    }
    hands = {"left": np.tile([0.2, 0.2, 0.9], (n, 1)), "right": np.linspace([0.3, -0.1, 0.9], [0.85, 0.0, 0.78], n)}
    return objects, hands


REACH = [call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78])]
ROLES = dict(target="Mug", effector="right_hand", carried=None)


class Frames(unittest.TestCase):
    def test_frame_reads_poses_contacts_and_touching(self):
        objects, hands = mug_round()
        arr = arrays_for(20, objects, hands, contacts=[(19, "Mug", "right_dex1_finger_1_link"),
                                                       (19, "Mug", "right_dex1_finger_2_link")])  # fmt: skip
        f = P.Frame(arr, META, 19)
        self.assertTrue(f.has_contacts)
        np.testing.assert_allclose(f.obj_pose("Mug")[0], [1.0, 0.0, 0.75])
        self.assertEqual(f.touching("Mug"), {"right"})
        self.assertEqual(f.gripped("Mug"), {"right"})
        self.assertEqual(f.touching("Can"), set())
        self.assertEqual(f.state(["Mug"])["objects"]["Mug"]["gripped"], ["right"])

    def test_frame_without_contact_channel_answers_unknown(self):
        objects, hands = mug_round()
        f = P.Frame(no_contact_channel(arrays_for(20, objects, hands)), META, 0)
        self.assertFalse(f.has_contacts)
        self.assertEqual(f.touching("Mug"), set())
        self.assertIsNone(f.touches("Mug", "Table"))


class Scope(unittest.TestCase):
    def test_reach_scope_has_roles_neighbour_overhang_and_regions_not_the_far_object(self):
        objects, hands = mug_round()
        f = P.Frame(arrays_for(20, objects, hands), META, 0)
        entities, scope = P.scope_of(f, ROLES, "reach", TASK, META, REACH)
        ids = set(scope["initial"])
        for want in ("Mug", "right_hand", "left_hand", "robot", "T", "Table", "Can", "Shelf", "region:corridor",
                     "region:pregrasp_target"):  # fmt: skip
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
        entities, scope = P.scope_of(f, dict(target="Mug", effector=None, carried=None), "walk", TASK, META,
                                     [call("walk_to", 0.0, 0.38, [0.4, 0.0])])  # fmt: skip
        self.assertIn("region:stance", scope["initial"])
        self.assertNotIn("right_hand", scope["initial"])
        self.assertEqual(entities["region:stance"]["radius_m"], 0.35)


class Graph(unittest.TestCase):
    @staticmethod
    def edges(g, r, s=None):
        return [e for e in g["edges"] if e["r"] == r and (s is None or e["s"] == s)]

    def test_graph_edges_before_the_grasp(self):
        objects, hands = mug_round()
        f = P.Frame(arrays_for(20, objects, hands, contacts=[(0, "Mug", "Table")]), META_G, 0)
        entities, _ = P.scope_of(f, ROLES, "reach", TASK, META_G, REACH)
        g = P.graph_of(f, entities, ROLES, META_G)
        self.assertEqual([(e["o"], e["v"], e["src"]) for e in self.edges(g, "supported_by", "Mug") if e["o"] == "T"],
                         [("T", True, "derived")])  # fmt: skip
        held = [e for e in self.edges(g, "held", "Mug") if e["o"] == "right_hand"][0]
        self.assertEqual((held["v"], held["src"]), (False, "predicate"))
        above = self.edges(g, "above", "Shelf")[0]
        self.assertTrue(above["v"])
        self.assertAlmostEqual(above["gap_m"], 1.05 - 0.80, places=3)
        beside = self.edges(g, "beside", "Can")[0]
        self.assertAlmostEqual(beside["gap_m"], 0.10 - 0.04 - 0.03, places=3)
        self.assertEqual(beside["side"], "+y")
        edge = self.edges(g, "near_edge", "Mug")[0]
        self.assertIn(edge["edge"], ("+y", "-y"))  # the mug sits at y 0 on a 0.4 m half-width table: a tie
        self.assertAlmostEqual(edge["dist_m"], 0.4, places=3)
        self.assertEqual(len(self.edges(g, "reach_vec")[0]["v"]), 3)
        self.assertEqual(g["states"]["right_hand"]["att"], "level")
        self.assertIn("v_mps", g["states"]["Mug"])
        self.assertEqual(g["states"]["robot"]["pelvis_z_m"], 0.79)
        self.assertTrue(any(e["o"] == "region:corridor" for e in self.edges(g, "blocking")))

    def test_unknown_contacts_make_support_approx_and_touching_unknown(self):
        objects, hands = mug_round()
        f = P.Frame(no_contact_channel(arrays_for(20, objects, hands)), META, 0)
        entities, _ = P.scope_of(f, ROLES, "reach", TASK, META, REACH)
        g = P.graph_of(f, entities, ROLES, META)
        touching = [e for e in self.edges(g, "touching", "Mug") if e["o"] == "right_hand"][0]
        self.assertEqual((touching["v"], touching["src"]), (None, "unknown"))
        sup = [e for e in self.edges(g, "supported_by", "Mug") if e["o"] == "T"][0]
        self.assertEqual((sup["v"], sup["src"]), (True, "approx"))


# -- a whole round: reach (ticks 0-19), grip (20-29), lift (30-39) -------------------------------------------------------
def grasp_round_arrays(n=40, close_both=True):
    """Finger 1 touches the mug at tick 22, finger 2 (when `close_both`) at 24; the mug leaves the table at 32."""
    mug = np.vstack([still([1.0, 0.0, 0.75], 32),
                     np.hstack([np.tile([1.0, 0.0], (8, 1)), np.linspace(0.75, 0.85, 8)[:, None], np.tile([1, 0, 0, 0], (8, 1))])])  # fmt: skip
    right = np.vstack([np.linspace([0.3, -0.1, 0.9], [0.85, 0.0, 0.78], 20), np.tile([0.85, 0.0, 0.78], (12, 1)),
                       np.hstack([np.tile([0.85, 0.0], (8, 1)), np.linspace(0.78, 0.88, 8)[:, None]])])  # fmt: skip
    contacts = [(k, "Mug", "Table") for k in range(0, 32)]
    contacts += [(k, "Mug", "right_dex1_finger_1_link") for k in range(22, n)]
    if close_both:
        contacts += [(k, "Mug", "right_dex1_finger_2_link") for k in range(24, n)]
    objects = {"Mug": mug, "Can": still([1.0, 0.10, 0.76], n), "Far": still([1.0, 0.39, 0.73], n)}
    return arrays_for(n, objects, {"left": np.tile([0.2, 0.2, 0.9], (n, 1)), "right": right}, contacts=contacts)


def grasp_round_rec(ok=True):
    return dict(label="pick_Mug_s1", task="pick_Mug", ok=ok, why="" if ok else "[goal] held(Mug): no hand touches it",
                fell=False, dropped=False, plan_ok=ok, plan_why="", error="", skills="abc123", set={"X": 1},
                failure_class="" if ok else "slip", events={},
                steps=[step("settle", 0.0, 0.0), step("pregrasp", 0.0, 0.40),
                       step("grip", 0.40, 0.60, ok=ok, check={"opening_cm": 6.1}, attempt=1), step("lift", 0.60, 0.78)],
                calls=[call("reach", 0.0, 0.38, "right", [0.85, 0.0, 0.78]), call("grip", 0.40, 0.58, "right"),
                       call("reach", 0.60, 0.78, "right", [0.85, 0.0, 0.88])])  # fmt: skip


class Expected(unittest.TestCase):
    def test_grip_expects_gripped_and_reach_expects_landing_and_no_blocking(self):
        ex = P.expected_of("grip", ROLES, TASK, dict(edges=[]))
        self.assertIn({"edge": ["Mug", "gripped", "right_hand"], "to": True}, ex)
        ex = P.expected_of("reach", ROLES, TASK, dict(edges=[]))
        self.assertIn({"edge": ["right_hand", "at", "region:pregrasp_target"], "to": True, "tol_cm": 3.0}, ex)
        self.assertIn({"edge": ["*", "blocking", "region:corridor"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "moved", None], "to": False}, ex)

    def test_rising_reach_while_holding_expects_held_and_support_to_drop(self):
        roles = dict(target="Mug", effector="right_hand", carried="Mug")
        g = dict(edges=[dict(s="Mug", r="supported_by", o="T", v=True, src="derived")])
        ex = P.expected_of("reach", roles, TASK, g, rising=True)
        self.assertIn({"edge": ["Mug", "supported_by", "T"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "held", "right_hand"], "to": True}, ex)

    def test_release_toward_the_goal_surface(self):
        task = dict(id="put", objects=["Mug"], goal=["on(Mug, T)"])
        roles = dict(target="Mug", effector="right_hand", carried="Mug")
        ex = P.expected_of("release", roles, task, dict(edges=[]))
        self.assertIn({"edge": ["Mug", "gripped", "right_hand"], "to": False}, ex)
        self.assertIn({"edge": ["Mug", "supported_by", "T"], "to": True}, ex)

    def test_walk(self):
        ex = P.expected_of("walk", dict(target="Mug", effector=None, carried=None), TASK, dict(edges=[]))
        self.assertEqual([e["edge"][1] for e in ex], ["at", "blocking"])


class Build(unittest.TestCase):
    def test_one_record_per_step_with_events_and_no_discrepancy_on_a_clean_grasp(self):
        recs = P.build(grasp_round_rec(ok=True), grasp_round_arrays(), META_T, TASK)
        self.assertEqual([r["step"] for r in recs], ["settle", "pregrasp", "grip", "lift"])
        self.assertEqual([r["kind"] for r in recs], ["none", "reach", "grip", "reach"])
        self.assertTrue(all(r["schema"] == P.SCHEMA for r in recs))
        self.assertEqual(recs[2]["record_id"], "pick_Mug_s1/2.1")
        grip, lift = recs[2], recs[3]
        flips = [(e["edge"], e["to"]) for e in grip["events"]]
        self.assertIn((["Mug", "touching", "right_hand"], True), flips)
        self.assertIn((["Mug", "gripped", "right_hand"], True), flips)
        self.assertEqual(grip["discrepancies"], [])
        self.assertEqual(grip["roles"]["effector"], "right_hand")
        self.assertEqual(grip["action"]["check"], {"opening_cm": 6.1})
        self.assertEqual(lift["roles"]["carried"], "Mug")
        self.assertIn((["Mug", "supported_by", "T"], False), [(e["edge"], e["to"]) for e in lift["events"]])
        self.assertIn((["Mug", "held", "right_hand"], True), [(e["edge"], e["to"]) for e in lift["events"]])
        self.assertEqual(lift["discrepancies"], [])
        self.assertEqual(lift["outcome"]["round_ok"], True)
        self.assertEqual(
            (lift["evidence"]["npz"], lift["evidence"]["k0"], lift["evidence"]["k1"]), ("pick_Mug_s1.npz", 30, 39)
        )

    def test_a_grip_that_closes_one_finger_only_shows_the_discrepancy(self):
        recs = P.build(grasp_round_rec(ok=False), grasp_round_arrays(close_both=False), META_T, TASK)
        grip = recs[2]
        self.assertEqual([d["edge"] for d in grip["discrepancies"]], [["Mug", "gripped", "right_hand"]])
        self.assertEqual(grip["discrepancies"][0]["observed"], False)
        self.assertEqual(grip["outcome"]["failure_class"], "slip")
        self.assertIsNone(recs[3]["roles"]["carried"])  # one finger is not a grip: nothing is carried into the lift

    def test_no_arrays_means_no_records_and_the_error_line_shape(self):
        self.assertEqual(P.build(dict(label="x", steps=[], calls=[]), None, META_T, TASK), [])
        e = P.error_line("x", "boom")
        self.assertEqual((e["schema"], e["label"], e["error"]), (P.SCHEMA, "x", "boom"))
        n = P.note_line("x", "no records: no skill covers this goal")
        self.assertEqual((n["schema"], n["label"], n["note"]), (P.SCHEMA, "x", "no records: no skill covers this goal"))
        self.assertNotIn("record_id", n)

    def test_write_and_read_roundtrip(self):
        recs = P.build(grasp_round_rec(), grasp_round_arrays(), META_T, TASK)
        with tempfile.TemporaryDirectory() as d:
            p = P.write(Path(d), "pick_Mug_s1", recs)
            self.assertEqual(p.name, "pick_Mug_s1.past.jsonl")
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ["pick_Mug_s1.past.jsonl"])
            back = P.read(p)
        self.assertEqual(len(back), len(recs))
        self.assertEqual(back[1]["record_id"], recs[1]["record_id"])
        json.dumps(back)  # plain JSON all the way down


def affordance_meta():
    """META_T with the scene file's affordance fields: an ADT-style mug, a gso-style can, a container."""
    meta = json.loads(json.dumps(META_T))
    mug, can, far = (meta["objects"][k] for k in ("Mug", "Can", "Far"))
    mug["semantic"].update(graspable="single", phrase="the mug", friction=0.7, adt={"category": "kitchen"},
                           drop_test={"ok": True})  # fmt: skip
    mug["provenance"] = {"graspable": "rule", "drop_test": "measured"}
    can["box"]["half_extent"] = [0.07, 0.03, 0.03]
    can["semantic"]["graspable"] = "single"
    can["gso"] = {"grasp_height_m": 0.79, "in_reach_band": True, "capability": "measured_envelope"}
    far["semantic"]["container"] = {"is": True, "opening_half_extent": [0.025, 0.025], "rim_height": 0.06}
    return meta


def affordance_model():
    import mujoco

    boxes = {"Mug": (0.04, 0.04, 0.05), "Can": (0.07, 0.03, 0.03), "Far": (0.03, 0.03, 0.03)}
    bodies = "".join(f'<body name="{k}"><freejoint/><geom type="box" size="{h[0]} {h[1]} {h[2]}"/></body>'
                     for k, h in boxes.items())  # fmt: skip
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")


class Affordance(unittest.TestCase):
    def test_static_fields_are_copied_with_their_provenance_and_the_drop_test_is_not(self):
        meta = affordance_meta()
        a = P.affordance_static(meta, "Mug")
        self.assertEqual(
            (a["graspable"], a["phrase"], a["friction"], a["category"]), ("single", "the mug", 0.7, "kitchen")
        )
        self.assertEqual((a["src"]["graspable"], a["src"]["phrase"], a["src"]["category"]), ("rule", "scene", "adt"))
        self.assertNotIn("drop_test", a)
        c = P.affordance_static(meta, "Can")
        self.assertEqual((c["grasp_height_m"], c["src"]["grasp_height_m"]), (0.79, "gso"))
        f = P.affordance_static(meta, "Far")
        self.assertEqual((f["opening_half_extent_m"], f["rim_height_m"]), ([0.025, 0.025], 0.06))
        self.assertEqual(P.hug_geometry(meta, "Can"), dict(span_x_m=0.14, span_y_m=0.06, height_m=0.06, mass_kg=0.4))

    def test_without_the_model_the_records_carry_the_scene_fields_and_no_one_hand_check(self):
        recs = P.build(grasp_round_rec(), grasp_round_arrays(), affordance_meta(), TASK)
        a = recs[1]["entities"]["Mug"]["affordance"]
        self.assertEqual((a["graspable"], a["src"]["hug"]), ("single", "derived:box"))
        self.assertNotIn("one_hand", a)
        self.assertNotIn("affordance", recs[1]["entities"]["T"])  # surfaces and furniture: none

    def test_the_one_hand_check_is_in_the_objects_frame_and_goes_stale_when_it_tips(self):
        arr = grasp_round_arrays()
        yaw90 = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
        tipped = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]  # 90 deg about x
        arr["obj_Can"] = np.array([[1.0, 0.10, 0.76, *(yaw90 if k < 20 else tipped)] for k in range(40)])
        recs = P.build(grasp_round_rec(), arr, affordance_meta(), TASK, model=affordance_model())
        pre, grip = recs[1], recs[2]
        can = pre["entities"]["Can"]["affordance"]
        oh = can["one_hand"]
        self.assertEqual(can["src"]["one_hand"], "derived:graspability")
        # 14 cm along its x, 6 cm along its y: the jaw closes across y, so the approach runs along x whatever the yaw
        self.assertAlmostEqual(abs(oh["narrowest"]["approach_deg"]) % 180.0, 0.0, delta=1.0)
        self.assertAlmostEqual(oh["narrowest"]["width_m"], 0.06, places=3)
        self.assertTrue(any(lo <= 0.0 <= hi for lo, hi in oh["side_fits_deg"]))
        self.assertFalse(any(lo <= 90.0 <= hi for lo, hi in oh["side_fits_deg"] if lo <= hi))
        self.assertFalse(can["stale"])
        self.assertTrue(pre["entities"]["Mug"]["affordance"]["one_hand"]["side_fits_deg"])  # 8 cm: fits all round
        g = grip["entities"]["Can"]["affordance"]
        self.assertTrue(g["stale"])
        self.assertAlmostEqual(g["tilt_deg"], 90.0, delta=0.5)
        json.dumps(recs, default=P._json_default)

    def test_a_check_that_raises_leaves_one_hand_out(self):
        class Broken:
            meta = affordance_meta()

            def object_pose(self, key):
                raise RuntimeError("no pose")

        a = P.affordance(affordance_meta(), "Mug", Broken())
        self.assertNotIn("one_hand", a)
        self.assertIn("RuntimeError: no pose", a["one_hand_error"])

    def test_runs_through_plus_minus_180_join(self):
        self.assertEqual(P._ranges([-180.0, -175.0, -5.0, 0.0, 5.0, 170.0, 175.0], 5.0), [[170.0, -175.0], [-5.0, 5.0]])


class OldRounds(unittest.TestCase):
    def test_without_a_call_log_the_kind_comes_from_the_trace_primitives(self):
        arr = grasp_round_arrays()
        arr["primitives"] = np.array(["reach", "grip", "hold"])
        arr["primitive"] = np.array([0] * 20 + [1] * 10 + [0] * 10, dtype=np.int16)
        rec = grasp_round_rec()
        rec["calls"] = []
        recs = P.build(rec, arr, META_T, TASK)
        self.assertEqual([r["kind"] for r in recs], ["none", "reach", "grip", "reach"])
        self.assertEqual([r["kind_src"] for r in recs], ["calls", "primitive", "primitive", "primitive"])
        self.assertIsNone(recs[2]["roles"]["effector"])  # the trace carries no side
        self.assertEqual(recs[2]["expected"], [])


class BlockingExemptions(unittest.TestCase):
    def test_the_room_shell_and_the_support_furniture_never_block_the_stance(self):
        meta = json.loads(json.dumps(META_G))
        meta["objects"]["Room"] = {"free": False, "body": None, "container": False, "init_pos": [0, 0, 0],
                                   "semantic": {"movable": False},
                                   "box": {"half_extent": [6.5, 4.9, 4.2], "centre": [1.0, 0.0, 2.0]}}  # fmt: skip
        meta["objects"]["Stool"] = {"free": False, "body": None, "container": False, "init_pos": [0, 0, 0],
                                    "semantic": {"movable": False},
                                    "box": {"half_extent": [0.15, 0.15, 0.35], "centre": [0.45, 0.0, 0.35]}}  # fmt: skip
        self.assertTrue(P.is_shell(meta, "Room"))
        self.assertFalse(P.is_shell(meta, "Table"))
        objects, hands = mug_round()
        f = P.Frame(arrays_for(20, objects, hands), meta, 0)
        roles = dict(target="Mug", effector=None, carried=None)
        entities, scope = P.scope_of(f, roles, "walk", TASK, meta, [call("walk_to", 0.0, 0.38, [0.4, 0.0])])
        self.assertNotIn("Room", scope["initial"])
        self.assertEqual(P.support_furniture(f, "Mug", meta), {"Table"})
        g = P.graph_of(f, entities, roles, meta)
        blockers = {e["s"] for e in g["edges"] if e["r"] == "blocking" and e["v"]}
        self.assertIn("Stool", blockers)
        self.assertNotIn("Table", blockers)
        self.assertNotIn("Room", blockers)


class ReachLanding(unittest.TestCase):
    def test_at_uses_the_motors_jaw_error_when_the_step_logged_one(self):
        g = dict(edges=[dict(s="right_hand", r="at", o="region:pregrasp_target", v=False, src="approx", err_cm=14.9)])
        P.measured_at(g, dict(step="pregrasp", jaw_err_cm=0.98))
        self.assertEqual((g["edges"][0]["v"], g["edges"][0]["src"], g["edges"][0]["err_cm"]), (True, "observed", 0.98))
        g2 = dict(edges=[dict(s="right_hand", r="at", o="region:pregrasp_target", v=False, src="approx", err_cm=14.9)])
        P.measured_at(g2, dict(step="pregrasp", check={"jaw_err_cm": 4.2}))
        self.assertEqual((g2["edges"][0]["v"], g2["edges"][0]["err_cm"]), (False, 4.2))

    def test_without_a_measured_error_the_wrist_to_jaw_offset_is_taken_off(self):
        objects, hands = mug_round()
        hands["right"] = np.tile([0.85 - P.JAW_OFF_M, 0.0, 0.78], (20, 1))  # the wrist one jaw offset behind the target
        f = P.Frame(arrays_for(20, objects, hands), META, 19)
        entities, _ = P.scope_of(f, ROLES, "reach", TASK, META, REACH)
        at = [e for e in P.graph_of(f, entities, ROLES, META)["edges"] if e["r"] == "at"][0]
        self.assertEqual((at["v"], at["src"]), (True, "approx"))
        self.assertLess(at["err_cm"], 0.5)
