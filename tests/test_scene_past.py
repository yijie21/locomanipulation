"""TODO 39: the PAST recorder's builder (src/locomani/tasks/scene/past.py) on hand-made rounds, no simulation."""

from __future__ import annotations

import json
import unittest

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
        f = P.Frame(arrays_for(20, objects, hands, contacts=[(0, "Mug", "Table")]), META_T, 0)
        entities, _ = P.scope_of(f, ROLES, "reach", TASK, META_T, REACH)
        g = P.graph_of(f, entities, ROLES, META_T)
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
