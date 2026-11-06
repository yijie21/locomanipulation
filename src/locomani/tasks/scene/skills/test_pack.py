"""The pack line's pure parts (pack.py, bimanual.py, 2026-11-05): `plan_for` sends one in(X, C) with held(C) (and
lifted / upright of C) to `pack.pack_into`, held(X) and in(X, C) of a two-hand X (by the scene file's features, never
the name) to the hug skills; the chain stops at a failed put-in, then hugs the container and reads the goal again; the
drop stance stands on the container's hug line with the carried offset kept where the pelvis stays outside the edge;
the box centre and underside follow a turned `box.centre`. Lives in the skills package because the loop's Engineer may
change nothing outside it; run with `make test TESTS="src/locomani/tasks/scene/skills/test_pack.py"`."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import locomani.tasks.scene.skills as skills
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene.skills import bimanual, held, into, pack, pick
from locomani.tasks.scene.skills.test_plan_for import World, yaw_quat

CARTON = dict(half_extent=[0.225, 0.16, 0.165], centre=[0.0, 0.0, 0.165], wall=0.01)


def carton_world(check=(True, "")):
    """test_plan_for's room with the open carton on the table T (approached from -y), 0.2 m in from that edge, its
    long side along it; `check` is what the world's goal check answers."""

    class CartonWorld(World):
        def __init__(self):
            super().__init__()
            self.checked, self.keys = [], []

        def check(self, g):
            self.checked.append(list(g))
            return lambda state: check

        def state(self, keys):
            self.keys.append(list(keys))
            return {}

    world = CartonWorld()
    world.meta["objects"]["Carton"] = dict(surface="T", free=True, init_pos=[0.0, -0.1, 0.7], init_quat=yaw_quat(0.0),
                                           box=dict(CARTON), semantic=dict(graspable="bimanual", phrase="open carton"))  # fmt: skip
    world.free = {"Can": 7, "Carton": 8}
    return world


def ctx_for(world, goal):
    ctx = SimpleNamespace(world=world, task=SimpleNamespace(goal=goal), motor=SimpleNamespace(t=1.0), log=[])
    ctx.note = lambda name, out, **kw: (ctx.log.append(dict(step=name, ok=out.ok, why=out.why, **kw)), out)[1]
    return ctx


class PlanFor(unittest.TestCase):
    def test_in_with_held_and_lifted_of_the_container_goes_to_the_pack_chain_and_other_mixes_have_no_plan(self):
        ctx = ctx_for(carton_world(), [])
        with mock.patch.object(pack, "pack_into", return_value=Outcome(True)) as p:
            skills.plan_for(SimpleNamespace(goal=["in(Can, Carton)", "held(Carton)", "lifted(Carton, 0.20, 60)"]))(ctx)
            skills.plan_for(SimpleNamespace(goal=["held(Carton)", "in(Can, Carton)"]))(ctx)
            skills.plan_for(SimpleNamespace(goal=["in(Can, Carton)", "held(Carton)", "upright(Carton)"]))(ctx)
        self.assertEqual(p.call_args_list, [mock.call(ctx, "Can", "Carton")] * 3)
        for goal in (
            ["in(Can, Carton)", "held(Can)"],  # held of the object, not the container
            ["in(Can, Carton)", "held(Carton)", "upright(Can)"],
            ["in(Can, Carton)", "lifted(Carton, 0.20, 60)"],  # no held
            ["in(Can, Carton)", "in(Book, Carton)", "held(Carton)"],
        ):
            self.assertIsNone(skills.plan_for(SimpleNamespace(goal=goal)), goal)

    def test_a_two_hand_object_by_its_features_gets_the_hug_pick_and_the_two_hand_put_in(self):
        world = carton_world()
        ctx = ctx_for(world, [])
        with (
            mock.patch.object(bimanual, "pick", return_value=Outcome(True)) as hug,
            mock.patch.object(bimanual, "place_into", return_value=Outcome(True)) as put,
            mock.patch.object(held, "pick_held", return_value=Outcome(True)) as one,
            mock.patch.object(into, "place_into", return_value=Outcome(True)) as one_in,
        ):
            skills.plan_for(SimpleNamespace(goal=["held(Carton)", "lifted(Carton, 0.20, 60)"]))(ctx)
            skills.plan_for(SimpleNamespace(goal=["held(Can)"]))(ctx)
            skills.plan_for(SimpleNamespace(goal=["in(Carton, Bin)"]))(ctx)
            skills.plan_for(SimpleNamespace(goal=["in(Can, Carton)"]))(ctx)
            world.meta["objects"]["Can"]["bimanual"] = True  # the box line's flag counts too
            skills.plan_for(SimpleNamespace(goal=["held(Can)"]))(ctx)
        self.assertEqual(hug.call_args_list, [mock.call(ctx, "Carton"), mock.call(ctx, "Can")])
        self.assertEqual(one.call_args_list, [mock.call(ctx, "Can", pick.pick, False)])
        self.assertEqual(put.call_args_list, [mock.call(ctx, "Carton", "Bin")])
        self.assertEqual(one_in.call_args_list, [mock.call(ctx, "Can", "Carton")])
        self.assertFalse(bimanual.two_handed(world, "nothing"))


class Chain(unittest.TestCase):
    GOAL = ["in(Can, Carton)", "held(Carton)", "lifted(Carton, 0.20, 60)"]

    def test_a_failed_put_in_ends_the_plan_before_the_lift(self):
        world = carton_world()
        ctx = ctx_for(world, self.GOAL)
        with (
            mock.patch.object(into, "place_into", return_value=Outcome(False, "no release over Carton")) as put,
            mock.patch.object(bimanual, "pick") as hug,
        ):
            out = pack.pack_into(ctx, "Can", "Carton")
        self.assertEqual((out.ok, out.why), (False, "put-in of Can: no release over Carton"))
        self.assertEqual((put.call_count, hug.call_count), (1, 0))
        self.assertEqual([(r["step"], r["hands"]) for r in ctx.log], [("pack_put_in", "one")])

    def test_the_put_in_then_the_hug_then_the_goal_read_again_on_both_objects(self):
        world = carton_world(check=(False, "[goal] in(Can, Carton): outside"))
        ctx = ctx_for(world, self.GOAL)
        with (
            mock.patch.object(into, "place_into", return_value=Outcome(True)),
            mock.patch.object(bimanual, "pick", return_value=Outcome(True, "", 3.0, 9.0)) as hug,
        ):
            out = pack.pack_into(ctx, "Can", "Carton")
        self.assertEqual(hug.call_args_list, [mock.call(ctx, "Carton")])
        self.assertEqual((out.ok, out.why, out.t1), (False, "[goal] in(Can, Carton): outside", 9.0))
        self.assertEqual((world.checked, world.keys), ([self.GOAL], [["Can", "Carton"]]))
        self.assertEqual([r["step"] for r in ctx.log], ["pack_put_in", "pack_lift"])
        world = carton_world()
        ctx = ctx_for(world, self.GOAL)
        with (
            mock.patch.object(into, "place_into", return_value=Outcome(True)),
            mock.patch.object(bimanual, "pick", return_value=Outcome(True)),
        ):
            self.assertTrue(pack.pack_into(ctx, "Can", "Carton").ok)

    def test_a_two_hand_object_is_put_in_with_both_hands(self):
        world = carton_world()
        world.meta["objects"]["Can"]["semantic"] = dict(graspable="bimanual")
        ctx = ctx_for(world, self.GOAL)
        with (
            mock.patch.object(bimanual, "place_into", return_value=Outcome(False, "hug pick failed: x")) as put,
            mock.patch.object(into, "place_into") as one,
        ):
            out = pack.pack_into(ctx, "Can", "Carton")
        self.assertEqual((put.call_count, one.call_count, ctx.log[0]["hands"]), (1, 0, "both"))
        self.assertIn("hug pick failed", out.why)


class Geometry(unittest.TestCase):
    def test_the_drop_stance_is_on_the_cartons_hug_line_with_the_carried_offset_clipped(self):
        world = carton_world()
        # the carton's nearest approach edge is -y (0.2 m in); its axis closest to the inward normal is y, so the
        # robot faces +y and the hands squeeze across the 0.45 m long side, which lies along the edge
        spot = bimanual.drop_stance(world, "Carton", 0.5)
        self.assertEqual((spot["edge"], round(spot["depth"], 6), round(spot["ahead"], 6)), ("-y", 0.2, 0.5))
        np.testing.assert_allclose(spot["xy"], [0.0, -0.6], atol=1e-9)
        self.assertAlmostEqual(spot["yaw"], np.pi / 2)
        np.testing.assert_allclose(spot["v"], [0.0, 1.0], atol=1e-9)
        # too short a carried offset would stand the pelvis inside EDGE_OUT_MIN_M of the edge: pushed out ...
        short = bimanual.drop_stance(world, "Carton", 0.3)
        self.assertAlmostEqual(short["ahead"], 0.2 + bimanual.EDGE_OUT_MIN_M)
        np.testing.assert_allclose(short["xy"], [0.0, -0.1 - 0.2 - bimanual.EDGE_OUT_MIN_M], atol=1e-9)
        # ... and a long one is capped at the hug's farthest grasp line
        self.assertAlmostEqual(bimanual.drop_stance(world, "Carton", 0.9)["ahead"], bimanual.AHEAD_MAX_M)
        g = bimanual.geometry(world, "Carton")
        self.assertEqual((round(g["w"], 3), round(g["depth_half"], 3), round(g["rim"], 3)), (0.225, 0.16, 1.03))

    def test_box_centre_and_underside_follow_a_turned_centre_offset(self):
        world = carton_world()
        world.meta["objects"]["Can"]["box"] = dict(half_extent=[0.05, 0.05, 0.1], centre=[0.3, 0.0, 0.1])
        world.poses["Can"] = ([1.0, 2.0, 0.5], yaw_quat(np.pi / 2))  # the body's x points along +y
        np.testing.assert_allclose(bimanual.box_centre(world, "Can"), [1.0, 2.3, 0.6], atol=1e-9)
        self.assertAlmostEqual(bimanual.underside(world, "Can"), 0.5)  # centre 0.1 up, half height 0.1
        world.poses["Can"] = ([1.0, 2.0, 0.5], yaw_quat(0.0))
        np.testing.assert_allclose(bimanual.box_centre(world, "Can"), [1.3, 2.0, 0.6], atol=1e-9)

    def test_the_library_lists_the_two_modules_parameters(self):
        params = skills.parameters(skills)
        self.assertEqual(params["bimanual.PUT_CLEAR_M"], bimanual.PUT_CLEAR_M)
        self.assertEqual(params["bimanual.SQUEEZE_M"], bimanual.SQUEEZE_M)
        self.assertIn("into.DROP_CLEAR_M", params)
        skills.apply_set(skills, {"bimanual.PUT_OVER_M": 0.05})
        self.assertEqual(bimanual.PUT_OVER_M, 0.05)
        bimanual.PUT_OVER_M = 0.03
