"""`into.place_into`'s pure parts: `plan_for` sends exactly one in(X, C), alone or with upright(X), to it; the drop
stances at a container on a surface (edge-normal, deep) and on the floor; the near drop at the pick's edge, its
T_est and its lift over what rests on the support along the carried path; the release check -- the footprint inside the
inner box by RELEASE_MARGIN_M, the underside above the rim -- its one correction by the commanded target, and no release
when it still fails (set down over a surface, else kept and raised). The fakes are test_plan_for.py's and
test_held.py's; run with `make test TESTS="src/locomani/tasks/scene/skills/test_into.py"`."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import locomani.tasks.scene.skills as skills
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import MotorConfig, Outcome
from locomani.primitives.navigator import build
from locomani.tasks.scene.skills import into, pick, place
from locomani.tasks.scene.skills.test_held import Gripper
from locomani.tasks.scene.skills.test_plan_for import Ctx, World, yaw_quat


def bowl_world(xy=(0.0, 0.0)):
    """test_plan_for's room with a bowl (inner half 0.06 m, rim 0.76 m) on the table T, the book off it."""
    world = World({"Book": ([2.0, -2.0, 0.13], yaw_quat(0.0))})
    world.meta["objects"]["Bowl"] = dict(surface="T", free=False, init_pos=[xy[0], xy[1], 0.73],
                                         init_quat=yaw_quat(0.0), box=dict(half_extent=[0.07, 0.07, 0.03], wall=0.01))  # fmt: skip
    world.occ, _ = build(world.meta, body_radius=0.35)
    return world


class PlanFor(unittest.TestCase):
    def test_one_in_alone_or_with_upright_of_the_same_object_goes_to_place_into(self):
        with mock.patch.object(into, "place_into", return_value=Outcome(True)) as p:
            skills.plan_for(SimpleNamespace(goal=["in(Can, Bowl)"]))("ctx")
            skills.plan_for(SimpleNamespace(goal=["in(Can, Bowl)", "upright(Can)"]))("ctx")
        self.assertEqual(p.call_args_list, [mock.call("ctx", "Can", "Bowl")] * 2)
        for goal in (
            ["in(Can, Bowl)", "upright(Book)"],
            ["in(Can, Bowl)", "in(Book, Bowl)"],
            ["in(Can, Bowl)", "on(Book, T)"],
            ["in(Can, Bowl)", "held(Can)"],
        ):
            self.assertIsNone(skills.plan_for(SimpleNamespace(goal=goal)), goal)


class Stances(unittest.TestCase):
    def test_a_container_on_a_surface_gets_the_listed_edge_first_and_a_deep_one_the_pelvis_outside_the_edge(self):
        with mock.patch.dict(pick._NAV, {}, clear=True):
            world = bowl_world()
            first = into.stances(world, "Bowl", "right", np.array([0.0, -2.0]))[0]
            self.assertEqual((first["edge"], round(first["depth"], 6)), ("-y", 0.3))
            self.assertAlmostEqual(first["face"], np.pi / 2)
            np.testing.assert_allclose(first["stance"], [-pick.STAND_SIDE_M, -pick.STAND_AHEAD_M], atol=1e-9)
            np.testing.assert_allclose(first["edge_xy"], [0.0, -0.3], atol=1e-9)
        with mock.patch.dict(pick._NAV, {}, clear=True):
            world = bowl_world((0.0, 0.25))  # 0.55 m in from -y
            first = into.stances(world, "Bowl", "right", np.array([0.0, -2.0]))[0]
            self.assertEqual(first["edge"], "-y")
            self.assertAlmostEqual(float(first["stance"][1]), -0.3 - into.EDGE_OUT_M)
            with mock.patch.object(into, "DEEP_MAX_M", 0.0):
                self.assertNotIn("-y", [s["edge"] for s in into.stances(world, "Bowl", "right", np.zeros(2))])

    def test_a_floor_container_gets_stances_around_it_facing_it(self):
        with mock.patch.dict(pick._NAV, {}, clear=True):
            world = bowl_world()
            world.meta["objects"]["Bin"] = dict(surface=None, free=False, init_pos=[-2.0, 1.5, 0.15],
                                                init_quat=yaw_quat(0.0), box=dict(half_extent=[0.15, 0.1, 0.15]))  # fmt: skip
            got = into.stances(world, "Bin", "right", np.array([-2.0, 0.0]))
            self.assertEqual(len(got), into.FLOOR_FACES)
            for s in got:
                ahead = R.rot_z(s["face"])[:2, :2].T @ (np.array([-2.0, 1.5]) - s["stance"])
                np.testing.assert_allclose(ahead, [into.FLOOR_AHEAD_M, -pick.STAND_SIDE_M], atol=1e-9)
            self.assertLess(
                np.linalg.norm(got[0]["stance"] - [-2.0, 0.0]), np.linalg.norm(got[-1]["stance"] - [-2.0, 0.0])
            )

    def test_the_near_drop_is_at_the_picks_edge_within_near_m_and_its_t_est_skips_the_back_off_and_route(self):
        with mock.patch.dict(pick._NAV, {}, clear=True):
            world = bowl_world()
            pick_stance = np.array([-0.32, -0.55])
            near = into.near_spot(world, "Bowl", "right", pick_stance, np.pi / 2)
            self.assertEqual(near["edge"], "-y")
            # the facing check at the NEAR_M it was written for (r1_in_single set 1.6)
            with mock.patch.object(into, "NEAR_M", 0.6):
                self.assertIsNone(into.near_spot(world, "Bowl", "right", pick_stance, 0.0))
            with mock.patch.object(into, "NEAR_M", 0.1):
                self.assertIsNone(into.near_spot(world, "Bowl", "right", pick_stance, np.pi / 2))
            world.poses["Can"] = ([-0.2, -0.15, 0.79], yaw_quat(0.0))
            world.meta["objects"]["Can"]["surface"] = "T"
            plan = (np.pi / 2, 0.15, np.pi / 2, pick_stance, np.array([-0.2, -0.3]))
            with mock.patch.object(pick, "plan_hand", return_value=(plan, "right", "")):
                ahead = into.plan_ahead(world, "Can", "Bowl", "right", pick_stance, MotorConfig())
                self.assertTrue(ahead["near"])
                self.assertIsNone(ahead["corners"])
                with mock.patch.object(into, "NEAR_M", 0.0):
                    far = into.plan_ahead(world, "Can", "Bowl", "right", pick_stance, MotorConfig())
            self.assertFalse(far["near"])
            self.assertIsNotNone(far["corners"])
            self.assertLess(ahead["t_est"], far["t_est"])


class NearLift(unittest.TestCase):
    """The near drop's pick lifts over what rests on the support along the carried path, and the near drop is not
    taken when that needs more than NEAR_LIFT_MAX_M."""

    def _tall(self, world, xy, top):
        h = (top - 0.7) / 2  # resting on the table T (top 0.7 m)
        box = dict(half_extent=[0.03, 0.03, h])
        world.meta["objects"]["Tall"] = dict(surface="T", free=False, init_pos=[xy[0], xy[1], 0.7 + h],
                                             init_quat=yaw_quat(0.0), box=box)  # fmt: skip

    def test_the_lift_clears_what_the_path_crosses_by_near_clear_m_else_it_is_lift_m(self):
        world = bowl_world()
        world.poses["Can"] = ([-0.3, -0.2, 0.79], yaw_quat(0.0))  # underside on the table top 0.70
        world.meta["objects"]["Can"]["surface"] = "T"
        self._tall(world, (0.0, -0.2), 0.94)
        stance = np.array([-0.3, -0.6])
        lift, crossed = into.near_lift(world, "Can", stance, dict(stance=np.array([0.1, -0.6])))
        self.assertEqual(crossed, ["Tall"])
        self.assertAlmostEqual(lift, 0.94 + into.NEAR_CLEAR_M - 0.70)
        lift, crossed = into.near_lift(world, "Can", stance, dict(stance=np.array([-0.3, -0.5])))
        self.assertEqual((lift, crossed), (into.grasp_program.LIFT_M, []))

    def test_a_near_drop_needing_more_than_near_lift_max_m_is_not_taken(self):
        with mock.patch.dict(pick._NAV, {}, clear=True):
            world = bowl_world()
            pick_stance = np.array([-0.32, -0.55])
            world.poses["Can"] = ([-0.2, -0.15, 0.79], yaw_quat(0.0))
            world.meta["objects"]["Can"]["surface"] = "T"
            plan = (np.pi / 2, 0.15, np.pi / 2, pick_stance, np.array([-0.2, -0.3]))
            with mock.patch.object(pick, "plan_hand", return_value=(plan, "right", "")):
                low = into.plan_ahead(world, "Can", "Bowl", "right", pick_stance, MotorConfig())
                self._tall(world, (-0.1, -0.075), 0.70 + into.NEAR_LIFT_MAX_M)
                high = into.plan_ahead(world, "Can", "Bowl", "right", pick_stance, MotorConfig())
        self.assertTrue(low["near"])
        self.assertEqual(low["crossed"], ["Bowl"])
        self.assertAlmostEqual(low["lift"], into.grasp_program.LIFT_M)
        self.assertFalse(high["near"])
        self.assertIsNone(high["lift"])

    def test_the_pick_lifts_by_the_near_lift_and_lift_m_is_restored(self):
        seen = {}

        def fake_pick(ctx, key, side):
            seen["lift"] = into.grasp_program.LIFT_M
            return Outcome(False, "stop here")

        world = World({"Can": ([-0.2, -0.15, 0.79], yaw_quat(0.0))})
        world.free = {"Can": 7}
        m = SimpleNamespace(t=0.0, sim=SimpleNamespace(d=SimpleNamespace(qpos=np.zeros(3))), cfg=MotorConfig())
        ahead = dict(near=True, lift=0.23, crossed=["Tall"])
        with (
            mock.patch.object(place, "blocking_neighbour", return_value=None),
            mock.patch.object(into, "plan_ahead", return_value=ahead),
            mock.patch.object(place, "_refusal", return_value=None),
            mock.patch.object(pick, "pick", side_effect=fake_pick),
            mock.patch.object(place, "_fail", return_value=Outcome(False, "failed")),
        ):
            ctx = Ctx(world, m)
            into.place_into(ctx, "Can", "Bowl")
        self.assertEqual(seen["lift"], 0.23)
        self.assertEqual(into.grasp_program.LIFT_M, 0.10)
        self.assertEqual(ctx.log[0]["step"], "near_lift")


class ReleaseFit(unittest.TestCase):
    def square(self, xy, z, half=0.03):
        return np.array([[xy[0] + sx * half, xy[1] + sy * half, z + sz * 0.05] for sx in (-1, 1) for sy in (-1, 1)
                         for sz in (-1, 1)])  # fmt: skip

    def test_inside_by_the_margin_above_the_rim_passes_else_says_why_and_the_move_back(self):
        half, wall, rim = np.array([0.07, 0.07, 0.03]), 0.01, 0.76
        ok, why, move = into.release_fit(self.square([1.0, 2.0], 0.84), [1.0, 2.0], rim, half, wall, 0.0)
        self.assertTrue(ok, why)
        np.testing.assert_allclose(move, [0.0, 0.0, rim + into.DROP_CLEAR_M - 0.79], atol=1e-9)
        room = 0.07 - wall - into.RELEASE_MARGIN_M - 0.03  # the square's centre may be this far off
        self.assertTrue(into.release_fit(self.square([1.0 + room - 1e-4, 2.0], 0.84), [1.0, 2.0], rim, half, wall,
                                         0.0)[0])  # fmt: skip
        ok, why, move = into.release_fit(self.square([1.0 + room + 0.005, 2.0], 0.84), [1.0, 2.0], rim, half, wall, 0.0)
        self.assertFalse(ok)
        self.assertIn("past the inner box", why)
        np.testing.assert_allclose(move[:2], [-(room + 0.005), 0.0], atol=1e-9)
        ok, why, _ = into.release_fit(self.square([1.0, 2.0], 0.80), [1.0, 2.0], rim, half, wall, 0.0)
        self.assertEqual((ok, why), (False, "underside 1.0 cm below the rim"))
        # in C's own frame: a container yawed 45 deg has the corners of an axis-aligned square on its axes
        self.assertTrue(into.release_fit(self.square([1.0, 2.0], 0.84, 0.045), [1.0, 2.0], rim, half, wall, 0.0)[0])
        self.assertFalse(into.release_fit(self.square([1.0, 2.0], 0.84, 0.045), [1.0, 2.0], rim, half, wall,
                                          np.pi / 4)[0])  # fmt: skip


class Biased(Gripper):
    """A Gripper whose jaw lands `bias` off every reach target."""

    bias = np.zeros(3)

    def reach(self, side, pos, rot=None, **kw):
        self.calls.append(("reach", np.asarray(pos, float), None if rot is None else np.asarray(rot, float)))
        self._carry(np.asarray(pos, float) + self.bias, self.rot if rot is None else np.asarray(rot, float))
        return Outcome(True)


class DropIn(unittest.TestCase):
    """The Can carried in the right jaw over a bin on the floor (rim 0.30 m), the robot facing +y."""

    def _held(self, bin_half=(0.15, 0.1), bias=(0.0, 0.0, 0.0)):
        world = World({"Can": ([-2.0, -2.5, 0.8], yaw_quat(0.0))})
        world.meta["objects"]["Bin"] = dict(surface=None, free=False, init_pos=[-2.0, -2.0, 0.15],
                                            init_quat=yaw_quat(0.3), box=dict(half_extent=[*bin_half, 0.15]))  # fmt: skip
        world.free = {"Can": 7}
        world.check = lambda goals: lambda state: (True, "")
        world.state = lambda keys: {}
        m = Biased(world, [-2.0, -2.5, 0.8], np.eye(3), yaw=np.pi / 2)
        m.bias = np.asarray(bias, float)
        spot = dict(xy=np.array([-2.0, -2.0]), face=np.pi / 2, edge_xy=np.array([-2.0, -2.1]), edge="floor4", depth=0.1)
        return world, m, Ctx(world, m), spot

    def _run(self, ctx, spot, z0=0.8):
        with mock.patch.object(pick, "object_vertices", return_value=np.zeros((0, 3))):
            return into._drop_in(ctx, "Can", "Bin", "right", spot, ("S",), z0)

    def test_centred_over_the_opening_it_opens_rises_retracts_and_steps_back(self):
        world, m, ctx, spot = self._held()
        self.assertTrue(self._run(ctx, spot).ok)
        self.assertEqual([r["step"] for r in ctx.log], ["over", "lower", "release_fit", "open", "rise", "retract",
                                                        "step_back"])  # fmt: skip
        np.testing.assert_allclose(m.calls[1][1], [-2.0, -2.0, 0.3 + into.DROP_CLEAR_M + 0.09], atol=1e-9)
        np.testing.assert_allclose(m.calls[0][1], m.calls[1][1] + [0.0, 0.0, into.ABOVE_M], atol=1e-9)

    def test_a_steady_landing_error_is_corrected_once_by_the_commanded_target(self):
        world, m, ctx, spot = self._held(bias=(0.12, 0.0, 0.0))
        self.assertTrue(self._run(ctx, spot).ok)
        fits = [r for r in ctx.log if r["step"] == "release_fit"]
        self.assertEqual([(r["ok"], r["attempt"]) for r in fits], [(False, 0), (True, 1)])
        self.assertIn("raise_again", [r["step"] for r in ctx.log])
        self.assertEqual((fits[0]["off_cm"], fits[1]["off_cm"]), (12.0, 0.0))  # the fake never lets go: at the fit

    def test_still_outside_it_never_opens_raised_over_the_floor_set_down_over_a_surface(self):
        world, m, ctx, spot = self._held(bin_half=(0.035, 0.035))
        out = self._run(ctx, spot)
        self.assertFalse(out.ok)
        self.assertIn("kept in the right hand", out.why)
        self.assertNotIn(("release",), m.calls)
        self.assertEqual(ctx.log[-1]["step"], "raise_back")
        self.assertAlmostEqual(float(world.object_pose("Can")[0][2]), 0.8 - into.RAISE_BELOW_START_M)
        world, m, ctx, spot = self._held(bin_half=(0.035, 0.035))
        with (
            mock.patch.object(place, "surface_below", return_value="S"),
            mock.patch.object(place, "_fail", return_value=Outcome(False, "set")) as fail,
        ):
            self.assertEqual(self._run(ctx, spot).why, "set")
        self.assertEqual(fail.call_args.args[3], ("S",))
        self.assertNotIn(("release",), m.calls)


if __name__ == "__main__":
    unittest.main()
