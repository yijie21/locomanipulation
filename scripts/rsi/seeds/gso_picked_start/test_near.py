"""held(X) and near(X, T, d) (near.py): `plan_for` sends held(X) with exactly one near(X, T, d) of the same X, upright(X)
or not, to `near.bring_near` over the PLAN_HELD pick; the end stance puts the carried object's predicted xy within
d - NEAR_MARGIN_M of T and off the footprint of a surface whose top is above its underside, nearest the start first;
a failed hand-off check keeps holding (the put-down's sets the object down); with no end stance on the skill's map the
search runs again on a narrower map, its route kept only beside low hulls and slabs; the pull-back with `follow` goes on
when the object came along. Run with `make test TESTS="src/locomani/tasks/scene/skills/test_near.py"`."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import locomani.tasks.scene.skills as skills
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.primitives.navigator import build
from locomani.tasks.scene.skills import near, pick, place
from locomani.tasks.scene.skills.test_plan_for import Ctx, World, yaw_quat


class PlanFor(unittest.TestCase):
    def test_held_and_one_near_of_the_same_object_go_to_bring_near_over_the_plan_held_pick(self):
        with mock.patch.object(near, "bring_near", return_value=Outcome(True)) as b:
            skills.plan_for(SimpleNamespace(goal=["held(Can)", "near(Can, T, 1.0)"]))("ctx")
            skills.plan_for(SimpleNamespace(goal=["near(Can, Book, 0.8)", "upright(Can)", "held(Can)"]))("ctx")
            with mock.patch.object(skills, "PLAN_HELD", "pick_simple"):
                skills.plan_for(SimpleNamespace(goal=["held(Can)", "near(Can, T, 1.0)"]))("ctx")
        self.assertEqual(b.call_args_list, [mock.call("ctx", "Can", "T", 1.0, pick.pick, False),
                                            mock.call("ctx", "Can", "Book", 0.8, pick.pick, True),
                                            mock.call("ctx", "Can", "T", 1.0, pick.pick_simple, False)])  # fmt: skip
        for goal in (
            ["near(Can, T, 1.0)"],
            ["held(Can)", "near(Book, T, 1.0)"],
            ["held(Can)", "near(Can, T, 1.0)", "near(Can, S, 1.0)"],
            ["held(Can)", "near(Can, T, 1.0)", "on(Can, S)"],
        ):
            self.assertIsNone(skills.plan_for(SimpleNamespace(goal=goal)), goal)


def carried(bottom: float, pel=(0.0, -1.5), yaw=np.pi / 2, off=(0.45, -0.1)):
    """The can held at `off` (pelvis yaw frame) from a pelvis at `pel` facing `yaw`, its underside at `bottom`."""
    xy = np.asarray(pel) + R.rot_z(yaw)[:2, :2] @ np.asarray(off)
    world = World({"Can": ([xy[0], xy[1], bottom + 0.09], yaw_quat(yaw))})
    m = SimpleNamespace(sim=SimpleNamespace(d=SimpleNamespace(qpos=np.r_[pel, 0.78])), yaw=lambda: yaw)
    return world, m


class EndStance(unittest.TestCase):
    def test_barred_is_the_grown_footprint_of_a_surface_above_the_underside_less_the_gap(self):
        world, _ = carried(0.72)
        self.assertEqual(near.barred(world, np.array([0.44, 0.0]), 0.72), ["T"])  # 4 cm off the table's +x side
        self.assertEqual(near.barred(world, np.array([0.46, 0.0]), 0.72), [])
        self.assertEqual(near.barred(world, np.array([0.0, 0.0]), 0.76), [])  # table top 0.70 is 6 cm below

    def test_the_predicted_object_is_near_the_target_off_any_slab_it_would_hit_nearest_the_start_first(self):
        stances, start = {}, np.array([-1.5, 0.0])  # from -x: d - margin 0.42 m ends inside the table's grown 0.45 m
        with mock.patch.dict(pick._NAV, {}, clear=True):
            for bottom in (0.72, 0.90):
                world, m = carried(bottom, pel=start, yaw=0.0)
                end, info = near.plan_end(world, m, "Can", "T", 0.57)
                self.assertIsNotNone(end, info)
                off = info["off"]
                np.testing.assert_allclose(off, [0.45, -0.1], atol=1e-9)
                o = end["stance"] + R.rot_z(end["face"])[:2, :2] @ off
                self.assertAlmostEqual(end["dist"], float(np.linalg.norm(o)), places=9)
                self.assertLessEqual(end["dist"], 0.57 - near.NEAR_MARGIN_M)
                self.assertEqual(near.barred(world, o, bottom), [])
                occ, _, _ = pick.nav_map(world)
                self.assertTrue(occ.is_free(end["stance"]))
                np.testing.assert_allclose(end["corners"][0], start)
                np.testing.assert_allclose(end["corners"][-1], end["stance"])
                stances[bottom] = end["stance"], o
            world, m = carried(0.72)
            end, info = near.plan_end(world, m, "Can", "T", 0.2)  # only over the table: barred at 0.72
            self.assertIsNone(end)
            self.assertEqual(info["n_clear"], 0)
        self.assertEqual(near.barred(world, stances[0.90][1], 0.72), ["T"])  # the 0.90 m underside's nearest: over T
        self.assertLess(np.linalg.norm(stances[0.90][0] - start), np.linalg.norm(stances[0.72][0] - start))


def gate_room(bar_top: float):
    """The can carried north from (0, -1), facing +y; a seat (top 0.45 m) against the north wall behind two bars (top
    `bar_top`) across the room at y 1.45-1.75 that leave a 0.72 m gap at x = 0 (0.65 m as the map marks it): closed at
    body radius 0.35, open at 0.28. The bars' and the seat's hulls end 2 cm below their tops."""
    world, m = carried(0.72, pel=(0.0, -1.0), off=(0.45, -0.1))
    world.meta["objects"] = {"Can": world.meta["objects"]["Can"]}
    surface = dict(yaw=0.0, approach_edges=[], room="r")
    world.meta["surfaces"] = {
        "Seat": dict(surface, xy=[0.0, 2.75], half_extent=[0.4, 0.2], flat_half_extent=[0.4, 0.2], top=0.45),
        "BarW": dict(surface, xy=[-1.68, 1.6], half_extent=[1.32, 0.15], top=bar_top),
        "BarE": dict(surface, xy=[1.68, 1.6], half_extent=[1.32, 0.15], top=bar_top),
    }
    boxes = [(s["xy"], s["half_extent"], s["top"] - 0.02) for s in world.meta["surfaces"].values()]
    hulls = [(np.asarray(c) + np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]]), z)
             for c, h, z in boxes]  # fmt: skip
    return world, m, hulls


class Narrow(unittest.TestCase):
    def test_tops_within_the_radius_are_the_highest_hull_or_slab_and_inf_beside_a_wall(self):
        world, _, hulls = gate_room(0.2)
        occ, _ = build(world.meta, body_radius=near.NARROW_RADIUS_M)
        with mock.patch.object(near, "hulls", return_value=hulls):
            tops = near.tops_within(world, occ, 0.35)
        at = lambda xy: float(tops[tuple(occ.cell_of(xy))])  # noqa: E731
        self.assertEqual(at([0.0, 2.25]), 0.45)  # 0.2 m south of the seat, 0.5 m from the bars' ends
        self.assertEqual(at([-1.0, 1.2]), 0.2)  # 0.25 m south of a bar
        self.assertEqual(at([-1.0, 0.0]), -np.inf)  # nothing within 0.35 m
        self.assertEqual(at([-2.8, 0.0]), np.inf)  # the room's border wall
        wide, _ = build(world.meta, body_radius=0.35)
        seen = {}
        self.assertTrue(near.low_route([np.array([0.0, -1.0]), np.array([0.0, 2.2])], wide, tops, seen))
        self.assertEqual(seen["top"], 0.45)
        self.assertTrue(near.low_route([np.array([-2.0, -1.0]), np.array([2.0, -1.0])], wide, tops, seen))
        self.assertEqual(seen["top"], 0.0)  # free on the wide map all along: nothing checked
        tops[tuple(occ.cell_of([0.0, 1.6]))] = near.NARROW_TOP_MAX_M + 0.01
        self.assertFalse(near.low_route([np.array([0.0, -1.0]), np.array([0.0, 2.2])], wide, tops, seen))

    def test_no_end_stance_on_the_skills_map_plans_again_narrower_beside_low_bars_only(self):
        for bar_top, narrow in ((0.2, True), (0.9, False)):
            world, m, hulls = gate_room(bar_top)
            with mock.patch.dict(pick._NAV, {}, clear=True), mock.patch.dict(near._NARROW, {}, clear=True), \
                    mock.patch.object(near, "hulls", return_value=hulls):  # fmt: skip
                wide, info = near.plan_end(world, m, "Can", "Seat", 0.8)
                self.assertIsNone(wide)
                end, info = near.plan_stance(world, m, "Can", "Seat", 0.8)
                closed = end is not None and not pick.nav_map(world)[0].is_free(end["stance"])
            if not narrow:  # 0.9 m bars: every route through the gap is refused
                self.assertIsNone(end)
                self.assertGreater(info["narrow_n"]["n_refused"], 0)
                continue
            self.assertTrue(end["narrow"])
            self.assertEqual(end["top"], 0.2)  # the bars' slabs: the nearest stance stands in the gap
            self.assertLessEqual(end["dist"], 0.8 - near.NARROW_MARGIN_M)
            self.assertGreater(end["dist"], 0.8 - near.NEAR_MARGIN_M)  # the skill's margin would refuse it
            self.assertGreater(end["stance"][1], 1.45)  # in the gap, which the skill's map closes
            self.assertTrue(closed)
            self.assertEqual(info["narrow_n"]["n_refused"], 0)

    def test_an_end_stance_on_the_skills_map_never_builds_the_narrow_one(self):
        end = dict(stance=np.zeros(2), face=0.0, corners=[], dist=0.5)
        with mock.patch.object(near, "plan_end", return_value=(end, {})) as plan, \
                mock.patch.object(near, "narrow_map") as narrow:  # fmt: skip
            self.assertIs(near.plan_stance("world", "m", "Can", "T", 0.8, start="s")[0], end)
        plan.assert_called_once_with("world", "m", "Can", "T", 0.8, "s")
        narrow.assert_not_called()


class PullBack(unittest.TestCase):
    """`place._handoff` after a low pick: a pull-back that ends outside the reach's tolerance is stuck, unless `follow`
    and the object kept its distance to the jaw (within PULL_FOLLOW_M) in a gripping hand."""

    def run_pull(self, follow: bool, obj_moves: float, grips=True):
        jaw = [np.array([0.5, 0.0, 0.6])]
        obj = [np.array([0.52, 0.0, 0.6])]

        def reach(side, target, rot):
            jaw[0] = np.array([0.33, 0.0, 0.6])  # 17 cm of the 20 back: 3 cm off
            obj[0] = obj[0] - [obj_moves, 0.0, 0.0]
            return Outcome(False, "jaw centre 3.0 cm off", 1.0, 2.0)

        world = World()
        world.free = {"Can": 7}
        world.object_pose = lambda key: (obj[0], np.array([1.0, 0.0, 0.0, 0.0]))
        m = SimpleNamespace(t=1.0, fell_at=None, yaw=lambda: 0.0, jaw_centre=lambda side: (jaw[0], np.eye(3)), reach=reach,
                            gripping=lambda body: {"right"} if grips else set(), ref=lambda t: SimpleNamespace(
                                quat=R.quat_wxyz(R.rot_y(np.radians(30.0)))),  # fmt: skip
                            posture=lambda: Outcome(True, "", 2.0, 3.0), trace=[], sim=SimpleNamespace(d=SimpleNamespace(
                                qpos=np.r_[0.0, 0.0, 0.78])), sidestep=lambda forward: Outcome(True, "", 3.0, 4.0))  # fmt: skip
        ctx = Ctx(world, m)
        with mock.patch.object(place, "_grip_checks", return_value=None):
            out = place._handoff(
                ctx, "Can", "right", ("S",), dict(top=True, t_est=20.0), None, near._keep, follow=follow
            )
        return out, ctx.log

    def test_a_pull_back_the_object_followed_goes_on_to_the_stand_only_with_follow(self):
        out, log = self.run_pull(True, 0.17)
        self.assertIsNone(out)
        self.assertEqual([r["step"] for r in log], ["pull_back", "stand", "back_off"])
        self.assertEqual((log[0]["followed"], log[0]["gap_change_cm"]), (True, 0.0))
        out, log = self.run_pull(False, 0.17)
        self.assertEqual(out.why, "pick failed: stuck after the lift (jaw centre 3.0 cm off)")
        self.assertNotIn("followed", log[0])

    def test_with_follow_a_pull_back_that_left_the_object_or_lost_the_grip_is_stuck(self):
        out, log = self.run_pull(True, 0.12)  # 5 cm farther from the jaw
        self.assertEqual(out.why, "pick failed: stuck after the lift (jaw centre 3.0 cm off)")
        self.assertEqual((log[0]["followed"], log[0]["gap_change_cm"]), (False, 5.0))
        out, log = self.run_pull(True, 0.17, grips=False)
        self.assertEqual(out.why, "pick failed: stuck after the lift (jaw centre 3.0 cm off)")


class HandOff(unittest.TestCase):
    def test_a_canister_held_at_the_fingertips_is_kept_in_the_hand_the_put_down_sets_it_down(self):
        world = World({"Can": ([2.06, 2.0, 0.60], yaw_quat(0.0))})  # 6 cm out along the wrist x from the jaw centre
        m = SimpleNamespace(t=5.0, jaw_centre=lambda side: (np.array([2.0, 2.0, 0.6]), np.eye(3)))
        ahead = dict(top=False, t_est=30.0)
        with mock.patch.object(place, "set_down", return_value="set down on S") as down:
            m.touching = lambda body: {"right"}
            kept = place._grip_checks(Ctx(world, m), "Can", "right", ("S",), ahead, None, near._keep)
            self.assertEqual((kept.ok, kept.why), (False, "pick failed: held at the fingertips (6.0 cm out)"))
            down.assert_not_called()
            m.fell_at = None
            world.free = {"Can": 7}
            put = place._grip_checks(Ctx(world, m), "Can", "right", ("S",), ahead, None)
            self.assertEqual(put.why, "pick failed: held at the fingertips (6.0 cm out) [set down on S]")
            down.assert_called_once()


if __name__ == "__main__":
    unittest.main()
