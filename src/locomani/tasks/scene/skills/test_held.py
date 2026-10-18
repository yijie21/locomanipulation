"""held(X) after the pick (held.py, cover.py) -- the predicate's resting rule, nothing added when X is under no other
surface, a level pull-out and a back-off from under a shelf, a side grasp levelled about X's centre for upright(X) --
and the pick's gripper shape (pick.py) -- the least jaw height level and pitched, a pitch only for a short object under
no other surface, the level grasp's targets unchanged, the covered via -- and its retry after a failed attempt. The
fakes are test_plan_for.py's; run with
`make test TESTS="src/locomani/tasks/scene/skills/test_held.py"`."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome, expmap
from locomani.primitives.navigator import build
from locomani.tasks.scene.skills import clearance, cover, grasp_program, held, pick
from locomani.tasks.scene.skills.test_plan_for import Ctx, Hand, World, yaw_quat


class Gripper(Hand):
    """A Hand holding the Can rigidly in the jaw (the object follows every reach and sidestep), facing `yaw`."""

    def __init__(self, world, jaw, rot, yaw: float = 0.0):
        super().__init__(jaw, rot)
        self.world, self._yaw = world, yaw

    def gripping(self, body):
        return {"right"}

    def yaw(self):
        return self._yaw

    def _carry(self, jaw, rot):
        pos, quat = self.world.object_pose("Can")
        rel = self.rot.T @ (pos - self.jaw)
        self.world.poses["Can"] = (jaw + rot @ rel, R.quat_wxyz(rot @ self.rot.T @ R.mat(quat)))
        self.jaw, self.rot = np.asarray(jaw, float), np.asarray(rot, float)

    def reach(self, side, pos, rot=None, **kw):
        self.calls.append(("reach", np.asarray(pos, float), None if rot is None else np.asarray(rot, float)))
        self._carry(np.asarray(pos, float), self.rot if rot is None else np.asarray(rot, float))
        return Outcome(True)

    def sidestep(self, left=0.0, forward=0.0):
        self.calls.append(("sidestep", forward))
        self._carry(self.jaw + forward * np.array([np.cos(self._yaw), np.sin(self._yaw), 0.0]), self.rot)
        return Outcome(True)


class HeldAfterPick(unittest.TestCase):
    """held(X) after the pick (held.py): the predicate's resting rule counts a shelf above X; a pick leaving X under no
    other surface is returned untouched; under a shelf the jaw pulls back level, then the robot backs off if X is still
    under it; upright(X) levels a side-grasped X about its centre."""

    def _world(self, pos, quat=None, upper=True):
        """The Can at `pos` (the shelf S: centre (2, 2), half 0.2 m, top 0.4 m), with a shelf U over S (top 0.9 m)."""
        world = World({"Can": (pos, quat if quat is not None else yaw_quat(0.0))})
        world.free = {"Can": 7}
        if upper:
            world.meta["surfaces"]["U"] = dict(xy=[2.0, 2.0], half_extent=[0.2, 0.2], yaw=0.0, top=0.9)
        return world

    def test_resting_on_counts_the_support_and_any_shelf_above_overhead_only_the_shelf_above(self):
        world = self._world([2.0, 2.0, 0.49])  # underside 0.40 on S, U above
        self.assertEqual((cover.resting_on(world, "Can"), cover.overhead(world, "Can")), (["S", "U"], ["U"]))
        world = self._world([2.0, 2.0, 0.59])  # lifted 10 cm
        self.assertEqual((cover.resting_on(world, "Can"), cover.overhead(world, "Can")), (["U"], ["U"]))
        world = self._world([1.75, 2.0, 0.59])  # centre 5 cm outside the shelves' footprint
        self.assertEqual(cover.resting_on(world, "Can"), [])
        world = self._world([2.0, 2.0, 0.495 + cover.RESTING_GAP_M], upper=False)  # underside 2.5 cm above S
        self.assertEqual(cover.resting_on(world, "Can"), [])

    def _run(self, world, m, upright=False):
        ctx = Ctx(world, m)
        pick_out = Outcome(True, "", 1.0, 2.0)
        return held.pick_held(ctx, "Can", lambda c, k: pick_out, upright), pick_out, ctx

    def test_a_pick_leaving_the_object_under_no_other_surface_is_returned_untouched(self):
        world = self._world([2.0, 2.0, 0.59], upper=False)
        m = Gripper(world, [1.98, 2.0, 0.59], np.eye(3))
        out, pick_out, ctx = self._run(world, m)
        self.assertIs(out, pick_out)
        self.assertEqual((m.calls, ctx.log), ([], []))

    def test_under_a_shelf_the_jaw_pulls_back_level_and_backs_off_only_while_still_under(self):
        world = self._world([1.95, 2.0, 0.59])
        m = Gripper(world, [1.93, 2.0, 0.59], np.eye(3))
        out, _, ctx = self._run(world, m)
        self.assertTrue(out.ok, out.why)
        self.assertEqual([c[0] for c in m.calls], ["reach"])
        np.testing.assert_allclose(m.calls[0][1], [1.93 - held.WITHDRAW_PULL_M, 2.0, 0.59])
        np.testing.assert_allclose(m.calls[0][2], np.eye(3))
        self.assertEqual(ctx.log[0]["under"], [])

        world = self._world([2.15, 2.0, 0.59])  # the pull leaves it 0.05 m inside U's edge
        m = Gripper(world, [2.13, 2.0, 0.59], np.eye(3))
        out, _, ctx = self._run(world, m)
        self.assertTrue(out.ok, out.why)
        self.assertEqual([c[0] for c in m.calls], ["reach", "sidestep"])
        self.assertEqual(m.calls[1][1], -held.WITHDRAW_BACK_M)
        self.assertEqual([r["under"] for r in ctx.log], [["U"], []])

        world = self._world([2.15, 2.0, 0.59])
        m = Gripper(world, [2.13, 2.0, 0.59], np.eye(3))
        m.sidestep = lambda **kw: Outcome(True)  # the back-off goes nowhere
        out, _, _ = self._run(world, m)
        self.assertFalse(out.ok)
        self.assertIn("resting on ['U']", out.why)

    def test_a_pick_with_its_own_pull_pulls_that_far_and_on_to_its_max_only_while_still_under(self):
        pick_out = Outcome(True, "", 1.0, 2.0, dict(withdraw=dict(pull_m=0.20, max_m=0.30)))
        for x, steps in ((1.99, [("pull_out", [], 1.0, 20.0)]),  # out, 1 cm past the edge: no second pull
                         (2.05, [("pull_out", ["U"], -5.0, 20.0), ("pull_on", [], 5.0, 30.0)])):  # fmt: skip
            world = self._world([x, 2.0, 0.59])  # S's and U's front edge, facing +x, at x 1.8
            m = Gripper(world, [x - 0.02, 2.0, 0.59], np.eye(3))
            ctx = Ctx(world, m)
            out = held.pick_held(ctx, "Can", lambda c, k: pick_out)
            self.assertTrue(out.ok, out.why)
            self.assertEqual([(r["step"], r["under"], r["past_edge_cm"], r["jaw_back_cm"]) for r in ctx.log], steps)
            pulls = [x - 0.02 - d for d in (0.20, 0.30)[: len(steps)]]  # both from the jaw where the pull began
            np.testing.assert_allclose([c[1] for c in m.calls], [[p, 2.0, 0.59] for p in pulls])

    def test_past_edge_is_outside_the_edge_a_robot_faces_least_over_the_surfaces(self):
        world = self._world([1.75, 2.0, 0.59])
        self.assertAlmostEqual(cover.past_edge(world, "Can", ["S", "U"], 0.0), 0.05)
        world.meta["surfaces"]["U"]["xy"] = [1.98, 2.0]  # U's front edge 2 cm farther out
        self.assertAlmostEqual(cover.past_edge(world, "Can", ["S", "U"], 0.0), 0.03)
        # the display shelves' frame: turned 90 deg, approached from its -y edge by a robot facing -x
        world.meta["surfaces"]["D"] = dict(xy=[0.0, 0.0], half_extent=[0.36, 0.177], yaw=np.pi / 2, top=0.4)
        world.poses["Can"] = ([0.20, 0.1, 0.5], yaw_quat(0.0))
        self.assertAlmostEqual(cover.past_edge(world, "Can", ["D"], np.pi - 0.03), 0.023)

    def test_only_a_covered_object_whose_one_fitting_side_is_narrow_takes_the_narrow_grasp(self):
        book, wallet, can = [0.0831, 0.0233, 0.1196], [0.054, 0.0434, 0.0138], [0.03, 0.03, 0.09]
        for half, upper, want in ((book, True, True), (book, False, False), (wallet, True, False), (can, True, False)):
            world = self._world([2.0, 2.0, 0.49], upper=upper)
            world.meta["objects"]["Can"]["box"]["half_extent"] = half
            self.assertEqual(pick.narrow_covered(world, "Can"), want, (half, upper))

    def test_upright_levels_a_side_grasp_about_the_object_centre(self):
        tip = expmap([np.radians(12.0), 0.0, 0.0])
        pos = np.array([0.1, -0.1, 0.95])
        rot = tip @ R.rot_z(np.pi / 2)
        world = self._world(pos, R.quat_wxyz(tip), upper=False)
        m = Gripper(world, pos - rot @ [0.02, 0.0, 0.0], rot)
        out, _, ctx = self._run(world, m, upright=True)
        self.assertTrue(out.ok, out.why)
        self.assertEqual(([c[0] for c in m.calls], ctx.log[0]["deg"]), (["reach"], 12.0))
        new_pos, new_quat = world.object_pose("Can")
        np.testing.assert_allclose(new_pos, pos, atol=1e-9)
        self.assertAlmostEqual(clearance.lean_deg(new_quat, yaw_quat(0.0)), 0.0, places=6)


class GripperShape(unittest.TestCase):
    """A side grasp whose level housing would reach below the support top pitches nose-down GRASP_PITCH_DEG with the
    jaw at least min_jaw above the top (pick.grasp_pitch); an object under another surface stays level and its reach
    rises COVERED_VIA_OUT_M outside the edge (pick.edge_via)."""

    def _world(self, half_z, h, upper=False):
        """The Can (box half height `half_z`) with its centre `h` above the shelf S (top 0.4 m), shelf U over S if
        `upper` (top 0.9 m)."""
        world = World({"Can": ([2.0, 2.0, 0.4 + h], yaw_quat(0.0))})
        world.meta["objects"]["Can"]["box"]["half_extent"] = [0.03, 0.03, half_z]
        if upper:
            world.meta["surfaces"]["U"] = dict(xy=[2.0, 2.0], half_extent=[0.2, 0.2], yaw=0.0, top=0.9)
        return world

    def test_min_jaw_is_3_9_cm_level_and_2_76_cm_at_15_deg_5_0_level_with_the_wrist(self):
        self.assertAlmostEqual(pick.min_jaw(0.0), 0.039, places=6)
        self.assertAlmostEqual(pick.min_jaw(np.radians(15.0)), 0.0276, places=4)
        wrist = pick.low_points(pick.WRIST_OVER_M)
        self.assertEqual(wrist, (*pick.GRIPPER_LOW_POINTS, pick.WRIST_LOW_POINT))
        self.assertAlmostEqual(pick.min_jaw(0.0, wrist), 0.050, places=6)
        self.assertEqual(pick.min_jaw(np.radians(15.0), wrist), pick.min_jaw(np.radians(15.0)))  # the fingertip's
        self.assertAlmostEqual(pick.min_jaw(np.radians(15.0)), 0.0276, places=4)

    def test_only_a_short_object_under_no_other_surface_is_pitched(self):
        pitch = np.radians(pick.GRASP_PITCH_DEG)
        for half_z, h, upper, want in (
            (0.0251, 0.0251, False, (pitch, pick.min_jaw(pitch) - 0.0251)),  # the candles: raised 0.25 cm
            (0.0264, 0.0264, False, (pitch, pick.min_jaw(pitch) - 0.0264)),  # the supply bin
            (0.0794, 0.0794, False, (0.0, 0.0)),  # the coffee can
            (0.0457, 0.0457, False, (0.0, 0.0)),  # the mug
            (0.0475, 0.0475, True, (0.0, 0.0)),  # the wolf, under the corner display's top shelf
            (0.0367, 0.0367, True, (0.0, 0.0)),  # the shelf plant, under the middle shelf
            (0.0251, 0.0251, True, (0.0, 0.0)),  # a candle under a shelf stays level
        ):
            world = self._world(half_z, h, upper)
            got = pick.grasp_pitch(world, "Can", world.object_pose("Can")[0])
            np.testing.assert_allclose(got, want, atol=1e-12, err_msg=str((half_z, upper)))
        world = self._world(0.03, 0.03)  # pitched, the centre already above min_jaw(pitch): no rise
        self.assertEqual(pick.grasp_pitch(world, "Can", world.object_pose("Can")[0]), (pitch, 0.0))

    def test_the_apple_is_pitched_only_where_its_wrist_lies_over_the_support(self):
        pitch = np.radians(pick.GRASP_PITCH_DEG)
        world = self._world(0.0413, 0.0405)  # the apple, centre 4.05 cm above the dining table
        pos = world.object_pose("Can")[0]
        for wrist, want in (
            (None, (0.0, 0.0)),
            (pick.WRIST_OVER_M - 1e-3, (0.0, 0.0)),
            (pick.WRIST_OVER_M, (pitch, 0.0)),
        ):
            self.assertEqual(pick.grasp_pitch(world, "Can", pos, wrist), want, msg=str(wrist))
        world = self._world(0.0794, 0.0794)  # the coffee can stays level with its wrist over the table
        self.assertEqual(pick.grasp_pitch(world, "Can", world.object_pose("Can")[0], 0.2), (0.0, 0.0))

    def test_the_wrist_lies_its_distance_behind_the_jaw_along_the_approach_in_from_the_edge(self):
        x = pick.WRIST_LOW_POINT[0]
        self.assertAlmostEqual(pick.wrist_in(dict(edge_depth_m=0.228, psi=1.6, face=1.6)), 0.228 + x)
        self.assertAlmostEqual(pick.wrist_in(dict(edge_depth_m=0.10, psi=0.0, face=np.pi / 3)), 0.10 + 0.5 * x)
        self.assertIsNone(pick.wrist_in(dict(psi=0.0, face=0.0)))

    def _pick(self, world, **info):
        """pick.pick with the walk, the via and the hand program stubbed: (grasp_program.run's args, the reaches)."""
        m = Hand([1.7, 2.0, 0.45], np.eye(3))
        plan = Outcome(True, info=dict(side="right", psi=0.3, face=0.0, edge_xy=np.array([1.8, 2.0]), **info))
        with (
            mock.patch.object(pick, "approach", return_value=plan),
            mock.patch.object(pick, "edge_via", return_value=()),
            mock.patch.object(grasp_program, "run", return_value=Outcome(True)) as run,
        ):
            pick.pick(Ctx(world, m), "Can")
        return run.call_args.args, m.calls

    def test_a_level_grasp_keeps_its_targets_and_a_pitched_one_turns_and_rises(self):
        world = self._world(0.0794, 0.0794)
        (_, _, _, Rg, tool, _), calls = self._pick(world)
        pos = world.object_pose("Can")[0]
        np.testing.assert_array_equal(Rg, R.rot_z(0.3))
        np.testing.assert_array_equal(tool, pos)
        np.testing.assert_array_equal(calls[0][1], pos - R.rot_z(0.3) @ np.array([grasp_program.PREGRASP_BACK, 0, 0]))

        world = self._world(0.0251, 0.0251)
        (_, _, _, Rg, tool, _), calls = self._pick(world)
        pitch = np.radians(pick.GRASP_PITCH_DEG)
        want_R = R.rot_z(0.3) @ R.rot_y(pitch)
        np.testing.assert_allclose(Rg, want_R, atol=1e-12)
        self.assertLess(Rg[2, 0], 0.0)  # the approach points down
        np.testing.assert_allclose(tool, [2.0, 2.0, 0.4 + pick.min_jaw(pitch)], atol=1e-12)
        np.testing.assert_allclose(calls[0][1], tool - want_R @ [grasp_program.PREGRASP_BACK, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(calls[0][2], want_R, atol=1e-12)

    def test_the_pick_pitches_the_apple_from_the_approach_depth_of_its_wrist(self):
        world = self._world(0.0413, 0.0405)
        (_, _, _, Rg, _, _), _ = self._pick(world, edge_depth_m=0.25)  # the wrist 0.11 m in from the edge (psi 0.3)
        np.testing.assert_allclose(Rg, R.rot_z(0.3) @ R.rot_y(np.radians(pick.GRASP_PITCH_DEG)), atol=1e-12)
        (_, _, _, Rg, _, _), _ = self._pick(world, edge_depth_m=0.15)  # 0.007 m in: off the slab, level
        np.testing.assert_array_equal(Rg, R.rot_z(0.3))

    def test_the_via_rises_farther_outside_the_edge_for_a_covered_object(self):
        plan = Outcome(True, info=dict(face=0.0, edge_xy=np.array([1.8, 2.0])))
        for upper, out in ((False, pick.VIA_OUT_M), (True, pick.COVERED_VIA_OUT_M)):
            world = self._world(0.0475, 0.0475, upper)
            m = Hand([1.75, 2.0, 0.38], np.eye(3))  # jaw below the top, 5 cm outside the edge
            (via,) = pick.edge_via(Ctx(world, m), "Can", "right", plan, np.array([1.9, 2.0, 0.45]))
            np.testing.assert_allclose(via, [1.8 - out, 2.0, 0.45], atol=1e-12, err_msg=str(upper))


class LowGrasp(unittest.TestCase):
    """A level side grasp whose pads would meet a hull wider than LOW_SPAN_M closes at the highest jaw below the centre
    where the collision hull spans at most LOW_WIDTH_M over the pads' height (pick.low_grasp, the round vase), and when
    its straight approach sweeps an object fixed on the support it comes in CLEAR_YAW_DEG turned away with the far
    hand (pick.clear_pass); every other grasp and plan is unchanged."""

    _world, _pick = GripperShape._world, GripperShape._pick
    A, C = 0.046, 0.08  # the vase's equator radius and half height

    def _vase(self, top=0.4):
        """The Can as a 9.2 x 9.2 x 16 cm box on the shelf S (top `top`), its hull an ellipsoid of rings: (world,
        verts)."""
        world = self._world(self.C, self.C)
        world.meta["objects"]["Can"]["box"]["half_extent"] = [self.A, self.A, self.C]
        world.meta["surfaces"]["S"]["top"] = top
        world.free, world.motor = {"Can": 7}, None
        t, a = np.linspace(-1.0, 1.0, 33), np.radians(np.arange(0.0, 360.0, 15.0))
        ring = [(self.A * np.sqrt(1 - u * u) * np.cos(b), self.A * np.sqrt(1 - u * u) * np.sin(b), self.C * u)
                for u in t for b in a]  # fmt: skip
        return world, world.object_pose("Can")[0] + np.asarray(ring)

    def test_band_width_is_the_hull_extent_between_the_planes(self):
        r, h = 0.05, 0.1
        diamond = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, h], [0, 0, -h]])
        self.assertAlmostEqual(pick.band_width(diamond, -0.5 * h, -0.25 * h, [1.0, 0.0, 0.0]), 1.5 * r)
        self.assertAlmostEqual(pick.band_width(diamond, -0.1 * h, 0.1 * h, [0.0, 1.0, 0.0]), 2 * r)
        self.assertEqual(pick.band_width(diamond, 1.1 * h, 1.2 * h, [1.0, 0.0, 0.0]), 0.0)

    def test_the_vase_closes_at_the_highest_jaw_whose_pads_span_at_most_low_width(self):
        world, verts = self._vase()
        pos, quat = world.object_pose("Can")
        across = R.rot_z(0.3)[:, 1]
        with mock.patch.object(pick, "object_vertices", return_value=verts):
            pitch, rise, width = pick.low_grasp(world, "Can", pos, quat, 0.3)
            self.assertEqual(pitch, np.radians(pick.LOW_PITCH_DEG))
            exact = -self.C * np.sqrt(1 - (pick.LOW_WIDTH_M / (2 * self.A)) ** 2) - grasp_program.PAD_Z  # the ellipsoid
            self.assertTrue(exact - pick.LOW_STEP_M - 0.004 < rise < exact + 0.004, (rise, exact))
            self.assertLessEqual(width, pick.LOW_WIDTH_M)
            z = pos[2] + rise + pick.LOW_STEP_M  # one step up the pads span more
            self.assertGreater(pick.band_width(verts, z - grasp_program.PAD_Z, z + grasp_program.PAD_Z, across),
                               pick.LOW_WIDTH_M)  # fmt: skip
            self.assertGreaterEqual(pos[2] + rise, 0.4 + pick.min_jaw(pitch))
            high, verts = self._vase(top=0.4 + 0.07)  # the gripper would reach below the top: none
            self.assertIsNone(pick.low_grasp(high, "Can", pos, quat, 0.3))
            world.meta["surfaces"]["U"] = dict(xy=[2.0, 2.0], half_extent=[0.2, 0.2], yaw=0.0, top=0.9)
            self.assertIsNone(pick.low_grasp(world, "Can", pos, quat, 0.3))  # under a shelf: none
        world = self._world(0.0794, 0.0794)  # the coffee can's 6 cm: none, its mesh never read
        self.assertIsNone(pick.low_grasp(world, "Can", *world.object_pose("Can"), 0.3))
        world, verts = self._vase()  # a round can whose 6.7 cm box spans 9.5 cm at 45 deg: its hull, 6.7 cm, fits
        world.meta["objects"]["Can"]["box"]["half_extent"] = [0.0335, 0.0335, self.C]
        can = pos + (verts - pos) * np.array([0.0335 / self.A, 0.0335 / self.A, 1.0])
        with mock.patch.object(pick, "object_vertices", return_value=can):
            self.assertGreater(pick.span(quat, [0.0335, 0.0335], np.pi / 4), pick.LOW_SPAN_M)
            self.assertIsNone(pick.low_grasp(world, "Can", pos, quat, np.pi / 4))

    def test_the_lowest_jaw_the_gripper_allows_closes_when_every_step_above_it_is_too_wide(self):
        world, verts = self._vase()
        pos, quat = world.object_pose("Can")
        across = R.rot_z(0.3)[:, 1]

        def width(z):
            return pick.band_width(verts, z - grasp_program.PAD_Z, z + grasp_program.PAD_Z, across)

        with mock.patch.object(pick, "object_vertices", return_value=verts):
            _, rise, _ = pick.low_grasp(world, "Can", pos, quat, 0.3)
            step = pos[2] + rise  # the first step that fits; the one above it is too wide
            hi, lo = step + pick.LOW_STEP_M, step
            for _ in range(30):  # the highest jaw that fits, between them
                mid = 0.5 * (hi + lo)
                hi, lo = (hi, mid) if width(mid) <= pick.LOW_WIDTH_M else (mid, lo)
            mj = pick.min_jaw(np.radians(pick.LOW_PITCH_DEG))
            for floor, fits in ((0.5 * (lo + step), True), (0.5 * (hi + step + pick.LOW_STEP_M), False)):
                world.meta["surfaces"]["S"]["top"] = floor - mj  # the steps stop above `step`
                low = pick.low_grasp(world, "Can", pos, quat, 0.3)
                if fits:
                    self.assertAlmostEqual(low[1], floor - pos[2])
                    self.assertLessEqual(low[2], pick.LOW_WIDTH_M)
                else:
                    self.assertIsNone(low)

    def _fixed(self, world, xy, half=0.048):
        """A fixed (not free) round-ish object on the shelf S, its box `half` wide, at `xy`."""
        world.meta["objects"]["Small"] = dict(surface="S", free=False, init_pos=[*xy, 0.4 + half],
                                              init_quat=yaw_quat(0.0),
                                              box=dict(half_extent=[half, half, half]))  # fmt: skip

    def test_the_strip_meets_a_fixed_object_behind_the_jaw_only(self):
        world, _ = self._vase()
        self.assertIsNone(pick.strip_blocker(world, "Can", np.pi / 2))  # nothing fixed on S
        self._fixed(world, (2.06, 1.88))  # 12 cm back toward the robot (approach +y), 6 cm to the right
        self.assertAlmostEqual(pick.strip_blocker(world, "Can", np.pi / 2), -0.06)
        self.assertIsNone(pick.strip_blocker(world, "Can", 0.0))  # approached along +x it is ahead and aside
        world.meta["objects"]["Small"]["free"] = True  # a free neighbour is not a blocker
        self.assertIsNone(pick.strip_blocker(world, "Can", np.pi / 2))

    def test_a_blocked_low_grasp_plans_turned_away_with_the_far_hand_any_other_plan_stands(self):
        world, _ = self._vase()
        plan = (np.pi / 2, 0.2, np.pi / 2, np.array([2.12, 1.6]), np.array([2.0, 1.8]))
        yaw = np.pi / 2 - np.radians(pick.CLEAR_YAW_DEG)
        turned = (np.pi / 2, 0.2, yaw, np.array([2.0, 1.6]), np.array([2.0, 1.8]))
        with (
            mock.patch.object(pick, "low_grasp", return_value=(0.0, -0.03, 0.09)),
            mock.patch.object(pick, "plan_approach", return_value=(turned, "")) as pa,
        ):
            self.assertIsNone(pick.clear_pass(world, np.zeros(2), "Can", plan))  # nothing in the strip
            self._fixed(world, (2.06, 1.88))  # on the approach's right
            self.assertEqual(pick.clear_pass(world, np.zeros(2), "Can", plan), (turned, "left", ""))
            self.assertEqual(pa.call_args.args[3:], ("left", None, True, -np.radians(pick.CLEAR_YAW_DEG)))
            world.meta["objects"]["Small"]["init_pos"][0] = 1.94  # on its left: the right hand, turned the other way
            self.assertEqual(pick.clear_pass(world, np.zeros(2), "Can", plan)[1], "right")
            self.assertEqual(pa.call_args.args[3:], ("right", None, True, np.radians(pick.CLEAR_YAW_DEG)))
        with mock.patch.object(pick, "low_grasp", return_value=None):  # no low grasp: the straight plan stands
            self.assertIsNone(pick.clear_pass(world, np.zeros(2), "Can", plan))

    def test_plan_approach_turns_by_yaw_off_only_a_box_that_fits_both_ways(self):
        world, _ = self._vase()
        occ, _ = build(world.meta, body_radius=0.35)
        world.occ = occ
        with mock.patch.dict(pick._NAV, {}, clear=True):
            plan, why = pick.plan_approach(world, np.array([2.0, 0.0]), "Can", "left", None, True, -0.6)
            self.assertIsNotNone(plan, why)
            self.assertAlmostEqual(plan[2], plan[0] - 0.6)
            world.meta["objects"]["Can"]["box"]["half_extent"] = [self.A, 0.06, self.C]  # one way only: no plan
            self.assertIsNone(pick.plan_approach(world, np.array([2.0, 0.0]), "Can", "left", None, True, -0.6)[0])

    def test_the_pick_enters_the_low_grasp_lowered_and_pitched_across_its_span(self):
        world, _ = self._vase()
        pitch = np.radians(pick.GRASP_PITCH_DEG)
        with mock.patch.object(pick, "low_grasp", return_value=(pitch, -0.05, 0.086)):
            (_, _, _, Rg, tool, width), calls = self._pick(world)
        want_R = R.rot_z(0.3) @ R.rot_y(pitch)
        np.testing.assert_allclose(Rg, want_R, atol=1e-12)
        np.testing.assert_allclose(tool, world.object_pose("Can")[0] - [0.0, 0.0, 0.05], atol=1e-12)
        np.testing.assert_allclose(calls[0][1], tool - want_R @ [grasp_program.PREGRASP_BACK, 0.0, 0.0], atol=1e-12)
        self.assertEqual(width, 0.086)

    def test_a_low_grasp_whose_pre_grasp_ends_off_fails_low_without_the_hand_program(self):
        world, _ = self._vase()
        plan = Outcome(True, info=dict(side="right", psi=0.3, face=0.0, edge_xy=np.array([1.8, 2.0])))
        for off, ran in ((pick.LOW_PREGRASP_MISS_M - 0.005, True), (pick.LOW_PREGRASP_MISS_M + 0.005, False)):
            m = Hand([1.7, 2.0, 0.45], np.eye(3))
            reach = m.reach

            def short(side, pos, rot=None, **kw):  # the pre-grasp ends `off` short along -x
                out = reach(side, pos, rot, **kw)
                m.jaw = m.jaw - [off, 0.0, 0.0]
                return out

            m.reach = short
            with (
                mock.patch.object(pick, "approach", return_value=plan),
                mock.patch.object(pick, "edge_via", return_value=()),
                mock.patch.object(pick, "low_grasp", return_value=(0.0, -0.03, 0.0908)),
                mock.patch.object(grasp_program, "run", return_value=Outcome(True)) as run,
            ):
                out = pick.pick_once(Ctx(world, m), "Can", "right", {})
            self.assertEqual((run.called, out.ok, out.info.get("low")), (ran, ran, True))
            if not ran:
                self.assertTrue(out.why.startswith(f"pregrasp missed by {100 * off:.1f} cm"), out.why)

    def test_a_low_grasp_marks_its_outcome_low_any_other_grasp_keeps_the_hand_programs(self):
        world, _ = self._vase()
        plan = Outcome(True, info=dict(side="right", psi=0.3, face=0.0, edge_xy=np.array([1.8, 2.0])))
        for low, want in (((0.0, -0.03, 0.0908), dict(shift_m=0.01, low=True)), (None, dict(shift_m=0.01))):
            with (
                mock.patch.object(pick, "approach", return_value=plan),
                mock.patch.object(pick, "edge_via", return_value=()),
                mock.patch.object(pick, "low_grasp", return_value=low),
                mock.patch.object(grasp_program, "run", return_value=Outcome(False, "x", 1.0, 2.0, dict(shift_m=0.01))),
            ):
                out = pick.pick_once(Ctx(world, Hand([1.7, 2.0, 0.45], np.eye(3))), "Can", "right", {})
            self.assertEqual((out.ok, out.why, out.t0, out.t1, out.info), (False, "x", 1.0, 2.0, want))


class Pinch(unittest.TestCase):
    """A pitched grasp whose pads would pass over the object (pick.needs_pinch: the car key) pinches it instead
    (pick.pick_pinch): wrist pitched PINCH_PITCH_DEG with the jaw at min_jaw over PINCH_LOW_POINTS, aimed at the part
    the pads can overlap (pick.grip_centre), entering level (grasp_program.run's entry_frame)."""

    TOP = 0.92  # the kitchen counter's height (the shelf S raised): above TOP_SURFACE_MAX_M, so no top-down pinch

    def _world(self, half_z, h):
        world = World({"Can": ([2.0, 2.0, self.TOP + h], yaw_quat(0.0))})
        world.meta["surfaces"]["S"]["top"] = self.TOP
        world.meta["objects"]["Can"]["box"]["half_extent"] = [0.0492, 0.0185, half_z]
        world.free = {"Can": 7}
        return world

    def test_min_jaw_over_the_true_outline_is_finger_limited_with_the_housing_a_cm_higher_at_25_deg(self):
        self.assertEqual(pick.PINCH_PITCH_DEG, 25.0)
        pitch = np.radians(25.0)
        self.assertAlmostEqual(pick.min_jaw(pitch, pick.PINCH_LOW_POINTS), 0.0213, places=4)
        s, c = np.sin(pitch), np.cos(pitch)
        low = {p: p[0] * s - p[1] * c for p in pick.PINCH_LOW_POINTS}
        self.assertEqual(max(low, key=low.get), (0.0321, -0.003))  # the finger's tapered front underside
        self.assertGreater(low[max(low, key=low.get)] - low[(-0.0441, -0.031)], 0.0065)  # the housing corner
        self.assertAlmostEqual(pick.min_jaw(np.radians(20.0), pick.PINCH_LOW_POINTS), 0.02035, places=4)
        self.assertEqual(pick.min_jaw(np.radians(15.0)), pick.min_jaw(np.radians(15.0), pick.GRIPPER_LOW_POINTS))

    def test_only_the_key_is_pinched_not_the_wallet_candles_or_bin(self):
        for half_z, want in ((0.0075, True), (0.0138, False), (0.0251, False), (0.0264, False)):
            world = self._world(half_z, half_z)
            pos = world.object_pose("Can")[0]
            pitch, rise = pick.grasp_pitch(world, "Can", pos)
            self.assertGreater(pitch, 0.0)
            self.assertEqual(pick.needs_pinch(world, "Can", pos, pitch, rise), want, msg=str(half_z))

    def test_grip_centre_is_the_middle_of_the_part_tall_enough_to_overlap(self):
        top = 0.4
        fob = [[x, y, top + z] for x in (-0.049, -0.008) for y in (-0.018, 0.018) for z in (0.0, 0.015)]
        blade = [[x, y, top + z] for x in (0.0, 0.049) for y in (-0.005, 0.005) for z in (0.0, 0.004)]
        verts = np.array(fob + blade) + [2.0, 2.0, 0.0]
        pos = np.array([2.0, 2.0, top + 0.0075])
        d = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(pick.grip_centre(verts, pos, top, d), [2.0 - 0.0285, 2.0], atol=1e-12)
        np.testing.assert_allclose(pick.grip_centre(verts, pos, top, -d), [2.0 - 0.0285, 2.0], atol=1e-12)
        np.testing.assert_array_equal(pick.grip_centre(np.zeros((0, 3)), pos, top, d), pos[:2])

    def test_the_key_is_reached_above_the_entry_line_dropped_onto_it_and_entered_level(self):
        world = self._world(0.0075, 0.0075)
        pos = world.object_pose("Can")[0]
        m = Hand([1.7, 2.0, 0.45], np.eye(3))
        psi = 0.3
        plan = Outcome(True, info=dict(side="right", psi=psi, face=0.0, edge_xy=np.array([1.8, 2.0])))
        centre = pos[:2] + [-0.02, 0.0]
        with (
            mock.patch.object(pick, "approach", return_value=plan),
            mock.patch.object(pick, "edge_via", return_value=()),
            mock.patch.object(pick, "object_vertices", return_value=np.zeros((0, 3))),
            mock.patch.object(pick, "grip_centre", return_value=centre),
            mock.patch.object(grasp_program, "run", return_value=Outcome(True)) as run,
        ):
            pick.pick(Ctx(world, m), "Can")
        (_, _, _, Rg, tool, width), kw = run.call_args
        pitch = np.radians(pick.PINCH_PITCH_DEG)
        level = R.rot_z(psi)
        np.testing.assert_allclose(Rg, level @ R.rot_y(pitch), atol=1e-12)
        np.testing.assert_allclose(kw["entry_frame"], R.rot_y(-pitch), atol=1e-12)
        np.testing.assert_allclose(Rg @ kw["entry_frame"], level, atol=1e-12)  # the entry runs level
        self.assertFalse(pick.is_top(world, "Can"))
        jaw_z = self.TOP + pick.min_jaw(pitch, pick.PINCH_LOW_POINTS, pick.PINCH_CLEAR_M)
        np.testing.assert_allclose(tool, [*(centre - pick.PINCH_X_M * level[:2, 0]), jaw_z], atol=1e-12)
        pre = tool - grasp_program.PREGRASP_BACK * level[:, 0]
        self.assertEqual([c[0] for c in m.calls], ["reach", "reach"])
        np.testing.assert_allclose(m.calls[0][1], pre + [0.0, 0.0, pick.PINCH_DROP_M], atol=1e-12)
        np.testing.assert_allclose(m.calls[1][1], pre, atol=1e-12)
        np.testing.assert_allclose(m.calls[1][2], Rg, atol=1e-12)
        self.assertAlmostEqual(width, pick.span(yaw_quat(0.0), [0.0492, 0.0185, 0.0075], psi))

    def _entry(self, entry_frame):
        """grasp_program.run on a jaw that goes exactly where it is commanded, wrist pitched 20 deg nose-down, the
        grip 0.19 m ahead level: the commanded jaw heights during the entry."""
        body, pitch = 7, np.radians(20.0)
        Rc = R.rot_y(pitch)
        state = dict(jaw=np.array([0.0, 0.0, 1.0]) - 0.14 * (Rc if entry_frame is None else np.eye(3))[:, 0])
        sim = SimpleNamespace(d=SimpleNamespace(xpos={body: np.array([0.19, 0.0, 1.0])}, ncon=0), m=None)
        zs = []

        def hand(side, pos=None, rot=None, v=None):
            if pos is not None:
                state["jaw"] = pos.copy()
                zs.append(float(pos[2]))

        m = SimpleNamespace(sim=sim, servo_on={}, servo={}, fingers={"right": (1, 2)}, jaw=np.zeros(2), hand=hand,
                            jaw_centre=lambda side: (state["jaw"].copy(), Rc), adopt_hand=lambda side: None,
                            commanded_jaw=lambda side: state["jaw"].copy())  # fmt: skip
        m.t = 0.0

        def tick():
            m.t += R.DT
            return True

        m.tick = tick
        ctx = SimpleNamespace(world=SimpleNamespace(free={"Can": body}), motor=m, note=lambda name, out, **kw: out)
        out = grasp_program.run(ctx, "Can", "right", Rc, np.array([0.19, 0.0, 1.0]), 0.037, entry_frame=entry_frame)
        return out, zs

    def test_the_level_entry_keeps_the_jaw_height_and_the_pitched_one_descends(self):
        out, zs = self._entry(R.rot_y(-np.radians(20.0)))
        self.assertEqual(out.why, "no bilateral pad contact after the close")
        np.testing.assert_allclose(zs, 1.0, atol=1e-9)
        out, zs = self._entry(None)
        self.assertGreater(zs[0] - min(zs), 0.03)  # along the pitched wrist x, from 4.8 cm above the grip


class YawedPass(unittest.TestCase):
    """plan_hand's last pass (pick.plan_approach yawed): a side up to FACE_TOLERANCE_YAWED_DEG off the facing, the
    stance placed from the wrist, only where the 40 deg passes plan nothing; at psi = face its stance is today's."""

    def _world(self, half, yaw):
        world = World({"Can": ([0.0, -0.2, 0.79], yaw_quat(yaw))})
        world.meta["objects"]["Can"]["surface"] = "T"  # the 0.7 m table, 0.1 m in from -y (face +90 deg), 0.4 m from x
        world.meta["objects"]["Can"]["box"]["half_extent"] = half
        world.occ, _ = build(world.meta, body_radius=0.35)
        return world

    def test_at_psi_equal_to_the_facing_the_wrist_placed_stance_is_bit_identical(self):
        world = self._world([0.03, 0.03, 0.09], 0.4)  # both sides fit: psi = face
        with mock.patch.dict(pick._NAV, {}, clear=True):
            plain, why = pick.plan_approach(world, np.array([0.0, -2.0]), "Can", "right")
            yawed, _ = pick.plan_approach(world, np.array([0.0, -2.0]), "Can", "right", yawed=True)
        self.assertIsNotNone(plain, why)
        self.assertEqual(plain[2], plain[0])
        np.testing.assert_array_equal(yawed[3], plain[3])

    def test_a_side_58_deg_off_plans_only_in_the_yawed_pass_from_the_wrist(self):
        world = self._world([0.03, 0.06, 0.09], np.radians(58.0))  # narrow side x: psi 148 deg, the face 90
        start = np.array([0.0, -2.0])
        with mock.patch.dict(pick._NAV, {}, clear=True):
            for side in ("right", "left"):
                self.assertIsNone(pick.plan_approach(world, start, "Can", side)[0])
            plan, hand, why = pick.plan_hand(world, start, "Can", "right")
        self.assertEqual((hand, why), ("right", ""))
        face, _, psi, stance, _ = plan
        self.assertAlmostEqual(np.degrees(R.wrap(psi - face)), 58.0)
        pos = np.array([0.0, -0.2])
        wrist = pos - pick.WRIST_JAW_M * np.array([np.cos(psi), np.sin(psi)])
        want = wrist - (R.rot_z(face) @ [pick.STAND_AHEAD_M - pick.WRIST_JAW_M, -pick.STAND_SIDE_M, 0.0])[:2]
        np.testing.assert_allclose(stance, want, atol=1e-12)

    def test_no_yawed_plan_above_yawed_top_max(self):
        world = self._world([0.03, 0.06, 0.09], np.radians(58.0))
        world.meta["surfaces"]["T"]["top"] = pick.YAWED_TOP_MAX_M + 0.12  # the 1.12 m shelf
        with mock.patch.dict(pick._NAV, {}, clear=True):
            plan, hand, why = pick.plan_hand(world, np.array([0.0, -2.0]), "Can", "right")
        self.assertIsNone(plan)
        self.assertEqual(why, "no side fits the jaw within reach of an approachable edge")

    def test_a_plan_of_the_40_deg_pass_is_never_replaced(self):
        world = self._world([0.03, 0.06, 0.09], np.radians(30.0))  # psi 120 deg: 30 deg off
        start = np.array([0.0, -2.0])
        with mock.patch.dict(pick._NAV, {}, clear=True):
            first, _ = pick.plan_approach(world, start, "Can", "right")
            plan, hand, _ = pick.plan_hand(world, start, "Can", "right")
        self.assertEqual(hand, "right")
        for a, b in zip(plan, first):
            np.testing.assert_array_equal(a, b)


class ContinuousClose(unittest.TestCase):
    """grasp_program.run's close (CONTINUOUS_CLOSE): a jaw more than CONTINUOUS_MIN_M behind the entry's last target
    keeps that target through the close and lifts from it; a jaw at its target keeps bfm_grasp's reset, step for step."""

    def _run(self, speed, continuous):
        """A level grasp of an object 0.19 m ahead on a jaw that moves toward its command at most `speed` (m/s, None:
        it is where it was commanded), the pads touching once the jaw is closed: the commanded jaw positions and the
        grasp note."""
        body, Rc = 7, np.eye(3)
        state = dict(jaw=np.array([0.05, 0.0, 1.0]))
        state["cmd"] = state["jaw"].copy()
        sim = SimpleNamespace(d=SimpleNamespace(xpos={body: np.array([0.19, 0.0, 1.0])}, ncon=0), m=None)
        cmds, notes = [], {}

        def hand(side, pos=None, rot=None, v=None):
            if pos is not None:
                state["cmd"] = pos.copy()
                cmds.append(pos.copy())

        m = SimpleNamespace(sim=sim, servo_on={}, servo={}, fingers={"right": (1, 2)}, jaw=np.zeros(2), hand=hand,
                            jaw_centre=lambda side: (state["jaw"].copy(), Rc), adopt_hand=lambda side: None,
                            commanded_jaw=lambda side: state["jaw"].copy())  # fmt: skip
        m.t = 0.0

        def tick():
            step = state["cmd"] - state["jaw"]
            n = float(np.linalg.norm(step))
            if speed is not None and n > speed * R.DT:
                step *= speed * R.DT / n
            state["jaw"] = state["jaw"] + step
            m.t += R.DT
            return True

        def note(name, out, **kw):
            notes.update(kw)
            return out

        m.tick = tick
        ctx = SimpleNamespace(world=SimpleNamespace(free={"Can": body}), motor=m, note=note)
        closed = lambda sim, f: {body} if m.jaw.min() <= grasp_program.JAW_CLOSED + 1e-9 else set()  # noqa: E731
        with (
            mock.patch.object(grasp_program, "_touching", closed),
            mock.patch.object(grasp_program, "CONTINUOUS_CLOSE", continuous),
        ):
            out = grasp_program.run(ctx, "Can", "right", Rc, np.array([0.19, 0.0, 1.0]), 0.037)
        self.assertEqual(notes["phases"], ["settle", "entry", "close", "lift", "recover"], out.why)
        return np.array(cmds), notes

    def test_a_lagging_jaw_keeps_the_entry_target_through_the_close_and_the_lift(self):
        cmds, notes = self._run(0.03, 1)
        self.assertTrue(notes["continuous_close"])
        self.assertGreater(notes["close_jump_m"], grasp_program.CONTINUOUS_MIN_M)
        self.assertLess(np.abs(np.diff(cmds, axis=0)).max(), 0.005)  # no step (the lift peaks at 1.9 mm a tick)
        reset, notes = self._run(0.03, 0)
        self.assertFalse(notes["continuous_close"])
        self.assertGreater(np.abs(np.diff(reset, axis=0)).max(), grasp_program.CONTINUOUS_MIN_M)  # bfm_grasp's reset

    def test_a_jaw_at_its_target_keeps_the_reset_step_for_step(self):
        cmds, notes = self._run(None, 1)
        self.assertFalse(notes["continuous_close"])
        self.assertLessEqual(notes["close_jump_m"], grasp_program.CONTINUOUS_MIN_M)
        reset, _ = self._run(None, 0)
        np.testing.assert_array_equal(cmds, reset)


class Arm(Hand):
    """A Hand on a robot standing at `xy` facing `yaw`, pelvis 0.8 m up: what pick's retry and back_out read (the
    servo integrals, the pelvis, the qpos for the map), the override released, what is between the fingers (`held`)."""

    def __init__(self, jaw, rot, xy=(0.0, 0.0), yaw=0.0):
        super().__init__(jaw, rot)
        self._yaw, self.held = yaw, set()
        self.sim = SimpleNamespace(d=SimpleNamespace(qpos=np.r_[xy, 0.8, R.quat_wxyz(R.rot_z(yaw))]))

    def yaw(self):
        return self._yaw

    def pelvis(self):
        return self.sim.d.qpos[:3].copy(), self.sim.d.qpos[3:7].copy()

    def hand(self, side, *args):
        self.calls.append(("hand", side))

    def between_fingers(self, side):
        return self.held if side == "right" else set()

    def sidestep(self, left=0.0, forward=0.0):
        self.calls.append(("sidestep", forward))
        return Outcome(True)


class Retry(unittest.TestCase):
    """pick's retry (pick.pick over pick_once): an attempt that failed in the hand program, without a fall, X still on
    its own support, under no other surface and between no fingers, backs out and runs once more; a fall, X lifted
    off, gripped or under a shelf, and a planning refusal are not retried. back_out pulls the jaw back level, opens
    it, steps back only onto a point free on the skill's map and then returns the arm to its walking pose; pick_once
    records that pose for side grasps only."""

    def _ctx(self, z=0.49):
        world = World({"Can": ([2.0, 2.0, z], yaw_quat(0.0))})  # on its own shelf S (top 0.4 m) at z 0.49
        world.free = {"Can": 7}
        return Ctx(world, Arm([1.7, 2.0, 0.45], np.eye(3)))

    def _attempts(self, ctx, outcomes, reached=True, back=None):
        """pick.pick with pick_once returning `outcomes` in turn (recording a walking pose when `reached`) and back_out
        stubbed (`back`: its side effect): (pick_once's calls, back_out's calls, the result)."""

        def once(ctx_, key, side, rest):
            if reached:
                rest.update(side="right", jaw=np.zeros(3), rot=np.eye(3))
            return outcomes.pop(0)

        with (
            mock.patch.object(pick, "pick_once", side_effect=once) as po,
            mock.patch.object(pick, "back_out", side_effect=back or (lambda c, r: Outcome(True))) as bo,
        ):
            out = pick.pick(ctx, "Can")
        return po.call_count, bo.call_count, out

    def test_a_failed_attempt_is_retried_once_after_backing_out(self):
        ctx = self._ctx()
        n, backs, out = self._attempts(ctx, [Outcome(False, "target knocked 5 cm in settle"), Outcome(True)])
        self.assertEqual((n, backs, out.ok), (2, 1, True))
        (row,) = [r for r in ctx.log if r["step"] == "retry"]
        self.assertEqual((row["attempt"], row["after"]), (1, "target knocked 5 cm in settle"))
        n, backs, out = self._attempts(self._ctx(), [Outcome(False, "a"), Outcome(False, "b"), Outcome(True)])
        self.assertEqual((n, backs, out.why), (1 + pick.RETRY_ATTEMPTS, pick.RETRY_ATTEMPTS, "b"))

    def test_no_retry_after_a_fall_or_a_planning_refusal(self):
        ctx = self._ctx()
        ctx.motor.fell_at = 12.0
        self.assertEqual(self._attempts(ctx, [Outcome(False, "fell in entry"), Outcome(True)])[:2], (1, 0))
        refusal = Outcome(False, "no side fits the jaw within reach of an approachable edge")
        self.assertEqual(self._attempts(self._ctx(), [refusal, Outcome(True)], reached=False)[:2], (1, 0))

    def test_no_retry_when_x_is_off_its_support_or_between_fingers(self):
        self.assertEqual(self._attempts(self._ctx(z=0.6), [Outcome(False, "x"), Outcome(True)])[:2], (1, 0))
        ctx = self._ctx()
        ctx.motor.held = {7}
        self.assertEqual(self._attempts(ctx, [Outcome(False, "x"), Outcome(True)])[:2], (1, 0))
        ctx = self._ctx()
        ctx.world.meta["objects"]["Can"]["surface"] = "T"  # resting on S, which is not its own support
        self.assertEqual(self._attempts(ctx, [Outcome(False, "x"), Outcome(True)])[:2], (1, 0))

    def test_no_retry_for_an_object_under_another_surface(self):
        ctx = self._ctx()
        ctx.world.meta["surfaces"]["U"] = dict(xy=[2.0, 2.0], half_extent=[0.2, 0.2], yaw=0.0, top=0.9)  # over S
        self.assertEqual(cover.overhead(ctx.world, "Can"), ["U"])
        self.assertEqual(self._attempts(ctx, [Outcome(False, "entry timed out"), Outcome(True)])[:2], (1, 0))

    def test_no_retry_after_a_failed_low_grasp(self):
        low = Outcome(False, "target knocked 2 cm in settle", info=dict(low=True))
        self.assertEqual(self._attempts(self._ctx(), [low, Outcome(True)])[:2], (1, 0))

    def test_a_fall_while_backing_out_ends_the_pick(self):
        ctx = self._ctx()

        def fall(c, rest):
            c.motor.fell_at = 20.0
            return Outcome(False, "fell at 20.00 s")

        n, backs, out = self._attempts(ctx, [Outcome(False, "x"), Outcome(True)], back=fall)
        self.assertEqual((n, backs, out.why), (1, 1, "fell at 20.00 s"))

    def _back_out(self, free: bool):
        yaw = 0.5
        rot = R.rot_z(yaw) @ R.rot_y(np.radians(15.0))
        m = Arm([0.6, 0.3, 0.9], rot, xy=(0.1, -0.2), yaw=yaw)
        ctx = Ctx(World(), m)
        rest = dict(side="right", jaw=np.array([0.1, -0.25, -0.1]), rot=R.rot_y(0.2))
        occ = SimpleNamespace(is_free=mock.Mock(return_value=free))
        with mock.patch.object(pick, "nav_map", return_value=(occ, None, None)):
            pick.back_out(ctx, rest)
        back = m.sim.d.qpos[:2] - pick.RETRY_BACK_M * np.array([np.cos(yaw), np.sin(yaw)])
        np.testing.assert_allclose(occ.is_free.call_args.args[0], back, atol=1e-12)
        return m, ctx, rot, rest

    def test_back_out_pulls_back_level_opens_steps_back_onto_a_free_point_and_returns_the_arm(self):
        m, ctx, rot, rest = self._back_out(True)
        self.assertEqual(m.servo, {})  # both integrals dropped
        self.assertEqual([c[0] for c in m.calls], ["hand", "reach", "release", "sidestep", "reach"])
        u = np.array([np.cos(0.5), np.sin(0.5), 0.0])
        np.testing.assert_allclose(m.calls[1][1], np.array([0.6, 0.3, 0.9]) - pick.RETRY_OUT_M * u, atol=1e-12)
        np.testing.assert_array_equal(m.calls[1][2], rot)  # level: same height, same rotation
        Rz = R.rot_z(0.5)
        self.assertEqual(m.calls[3], ("sidestep", -pick.RETRY_BACK_M))  # off the slab before the arm comes down
        np.testing.assert_allclose(m.calls[4][1], np.array([0.1, -0.2, 0.8]) + Rz @ rest["jaw"], atol=1e-12)
        np.testing.assert_allclose(m.calls[4][2], Rz @ rest["rot"], atol=1e-12)
        self.assertEqual([r["step"] for r in ctx.log], ["back_out", "open", "step_back", "arm_rest"])

    def test_back_out_skips_the_step_onto_a_point_not_free(self):
        m, ctx, _, _ = self._back_out(False)
        self.assertEqual([c[0] for c in m.calls], ["hand", "reach", "release", "reach"])
        self.assertEqual(ctx.log[2]["skipped"], "not free on the skill's map")

    def test_pick_once_records_the_walking_pose_for_a_side_grasp_only(self):
        world = World()  # the Can 9 cm above the shelf's top: a level side grasp
        m = Arm([0.3, -0.2, 0.7], R.rot_z(0.5), xy=(1.5, 1.8), yaw=0.5)
        plan = Outcome(True, info=dict(side="right", psi=0.3, face=0.0, edge_xy=np.array([1.8, 2.0])))
        rest = {}
        with (
            mock.patch.object(pick, "approach", return_value=plan),
            mock.patch.object(pick, "edge_via", return_value=()),
            mock.patch.object(grasp_program, "run", return_value=Outcome(False, "x")),
        ):
            pick.pick_once(Ctx(world, m), "Can", "right", rest)
            self.assertEqual(rest["side"], "right")
            pel, Rz = np.array([1.5, 1.8, 0.8]), R.rot_z(0.5)
            np.testing.assert_allclose(pel + Rz @ rest["jaw"], [0.3, -0.2, 0.7], atol=1e-12)
            np.testing.assert_allclose(Rz @ rest["rot"], R.rot_z(0.5), atol=1e-12)
            top = {}
            with (
                mock.patch.object(pick, "is_top", return_value=True),
                mock.patch.object(pick, "pick_top", return_value=Outcome(False, "x")),
            ):
                pick.pick_once(Ctx(world, m), "Can", "right", top)
            self.assertEqual(top, {})


if __name__ == "__main__":
    unittest.main()
