"""The put-down's pure parts (place.py): `plan_for` sends a goal of on(...) and upright(...) to `place.place_on` once per
on(X, S) in goal order and keeps held(X) on the pick; a spot sees what rests on the target by its pose, not by the
scene file's `surface` field; the drop guards -- a neighbour overlapping the object refuses the pick, the object's
offset out along the fingers, the surface a failed carry sets it down on; the creep budget -- the planned
lift-to-release time, the pad margin, the plan made before the pick -- and the pick's settle abort, inert unless set,
backing out from the measured jaw; the put-down's pick switches (inert at None), its stance and the spot rule for a
side-grasped versus a top-pinched object; the release -- a side-grasped object levelled before it is lowered, unloaded,
centred, retracted level, a top-pinched one released as before (held(X) after the pick and the pick's gripper shape:
test_held.py). Lives in the skills package because the loop's Engineer may change nothing outside it; run with
`make test TESTS="src/locomani/tasks/scene/skills/test_plan_for.py"`."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import locomani.tasks.scene.skills as skills
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import FF_SHORT, MotorConfig, Outcome, Profile, expmap, turn_seconds
from locomani.primitives.navigator import build
from locomani.tasks.scene.skills import clearance, grasp_program, held, pick, place


def yaw_quat(y):
    return [float(np.cos(y / 2)), 0.0, 0.0, float(np.sin(y / 2))]


def meta():
    """A 6 x 6 m room, a 0.8 x 0.6 m table (top 0.7 m, approached from -y) with a book on it whose scene-file surface
    is the shelf it came from, a 0.9 m tall cabinet next to it (underside on the floor) and the can to place."""
    table = dict(xy=[0.0, 0.0], half_extent=[0.4, 0.3], flat_half_extent=[0.4, 0.3], yaw=0.0, top=0.7,
                 approach_edges=["-y"], room="r", of_object="Table")  # fmt: skip
    shelf = dict(xy=[2.0, 2.0], half_extent=[0.2, 0.2], yaw=0.0, top=0.4, approach_edges=["-y"], room="r")
    objects = {
        "Book": dict(
            surface="S",
            free=True,
            body="Book_object",
            init_pos=[0.0, 0.0, 0.83],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.38, 0.27, 0.13]),
        ),  # fmt: skip
        "Cabinet": dict(
            surface=None,
            free=False,
            init_pos=[0.7, 0.0, 0.45],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.2, 0.2, 0.45]),
        ),  # fmt: skip
        "Can": dict(
            surface="S",
            free=True,
            body="Can_object",
            init_pos=[2.0, 2.0, 0.49],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.03, 0.03, 0.09]),
        ),  # fmt: skip
    }
    room = dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0]), walls=[], obstacles=[],
                spawn={"r": [[[-2.8, -2.8], [2.8, -2.8], [2.8, 2.8], [-2.8, 2.8]]]})  # fmt: skip
    return dict(objects=objects, surfaces={"T": table, "S": shelf}, room=room)


class World:
    """What plan_spot reads: the scene file, the objects' poses, the navigation body radius."""

    def __init__(self, poses=None):
        self.meta = meta()
        self.poses = poses or {}
        self.occ = SimpleNamespace(body_radius=0.35)

    def object_pose(self, key):
        o = self.meta["objects"][key]
        pos, quat = self.poses.get(key, (o["init_pos"], o["init_quat"]))
        return np.asarray(pos, float), np.asarray(quat, float)


class PlanFor(unittest.TestCase):
    def test_held_alone_or_with_upright_of_the_same_object_goes_to_the_pick_and_other_goals_have_no_plan(self):
        with mock.patch.object(held, "pick_held", return_value=Outcome(True)) as h:
            skills.plan_for(SimpleNamespace(goal=["held(Can)"]))("ctx")
            skills.plan_for(SimpleNamespace(goal=["upright(Can)", "held(Can)"]))("ctx")
        self.assertEqual(h.call_args_list, [mock.call("ctx", "Can", pick.pick, False),
                                            mock.call("ctx", "Can", pick.pick, True)])  # fmt: skip
        for goal in (
            ["upright(Can)"],
            ["held(Can)", "upright(Book)"],
            ["on(Can, T)", "held(Can)"],
            ["on(Can, T)", "near(Can, T, 1)"],
        ):
            self.assertIsNone(skills.plan_for(SimpleNamespace(goal=goal)), goal)

    def test_on_and_upright_place_each_on_goal_in_order_and_stop_at_a_failure(self):
        calls = []

        def place_on(ctx, key, surface):
            calls.append((key, surface))
            return Outcome(key != "Fail", "" if key != "Fail" else "pick refused: no side fits")

        with mock.patch.object(place, "place_on", side_effect=place_on):
            out = skills.plan_for(SimpleNamespace(goal=["on(Can, T)", "upright(Can)"]))("ctx")
            self.assertTrue(out.ok)
            self.assertEqual(calls, [("Can", "T")])
            calls.clear()
            out = skills.plan_for(SimpleNamespace(goal=["on(Book, S)", "on(Can, T)"]))("ctx")
            self.assertEqual(calls, [("Book", "S"), ("Can", "T")])
            calls.clear()
            out = skills.plan_for(SimpleNamespace(goal=["on(Fail, S)", "on(Can, T)"]))("ctx")
            self.assertEqual((calls, out.ok, out.why), ([("Fail", "S")], False, "pick refused: no side fits"))


class Spot(unittest.TestCase):
    def test_what_rests_on_the_target_counts_by_pose_and_furniture_does_not(self):
        rest = place.resting_on(World(), "T", "Can")
        self.assertEqual([tuple(np.round(f[0], 3)) for f in rest], [(0.0, 0.0)])  # the book, not the cabinet
        moved = World({"Book": ([2.0, -2.0, 0.13], yaw_quat(0.0))})
        self.assertEqual(place.resting_on(moved, "T", "Can"), [])

    def test_spot_clears_the_book_at_the_listed_edge_with_a_routed_stance(self):
        world = World({"Book": ([0.0, 0.11, 0.83], yaw_quat(0.0))})  # free along -y at the first depth only
        occ, _ = build(world.meta, body_radius=0.35)
        world.occ = occ
        with mock.patch.dict(pick._NAV, {}, clear=True):
            spot, why = place.plan_spot(world, "Can", "T", "right", np.array([0.0, -2.0]), 0.0)
            self.assertIsNotNone(spot, why)
            self.assertEqual((spot["edge"], spot["depth"]), ("-y", 0.06))
            self.assertAlmostEqual(float(spot["xy"][1]), -0.24, places=6)
            self.assertAlmostEqual(spot["face"], np.pi / 2)
            np.testing.assert_allclose(spot["stance"], spot["xy"] - [pick.STAND_SIDE_M, pick.STAND_AHEAD_M], atol=1e-9)
            full = World({"Book": ([0.0, 0.0, 0.83], yaw_quat(0.0)), "Can": ([2.0, 2.0, 0.49], yaw_quat(0.0))})
            full.meta["objects"]["Book"]["box"]["half_extent"] = [0.42, 0.32, 0.13]
            full.occ = occ
            spot, why = place.plan_spot(full, "Can", "T", "right", np.array([0.0, -2.0]), 0.0)
            self.assertIsNone(spot)
            self.assertTrue(why.startswith("no spot: "), why)


class DropGuards(unittest.TestCase):
    def test_a_neighbour_overlapping_the_object_on_its_support_blocks_the_pick(self):
        self.assertIsNone(place.blocking_neighbour(World(), "Can"))  # the book rests on T, whatever its surface field
        # the book on the shelf, its near face 1 cm into the can's footprint (books in a row: -1.2 to -1.5 cm)
        into = World({"Book": ([2.0, 2.0 + 0.03 + 0.27 - 0.01, 0.53], yaw_quat(0.0))})
        self.assertEqual(place.blocking_neighbour(into, "Can"), "Book")
        clear = World({"Book": ([2.0, 2.0 + 0.03 + 0.27 + 0.02, 0.53], yaw_quat(0.0))})  # 2 cm clear
        self.assertIsNone(place.blocking_neighbour(clear, "Can"))
        # the can 0.5 cm short of the book's -y face (y 1.73) square on, but turned 45 deg its corner (4.2 cm) reaches in
        square = World({"Can": ([2.0, 1.695, 0.49], yaw_quat(0.0)), "Book": ([2.0, 2.0, 0.53], yaw_quat(0.0))})
        self.assertIsNone(place.blocking_neighbour(square, "Can"))
        turned = World({"Can": ([2.0, 1.695, 0.49], yaw_quat(np.pi / 4)), "Book": ([2.0, 2.0, 0.53], yaw_quat(0.0))})
        self.assertEqual(place.blocking_neighbour(turned, "Can"), "Book")

    def test_held_out_is_the_object_centre_along_the_fingers(self):
        jaw = np.array([0.5, -0.2, 0.9])
        side = R.rot_z(0.7)  # side grasp: wrist x level
        self.assertAlmostEqual(place.held_out(jaw, side, jaw + side @ [0.033, 0.004, -0.01]), 0.033)
        self.assertGreater(place.held_out(jaw, side, jaw + side @ [0.033, 0.0, 0.0]), place.HELD_OUT_MAX_M)
        self.assertLess(place.held_out(jaw, side, jaw + side @ [0.026, 0.0, 0.0]), place.HELD_OUT_MAX_M)
        top = R.rot_z(0.7) @ R.rot_y(np.pi / 2)  # top-down pinch: wrist x down, out along the fingers is below
        self.assertAlmostEqual(place.held_out(jaw, top, jaw - [0.0, 0.0, 0.025]), 0.025)

    def test_set_down_takes_the_highest_of_support_and_target_under_the_object(self):
        lifted = World({"Can": ([0.1, -0.1, 0.70 + 0.09 + 0.10], yaw_quat(0.0))})  # 10 cm over the table T
        self.assertEqual(place.surface_below(lifted, "Can", ("S", "T")), "T")
        self.assertIsNone(place.surface_below(lifted, "Can", ("S",)))
        resting = World({"Can": ([0.1, -0.1, 0.70 + 0.09 + 0.005], yaw_quat(0.0))})  # still on it
        self.assertEqual(place.surface_below(resting, "Can", ("T",)), "T")
        edge = World({"Can": ([0.1, -0.3 + 0.02, 0.89], yaw_quat(0.0))})  # centre 2 cm inside the -y edge
        self.assertIsNone(place.surface_below(edge, "Can", ("T", "S")))
        below = World({"Can": ([0.1, -0.1, 0.60], yaw_quat(0.0))})  # under the table top
        self.assertIsNone(place.surface_below(below, "Can", ("T",)))


class CreepBudget(unittest.TestCase):
    def test_walk_seconds_are_walk_tos_turns_and_profiles_plus_the_leg_settle(self):
        cfg = MotorConfig()
        leg = Profile(cfg.walk_speed, cfg.walk_acc, 1.0 + FF_SHORT[0]).t_end
        straight = place.walk_seconds(cfg, [0.0, 0.0], 0.0, [[0.0, 0.0], [1.0, 0.0]], 0.0)
        self.assertAlmostEqual(straight, leg + place.LEG_SETTLE_S)
        # facing +y: turn to the leg first (past face_travel_deg), then to the goal yaw at its end
        turned = place.walk_seconds(cfg, [0.0, 0.0], np.pi / 2, [[0.0, 0.0], [1.0, 0.0]], -np.pi / 2)
        self.assertAlmostEqual(turned, straight + turn_seconds(np.pi / 2, cfg.turn_rate) * 2)
        # a step back keeps the heading: no turn
        back = place.walk_seconds(cfg, [0.0, 0.0], 0.0, [[0.0, 0.0], [-0.3, 0.0]], 0.0, fixed=True)
        self.assertAlmostEqual(
            back, Profile(cfg.walk_speed, cfg.walk_acc, 0.3 + FF_SHORT[0]).t_end + place.LEG_SETTLE_S
        )

    def test_lift_to_release_adds_lift_hold_back_off_carry_reach_over_and_lowering(self):
        cfg = MotorConfig()
        corners = [[-0.3, 0.0], [-0.3, 2.0]]
        t = place.lift_to_release_s(cfg, [0.0, 0.0], 0.0, corners, np.pi / 2, 0.06, 0.0)
        carry = place.walk_seconds(cfg.__class__(turn_rate=place.CARRY_TURN_RATE), [-0.3, 0.0], 0.0, corners, np.pi / 2)
        back = place.walk_seconds(cfg, [0.0, 0.0], 0.0, [[0.0, 0.0], [-0.3, 0.0]], 0.0, fixed=True)
        reach = 2 * max(cfg.reach_min_s, (0.06 + place.VIA_OUT_M) / cfg.reach_speed) + cfg.reach_settle_s
        lower = place.ABOVE_M / place.DESCENT_SPEED + cfg.reach_settle_s
        lift = grasp_program.SIMPLE["lift_seconds"] + grasp_program.HOLD_S
        self.assertAlmostEqual(t, lift + back + carry + reach + lower)
        self.assertGreater(place.lift_to_release_s(cfg, [0.0, 0.0], 0.0, corners + [[2.0, 2.0]], 0.0, 0.06, 0.0), t)

    def test_grip_margin_is_the_top_above_the_grasp_less_the_pad_half_height(self):
        self.assertAlmostEqual(place.grip_margin(0.049), 0.049 - grasp_program.PAD_Z)  # the clock: 3.5 cm
        self.assertLess(place.grip_margin(0.0138), 0.0)  # the wallet's 2.8 cm: none
        box = dict(half_extent=[0.03, 0.03, 0.09])
        jaw, rot = np.array([0.5, 0.2, 1.0]), R.rot_z(0.7)  # side grasp, wrist z up
        planned = place.grip_margin(0.09)
        self.assertAlmostEqual(place.grip_margin_now(jaw, rot, jaw, yaw_quat(0.3), box), planned)
        slid = place.grip_margin_now(jaw, rot, jaw - [0.0, 0.0, 0.02], yaw_quat(0.3), box)  # 2 cm down the pads
        self.assertAlmostEqual(slid, planned - 0.02)
        # the clock over a 30 s carry at 0.15 cm/s needs 4.5 cm; the jam's 7.6 cm covers 43.8 s
        with mock.patch.object(place, "CREEP_RATE_M_S", 0.0012):  # the rate these numbers were worked at
            self.assertTrue(place.creep_short(place.grip_margin(0.049), 30.0))
            self.assertFalse(place.creep_short(place.grip_margin(0.09), 43.8))

    def test_plan_ahead_plans_the_spot_and_carry_before_the_pick_or_says_why_not(self):
        world = World({"Book": ([0.0, 0.11, 0.83], yaw_quat(0.0))})  # free along -y
        occ, _ = build(world.meta, body_radius=0.35)
        world.occ = occ
        with mock.patch.dict(pick._NAV, {}, clear=True):
            ahead = place.plan_ahead(world, "Can", "T", "right", np.array([0.0, -2.0]), MotorConfig())
            self.assertIsNotNone(ahead["spot"], ahead["why"])
            self.assertEqual((ahead["hand"], ahead["top"]), ("right", False))
            self.assertAlmostEqual(ahead["margin"], 0.09 - grasp_program.PAD_Z)
            np.testing.assert_allclose(ahead["corners"][-1], ahead["spot"]["stance"])
            self.assertGreater(ahead["t_est"], grasp_program.SIMPLE["lift_seconds"] + grasp_program.HOLD_S)
            world.poses["Book"] = ([0.0, 0.0, 0.83], yaw_quat(0.0))
            world.meta["objects"]["Book"]["box"]["half_extent"] = [0.42, 0.32, 0.13]  # covers the table
            ahead = place.plan_ahead(world, "Can", "T", "right", np.array([0.0, -2.0]), MotorConfig())
            self.assertIsNone(ahead["spot"])
            self.assertTrue(ahead["why"].startswith("no spot: "), ahead["why"])


class SettleAbort(unittest.TestCase):
    """grasp_program.run on a jaw that stays 5 cm short of the pre-grasp: settle_abort_m None (held(X)'s) goes on into
    the entry as before; set, the hand backs out and the program fails before any entry."""

    def run_program(self, abort):
        body = 7
        jaw = np.array([0.0, 0.0, 1.0])
        sim = SimpleNamespace(d=SimpleNamespace(xpos={body: np.array([0.19, 0.0, 1.0])}, ncon=0), m=None)
        self.hands = []
        m = SimpleNamespace(sim=sim, servo_on={}, servo={}, fingers={"right": (1, 2)}, jaw=np.zeros(2),
                            jaw_centre=lambda side: (jaw, np.eye(3)), commanded_jaw=lambda side: jaw.copy(),
                            hand=lambda side, pos=None, *a: self.hands.append(None if pos is None else pos.copy()),
                            adopt_hand=lambda side: None)  # fmt: skip
        m.t = 0.0

        def tick():
            m.t += R.DT
            return True

        m.tick = tick
        notes = []
        ctx = SimpleNamespace(world=SimpleNamespace(free={"Can": body}), motor=m,
                              note=lambda name, out, **kw: notes.append((name, out, kw)) or out)  # fmt: skip
        with mock.patch.dict(grasp_program.SIMPLE, settle_abort_m=abort):
            out = grasp_program.run(ctx, "Can", "right", np.eye(3), np.array([0.19, 0.0, 1.0]), 0.06)
        return out, notes[-1][2]["phases"]

    def test_inert_at_none_and_aborting_when_set(self):
        self.assertIsNone(grasp_program.SIMPLE["settle_abort_m"])
        out, phases = self.run_program(None)
        self.assertIn("entry", phases)
        self.assertNotIn("abort", phases)
        out, phases = self.run_program(0.035)
        self.assertEqual(phases, ["settle", "abort"])
        self.assertEqual((out.ok, out.why), (False, "pregrasp missed by 5.0 cm"))
        # the settle commands the pre-grasp 5 cm ahead plus its feedback; the abort starts at the measured jaw, its
        # feedback zeroed, and only backs out from there
        xs = [float(h[0]) for h in self.hands if h is not None]
        k = next(i for i, x in enumerate(xs) if abs(x) < 1e-9)
        self.assertGreater(min(xs[:k]), 0.04)
        self.assertLessEqual(max(xs[k:]), 1e-9)


class PutDownPick(unittest.TestCase):
    """The put-down's switches: off (held(X)'s pick) unless place_on turns them on; on, a right-hand side grasp on a
    surface above 0.6 m stands at PICK_SIDE_AHEAD_M, the left hand, a top-down pinch and every put-down stance at
    pick.STAND_AHEAD_M."""

    def test_switches_are_inert_at_none_and_restored_after_the_put_down(self):
        self.assertIsNone(pick.SIDE_AHEAD_M)
        self.assertEqual(pick.stand_ahead(0.82, "right", False), pick.STAND_AHEAD_M)
        self.assertEqual(grasp_program.SIMPLE["entry_capture_depth_m"], 0.025)
        with self.assertRaises(RuntimeError), place._pick_switches():
            self.assertEqual(pick.SIDE_AHEAD_M, place.PICK_SIDE_AHEAD_M)
            raise RuntimeError
        self.assertIsNone(pick.SIDE_AHEAD_M)

    def test_only_a_right_hand_side_grasp_above_0_6_m_stands_back(self):
        with place._pick_switches():
            self.assertEqual(pick.stand_ahead(0.82, "right", False), 0.46)
            self.assertEqual(pick.stand_ahead(0.82, "left", False), pick.STAND_AHEAD_M)
            self.assertEqual(pick.stand_ahead(0.82, "right", True), pick.STAND_AHEAD_M)
            self.assertEqual(pick.stand_ahead(0.51, "right", False), pick.STAND_AHEAD_M)
        world = World({"Book": ([0.0, 0.11, 0.83], yaw_quat(0.0)), "Can": ([0.1, -0.2, 0.79], yaw_quat(0.0))})
        world.meta["objects"]["Can"]["surface"] = "T"  # on the 0.7 m table, approached from -y
        occ, _ = build(world.meta, body_radius=0.35)
        world.occ = occ
        with mock.patch.dict(pick._NAV, {}, clear=True), place._pick_switches():
            for side, ahead in (("right", 0.46), ("left", pick.STAND_AHEAD_M)):
                plan, why = pick.plan_approach(world, np.array([0.0, -2.0]), "Can", side)
                self.assertIsNotNone(plan, why)
                self.assertAlmostEqual(float(-0.2 - plan[3][1]), ahead, msg=side)
            spot, why = place.plan_spot(world, "Can", "T", "right", np.array([0.0, -2.0]), 0.0)
            np.testing.assert_allclose(spot["stance"], spot["xy"] - [pick.STAND_SIDE_M, pick.STAND_AHEAD_M], atol=1e-9)

    def test_a_side_grasped_spot_sits_off_the_ends_and_inside_the_top_a_pinched_one_keeps_the_old_rule(self):
        world = World({"Book": ([0.0, 0.2, 0.83], yaw_quat(0.0))})  # free along -y to 0.13 m deep
        occ, _ = build(world.meta, body_radius=0.35)
        world.occ = occ
        can = world.meta["objects"]["Can"]
        start = np.array([-2.0, -2.0])  # nearest the -x end of the -y edge
        with mock.patch.dict(pick._NAV, {}, clear=True):
            spot, why = place.plan_spot(world, "Can", "T", "right", start, 0.0)  # side grasp (half height 9 cm)
            self.assertEqual((spot["edge"], spot["depth"]), ("-y", 0.06), why)
            self.assertAlmostEqual(float(spot["xy"][0]), -0.4 + place.PLACE_SIDE_MIN_M)
            can["box"]["half_extent"] = [0.03, 0.06, 0.09]  # carried 45 deg turned: 6.4 cm reach across the edge
            spot, why = place.plan_spot(world, "Can", "T", "right", start, np.pi / 4)
            self.assertEqual(spot["depth"], 0.08, why)  # at 0.06 it would end 1.4 cm from the -y edge
            self.assertAlmostEqual(float(spot["xy"][0]), -0.32)  # at -0.34, 1.4 cm from the -x end
            self.assertTrue(place.clearance.inside(place.clearance.carried_footprint(spot["xy"], [0.03, 0.06],
                                                                                      spot["face"], np.pi / 4),
                                                   world.meta["surfaces"]["T"], place.SPOT_INSIDE_M))  # fmt: skip
            can["box"]["half_extent"] = [0.03, 0.06, 0.015]  # flat on the 0.4 m shelf: a top-down pinch
            self.assertTrue(pick.is_top(world, "Can"))
            spot, why = place.plan_spot(world, "Can", "T", "right", start, np.pi / 4)
            self.assertEqual(spot["depth"], 0.06, why)
            self.assertAlmostEqual(float(spot["xy"][0]), -0.4 + place.SPOT_SIDE_M)


class Hand:
    """What the lowering and the release drive: a jaw that goes where a reach sends it (rot None keeps the attitude),
    the servo's lead and turn integral, the motions in order."""

    def __init__(self, jaw, rot):
        self.jaw, self.rot = np.asarray(jaw, float), np.asarray(rot, float)
        self.servo = {("pos", "right"): np.array([0.0, 0.0, 0.03]), ("turn", "right"): np.zeros(3)}
        self.calls, self.trace, self.fell_at, self.t = [], [], None, 0.0

    def jaw_centre(self, side):
        return self.jaw.copy(), self.rot.copy()

    def pelvis(self):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def yaw(self):
        return 0.0

    def reach(self, side, pos, rot=None, **kw):
        self.calls.append(("reach", np.asarray(pos, float), None if rot is None else np.asarray(rot, float)))
        self.jaw, self.rot = np.asarray(pos, float), self.rot if rot is None else np.asarray(rot, float)
        return Outcome(True)

    def release(self, side):
        self.calls.append(("release",))
        return Outcome(True)

    def sidestep(self, **kw):
        self.calls.append(("sidestep",))
        return Outcome(True)

    def touching(self, body):
        return set()


class Ctx:
    def __init__(self, world, motor):
        self.world, self.motor, self.log = world, motor, []

    def note(self, name, out, **extra):
        self.log.append(dict(step=name, ok=out.ok, **out.info, **extra))
        return out


class Release(unittest.TestCase):
    """A side-grasped object is lowered with its measured up axis levelled (at most LEVEL_MAX_DEG) and let go by
    unload, open, centre, a level retract and a rise; a top-pinched one keeps the carried wrist, the release and the
    rising retract."""

    def test_levelling_is_identity_when_level_takes_the_up_axis_to_vertical_and_clamps(self):
        corr, deg = clearance.level_rotation([0.0, 0.0, 1.0], place.LEVEL_MAX_DEG)
        np.testing.assert_allclose(corr, np.eye(3))
        self.assertEqual(deg, 0.0)
        u = R.rot_z(0.4) @ R.rot_y(np.radians(12.0)) @ [0.0, 0.0, 1.0]
        corr, deg = clearance.level_rotation(u, 25.0)
        np.testing.assert_allclose(corr @ u, [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(deg, 12.0)
        u = expmap([np.radians(40.0), 0.0, 0.0]) @ [0.0, 0.0, 1.0]
        corr, deg = clearance.level_rotation(u, 25.0)
        self.assertAlmostEqual(deg, 25.0)
        v = corr @ u
        self.assertAlmostEqual(float(np.degrees(np.arccos(v[2]))), 15.0)
        self.assertAlmostEqual(float(v[0]), 0.0)  # turned in the plane of u and z

    def test_up_axis_is_the_rest_vertical_turned_by_the_pose_now(self):
        rest = R.rot_z(0.7) @ expmap([0.0, np.radians(4.0), 0.0])  # a start turns the rest pose about z only
        tip = expmap([np.radians(10.0), 0.0, 0.0])
        now = R.quat_wxyz(tip @ R.rot_z(1.1) @ rest)
        np.testing.assert_allclose(clearance.up_axis(now, R.quat_wxyz(rest)), tip @ [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(clearance.lean_deg(now, R.quat_wxyz(rest)), 10.0)

    def test_centre_move_follows_the_offset_sign_and_clamps(self):
        self.assertEqual(place.centre_move(0.0015), 0.0)
        self.assertEqual(place.centre_move(-place.CENTRE_MIN_M), 0.0)
        self.assertAlmostEqual(place.centre_move(0.012), 0.012)
        self.assertAlmostEqual(place.centre_move(-0.009), -0.009)
        self.assertEqual(place.centre_move(0.05), place.CENTRE_MAX_M)
        self.assertEqual(place.centre_move(-0.05), -place.CENTRE_MAX_M)

    def _held(self, top: bool):
        """The Can held 12 deg tipped about x, 2 cm out along the fingers of a side grasp (wrist x along +y, tipped)."""
        tip = expmap([np.radians(12.0), 0.0, 0.0])
        pos = np.array([0.1, -0.1, 0.95])
        rot = tip @ R.rot_z(np.pi / 2)
        world = World({"Can": (pos, R.quat_wxyz(tip))})
        world.free = {"Can": 7}
        if top:
            world.meta["objects"]["Can"]["box"]["half_extent"] = [0.03, 0.06, 0.015]
        return world, Hand(pos - rot @ [0.02, 0.0, 0.0], rot), pos, tip

    def test_a_side_grasp_is_lowered_levelled_onto_the_spot(self):
        world, m, pos, tip = self._held(top=False)
        rot0, jaw0 = m.rot.copy(), m.jaw.copy()
        off, level = place._lower_onto(Ctx(world, m), "Can", "right", [0.1, -0.1], "T", [0.1, -0.3], np.pi / 2)
        self.assertEqual((off, round(level, 6)), ([], 12.0))
        self.assertEqual(len(m.calls), 2)
        for _, _, rot in m.calls:
            np.testing.assert_allclose(rot @ rot0.T @ tip @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-9)
        corr = m.calls[-1][2] @ rot0.T
        centre = m.calls[-1][1] + corr @ (pos - jaw0)  # where the levelled object's centre ends
        np.testing.assert_allclose(centre, [0.1, -0.1, 0.7 + 0.09 + place.PLACE_CLEAR_M], atol=1e-9)

    def test_a_top_pinch_is_lowered_with_the_carried_wrist(self):
        world, m, pos, _ = self._held(top=True)
        self.assertTrue(pick.is_top(world, "Can"))
        jaw0 = m.jaw.copy()
        off, level = place._lower_onto(Ctx(world, m), "Can", "right", [0.1, -0.1], "T", [0.1, -0.3], np.pi / 2)
        self.assertEqual((off, level), ([], None))
        self.assertEqual([c[2] for c in m.calls], [None, None])
        np.testing.assert_allclose(m.calls[-1][1], [0.1, -0.1, 0.7 + 0.015 + place.PLACE_CLEAR_M] - (pos - jaw0))

    def test_release_unloads_opens_centres_retracts_level_then_rises(self):
        world, m, pos, _ = self._held(top=False)
        m.jaw = pos - m.rot @ [0.02, 0.012, 0.0]  # the object 1.2 cm toward +wrist y
        jaw0, rot0 = m.jaw.copy(), m.rot.copy()
        ctx = Ctx(world, m)
        place._release(ctx, "Can", "right", 12.0)
        self.assertEqual([r["step"] for r in ctx.log], ["unload", "open", "centre", "retract", "rise", "release"])
        self.assertNotIn(("pos", "right"), m.servo)
        self.assertNotIn(("turn", "right"), m.servo)
        goals = [c[1] for c in m.calls if c[0] == "reach"]
        np.testing.assert_allclose(goals[0], jaw0)
        np.testing.assert_allclose(goals[1], jaw0 + 0.012 * rot0[:, 1], atol=1e-9)
        np.testing.assert_allclose(goals[2], goals[1] - place.RETRACT_M * rot0[:, 0], atol=1e-9)
        np.testing.assert_allclose(goals[3], goals[2] + [0.0, 0.0, place.RETRACT_RISE_M], atol=1e-9)
        for c in m.calls:
            if c[0] == "reach":
                np.testing.assert_allclose(c[2], rot0)
        row = ctx.log[-1]
        self.assertEqual((row["lead_cm"], row["centre_cm"], row["level_deg"]), ([0.0, 0.0, 3.0], 1.2, 12.0))
        self.assertEqual(row["touching_open"], [])

    def test_a_top_pinch_put_down_releases_and_retracts_as_before_a_side_grasp_through_release(self):
        world, m, _, _ = self._held(top=True)
        jaw0, rot0 = m.jaw.copy(), m.rot.copy()
        ctx = Ctx(world, m)
        spot = dict(xy=[0.1, -0.1], edge_xy=[0.1, -0.3], face=np.pi / 2, edge="-y")
        with (
            mock.patch.object(place, "_lower_onto", return_value=([], None)),
            mock.patch.object(place, "surface_below", return_value="T"),
            mock.patch.object(place, "resting", return_value=(True, "")),
            mock.patch.object(place, "_release") as release,
        ):
            self.assertTrue(place._put_down(ctx, "Can", "T", "right", spot, ("S", "T"), {}).ok)
        release.assert_not_called()
        self.assertEqual([r["step"] for r in ctx.log], ["release", "retract", "step_back"])
        self.assertEqual([c[0] for c in m.calls], ["release", "reach", "sidestep"])
        up = jaw0 - place.RETRACT_M * rot0[:, 0] + [0.0, 0.0, place.RETRACT_UP_M]
        np.testing.assert_allclose(m.calls[1][1], up)
        self.assertIsNone(m.calls[1][2])
        self.assertIn(("pos", "right"), m.servo)
        world, m, _, _ = self._held(top=False)
        ctx = Ctx(world, m)
        with (
            mock.patch.object(place, "_lower_onto", return_value=([], 7.5)),
            mock.patch.object(place, "surface_below", return_value="T"),
            mock.patch.object(place, "resting", return_value=(True, "")),
            mock.patch.object(place, "_release", return_value=Outcome(True)) as release,
        ):
            self.assertTrue(place._put_down(ctx, "Can", "T", "right", spot, ("S", "T"), {}).ok)
        release.assert_called_once_with(ctx, "Can", "right", 7.5)


if __name__ == "__main__":
    unittest.main()
