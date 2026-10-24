"""TODO 33.9 (plan section 2.1, D8): grasps written as keypoint constraints -- the named points, the solver (free angles
sampled, the support a hard floor, what lies between the fingers), the chooser's scoring and the execution through
stage() on a kinematic fake motor."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.backends.scalebfm.motor import Outcome  # noqa: E402
from locomani.tasks.scene import body  # noqa: E402
from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402
from locomani.tasks.scene import keypoints as K  # noqa: E402

TOP = 0.70


def scene(half, z=None, yaw_deg=0.0, extra_meta=None) -> G.StaticScene:
    z = TOP + half[2] if z is None else z
    q = [math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0, math.sin(math.radians(yaw_deg) / 2)]
    xml = (f'<mujoco><worldbody><body name="obj" pos="0 0 {z}" quat="{q[0]} 0 0 {q[3]}"><freejoint/>'
           f'<geom type="box" size="{half[0]} {half[1]} {half[2]}"/></body></worldbody></mujoco>')  # fmt: skip
    d = Path(tempfile.mkdtemp())
    meta = dict(room=dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0])),
                surfaces={"table": dict(xy=[0.0, 0.0], top=TOP, half_extent=[0.30, 0.30, 0.01], yaw=0.0)},
                objects={"obj": dict(body="obj", surface="table", init_pos=[0, 0, z], init_quat=q, free=True,
                                     box=dict(half_extent=list(half)), **(extra_meta or {}))})  # fmt: skip
    (d / "scene_meta.json").write_text(json.dumps(meta))
    return G.StaticScene(d, model=mujoco.MjModel.from_xml_string(xml))


def table(path: Path) -> None:
    """A synthetic envelope: everything within 0.65 m lands, beyond misses; the straight-down wrist misses."""
    rows = [[x, y, zz, att, pel, hand] for x in np.arange(0.2, 0.95, 0.1) for y in (-0.05, 0.05, 0.15)
            for zz in np.arange(0.4, 1.2, 0.1) for att in (0.0, 1.0, 2.0) for pel in (0.78, 0.68, 0.62)
            for hand in (0.0, 1.0)]  # fmt: skip
    X = np.array(rows)
    miss = ((X[:, 0] > 0.65) | (X[:, 3] == 1.0)).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32),
             y_err=np.where(miss > 0, 10.0, 1.0).astype(np.float32), mu=mu, sd=sd,
             features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15), miss_cm=np.array(3.0),
             stand_z=np.array(0.78))  # fmt: skip


def orthonormal(R) -> bool:
    R = np.asarray(R, float)
    return np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


class Points(unittest.TestCase):
    def test_named_points_of_a_box(self):
        s = scene((0.04, 0.02, 0.05))
        pts = K.points(s, "obj")
        self.assertTrue({"centre", "top", "face:+x", "face:-z", "hull:+x", "hull:-y"} <= set(pts))
        np.testing.assert_allclose(pts["face:+x"].xyz, [0.04, 0.0, TOP + 0.05], atol=1e-6)
        np.testing.assert_allclose(pts["face:+x"].normal, [1, 0, 0], atol=1e-6)
        self.assertAlmostEqual(float(pts["top"].xyz[2]), TOP + 0.10, places=6)
        z = G.grasp_height(s, "obj")
        np.testing.assert_allclose(pts["hull:-y"].xyz, [0.0, -0.02, z], atol=1e-6)  # on the line through the middle

    def test_a_turned_box_turns_its_points(self):
        s = scene((0.04, 0.02, 0.05), yaw_deg=90)
        pts = K.points(s, "obj")
        np.testing.assert_allclose(pts["face:+x"].xyz[:2], [0.0, 0.04], atol=1e-6)
        np.testing.assert_allclose(pts["hull:+y"].normal, [-1, 0, 0], atol=1e-6)


class Solve(unittest.TestCase):
    def test_a_level_pinch_across_y_comes_in_along_x_wrist_up(self):
        s = scene((0.04, 0.02, 0.05))  # 4 cm across y
        poses = K.solve(s, "obj", dict(pinch=("hull:-y", "hull:+y"), approach="level"))
        self.assertEqual(len(poses), 2)
        for g in poses:
            self.assertTrue(orthonormal(g.rot))
            self.assertAlmostEqual(abs(float(g.approach[0])), 1.0, places=6)  # along +-x, square to the closing axis
            self.assertAlmostEqual(abs(float(g.closing[1])), 1.0, places=6)
            self.assertGreater(float(g.rot[2, 2]), 0.99)  # the wrist's z up
            self.assertAlmostEqual(g.width, 0.04, places=3)
            self.assertAlmostEqual(g.margin, 0.5 * (G.JAW_OPEN_M - 0.04), places=3)
            self.assertEqual(g.att, "level")

    def test_too_wide_is_dropped_and_a_closing_that_disagrees_gives_nothing(self):
        s = scene((0.04, 0.06, 0.05))  # 12 cm across y: wider than the open jaw
        self.assertEqual(K.solve(s, "obj", dict(pinch=("hull:-y", "hull:+y"), approach="level")), [])
        self.assertTrue(K.solve(s, "obj", dict(pinch=("hull:-x", "hull:+x"), approach="level")))
        self.assertEqual(K.solve(s, "obj", dict(pinch=("hull:-x", "hull:+x"), closing="vertical")), [])

    def test_the_support_is_a_floor_and_a_flat_object_sits_low_in_the_fingers(self):
        s = scene((0.05, 0.02, 0.008))  # a card 1.6 cm high
        poses = K.solve(s, "obj", dict(centre="centre", approach="down", closing="horizontal"))
        self.assertTrue(poses)
        for g in poses:
            tip = float(g.centre[2]) + K.FINGERTIP_M * float(g.approach[2])
            self.assertGreaterEqual(tip, TOP + K.TIP_CLEAR_M - 1e-9)  # raised off the table
            self.assertIn("raised", g.note)
        level = K.solve(s, "obj", dict(centre="centre", approach="level", closing="horizontal"))
        self.assertEqual(level, [])  # a level jaw's housing keeps its pads above a 1.6 cm card: nothing between them

    def test_a_level_jaw_raised_over_a_flat_object_is_not_offered(self):
        """Run 33.9.4: a 2 cm remote taken level with the jaw raised off the table to 3.8 cm closed on nothing, twice --
        the fingers' height range must overlap the object's."""
        s = scene((0.08, 0.025, 0.01))  # 2 cm tall
        poses = K.solve(s, "obj", dict(pinch=("hull:-y", "hull:+y"), twist=30))
        self.assertTrue(poses)
        for g in poses:
            e = np.cross(g.rot[:, 0], g.rot[:, 1])
            span = K.FINGER_REACH_M * abs(float(g.rot[2, 0])) + K.FINGER_HALF_W_M * abs(float(e[2]))
            self.assertGreaterEqual(
                min(TOP + 0.02, float(g.centre[2]) + span) - max(TOP, float(g.centre[2]) - span), 0.01 - 1e-9
            )
        self.assertNotIn("level", {g.att for g in poses})

    def test_a_twisted_pinch_of_a_long_bar_meets_only_the_strip_under_the_fingers(self):
        s = scene((0.15, 0.015, 0.01))  # a 30 x 3 cm bar
        poses = K.solve(s, "obj", dict(pinch=("hull:-y", "hull:+y"), approach="down", twist=45))
        twisted = [g for g in poses if abs(float(g.closing[1])) < math.cos(math.radians(10))]
        self.assertTrue(twisted)
        for g in twisted:  # at 45 deg: 3 / cos 45 + the fingers' 2 cm width x tan 45 = 6.2 cm, not the bar's length
            self.assertLess(g.width, 0.07)
            whole = G.width_across(G.section(s, "obj", float(g.centre[2]) - 0.01), g.closing)
            self.assertGreater(whole, g.width + 0.03)  # the whole section across the same axis

    def test_an_approach_into_a_face_and_one_from_below_never(self):
        s = scene((0.04, 0.03, 0.05))
        poses = K.solve(s, "obj", dict(centre="centre", approach=("normal", "face:+x"), closing="horizontal"))
        self.assertEqual(len(poses), 1)
        np.testing.assert_allclose(poses[0].approach, [-1, 0, 0], atol=1e-6)
        free = K.solve(s, "obj", dict(pinch=("hull:-y", "hull:+y")))
        self.assertTrue(free)
        self.assertTrue(all(float(g.approach[2]) <= K.UP_MAX for g in free))


class Plan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = str(Path(self.tmp.name) / "table.npz")
        table(Path(self.table))
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_options_carry_the_rotation_and_stand_clear(self):
        s = scene((0.04, 0.02, 0.05))
        opts = K.plan(s, "obj", dict(pinch=("hull:-y", "hull:+y")), start_xy=(-2.0, 0.0), table=self.table)
        self.assertTrue(opts)
        self.assertTrue(all(o.kind == "keypoint" and o.rot is not None and orthonormal(o.rot) for o in opts))
        self.assertTrue(all(o.clear_m >= C.STANCE_CLEAR_M for o in opts))
        self.assertNotEqual(opts[0].att, "down")  # the synthetic grid: a straight-down wrist misses
        keys = [(o.att, o.hand, round(o.approach_deg), round(o.distance_m, 2), o.pelvis_z) for o in opts]
        self.assertEqual(len(keys), len(set(keys)))  # no repeats

    def test_a_pinch_from_above_faces_within_the_twist_window(self):
        s = scene((0.05, 0.02, 0.008))
        opts = K.plan(s, "obj", dict(centre="centre", approach="down", closing="horizontal"), table=self.table)
        for o in opts:
            k = np.asarray(o.rot)[:, 1]
            square = math.degrees(math.atan2(k[1], k[0])) + 90.0
            off = min(abs(C._wrap(o.approach_deg - square)), abs(C._wrap(o.approach_deg - square - 180.0)))
            self.assertLessEqual(off, C.TWIST_MAX_DEG + 1e-3)  # the rotation is stored to 6 decimals
            f = np.array([math.cos(math.radians(o.approach_deg)), math.sin(math.radians(o.approach_deg)), 0.0])
            self.assertGreaterEqual(float(np.asarray(o.rot)[:, 2] @ f), -1e-6)  # wrist z toward the far side


class FakeMotor:
    """Kinematic: a walk lands where asked, a reach puts the jaw there, a grip takes the object between the fingers, a
    reach with the object held carries it."""

    def __init__(self, world, key):
        self.world, self.key, self.t, self.fell_at, self.trace = world, key, 0.0, None, []
        self.sim = SimpleNamespace(d=SimpleNamespace(qpos=np.array([-1.5, 0.0, 0.78])))
        self.jaw, self.yaw_v, self.held, self.calls = np.array([-1.3, -0.2, 0.9]), 0.0, False, []

    def _out(self, name, **info):
        t0 = self.t
        self.t += 1.0
        self.calls.append(name)
        return Outcome(True, "", t0, self.t, info)

    def walk_to(self, xy, yaw=None, **kw):
        self.sim.d.qpos[:2] = np.asarray(xy, float)
        if yaw is not None:
            self.yaw_v = float(yaw)
        return self._out("walk_to")

    def posture(self, z=None, **kw):
        return self._out("posture")

    def pelvis(self):
        return self.sim.d.qpos.copy(), np.array([math.cos(self.yaw_v / 2), 0, 0, math.sin(self.yaw_v / 2)])

    def yaw(self):
        return self.yaw_v

    def jaw_centre(self, side):
        return self.jaw.copy(), np.eye(3)

    def reach(self, side, pos, rot=None, *, via=(), speed=None, **kw):
        dz = np.asarray(pos, float) - self.jaw
        self.jaw = np.asarray(pos, float).copy()
        if self.held:
            p, q = self.world.object_pose(self.key)
            self.world.placements[self.key] = (np.asarray(p, float) + dz, q)
        return self._out("reach", via=len(via))

    def grip(self, side, **kw):
        self.held = True
        return self._out("grip")

    def jaw_opening(self, side):
        return 0.04 if self.held else 0.095

    def between_fingers(self, side):
        return {1} if self.held else set()

    def gripping(self, body_id):
        return {"right"} if self.held else set()


class Execute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = str(Path(self.tmp.name) / "table.npz")
        table(Path(self.table))
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_walk_pregrasp_insert_grip_lift_through_stage(self):
        s = scene((0.04, 0.02, 0.05))
        C.navigation(s)
        s.free = {"obj": 1}
        m = FakeMotor(s, "obj")
        log = []

        def note(name, out, **kw):  # as SkillContext.note builds the row: a clashing key raises here too
            dict(step=name, ok=out.ok, why=out.why, t0=out.t0, t1=out.t1, **out.info, **kw)
            log.append((name, out.ok))
            return out

        ctx = SimpleNamespace(world=s, motor=m, note=note)
        spec = dict(pinch=("hull:-y", "hull:+y"), approach="level")
        out = K.pick(ctx, "obj", spec, table=self.table, hands=("right",))
        self.assertTrue(out.ok, out.why)
        self.assertTrue(out.info["held"])
        steps = [n for n, _ in log]
        for name in ("keypoint_plan", "walk", "pregrasp", "insert", "grip", "lift"):
            self.assertIn(name, steps)
        self.assertLess(steps.index("pregrasp"), steps.index("insert"))
        self.assertGreater(float(s.object_pose("obj")[0][2]), TOP + 0.05 + 0.05)  # lifted

    def test_no_grasp_says_so(self):
        s = scene((0.09, 0.09, 0.06))  # 18 cm every way
        C.navigation(s)
        s.free = {"obj": 1}
        m = FakeMotor(s, "obj")
        ctx = SimpleNamespace(world=s, motor=m, note=lambda name, out, **kw: out)
        out = K.pick(ctx, "obj", dict(pinch=("hull:-y", "hull:+y")), table=self.table)
        self.assertFalse(out.ok)
        self.assertIn("no grasp", out.why)


if __name__ == "__main__":
    unittest.main()
