"""TODO 33.1.4 (user 2026-10-23 "继续做 33.1.4，先做预测器"): the body model's reach predictor -- a k-nearest-neighbour
table over the envelope grid, queried in the pelvis heading frame or from a world point and pelvis pose."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

from locomani.tasks.scene import body  # noqa: E402


def _table(path: Path) -> None:
    """A synthetic grid: reaches beyond 0.6 m forward or below 0.3 m miss, the rest land."""
    xs, ys, zs = np.meshgrid(np.arange(0.2, 1.0, 0.1), [-0.05, 0.05, 0.15], np.arange(0.1, 1.5, 0.1), indexing="ij")
    rows = []
    for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel()):
        for att in (0.0, 1.0):
            for hand in (0.0, 1.0):
                rows.append([x, y, z, att, 0.78, hand])
    X = np.array(rows)
    miss = ((X[:, 0] > 0.6) | (X[:, 2] < 0.3)).astype(float)
    err = np.where(miss > 0, 12.0, 1.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32), y_err=err.astype(np.float32),
             mu=mu, sd=sd, features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15),
             miss_cm=np.array(3.0), stand_z=np.array(0.78))  # fmt: skip


class Reach(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "table.npz"
        _table(self.path)
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_table_separates_the_planted_regions(self):
        near = body.reach(0.45, 0.05, 0.9, "level", None, "right", path=str(self.path))
        far = body.reach(0.85, 0.05, 0.9, "level", None, "right", path=str(self.path))
        low = body.reach(0.45, 0.05, 0.1, "down", 0.78, "left", path=str(self.path))
        self.assertLess(near["p_miss"], 0.2)
        self.assertGreater(far["p_miss"], 0.8)
        self.assertGreater(low["p_miss"], 0.6)  # at the planted border (z < 0.3 misses) the neighbours mix
        self.assertLess(near["err_cm"], far["err_cm"])
        self.assertTrue(near["inside"])
        self.assertEqual((near["n"], near["att"], near["hand"], near["pelvis"]), (15, "level", "right", 0.78))
        # a query far outside the grid says so
        self.assertFalse(body.reach(3.0, 2.0, 4.0, "level", 0.4, "right", path=str(self.path))["inside"])

    def test_world_points_turn_into_the_heading_frame_with_the_outward_sign_per_hand(self):
        q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # yaw +90 deg: the robot faces +y
        pelvis = (1.0, 2.0, 0.78)
        x, y_out, z = body.grid_coords((1.0, 2.5, 0.9), pelvis, q, "right")  # 0.5 m ahead, on the centre line
        self.assertAlmostEqual(x, 0.5, places=6)
        self.assertAlmostEqual(y_out, 0.0, places=6)
        self.assertAlmostEqual(z, 0.9)
        # a point to the robot's right (world +x when facing +y) is outward for the right hand, inward for the left
        xr, out_r, _ = body.grid_coords((1.2, 2.5, 0.9), pelvis, q, "right")
        xl, out_l, _ = body.grid_coords((1.2, 2.5, 0.9), pelvis, q, "left")
        self.assertAlmostEqual(out_r, 0.2, places=6)
        self.assertAlmostEqual(out_l, -0.2, places=6)
        self.assertAlmostEqual(xr, xl)
        r = body.reach_world((1.0, 2.5, 0.9), pelvis, q, "right", pinch=True, path=str(self.path))
        self.assertEqual(r["att"], "down")
        self.assertLess(r["p_miss"], 0.2)

    def test_the_command_line_prints_json(self):
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            body.main(["--table", str(self.path), "reach", "--x", "0.45", "--z", "0.9", "--hand", "left"])
        out = json.loads(buf.getvalue())
        self.assertIn("p_miss", out)
        self.assertEqual(out["hand"], "left")


class Fit(unittest.TestCase):
    def test_auc_and_dedupe(self):
        import body_fit as BF

        self.assertAlmostEqual(BF.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)
        self.assertAlmostEqual(BF.auc(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 1])), 0.5)  # all tied: chance
        self.assertTrue(math.isnan(BF.auc(np.array([0.2, 0.3]), np.array([1, 1]))))
        rows = np.array([("a", "h1", "pre", 1.0), ("a", "h1", "pre", 1.0), ("a", "h2", "pre", 1.0), ("b", "h1", "pre", 2.0)],
                        dtype=[("label", "U4"), ("skills", "U4"), ("step", "U4"), ("t0", "f4")])  # fmt: skip
        self.assertEqual(len(BF.dedupe(rows)), 3)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _bias_table(path: Path, miss) -> None:
    """A synthetic free-space grid where every reach misses by `miss` (heading frame, metres)."""
    xs, ys, zs = np.meshgrid(np.arange(0.2, 0.8, 0.1), [-0.05, 0.05, 0.15], np.arange(0.6, 1.3, 0.1), indexing="ij")
    X = np.array([[x, y, z, a, 0.78, h] for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel()) for a in (0, 1, 2)
                  for h in (0, 1)])  # fmt: skip
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), E=np.tile(np.asarray(miss, float), (len(X), 1)), mu=mu, sd=sd,
             k=np.array(8))  # fmt: skip


class Compensation(unittest.TestCase):
    """TODO 33.1.4 (2): the free-space miss the motor's `reach_offset` hook takes off a one-hand reach's target."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        body.bias_table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()
        body.bias_table.cache_clear()

    def test_the_offset_is_the_heading_frame_miss_turned_into_the_world(self):
        path = self.dir / "bias.npz"
        _bias_table(path, [0.02, 0.01, -0.005])
        b = body.bias(0.5, 0.05, 0.9, "level", 0.78, "right", str(path))
        np.testing.assert_allclose(b["miss"], [0.02, 0.01, -0.005], atol=1e-6)
        self.assertTrue(b["inside"])
        offset = body.reach_offset(str(path))
        yaw = math.pi / 2  # facing +y: ahead is world +y, left is world -x
        pelvis, q = np.array([1.0, 2.0, 0.78]), np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        target = pelvis + np.array([0.05, 0.5, 0.12])  # 0.5 m ahead, 5 cm out to the right hand's side
        np.testing.assert_allclose(offset("right", target, _rot_z(yaw), pelvis, q, None), [-0.01, 0.02, -0.005],
                                   atol=1e-6)  # fmt: skip
        self.assertIsNone(offset("right", pelvis + np.array([0.0, 3.0, 0.12]), _rot_z(yaw), pelvis, q, None))

    def test_a_large_miss_is_capped_and_a_missing_table_gives_no_hook(self):
        path = self.dir / "bias.npz"
        _bias_table(path, [0.3, 0.0, 0.0])
        self.assertEqual(body.bias(0.5, 0.05, 0.9, path=str(path))["miss_cm"], 100 * body.BIAS_MAX_M)
        self.assertIsNone(body.reach_offset(str(self.dir / "missing.npz")))


class Prediction(unittest.TestCase):
    """TODO 33.1.2: every one-hand reach's record carries the body model's prediction; the sealed report scores it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "table.npz"
        _table(self.path)
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_wrist_rotation_maps_onto_the_grids_three_attitudes_whatever_the_heading(self):
        for yaw in (0.0, 1.0, -2.5):
            self.assertEqual(body.attitude(_rot_z(yaw)), "level")
            self.assertEqual(body.attitude(_rot_z(yaw) @ _rot_y(math.pi / 4)), "tilt45")
            self.assertEqual(body.attitude(_rot_z(yaw) @ _rot_y(math.pi / 2)), "down")
            self.assertEqual(body.attitude(_rot_z(yaw) @ _rot_y(math.radians(15))), "level")
            self.assertEqual(body.attitude(_rot_z(yaw) @ _rot_y(math.radians(75))), "down")

    def test_the_hook_predicts_from_the_command_and_the_pelvis_pose(self):
        note = body.reach_note(str(self.path))
        pelvis, q = np.array([0.0, 0.0, 0.78]), np.array([1.0, 0.0, 0.0, 0.0])  # facing +x
        near = note("right", np.array([0.45, -0.05, 0.9]), np.eye(3), pelvis, q, None, True)
        far = note("right", np.array([0.85, -0.05, 0.9]), np.eye(3), pelvis, q, None, False)
        self.assertEqual(set(near), {"pred_p_miss", "pred_err_cm", "pred_att", "pred_inside", "pred_open"})
        self.assertLess(near["pred_p_miss"], 0.2)
        self.assertGreater(far["pred_p_miss"], 0.8)
        self.assertEqual((near["pred_att"], near["pred_open"], far["pred_open"]), ("level", True, False))
        self.assertIsNone(body.reach_note(str(Path(self.tmp.name) / "missing.npz")))

    def test_the_score_counts_open_jaw_reaches_and_the_failures_seen_coming(self):
        def rd(ok, *reaches):
            steps = [dict(step="walk")] + [dict(step="pre", pred_p_miss=p, pred_err_cm=pe, jaw_err_cm=e, pred_open=o)
                                           for p, pe, e, o in reaches]  # fmt: skip
            return dict(ok=ok, steps=steps)

        rounds = [
            rd(True, (0.1, 1.0, 1.0, True)),
            rd(False, (0.9, 10.0, 12.0, True)),  # a miss the model saw coming
            rd(False, (0.2, 2.0, 6.0, True), (0.95, 9.0, 0.5, False)),  # a surprise; the held-object reach is left out
            dict(ok=False, steps=[dict(step="walk", ok=False)]),  # no reach: not scored
        ]
        sc = body.score(rounds)
        self.assertEqual((sc["rounds"], sc["reaches"], sc["missed"]), (3, 3, 2))
        self.assertEqual(sc["auc"], 1.0)  # both misses (0.9, 0.2) rank above the landed reach (0.1)
        self.assertEqual((sc["failed_rounds"], sc["failed_with_risky_reach"]), (2, 1))
        self.assertEqual(sc["err_cm"], round((1.0 + 12.0 + 6.0) / 3, 2))


if __name__ == "__main__":
    unittest.main()


class Curves(unittest.TestCase):
    """TODO 33.1.7: the design probes' curves, read at run time, and the measured best level of a parameter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "curves.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, *rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    @staticmethod
    def curve(levels, name="pick.X", factor="param", tasks=("t1",), machine="m1", controller="scalebfm", inc="h1"):
        return dict(family="f", incumbent=inc, factor=factor, name=name, tasks=list(tasks), machine=machine,
                    controller=controller, scope="skill" if factor == "param" else "start",
                    levels=[dict(level=lv, n=n, ok=ok) for lv, n, ok in levels])  # fmt: skip

    def test_curves_filter_by_name_factor_task_machine_and_controller(self):
        self.write(self.curve([(0.3, 4, 1)]), self.curve([(0.3, 4, 2)], tasks=("t2",), machine="m2"),
                   self.curve([(0, 4, 3)], name=None, factor="spawn_bearing", controller="amo"))  # fmt: skip
        p = str(self.path)
        self.assertEqual(len(body.curves(path=p)), 3)
        self.assertEqual(len(body.curves(name="pick.X", path=p)), 2)
        self.assertEqual(len(body.curves(factor="spawn_bearing", path=p)), 1)
        self.assertEqual(len(body.curves(task="t2", path=p)), 1)
        self.assertEqual(len(body.curves(machine="m1", path=p)), 2)
        self.assertEqual(len(body.curves(controller="scalebfm", path=p)), 2)
        self.assertEqual(body.curves(path=str(Path(self.tmp.name) / "none.jsonl")), [])

    def test_best_level_needs_rounds_and_a_margin_pooled_over_curves(self):
        # 0.38 m: 5/6 and 4/5 over two curves (9/11), 0.30 m: 2/6 and 1/5 (3/11) -- a clear winner
        self.write(self.curve([(0.30, 6, 2), (0.38, 6, 5)]), self.curve([(0.30, 5, 1), (0.38, 5, 4)], inc="h2"))
        b = body.best_level("pick.X", path=str(self.path))
        self.assertEqual(b["level"], 0.38)
        self.assertEqual((b["n"], b["curves"], b["incumbents"]), (11, 2, ["h1", "h2"]))
        self.assertAlmostEqual(b["levels"][0.30]["rate"], 3 / 11, places=3)
        # run D's iteration 18: 5/9, 6/9, 6/9, 6/9 -- no level measurably better, the skill keeps its default
        self.write(self.curve([(0.30, 9, 5), (0.38, 9, 6), (0.42, 9, 6), (0.46, 9, 6)]))
        self.assertIsNone(body.best_level("pick.X", path=str(self.path)))
        # too few rounds, or one level only
        self.write(self.curve([(0.30, 2, 0), (0.38, 3, 3)]))
        self.assertIsNone(body.best_level("pick.X", path=str(self.path)))
        self.write(self.curve([(0.38, 9, 9)]))
        self.assertIsNone(body.best_level("pick.X", path=str(self.path)))

    def test_gathering_sandboxes_tags_and_dedupes(self):
        import body_curves as BC

        root = Path(self.tmp.name)
        out = root / "origin_body"
        sbs = []
        for name, rows in (("sb1", [self.curve([(0.3, 4, 1)])]), ("sb2", [self.curve([(0.3, 4, 1)]),
                                                                        self.curve([(0.4, 4, 3)], inc="h9")])):  # fmt: skip
            d = root / name / BC.SUB
            d.mkdir(parents=True)
            run = [dict(r, iter=i) for i, r in enumerate(rows)]
            for r in run:
                r.pop("scope")
                r.pop("machine")
            (d / "curves_run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run))
            (d / "constraints.jsonl").write_text(json.dumps(dict(family="f", incumbent="h1", iter=0, label="s1",
                                                                 kind="isolated_failure")) + "\n")  # fmt: skip
            sbs.append(root / name)
        self.assertEqual(BC.build(sbs, out), {"curves.jsonl": 2, "constraints.jsonl": 1})  # sb2's first = sb1's
        got = [json.loads(x) for x in (out / "curves.jsonl").read_text().splitlines()]
        self.assertEqual([c["source"] for c in got], ["sb1", "sb2"])
        self.assertTrue(all(c["machine"] and c["scope"] == "skill" for c in got))
        self.assertEqual(BC.build(sbs, out), {"curves.jsonl": 0, "constraints.jsonl": 0})  # nothing new the second time
