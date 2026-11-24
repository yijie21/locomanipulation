"""TODO 49.1 (user 2026-11-24): a run's switches -- benchmark, truth or perception targets, stereo or RGB-D, depth
noise, self pose, the pelvis ScaleBFM is told, controller -- resolved once and checked against each benchmark's
capability table before anything starts."""

import io
import itertools
import unittest
from contextlib import redirect_stdout

import numpy as np

from locomani import profiles
from locomani.bench import BENCHES, BenchOptions, registry, resolve
from locomani.bench import __main__ as cli
from locomani.bench import options as O
from locomani.bench.spec import AXES, PLANNED, READY, BenchSpec
from locomani.perception.sensor import Frame


class Profiles(unittest.TestCase):
    def test_every_benchmark_has_a_profile_that_resolves_to_itself_within_its_table(self):
        self.assertEqual(profiles.names("bench"), sorted(BENCHES))
        for name in BENCHES:
            o = profiles.load("bench", name)
            self.assertIsInstance(o, BenchOptions)
            self.assertEqual(o.bench, name)
            self.assertEqual(O.consistency_errors(o), [], name)
            self.assertEqual(O.support_errors(o, allow_planned=True), [], name)
            r = resolve(name, allow_planned=True)
            self.assertEqual((r.options, r.explicit, r.normalized), (o, (), ()), name)

    def test_the_profiles_keep_the_loops_settings_so_far(self):
        # SIMPLE's loops: stereo depth; leg odometry for the belief and for ScaleBFM's root (OdoRoot as root_source)
        s = resolve("simple", allow_planned=True).options
        self.assertEqual((s.target, s.sensor, s.self_pose, s.controller_root, s.controller),
                         ("perception", "stereo", "odometry", "odometry", "scalebfm"))  # fmt: skip
        self.assertEqual(resolve("locomanilab", allow_planned=True).options.sensor, "rgbd")
        for name in ("bigym2", "humanoidarena"):
            self.assertEqual(resolve(name, allow_planned=True).options.controller, "native")


class Precedence(unittest.TestCase):
    def test_code_default_then_profile_then_override(self):
        self.assertEqual(BenchOptions().self_pose, "odometry")  # code default
        self.assertEqual(resolve("bigym2", allow_planned=True).options.self_pose, "truth")  # the profile's
        r = resolve("bigym2", ["self_pose=odometry"], allow_planned=True)
        self.assertEqual(r.options.self_pose, "odometry")  # the override's
        self.assertEqual(r.explicit, ("self_pose",))

    def test_an_explicit_value_equal_to_the_default_is_still_recorded_as_explicit(self):
        r = resolve("simple", ["sensor=stereo"], allow_planned=True)
        self.assertEqual(r.explicit, ("sensor",))
        self.assertEqual(r.record()["explicit"], ["sensor"])

    def test_the_benchmark_can_come_from_an_override_but_not_twice(self):
        self.assertEqual(resolve(None, ["bench=locomanilab"], allow_planned=True).options.bench, "locomanilab")
        with self.assertRaisesRegex(ValueError, "two benchmarks"):
            resolve("simple", ["bench=bigym2"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "unknown benchmark"):
            resolve("bigym", allow_planned=True)


class InvalidInput(unittest.TestCase):
    def test_an_unknown_choice_or_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            resolve("simple", ["target=xray"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "unknown option"):
            resolve("simple", ["camera=stereo"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "unknown field"):
            BenchOptions.from_dict({"bench": "simple", "track": "official"})

    def test_truth_takes_no_sensor(self):
        r = resolve("simple", ["target=truth"], allow_planned=True)
        self.assertEqual((r.options.sensor, r.normalized), ("none", ("sensor",)))
        self.assertEqual(r.options.signature(), "simple/truth/pose-odometry/root-odometry/scalebfm")
        with self.assertRaisesRegex(ValueError, "takes no sensor"):
            resolve("simple", ["target=truth", "sensor=stereo"], allow_planned=True)

    def test_perception_needs_a_sensor(self):
        with self.assertRaisesRegex(ValueError, "needs sensor"):
            resolve("simple", ["sensor=none"], allow_planned=True)

    def test_depth_noise_belongs_to_rgbd(self):
        r = resolve("simple", ["sensor=rgbd", "depth_noise=realsense"], allow_planned=True)
        self.assertEqual(r.options.signature(), "simple/perception-rgbd-realsense/pose-odometry/root-odometry/scalebfm")
        with self.assertRaisesRegex(ValueError, "rgbd only"):
            resolve("simple", ["depth_noise=realsense"], allow_planned=True)


class CapabilityTables(unittest.TestCase):
    def test_a_value_the_benchmark_lacks_is_rejected_with_its_reason(self):
        with self.assertRaisesRegex(ValueError, "does not support sensor=stereo.*second rendered camera"):
            resolve("locomanilab", ["sensor=stereo"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "does not support controller=scalebfm"):
            resolve("bigym2", ["controller=scalebfm"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "does not support controller_root=odometry"):
            resolve("humanoidarena", ["controller_root=odometry"], allow_planned=True)

    def test_a_planned_value_stops_a_run_unless_allowed(self):
        with self.assertRaisesRegex(ValueError, "planned, not implemented yet"):
            resolve(
                "humanoidarena", ["target=perception", "sensor=rgbd"]
            )  # perception on its front camera is still planned
        r = resolve("humanoidarena", ["target=perception", "sensor=rgbd"], allow_planned=True)
        self.assertTrue(r.planned)
        self.assertFalse(resolve("humanoidarena").planned)  # the truth profile is what the bed runs

    def test_simple_runs_the_switches_49_2_verified(self):
        # TODO 49.2: the loops' switches, truth targets, RGB-D with and without noise are ready on SIMPLE
        for over in (
            [],
            ["target=truth"],
            ["sensor=rgbd"],
            ["sensor=rgbd", "depth_noise=realsense"],
            ["self_pose=truth"],
            ["controller_root=truth"],
            ["self_pose=truth", "controller_root=truth"],
        ):
            r = resolve("simple", over)
            self.assertEqual(r.planned, (), over)
        self.assertNotIn("depth_noise=none", resolve("simple", allow_planned=True).planned)  # no noise without rgbd

    def test_every_table_covers_every_axis_and_every_value_is_a_known_choice(self):
        choices = {f: BenchOptions.__dataclass_fields__[f].metadata["choices"] for f in AXES}
        for name in BENCHES:
            spec = registry.spec(name)
            for axis, value, status in spec.table():
                self.assertIn(value, choices[axis], f"{name}.{axis}")
                self.assertIn(status, (READY, PLANNED))
                if axis == "sensor":
                    self.assertNotEqual(value, "none", f"{name}: sensor=none is implied by target=truth")

    def test_a_table_with_a_missing_axis_or_a_bad_status_is_refused(self):
        full = {a: {"x": PLANNED} for a in AXES}
        with self.assertRaisesRegex(ValueError, "missing"):
            BenchSpec("b", "B", "", {a: v for a, v in full.items() if a != "controller"})
        with self.assertRaisesRegex(ValueError, "statuses"):
            BenchSpec("b", "B", "", {**full, "target": {"truth": "soon"}})


class Signatures(unittest.TestCase):
    def test_different_switches_never_share_a_signature(self):
        seen = {}
        grid = itertools.product(BENCHES, O.TARGETS, O.SENSORS, O.DEPTH_NOISE, O.SELF_POSE, O.CONTROLLER_ROOT,
                                 O.CONTROLLERS)  # fmt: skip
        for b, t, s, n, p, r, c in grid:
            o = BenchOptions(b, t, s, n, p, r, c)
            if O.consistency_errors(o):
                continue
            sig = o.signature()
            self.assertNotIn(sig, seen, f"{o} and {seen.get(sig)}")
            seen[sig] = o


class Entry(unittest.TestCase):
    def test_show_prints_the_resolved_record_and_table_prints_every_benchmark(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["show", "--bench", "simple", "--bench-set", "target=truth"]), 0)
        self.assertIn('"signature": "simple/truth/', out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["table"]), 0)
        for name in BENCHES:
            self.assertIn(f"({name})", out.getvalue())

    def test_show_reports_an_impossible_combination_as_an_error(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["show", "--bench", "locomanilab", "--bench-set", "sensor=stereo"]), 2)


class FrameShape(unittest.TestCase):
    def test_a_frame_checks_its_image_and_depth_agree(self):
        rgb, depth = np.zeros((4, 6, 3), np.uint8), np.zeros((4, 6), np.float32)
        f = Frame(rgb, depth, np.eye(3), np.eye(4), "rgbd", 0.0)
        self.assertEqual(f.depth.shape, (4, 6))
        with self.assertRaisesRegex(ValueError, "differ"):
            Frame(rgb, np.zeros((5, 6), np.float32), np.eye(3), np.eye(4), "rgbd", 0.0)


if __name__ == "__main__":
    unittest.main()
